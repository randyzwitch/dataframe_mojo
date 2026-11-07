"""Fused global moment scans with bounded mathematical finishing.

Integer statistical inputs use Float64 conversion; exact moments avoid
cancellation before the final Float64 operations. See docs/design/metal-moments.md.
"""
from std.time import perf_counter_ns
from std.math import sqrt
from dataframe import Column, DataFrame, DataType, LazyFrame, Series
from dataframe.expr import VAR, STD, SUM, MEAN
from .moment_plan import MomentPlan, plan_moments, moment_output_sources
from .moment_finish import finish_variance
from .moment_compatibility import (
    require_cpu_moment_compatibility,
    require_cpu_moment_precision,
)
from .multi_plan import ReductionOutput
from .composition_plan import has_trees
from .composition_reduce import (
    ComposedScan,
    _mixed_dispatch,
    finish_composition,
)
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.reduction_inputs import HandoffTimings


def moment_masks(plan: MomentPlan) raises -> Tuple[UInt32, UInt32]:
    var selected = UInt32(0)
    var cpu_sum_order = UInt32(0)
    if len(plan.needs_moments) != len(plan.selected.bundle.fields):
        raise Error("Metal moment field mask does not match its schema")
    for field in range(len(plan.needs_moments)):
        selected |= UInt32(plan.needs_moments[field]) << UInt32(field)
    for output in plan.outputs:
        if output.op == SUM or output.op == MEAN:
            cpu_sum_order |= UInt32(1) << UInt32(output.field)
    return (selected, cpu_sum_order)


def _dispatch_global_moments[
    precise: Bool, wide: Bool
](
    plan: MomentPlan,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> ComposedScan:
    var masks = moment_masks(plan)
    var program = Optional(plan.selected.program.copy())
    if has_trees(plan.selected.bundle):
        return _mixed_dispatch[
            precise, wide, mixed=True, compound=True, trees=True, moments=True
        ](
            plan.selected.bundle,
            frame,
            scope,
            zero_copy,
            timings,
            program,
            masks[0],
            masks[1],
        )
    return _mixed_dispatch[
        precise, wide, mixed=True, compound=True, moments=True
    ](
        plan.selected.bundle,
        frame,
        scope,
        zero_copy,
        timings,
        program,
        masks[0],
        masks[1],
    )


def finish_global_moments_math(
    plan: MomentPlan, scan: ComposedScan, cpu_compatible: Bool = False
) raises -> DataFrame:
    """Finish exact moments, retaining ordinary aggregate output semantics.

    The internal mathematical primitive can be tested independently. Public
    execution requests CPU-compatible completion and adjudicates sensitive
    centering through the ordinary CPU executor.
    """
    if plan.grouped:
        raise Error("Global moment finishing cannot consume grouped states")
    var bundle = plan.selected.bundle.copy()
    bundle.outputs = List[ReductionOutput]()
    for output in plan.outputs:
        if output.op != VAR and output.op != STD:
            bundle.outputs.append(
                ReductionOutput(
                    output.field,
                    output.name.copy(),
                    output.op,
                    Int(output.parameter),
                )
            )
    var ordinary = finish_composition(
        bundle,
        (scan.integers.copy(), scan.floating.copy(), scan.counts.copy()),
        scan.extrema,
    )
    var columns = List[Series]()
    var sources = moment_output_sources(plan.outputs)
    var cached = List[Optional[Float64]](
        length=len(plan.outputs), fill=Optional[Float64]()
    )
    for i in range(len(plan.outputs)):
        ref output = plan.outputs[i]
        if output.op != VAR and output.op != STD:
            columns.append(ordinary.column(output.name).copy())
            continue
        var answer = Optional[Float64]()
        if sources[i] != i:
            answer = cached[sources[i]].copy()
        else:
            var field = output.field
            if (
                field < 0
                or field >= len(scan.moment_data)
                or len(scan.moment_data[field]) != 28
            ):
                raise Error("Metal moment result does not match its schema")
            ref data = scan.moment_data[field]
            var floating = (
                plan.selected.bundle.fields[field].dtype == DataType.FLOAT32
            )
            var sums = List[Int128]()
            var squares = List[Int128]()
            for bin in range(9 if floating else 2):
                sums.append(data[bin])
            for bin in range(18 if floating else 4):
                squares.append(data[9 + bin])
            answer = finish_variance(
                sums,
                squares,
                scan.counts[field],
                Int(output.parameter),
                -149 if floating else 0,
                False,
                data[27] != 0,
            )
            if cpu_compatible and answer:
                require_cpu_moment_compatibility(
                    sums,
                    scan.counts[field],
                    Int(output.parameter),
                    -149 if floating else 0,
                    answer.value(),
                )
                if answer.value() != 0:
                    require_cpu_moment_precision(squares, scan.counts[field])
            cached[i] = answer.copy()
        if answer and output.op == STD:
            answer = sqrt(answer.value())
        columns.append(
            Series(
                output.name,
                Column[Float64](
                    [answer.value() if answer else Float64(0)], [Bool(answer)]
                ),
            )
        )
    return DataFrame(columns^)


def execute_global_moments_math(
    region: LazyFrame, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var matched = plan_moments(region)
    if not matched or matched.value().grouped:
        raise Error("Metal global moments did not accept this region")
    var plan = matched.take()
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    if frame.height() >= Int(Int32.MAX):
        raise Error("Metal unsupported [handoff]: 2**31 or more rows")
    var timings = HandoffTimings()
    if frame.height() == 0:
        var nf = len(plan.selected.bundle.fields)
        var moments = List[List[Int128]]()
        var extrema = List[Int64](length=nf * 4, fill=0)
        for field in range(nf):
            moments.append(List[Int128](length=28, fill=0))
            extrema[field * 4] = Int64.MAX
            extrema[field * 4 + 1] = Int64.MIN
        var empty = ComposedScan(
            List[Int128](length=nf, fill=0),
            List[Float64](length=nf, fill=0),
            List[Int](length=nf, fill=0),
            extrema^,
            moments^,
            List[UInt64](length=nf, fill=0),
        )
        return (finish_global_moments_math(plan, empty), timings^)
    comptime if not METAL_TARGET:
        raise Error("Metal moments require an Apple compilation target")
    else:
        var scope = Scope(service=shared_service())
        var started = Int(perf_counter_ns())
        _ = scope.context()
        timings.context_ns = Int(perf_counter_ns()) - started
        var result: ComposedScan
        if plan.selected.bundle.precise_float:
            try:
                result = _dispatch_global_moments[True, False](
                    plan, frame, scope, zero_copy, timings
                )
            except error:
                if String(error) != "Metal exact bins need wide scan":
                    raise error^
                result = _dispatch_global_moments[True, True](
                    plan, frame, scope, zero_copy, timings
                )
        else:
            result = _dispatch_global_moments[False, False](
                plan, frame, scope, zero_copy, timings
            )
        started = Int(perf_counter_ns())
        var answer = finish_global_moments_math(
            plan, result, cpu_compatible=True
        )
        timings.finish_ns = Int(perf_counter_ns()) - started
        scope.mark_success()
        return (answer^, timings^)
