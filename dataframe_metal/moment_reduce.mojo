"""Unrouted global moment scans and mathematical finishing.

This module supplies the fused execution foundation. It deliberately has no
provider registration: CPU compatibility/adjudication and grouped finishing
must be completed before public VAR/STD handoffs are enabled.
"""
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, LazyFrame, Series
from dataframe.expr import VAR, STD, SUM, MEAN
from .moment_plan import MomentPlan, plan_moments
from .moment_finish import finish_variance
from .multi_plan import ReductionOutput
from .composition_plan import has_trees
from .composition_reduce import (
    ComposedScan,
    _mixed_dispatch,
    finish_composition,
)
from .accelerator.gpu.query_scope import Scope, shared_service
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
    plan: MomentPlan, scan: ComposedScan
) raises -> DataFrame:
    """Finish exact moments, retaining ordinary aggregate output semantics.

    Mathematical accuracy alone does not prove agreement with ill-conditioned
    CPU variance. This function is internal to the unrouted execution module.
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
    for output in plan.outputs:
        if output.op != VAR and output.op != STD:
            columns.append(ordinary.column(output.name).copy())
            continue
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
        var answer = finish_variance(
            sums,
            squares,
            scan.counts[field],
            Int(output.parameter),
            -149 if floating else 0,
            output.op == STD,
            data[27] != 0,
        )
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
        raise Error("Metal moments require fewer than 2**31 input rows")
    var scope = Scope(service=shared_service())
    var timings = HandoffTimings()
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
    var answer = finish_global_moments_math(plan, result)
    scope.mark_success()
    return (answer^, timings^)
