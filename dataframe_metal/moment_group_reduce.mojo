"""Unrouted grouped moment execution and exact mathematical finishing.

The ordinary and moment aggregates share tuple discovery, predicate/value
evaluation, group counters and one readback. CPU numerical compatibility
must be established separately before registering public VAR/STD handoffs.
"""
from std.memory import bitcast
from std.time import perf_counter_ns
from std.math import sqrt
from dataframe import Column, DataFrame, DataType, LazyFrame, Series
from dataframe.expr import VAR, STD, SUM, MEAN
from .moment_plan import MomentPlan, plan_moments, moment_output_sources
from .moment_reduce import moment_masks
from .moment_finish import finish_variance
from .group_layout import GROUP_VALUE_WORDS, GROUP_MOMENT_WORDS
from .group_partials import GroupedPartials
from .group_general_reduce import general_partials
from .group_finish import finish_grouped
from .group_output import group_order
from .multi_plan import ReductionOutput
from .accelerator.gpu.group_moments import (
    GROUP_SQUARES,
    GROUP_MOMENT_SUM,
    GROUP_MOMENT_SPECIAL,
)
from .accelerator.gpu.query_scope import Scope, shared_service
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.gpu.reduction_inputs import HandoffTimings


def _signed_bin(words: List[UInt32], at: Int) -> Int128:
    return Int128(
        bitcast[DType.int64](UInt64(words[at]) | (UInt64(words[at + 1]) << 32))
    )


def finish_grouped_moments_math(
    plan: MomentPlan, frame: DataFrame, partials: GroupedPartials
) raises -> DataFrame:
    if not plan.grouped:
        raise Error("Grouped moment finishing requires grouped states")
    var grouped = plan.grouped.value().copy()
    var nf = len(grouped.selected.bundle.fields)
    var capacity = len(partials.first)
    var blocks = partials.blocks
    var entries = capacity * blocks * nf
    if (
        nf < 1
        or blocks < 1
        or len(partials.states) != entries * GROUP_MOMENT_WORDS
        or len(partials.rows) != capacity * blocks
    ):
        raise Error("Metal grouped moment result does not match its schema")
    grouped.outputs = List[ReductionOutput]()
    var ordinary_fields = False
    for field in range(nf):
        grouped.selected.bundle.fields[field].needs_sum = False
    for output in plan.outputs:
        if output.op != VAR and output.op != STD:
            ordinary_fields = ordinary_fields or output.field >= 0
            grouped.outputs.append(
                ReductionOutput(
                    output.field,
                    output.name.copy(),
                    output.op,
                    Int(output.parameter),
                )
            )
            if output.field >= 0 and (output.op == SUM or output.op == MEAN):
                grouped.selected.bundle.fields[output.field].needs_sum = True
    # Key ordering and row counts need no per-value state. Moment-only and
    # moment-plus-LEN queries avoid both compaction and ordinary decoding.
    var ordinary_states = List[UInt32]()
    if ordinary_fields:
        ordinary_states = List[UInt32](
            length=entries * GROUP_VALUE_WORDS, fill=0
        )
        for entry in range(entries):
            for word in range(GROUP_VALUE_WORDS):
                ordinary_states[
                    entry * GROUP_VALUE_WORDS + word
                ] = partials.states[entry * GROUP_MOMENT_WORDS + word]
    else:
        grouped.selected.bundle.fields.clear()
    var ordinary = finish_grouped(
        grouped,
        frame,
        blocks,
        partials.first,
        partials.rows,
        ordinary_states,
        partials.extrema,
    )
    var columns = List[Series]()
    for name in grouped.keys.names:
        columns.append(ordinary.column(name).copy())
    var order = group_order(partials.first)
    var sources = moment_output_sources(plan.outputs)
    var shared = List[Bool](length=len(plan.outputs), fill=False)
    for i in range(len(sources)):
        if sources[i] != i:
            shared[sources[i]] = True
    var cached_values = List[List[Float64]]()
    var cached_validity = List[List[Bool]]()
    for _ in plan.outputs:
        cached_values.append(List[Float64]())
        cached_validity.append(List[Bool]())
    for i in range(len(plan.outputs)):
        ref output = plan.outputs[i]
        if output.op != VAR and output.op != STD:
            columns.append(ordinary.column(output.name).copy())
            continue
        var values = List[Float64]()
        var validity = List[Bool]()
        if sources[i] == i:
            var field = output.field
            if field < 0 or field >= nf:
                raise Error("Metal grouped moment output has an invalid field")
            var floating = (
                plan.selected.bundle.fields[field].dtype == DataType.FLOAT32
            )
            for slot in order:
                var sums = List[Int128](length=9 if floating else 2, fill=0)
                var squares = List[Int128](length=18 if floating else 4, fill=0)
                var count = 0
                var non_finite = False
                for block in range(blocks):
                    var at = (
                        (slot * blocks + block) * nf + field
                    ) * GROUP_MOMENT_WORDS
                    count += Int(partials.states[at])
                    non_finite = non_finite or Bool(
                        partials.states[at + GROUP_MOMENT_SPECIAL]
                    )
                    for bin in range(len(sums)):
                        sums[bin] += _signed_bin(
                            partials.states,
                            at
                            + (10 if floating else GROUP_MOMENT_SUM)
                            + bin * 2,
                        )
                    for bin in range(len(squares)):
                        squares[bin] += _signed_bin(
                            partials.states, at + GROUP_SQUARES + bin * 2
                        )
                var answer = finish_variance(
                    sums,
                    squares,
                    count,
                    Int(output.parameter),
                    -149 if floating else 0,
                    False,
                    non_finite,
                )
                values.append(answer.value() if answer else Float64(0))
                validity.append(Bool(answer))
            if shared[i]:
                cached_values[i] = values.copy()
                cached_validity[i] = validity.copy()
        else:
            values = cached_values[sources[i]].copy()
            validity = cached_validity[sources[i]].copy()
        if output.op == STD:
            for row in range(len(values)):
                values[row] = sqrt(values[row])
        columns.append(Series(output.name, Column[Float64](values^, validity)))
    return DataFrame(columns^)


def execute_grouped_moments_math(
    region: LazyFrame, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var matched = plan_moments(region)
    if not matched or not matched.value().grouped:
        raise Error("Metal grouped moments did not accept this region")
    var plan = matched.take()
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    if frame.height() >= Int(Int32.MAX):
        raise Error("Metal grouped moments require fewer than 2**31 input rows")
    var timings = HandoffTimings()
    if frame.height() == 0:
        var empty = GroupedPartials(
            List[UInt32](), List[UInt32](), List[UInt32](), List[Int64](), 1
        )
        return (finish_grouped_moments_math(plan, frame, empty), timings^)
    comptime if not METAL_TARGET:
        raise Error("Metal grouped moments require an Apple compilation target")
    else:
        var scope = Scope(service=shared_service())
        var started = Int(perf_counter_ns())
        _ = scope.context()
        timings.context_ns = Int(perf_counter_ns()) - started
        var masks = moment_masks(plan)
        var partials = general_partials[True](
            plan.grouped.value(),
            frame,
            scope,
            zero_copy,
            timings,
            moment_mask=masks[0],
        )
        started = Int(perf_counter_ns())
        var answer = finish_grouped_moments_math(plan, frame, partials)
        timings.finish_ns = Int(perf_counter_ns()) - started
        scope.mark_success()
        return (answer^, timings^)
