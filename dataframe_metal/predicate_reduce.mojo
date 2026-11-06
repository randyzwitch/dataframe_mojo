"""One reduction scan over checked native AND/OR predicates."""
from std.time import perf_counter_ns
from dataframe import DataFrame, LazyFrame
from .composition_plan import has_extrema
from .composition_reduce import (
    ComposedScan,
    _mixed_dispatch,
    finish_composition,
)
from .predicate_plan import (
    PredicateComposition,
    PredicateProgram,
    plan_predicates,
)
from .accelerator.gpu.reduction_inputs import HandoffTimings
from .accelerator.gpu.query_scope import Scope, shared_service


def _dispatch[
    precise: Bool, wide: Bool
](
    selected: PredicateComposition,
    frame: DataFrame,
    scope: Scope,
    zero_copy: Bool,
    mut timings: HandoffTimings,
) raises -> ComposedScan:
    var program = Optional[PredicateProgram](selected.program.copy())
    if has_extrema(selected.bundle):
        return _mixed_dispatch[precise, wide, mixed=True, compound=True](
            selected.bundle, frame, scope, zero_copy, timings, program
        )
    return _mixed_dispatch[precise, wide, mixed=False, compound=True](
        selected.bundle, frame, scope, zero_copy, timings, program
    )


def execute_predicates(
    region: LazyFrame, zero_copy: Bool
) raises -> Tuple[DataFrame, HandoffTimings]:
    var matched = plan_predicates(region)
    if not matched:
        raise Error(
            "Metal unsupported [handoff]: predicate reduction was not accepted"
        )
    var selected = matched.take()
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    if frame.height() >= Int(Int32.MAX):
        raise Error("Metal unsupported [handoff]: 2**31 or more rows")
    var scope = Scope(service=shared_service())
    var timings = HandoffTimings()
    var started = Int(perf_counter_ns())
    _ = scope.context()
    timings.context_ns = Int(perf_counter_ns()) - started
    var result: ComposedScan
    if selected.bundle.precise_float:
        try:
            result = _dispatch[True, False](
                selected, frame, scope, zero_copy, timings
            )
        except error:
            if String(error) != "Metal exact bins need wide scan":
                raise error^
            result = _dispatch[True, True](
                selected, frame, scope, zero_copy, timings
            )
    else:
        result = _dispatch[False, False](
            selected, frame, scope, zero_copy, timings
        )
    var answer = finish_composition(
        selected.bundle,
        (result.integers.copy(), result.floating.copy(), result.counts.copy()),
        result.extrema,
    )
    scope.mark_success()
    return (answer^, timings^)
