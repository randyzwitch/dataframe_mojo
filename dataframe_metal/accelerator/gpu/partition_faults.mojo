"""Ordered operand/control faults for one CPU OVER state preparation.

The caller proves original sources already observed and row errors are ranked
occurrence tokens. Exact scheduler phases/ordinals define whole-run evaluation;
no source value is exported. Run order precedes state/operand/control phase,
then operand batch/AST/row or control group order. No error categories are ORed.
"""
from max.gpu import global_idx
from std.memory import Pointer
from .ordered_faults import rank

comptime Fault = Tuple[
    UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32
]
comptime OPERAND = UInt32(1)
comptime CONTROL = UInt32(2)
comptime WORDS = 8


def _empty() -> Fault:
    return (
        UInt32(0),
        UInt32(0),
        UInt32(0),
        UInt32(0),
        UInt32(0),
        UInt32(0),
        UInt32(0),
        UInt32(0),
    )


def _earlier(left: Fault, right: Fault) -> Bool:
    if not left[0]:
        return False
    if not right[0]:
        return True
    # kind/location/category never override semantic run/phase/batch/AST/row
    # order. Location breaks only an otherwise identical event for determinism.
    comptime for at in range(1, 7):
        if left[at] != right[at]:
            return left[at] < right[at]
    return False


def _write(output: Pointer[UInt32, MutAnyOrigin], base: Int, fault: Fault):
    comptime for at in range(WORDS):
        output[unsafe_offset=base + at] = fault[at]


def operand_partials(
    errors: Pointer[UInt32, ImmutAnyOrigin],
    phase_by_row: Pointer[UInt32, ImmutAnyOrigin],
    ordinal_by_row: Pointer[UInt32, ImmutAnyOrigin],
    partials: Pointer[UInt32, MutAnyOrigin],
    target: UInt32,
    rows: UInt32,
    stride: UInt32,
    slices: UInt32,
    stage: UInt32,
    batch_rows: UInt32,
):
    """One bounded slice of original rows; supplied ordinals may be gathered.

    At least one slice and positive batch size are host preflight invariants.
    A fallback schedule has phase0/original-row ordinals. Contiguous schedules
    have gathered run-local ordinals, so original row order cannot win by mistake.
    """
    var slice = global_idx.x
    if slice >= Int(slices):
        return
    var width = (Int(rows) + Int(slices) - 1) // Int(slices)
    var begin = slice * width
    var end = min(Int(rows), begin + width)
    var winner = _empty()
    for row in range(begin, end):
        var token = errors[unsafe_offset=Int(target) * Int(stride) + row]
        if not token:
            continue
        var ordinal = ordinal_by_row[unsafe_offset=row]
        var event: Fault = (
            OPERAND,
            phase_by_row[unsafe_offset=row],
            stage,
            ordinal // batch_rows,
            rank(token),
            ordinal,
            UInt32(row),
            token,
        )
        if _earlier(event, winner):
            winner = event
    _write(partials, slice * WORDS, winner)


def control_partials(
    controls: Pointer[UInt32, ImmutAnyOrigin],
    phase_by_group: Pointer[UInt32, ImmutAnyOrigin],
    rank_by_group: Pointer[UInt32, ImmutAnyOrigin],
    partials: Pointer[UInt32, MutAnyOrigin],
    groups: UInt32,
    slices: UInt32,
    output_offset: UInt32,
    stage: UInt32,
):
    """Each control already retains its first original probability/edge fault.

    Controls have two words/group: used edges then failure. Their first failure
    is sufficient because CPU processes all of one group's controls before the
    next group. output_offset counts32-byte partial records, not scalar words.
    """
    var slice = global_idx.x
    if slice >= Int(slices):
        return
    var width = (Int(groups) + Int(slices) - 1) // Int(slices)
    var begin = slice * width
    var end = min(Int(groups), begin + width)
    var winner = _empty()
    for group in range(begin, end):
        var failure = controls[unsafe_offset=group * 2 + 1]
        if not failure:
            continue
        var event: Fault = (
            CONTROL,
            phase_by_group[unsafe_offset=group],
            stage,
            UInt32(0),
            rank_by_group[unsafe_offset=group],
            UInt32(0),
            UInt32(group),
            failure,
        )
        if _earlier(event, winner):
            winner = event
    _write(partials, (Int(output_offset) + slice) * WORDS, winner)


def finish(
    partials: Pointer[UInt32, ImmutAnyOrigin],
    result: Pointer[UInt32, MutAnyOrigin],
    count: UInt32,
):
    if global_idx.x != 0:
        return
    var winner = _empty()
    for item in range(Int(count)):
        var base = item * WORDS
        var event: Fault = (
            partials[unsafe_offset=base],
            partials[unsafe_offset=base + 1],
            partials[unsafe_offset=base + 2],
            partials[unsafe_offset=base + 3],
            partials[unsafe_offset=base + 4],
            partials[unsafe_offset=base + 5],
            partials[unsafe_offset=base + 6],
            partials[unsafe_offset=base + 7],
        )
        if _earlier(event, winner):
            winner = event
    _write(result, 0, winner)
