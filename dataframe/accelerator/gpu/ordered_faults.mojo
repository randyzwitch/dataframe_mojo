"""Occurrence-ranked errors for the private one-row CUT scalar producer.

Legacy row kernels and their error flags remain unchanged. The host validates
rank geometry and observes tokens before a result can leave this region.
"""


def pack(rank: UInt32, category: UInt32) -> UInt32:
    return ((rank + UInt32(1)) << UInt32(7)) | category


def rank(token: UInt32) -> UInt32:
    return (token >> UInt32(7)) - UInt32(1)


def category(token: UInt32) -> UInt32:
    return token & UInt32(127)


def merge(left: UInt32, right: UInt32) -> UInt32:
    if not left:
        return right
    if not right:
        return left
    return left if (left >> UInt32(7)) <= (right >> UInt32(7)) else right


def earlier(
    left: UInt32,
    left_row: UInt32,
    right: UInt32,
    right_row: UInt32,
    batch_rows: UInt32 = 8192,
) -> Bool:
    """Compare events within one materialized projection/producer phase.

    Batch size is positive; worker partitions are contiguous and aligned to
    this size. Projection and phase ordering belong to the outer observer.
    """
    if not left:
        return False
    if not right:
        return True
    var a = left_row // batch_rows
    var b = right_row // batch_rows
    if a != b:
        return a < b
    if (left >> UInt32(7)) != (right >> UInt32(7)):
        return (left >> UInt32(7)) < (right >> UInt32(7))
    return left_row < right_row


# Select one output fault without category-priority OR. Production admission
# currently permits exactly one row; the batch comparator also serves probes.
from max.gpu import global_idx
from std.memory import Pointer


def observe(
    errors: Pointer[UInt32, ImmutAnyOrigin],
    result: Pointer[UInt32, MutAnyOrigin],
    target: UInt32,
    rows: UInt32,
    stride: UInt32,
    batch_rows: UInt32 = 8192,
):
    if global_idx.x != 0:
        return
    var winner = UInt32(0)
    var winner_row = UInt32.MAX
    for row in range(Int(rows)):
        var error = errors[unsafe_offset=Int(target) * Int(stride) + row]
        if earlier(error, UInt32(row), winner, winner_row, batch_rows):
            winner = error
            winner_row = UInt32(row)
    result[unsafe_offset=0] = winner
    result[unsafe_offset=1] = winner_row


def observe_scalar_inputs(
    errors: Pointer[UInt32, ImmutAnyOrigin],
    result: Pointer[UInt32, MutAnyOrigin],
    columns: UInt32,
):
    """Foreign regions must have observed every visible one-row input fault.

    The host proves stride==rows==1. Never reinterpret a legacy flag as a rank.
    This control pass precedes ordered evaluation on the same stream.
    """
    if global_idx.x != 0:
        return
    var failure = UInt32(0)
    var source = UInt32.MAX
    for column in range(Int(columns)):
        if not failure and errors[unsafe_offset=column]:
            failure = errors[unsafe_offset=column]
            source = UInt32(column)
    result[unsafe_offset=0] = failure
    result[unsafe_offset=1] = source
