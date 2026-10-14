"""Stable dataframe compaction using Mojo block scans and device gathers."""
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives.block import prefix_sum
from std.memory import Pointer


def filter_scan(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    positions: Pointer[UInt32, MutAnyOrigin],
    blocks: Pointer[UInt32, MutAnyOrigin],
    slot: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    # All lanes participate, including padding in the final partial block.
    var row = global_idx.x
    var selected = UInt32(0)
    var failed = UInt32(0)
    if row < Int(rows):
        var at = Int(slot) * Int(stride) + row
        selected = UInt32(
            validity[unsafe_offset=at] != 0 and words[unsafe_offset=at] != 0
        )
        failed = UInt32(errors[unsafe_offset=at] != 0)
    var prefix = prefix_sum[block_size=256](selected)
    var failures = prefix_sum[block_size=256](failed)
    if row < Int(rows):
        positions[unsafe_offset=row] = prefix - selected
    if thread_idx.x == 255:
        var count = (Int(rows) + 255) // 256
        blocks[unsafe_offset=block_idx.x] = prefix
        blocks[unsafe_offset=count + block_idx.x] = failures


def filter_offsets(
    blocks: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, MutAnyOrigin],
    count: UInt32,
):
    var total = UInt32(0)
    var failed = UInt32(0)
    for block in range(Int(count)):
        var selected = blocks[unsafe_offset=block]
        blocks[unsafe_offset=block] = total
        total += selected
        failed |= blocks[unsafe_offset=Int(count) + block]
    metadata[unsafe_offset=0] = total
    metadata[unsafe_offset=1] = failed


def filter_gather(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    positions: Pointer[UInt32, ImmutAnyOrigin],
    blocks: Pointer[UInt32, ImmutAnyOrigin],
    gathers: Pointer[Int64, ImmutAnyOrigin],
    count: UInt32,
    predicate: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var mask = Int(predicate) * Int(stride) + row
    if validity[unsafe_offset=mask] == 0 or words[unsafe_offset=mask] == 0:
        return
    var target = Int(
        positions[unsafe_offset=row] + blocks[unsafe_offset=row // 256]
    )
    for column in range(Int(count)):
        var base = Int(gathers[unsafe_offset=column]) * Int(stride)
        output_words[unsafe_offset=base + target] = words[
            unsafe_offset=base + row
        ]
        if slots:
            var high_base = Int(stride) * Int(slots)
            output_words[unsafe_offset=high_base + base + target] = words[
                unsafe_offset=high_base + base + row
            ]
        output_validity[unsafe_offset=base + target] = validity[
            unsafe_offset=base + row
        ]
        output_errors[unsafe_offset=base + target] = errors[
            unsafe_offset=base + row
        ]
