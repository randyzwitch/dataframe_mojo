"""Bounded integer reduction kernels over a host-certified expression shape.

The host binds an optional `x > threshold` predicate and optional checked
`(x + addend) * factor` expression followed by an explicit Int64 cast and SUM.
No runtime expression descriptors or opcodes reach these kernels. All surviving
row values remain Int32, so an Int64 accumulator is exact for <= Int32.MAX rows.
The launch caps the number of partials; grid-stride loops handle larger inputs.
"""

# Kernel arguments have explicit device lifetimes: callers retain their native
# input and private scratch owners until final synchronization. Untracked
# origins let the compiled function type be retained in a concrete cache slot.
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives.block import prefix_sum
from std.memory import Pointer


def partials[
    nullable: Bool, filtered: Bool, arithmetic: Bool
](
    values: Pointer[Int32, UntrackedOrigin[mut=False]],
    validity: Pointer[UInt8, UntrackedOrigin[mut=False]],
    validity_offset: UInt32,
    rows: UInt32,
    stride: UInt32,
    threshold: Int32,
    addend: Int32,
    factor: Int32,
    sums: Pointer[Int64, MutUntrackedOrigin],
    counts: Pointer[UInt32, MutUntrackedOrigin],
    faults: Pointer[UInt32, MutUntrackedOrigin],
):
    var total = Int64(0)
    var count = UInt32(0)
    var failures = UInt32(0)
    var row = global_idx.x
    while row < Int(rows):
        var present = True
        comptime if nullable:
            var bit = Int(validity_offset) + row
            present = (
                validity[unsafe_offset=bit // 8] >> UInt8(bit % 8)
            ) & 1 != 0
        if present:
            var value = Int64(values[unsafe_offset=row])
            var selected = True
            comptime if filtered:
                selected = value > Int64(threshold)
            if selected:
                var failed = False
                comptime if arithmetic:
                    value += Int64(addend)
                    failed = value < Int64(Int32.MIN) or value > Int64(
                        Int32.MAX
                    )
                    if not failed:
                        value *= Int64(factor)
                        failed = value < Int64(Int32.MIN) or value > Int64(
                            Int32.MAX
                        )
                if failed:
                    failures += 1
                else:
                    total += value
                    count += 1
        row += Int(stride)
    var block_sum = prefix_sum[block_size=256](total)
    var block_count = prefix_sum[block_size=256](count)
    var block_failures = prefix_sum[block_size=256](failures)
    if thread_idx.x == 255:
        sums[unsafe_offset=block_idx.x] = block_sum
        counts[unsafe_offset=block_idx.x] = block_count
        faults[unsafe_offset=block_idx.x] = block_failures


def finish(
    sums: Pointer[Int64, UntrackedOrigin[mut=False]],
    counts: Pointer[UInt32, UntrackedOrigin[mut=False]],
    faults: Pointer[UInt32, UntrackedOrigin[mut=False]],
    partial_count: UInt32,
    result: Pointer[Int64, MutUntrackedOrigin],
):
    """Publish sum, valid count and fault count in one small control record."""
    var total = Int64(0)
    var count = UInt32(0)
    var failures = UInt32(0)
    var part = thread_idx.x
    while part < Int(partial_count):
        total += sums[unsafe_offset=part]
        count += counts[unsafe_offset=part]
        failures += faults[unsafe_offset=part]
        part += 256
    var block_sum = prefix_sum[block_size=256](total)
    var block_count = prefix_sum[block_size=256](count)
    var block_failures = prefix_sum[block_size=256](failures)
    if thread_idx.x == 255:
        result[unsafe_offset=0] = block_sum
        result[unsafe_offset=1] = Int64(block_count)
        result[unsafe_offset=2] = Int64(block_failures)
