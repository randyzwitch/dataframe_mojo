"""Decode bounded GPU partial states; never discover groups on the CPU."""
from std.bit import bit_width
from std.memory import bitcast
from dataframe import DataFrame, DataType
from .group_gather import gather_group_keys
from .group_plan import GroupedComposition
from .group_output import GroupOutput, group_order
from dataframe.expr import MIN, MAX
from .accelerator.faults import data_fault
from .accelerator.gpu.group_state import GROUP_STATE_WORDS


comptime GROUP_NONFINITE_FLOAT = "grouped non-finite Float32 sums require CPU adjudication"
comptime GROUP_ROUNDED_FLOAT = "grouped Float32 sum needs CPU accumulation order"


def _finish_group_float(
    bins: List[Int128], count: Int, least: Int, greatest: Int, special: UInt32
) raises -> Float64:
    if special & 7:
        raise data_fault(GROUP_NONFINITE_FLOAT)
    var growth = Int(bit_width(UInt64(max(1, count) - 1)))
    if least < 512 and greatest - least + 1 + growth > 53:
        raise data_fault(GROUP_ROUNDED_FLOAT)
    # Every input is a multiple of 2**(least - 149). The bound
    # above proves even the sum of their absolute magnitudes fits
    # in 53 bits at that scale. Splitting each input into same-sign
    # radix digits preserves this bound for every partial sum.
    var result = Float64(0)
    for bin in range(least // 32, greatest // 32 + 1):
        var scale = bitcast[DType.float64](UInt64(1023 - 149 + 32 * bin) << 52)
        result += bins[bin].cast[DType.float64]() * scale
    return result


def finish_grouped(
    plan: GroupedComposition,
    frame: DataFrame,
    blocks: Int,
    first: List[UInt32],
    rows: List[UInt32],
    states: List[UInt32],
    resolved: List[Int64],
) raises -> DataFrame:
    ref bundle = plan.selected.bundle
    var nf = len(bundle.fields)
    var order = group_order(first)
    var answer = gather_group_keys(frame, plan.keys, first, order)
    var columns = List[GroupOutput](capacity=len(plan.outputs))
    var needs_extrema = List[Bool](length=nf, fill=False)
    for output in plan.outputs:
        if output.op == MIN or output.op == MAX:
            needs_extrema[output.field] = True
        columns.append(
            GroupOutput(
                output,
                DataType.INT64 if output.field
                < 0 else bundle.fields[output.field].dtype,
                output.name in plan.row_counts,
                len(order),
            )
        )
    var integers = List[Int128](length=nf, fill=0)
    var floating = List[Float64](length=nf, fill=0)
    var counts = List[Int](length=nf, fill=0)
    var extrema = List[Int64](length=nf * 4, fill=0)
    var bins = List[Int128](length=9, fill=0)
    for slot in order:
        var kept = 0
        if plan.row_counts:
            for block in range(blocks):
                kept += Int(rows[slot * blocks + block])
        for field in range(nf):
            integers[field] = 0
            floating[field] = 0
            counts[field] = 0
            var needs_sum = bundle.fields[field].needs_sum
            var extrema_used = needs_extrema[field]
            if needs_sum and bundle.fields[field].dtype == DataType.FLOAT32:
                for bin in range(9):
                    bins[bin] = 0
            var special = UInt32(0)
            var least = 512
            var greatest = 0
            var minimum = Int64.MAX
            var maximum = Int64.MIN
            var numeric = 0
            var dtype = bundle.fields[field].dtype
            for block in range(blocks):
                var index = (slot * blocks + block) * nf + field
                var at = index * GROUP_STATE_WORDS
                var count = Int(states[at])
                counts[field] += count
                if dtype == DataType.FLOAT32:
                    if needs_sum or extrema_used:
                        special |= states[at + 6]
                    if needs_sum:
                        var lo = Int(states[at + 8])
                        var hi = Int(states[at + 9])
                        least = min(least, lo)
                        greatest = max(greatest, hi)
                        for bin in range(lo // 32, hi // 32 + 1):
                            var raw = UInt64(states[at + 10 + bin * 2]) | (
                                UInt64(states[at + 11 + bin * 2]) << 32
                            )
                            bins[bin] += Int128(bitcast[DType.int64](raw))
                    if extrema_used and states[at + 7]:
                        minimum = min(minimum, Int64(states[at + 4]))
                        maximum = max(maximum, Int64(~states[at + 5]))
                        numeric += Int(states[at + 7])
                else:
                    if needs_sum:
                        var lower = UInt64(states[at + 1]) | (
                            UInt64(states[at + 2]) << 32
                        )
                        integers[field] += (
                            Int128(bitcast[DType.int32](states[at + 3])) << 64
                        ) + Int128(lower)
                    if not extrema_used:
                        continue
                    if dtype == DataType.INT32 and states[at + 7]:
                        minimum = min(
                            minimum,
                            Int64(
                                bitcast[DType.int32](
                                    states[at + 4] ^ 0x80000000
                                )
                            ),
                        )
                        maximum = max(
                            maximum,
                            Int64(
                                bitcast[DType.int32](
                                    (~states[at + 5]) ^ 0x80000000
                                )
                            ),
                        )
                        numeric += Int(states[at + 7])
                    elif dtype == DataType.INT64 and count:
                        if states[at + 4] != UInt32.MAX:
                            minimum = min(minimum, resolved[index * 2])
                            maximum = max(maximum, resolved[index * 2 + 1])
                            numeric += count
            if dtype == DataType.FLOAT32 and needs_sum:
                floating[field] = _finish_group_float(
                    bins, counts[field], least, greatest, special
                )
            extrema[field * 4] = minimum
            extrema[field * 4 + 1] = maximum
            extrema[field * 4 + 2] = Int64(numeric)
            extrema[field * 4 + 3] = Int64(
                (special & 1)
                | (UInt32(Bool(special & 8)) << 1)
                | (UInt32(Bool(special & 16)) << 2)
            )
        for i in range(len(plan.outputs)):
            var field = plan.outputs[i].field
            if field < 0:
                columns[i].append(0, 0, 0, 0, 0, 0, 0, kept)
                continue
            columns[i].append(
                integers[field],
                floating[field],
                counts[field],
                extrema[field * 4],
                extrema[field * 4 + 1],
                Int(extrema[field * 4 + 2]),
                UInt32(extrema[field * 4 + 3]),
                kept,
            )
    while columns:
        answer.append(columns.pop(0).finish())
    return DataFrame(answer^)
