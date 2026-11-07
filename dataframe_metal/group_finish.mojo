"""Decode bounded GPU partial states; never discover groups on the CPU."""
from std.memory import bitcast
from dataframe import Column, DataFrame, DataType, Series
from .group_plan import GroupedComposition
from .group_output import GroupOutput, group_order
from .exact_float import finish_float_bins
from .accelerator.faults import data_fault
from .accelerator.gpu.group_state import GROUP_STATE_WORDS


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
    var keys32 = List[List[Int32]]()
    var keys64 = List[List[Int64]]()
    var key_valid = List[List[Bool]]()
    for _ in plan.keys.names:
        keys32.append(List[Int32]())
        keys64.append(List[Int64]())
        key_valid.append(List[Bool]())
    var columns = List[GroupOutput]()
    for output in plan.outputs:
        columns.append(
            GroupOutput(
                output,
                DataType.INT64 if output.field
                < 0 else bundle.fields[output.field].dtype,
                output.name in plan.row_counts,
            )
        )
    var integers = List[Int128](length=nf, fill=0)
    var floating = List[Float64](length=nf, fill=0)
    var counts = List[Int](length=nf, fill=0)
    var extrema = List[Int64](length=nf * 4, fill=0)
    var bins = List[Int128](length=9, fill=0)
    for slot in order:
        var kept = 0
        for block in range(blocks):
            kept += Int(rows[slot * blocks + block])
        for field in range(nf):
            integers[field] = 0
            floating[field] = 0
            counts[field] = 0
            if bundle.fields[field].dtype == DataType.FLOAT32:
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
                special |= states[at + 6]
                if dtype == DataType.FLOAT32:
                    least = min(least, Int(states[at + 8]))
                    greatest = max(greatest, Int(states[at + 9]))
                    for bin in range(9):
                        var raw = UInt64(states[at + 10 + bin * 2]) | (
                            UInt64(states[at + 11 + bin * 2]) << 32
                        )
                        bins[bin] += Int128(bitcast[DType.int64](raw))
                    if states[at + 7]:
                        minimum = min(minimum, Int64(states[at + 4]))
                        maximum = max(maximum, Int64(~states[at + 5]))
                        numeric += Int(states[at + 7])
                else:
                    var lower = UInt64(states[at + 1]) | (
                        UInt64(states[at + 2]) << 32
                    )
                    integers[field] += (
                        Int128(bitcast[DType.int32](states[at + 3])) << 64
                    ) + Int128(lower)
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
            if dtype == DataType.FLOAT32 and bundle.fields[field].needs_sum:
                if special & 7:
                    raise data_fault(
                        "grouped non-finite Float32 sums require CPU adjudication"
                    )
                floating[field] = finish_float_bins(
                    bins,
                    Bool(special & 1),
                    Bool(special & 2),
                    Bool(special & 4),
                )
                var growth = 0
                var n = max(1, counts[field]) - 1
                while n:
                    growth += 1
                    n >>= 1
                if (
                    (special & 7) == 0
                    and least < 512
                    and greatest - least + 1 + growth > 53
                ):
                    raise data_fault(
                        "grouped Float32 sum needs CPU accumulation order"
                    )
            extrema[field * 4] = minimum
            extrema[field * 4 + 1] = maximum
            extrema[field * 4 + 2] = Int64(numeric)
            extrema[field * 4 + 3] = Int64(
                (special & 1)
                | (UInt32(Bool(special & 8)) << 1)
                | (UInt32(Bool(special & 16)) << 2)
            )
        # Copy only finalized keys. Returning source slices would keep the
        # entire input key allocation alive behind a tiny grouped result.
        for i in range(len(plan.keys.names)):
            var key = frame.item(Int(first[slot]), plan.keys.names[i])
            var present = not key.is_null()
            key_valid[i].append(present)
            if plan.keys.dtypes[i] == DataType.INT32:
                keys32[i].append(key.int32() if present else Int32(0))
            else:
                keys64[i].append(key.int64() if present else Int64(0))
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
    var answer = List[Series]()
    for i in range(len(plan.keys.names)):
        var narrow_keys = keys32.pop(0)
        var wide_keys = keys64.pop(0)
        var validity = key_valid.pop(0)
        if plan.keys.dtypes[i] == DataType.INT32:
            answer.append(
                Series(
                    plan.keys.names[i], Column[Int32](narrow_keys^, validity)
                )
            )
        else:
            answer.append(
                Series(plan.keys.names[i], Column[Int64](wide_keys^, validity))
            )
    while columns:
        answer.append(columns.pop(0).finish())
    return DataFrame(answer^)
