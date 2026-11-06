"""Decode bounded GPU partial states; never discover groups on the CPU."""
from std.memory import bitcast
from dataframe import Column, DataFrame, DataType, Series
from .group_plan import GroupedComposition
from .composition_reduce import finish_composition
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
    var order = List[Int]()
    for slot in range(len(first)):
        if first[slot] != UInt32.MAX:
            # At most256 output groups. Sort only their GPU-recorded first
            # row numbers, preserving maintain_order without scanning keys.
            order.append(slot)
            var i = len(order) - 1
            while i > 0 and first[order[i]] < first[order[i - 1]]:
                var previous = order[i - 1]
                order[i - 1] = order[i]
                order[i] = previous
                i -= 1
    var empty = finish_composition(
        bundle,
        (
            List[Int128](length=nf, fill=0),
            List[Float64](length=nf, fill=0),
            List[Int](length=nf, fill=0),
        ),
        List[Int64](length=nf * 4, fill=0),
    )
    var keys32 = List[Int32]()
    var keys64 = List[Int64]()
    var key_valid = List[Bool]()
    var columns = List[List[Series]]()
    columns.append(List[Series]())
    for output in bundle.outputs:
        columns.append(List[Series]([empty.column(output.name).slice(0, 0)]))
    for slot in order:
        var integers = List[Int128](length=nf, fill=0)
        var floating = List[Float64](length=nf, fill=0)
        var counts = List[Int](length=nf, fill=0)
        var extrema = List[Int64](length=nf * 4, fill=0)
        var kept = 0
        for block in range(blocks):
            kept += Int(rows[slot * blocks + block])
        for field in range(nf):
            var bins = List[Int128](length=9, fill=0)
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
        var output = finish_composition(
            bundle, (integers^, floating^, counts^), extrema
        )
        # Copy only finalized keys. Returning source slices would keep the
        # entire input key allocation alive behind a tiny grouped result.
        var key = frame.item(Int(first[slot]), plan.key)
        var present = not key.is_null()
        key_valid.append(present)
        if plan.key_dtype == DataType.INT32:
            keys32.append(key.int32() if present else Int32(0))
        else:
            keys64.append(key.int64() if present else Int64(0))
        for i in range(len(bundle.outputs)):
            ref item = bundle.outputs[i]
            if item.name in plan.row_counts:
                columns[i + 1].append(
                    Series(item.name, Column[Int64]([Int64(kept)]))
                )
            else:
                columns[i + 1].append(output.column(item.name))
    var answer = List[Series]()
    if plan.key_dtype == DataType.INT32:
        answer.append(Series(plan.key, Column[Int32](keys32^, key_valid)))
    else:
        answer.append(Series(plan.key, Column[Int64](keys64^, key_valid)))
    for i in range(1, len(columns)):
        answer.append(Series._from_chunks(columns[i]))
    return DataFrame(answer^)
