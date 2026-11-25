"""Size the independent group and value hash tables from a bounded row sample."""
from std.bit import next_power_of_two
from std.collections import Dict
from std.memory import bitcast
from std.random.philox import Random
from dataframe import DataFrame, DataType
from .accelerator.gpu.distinct_value import distinct_canonical
from .distinct_layout import MAX_DISTINCT_CAPACITY
from .group_properties import GroupSample


def chunked_distinct_capacities(
    frame: DataFrame, key: String, field: String
) raises -> Tuple[Int, Int]:
    var keys = frame.column(key)
    var vals = frame.column(field)
    var dtype = vals.dtype()
    var rows = frame.height()
    var samples = min(4096, rows)
    var groups = Dict[Int64, Int]()
    var pairs = Dict[UInt128, Int]()
    var random = Random(seed=0x9E3779B97F4A7C15)
    var singles = 0
    var doubles = 0
    for i in range(samples):
        var row = i
        if samples < rows:
            var draw = random.step()
            var word = (UInt64(draw[0]) << 32) | UInt64(draw[1])
            row = Int(word % UInt64(rows))
        var k = keys.get(row)
        # 2^32 is distinct from every valid unsigned 32-bit key.
        var key_id = Int64(1) << 32
        if not k.is_null():
            key_id = Int64(bitcast[DType.uint32](k.int32()))
        groups[key_id] = 1
        var v = vals.get(row)
        var pair = UInt128(key_id) << 65
        if not v.is_null():
            pair |= UInt128(1) << 64
            var value = UInt64(0)
            if dtype == DataType.INT64:
                value = bitcast[DType.uint64](v.int64())
            elif dtype == DataType.INT32:
                value = UInt64(bitcast[DType.uint32](v.int32()))
            elif dtype == DataType.FLOAT32:
                value = distinct_canonical(
                    UInt64(bitcast[DType.uint32](v.float32())), UInt32(2)
                )
            else:
                value = UInt64(v.bool())
            pair |= UInt128(value)
        var seen = pairs.get(pair, 0)
        pairs[pair] = seen + 1
        singles += Int(seen == 0) - Int(seen == 1)
        doubles += Int(seen == 1) - Int(seen == 2)

    var estimated = len(pairs)
    if samples < rows:
        # Bias-corrected Chao1 estimates unobserved pairs from singletons
        # and doubletons. It is a sizing hint; GPU overflow remains checked.
        if singles == samples:
            estimated = rows
        else:
            estimated += (singles * (singles - 1)) // (2 * (doubles + 1))
    var nvalues = next_power_of_two(
        max(512, min(MAX_DISTINCT_CAPACITY, 2 * min(rows, estimated)))
    )
    var ngroups = GroupSample(len(groups), 0, 0, samples, rows).capacity()
    return (ngroups, nvalues)

