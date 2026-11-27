"""Size the independent group and value hash tables from a bounded row sample."""
from std.bit import next_power_of_two
from std.collections import Dict
from std.memory import bitcast
from std.random.philox import Random
from dataframe import AnyValue, DataFrame, DataType, Series
from .accelerator.gpu.distinct_value import distinct_canonical
from .distinct_layout import MAX_DISTINCT_CAPACITY
from .group_properties import GroupSample


struct _RowSample(Movable):
    """Read increasing row positions, reusing each physical chunk view."""

    var source: Series
    var part: Series
    var start: Int
    var end: Int

    def __init__(out self, var source: Series):
        self.start = 0
        self.end = 0 if source.is_chunked() else len(source)
        self.part = source.copy()
        self.source = source^

    def get(mut self, row: Int) raises -> AnyValue:
        if row >= self.end:
            var current = self.source._chunk_at(row)
            self.start = row - current[1]
            self.part = current[0].copy()
            self.end = self.start + len(self.part)
        return self.part.get(row - self.start)


def chunked_distinct_capacities(
    frame: DataFrame, key: String, field: String
) raises -> Tuple[Int, Int]:
    var keys = _RowSample(frame.column(key))
    var vals = _RowSample(frame.column(field))
    var dtype = vals.source.dtype()
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
            # One random position in each stratum, in increasing row order.
            var start = i * rows // samples
            var stop = (i + 1) * rows // samples
            row = start + Int(word % UInt64(stop - start))
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
        max(
            512,
            min(
                min(2 * rows, MAX_DISTINCT_CAPACITY), max(16384, 2 * estimated)
            ),
        )
    )
    var ngroups = GroupSample(len(groups), 0, 0, samples, rows).capacity()
    return (ngroups, nvalues)
