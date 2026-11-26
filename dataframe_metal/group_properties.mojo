"""Bounded key samples distinguish grouped placement costs.

These samples describe cost only. GPU hashing independently discovers every
actual group, checks its cardinality limit, and computes the complete result.
"""
from std.bit import next_power_of_two
from std.collections import Dict
from std.random.philox import Random
from dataframe import Column, DataFrame, DataType, LazyFrame, Series


def group_properties(region: LazyFrame, key: String) raises -> String:
    ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
    var source = frame.column(key)
    var samples = min(4096, frame.height())
    if not samples:
        return ":sample:empty"
    var counts = Dict[Int64, Int]()
    var nulls = 0
    var hottest = 0
    var random = Random(seed=0x9E3779B97F4A7C15)
    for i in range(samples):
        # Randomized positions avoid aliasing periodic keys with a fixed
        # sampling stride. Small inputs are examined once in natural order.
        var row = i
        if frame.height() > 4096:
            var draw = random.step()
            var value = (UInt64(draw[0]) << 32) | UInt64(draw[1])
            row = Int(value % UInt64(frame.height()))
        var item = source.get(row)
        if item.is_null():
            nulls += 1
        else:
            var value = (
                Int64(item.int32()) if source.dtype()
                == DataType.INT32 else item.int64()
            )
            if value in counts:
                counts[value] += 1
            else:
                counts[value] = 1
            hottest = max(hottest, counts[value])
    var groups = len(counts) + Int(nulls > 0)
    var bucket = next_power_of_two(groups)
    hottest = max(hottest, nulls)
    return (
        ":sample:v1:philox:groups"
        + String(bucket)
        + ":hot"
        + String((hottest * 8 + samples - 1) // samples)
        + ":null"
        + String((nulls * 8 + samples - 1) // samples)
    )


@fieldwise_init
struct GroupSample(Copyable, Movable):
    var distinct: Int
    var hottest: Int
    var nulls: Int
    var samples: Int
    var population: Int

    def global_state(self) -> Bool:
        return self.distinct > 256

    def capacity(self) -> Int:
        if not self.global_state():
            return 512
        var wanted = self.distinct * 2
        if self.samples < self.population:
            # A mostly-distinct sample cannot bound the full cardinality.
            wanted = (
                131072 if self.distinct * 4
                >= self.samples * 3 else self.distinct * 4
            )
        return next_power_of_two(min(max(wanted, 512), 131072))

    def key(self) -> String:
        if not self.samples:
            return ":sample:v2:philox:empty"
        var bucket = next_power_of_two(self.distinct)
        return (
            ":sample:v2:philox:groups"
            + String(bucket)
            + ":hot"
            + String((self.hottest * 8 + self.samples - 1) // self.samples)
            + ":null"
            + String((self.nulls * 8 + self.samples - 1) // self.samples)
            + ":state"
            + ("global" if self.global_state() else "workgroup")
            + ":capacity"
            + String(self.capacity())
        )


def _extend_sample[D: DType](
    source: Series, positions: List[Int], mut codes: List[Int], mut nulls: List[Bool]
) raises -> Int:
    # A pair of the previous tuple's code and the next signed integer identifies
    # a new tuple exactly. Complementing the nonnegative prefix marks a null at
    # this position, leaving every Int64 value available as a real key.
    var seen = Dict[Tuple[Int, Int64], Int]()
    var at = 0
    while at < len(positions):
        # Borrow only the next sampled array. A column may have more physical
        # chunks than there are samples, so building views for all is unbounded.
        var position = positions[at]
        var window = (
            source._chunk_at(position) if source.is_chunked() else (
                source.copy(), position
            )
        )
        var base = position - window[1]
        ref column = window[0]._data[Column[Scalar[D]]]
        var end = base + len(column)
        var constant = column.is_constant()
        while at < len(positions) and positions[at] < end:
            var row = positions[at] - base
            var present = column.constant_valid() if constant else column.is_valid(row)
            var prefix = codes[at]
            var value = Int64(0)
            if present:
                value = Int64(column.constant_value() if constant else column._get(row))
            else:
                prefix = ~prefix
                nulls[at] = True
            var key = (prefix, value)
            var code = seen.get(key, -1)
            if code < 0:
                code = len(seen)
                seen[key] = code
            codes[at] = code
            at += 1
    return len(seen)


def sample_group_tuples(
    frame: DataFrame, keys: List[String]
) raises -> GroupSample:
    var population = frame.height()
    var samples = min(4096, population)
    var columns = List[Series](capacity=len(keys))
    var chunked = False
    for key in keys:
        var column = frame.column(key)
        chunked = chunked or column.is_chunked()
        columns.append(column^)
    var positions = List[Int](capacity=samples)
    var random = Random(seed=0x9E3779B97F4A7C15)
    for i in range(samples):
        var row = i
        if samples < population:
            var draw = random.step()
            var value = (UInt64(draw[0]) << 32) | UInt64(draw[1])
            row = Int(value % UInt64(population))
        positions.append(row)
    # Counts are independent of visit order. Sorting the bounded sample lets
    # each key walk its own physical chunks once, even at different boundaries.
    if chunked and samples < population:
        sort(positions)
    var codes = List[Int](length=samples, fill=0)
    var has_null = List[Bool](length=samples, fill=False)
    var distinct = Int(samples > 0)
    for column in columns:
        if column.dtype() == DataType.INT32:
            distinct = _extend_sample[DType.int32](column, positions, codes, has_null)
        elif column.dtype() == DataType.INT64:
            distinct = _extend_sample[DType.int64](column, positions, codes, has_null)
        else:
            raise Error("Grouped Metal samples require integer keys")
    var counts = List[Int](length=distinct, fill=0)
    var hottest = 0
    var nulls = 0
    for i in range(samples):
        counts[codes[i]] += 1
        hottest = max(hottest, counts[codes[i]])
        nulls += Int(has_null[i])
    return GroupSample(distinct, hottest, nulls, samples, population)


def sample_group_stride_reuse(
    frame: DataFrame, keys: List[String], stride: Int
) raises -> Bool:
    """Bounded dispatch hint only; the cached kernel still compares every key."""
    if stride <= 0 or frame.height() < stride * 4:
        return False
    var columns = List[Series]()
    for name in keys:
        columns.append(frame.column(name))
    var repeated = 0
    var random = Random(seed=0xBADC0FFEE)
    for i in range(128):
        var draw = random.step()
        var value = (UInt64(draw[0]) << 32) | UInt64(draw[1])
        var row = Int(value % UInt64(frame.height() - stride))
        var same = True
        for column in columns:
            var a = column.get(row)
            var b = column.get(row + stride)
            if a.is_null() or b.is_null():
                same = same and a.is_null() == b.is_null()
            elif column.dtype() == DataType.INT32:
                same = same and a.int32() == b.int32()
            else:
                same = same and a.int64() == b.int64()
        repeated += Int(same)
    return repeated >= 96
