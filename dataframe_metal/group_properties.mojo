"""Bounded key samples distinguish grouped placement costs.

These samples describe cost only. GPU hashing independently discovers every
actual group, checks its cardinality limit, and computes the complete result.
"""
from std.collections import Dict
from dataframe import DataFrame, DataType, LazyFrame, Series


def group_properties(region: LazyFrame, key: String) raises -> String:
    ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
    var source = frame.column(key)
    var samples = min(4096, frame.height())
    if not samples:
        return ":sample:empty"
    var counts = Dict[Int64, Int]()
    var nulls = 0
    var hottest = 0
    var random = UInt64(0x9E3779B97F4A7C15)
    for i in range(samples):
        # Randomized positions avoid aliasing periodic keys with a fixed
        # sampling stride. Small inputs are examined once in natural order.
        random = random * UInt64(6364136223846793005) + UInt64(1)
        var row = i if frame.height() <= 4096 else Int(
            (random >> 32) % UInt64(frame.height())
        )
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
    var bucket = 1
    while bucket < groups:
        bucket <<= 1
    hottest = max(hottest, nulls)
    return (
        ":sample:v1:groups"
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

    def global_state(self) -> Bool:
        return self.distinct > 256

    def capacity(self) -> Int:
        if not self.global_state():
            return 512
        var wanted = self.distinct * 2
        if self.samples == 4096:
            # A mostly-distinct sample cannot bound the full cardinality.
            wanted = (
                131072 if self.distinct * 4
                >= self.samples * 3 else self.distinct * 4
            )
        var capacity = 512
        while capacity < wanted and capacity < 131072:
            capacity <<= 1
        return capacity

    def key(self) -> String:
        if not self.samples:
            return ":sample:v2:empty"
        var bucket = 1
        while bucket < self.distinct:
            bucket <<= 1
        return (
            ":sample:v2:groups"
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


def sample_group_tuples(
    frame: DataFrame, keys: List[String]
) raises -> GroupSample:
    var columns = List[Series]()
    for key in keys:
        columns.append(frame.column(key))
    var samples = min(4096, frame.height())
    var counts = Dict[String, Int]()
    var nulls = 0
    var hottest = 0
    var random = UInt64(0x9E3779B97F4A7C15)
    for i in range(samples):
        random = random * UInt64(6364136223846793005) + UInt64(1)
        var row = i if frame.height() <= 4096 else Int(
            (random >> 32) % UInt64(frame.height())
        )
        var identity = String()
        var has_null = False
        for column in columns:
            var item = column.get(row)
            if item.is_null():
                identity += ":null"
                has_null = True
            else:
                var value = (
                    Int64(item.int32()) if column.dtype()
                    == DataType.INT32 else item.int64()
                )
                identity += ":v" + String(value)
        if has_null:
            nulls += 1
        if identity in counts:
            counts[identity] += 1
        else:
            counts[identity] = 1
        hottest = max(hottest, counts[identity])
    return GroupSample(len(counts), hottest, nulls, samples)
