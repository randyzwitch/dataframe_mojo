"""Bounded key samples distinguish grouped placement costs.

These samples describe cost only. GPU hashing independently discovers every
actual group, checks its cardinality limit, and computes the complete result.
"""
from std.collections import Dict
from dataframe import DataType, LazyFrame


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
