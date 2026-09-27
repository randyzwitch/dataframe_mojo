"""Independent lookup and calendar/panel joins. Args: rows shape reps.

The build table has its own domain, independent of the sampled probe keys.
A panel has four sensors for every minute; missing/shuffled panels exercise
rejection. Timed public joins include output gathering.
"""
from std.sys import argv
from std.time import monotonic
from dataframe import Column, DataFrame, DataType, Series


def main() raises:
    var args = argv()
    var n = Int(String(args[1]))
    var shape = String(args[2])
    var reps = Int(String(args[3]))
    var panel = shape == "panel" or shape == "missing" or shape == "shuffled"
    var copies = 4 if panel else 1
    var domain = n // copies
    var step = 1 if shape == "lookup" else 60_000_000
    var base = 1 if shape == "lookup" else 1_700_000_000_000_000
    var dtype = DataType.INT64 if shape == "lookup" else DataType.datetime("us")
    var keys = List[Int64](capacity=n)
    var payload = List[Int64](capacity=n)
    for slot in range(domain):
        for sensor in range(copies):
            if shape == "missing" and slot % 101 == 0 and sensor == 0:
                continue
            keys.append(Int64(base + slot * step))
            payload.append(Int64(sensor))
    var state = UInt64(271269)
    if shape == "shuffled":
        for i in range(len(keys) - 1, 0, -1):
            state = state * 6364136223846793005 + 1442695040888963407
            var j = Int((state >> 16) % UInt64(i + 1))
            var key = keys[i]
            keys[i] = keys[j]
            keys[j] = key
            var value = payload[i]
            payload[i] = payload[j]
            payload[j] = value
    var probes = List[Int64](capacity=n)
    var expected = 0
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var slot = Int((state >> 16) % UInt64(domain + domain // 10))
        probes.append(Int64(base + slot * step))
        if slot < domain:
            expected += copies - Int(shape == "missing" and slot % 101 == 0)
    var left = DataFrame(
        [Series("k", Column[Int64](probes^)).with_dtype(dtype)]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64](keys^)).with_dtype(dtype),
            Series("sensor", Column[Int64](payload^)),
        ]
    )
    for rep in range(reps + 1):
        var start = monotonic()
        var result = left.join(right, "k")
        var elapsed = monotonic() - start
        if result.height() != expected or result.column("k").dtype() != dtype:
            raise Error("wrong join result")
        if rep > 0:
            print(elapsed)
