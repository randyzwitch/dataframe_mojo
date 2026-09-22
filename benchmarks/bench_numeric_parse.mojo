"""Float parser microbenchmark: 10 million conversions per input family.

Build against each revision with `mojo build -I .` and compare binaries on
an idle machine. This isolates conversion; CSV results are measured with
bench_vs_polars.mojo. Inputs vary to prevent constant folding.
"""
from std.time import monotonic

from dataframe.parse import parse_float64


comptime ITERATIONS = 10000000


def timed(fields: List[String]) raises -> Int:
    var total = Float64(0)
    var start = monotonic()
    for i in range(ITERATIONS):
        total += parse_float64(StringSlice(fields[i & 15]))
    var elapsed = monotonic() - start
    if total == Float64(0):
        raise Error("unexpected checksum")
    return elapsed


def main() raises:
    var plain: List[String] = [
        "123.456",
        "234.567",
        "345.678",
        "456.789",
        "567.891",
        "678.912",
        "789.123",
        "891.234",
        "912.345",
        "123.457",
        "234.568",
        "345.679",
        "456.781",
        "567.892",
        "678.913",
        "789.124",
    ]
    var wide: List[String] = [
        "49.970000000000006",
        "59.970000000000006",
        "69.970000000000006",
        "79.970000000000006",
        "89.970000000000006",
        "99.970000000000006",
        "19.970000000000006",
        "29.970000000000006",
        "39.970000000000006",
        "49.980000000000004",
        "59.980000000000004",
        "69.980000000000004",
        "79.980000000000004",
        "89.980000000000004",
        "99.980000000000004",
        "19.980000000000004",
    ]
    var exponent: List[String] = [
        "1.2345e7",
        "2.3456e7",
        "3.4567e7",
        "4.5678e7",
        "5.6789e7",
        "6.7891e7",
        "7.8912e7",
        "8.9123e7",
        "9.1234e7",
        "1.3456e7",
        "2.4567e7",
        "3.5678e7",
        "4.6789e7",
        "5.7891e7",
        "6.8912e7",
        "7.9123e7",
    ]
    print("plain_ns,", timed(plain), sep="")
    print("wide_ns,", timed(wide), sep="")
    print("exponent_ns,", timed(exponent), sep="")
