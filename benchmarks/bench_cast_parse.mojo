"""Measure public String-to-number casts, including column access and output.

Build this file against two source trees with ``mojo build -I <tree>`` and run
binaries alternately on an idle machine. Each timing covers 12 casts of a
100,000-row String column; input construction is outside the timed section.
"""
from std.time import monotonic
from dataframe import Column, DataType, Series
from dataframe.cast import cast_series

comptime ROWS = 100000
comptime REPS = 12


def run_case(fields: List[String], target: DataType) raises -> Int:
    var input = Series("s", Column[String](fields.copy()))
    var mask = List[Bool]()
    var checksum = 0
    var start = monotonic()
    for _ in range(REPS):
        var output = cast_series(input, target, True, 0, mask)
        checksum += len(output)
    var elapsed = monotonic() - start
    if checksum != ROWS * REPS:
        raise Error("bad cast checksum")
    return elapsed


def main() raises:
    var short_ints = List[String](capacity=ROWS)
    var wide_ints = List[String](capacity=ROWS)
    var decimals = List[String](capacity=ROWS)
    var exponents = List[String](capacity=ROWS)
    var short = ["1", "2", "3", "4", "5", "6", "7", "8"]
    var wide = [
        "123456789012345678",
        "123456789012345679",
        "123456789012345680",
        "123456789012345681",
        "123456789012345682",
        "123456789012345683",
        "123456789012345684",
        "123456789012345685",
    ]
    var decimal = [
        "1.25",
        "2.5",
        "3.75",
        "4.125",
        "5.5",
        "6.75",
        "7.25",
        "8.875",
    ]
    var exponent = [
        "1.25e7",
        "2.5e7",
        "3.75e7",
        "4.125e7",
        "5.5e7",
        "6.75e7",
        "7.25e7",
        "8.875e7",
    ]
    for i in range(ROWS):
        short_ints.append(short[i & 7])
        wide_ints.append(wide[i & 7])
        decimals.append(decimal[i & 7])
        exponents.append(exponent[i & 7])
    print("int64_short_ns,", run_case(short_ints, DataType.INT64), sep="")
    print("int64_wide_ns,", run_case(wide_ints, DataType.INT64), sep="")
    print("float64_plain_ns,", run_case(decimals, DataType.FLOAT64), sep="")
    print("float64_exponent_ns,", run_case(exponents, DataType.FLOAT64), sep="")
    print("float32_plain_ns,", run_case(decimals, DataType.FLOAT32), sep="")
