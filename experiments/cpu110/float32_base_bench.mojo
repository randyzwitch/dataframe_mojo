from std.memory import bitcast
from std.time import monotonic

from dataframe.parse import parse_float64


comptime ITERATIONS = 20000000


def timed(fields: List[String]) raises -> Int:
    var total = UInt32(0)
    var start = monotonic()
    for i in range(ITERATIONS):
        total ^= bitcast[DType.uint32](
            Float32(parse_float64(StringSlice(fields[i & 15])))
        )
    if total == UInt32.MAX:
        raise Error("unexpected checksum")
    return monotonic() - start


def main() raises:
    var short: List[String] = [
        "1.25",
        "-2.5",
        "123.456",
        "-7654.321",
        "0.000001",
        "-0.00001",
        "16777.216",
        "-12345.67",
        "42",
        "-99.9",
        "0.125",
        "987.6543",
        "-0.75",
        "1234.567",
        "12.34567",
        "-456.789",
    ]
    var fallback: List[String] = [
        "1.0000000596046448",
        "-1.0000000596046448",
        "1.2345678901234567",
        "-1.2345678901234567",
        "1e-20",
        "-1e20",
        "16777216.01",
        "-16777216.01",
        "0.00000001",
        "-0.00000001",
        "99999999.99",
        "-99999999.99",
        "3.141592653589793",
        "-2.718281828459045",
        "1.7976931348623157e308",
        "5e-324",
    ]
    print("short_ns,", timed(short), sep="")
    print("fallback_ns,", timed(fallback), sep="")
