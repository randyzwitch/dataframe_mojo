"""Standalone EmberJSON rounding/range/performance reproducer; no dataframe.

Float64 timings diagnose the narrower-target overhead, not a valid substitute
for correctly rounded Float32. The first assertions demonstrate why a plain
Float64-to-Float32 cast is insufficient.
"""
from emberjson import Parser, from_json
from std.memory import bitcast
from std.testing import assert_equal
from std.time import perf_counter_ns


def _rounding() raises:
    var above = String("1.000000059604644775390625000001")
    assert_equal(
        bitcast[DType.uint32](from_json[Float32](above)), UInt32(0x3F800001)
    )
    assert_equal(
        bitcast[DType.uint32](from_json[Float64](above).cast[DType.float32]()),
        UInt32(0x3F800000),
    )
    var cases = List[String](
        [
            "1.000000059604644775390625",
            "-1.000000059604644775390625000001",
            "7.00649232162408535461864791644958065640130970938257885878534141944895541342930300743319094181060791015625e-46",
            "7.006492321624085354618647916449580656401309709382579e-46",
            "1.17549435082228750796873653722224568e-38",
            "340282356779733661637539395458142568447",
            "-0.0",
            "1e-200",
        ]
    )
    var expected: List[UInt32] = [
        0x3F800000,
        0xBF800001,
        0,
        1,
        0x00800000,
        0x7F7FFFFF,
        0x80000000,
        0,
    ]
    for i in range(len(cases)):
        assert_equal(
            bitcast[DType.uint32](from_json[Float32](cases[i])), expected[i]
        )
    print("rounding guards PASS")


@no_inline
def _run[D: DType](values: List[String]) raises -> UInt64:
    var checksum = UInt64(0)
    for _ in range(10000):
        for text in values:
            var parser = Parser(StringSlice(text))
            var value = parser.expect_float[D]()
            comptime if D == DType.float32:
                checksum += UInt64(bitcast[DType.uint32](value))
            else:
                checksum += bitcast[DType.uint64](value)
    return checksum


def _timings(name: String, values: List[String]) raises:
    var expected32 = _run[DType.float32](values)
    var expected64 = _run[DType.float64](values)
    for run in range(8):
        # Alternate order; each type has its own checked checksum.
        for slot in range(2):
            var f32 = (run + slot) % 2 == 0
            var start = perf_counter_ns()
            var checksum = _run[DType.float32](values) if f32 else _run[
                DType.float64
            ](values)
            var elapsed = perf_counter_ns() - start
            assert_equal(checksum, expected32 if f32 else expected64)
            if run:
                print(
                    name,
                    "float32" if f32 else "float64",
                    len(values) * 10000,
                    elapsed,
                    checksum,
                )


def main() raises:
    _rounding()
    # These are legitimate JSON errors. A CSV adapter needs to distinguish
    # range overflow (infinity is accepted) from malformed input.
    for text in List[String](
        [
            "1e40",
            "1e400",
            "340282356779733661637539395458142568448",
            "1e400x",
            "1.",
            ".5",
            "+1.5",
        ]
    ):
        try:
            _ = from_json[Float32](text)
            print("accepted", text)
        except error:
            print("diagnostic", text, error.kind, error.message)
    _timings(
        "normal",
        List[String](
            ["1.25", "-37.125", "1348.123", "0.1", "123456.75", "0.00012345"]
        ),
    )
    _timings(
        "scientific",
        List[String](
            ["1e-30", "2.7182818e5", "-1.2e12", "3.4e38", "9.8765E-20", "5e-7"]
        ),
    )
    _timings(
        "subnormal",
        List[String](
            [
                "1e-45",
                "1e-46",
                "1.1754942e-38",
                "1.401298464324817e-45",
                "7.006492321624085e-46",
                "-1e-40",
            ]
        ),
    )
