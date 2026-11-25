"""Grouped means preserve the public floating-point result across native spans."""
# contract-group: execution.backends
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, col
from tests.support.metal_outcomes import require_metal


def _split[
    D: DType
](
    name: String, values: List[Scalar[D]], valid: List[Bool], at: Int
) raises -> Series:
    return Series._from_chunks(
        [
            Series(name, Column[Scalar[D]](values.copy(), valid.copy())).slice(
                0, at
            ),
            Series(name, Column[Scalar[D]](values.copy(), valid.copy())).slice(
                at, len(values) - at
            ),
        ]
    )


# requires: metal
def test_metal_fractional_grouped_means_preserve_rounding_nulls_and_order() raises:
    var runtime = require_metal()
    var high = (Int64(1) << 40) + 7
    var keys: List[Int64] = [high, 7, high, 7, high, 0, 9, 0, 11, 9, 11]
    var key_valid = List[Bool](length=len(keys), fill=True)
    key_valid[5] = False
    key_valid[7] = False
    var inf = bitcast[DType.float64](UInt64(0x7FF0000000000000))
    var tiny = bitcast[DType.float64](UInt64(1))
    var large = bitcast[DType.float64](UInt64(0x4630000000000000))
    var doubles: List[Float64] = [
        large,
        -tiny,
        1,
        0,
        -large,
        inf,
        0,
        -inf,
        0,
        0,
        6,
    ]
    var d_valid = List[Bool](length=len(keys), fill=True)
    d_valid[6] = False
    d_valid[9] = False
    d_valid[8] = False
    var max32 = bitcast[DType.float32](UInt32(0x7F7FFFFF))
    var sub32 = bitcast[DType.float32](UInt32(1))
    var quiet = bitcast[DType.float32](UInt32(0x7FC12345))
    var singles: List[Float32] = [
        max32,
        sub32,
        1,
        sub32,
        -max32,
        quiet,
        2,
        1,
        3,
        6,
        5,
    ]
    var s_valid = List[Bool](length=len(keys), fill=True)
    s_valid[8] = False
    s_valid[10] = False
    var ints = List[Int32]()
    var wide = List[Int64]()
    for row in range(len(keys)):
        ints.append(Int32(1 + row))
        wide.append((Int64(1) << 40) + Int64(row))
    var all_valid = List[Bool](length=len(keys), fill=True)
    var frame = DataFrame(
        [
            _split[DType.int64]("key", keys, key_valid, 5),
            _split[DType.float64]("d", doubles, d_valid, 3),
            _split[DType.float32]("f", singles, s_valid, 7),
            _split[DType.int32]("n", ints, all_valid, 4),
            _split[DType.int64]("w", wide, all_valid, 9),
        ]
    )
    for source in [frame.copy(), frame.rechunk()]:
        var plan = (
            source.lazy()
            .group_by("key", maintain_order=True)
            .agg(
                [
                    col("d").mean().alias("d"),
                    col("f").mean().alias("f"),
                    col("n").mean().alias("n"),
                    col("w").mean().alias("w"),
                ]
            )
        )
        var cpu = plan.collect(engine="cpu")
        var prof = plan.profile(accelerator=runtime)
        ref result = prof[0]
        ref profile = prof[1]
        assert_equal(result.height(), 5)
        assert_true(result.column("key").equals(cpu.column("key")))
        assert_equal(result.item(0, "key").int64(), high)
        assert_equal(result.item(1, "key").int64(), Int64(7))
        assert_true(result.item(2, "key").is_null())
        assert_equal(result.item(3, "key").int64(), Int64(9))
        assert_equal(result.item(4, "key").int64(), Int64(11))
        var completed = Int64(0)
        for row in range(profile.height()):
            assert_equal(
                profile.item(row, "fallback_reason").string(), String("")
            )
            var launches = profile.item(row, "gpu_launches").int64()
            assert_equal(
                launches, profile.item(row, "gpu_completed_launches").int64()
            )
            if launches:
                assert_equal(
                    profile.item(row, "result_device").string(), String("metal")
                )
                assert_equal(
                    profile.item(row, "device_status").string(),
                    String("completed"),
                )
            completed += launches
        assert_true(completed > 0)
        for field in ["d", "f", "n", "w"]:
            for row in range(result.height()):
                var expected = cpu.item(row, field)
                var actual = result.item(row, field)
                assert_equal(actual.is_null(), expected.is_null())
                if expected.is_null():
                    continue
                if isnan(expected.float64()):
                    assert_true(isnan(actual.float64()))
                else:
                    assert_equal(
                        bitcast[DType.uint64](actual.float64()),
                        bitcast[DType.uint64](expected.float64()),
                    )
        assert_equal(
            bitcast[DType.uint64](result.item(0, "d").float64()), UInt64(0)
        )
        assert_equal(
            bitcast[DType.uint64](result.item(1, "d").float64()),
            UInt64(0x8000000000000000),
        )
        assert_true(isnan(result.item(2, "d").float64()))
        assert_true(result.item(3, "d").is_null())
        assert_equal(result.item(4, "d").float64(), Float64(6))
        assert_equal(result.item(1, "f").float64(), Float64(sub32))
        assert_true(isnan(result.item(2, "f").float64()))
        assert_true(result.item(4, "f").is_null())
