"""Pure CPU literal witnesses for alignment and diagnostic distances.

Build with -I . -I benchmarks; no GPU context or accelerator is needed.
"""
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, lit
from metal_suite_diagnostics import align, column_metrics, emit_comparison


def source() raises -> DataFrame:
    var keys: List[Int32] = [99, -1, 2]
    var valid: List[Bool] = [False, True, True]
    var floating: List[Float64] = [Float64(1), Float64(2), Float64(3)]
    var integer_mean: List[Float64] = [Float64(4), Float64(5), Float64(6)]
    var columns: List[Series] = [
        Series("k", Column[Int32](keys^, valid^)),
        Series("floating_mean", Column[Float64](floating^)),
        Series("integer_mean", Column[Float64](integer_mean^)),
    ]
    return DataFrame(columns^)


def changed() raises -> DataFrame:
    var keys: List[Int32] = [2, 99, -1]
    var valid: List[Bool] = [True, False, True]
    var floating: List[Float64] = [
        Float64(3),
        Float64(1),
        bitcast[DType.float64](UInt64(0x4000000000000001)),
    ]
    var integer_mean: List[Float64] = [Float64(6), Float64(4), Float64(5)]
    var columns: List[Series] = [
        Series("k", Column[Int32](keys^, valid^)),
        Series("floating_mean", Column[Float64](floating^)),
        Series("integer_mean", Column[Float64](integer_mean^)),
    ]
    return DataFrame(columns^)


def float32_source(altered: Bool) raises -> DataFrame:
    var rows: List[Int32] = [0, 1, 2]
    var values: List[Float32] = [
        Float32(1),
        Float32(0),
        bitcast[DType.float32](UInt32(0x7F800001)),
    ]
    if altered:
        values[0] = bitcast[DType.float32](UInt32(0x3F800001))
        values[1] = bitcast[DType.float32](UInt32(0x80000000))
        values[2] = bitcast[DType.float32](UInt32(0xFF800123))
    var valid: List[Bool] = [True, True, False]
    var columns: List[Series] = [
        Series("row", Column[Int32](rows^)),
        Series("value", Column[Float32](values^, valid^)),
    ]
    return DataFrame(columns^)


def special_source(altered: Bool) raises -> DataFrame:
    var rows: List[Int32] = [0, 1, 2, 3]
    var values: List[Float64] = [
        Float64(7),
        bitcast[DType.float64](UInt64(0x7FF0000000000000)),
        bitcast[DType.float64](UInt64(0x7FF8000000000123)),
        bitcast[DType.float64](UInt64(0x7FF8000000000123)),
    ]
    var valid: List[Bool] = [False, True, True, True]
    if altered:
        valid[0] = True
        values[1] = bitcast[DType.float64](UInt64(0xFFF0000000000000))
        values[2] = Float64(1)
        values[3] = bitcast[DType.float64](UInt64(0xFFF8000000000456))
    var columns: List[Series] = [
        Series("row", Column[Int32](rows^)),
        Series("value", Column[Float64](values^, valid^)),
    ]
    return DataFrame(columns^)


def main() raises:
    var left = source()
    var right = changed()
    var keys: List[String] = ["k"]
    var aligned = align(left, right, keys)
    assert_true(aligned.available)
    assert_equal(aligned.reason, "verified_unique_keys")
    var metrics = column_metrics(aligned.left, aligned.right, "floating_mean")
    assert_equal(metrics.mismatches, 1)
    assert_equal(metrics.null_mismatches, 0)
    assert_equal(metrics.finite_float_mismatches, 1)
    assert_equal(metrics.max_ulp, UInt64(1))
    assert_equal(metrics.max_absolute, (Float64(1) / Float64(2251799813685248)))
    assert_equal(
        metrics.max_relative,
        (Float64(1) / Float64(2251799813685248))
        / bitcast[DType.float64](UInt64(0x4000000000000001)),
    )
    var exact = column_metrics(aligned.left, aligned.right, "integer_mean")
    assert_equal(exact.mismatches, 0)
    assert_equal(exact.finite_float_mismatches, 0)
    var unavailable = align(left, right, List[String]())
    assert_true(not unavailable.available)
    var nonunique: List[String] = ["floating_mean"]
    # Different keys must not become an ordinal numeric comparison.
    var membership = align(left, right, nonunique)
    assert_true(not membership.available)
    assert_equal(membership.reason, "key_membership_differs")
    var duplicate = left.with_columns([lit(Int32(0)).alias("k")])
    var duplicates = align(duplicate, duplicate, keys)
    assert_true(not duplicates.available)
    assert_equal(duplicates.reason, "declared_keys_not_unique")
    var a32 = float32_source(False)
    var b32 = float32_source(True)
    var row_keys: List[String] = ["row"]
    var aligned32 = align(a32, b32, row_keys)
    assert_true(aligned32.available)
    var single = column_metrics(aligned32.left, aligned32.right, "value")
    assert_equal(single.mismatches, 1)
    assert_equal(single.finite_float_mismatches, 1)
    assert_equal(single.null_mismatches, 0)
    assert_equal(single.nonfinite_float_mismatches, 0)
    assert_equal(single.max_ulp, UInt64(1))
    var special_a = special_source(False)
    var special_b = special_source(True)
    var special = column_metrics(special_a, special_b, "value")
    assert_equal(special.mismatches, 3)
    assert_equal(special.null_mismatches, 1)
    assert_equal(special.nonfinite_float_mismatches, 2)
    assert_equal(special.finite_float_mismatches, 0)
    # NaN payload changes and signed zero follow DataFrame.equals semantics.
    emit_comparison(left, right, "cpu_metal", keys, List[String]())
    emit_comparison(left, left, "cpu_repeat", keys, List[String]())
    print("PASS null/negative key alignment and per-column one-ULP diagnostics")
