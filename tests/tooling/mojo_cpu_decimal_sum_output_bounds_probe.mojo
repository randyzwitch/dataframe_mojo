"""Independent CPU literals for successful GROUP output magnitude receipts.

Standalone proof entry; no benchmark query is rewritten. Not yet executed.
"""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, col, lit
from dataframe.dtype import DataType


def pow10(n: Int) -> Int128:
    var value = Int128(1)
    for _ in range(n):
        value *= 10
    return value


def physical_width_is_not_declared_precision() raises:
    var raw = List[Int64]()
    raw.append(Int64.MAX)
    raw.append(Int64.MAX)
    raw.append(Int64.MAX)
    raw.append(Int64.MIN)
    raw.append(Int64.MIN)
    var keys = List[Int32]([0, 0, 1, 2, 2])
    var valid = List[Bool]([True, True, False, True, True])
    var dtype = DataType.decimal(18, 0, 64)
    var frame = DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("v", Column[Int64](raw^, valid^)).with_dtype(dtype),
        ]
    )
    var query = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(col("v").sum(min_count=1).alias("s"))
        .with_columns((col("s") * lit("2").cast(dtype)).alias("p"))
        .sort("k")
    )
    for streaming in [False, True]:
        var result = query.collect(
            engine="cpu", optimize=False, streaming=streaming
        )
        assert_equal(result.height(), 3)
        assert_equal(result.column("s").dtype(), DataType.decimal(38, 0, 128))
        assert_equal(result.column("p").dtype(), DataType.decimal(38, 0, 128))
        var sums = result.column("s").numeric[DType.int128]()
        var products = result.column("p").numeric[DType.int128]()
        assert_equal(sums.value(0), (Int128(1) << 64) - 2)
        assert_equal(products.value(0), (Int128(1) << 65) - 4)
        assert_true(not sums.is_valid(1))
        assert_true(not products.is_valid(1))
        assert_equal(sums.value(2), -(Int128(1) << 64))
        assert_equal(products.value(2), -(Int128(1) << 65))


def successful_sum_does_not_make_later_multiply_infallible() raises:
    var input_dtype = DataType.decimal(18, 18, 64)
    # Logical 2 requires precision19 at scale18. Its raw word fits Int64.
    var factor_dtype = DataType.decimal(19, 18, 128)
    var input_word = Int64(999999999999999999)
    for n in [1, 60, 100]:
        var raw = List[Int64]()
        var keys = List[Int32]()
        for _ in range(n):
            raw.append(input_word)
            keys.append(Int32(0))
        var frame = DataFrame(
            [
                Series("k", Column[Int32](keys^)),
                Series("v", Column[Int64](raw^)).with_dtype(input_dtype),
            ]
        )
        var grouped = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(col("v").sum(min_count=1).alias("s"))
        )
        var multiplied = grouped.with_columns(
            (col("s") * lit("2").cast(factor_dtype)).alias("p")
        )
        for streaming in [False, True]:
            var total = grouped.collect(
                engine="cpu", optimize=False, streaming=streaming
            )
            assert_equal(
                total.column("s").dtype(), DataType.decimal(38, 18, 128)
            )
            var words = total.column("s").numeric[DType.int128]()
            assert_equal(words.value(0), Int128(n) * Int128(input_word))
            if n == 1:
                var result = multiplied.collect(
                    engine="cpu", optimize=False, streaming=streaming
                )
                assert_equal(
                    result.column("p").dtype(), DataType.decimal(38, 36, 128)
                )
                var products = result.column("p").numeric[DType.int128]()
                assert_equal(
                    products.value(0),
                    Int128(input_word) * Int128(2) * pow10(18),
                )
            elif n == 60:
                with assert_raises(
                    contains="value exceeds decimal precision 38"
                ):
                    _ = multiplied.collect(
                        engine="cpu", optimize=False, streaming=streaming
                    )
            else:
                with assert_raises(contains="decimal multiplication overflow"):
                    _ = multiplied.collect(
                        engine="cpu", optimize=False, streaming=streaming
                    )


def q64_scaled_literal_and_comparison_alignment() raises:
    var dtype = DataType.decimal(18, 2, 64)
    var keys = List[Int32]([0, 1, 2])
    var sales = List[Int64]([1000, 1000, Int64.MIN])
    var refunds = List[Int64]([400, 600, Int64.MAX])
    var valid = List[Bool]([True, True, False])
    var frame = DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("sale", Column[Int64](sales^, valid)).with_dtype(dtype),
            Series("refund", Column[Int64](refunds^, valid^)).with_dtype(dtype),
        ]
    )
    var grouped = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                col("sale").sum(min_count=1).alias("sale"),
                col("refund").sum(min_count=1).alias("refund"),
            ]
        )
    )
    var factor = lit("2").cast(dtype)
    var query = grouped.filter(col("sale") > factor * col("refund")).select(
        ["k"]
    )
    for streaming in [False, True]:
        var products = grouped.with_columns(
            (factor * col("refund")).alias("p")
        ).collect(engine="cpu", optimize=False, streaming=streaming)
        assert_equal(products.column("p").dtype(), DataType.decimal(38, 4, 128))
        var words = products.column("p").numeric[DType.int128]()
        assert_equal(words.value(0), Int128(80000))
        assert_equal(words.value(1), Int128(120000))
        assert_true(not words.is_valid(2))
        var result = query.collect(
            engine="cpu", optimize=False, streaming=streaming
        )
        assert_equal(result.height(), 1)
        var actual = result.column("k").numeric[DType.int32]()
        assert_equal(actual.value(0), Int32(0))


def main() raises:
    var failures = 0
    try:
        physical_width_is_not_declared_precision()
        print("PASS physical_width_is_not_declared_precision")
    except e:
        failures += 1
        print("FAIL physical_width_is_not_declared_precision:", e)
    try:
        successful_sum_does_not_make_later_multiply_infallible()
        print("PASS successful_sum_does_not_make_later_multiply_infallible")
    except e:
        failures += 1
        print("FAIL successful_sum_does_not_make_later_multiply_infallible:", e)
    try:
        q64_scaled_literal_and_comparison_alignment()
        print("PASS q64_scaled_literal_and_comparison_alignment")
    except e:
        failures += 1
        print("FAIL q64_scaled_literal_and_comparison_alignment:", e)
    if failures:
        raise Error("GROUP output bounds CPU literal proof failed")
