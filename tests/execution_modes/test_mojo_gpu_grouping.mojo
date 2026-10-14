"""Direct Mojo grouped reductions, nullable keys and stable representatives."""

from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    cov,
    corr,
    lit,
)
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_moments import near


def _expressions() -> List[Expr]:
    return [
        col("x").sum().alias("sum"),
        col("x").mean().alias("mean"),
        col("x").min().alias("min"),
        col("x").max().alias("max"),
        col("x").count().alias("count"),
        col("x").len().alias("length"),
        col("x").null_count().alias("nulls"),
        col("x").first().alias("first"),
        col("x").last().alias("last"),
        col("x").arg_min().alias("arg_min"),
        col("x").arg_max().alias("arg_max"),
        col("flag").any(ignore_nulls=False).alias("any"),
        col("flag").all(ignore_nulls=False).alias("all"),
        col("x").var().alias("variance"),
        col("x").std().alias("standard"),
        col("x").skew(False).alias("skew"),
        col("x").kurtosis(True, False).alias("kurtosis"),
        cov(col("x"), col("y")).alias("covariance"),
        corr(col("x"), col("y")).alias("correlation"),
    ]


def _check(query: LazyFrame) raises -> DataFrame:
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    near(actual, query.collect(engine="cpu", streaming=False))
    return actual^


def _layout_case[D: DType](n: Int, chunked: Bool) raises:
    var keys = List[Scalar[D]]()
    var second = List[Int32]()
    var xs = List[Int64]()
    var ys = List[Float32]()
    var flags = List[Bool]()
    var kv = List[Bool]()
    var sv = List[Bool]()
    var xv = List[Bool]()
    var yv = List[Bool]()
    for row in range(n + 6):
        var key = Scalar[D](row % 5)
        comptime if D == DType.float32:
            var encodings: List[UInt32] = [
                0,
                0x80000000,
                0x7FC00001,
                0xFFC00002,
                0x3F800000,
            ]
            key = bitcast[DType.float32](encodings[row % 5]).cast[D]()
        elif D == DType.float64:
            var encodings: List[UInt64] = [
                0,
                0x8000000000000000,
                0x7FF8000000000001,
                0xFFF8000000000002,
                0x3FF0000000000000,
            ]
            key = bitcast[DType.float64](encodings[row % 5]).cast[D]()
        keys.append(key)
        second.append(Int32(row % 3))
        xs.append(Int64(row % 11 - 5))
        ys.append(Float32(row % 11 - 5) * 2 + Float32(row % 3))
        flags.append(row % 2 == 0)
        kv.append(row % 13 != 0)
        sv.append(row % 17 != 0)
        xv.append(row % 7 != 0)
        yv.append(row % 5 != 0)
    var columns = List[Series](
        [
            Series("k", Column[Scalar[D]](keys^, kv^)),
            Series("j", Column[Int32](second^, sv^)),
            Series("x", Column[Int64](xs^, xv.copy())),
            Series("y", Column[Float32](ys^, yv^)),
            Series("flag", Column[Bool](flags^, xv^)),
        ]
    )
    for i in range(len(columns)):
        columns[i] = columns[i].slice(3, n)
        if chunked:
            columns[i] = Series._from_chunks(
                [
                    columns[i].slice(0, n // 2),
                    columns[i].slice(n // 2, 0),
                    columns[i].slice(n // 2, n - n // 2),
                ]
            )
    var frame = DataFrame(columns^)
    var result = _check(
        frame.lazy()
        .group_by(["k", "j"], maintain_order=True)
        .agg(_expressions())
    )
    assert_true(result.height() <= n)
    if n == 0:
        assert_equal(result.height(), 0)
    else:
        var total = Int64(0)
        for row in range(result.height()):
            total += result.item(row, "length").int64()
        assert_equal(total, Int64(n))


def test_mojo_gpu_grouping_mixed_keys_all_reductions_and_nullable_layouts() raises:
    comptime types = (DType.int8, DType.uint64, DType.float32, DType.float64)
    comptime for i in range(len(types)):
        comptime D = types[i]
        for n in List[Int]([0, 1, 1027]):
            _layout_case[D](n, False)
            _layout_case[D](n, True)


def test_mojo_gpu_grouping_key_only_stability_distinct_groups_and_compositions() raises:
    var keys = List[UInt64]()
    for row in range(1027):
        keys.append(UInt64.MAX - UInt64(row))
    var frame = DataFrame([Series("k", Column[UInt64](keys.copy()))])
    var result = _check(
        frame.lazy().group_by("k", maintain_order=True).agg(List[Expr]())
    )
    assert_equal(result.height(), 1027)
    assert_true(result.column("k").equals(frame.column("k")))
    _ = _check(frame.lazy().group_by("k").agg(List[Expr]()).sort("k"))
    var source = DataFrame(
        [
            Series(
                "k",
                Column[Bool](
                    [True, False, True, False, True, False],
                    [True, True, False, True, True, False],
                ),
            ),
            Series("x", Column[Int64]([4, 3, 2, 1, 0, -1])),
        ]
    )
    result = _check(
        source.lazy()
        .group_by("k", maintain_order=True)
        .agg([col("x").mean().alias("total"), col("x").len().alias("length")])
        .filter(col("length") > lit(Int64(1)))
        .sort("total")
        .select_exprs([col("k"), col("total") * lit(Float64(2))])
    )
    assert_equal(result.height(), 3)


def test_mojo_gpu_grouping_exact_wide_sum_local_arg_positions_and_overflow() raises:
    var source = DataFrame(
        [
            Series("k", Column[Int32]([2, 1, 2, 1, 2, 1])),
            Series(
                "x", Column[Int64]([Int64.MAX, 8, Int64.MAX, 2, -Int64.MAX, 4])
            ),
        ]
    )
    var result = _check(
        source.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                col("x").sum().alias("sum"),
                col("x").arg_min().alias("min_at"),
                col("x").arg_max().alias("max_at"),
            ]
        )
    )
    assert_equal(result.item(0, "sum").int64(), Int64.MAX)
    assert_equal(result.item(0, "min_at").numeric[DType.uint32](), UInt32(2))
    assert_equal(result.item(0, "max_at").numeric[DType.uint32](), UInt32(0))
    assert_equal(result.item(1, "min_at").numeric[DType.uint32](), UInt32(1))
    var overflow = DataFrame(
        [
            Series("k", Column[Int32]([0, 0])),
            Series("x", Column[UInt64]([UInt64.MAX, 1])),
        ]
    )
    with assert_raises(contains="overflow"):
        _ = (
            overflow.lazy()
            .group_by("k")
            .agg(col("x").sum())
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
    _ = _check(overflow.lazy().group_by("k").agg(col("x").sum(min_count=3)))
