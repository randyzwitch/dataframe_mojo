"""Exact ordinal decimal extrema, including null rows and worker merges."""
# contract-group: public.apple_parity
# contract-group: expr.decimal_extrema
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.lifecycle_jobs import WorkerSetting


def fixture(width: Int, scale: Int) raises -> DataFrame:
    var magnitude = Int128(1) << Int128(
        20 if width == 32 else 54 if width == 64 else 90
    )
    var values: List[Int128] = [
        0,
        magnitude,
        -magnitude,
        magnitude + 1,
        -magnitude,
        5,
        Int128.MIN,
        -5,
        magnitude,
    ]
    var valid: List[Bool] = [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
    ]
    var source = Series("x", Column[Int128](values^, valid)).with_dtype(
        DataType.decimal(38, scale)
    )
    if width != 128:
        source = source.cast(
            DataType.decimal(9 if width == 32 else 18, scale, width)
        )
    return DataFrame(
        [source^, Series("g", Column[Int32]([1, 0, 1, 0, 1, 2, 2, 2, 3]))]
    )


def expressions() -> List[Expr]:
    return [col("x").arg_min().alias("lo"), col("x").arg_max().alias("hi")]


def assert_indices(
    actual: DataFrame, minima: List[Int64], maxima: List[Int64]
) raises:
    assert_equal(actual.height(), len(minima))
    assert_true(actual.column("lo").dtype() == DataType.UINT32)
    assert_true(actual.column("hi").dtype() == DataType.UINT32)
    for row in range(len(minima)):
        for name in [String("lo"), String("hi")]:
            var expected = minima[row] if name == "lo" else maxima[row]
            var value = actual.item(row, name)
            assert_equal(value.is_null(), expected < 0)
            if expected >= 0:
                assert_equal(value.numeric[DType.uint32](), UInt32(expected))


def parallel_fixture() raises -> DataFrame:
    var rows = 131073
    var magnitude = Int128(1) << 90
    var values = List[Int128](length=rows, fill=0)
    values[1] = -magnitude
    values[73729] = -magnitude
    values[73728] = magnitude
    values[rows - 1] = magnitude
    return DataFrame(
        [
            Series("x", Column[Int128](values^)).with_dtype(
                DataType.decimal(38, 6)
            )
        ]
    )


def test_decimal_arg_extrema_exact_widths_scale_nulls_groups_and_layouts() raises:
    for width in [32, 64, 128]:
        for scale in [0, 2, 6]:
            var raw = fixture(width, scale)
            for frame in [raw.copy(), offset_layout(raw), chunk_layout(raw)]:
                for streaming in [False, True]:
                    assert_indices(
                        frame.lazy()
                        .select_exprs(expressions())
                        .collect(engine="cpu", streaming=streaming),
                        [2],
                        [3],
                    )
                    assert_indices(
                        frame.lazy()
                        .group_by("g")
                        .agg(expressions())
                        .sort("g")
                        .collect(engine="cpu", streaming=streaming),
                        [0, 1, 2, -1],
                        [1, 1, 0, -1],
                    )
                    assert_indices(
                        frame.head(0)
                        .lazy()
                        .select_exprs(expressions())
                        .collect(engine="cpu", streaming=streaming),
                        [-1],
                        [-1],
                    )
                    assert_indices(
                        frame.head(0)
                        .lazy()
                        .group_by("g")
                        .agg(expressions())
                        .collect(engine="cpu", streaming=streaming),
                        [],
                        [],
                    )
                    assert_indices(
                        frame.tail(1)
                        .lazy()
                        .select_exprs(expressions())
                        .collect(engine="cpu", streaming=streaming),
                        [-1],
                        [-1],
                    )


def test_decimal_arg_extrema_preserve_first_ties_across_parallel_workers() raises:
    var setting = WorkerSetting(2)
    var raw = parallel_fixture()
    for frame in [raw.copy(), offset_layout(raw), chunk_layout(raw)]:
        for streaming in [False, True]:
            assert_indices(
                frame.lazy()
                .select_exprs(expressions())
                .collect(engine="cpu", streaming=streaming),
                [1],
                [73728],
            )
    _ = setting^
