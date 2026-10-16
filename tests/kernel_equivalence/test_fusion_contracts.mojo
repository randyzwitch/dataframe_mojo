"""Fusion eligibility follows resolved types; enabled/disabled paths obey scalar math."""
# contract-group: kernel.row_ops
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe.binding import bind
from dataframe.execution import evaluate, FUSED_CONTIGUOUS_ROWS
from dataframe.fusion import fused
from tests.support.coverage_compare import assert_float
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_untyped_float_adoption_preserves_fusion_and_leaf_comparisons_use_bitmap_path() raises:
    var values: List[Float64] = [-2, 0, 1, 2, 4, 99]
    var valid: List[Bool] = [True, True, True, True, True, False]
    var raw = DataFrame(
        [
            Series("x", Column[Float64](values.copy(), valid.copy())),
            Series("f32", Column[Float32]([-2, 0, 1, 2, 4, 99], valid.copy())),
            Series("i", Column[Int64]([-2, 0, 1, 2, 4, 99], valid.copy())),
        ]
    )
    var expressions: List[Expr] = [
        (col("x") * 2 + 1) > 3,
        (col("x") * lit(Float64(2)) + lit(Float64(1))) > lit(Float64(3)),
        col("x") > 1,
        (col("f32") * 2 + 1) > 3,
        (col("i") * 2 + 1) > 3,
    ]
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for operation in range(len(expressions)):
            for fuse in [False, True]:
                var bound = bind(
                    expressions[operation].copy().alias("r"),
                    frame._columns,
                    fuse=fuse,
                )
                assert_equal(
                    bound.fusible[len(bound.fusible) - 1],
                    fuse and operation < 2,
                )
                assert_true(
                    bound.dtypes[len(bound.dtypes) - 1] == DataType.BOOL
                )
                for batch in [1, 3, 64]:
                    var actual = evaluate(
                        bound, frame._columns, frame.height(), batch_size=batch
                    )
                    assert_true(actual.dtype() == DataType.BOOL)
                    assert_equal(len(actual), len(values))
                    for row in range(len(values)):
                        assert_equal(actual.get(row).is_null(), not valid[row])
                        if valid[row]:
                            assert_equal(
                                actual.get(row).bool(), values[row] > 1
                            )


def _fusion_source(n: Int) raises -> DataFrame:
    var x = List[Float64](length=7, fill=999)
    var y = List[Float64](length=3, fill=999)
    var xv = List[Bool](length=7, fill=False)
    var yv = List[Bool](length=3, fill=False)
    for row in range(n):
        x.append(Float64(row % 17 - 8) / 4)
        y.append(Float64(row % 13 - 6) / 2)
        xv.append(row % 5 != 1)
        yv.append(row % 7 != 2)
    var a = Series("x", Column[Float64](x^, xv^)).slice(7, n)
    var b = Series("y", Column[Float64](y^, yv^)).slice(3, n)
    var cut = n // 2
    return DataFrame(
        [
            Series._from_chunks([a.slice(0, cut), a.slice(cut, n - cut)]),
            Series._from_chunks(
                [b.slice(0, cut + 1), b.slice(cut + 1, n - cut - 1)]
            ),
        ]
    )


def _check_fusion(
    actual: Series, offset: Int, count: Int, comparison: Bool
) raises:
    assert_equal(len(actual), count)
    assert_true(
        actual.dtype() == (DataType.BOOL if comparison else DataType.FLOAT64)
    )
    for position in range(count):
        var row = position + offset
        var valid = row % 5 != 1 and row % 7 != 2
        assert_equal(actual.get(position).is_null(), not valid)
        if valid:
            var x = Float64(row % 17 - 8) / 4
            var y = Float64(row % 13 - 6) / 2
            var expected = x * x - y * y
            if comparison:
                assert_equal(actual.get(position).bool(), expected > 0)
            else:
                assert_float(actual.get(position).float64(), expected)


def test_fused_repeated_sources_and_interior_chunk_windows_match_scalar_math_at_all_lane_widths() raises:
    var frame = _fusion_source(65)
    for comparison in [False, True]:
        var expression = col("x") * col("x") - col("y") * col("y")
        if comparison:
            expression = expression > lit(Float64(0))
        var bound = bind(expression, frame._columns)
        assert_true(bound.fusible[len(bound.fusible) - 1])
        for offset in [0, 1, 7, 31, 33]:
            var count = min(29, 65 - offset)
            comptime for index in range(3):
                comptime width = (1, 4, 8)[index]
                _check_fusion(
                    fused[width](
                        bound,
                        frame._columns,
                        len(bound.expr._nodes) - 1,
                        offset,
                        count,
                    ),
                    offset,
                    count,
                    comparison,
                )


def test_fused_chunk_reassembly_threshold_preserves_repeated_source_cells() raises:
    for n in [
        FUSED_CONTIGUOUS_ROWS - 1,
        FUSED_CONTIGUOUS_ROWS,
        FUSED_CONTIGUOUS_ROWS + 1,
    ]:
        var frame = _fusion_source(n)
        for batch in [1024, 65536]:
            var actual = frame.with_columns(
                (col("x") * col("x") - col("y") * col("y")).alias("out"),
                batch_size=batch,
            )
            _check_fusion(actual.column("out"), 0, n, False)
