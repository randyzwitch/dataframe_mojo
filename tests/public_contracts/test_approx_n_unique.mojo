"""Approximate cardinality scans every row and merges bounded group states."""
# contract-group: reduce.distinct_nulls
from std.math import abs
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, when
from dataframe.approx_distinct import CardinalitySketch
from dataframe.aggregate import Reducer
from dataframe.expr import APPROX_N_UNIQUE, SUM
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.reduction_types_reference import (
    typed_reduction_columns,
    fixture_codes,
    reduction_validity,
)
from tests.support.group_strategy_reference import ScopedEnvironment


def _count(result: DataFrame, name: String = "n") raises -> Int:
    var column = result.column(name)
    assert_equal(column.dtype(), DataType.UINT32)
    assert_equal(column.null_count(), 0)
    assert_equal(len(column), 1)
    return Int(column._data[Column[UInt32]]._get(0))


def test_approx_n_unique_counts_null_nan_zero_and_all_scalar_types() raises:
    for mode in range(3):
        var valid = reduction_validity(mode)
        for source in typed_reduction_columns(valid):
            var distinct = Dict[Int64, Bool]()
            var codes = fixture_codes(source.dtype())
            for row in range(len(codes)):
                distinct[codes[row] if valid[row] else -1] = True
            var expected = len(distinct)
            var raw = DataFrame([source.copy()])
            for frame in [raw.copy(), offset_layout(raw), chunk_layout(raw)]:
                assert_equal(
                    _count(
                        frame.select(
                            col("x").approx_n_unique().alias("n"), batch_size=2
                        )
                    ),
                    expected,
                )
                assert_equal(
                    _count(
                        frame.lazy()
                        .select(col("x").approx_n_unique().alias("n"))
                        .collect(engine="cpu", batch_size=2)
                    ),
                    expected,
                )
            assert_equal(
                _count(
                    raw.clear().select(col("x").approx_n_unique().alias("n"))
                ),
                0,
            )
    var nan = Float64(0) / Float64(0)
    var special = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [nan, nan, -0.0, 0.0, 99, 99],
                    [True, True, True, True, False, False],
                ),
            )
        ]
    )
    assert_equal(
        _count(special.select(col("x").approx_n_unique().alias("n"))), 3
    )
    for n in [0, 1, 17]:
        var nulls = DataFrame([Series.full_null("x", DataType.NULL, n)])
        assert_equal(
            _count(nulls.select(col("x").approx_n_unique().alias("n"))),
            Int(n > 0),
        )


def test_approx_n_unique_grouped_windows_and_guarded_errors() raises:
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 0, 0, 1, 1, 2])),
            Series(
                "x",
                Column[String](
                    ["a", "a", "a", "b", "bad", "bad", "bad"],
                    [True, True, True, True, False, False, False],
                ),
            ),
        ]
    )
    var expression = col("x").approx_n_unique().alias("n")
    for source in [frame.copy(), offset_layout(frame), chunk_layout(frame)]:
        for result in [
            source.group_by("g").agg(expression).sort("g"),
            source.lazy()
            .group_by("g")
            .agg(expression)
            .sort("g")
            .collect(engine="cpu", batch_size=2),
        ]:
            assert_equal(result.column("n").dtype(), DataType.UINT32)
            var counts = result.column("n")._data[Column[UInt32]].copy()
            assert_equal(counts._get(0), 2)
            assert_equal(counts._get(1), 2)
            assert_equal(counts._get(2), 1)
        var window = (
            source.lazy()
            .select(expression.over("g"))
            .collect(engine="cpu", batch_size=2)
        )
        for row in range(7):
            assert_equal(
                window.column("n")._data[Column[UInt32]]._get(row),
                UInt32(1 if row == 6 else 2),
            )
    var guarded = DataFrame(
        [
            Series("keep", Column[Bool]([True, False, True])),
            Series("x", Column[String](["1", "bad", "2"])),
        ]
    )
    var values = when(col("keep")).then(col("x").cast(DataType.INT64)).end()
    assert_equal(
        _count(
            guarded.lazy()
            .select(values.approx_n_unique().alias("n"))
            .collect(engine="cpu", batch_size=2)
        ),
        3,
    )


def test_approx_n_unique_full_input_accuracy_and_parallel_determinism() raises:
    for cardinality in [1, 10, 100, 1000, 10000, 100000]:
        var values = List[Int64]()
        for row in range(cardinality * 2):
            values.append(Int64(row % cardinality))
        var frame = DataFrame([Series("x", Column[Int64](values^))])
        var expected = -1
        for threads in [String("1"), "4"]:
            var scoped = ScopedEnvironment("DATAFRAME_THREADS", threads)
            var observed = _count(
                frame.select(
                    col("x").approx_n_unique().alias("n"), batch_size=127
                )
            )
            if expected < 0:
                expected = observed
            assert_equal(observed, expected)
            # A deterministic fixture budget, not a promised hard error bound.
            assert_true(
                abs(Float64(observed - cardinality))
                <= max(1.0, 0.20 * Float64(cardinality))
            )
            _ = scoped^
    # Values introduced late must be counted, rather than extrapolating the
    # planner statistics' first rows from each vector.
    var late = List[Int64](length=8192, fill=0)
    for row in range(614, len(late)):
        late[row] = Int64(row)
    var result = _count(
        DataFrame([Series("x", Column[Int64](late^))]).select(
            col("x").approx_n_unique().alias("n")
        )
    )
    assert_true(result > 6000)


def test_approx_n_unique_sketch_merge_growth_and_mapping_preserve_state() raises:
    var source = Series("x", Column[Int64]([1, 2, 3, 4, 1, 2]))
    var whole = Reducer(APPROX_N_UNIQUE, DataType.INT64, 2, 0, 0)
    var groups: List[Int] = [0, 0, 0, 1, 1, 1]
    whole.update(source, 0, True, groups)
    var left = Reducer(APPROX_N_UNIQUE, DataType.INT64, 1, 0, 0)
    left.update(source.slice(0, 3), 0, False, List[Int]())
    left.grow(2)
    var right = Reducer(APPROX_N_UNIQUE, DataType.INT64, 1, 0, 0)
    right.update(source.slice(3, 3), 0, False, List[Int]())
    left.merge(right, groups=[Int(1)])
    assert_true(left.finish().equals(whole.finish()))
    var remapped = Reducer(APPROX_N_UNIQUE, DataType.INT64, 1, 0, 0)
    remapped.merge(whole, sources=[Int(1)])
    assert_equal(remapped.finish()._data[Column[UInt32]]._get(0), 3)
    var empty = CardinalitySketch()
    assert_equal(empty.estimate(), 0)
    assert_equal(len(empty.registers), 256)
    var unrelated = Reducer(SUM, DataType.INT64, 1, 0, 0)
    assert_equal(len(unrelated.sketches), 0)
