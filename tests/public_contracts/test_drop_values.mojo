"""Length-changing expression filters preserve sequence and group boundaries."""
# contract-group: expr.order_windows
# contract-group: reduce.grouped
from std.testing import assert_equal, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    ListColumn,
    StructColumn,
    col,
    lit,
    when,
)
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.reduction_types_reference import (
    typed_reduction_columns,
    reduction_validity,
)


def _layouts(source: Series, picks: List[Int], nans: Bool = False) raises:
    var expression = col("x").drop_nans() if nans else col("x").drop_nulls()
    var expected_source = source.cast(
        DataType.STRING
    ) if source.dtype().is_categorical() else source.copy()
    var expected = DataFrame([expected_source.take(picks)])
    var raw = DataFrame([source.copy()])
    for frame in [raw.copy(), offset_layout(raw), chunk_layout(raw)]:
        assert_frame(frame.select(expression, batch_size=2), expected)
        for optimize in [False, True]:
            for streaming in [False, True]:
                assert_frame(
                    frame.lazy()
                    .select(expression)
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=2,
                    ),
                    expected,
                )
    assert_frame(raw.clear().select(expression), expected.clear())


def test_drop_nulls_and_nans_cover_scalar_types_nested_values_and_layouts() raises:
    for mode in range(3):
        var valid = reduction_validity(mode)
        var picks = List[Int]()
        for row in range(len(valid)):
            if valid[row]:
                picks.append(row)
        for source in typed_reduction_columns(valid):
            _layouts(source, picks)
            _layouts(source, [Int(0), 1, 2, 3, 4, 5], nans=True)
    for n in [0, 1, 8, 17]:
        _layouts(Series.full_null("x", DataType.NULL, n), List[Int]())
    var child = Series("item", Column[Int64]([1, 2, 3]))
    var lists = Series(
        "x", ListColumn([Int64(0), 1, 2, 3], child^, [UInt8(0b101)])
    )
    _layouts(lists, [Int(0), 2])
    _layouts(lists, [Int(0), 1, 2], nans=True)
    var fields: List[Series] = [Series("value", Column[Int64]([1, 2, 3]))]
    _layouts(
        Series("x", StructColumn(fields^, [UInt8(0b101)], length=3)),
        [Int(0), 2],
    )


def test_drop_nans_keeps_nulls_signed_zero_infinities_and_all_bit_offsets() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    for offset in range(8):
        var values = List[Float64](length=offset, fill=99)
        values.extend([Float64(1), nan, -0.0, 0.0, inf, -inf, nan, 99, 2])
        var valid = List[Bool](length=offset + 9, fill=True)
        valid[offset + 7] = False
        var source = Series("x", Column[Float64](values^, valid)).slice(
            offset, 9
        )
        _layouts(source, [Int(0), 2, 3, 4, 5, 7, 8], nans=True)
        _layouts(
            source.cast(DataType.FLOAT32), [Int(0), 2, 3, 4, 5, 7, 8], nans=True
        )
        _layouts(source, [Int(0), 1, 2, 3, 4, 5, 6, 8])


def test_drop_lengths_align_scalars_and_preserve_reduction_axes() raises:
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([99, 1, 2], [False, True, True])),
            Series("y", Column[Int64]([10, 20, 30])),
        ]
    )
    var expression = col("x").drop_nulls()
    var expected = DataFrame(
        [
            Series("constant", Column[Int64]([7, 7])),
            Series("x", Column[Int64]([1, 2])),
        ]
    )
    var selected: List[Expr] = [
        lit(Int64(7)).alias("constant"),
        expression.copy(),
    ]
    assert_frame(frame.select_exprs(selected), expected)
    assert_frame(
        frame.lazy().select_exprs(selected).collect(engine="cpu", batch_size=1),
        expected,
    )
    var sum = frame.select(
        (expression + col("y").sum()).alias("x"), batch_size=1
    )
    assert_frame(sum, DataFrame([Series("x", Column[Int64]([61, 62]))]))
    var reduced = frame.select_exprs(
        [
            expression.len().alias("n"),
            expression.first().alias("first"),
            expression.last().alias("last"),
        ]
    )
    assert_equal(reduced.item(0, "n").int64(), 2)
    assert_equal(reduced.item(0, "first").int64(), 1)
    assert_equal(reduced.item(0, "last").int64(), 2)
    with assert_raises(contains="different lengths"):
        _ = frame.select_exprs([expression.copy(), col("y")])
    with assert_raises(contains="different lengths"):
        _ = (
            frame.lazy()
            .select(expression + col("y"))
            .collect(engine="cpu", batch_size=1)
        )
    with assert_raises():
        _ = frame.with_columns(expression)
    var empty = frame.clear().select_exprs(selected)
    assert_equal(empty.height(), 0)
    assert_equal(empty.column("constant").dtype(), DataType.INT64)
    var singleton = DataFrame(
        [
            Series("x", Column[Int64]([99, 2, 99], [False, True, False])),
            Series("y", Column[Int64]([10, 20, 30])),
        ]
    )
    assert_frame(
        singleton.select((col("x").drop_nulls() + col("y")).alias("x")),
        DataFrame([Series("x", Column[Int64]([12, 22, 32]))]),
    )
    with assert_raises(contains="different lengths"):
        _ = singleton.select_exprs([col("x").drop_nulls(), col("y")])
    var windowed = singleton.with_columns(lit(Int64(0)).alias("g"))
    var independent = col("x").drop_nulls() + col("y").drop_nulls().sum().over(
        "g"
    )
    assert_frame(
        windowed.select(independent.alias("x")),
        DataFrame([Series("x", Column[Int64]([62, 62, 62]))]),
    )


def test_drop_values_apply_per_group_as_lists_and_inside_reductions_and_over() raises:
    var nan = Float64(0) / Float64(0)
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 0, 1, 2])),
            Series(
                "x",
                Column[Float64](
                    [1, nan, 99, 3, 99], [True, True, False, True, False]
                ),
            ),
        ]
    )
    var expected_nulls = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 2])),
            Series(
                "values",
                ListColumn(
                    [Int64(0), 1, 3, 3],
                    Series("item", Column[Float64]([1, nan, 3])),
                ),
            ),
        ]
    )
    var expected_nans = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 2])),
            Series(
                "values",
                ListColumn(
                    [Int64(0), 2, 3, 4],
                    Series(
                        "item",
                        Column[Float64](
                            [1, 99, 3, 99], [True, False, True, False]
                        ),
                    ),
                ),
            ),
        ]
    )
    for source in [frame.copy(), offset_layout(frame), chunk_layout(frame)]:
        var drop = col("x").drop_nulls().alias("values")
        assert_frame(source.group_by("g").agg(drop).sort("g"), expected_nulls)
        assert_frame(
            source.lazy()
            .group_by("g")
            .agg(drop)
            .sort("g")
            .collect(engine="cpu", batch_size=2),
            expected_nulls,
        )
        var drop_nan = col("x").drop_nans().alias("values")
        assert_frame(
            source.group_by("g").agg(drop_nan).sort("g"), expected_nans
        )
        assert_frame(
            source.lazy()
            .group_by("g")
            .agg(drop_nan)
            .sort("g")
            .collect(engine="cpu", batch_size=2),
            expected_nans,
        )
        var counts = (
            source.lazy()
            .select(col("x").drop_nulls().len().over("g").alias("n"))
            .collect(engine="cpu", batch_size=2)
        )
        assert_frame(
            counts, DataFrame([Series("n", Column[Int64]([1, 2, 1, 2, 0]))])
        )
        var no_nans = source.select(col("g").drop_nans().over("g"))
        assert_frame(no_nans, source.select([String("g")]))
        with assert_raises(contains="different length"):
            _ = source.select(col("x").drop_nulls().over("g"))
    var combined = (
        frame.lazy()
        .group_by("g")
        .agg(
            [
                col("x").drop_nulls().drop_nans().len().alias("n"),
                col("x").drop_nulls().drop_nans().first().alias("first"),
            ]
        )
        .sort("g")
        .collect(engine="cpu", batch_size=1)
    )
    assert_frame(
        combined,
        DataFrame(
            [
                Series("g", Column[Int64]([0, 1, 2])),
                Series("n", Column[Int64]([1, 1, 0])),
                Series(
                    "first", Column[Float64]([1, 3, 0], [True, True, False])
                ),
            ]
        ),
    )


def test_drop_values_preserve_guarded_errors_and_lazy_order_boundaries() raises:
    var frame = DataFrame(
        [
            Series("x", Column[String](["1", "bad", "2"])),
            Series("keep", Column[Bool]([True, False, True])),
        ]
    )
    var guarded = (
        when(col("keep")).then(col("x").cast(DataType.INT64)).end().drop_nulls()
    )
    assert_frame(
        frame.lazy()
        .select(guarded.alias("x"))
        .collect(engine="cpu", batch_size=1),
        DataFrame([Series("x", Column[Int64]([1, 2]))]),
    )
    var integers = DataFrame(
        [Series("x", Column[Int64]([1, 99, 2], [True, False, True]))]
    )
    var values = col("x").drop_nulls()
    var observed = (
        when(values == 1)
        .then(values * lit(Int64.MAX))
        .otherwise(lit(Int64(0)))
        .alias("x")
    )
    assert_frame(
        integers.select(observed, batch_size=1),
        DataFrame([Series("x", Column[Int64]([Int64.MAX, 0]))]),
    )
    var ordered = DataFrame(
        [Series("x", Column[Int64]([3, 99, 1, 2], [True, False, True, True]))]
    )
    for optimize in [False, True]:
        var compact = (
            ordered.lazy()
            .select(col("x").drop_nulls())
            .head(2)
            .collect(
                engine="cpu", optimize=optimize, streaming=True, batch_size=1
            )
        )
        assert_frame(compact, DataFrame([Series("x", Column[Int64]([3, 1]))]))


def test_drop_nested_counts_and_null_predicates_use_parent_validity() raises:
    var list_values = Series(
        "x",
        ListColumn(
            [Int64(0), 1, 2, 3],
            Series("item", Column[Int64]([1, 2, 3])),
            [UInt8(0b101)],
        ),
    )
    var fields: List[Series] = [Series("field", Column[Int64]([1, 2, 3]))]
    var struct_values = Series(
        "x", StructColumn(fields^, [UInt8(0b101)], length=3)
    )
    for values in [list_values.copy(), struct_values.copy()]:
        var frame = DataFrame(
            [values.copy(), Series("g", Column[Int64]([0, 1, 0]))]
        )
        var expressions: List[Expr] = [
            col("x").count().alias("count"),
            col("x").null_count().alias("nulls"),
            col("x").len().alias("len"),
            col("x").has_nulls().alias("has"),
            col("x").drop_nulls().len().alias("kept"),
        ]
        var expected = DataFrame(
            [
                Series("count", Column[Int64]([2])),
                Series("nulls", Column[Int64]([1])),
                Series("len", Column[Int64]([3])),
                Series("has", Column[Bool]([True])),
                Series("kept", Column[Int64]([2])),
            ]
        )
        for source in [frame.copy(), offset_layout(frame), chunk_layout(frame)]:
            assert_frame(
                source.select_exprs(expressions, batch_size=1), expected
            )
            assert_frame(
                source.lazy()
                .select_exprs(expressions)
                .collect(engine="cpu", batch_size=1),
                expected,
            )
            for k in range(4):
                var name = expressions[k]._name
                var expected_counts: List[Int64] = [2, 0]
                if k == 1:
                    expected_counts = [0, 1]
                elif k == 2:
                    expected_counts = [2, 1]
                var expected_column = Series(
                    name, Column[Bool]([False, True])
                ) if k == 3 else Series(name, Column[Int64](expected_counts^))
                var expected_group = DataFrame(
                    [Series("g", Column[Int64]([0, 1])), expected_column^]
                )
                assert_frame(
                    source.group_by("g").agg(expressions[k]).sort("g"),
                    expected_group,
                )
                assert_frame(
                    source.lazy()
                    .group_by("g")
                    .agg(expressions[k])
                    .sort("g")
                    .collect(engine="cpu", batch_size=1),
                    expected_group,
                )
            var grouped = (
                source.lazy()
                .group_by("g")
                .agg(expressions)
                .sort("g")
                .collect(engine="cpu", batch_size=1)
            )
            assert_frame(
                grouped,
                DataFrame(
                    [
                        Series("g", Column[Int64]([0, 1])),
                        Series("count", Column[Int64]([2, 0])),
                        Series("nulls", Column[Int64]([0, 1])),
                        Series("len", Column[Int64]([2, 1])),
                        Series("has", Column[Bool]([False, True])),
                        Series("kept", Column[Int64]([2, 0])),
                    ]
                ),
            )


def test_drop_sequences_compose_with_distinct_flags_after_compaction() raises:
    var frame = DataFrame(
        [
            Series(
                "x", Column[Int64]([1, 99, 1, 2], [True, False, True, True])
            ),
            Series("g", Column[Int64]([0, 0, 0, 0])),
        ]
    )
    var first = col("x").drop_nulls().is_first_distinct().alias("first")
    var expected = DataFrame(
        [Series("first", Column[Bool]([True, False, True]))]
    )
    assert_frame(frame.select(first), expected)
    assert_frame(
        frame.lazy().select(first).collect(engine="cpu", batch_size=1), expected
    )
    var counts = first.sum().over("g").alias("n")
    assert_frame(
        frame.select(counts),
        DataFrame([Series("n", Column[UInt32]([2, 2, 2, 2]))]),
    )
