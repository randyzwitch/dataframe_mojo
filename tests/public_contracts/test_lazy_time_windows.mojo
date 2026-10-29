"""Lazy group_by_dynamic and rolling against the eager methods and literals."""
# contract-group: lazy.time_windows
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, LazyFrame, Series, col
from tests.support.coverage_compare import assert_frame

comptime HOUR = 3600000


def _times(name: String, values: List[Int64]) raises -> Series:
    return Series(name, Column[Int64](values.copy())).with_dtype(
        DataType.datetime("ms")
    )


def _source() raises -> DataFrame:
    """Sorted by t within each g; x has nulls; `unused` is never read."""
    var x: List[Int64] = [4, 0, 1, 7, 0, -2]
    var valid: List[Bool] = [True, False, True, True, False, True]
    return DataFrame(
        [
            _times("t", [0, 0, HOUR, 2 * HOUR, 3 * HOUR, 5 * HOUR]),
            Series("g", Column[Int64]([0, 1, 0, 1, 0, 1])),
            Series("x", Column[Int64](x^, valid)),
            Series("unused", Column[Int64]([9, 9, 9, 9, 9, 9])),
        ]
    )


def _aggregates() -> List[Expr]:
    return [
        col("x").sum().alias("sum"),
        col("x").count().alias("count"),
        col("x").len().alias("len"),
        col("x").first().alias("first"),
        col("x").last().alias("last"),
    ]


def _assert_collects(plan: LazyFrame, expected: DataFrame) raises:
    """Every executor and optimizer setting gives the eager result."""
    for optimize in [False, True]:
        for streaming in [False, True]:
            for batch_size in [1, 65536]:
                assert_frame(
                    plan.collect(
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=batch_size,
                        engine="cpu",
                    ),
                    expected,
                )
    var schema = plan.collect_schema()
    var fields = expected.schema()
    assert_equal(len(schema), len(fields))
    for k in range(len(fields)):
        assert_equal(schema[k], fields[k].name + ": " + fields[k].dtype.name())


def _assert_ints(frame: DataFrame, name: String, values: List[Int64]) raises:
    assert_equal(frame.height(), len(values))
    for row in range(len(values)):
        assert_equal(frame.item(row, name).to_physical(), values[row])


def test_lazy_group_by_dynamic_matches_eager_closed_and_labels() raises:
    var source = _source()
    for closed in [
        String("left"),
        String("right"),
        String("both"),
        String("none"),
    ]:
        for label in [String("left"), String("right"), String("datapoint")]:
            for offset in [String("0h"), String("30m")]:
                var eager = source.group_by_dynamic(
                    "t",
                    "2h",
                    period="3h",
                    offset=offset,
                    closed=closed,
                    label=label,
                ).agg(_aggregates())
                var plan = (
                    source.lazy()
                    .group_by_dynamic(
                        "t",
                        "2h",
                        period="3h",
                        offset=offset,
                        closed=closed,
                        label=label,
                    )
                    .agg(_aggregates())
                )
                _assert_collects(plan, eager)
    # Independent literals: [0, 2h) holds rows 0-2, [2h, 4h) rows 3-4 and
    # [4h, 6h) row 5; x is 4, null, 1, 7, null, -2.
    var plan = source.lazy().group_by_dynamic("t", "2h").agg(_aggregates())
    var actual = plan.collect(engine="cpu")
    assert_equal(
        actual.columns(), ["t", "sum", "count", "len", "first", "last"]
    )
    _assert_ints(actual, "t", [0, 2 * HOUR, 4 * HOUR])
    _assert_ints(actual, "sum", [5, 7, -2])
    _assert_ints(actual, "count", [2, 1, 1])
    _assert_ints(actual, "len", [3, 2, 1])
    assert_true(
        "GROUP_BY_DYNAMIC t every 2h period 2h offset 0d" in plan.explain()
    )


def test_lazy_group_by_dynamic_by_keys_and_nulls() raises:
    var source = _source()
    var aggregates: List[Expr] = [
        col("x").count().alias("count"),
        col("x").len().alias("len"),
    ]
    var plan = (
        source.lazy().group_by_dynamic("t", "2h", group_by="g").agg(aggregates)
    )
    _assert_collects(
        plan, source.group_by_dynamic("t", "2h", group_by="g").agg(aggregates)
    )
    # g = 0 holds t 0, 1h, 3h (x 4, 1, null); g = 1 holds t 0, 2h, 5h
    # (x null, 7, -2).
    var actual = plan.collect(engine="cpu")
    assert_equal(actual.columns(), ["g", "t", "count", "len"])
    _assert_ints(actual, "g", [0, 0, 1, 1, 1])
    _assert_ints(actual, "t", [0, 2 * HOUR, 0, 2 * HOUR, 4 * HOUR])
    _assert_ints(actual, "count", [2, 0, 0, 1, 1])
    _assert_ints(actual, "len", [2, 1, 1, 1, 1])
    # Null keys form their own group, as in the eager method.
    var keys: List[Int64] = [0, 0, 0, 0, 0, 0]
    var key_valid: List[Bool] = [False, True, False, True, False, True]
    var nullable = source.with_column(
        Series("g", Column[Int64](keys^, key_valid))
    )
    _assert_collects(
        nullable.lazy()
        .group_by_dynamic("t", "2h", group_by=["g"])
        .agg(aggregates),
        nullable.group_by_dynamic("t", "2h", group_by=["g"]).agg(aggregates),
    )


def test_lazy_rolling_matches_eager_and_literals() raises:
    var source = _source()
    for closed in [
        String("left"),
        String("right"),
        String("both"),
        String("none"),
    ]:
        for offset in [String("-2h"), String("-1h")]:
            for grouped in range(2):
                var by: List[String] = ["g"] if grouped else []
                _assert_collects(
                    source.lazy()
                    .rolling(
                        "t", "2h", offset=offset, closed=closed, group_by=by
                    )
                    .agg(_aggregates()),
                    source.rolling(
                        "t", "2h", offset=offset, closed=closed, group_by=by
                    ).agg(_aggregates()),
                )
    # Windows (t - 2h, t]: one per row, empty-of-values windows included.
    var plan = source.lazy().rolling("t", "2h").agg(_aggregates())
    var actual = plan.collect(engine="cpu")
    _assert_ints(actual, "t", [0, 0, HOUR, 2 * HOUR, 3 * HOUR, 5 * HOUR])
    _assert_ints(actual, "len", [2, 2, 3, 2, 2, 1])
    _assert_ints(actual, "count", [1, 1, 2, 2, 1, 1])
    _assert_ints(actual, "sum", [4, 4, 5, 8, 7, -2])
    assert_true("ROLLING t period 2h offset -2h closed right" in plan.explain())


def test_lazy_time_windows_compose_without_crossing_filters_or_slices() raises:
    var source = _source()
    var aggregates: List[Expr] = [
        col("x").sum().alias("sum"),
        col("x").len().alias("len"),
    ]
    # A filter before the windows changes which rows they see.
    var kept = source.filter(col("x").is_not_null())
    _assert_collects(
        source.lazy()
        .filter(col("x").is_not_null())
        .group_by_dynamic("t", "2h")
        .agg(aggregates),
        kept.group_by_dynamic("t", "2h").agg(aggregates),
    )
    # Filters, slices and selections above stay above: a filter on the
    # keys, one on an aggregate and a head over the windows.
    var eager = source.group_by_dynamic("t", "2h", group_by="g").agg(aggregates)
    var key_filter = (
        source.lazy()
        .group_by_dynamic("t", "2h", group_by="g")
        .agg(aggregates)
        .filter(col("g") == 1)
        .select(["t", "len"])
    )
    _assert_collects(
        key_filter, eager.filter(col("g") == 1).select(["t", "len"])
    )
    var text = key_filter.explain()
    var filter_at = text.find("FILTER")
    var window_at = text.find("GROUP_BY_DYNAMIC")
    assert_true(filter_at >= 0 and window_at > filter_at)
    # Projection pruning reads only the index, keys and aggregated columns.
    assert_true("unused" not in text)
    assert_true("[project t, g, x]" in text)
    var rolling_head = (
        source.lazy()
        .rolling("t", "2h", group_by="g")
        .agg(aggregates)
        .filter(col("len") == 2)
        .head(2)
    )
    _assert_collects(
        rolling_head,
        source.rolling("t", "2h", group_by="g")
        .agg(aggregates)
        .filter(col("len") == 2)
        .head(2),
    )
    var rolling_text = rolling_head.explain()
    assert_true(rolling_text.find("SLICE") < rolling_text.find("ROLLING"))
    assert_true(rolling_text.find("FILTER") < rolling_text.find("ROLLING"))
    # A sort below supplies the order; sums and lengths ignore tie order.
    var reversed = source.take([5, 4, 3, 2, 1, 0])
    _assert_collects(
        reversed.lazy()
        .sort("t")
        .rolling("t", "2h")
        .agg(aggregates)
        .select(["sum", "len"]),
        source.rolling("t", "2h").agg(aggregates).select(["sum", "len"]),
    )


def test_lazy_time_windows_empty_input_and_errors() raises:
    var empty = _source().clear()
    _assert_collects(
        empty.lazy().group_by_dynamic("t", "1h").agg(_aggregates()),
        empty.group_by_dynamic("t", "1h").agg(_aggregates()),
    )
    _assert_collects(
        empty.lazy().rolling("t", "1h", group_by="g").agg(_aggregates()),
        empty.rolling("t", "1h", group_by="g").agg(_aggregates()),
    )
    assert_equal(
        empty.lazy()
        .rolling("t", "1h")
        .agg(_aggregates())
        .collect(engine="cpu")
        .height(),
        0,
    )
    # Settings are checked when the plan runs, not when built.
    var bad = _source().lazy().group_by_dynamic("t", "1h", closed="bad")
    with assert_raises():
        _ = bad.agg(_aggregates()).collect(engine="cpu")
    var stamps: List[Int64] = [0, 0, HOUR, 2 * HOUR, 3 * HOUR, 5 * HOUR]
    var present: List[Bool] = [True, True, False, True, True, True]
    var nulls = (
        _source()
        .with_column(
            Series("t", Column[Int64](stamps^, present)).with_dtype(
                DataType.datetime("ms")
            )
        )
        .lazy()
    )
    with assert_raises(contains="nulls"):
        _ = nulls.rolling("t", "1h").agg(_aggregates()).collect(engine="cpu")
    # A null still raises when the index is also out of order.
    var shuffled = nulls.collect(engine="cpu").take([5, 2, 0, 4, 1, 3]).lazy()
    with assert_raises(contains="nulls"):
        _ = shuffled.rolling("t", "1h").agg(_aggregates()).collect(engine="cpu")
    with assert_raises(contains="nulls"):
        _ = (
            shuffled.group_by_dynamic("t", "1h", group_by="g")
            .agg(_aggregates())
            .collect(engine="cpu")
        )


def _indexed(frame: DataFrame) raises -> DataFrame:
    var ids = List[Int64]()
    for row in range(frame.height()):
        ids.append(Int64(row))
    return frame.with_column(Series("i", Column[Int64](ids^)))


def _duplicates() raises -> DataFrame:
    """Unsorted times with ties inside and across groups."""
    var x: List[Int64] = [1, 2, 3, 0, 5, 6, 7]
    var valid: List[Bool] = [True, True, True, False, True, True, True]
    return _indexed(
        DataFrame(
            [
                _times(
                    "t",
                    [2 * HOUR, 0, 2 * HOUR, HOUR, 0, 2 * HOUR, 4 * HOUR],
                ),
                Series("g", Column[Int64]([1, 0, 0, 1, 0, 1, 0])),
                Series("x", Column[Int64](x^, valid)),
            ]
        )
    )


def _unsorted_frames() raises -> List[DataFrame]:
    # Group 1 appears first here, group 0 first once sorted by t.
    return [_indexed(_source()).take([5, 2, 4, 1, 3, 0]), _duplicates()]


def _group_order(frame: DataFrame, by: List[String]) raises -> List[Int64]:
    """Group key values in first-appearance order (one entry if no by)."""
    var seen = List[Int64]()
    for row in range(frame.height()):
        var key = (
            Int64(0) if len(by) == 0 else frame.item(row, "g").to_physical()
        )
        if key not in seen:
            seen.append(key)
    return seen^


def _rows_by_group(
    frame: DataFrame, by: List[String], groups: List[Int64], name: String
) raises -> List[Int]:
    """Rows of `frame` grouped in `groups` order, keeping frame order inside
    each group; `name` is the column that holds the group key."""
    var rows = List[Int]()
    for key in groups:
        for row in range(frame.height()):
            if len(by) == 0 or frame.item(row, name).to_physical() == key:
                rows.append(row)
    return rows^


def _presorted(frame: DataFrame) raises -> DataFrame:
    return frame.sort("t", maintain_order=True)


def _rolling_expected(
    frame: DataFrame, offset: String, closed: String, by: List[String]
) raises -> DataFrame:
    """Rolling on presorted input, rows put in the order rolling gives the
    unsorted input: groups by first appearance, input order inside each."""
    var sorted = _presorted(frame)
    var result = sorted.rolling(
        "t", "2h", offset=offset, closed=closed, group_by=by
    ).agg(_aggregates())
    # Output row r of the presorted run belongs to input row `owner[r]`.
    var owner = List[Int]()
    for row in _rows_by_group(sorted, by, _group_order(sorted, by), "g"):
        owner.append(Int(sorted.item(row, "i").to_physical()))
    var position = List[Int](length=frame.height(), fill=-1)
    for r in range(len(owner)):
        position[owner[r]] = r
    var order = List[Int]()
    for row in _rows_by_group(frame, by, _group_order(frame, by), "g"):
        order.append(position[Int(frame.item(row, "i").to_physical())])
    return result.take(order)


def _dynamic_expected(
    frame: DataFrame, closed: String, label: String, by: List[String]
) raises -> DataFrame:
    """Dynamic windows of presorted input, groups in the unsorted input's
    first-appearance order and windows in time order."""
    var result = (
        _presorted(frame)
        .group_by_dynamic(
            "t", "2h", period="3h", closed=closed, label=label, group_by=by
        )
        .agg(_aggregates())
    )
    return result.take(_rows_by_group(result, by, _group_order(frame, by), "g"))


def test_time_windows_accept_an_unsorted_index() raises:
    for frame in _unsorted_frames():
        for grouped in range(2):
            var by: List[String] = ["g"] if grouped else []
            for closed in [
                String("left"),
                String("right"),
                String("both"),
                String("none"),
            ]:
                for offset in [String("-2h"), String("-1h")]:
                    var expected = _rolling_expected(frame, offset, closed, by)
                    assert_frame(
                        frame.rolling(
                            "t",
                            "2h",
                            offset=offset,
                            closed=closed,
                            group_by=by,
                        ).agg(_aggregates()),
                        expected,
                    )
                    _assert_collects(
                        frame.lazy()
                        .rolling(
                            "t",
                            "2h",
                            offset=offset,
                            closed=closed,
                            group_by=by,
                        )
                        .agg(_aggregates()),
                        expected,
                    )
                for label in [String("left"), String("datapoint")]:
                    var expected = _dynamic_expected(frame, closed, label, by)
                    assert_frame(
                        frame.group_by_dynamic(
                            "t",
                            "2h",
                            period="3h",
                            closed=closed,
                            label=label,
                            group_by=by,
                        ).agg(_aggregates()),
                        expected,
                    )
                    _assert_collects(
                        frame.lazy()
                        .group_by_dynamic(
                            "t",
                            "2h",
                            period="3h",
                            closed=closed,
                            label=label,
                            group_by=by,
                        )
                        .agg(_aggregates()),
                        expected,
                    )


def test_rolling_by_expressions_accept_an_unsorted_index() raises:
    var expressions: List[Expr] = [
        col("x").rolling_sum_by("t", "2h").alias("sum"),
        col("x").rolling_mean_by("t", "2h", closed="both").alias("mean"),
        col("x").rolling_min_by("t", "2h").alias("min"),
        col("x").rolling_max_by("t", "3h", closed="left").alias("max"),
        col("x").rolling_std_by("t", "3h").alias("std"),
        col("x").rolling_var_by("t", "2h", ddof=0).alias("var"),
        col("x").rolling_sum_by("t", "2h").over("g").alias("sum_g"),
        col("x").rolling_max_by("t", "3h").over("g").alias("max_g"),
    ]
    for frame in _unsorted_frames():
        var sorted = _presorted(frame).with_columns(expressions)
        var position = List[Int](length=frame.height(), fill=-1)
        for row in range(sorted.height()):
            position[Int(sorted.item(row, "i").to_physical())] = row
        var order = List[Int]()
        for row in range(frame.height()):
            order.append(position[Int(frame.item(row, "i").to_physical())])
        var expected = sorted.take(order)
        assert_frame(frame.with_columns(expressions), expected)
        for streaming in [False, True]:
            assert_frame(
                frame.lazy()
                .with_columns(expressions)
                .collect(streaming=streaming, engine="cpu"),
                expected,
            )


def test_lazy_time_windows_run_on_the_cpu_under_automatic_placement() raises:
    var source = _source()
    var aggregates: List[Expr] = [col("x").sum().alias("sum")]
    for rolling in [False, True]:
        var plan = (
            source.lazy()
            .rolling("t", "2h", group_by="g")
            .agg(aggregates) if rolling else source.lazy()
            .group_by_dynamic("t", "2h", group_by="g")
            .agg(aggregates)
        ).select(["t", "sum"])
        assert_true(
            plan.collect(engine="auto").equals(plan.collect(engine="cpu"))
        )
