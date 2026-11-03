"""Segment statistics and table filters against evaluated predicates."""
# contract-group: kernel.table_filters
from std.memory import ArcPointer
from std.testing import assert_equal, assert_false, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe.kernels.segment_statistics import (
    compute_table_statistics,
    ensure_statistics,
    series_statistics,
    window_statistics,
)
from dataframe.storage.statistics import (
    COMPARE_GREATERTHAN,
    FILTER_ALWAYS_FALSE,
    FILTER_ALWAYS_TRUE,
    FILTER_FALSE_OR_NULL,
    FILTER_TRUE_OR_NULL,
    NO_PRUNING_POSSIBLE,
    STATISTICS_VECTOR_SIZE,
    STRING_STATS,
)
from dataframe.table_filter import (
    DynamicFilterData,
    TableFilterSet,
    check_statistics,
    set_dynamic_value,
    table_filters,
)
from dataframe.table_scan import zonemap_rows
from dataframe.storage.string import StringColumn
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def _without_statistics(frame: DataFrame) raises -> DataFrame:
    """A merged copy of the frame's rows, in new buffers with no
    statistics."""
    var columns = List[Series]()
    var half = frame.height() // 2
    for column in frame._columns:
        columns.append(
            Series._from_chunks(
                [column.slice(0, half), column.slice(half, len(column) - half)]
            ).rechunk()
        )
    return DataFrame(columns^)


def _counts(frame: DataFrame, predicate: Expr) raises -> Tuple[Int, Int, Int]:
    """Rows the predicate makes true, false and null, evaluated without
    statistics."""
    var plain = _without_statistics(frame)
    var true = plain.filter(predicate.copy()).height()
    var null = plain.filter(predicate.is_null()).height()
    return (true, plain.height() - true - null, null)


def _check_sound(frame: DataFrame, predicate: Expr) raises -> Int:
    """Every vector's statistics answer agrees with the predicate's values
    on its rows; returns how many vectors were decided."""
    var set = table_filters([predicate.copy()], frame._columns)
    assert_equal(len(set.filters), 1, "a table filter for the predicate")
    ref filter = set.filters[0]
    var column = frame.column(filter.column)
    var found = window_statistics(column)
    assert_true(found)
    ref w = found.value()
    var decided = 0
    for v in range(w.first, w.last):
        var start = max(v * STATISTICS_VECTOR_SIZE - w.shift, 0)
        var end = min(
            (v + 1) * STATISTICS_VECTOR_SIZE - w.shift, frame.height()
        )
        var counts = _counts(frame.slice(start, end - start), predicate)
        var answer = check_statistics(filter, w.record[].vectors[v])
        # ALWAYS_FALSE: no row passes (a null does not either).
        if answer == FILTER_ALWAYS_FALSE:
            assert_equal(counts[0], 0, "always false")
        elif answer == FILTER_ALWAYS_TRUE:
            assert_equal(counts[1] + counts[2], 0, "always true")
        elif answer == FILTER_FALSE_OR_NULL:
            assert_equal(counts[0], 0, "false or null")
        elif answer == FILTER_TRUE_OR_NULL:
            assert_equal(counts[1], 0, "true or null")
        if answer != NO_PRUNING_POSSIBLE:
            decided += 1
    return decided


def _ints(n: Int, nullable: Bool) raises -> Series:
    """Clustered values: vector v holds 100 * v .. 100 * v + 99, a vector
    of nulls, and a vector of one value."""
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(n):
        var v = row // STATISTICS_VECTOR_SIZE
        values.append(Int64(100 * v + row % 100))
        valid.append(not nullable or (v != 2 and row % 7 != 3))
    for row in range(2 * STATISTICS_VECTOR_SIZE, 3 * STATISTICS_VECTOR_SIZE):
        if row < n:
            values[row] = 250
    for row in range(3 * STATISTICS_VECTOR_SIZE, 4 * STATISTICS_VECTOR_SIZE):
        if row < n:
            values[row] = 333
            valid[row] = True
    return Series("x", Column[Int64](values^, valid))


def test_integer_statistics_answer_every_filter_kind_soundly() raises:
    var n = 10 * STATISTICS_VECTOR_SIZE + 17
    for nullable in [False, True]:
        var x = _ints(n, nullable)
        ensure_statistics(x)
        var frame = DataFrame([x^])
        var predicates: List[Expr] = [
            col("x") == lit(Int64(333)),
            col("x") != lit(Int64(333)),
            col("x") < lit(Int64(300)),
            lit(Int64(300)) <= col("x"),
            col("x") > lit(Int64(450)),
            col("x") >= lit(Int64(450)),
            col("x").eq_missing(lit(Int64(333))),
            col("x").ne_missing(lit(Int64(333))),
            col("x").is_in([lit(Int64(5)), lit(Int64(333))]),
            col("x").is_null(),
            col("x").is_not_null(),
            (col("x") < lit(Int64(50))) | (col("x") > lit(Int64(900))),
            (col("x") >= lit(Int64(100))) & (col("x") <= lit(Int64(199))),
            col("x").cast(DataType.INT32) > lit(Int32(10)),
        ]
        for i in range(len(predicates) - 1):
            var decided = _check_sound(frame, predicates[i])
            assert_true(decided > 0, "some vector is decided")
        # A cast column is not a table filter.
        assert_false(
            table_filters(
                [predicates[len(predicates) - 1].copy()], frame._columns
            ).has_filters()
        )
        for predicate in predicates:
            # The same rows with and without statistics (a merged copy has
            # none).
            var copied = _without_statistics(frame)
            assert_false(series_statistics(copied.column("x")))
            var expected = copied.filter(predicate.copy())
            var actual = frame.filter(predicate.copy())
            assert_equal(actual.height(), expected.height())
            for row in range(actual.height()):
                assert_equal(
                    actual.item(row, "x").is_null(),
                    expected.item(row, "x").is_null(),
                )
                if not actual.item(row, "x").is_null():
                    assert_equal(
                        actual.item(row, "x").int64(),
                        expected.item(row, "x").int64(),
                    )


def test_float_decimal_date_and_string_statistics() raises:
    var n = 3 * STATISTICS_VECTOR_SIZE
    var floats = List[Float64]()
    var dates = List[Int64]()
    var cents = List[Int64]()
    var texts = List[String]()
    for row in range(n):
        var v = row // STATISTICS_VECTOR_SIZE
        floats.append(Float64(v) + Float64(row % 10) / 10.0)
        dates.append(Int64(10000 + 10 * v + row % 5))
        cents.append(Int64(1000 * v + row % 100))
        texts.append(
            String("alpha") if v
            == 0 else String("beta-long-prefix-" + String(row % 3)) if v
            == 1 else String("gamma")
        )
    floats[2 * STATISTICS_VECTOR_SIZE + 5] = Float64(0.0) / Float64(0.0)
    var frame = DataFrame(
        [
            Series("f", Column[Float64](floats^)),
            Series("d", Column[Int64](dates^)).with_dtype(DataType.DATE),
            Series("m", Column[Int64](cents^)).with_dtype(
                DataType.decimal(12, 2, 64)
            ),
            Series("s", StringColumn(texts)),
        ]
    )
    compute_table_statistics(frame._columns)
    var s = series_statistics(frame.column("s"))
    assert_true(s)
    assert_equal(s.value().stats_type, STRING_STATS)
    assert_equal(s.value().min_string_length, 5)
    assert_equal(s.value().max_string_length, 18)
    var predicates: List[Expr] = [
        col("f") > lit(1.5),
        col("f") < lit(0.95),
        col("m") >= lit("10.00").cast(DataType.decimal(12, 2, 64)),
        col("m") < lit("0.990").cast(DataType.decimal(12, 3, 64)),
        col("s") == lit("gamma"),
        col("s") < lit("b"),
        col("s") > lit("beta-long-prefix-9"),
        col("s").is_in(["alpha", "zeta"]),
    ]
    for p in predicates:
        _ = _check_sound(frame, p)
    # The NaN vector rules nothing out on the float column.
    var f = table_filters([col("f") > lit(100.0)], frame._columns)
    var stats = window_statistics(frame.column("f"))
    assert_equal(
        check_statistics(f.filters[0], stats.value().record[].vectors[2]),
        NO_PRUNING_POSSIBLE,
    )
    assert_equal(
        check_statistics(f.filters[0], stats.value().record[].vectors[0]),
        FILTER_ALWAYS_FALSE,
    )
    # A constant the column cannot hold exactly is evaluated as written.
    assert_false(
        table_filters(
            [col("m") > lit("1.005").cast(DataType.decimal(12, 3, 64))],
            frame._columns,
        ).has_filters()
    )
    # The chunk loop skips single vectors the row groups' check keeps.
    var gamma = table_filters([col("s") == lit("gamma")], frame._columns)
    assert_equal(
        zonemap_rows(gamma, frame._columns, 0, STATISTICS_VECTOR_SIZE),
        FILTER_ALWAYS_FALSE,
    )
    assert_equal(
        zonemap_rows(
            gamma,
            frame._columns,
            2 * STATISTICS_VECTOR_SIZE,
            3 * STATISTICS_VECTOR_SIZE,
        ),
        NO_PRUNING_POSSIBLE,
    )
    assert_equal(
        frame.filter(col("s") == lit("gamma")).height(),
        STATISTICS_VECTOR_SIZE,
    )


def test_dynamic_filter_prunes_once_set() raises:
    var x = _ints(6 * STATISTICS_VECTOR_SIZE, False)
    ensure_statistics(x)
    var frame = DataFrame([x^])
    var data = ArcPointer(
        DynamicFilterData(DataType.INT64, COMPARE_GREATERTHAN)
    )
    var set = TableFilterSet()
    set.push_dynamic("x", data)
    var first = STATISTICS_VECTOR_SIZE
    assert_equal(
        zonemap_rows(set, frame._columns, 0, first), NO_PRUNING_POSSIBLE
    )
    assert_true(
        set_dynamic_value(
            data, COMPARE_GREATERTHAN, Series("b", Column[Int64]([Int64(420)]))
        )
    )
    assert_equal(
        zonemap_rows(set, frame._columns, 0, first), FILTER_ALWAYS_FALSE
    )
    assert_equal(
        zonemap_rows(set, frame._columns, 4 * first, 5 * first),
        NO_PRUNING_POSSIBLE,
    )


def test_eager_join_skips_probe_vectors_outside_the_build_key_range() raises:
    # Three row groups: build keys 512 and 533 are in the first.
    var probe = _ints(3 * 60 * STATISTICS_VECTOR_SIZE, False)
    ensure_statistics(probe)
    var ids = List[Int64]()
    for i in range(len(probe)):
        ids.append(Int64(i))
    var right = DataFrame([probe^, Series("id", Column[Int64](ids^))])
    var left = DataFrame(
        [
            Series("x", Column[Int64]([Int64(512), Int64(533), Int64(9999)])),
            Series("l", Column[Int64]([1, 2, 3])),
        ]
    )
    var joined = left.join(right, ["x"], "inner")
    var plain = left.join(_without_statistics(right), ["x"], "inner")
    assert_true(joined.height() > 0)
    assert_equal(joined.height(), plain.height())
    for row in range(joined.height()):
        assert_equal(
            joined.item(row, "id").int64(), plain.item(row, "id").int64()
        )


def test_pipeline_skips_ruled_out_vectors_and_matches_a_frame_without_statistics() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    # Two row groups: the filter keeps vector 5 of the first (500..599)
    # only, so the row-group check keeps the first and the chunk loop
    # skips its other vectors.
    var x = _ints(2 * 60 * STATISTICS_VECTOR_SIZE, True)
    ensure_statistics(x)
    var ids = List[Int64]()
    for i in range(len(x)):
        ids.append(Int64(i % 13))
    var frame = DataFrame([x^, Series("g", Column[Int64](ids^))])
    var plain = _without_statistics(frame)
    for threads in [1, 4]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        var observed = StrategyTrace()
        var actual = (
            frame.lazy()
            .filter(
                (col("x") >= lit(Int64(510))) & (col("x") < lit(Int64(540)))
            )
            .group_by(["g"])
            .agg([col("x").sum().alias("s"), col("x").count().alias("n")])
            .sort(["g"])
            .collect()
        )
        assert_true(
            "dataframe-path: scan.vector_skip\n" in observed.text(),
            "no vector skipped",
        )
        _ = observed^
        var expected = (
            plain.lazy()
            .filter(
                (col("x") >= lit(Int64(510))) & (col("x") < lit(Int64(540)))
            )
            .group_by(["g"])
            .agg([col("x").sum().alias("s"), col("x").count().alias("n")])
            .sort(["g"])
            .collect()
        )
        assert_equal(actual.height(), expected.height())
        for row in range(actual.height()):
            assert_equal(
                actual.item(row, "s").int64(), expected.item(row, "s").int64()
            )
            assert_equal(
                actual.item(row, "n").int64(), expected.item(row, "n").int64()
            )
        _ = workers^
    _ = trace^
