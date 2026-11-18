"""Categorical columns read from files run in the morsel pipeline on several
workers: each batch of a file carries its own dictionary, and the pipeline's
answers equal the in-memory answers with the categorical type kept."""
# contract-group: planner.routing
from std.os import getenv, makedirs
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    lit,
    read_csv,
    scan_csv,
)
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime _ROWS = 1200
comptime _BATCH = 50


def _schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("k"),
            CsvField.categorical("s"),
            CsvField.categorical("t"),
            CsvField.int64("v"),
        ]
    )


def _label(row: Int) -> String:
    # Values that change from batch to batch, so batches meet them in
    # different orders and each batch's dictionary is its own; every
    # thirteenth row is null.
    if row % 13 == 5:
        return ""
    var block = row // _BATCH
    return "s" + String((row * 7 + block * 11) % (5 + block % 9))


def _text(first: Int, last: Int) -> String:
    var text = String("k,s,t,v\n")
    for row in range(first, last):
        text += String(row) + "," + _label(row) + ","
        text += ("t" + String((row // _BATCH) % 4)) if row % 17 else ""
        text += "," + String((row * 31) % 101) + "\n"
    return text^


def _directory() raises -> String:
    var directory = getenv("COVERAGE_TEST_TMP")
    assert_true(Bool(directory))
    return directory + "/categorical-file-sources"


def _one_file() raises -> String:
    var directory = _directory()
    makedirs(directory, exist_ok=True)
    var path = directory + "/one.csv"
    with open(path, "w") as file:
        file.write(_text(0, _ROWS))
    return path^


def _files() raises -> String:
    """Four files, each with its own values: a dataset's batches."""
    var directory = _directory() + "/parts"
    makedirs(directory, exist_ok=True)
    var step = _ROWS // 4
    for part in range(4):
        with open(directory + "/part" + String(part) + ".csv", "w") as file:
            file.write(_text(part * step, (part + 1) * step))
    return directory + "/*.csv"


def _strings(frame: DataFrame) raises -> DataFrame:
    """The frame with categorical columns as their values: two dictionaries
    may order the same values differently."""
    var columns = List[Series]()
    for column in frame._columns:
        if column.dtype().is_categorical():
            columns.append(column.cast(DataType.STRING))
        else:
            columns.append(column.copy())
    return DataFrame(columns^, height=frame.height())


def _same(actual: DataFrame, expected: DataFrame, sort_by: List[String]) raises:
    assert_equal(actual.columns(), expected.columns())
    assert_equal(actual.height(), expected.height())
    for name in expected.columns():
        assert_equal(
            actual.column(name).dtype().is_categorical(),
            expected.column(name).dtype().is_categorical(),
            name,
        )
    var left = _strings(actual)
    var right = _strings(expected)
    if len(sort_by):
        left = left.sort(sort_by, nulls_last=True)
        right = right.sort(sort_by, nulls_last=True)
    assert_true(left.equals(right), String(left) + "\n" + String(right))


def _check(query: LazyFrame, expected: DataFrame, sort_by: List[String]) raises:
    """The query over the file on four workers, in batches, in the morsel
    pipeline, against the in-memory answer."""
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    for optimize in [False, True]:
        var trace = StrategyTrace()
        var actual = query.collect(
            engine="cpu", streaming=True, optimize=optimize, batch_size=_BATCH
        )
        var paths = trace.text()
        _ = trace^
        _same(actual, expected, sort_by)
        assert_true(paths.find("dataframe-strategy: batch_stream") < 0, paths)
    _ = setting^
    _ = threads^


def _dimension() raises -> DataFrame:
    var keys = List[Int64]()
    var names = List[String]()
    for key in range(0, _ROWS, 3):
        keys.append(Int64(key))
        names.append("d" + String(key % 7))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("d", Column[String](names^)),
        ]
    )


def _queries(scan: LazyFrame, frame: LazyFrame) raises:
    var no_sort = List[String]()
    # Filters and projections, rows in source order.
    var filtered = scan.filter(
        (col("s") == "s3") | (col("v") > 90)
    ).with_columns([(col("v") * 2).alias("w")])
    var expected = frame.filter(
        (col("s") == "s3") | (col("v") > 90)
    ).with_columns([(col("v") * 2).alias("w")])
    _check(filtered, expected.collect(streaming=False), no_sort)
    # Groupings on a categorical key and a pair of them; the key stays
    # categorical.
    var aggregates: List[Expr] = [
        col("v").sum().alias("total"),
        col("k").count().alias("rows"),
        col("t").n_unique().alias("t_unique"),
    ]
    _check(
        scan.group_by(["s"]).agg(aggregates),
        frame.group_by(["s"]).agg(aggregates).collect(streaming=False),
        ["s"],
    )
    _check(
        scan.group_by(["s"], maintain_order=True).agg(aggregates),
        frame.group_by(["s"], maintain_order=True)
        .agg(aggregates)
        .collect(streaming=False),
        no_sort,
    )
    var pair: List[Expr] = [
        col("v").max().alias("top"),
        col("k").min().alias("first"),
    ]
    _check(
        scan.filter(col("v") > 10).group_by(["s", "t"]).agg(pair),
        frame.filter(col("v") > 10)
        .group_by(["s", "t"])
        .agg(pair)
        .collect(streaming=False),
        ["s", "t"],
    )
    # Ungrouped reductions of a categorical column.
    var reductions: List[Expr] = [
        col("s").n_unique().alias("distinct"),
        col("s").first().alias("first"),
        col("t").last().alias("last"),
        col("s").count().alias("valid"),
    ]
    _check(
        scan.select_exprs(reductions),
        frame.select_exprs(reductions).collect(streaming=False),
        no_sort,
    )
    # A join probed by the file's rows, carrying its categoricals.
    var dimension = _dimension().lazy()
    _check(
        scan.join(dimension, on=["k"], how="inner"),
        frame.join(dimension, on=["k"], how="inner").collect(streaming=False),
        ["k"],
    )
    _check(
        scan.join(dimension, on=["k"], how="left")
        .group_by(["t", "d"])
        .agg([col("s").n_unique().alias("labels")]),
        frame.join(dimension, on=["k"], how="left")
        .group_by(["t", "d"])
        .agg([col("s").n_unique().alias("labels")])
        .collect(streaming=False),
        ["t", "d"],
    )
    # Sorted by a categorical's values, first rows only; `k` breaks ties.
    _check(
        scan.sort(["s", "k"], nulls_last=True).head(40),
        frame.sort(["s", "k"], nulls_last=True)
        .head(40)
        .collect(streaming=False),
        no_sort,
    )
    _check(
        scan.sort(
            ["t", "s", "k"],
            descending=[True, False, True],
            nulls_last=[True, True, True],
        ).head(25),
        frame.sort(
            ["t", "s", "k"],
            descending=[True, False, True],
            nulls_last=[True, True, True],
        )
        .head(25)
        .collect(streaming=False),
        no_sort,
    )
    # Head and slice over steps: the first rows in source order.
    _check(
        scan.filter(col("v") > 50).head(30),
        frame.filter(col("v") > 50).head(30).collect(streaming=False),
        no_sort,
    )
    _check(
        scan.filter(col("s").is_not_null()).slice(70, 45),
        frame.filter(col("s").is_not_null())
        .slice(70, 45)
        .collect(streaming=False),
        no_sort,
    )
    # A sort's first no rows: the columns, no row.
    _check(
        scan.sort("s").head(0),
        frame.sort("s").head(0).collect(streaming=False),
        no_sort,
    )


def test_categorical_file_batches_with_their_own_dictionaries_run_in_the_pipeline() raises:
    var path = _one_file()
    var frame = read_csv(path, _schema(), null_values=[""])
    assert_true(frame.column("s").dtype().is_categorical())
    _queries(scan_csv(path, _schema(), null_values=[""]), frame.lazy())


def test_categorical_dataset_files_with_their_own_dictionaries_run_in_the_pipeline() raises:
    var pattern = _files()
    var frame = read_csv(pattern, _schema(), null_values=[""])
    assert_equal(frame.height(), _ROWS)
    _queries(scan_csv(pattern, _schema(), null_values=[""]), frame.lazy())


def test_categorical_file_scan_routes_to_the_morsel_pipeline() raises:
    var path = _one_file()
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var result = (
        scan_csv(path, _schema(), null_values=[""])
        .group_by(["s"])
        .agg([col("v").sum()])
        .collect(engine="cpu", streaming=True, batch_size=_BATCH)
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(result.column("s").dtype().is_categorical())
    assert_true(paths.find("dataframe-strategy: table_pipeline") >= 0, paths)
    assert_true(paths.find("dataframe-strategy: batch_stream") < 0, paths)
