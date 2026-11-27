"""Categorical columns read from files run in the morsel pipeline on several
workers: each batch of a file carries its own dictionary, and the pipeline's
answers equal the in-memory answers with the categorical type kept."""
# contract-group: planner.routing
from std.memory import ArcPointer
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
from dataframe.categorical import ScanDictionary
from dataframe.dtype import CategoricalDictionary
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
    """The query over the file on four workers, in batches, against the
    in-memory answer."""
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    for optimize in [False, True]:
        var actual = query.collect(
            engine="cpu", streaming=True, optimize=optimize, batch_size=_BATCH
        )
        _same(actual, expected, sort_by)
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
    # Uniques keep each kept row's categorical values in the column's own
    # type: the values are known only once the batches are read.
    for keep in ["first", "last"]:
        _check(
            scan.unique(["t"], keep=keep, maintain_order=True),
            frame.unique(["t"], keep=keep, maintain_order=True).collect(
                streaming=False
            ),
            no_sort,
        )
    _check(
        scan.unique(["k"], keep="first", maintain_order=True),
        frame.unique(["k"], keep="first", maintain_order=True).collect(
            streaming=False
        ),
        no_sort,
    )
    _check(
        scan.unique(["s", "t"], keep="any"),
        frame.unique(["s", "t"], keep="any").collect(streaming=False),
        ["s", "t"],
    )
    var whole = scan.unique(["k"], keep="first", maintain_order=True).collect()
    assert_true(whole.column("s").dtype().is_categorical())
    _same(
        whole,
        frame.unique(["k"], keep="first", maintain_order=True).collect(
            streaming=False
        ),
        no_sort,
    )
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


def test_unordered_unique_keeps_categoricals_from_every_part() raises:
    """An unordered unique finishes its groups in parts on several
    workers, and each part encodes its kept categorical labels on its
    own. The parts' rows join as one column holding every label, on one
    worker and on eight, eager and lazy alike."""
    var path = _one_file()
    var frame = read_csv(path, _schema(), null_values=[""])
    var scan = scan_csv(path, _schema(), null_values=[""])
    var subsets: List[List[String]] = [["k"], ["v"], ["t"]]
    for threads in ["1", "8"]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
        for keep in ["first", "last", "any"]:
            for subset in subsets:
                var expected = frame.unique(subset, keep=keep)
                if keep == "any":
                    # Any row of a group may be kept: compare the keys.
                    expected = expected.select(subset)
                for optimize in [False, True]:
                    var actual = scan.unique(subset, keep=keep).collect(
                        engine="cpu",
                        streaming=True,
                        optimize=optimize,
                        batch_size=_BATCH,
                    )
                    if keep == "any":
                        actual = actual.select(subset)
                    _same(actual, expected, subset)
        _ = workers^


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
    assert_true("dataframe-strategy: table_pipeline" in paths, paths)


def test_scan_dictionary_views_share_the_values_they_grow_from() raises:
    """512 batches, each bringing one new value: every batch's dictionary
    is the values so far, its codes keep their values as later batches
    add more, all of them compare equal, and together they hold under
    three times the entries the values need, not a copy per batch."""
    comptime batches = 512
    var scan = ScanDictionary()
    var types = List[DataType]()
    for i in range(batches):
        var dictionary = CategoricalDictionary()
        dictionary.append(String(i))
        var added = scan.add(ArcPointer(dictionary^))
        assert_equal(len(added[0]), 1)
        assert_equal(Int(added[0][0]), i)
        types.append(added[1])
    assert_equal(len(scan.snapshots), batches)
    var final = len(scan.values.offsets[])
    assert_equal(final, batches + 1)
    # Entries held: each distinct offsets list once, at the length of the
    # longest view of it.
    var addresses = List[Int]()
    var lengths = List[Int]()
    for k in range(batches):
        ref view = scan.snapshots[k][]
        assert_equal(len(view), k + 1)
        # A view taken early still reads its values after later growth.
        for code in range(len(view)):
            assert_equal(String(view.get(code)), String(code))
        assert_true(types[k] == types[0], "batch types compare equal")
        assert_true(types[k] == scan.schema_type(), "and equal the schema's")
        var address = Int(view.offsets.unsafe_ptr())
        var found = False
        for a in range(len(addresses)):
            if addresses[a] == address:
                lengths[a] = max(lengths[a], len(view.offsets))
                found = True
        if not found:
            addresses.append(address)
            lengths.append(len(view.offsets))
    var held = 0
    for length in lengths:
        held += length
    assert_true(held < 3 * final, "views hold " + String(held) + " entries")
    scan.settle()
    for k in range(batches):
        ref view = scan.snapshots[k][]
        assert_equal(len(view), batches)
        assert_equal(String(view.get(batches - 1)), String(batches - 1))
    assert_equal(len(scan.schema[]), batches)
    assert_true(types[0] == scan.schema_type())


def test_every_batch_is_handed_out_once_across_many_workers() raises:
    # Eight workers take batches of ten rows from one file source, and
    # each batch's categorical dictionary joins the scan's shared one: the
    # cut and the join run under the source's lock on whichever worker
    # takes the batch, so every row is counted once and every value keeps
    # its spelling, on every repetition.
    var path = _one_file()
    var frame = read_csv(path, _schema(), null_values=[""])
    var expected = (
        frame.lazy()
        .group_by(["s"])
        .agg([col("v").sum().alias("v"), col("k").count().alias("n")])
        .collect()
    )
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "8")
    for _ in range(8):
        var actual = (
            scan_csv(path, _schema(), null_values=[""])
            .group_by(["s"])
            .agg([col("v").sum().alias("v"), col("k").count().alias("n")])
            .collect(engine="cpu", streaming=True, batch_size=10)
        )
        _same(actual, expected, ["s"])
    _ = threads^
