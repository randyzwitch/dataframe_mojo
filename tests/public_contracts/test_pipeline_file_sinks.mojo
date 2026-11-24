"""File sinks write what collect returns, from the morsel pipeline on several
workers: in the same order where the query defines one."""
# contract-group: io.sinks
from std.os import getenv, mkdir
from std.os.path import exists
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    LazyFrame,
    Series,
    col,
    read_parquet,
    scan_csv,
    write_csv,
)
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 40_000
comptime BATCH = 1000


def _path(name: String) raises -> String:
    var root = getenv("COVERAGE_TEST_TMP") + "/pipeline-file-sinks"
    if not exists(root):
        mkdir(root)
    return root + "/" + name


def _source(rows: Int) raises -> DataFrame:
    var keys = List[Int64]()
    var values = List[Int64]()
    var halves = List[Float64]()
    var text = List[String]()
    var colors: List[String] = ["red", "green", "blue", "gray", "teal"]
    var shades = List[String]()
    for row in range(rows):
        keys.append(Int64((row * 7) % 97))
        values.append(Int64(row))
        halves.append(Float64(row) * 0.5)
        text.append("r" + String(row))
        shades.append(colors[(row // 3) % 5])
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^)),
            Series("f", Column[Float64](halves^)),
            Series("s", Column[String](text^)),
            Series("c", Column[String](shades^)).cast(DataType.CATEGORICAL),
        ]
    )


def _dimension() raises -> DataFrame:
    # Every key but a few, once each: an inner join drops some rows.
    var keys = List[Int64]()
    var names = List[String]()
    for key in range(97):
        if key % 10 == 3:
            continue
        keys.append(Int64(key))
        names.append("d" + String(key))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("d", Column[String](names^)),
        ]
    )


def _file_source() raises -> LazyFrame:
    """The source's rows read from a CSV file in batches, `c` as a
    categorical whose dictionary grows over the scan."""
    var path = _path("source.csv")
    if not exists(path):
        write_csv(_source(ROWS), path)
    return scan_csv(
        path,
        CsvSchema(
            [
                CsvField.int64("k"),
                CsvField.int64("v"),
                CsvField.float64("f"),
                CsvField.string("s"),
                CsvField.categorical("c"),
            ]
        ),
    )


def _text(path: String) raises -> String:
    with open(path, "r") as file:
        return file.read()


def _sorted_lines(text: String) raises -> String:
    """The header, then the other lines in sorted order."""
    var lines = List[String]()
    for line in text.split("\n"):
        lines.append(String(line))
    var header = lines.pop(0)
    sort(lines)
    var out = header
    for line in lines:
        out += "\n" + line
    return out^


def _strings(frame: DataFrame) raises -> DataFrame:
    """The frame with categorical columns as their values: the file read
    back holds the same values under another dictionary."""
    var columns = List[Series]()
    for column in frame._columns:
        if column.dtype().is_categorical():
            columns.append(column.cast(DataType.STRING))
        else:
            columns.append(column.copy())
    return DataFrame(columns^, height=frame.height())


def _check(
    query: LazyFrame,
    name: String,
    ordered: Bool,
    sort_by: List[String] = List[String](),
) raises:
    """`query` into a CSV and a Parquet file on four workers equals its
    collected rows: the CSV text is what write_csv makes of them, and the
    Parquet file reads back as them; row for row when `ordered`, else as
    the same rows (sorted lines, or sorted by `sort_by`)."""
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    for optimize in [False, True]:
        var expected = query.collect(
            engine="cpu", streaming=True, optimize=optimize, batch_size=BATCH
        )
        var suffix = name + "-" + String(optimize)
        var csv = _path(suffix + ".csv")
        query.sink_csv(csv, batch_size=BATCH, optimize=optimize)
        var reference = _path(suffix + "-reference.csv")
        write_csv(expected, reference)
        if ordered:
            assert_equal(_text(csv), _text(reference), name)
        else:
            assert_equal(
                _sorted_lines(_text(csv)),
                _sorted_lines(_text(reference)),
                name,
            )
        var parquet = _path(suffix + ".parquet")
        query.sink_parquet(
            parquet, batch_size=BATCH, row_group_size=3000, optimize=optimize
        )
        var back = _strings(read_parquet(parquet))
        var want = _strings(expected)
        assert_equal(back.columns(), want.columns(), name)
        if not ordered:
            back = back.sort(sort_by)
            want = want.sort(sort_by)
        assert_true(back.equals(want), name)
    _ = threads^


def _queries(source: LazyFrame, name: String) raises:
    var dimension = _dimension().lazy()
    # Filtered and projected rows: written by the workers as they go.
    _check(
        source.filter(col("k") < 60).with_columns((col("v") * 2).alias("w")),
        name + "-filtered",
        True,
    )
    # Every row as it is.
    _check(source, name + "-whole", True)
    # Joined rows (the join's order is unspecified).
    _check(
        source.join(dimension, left_on=["k"], right_on=["k"]),
        name + "-inner",
        False,
        ["v"],
    )
    _check(
        source.join(dimension, left_on=["k"], right_on=["k"], how="left"),
        name + "-left",
        False,
        ["v"],
    )
    # Groups, in first-appearance order or any order.
    _check(
        source.group_by(["k"], maintain_order=True).agg(
            [col("v").sum().alias("total"), col("f").max().alias("top")]
        ),
        name + "-ordered-groups",
        True,
    )
    _check(
        source.group_by(["c", "k"]).agg([col("v").min().alias("least")]),
        name + "-groups",
        False,
        ["c", "k"],
    )
    # A full sort, its first rows, and slices of the filtered rows.
    _check(source.sort("v", descending=True), name + "-sorted", True)
    _check(
        source.sort(["k", "v"]).head(250),
        name + "-top",
        True,
    )
    _check(
        source.filter(col("k") != 3).head(2500),
        name + "-head",
        True,
    )
    _check(
        source.filter(col("k") > 10).slice(5000, 3000),
        name + "-slice",
        True,
    )
    # Distinct rows, ordered and not.
    _check(
        source.select(["k", "c"]).unique(maintain_order=True),
        name + "-unique-ordered",
        True,
    )
    _check(
        source.select(["k", "v", "s"]).unique(
            ["k"], keep="first", maintain_order=True
        ),
        name + "-unique-first",
        True,
    )
    _check(
        source.select(["c"]).unique(),
        name + "-unique",
        False,
        ["c"],
    )
    # No rows at all, through each route.
    _check(source.filter(col("k") < 0), name + "-empty", True)
    _check(
        source.filter(col("k") < 0)
        .group_by(["k"])
        .agg([col("v").sum().alias("total")]),
        name + "-empty-groups",
        True,
    )
    _check(
        source.filter(col("k") < 0).sort("v").head(5),
        name + "-empty-top",
        True,
    )


def test_frame_queries_sink_their_collected_rows() raises:
    _queries(_source(ROWS).lazy(), "frame")


def test_file_queries_sink_their_collected_rows() raises:
    _queries(_file_source(), "file")


def test_rows_from_the_pipeline_go_to_the_file_as_morsels_complete() raises:
    """Rows straight from the steps are written by the workers; a growing
    categorical dictionary waits for the scan's end."""
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    _source(ROWS).lazy().filter(col("k") < 60).sink_csv(
        _path("traced-frame.csv"), batch_size=BATCH
    )
    var frame_paths = trace.text()
    _ = trace^
    trace = StrategyTrace()
    _file_source().select(["k", "v", "s"]).sink_parquet(
        _path("traced-file.parquet"), batch_size=BATCH
    )
    var file_paths = trace.text()
    _ = trace^
    trace = StrategyTrace()
    _file_source().sink_csv(_path("traced-dictionary.csv"), batch_size=BATCH)
    var dictionary_paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        "dataframe-path: pipeline.file_sink" in frame_paths,
        frame_paths,
    )
    assert_true(
        "dataframe-path: pipeline.file_sink" in file_paths,
        file_paths,
    )
    assert_true(
        "dataframe-path: pipeline.file_sink" not in dictionary_paths,
        dictionary_paths,
    )
