"""Joins on categorical keys whose two sides were encoded with different
dictionaries: two scans of one file, two files, or two in-memory frames. A
code means a value only within its own dictionary, so the join compares and
emits values, not codes: a lazy join, streamed or not, gives the eager
join's rows and labels, and right-only rows keep their own labels."""
# contract-group: table.join
from std.memory import ArcPointer
from std.os import getenv, makedirs
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
    read_csv,
    scan_csv,
)


def _directory() raises -> String:
    var directory = getenv("COVERAGE_TEST_TMP")
    assert_true(Bool(directory))
    directory += "/categorical-join-dictionaries"
    makedirs(directory, exist_ok=True)
    return directory^


def _file(name: String, text: String) raises -> String:
    var path = _directory() + "/" + name
    with open(path, "w") as file:
        file.write(text)
    return path^


def _schema() raises -> CsvSchema:
    return CsvSchema([CsvField.categorical("k")])


def _labels(frame: DataFrame) raises -> DataFrame:
    """The frame with categorical columns as their values, rows sorted: two
    dictionaries may give the same values different codes."""
    var columns = List[Series]()
    for column in frame._columns:
        assert_true(column.dtype().is_categorical(), column.name())
        columns.append(column.cast(DataType.STRING))
    return DataFrame(columns^, height=frame.height()).sort(
        frame.columns(), nulls_last=True
    )


def _same(query: LazyFrame, expected: DataFrame) raises:
    var want = _labels(expected)
    for optimize in [True, False]:
        var got = _labels(query.collect(optimize=optimize))
        assert_equal(got.columns(), want.columns())
        assert_true(got.equals(want), String(got) + "\n" + String(want))


def _cat(name: String, values: List[Optional[String]]) raises -> Series:
    var strings = List[String]()
    var valid = List[Bool]()
    for value in values:
        strings.append(value.value() if value else String(""))
        valid.append(Bool(value))
    return Series(name, Column[String](strings^, valid)).cast(
        DataType.CATEGORICAL
    )


def test_full_self_join_of_a_scan_prefix_keeps_right_only_labels() raises:
    # The last value is not among the first ten: its row is right-only, and
    # its code exists only in the right scan's dictionary.
    var path = _file(
        "prefix.csv",
        "k\ngreen\nblack\n\n\n\nwhite\nred\n\nblue\ngreen\nnew\n",
    )
    var scan = scan_csv(path, _schema())
    var frame = read_csv(path, _schema())
    _same(
        scan.head(10).join(scan, on="k", how="full"),
        frame.head(10).join(frame, on=["k"], how="full"),
    )
    _same(
        scan.head(10).join(scan, on="k", how="full", coalesce=False),
        frame.head(10).join(frame, on=["k"], how="full", coalesce=False),
    )


def test_joins_of_two_scans_compare_values_not_codes() raises:
    # Each file's dictionary gives code 0 to a different value.
    var left_path = _file("left.csv", "k\nred\nblack\n")
    var right_path = _file("right.csv", "k\ngreen\nblack\nred\n\n")
    var left = scan_csv(left_path, _schema())
    var right = scan_csv(right_path, _schema())
    var left_frame = read_csv(left_path, _schema())
    var right_frame = read_csv(right_path, _schema())
    for how in ["inner", "left", "right", "full"]:
        for coalesce in [True, False]:
            _same(
                left.join(right, on="k", how=how, coalesce=coalesce),
                left_frame.join(
                    right_frame, on=["k"], how=how, coalesce=coalesce
                ),
            )
    # One row on each side, no match: a right join keeps only the right
    # row, once.
    var red = _file("red.csv", "k\nred\n")
    var green = _file("green.csv", "k\ngreen\n")
    var result = (
        scan_csv(red, _schema())
        .join(scan_csv(green, _schema()), on="k", how="right")
        .collect()
    )
    assert_equal(result.height(), 1)
    assert_equal(
        result.column("k").cast(DataType.STRING).get(0).string(), "green"
    )


def test_self_join_of_one_row_scan_matches_its_row_once() raises:
    var path = _file("one.csv", "k\ngreen\n")
    var scan = scan_csv(path, _schema())
    var result = scan.join(scan, on="k", how="full").collect()
    assert_equal(result.height(), 1)
    assert_equal(
        result.column("k").cast(DataType.STRING).get(0).string(), "green"
    )


def test_joins_of_frames_with_different_dictionaries() raises:
    var left = DataFrame([_cat("k", [Optional[String]("black")])])
    var right = DataFrame(
        [_cat("k", [Optional[String]("uni"), Optional[String]("black")])]
    )
    var empty = DataFrame([_cat("k", List[Optional[String]]())])
    var null = DataFrame([_cat("k", [Optional[String](None)])])
    var red = DataFrame([_cat("k", [Optional[String]("red")])])
    var green = DataFrame([_cat("k", [Optional[String]("green")])])
    var pairs = [
        (left.copy(), right.copy()),
        (empty.copy(), null.copy()),
        (red.copy(), green.copy()),
    ]
    for pair in pairs:
        for how in ["inner", "left", "right", "full"]:
            for coalesce in [True, False]:
                _same(
                    pair[0]
                    .lazy()
                    .join(pair[1].lazy(), on="k", how=how, coalesce=coalesce),
                    pair[0].join(pair[1], on=["k"], how=how, coalesce=coalesce),
                )
    # The labels of an inner join that keeps both keys are the matched
    # value, not whatever value the other dictionary gives the same code.
    var kept = (
        left.lazy()
        .join(right.lazy(), on="k", how="inner", coalesce=False)
        .collect()
    )
    assert_equal(kept.height(), 1)
    assert_equal(
        kept.column("k").cast(DataType.STRING).get(0).string(), "black"
    )
    assert_equal(
        kept.column("k_right").cast(DataType.STRING).get(0).string(), "black"
    )


def test_right_key_of_a_frame_self_join_selects_alone() raises:
    # The key carries Arrow field metadata (one entry, k -> v), as a column
    # read from an Arrow file may: the right key's rows, read back from the
    # join's build rows, keep it, so they concatenate with the column's
    # other pieces.
    var key = _cat("k", [Optional[String]("green")])
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    key._field_metadata = ArcPointer(metadata^)
    var frame = DataFrame([key^])
    var picked = (
        frame.lazy()
        .join(frame.lazy(), on="k", how="inner", coalesce=False)
        .select(col("k_right").alias("e"))
        .collect()
    )
    assert_equal(picked.height(), 1)
    assert_equal(
        picked.column("e").cast(DataType.STRING).get(0).string(), "green"
    )
