"""External physical coercions and dictionary values after row transformations."""
# contract-group: kernel.row_ops
# contract-group: io.arrow
# contract-group: io.parquet
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    col,
    when,
    read_parquet,
)
from tests.support.parquet_runtime import Observer


def test_parquet_external_float16_and_utf8_view_coerce_without_changing_null_cells() raises:
    var observer = Observer()
    var frame = read_parquet("tests/fixtures/parquet/coercions.parquet")
    assert_equal(frame.columns(), List[String](["half", "view"]))
    assert_equal(frame.column("half").dtype(), DataType.FLOAT32)
    assert_equal(frame.column("view").dtype(), DataType.STRING)
    assert_equal(frame.item(0, "half").float32(), Float32(1.5))
    assert_equal(frame.item(2, "half").float32(), Float32(-2.25))
    assert_equal(frame.item(0, "view").string(), "long external string")
    assert_equal(frame.item(2, "view").string(), "é")
    for name in frame.columns():
        assert_true(frame.item(1, name).is_null())
    observer.assert_balanced()


def assert_dictionary_rows(frame: DataFrame, rows: List[Int]) raises:
    var words: List[String] = [
        "red",
        "",
        "blue",
        "poison",
        "red",
        "green",
        "red",
        "blue",
        "last",
    ]
    var values: List[Int64] = [0, 1, 2, 3, 100, 101, 102, 200, 201]
    assert_equal(frame.height(), len(rows))
    # This is a route witness, separate from independently checked cells below.
    assert_true(
        Bool(frame.column("s")._dictionary_codes()),
        "fixture must exercise retained dictionary code dispatch",
    )
    for row in range(len(rows)):
        assert_equal(frame.item(row, "x").int64(), values[rows[row]])
        assert_equal(frame.item(row, "s").is_null(), rows[row] == 3)
        if rows[row] != 3:
            assert_equal(frame.item(row, "s").string(), words[rows[row]])
    var derived = frame.select_exprs(
        [
            (col("s") == "red").alias("red"),
            col("s").str().to_uppercase().alias("upper"),
            col("s").str().len_chars().alias("length"),
            col("s").is_in([String("blue"), String("last")]).alias("member"),
            when(col("s") == "red")
            .then(col("x"))
            .otherwise(-1)
            .alias("picked"),
        ]
    )
    for row in range(len(rows)):
        var index = rows[row]
        var present = index != 3
        for name in ["red", "upper", "length", "member"]:
            assert_equal(derived.item(row, name).is_null(), not present)
        if present:
            assert_equal(derived.item(row, "red").bool(), words[index] == "red")
            var upper: List[String] = [
                "RED",
                "",
                "BLUE",
                "POISON",
                "RED",
                "GREEN",
                "RED",
                "BLUE",
                "LAST",
            ]
            assert_equal(derived.item(row, "upper").string(), upper[index])
            assert_equal(
                derived.item(row, "length").int64(),
                Int64(words[index].byte_length()),
            )
            assert_equal(
                derived.item(row, "member").bool(),
                words[index] == "blue" or words[index] == "last",
            )
        assert_equal(
            derived.item(row, "picked").int64(),
            values[index] if present and words[index] == "red" else -1,
        )
    var grouped = frame.group_by("s").agg(col("x").sum())
    var seen = List[String]()
    for row in range(grouped.height()):
        var key = (
            String("<null>") if grouped.item(row, "s")
            .is_null() else grouped.item(row, "s")
            .string()
        )
        for previous in seen:
            assert_true(key != previous)
        seen.append(key)
        var total: Int64 = 0
        for index in rows:
            var want = String("<null>") if index == 3 else words[index]
            if want == key:
                total += values[index]
        assert_equal(grouped.item(row, "x").int64(), total)


def test_parquet_dictionary_codes_stay_aligned_after_gathers_and_outer_join_null_extension() raises:
    var observer = Observer()
    var source = read_parquet("tests/fixtures/parquet/dictionary.parquet")
    observer.assert_balanced()
    assert_dictionary_rows(source, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert_dictionary_rows(
        source.take([8, 0, 3, 2, 4, 7, 0]), [8, 0, 3, 2, 4, 7, 0]
    )
    assert_dictionary_rows(
        source.filter(col("x") >= 100).slice(1, 3), [5, 6, 7]
    )
    assert_dictionary_rows(
        source.sort("x", descending=True), [8, 7, 6, 5, 4, 3, 2, 1, 0]
    )
    var left = DataFrame(
        [
            Series("x", Column[Int64]([0, 100, 777])),
            Series("weight", Column[Int64]([1, 2, 4])),
        ]
    )
    var joined = left.join(source, on="x", how="left").sort("x")
    assert_equal(joined.height(), 3)
    assert_true(Bool(joined.column("s")._dictionary_codes()))
    assert_equal(joined.item(0, "s").string(), "red")
    assert_equal(joined.item(1, "s").string(), "red")
    assert_true(joined.item(2, "s").is_null())
    var groups = joined.group_by("s").agg(col("weight").sum())
    assert_equal(groups.height(), 2)
    for row in range(2):
        if groups.item(row, "s").is_null():
            assert_equal(groups.item(row, "weight").int64(), Int64(4))
        else:
            assert_equal(groups.item(row, "s").string(), "red")
            assert_equal(groups.item(row, "weight").int64(), Int64(3))
