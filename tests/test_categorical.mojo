"""Dictionary-encoded strings: the Categorical type (#106).

A categorical must behave like the String column it encodes: every result
here is compared with the same operation on the strings. It stores UInt32
codes plus a dictionary, and grouping, joins and unique work on the codes.
"""
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
    concat,
    lit,
    read_csv,
    write_csv,
)
from dataframe.arrow import ArrowArray, ArrowSchema, export_arrow, import_arrow

comptime CSV_PATH = "/tmp/dataframe_mojo_categorical.csv"


def strings() raises -> DataFrame:
    var words: List[String] = ["pear", "apple", "fig", "apple", "", "kiwi"]
    var keys = List[String]()
    var valid = List[Bool]()
    var values = List[Int64]()
    for i in range(60):
        keys.append(words[(i * 7) % len(words)])
        valid.append(i % 9 != 4)
        values.append(Int64(i % 11))
    return DataFrame(
        [
            Series("k", StringColumn(keys, valid)),
            Series("v", Column[Int64](values^)),
        ]
    )


def categorical() raises -> DataFrame:
    return strings().with_columns([col("k").cast("categorical")])


def as_strings(frame: DataFrame) raises -> DataFrame:
    """Every categorical column cast back to String, for comparison."""
    var exprs = List[Expr]()
    for field in frame.schema():
        var e = col(field.name)
        if field.dtype.is_categorical():
            e = e.cast("string").alias(field.name)
        exprs.append(e^)
    return frame.select_exprs(exprs)


def same(result: DataFrame, expected: DataFrame, label: String) raises:
    assert_true(as_strings(result).equals(expected), label)


def test_casts_names_and_cells() raises:
    var data = categorical()
    var dtype = data.column("k").dtype()
    assert_true(dtype.is_categorical())
    assert_equal(dtype.name(), "categorical")
    assert_true(DataType.parse("categorical").is_categorical())
    assert_equal(len(dtype.dictionary()[]), 5)
    same(data, strings(), "round trip")
    assert_equal(data.column("k").get(0).string(), "pear")
    assert_true(data.column("k").get(4).is_null())
    # A bare column keeps its type; a derived one is its strings.
    assert_true(
        data.select_exprs([col("k").alias("c")])
        .column("c")
        .dtype()
        .is_categorical()
    )
    assert_true(
        data.select_exprs([col("k").str().len_bytes()]).equals(
            strings().select_exprs([col("k").str().len_bytes()])
        )
    )
    assert_true("apple" in String(data))
    var empty = (
        strings().slice(0, 0).with_columns([col("k").cast("categorical")])
    )
    assert_equal(empty.height(), 0)
    assert_true(empty.column("k").dtype().is_categorical())


def test_expressions_read_the_strings() raises:
    var cats = categorical()
    var plain = strings()
    var exprs: List[Expr] = [
        col("k").str().to_uppercase().alias("upper"),
        col("k").str().contains("p").alias("has_p"),
        (col("k") == lit("fig")).alias("is_fig"),
        col("k").is_null().alias("missing"),
        col("k").n_unique().alias("distinct"),
        col("k").max().alias("largest"),
    ]
    for e in exprs:
        assert_true(
            cats.select_exprs([e.copy()]).equals(plain.select_exprs([e.copy()]))
        )
    same(
        cats.filter(col("k") == lit("apple")),
        plain.filter(col("k") == lit("apple")),
        "filter",
    )


def test_grouping_unique_and_counts_use_codes() raises:
    var cats = categorical()
    var plain = strings()
    same(
        cats.group_by("k", maintain_order=True).agg([col("v").sum()]),
        plain.group_by("k", maintain_order=True).agg([col("v").sum()]),
        "group_by",
    )
    var grouped = cats.group_by("k", maintain_order=True).agg([col("v").sum()])
    assert_true(grouped.column("k").dtype().is_categorical())
    same(
        cats.unique(["k"], keep="first", maintain_order=True),
        plain.unique(["k"], keep="first", maintain_order=True),
        "unique",
    )
    same(
        cats.column("k").value_counts(sort=True),
        plain.column("k").value_counts(sort=True),
        "value_counts",
    )


def test_joins_across_dictionaries_and_strings() raises:
    var cats = categorical()
    var plain = strings()
    # The right side's dictionary differs (other values, other order).
    var right_plain = DataFrame(
        [
            Series("k", StringColumn(["kiwi", "plum", "apple", "", "fig"])),
            Series("w", Column[Int64]([1, 2, 3, 4, 5])),
        ]
    )
    var right_cats = right_plain.with_columns([col("k").cast("categorical")])
    for how in ["inner", "left", "semi", "anti"]:
        var expected = plain.join(right_plain, "k", how)
        same(cats.join(right_cats, "k", how), expected, "cat-cat " + how)
        same(cats.join(right_plain, "k", how), expected, "cat-string " + how)


def test_concat_unions_dictionaries() raises:
    var first = categorical()
    var other = DataFrame(
        [
            Series("k", StringColumn(["plum", "apple", "zucchini"])),
            Series("v", Column[Int64]([1, 2, 3])),
        ]
    )
    var joined = concat(
        [first.copy(), other.with_columns([col("k").cast("categorical")])]
    )
    assert_true(joined.column("k").dtype().is_categorical())
    same(joined, concat([strings(), other.copy()]), "vertical")
    var diagonal = concat(
        [
            first.copy(),
            other.select(["k"]).with_columns([col("k").cast("categorical")]),
        ],
        "diagonal",
    )
    same(
        diagonal,
        concat([strings(), other.select(["k"])], "diagonal"),
        "diagonal",
    )


def test_sorting_is_by_value() raises:
    var cats = categorical()
    var plain = strings()
    for descending in [False, True]:
        for nulls_last in [False, True]:
            same(
                cats.sort(
                    ["k", "v"],
                    descending=[descending, False],
                    nulls_last=[nulls_last, True],
                ),
                plain.sort(
                    ["k", "v"],
                    descending=[descending, False],
                    nulls_last=[nulls_last, True],
                ),
                "sort",
            )
    same(cats.top_k(5, ["k", "v"]), plain.top_k(5, ["k", "v"]), "top_k")
    same(
        cats.lazy().sort(["k", "v"]).head(7).collect(),
        plain.lazy().sort(["k", "v"]).head(7).collect(),
        "lazy top-k",
    )


def test_csv_arrow_and_lazy() raises:
    var cats = categorical()
    write_csv(cats, CSV_PATH)
    assert_true(read_csv(CSV_PATH).equals(strings()))
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow(cats, array, schema)
    var back = import_arrow(array, schema)
    assert_true(back.column("k").dtype().is_categorical())
    assert_true(back.equals(cats))
    same(
        cats.lazy()
        .group_by("k", maintain_order=True)
        .agg([col("v").sum()])
        .collect(batch_size=16),
        strings().group_by("k", maintain_order=True).agg([col("v").sum()]),
        "lazy streaming group_by",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
