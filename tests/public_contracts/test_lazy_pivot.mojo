"""Lazy pivot gives DataFrame.pivot's cells under every executor setting,
with its columns named up front (Polars 1.44's `on_columns`): listed names
in their order, a name no row has as a column of nulls of the aggregate's
type. Filters on index columns move below it; other filters stay above."""
# contract-group: lazy.pivot
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    LazyFrame,
    Series,
    col,
    lit,
)
from dataframe.accelerator.metadata import prune_projections, schema_frame
from tests.support.coverage_compare import assert_frame

# Stands for a null Int64 in literal expectations.
comptime NULL = Int64.MIN


def source() raises -> DataFrame:
    """Rows keyed by (k, g), with `o` naming the new columns (one null) and
    two cells (x/a, z/c) holding two rows."""
    return DataFrame(
        [
            Series(
                "k",
                Column[String](["x", "x", "y", "z", "z", "x", "y", "z"]),
            ),
            Series("g", Column[Int64]([1, 1, 2, 1, 1, 1, 2, 1])),
            Series(
                "o",
                Column[String](
                    ["a", "b", "a", "c", "", "a", "b", "c"],
                    [True, True, True, True, False, True, True, True],
                ),
            ),
            Series(
                "v",
                Column[Int64](
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [True, True, True, True, True, False, True, True],
                ),
            ),
            Series("n", Column[Int64]([10, 20, 30, 40, 50, 60, 70, 80])),
        ]
    )


def unique_cells() raises -> DataFrame:
    """The source without the second row of each doubled cell."""
    return source().head(5)


def check(plan: LazyFrame, expected: DataFrame) raises:
    """Every executor setting gives `expected`, and the schema matches."""
    for optimize in [True, False]:
        for streaming in [True, False]:
            for batch_size in [65536, 2]:
                assert_frame(
                    plan.collect(
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=batch_size,
                    ),
                    expected,
                )
    var schema = List[String]()
    for field in expected.schema():
        schema.append(field.name + ": " + field.dtype.name())
    assert_equal(plan.collect_schema(), schema)
    var metadata = schema_frame(plan, len(plan._nodes) - 1)
    assert_equal(metadata.columns(), expected.columns())
    var pruned = plan.copy()
    prune_projections(pruned)
    assert_frame(pruned.collect(optimize=False, streaming=False), expected)


def check_fails(plan: LazyFrame) raises:
    for optimize in [True, False]:
        for streaming in [True, False]:
            with assert_raises():
                _ = plan.collect(optimize=optimize, streaming=streaming)


def line_of(text: String, operator: String) -> Int:
    """The line of `text` (from 0) that starts with `operator` once
    indented, or -1."""
    var lines = text.split("\n")
    for k in range(len(lines)):
        if String(lines[k].strip()).startswith(operator):
            return k
    return -1


def assert_ints(frame: DataFrame, name: String, expected: List[Int64]) raises:
    """Column `name` holds `expected`, NULL meaning a null."""
    assert_equal(frame.height(), len(expected))
    for row in range(len(expected)):
        var value = frame.item(row, name)
        if expected[row] == NULL:
            assert_true(value.is_null(), name + " row " + String(row))
        else:
            assert_true(not value.is_null(), name + " row " + String(row))
            assert_equal(value.int64(), expected[row])


def test_all_columns_match_eager() raises:
    var frame = source()
    var names: List[String] = ["a", "b", "c", "null"]
    var index_sets: List[List[String]] = [["k"], ["k", "g"], []]
    for index in index_sets:
        for function in [
            "first",
            "last",
            "sum",
            "mean",
            "min",
            "max",
            "count",
            "len",
            "median",
        ]:
            check(
                frame.lazy().pivot(
                    "o",
                    names,
                    index=index,
                    values="v",
                    aggregate_function=function,
                ),
                frame.pivot(
                    "o", index=index, values="v", aggregate_function=function
                ),
            )
        # Without an aggregate each cell holds one row; with no index,
        # that leaves one row per `on` value.
        var once = unique_cells() if len(index) > 0 else source().slice(1, 4)
        var expected = once.pivot("o", index=index, values="v")
        check(
            once.lazy().pivot(
                "o",
                List[String](expected.columns()[len(index) :]),
                index=index,
                values="v",
            ),
            expected,
        )


def test_named_columns_choose_and_order_output() raises:
    var frame = source()
    var plan = frame.lazy().pivot(
        "o", ["c", "a", "q"], index=["k"], values="v", aggregate_function="sum"
    )
    var wide = frame.pivot(
        "o", index=["k"], values="v", aggregate_function="sum"
    )
    var expected = DataFrame(
        [
            wide.column("k"),
            wide.column("c"),
            wide.column("a"),
            Series.full_null("q", DataType.INT64, wide.height()),
        ]
    )
    check(plan, expected)
    # Literal cells: x has a: 1 + null, y has a: 3, z has c: 4 + 8.
    var collected = plan.collect()
    assert_equal(collected.columns(), ["k", "c", "a", "q"])
    assert_ints(collected, "a", [1, 3, NULL])
    assert_ints(collected, "c", [NULL, NULL, 12])
    # A name no row has takes the aggregate's type, also with no rows.
    var means = frame.lazy().pivot(
        "o", ["q"], index=["k"], values="v", aggregate_function="mean"
    )
    assert_equal(means.collect_schema(), ["k: string", "q: float64"])
    assert_equal(
        frame.lazy()
        .head(0)
        .pivot("o", ["a", "b"], index=["k", "g"], values="v")
        .collect_schema(),
        ["k: string", "g: int64", "a: int64", "b: int64"],
    )
    # Names from `on` values: an Int64 column names them by their text.
    var numbered = frame.lazy().pivot(
        "g",
        Series("g", Column[Int64]([2, 1])),
        index=["k"],
        values="n",
        aggregate_function="max",
    )
    var by_g = frame.pivot(
        "g", index=["k"], values="n", aggregate_function="max"
    )
    check(
        numbered,
        DataFrame([by_g.column("k"), by_g.column("2"), by_g.column("1")]),
    )


def test_errors() raises:
    var frame = source()
    with assert_raises(contains="collides"):
        _ = frame.lazy().pivot("o", ["k"], index=["k"], values="v")
    with assert_raises(contains="collides"):
        _ = frame.lazy().pivot("o", ["a", "a"], index=["k"], values="v")
    # Doubled cells without an aggregate, an unknown aggregate and unknown
    # columns raise when the plan runs, as eagerly.
    check_fails(frame.lazy().pivot("o", ["a"], index=["k"], values="v"))
    check_fails(
        frame.lazy().pivot(
            "o", ["a"], index=["k"], values="v", aggregate_function="mode"
        )
    )
    check_fails(frame.lazy().pivot("missing", ["a"], index=["k"], values="v"))
    check_fails(frame.lazy().pivot("o", ["a"], index=["k"], values="missing"))


def test_filters_and_projections() raises:
    var frame = source()
    var names: List[String] = ["a", "b"]
    # A filter on index columns moves below the pivot.
    var on_index = (
        frame.lazy()
        .pivot(
            "o", names, index=["k", "g"], values="v", aggregate_function="max"
        )
        .filter(col("k") != "y")
    )
    var wide = frame.pivot(
        "o", index=["k", "g"], values="v", aggregate_function="max"
    )
    var expected = wide.select(["k", "g", "a", "b"]).filter(col("k") != "y")
    check(on_index, expected)
    var text = on_index.explain(streaming=False)
    assert_true(line_of(text, "PIVOT on o") < line_of(text, "FILTER"), text)
    # A filter on a new column stays above it.
    var on_cell = (
        frame.lazy()
        .pivot("o", names, index=["k"], values="v", aggregate_function="max")
        .filter(col("a") > 2)
    )
    check(
        on_cell,
        frame.pivot("o", index=["k"], values="v", aggregate_function="max")
        .select(["k", "a", "b"])
        .filter(col("a") > 2),
    )
    text = on_cell.explain(streaming=False)
    assert_true(line_of(text, "FILTER") < line_of(text, "PIVOT on o"), text)
    # The scan reads the index, `on` and values columns only.
    assert_true(text.find("SCAN frame [project k, o, v]") >= 0, text)
