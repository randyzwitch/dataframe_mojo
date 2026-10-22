"""Lazy describe, clear, gather, top_k, bottom_k and diagonal and
horizontal concat give the eager DataFrame results under every executor
setting. Filters written above a gather stay above it, filters and column
selections above a diagonal concat reach every input, and column
selections above a horizontal concat reach both inputs."""
# contract-group: lazy.frame_verbs
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    LazyFrame,
    Series,
    col,
    concat,
    lit,
)
from dataframe_metal.accelerator.metadata import prune_projections, schema_frame
from tests.support.coverage_compare import assert_frame

# Stands for a null Int64 in literal expectations.
comptime NULL = Int64.MIN


def source() raises -> DataFrame:
    """Seven rows with nulls; `u` is unique, so sorts on it have no ties."""
    return DataFrame(
        [
            Series(
                "k",
                Column[Int64](
                    [1, 0, 3, 4, 0, 6, 7],
                    [True, False, True, True, False, True, True],
                ),
            ),
            Series(
                "s",
                Column[String](
                    ["a", "b", "", "d", "e", "f", ""],
                    [True, True, False, True, True, True, False],
                ),
            ),
            Series(
                "f",
                Column[Float64](
                    [0.5, 1.5, 2.5, 0.0, 4.5, 5.5, 6.5],
                    [True, True, True, False, True, True, True],
                ),
            ),
            Series("u", Column[Int64]([30, 10, 60, 20, 50, 0, 40])),
        ]
    )


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


def line_of(text: String, operator: String) -> Int:
    """The line of `text` (from 0) that starts with `operator` once
    indented, or -1."""
    var lines = text.split("\n")
    for k in range(len(lines)):
        if String(lines[k].strip()).startswith(operator):
            return k
    return -1


def test_describe() raises:
    var frame = source()
    assert_frame(frame.lazy().describe(), frame.describe())
    assert_frame(
        frame.lazy().filter(col("k") > 1).describe([0.1, 0.9], "linear"),
        frame.filter(col("k") > 1).describe([0.1, 0.9], "linear"),
    )
    var no_percentiles = List[Float64]()
    assert_frame(
        frame.lazy().select(["s", "u"]).describe(no_percentiles),
        frame.select(["s", "u"]).describe(no_percentiles),
    )
    # Over no rows, and over no columns.
    assert_frame(frame.lazy().head(0).describe(), frame.head(0).describe())
    var empty = DataFrame(List[Series]())
    assert_frame(empty.lazy().describe(), empty.describe())
    with assert_raises(contains="percentiles"):
        _ = frame.lazy().describe([1.5])


def test_clear() raises:
    var frame = source()
    for n in [0, 1, 3, 9]:
        check(frame.lazy().clear(n), frame.clear(n))
        check(
            frame.lazy().filter(col("k") > 1).select(["s", "f"]).clear(n),
            frame.filter(col("k") > 1).select(["s", "f"]).clear(n),
        )
    var cleared = frame.clear(2)
    assert_equal(cleared.height(), 2)
    assert_equal(cleared.columns(), frame.columns())
    assert_ints(cleared, "k", [NULL, NULL])
    with assert_raises():
        _ = frame.lazy().clear(-1)
    with assert_raises():
        _ = frame.clear(-1)


def test_gather() raises:
    var frame = source()
    var lists: List[List[Int]] = [[], [0], [6, 0, 3], [-1, -7, 2, 2], [5, 5, 5]]
    for indices in lists:
        check(frame.lazy().gather(indices), frame.gather(indices))
        check(
            frame.lazy().gather(indices, null_on_oob=True),
            frame.gather(indices, null_on_oob=True),
        )
    assert_ints(frame.gather([-1, 0, 2]), "u", [40, 30, 60])
    # Out of range raises, eagerly and when the plan runs, unless asked for
    # a row of nulls.
    with assert_raises(contains="out of bounds"):
        _ = frame.gather([7])
    with assert_raises(contains="out of bounds"):
        _ = frame.gather([-8])
    check_fails(frame.lazy().gather([0, 7]))
    var padded = frame.gather([7, 1, -8], null_on_oob=True)
    check(frame.lazy().gather([7, 1, -8], null_on_oob=True), padded)
    assert_ints(padded, "u", [NULL, 10, NULL])
    # Indices from a plan: computed, with a null index giving a null row.
    var positions = DataFrame(
        [Series("at", Column[Int64]([2, 0, 5], [True, False, True]))]
    )
    var gathered = frame.lazy().gather(positions.lazy())
    var expected = frame.gather([2, 99, 5], null_on_oob=True)
    check(gathered, expected)
    assert_ints(gathered.collect(), "u", [60, NULL, 0])
    var shifted = frame.lazy().gather(
        positions.lazy()
        .select((col("at") - 1).alias("at"))
        .filter(col("at").is_not_null())
    )
    check(shifted, frame.gather([1, 4]))
    # The indices must be one integer column.
    check_fails(frame.lazy().gather(frame.lazy().select(["s"])))
    check_fails(frame.lazy().gather(frame.lazy().select(["k", "u"])))


def test_gather_keeps_filters_above_and_narrows_its_input() raises:
    var frame = source()
    var plan = frame.lazy().gather([6, 5, 4]).filter(col("k") > 5)
    check(plan, frame.gather([6, 5, 4]).filter(col("k") > 5))
    # Filtering first would make positions 6, 5 and 4 different rows.
    assert_ints(plan.collect(), "u", [40, 0])
    var text = plan.explain(streaming=False)
    assert_true(line_of(text, "FILTER") < line_of(text, "GATHER"), text)
    var selected = frame.lazy().gather([1, 2]).select(["u"])
    check(selected, frame.gather([1, 2]).select(["u"]))
    text = selected.explain(streaming=False)
    assert_true(text.find("SCAN frame [project u]") >= 0, text)
    # The index plan is read whole.
    assert_true(text.find("SCAN frame [project index]") < 0, text)


def test_top_and_bottom_k() raises:
    var frame = source()
    for k in [0, 1, 3, 7, 10]:
        for reverse in [False, True]:
            check(
                frame.lazy().top_k(k, "u", reverse=reverse),
                frame.top_k(k, "u", reverse=reverse),
            )
            check(
                frame.lazy().bottom_k(k, "u", reverse=reverse),
                frame.bottom_k(k, "u", reverse=reverse),
            )
            check(
                frame.lazy().top_k(k, ["k", "u"], reverse=reverse),
                frame.top_k(k, ["k", "u"], reverse=reverse),
            )
    # Nulls rank last whichever way the keys sort.
    assert_ints(frame.lazy().top_k(3, "k").collect(), "k", [7, 6, 4])
    assert_ints(frame.lazy().bottom_k(3, "k").collect(), "k", [1, 3, 4])
    assert_ints(
        frame.lazy().bottom_k(6, "k", reverse=True).collect(),
        "k",
        [7, 6, 4, 3, 1, NULL],
    )
    assert_ints(frame.top_k(2, "u", reverse=True), "u", [0, 10])
    var text = frame.lazy().top_k(2, "u").explain(streaming=False)
    assert_true(text.find("TOP_K 2 by u") >= 0, text)
    with assert_raises():
        _ = frame.lazy().top_k(-1, "u")


def test_vertical_and_diagonal_concat() raises:
    var frame = source()
    var first = frame.select(["k", "s"])
    var second = frame.select(["s", "f"]).head(3)
    var third = frame.select(["u", "k"]).tail(2)
    check(
        concat([first.lazy(), first.lazy(), first.lazy()]),
        concat([first.copy(), first.copy(), first.copy()]),
    )
    var diagonal = concat(
        [first.lazy(), second.lazy(), third.lazy()], "diagonal"
    )
    check(
        diagonal,
        concat([first.copy(), second.copy(), third.copy()], "diagonal"),
    )
    var expected = concat(
        [first.copy(), second.copy(), third.copy()], "diagonal"
    )
    assert_equal(expected.columns(), ["k", "s", "f", "u"])
    # A filter above reaches every input, below the select that aligns
    # it, except the input that lacks its column, where it stays above the
    # typed null.
    var narrowed = diagonal.filter(col("k") > 2).select(["k", "u"])
    check(
        narrowed,
        concat([first.copy(), second.copy(), third.copy()], "diagonal")
        .filter(col("k") > 2)
        .select(["k", "u"]),
    )
    var text = narrowed.explain(streaming=False)
    assert_true(line_of(text, "CONCAT") < line_of(text, "FILTER"), text)
    assert_equal(len(text.split("FILTER")), 4, text)
    # A name the inputs give two types raises, as eagerly.
    var other = DataFrame([Series("k", Column[Float64]([1.5]))])
    with assert_raises(contains="concat diagonal: column k"):
        _ = concat([first.lazy(), other.lazy()], "diagonal")
    with assert_raises(contains="concat diagonal: column k"):
        _ = concat([first.copy(), other.copy()], "diagonal")
    with assert_raises():
        _ = concat(List[LazyFrame](), "diagonal")
    with assert_raises():
        _ = concat([first.lazy()], "sideways")
    check(concat([first.lazy()], "diagonal"), first)


def test_horizontal_concat() raises:
    var frame = source()
    var left = frame.select(["k", "s"])
    var right = frame.select(["f", "u"]).with_columns(
        (col("u") * 2).alias("twice")
    )
    var plan = concat(
        [
            left.lazy(),
            frame.lazy()
            .select(["f", "u"])
            .with_columns((col("u") * 2).alias("twice")),
        ],
        "horizontal",
    )
    check(plan, concat([left.copy(), right.copy()], "horizontal"))
    var third = (
        frame.select(["u"]).with_columns(col("u").alias("v")).drop(["u"])
    )
    check(
        concat([left.lazy(), right.lazy(), third.lazy()], "horizontal"),
        concat([left.copy(), right.copy(), third.copy()], "horizontal"),
    )
    # A column selection reaches both inputs; a filter stays above.
    var selected = plan.select(["s", "twice"]).filter(col("twice") > 50)
    check(
        selected,
        concat([left.copy(), right.copy()], "horizontal")
        .select(["s", "twice"])
        .filter(col("twice") > 50),
    )
    var text = selected.explain(streaming=False)
    assert_true(text.find("SCAN frame [project s]") >= 0, text)
    assert_true(text.find("WITH_COLUMNS twice\n") >= 0, text)
    assert_true(line_of(text, "FILTER") < line_of(text, "HCONCAT"), text)
    # Heights must match and names must not repeat.
    check_fails(concat([left.lazy(), right.lazy().head(3)], "horizontal"))
    check_fails(concat([left.lazy(), left.lazy()], "horizontal"))
