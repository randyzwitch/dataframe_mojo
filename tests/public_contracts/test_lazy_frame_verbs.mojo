"""Lazy describe, clear, gather, top_k, bottom_k and diagonal and
horizontal concat give the eager DataFrame results under every executor
setting. Filters written above a gather stay above it, filters and column
selections above a diagonal concat reach every input, and column
selections above a horizontal concat reach both inputs."""
# contract-group: lazy.frame_verbs
from std.memory import ArcPointer
from std.os import getenv, makedirs
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    LazyFrame,
    Series,
    col,
    concat,
    lit,
    read_csv,
    scan_csv,
)
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import ScopedEnvironment

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
    assert_true("SCAN frame [project u]" in text, text)
    # The index plan is read whole.
    assert_true("SCAN frame [project index]" not in text, text)


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
    assert_true("TOP_K 2 by u" in text, text)
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
    # A name the inputs give two types that do not promote raises, as
    # eagerly.
    var other = DataFrame([Series("k", Column[String](["x"]))])
    with assert_raises(contains="concat diagonal: column k"):
        _ = concat([first.lazy(), other.lazy()], "diagonal")
    with assert_raises(contains="concat diagonal: column k"):
        _ = concat([first.copy(), other.copy()], "diagonal")
    with assert_raises():
        _ = concat(List[LazyFrame](), "diagonal")
    with assert_raises():
        _ = concat([first.lazy()], "sideways")
    check(concat([first.lazy()], "diagonal"), first)


def test_concat_of_joins_pruned_apart() raises:
    """A union of two joins whose inputs keep different columns: a filter
    above the second join reads a right column that nothing above the
    union reads, so the second join's right input keeps one more column
    than the first's. The union still stacks the same columns from both,
    over files and over frames, on one worker and on eight."""
    var directory = getenv("COVERAGE_TEST_TMP")
    assert_true(Bool(directory))
    directory += "/concat-of-joins"
    makedirs(directory, exist_ok=True)
    var path = directory + "/rows.csv"
    var text = String("k,a,d\n")
    for row in range(40):
        text += String(row % 3) + "," + String(row % 7) + ","
        text += String(row * 5) + "\n"
    with open(path, "w") as file:
        file.write(text)
    var schema = CsvSchema(
        [CsvField.int64("k"), CsvField.int64("a"), CsvField.int64("d")]
    )
    var frame = read_csv(path, schema)
    var sources = List[LazyFrame]()
    sources.append(scan_csv(path, schema))
    sources.append(frame.lazy())
    var eager = frame.head(3).join(
        frame.filter(col("a").is_in([lit(Int64(0)), lit(Int64(4))])),
        on=["k"],
        how="left",
    )
    var expected = (
        concat([eager.copy(), eager.filter(col("a_right") == col("a_right"))])
        .group_by(["a"])
        .agg([col("d_right").count().alias("n")])
        .sort("a")
    )
    var narrowed = (
        concat([frame.filter(col("a") > 2), frame.filter(col("d") > 50)])
        .select(["k"])
        .sort("k")
    )
    for threads in ["1", "8"]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
        for source in sources:
            var joined = source.head(3).join(
                source.filter(col("a").is_in([lit(Int64(0)), lit(Int64(4))])),
                on=["k"],
                how="left",
            )
            check(
                concat(
                    [
                        joined.copy(),
                        joined.filter(col("a_right") == col("a_right")),
                    ]
                )
                .group_by(["a"])
                .agg([col("d_right").count().alias("n")])
                .sort("a"),
                expected,
            )
            # Two filters that read different columns, neither of them
            # read above the union.
            check(
                concat(
                    [source.filter(col("a") > 2), source.filter(col("d") > 50)]
                )
                .select(["k"])
                .sort("k"),
                narrowed,
            )
        _ = workers^


def test_selected_columns_keep_field_metadata_in_a_union() raises:
    """A column passed through a select keeps its Arrow field metadata,
    both in the rows and in the schema the plan derives for the select,
    so a union of a selected input and an unselected one stacks. Found by
    the differential fuzzer (tests/fuzz): projection pushdown put a
    selection on one input of a union over Parquet data, and the derived
    schemas disagreed on a labelled column's metadata."""
    # Packed Arrow metadata: one entry, "tag" -> "raw".
    var metadata: List[UInt8] = [1, 0, 0, 0, 3, 0, 0, 0, 116, 97, 103]
    metadata += [3, 0, 0, 0, 114, 97, 119]
    var label = Series(
        "s", Column[String](["a", "b", "c"], [True, False, True])
    )
    label._field_metadata = ArcPointer(metadata^)
    var frame = DataFrame([label^, Series("u", Column[Int64]([1, 2, 3]))])
    var expected = concat([frame.copy(), frame.copy()])
    for threads in ["1", "8"]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
        var plans = List[LazyFrame]()
        plans.append(concat([frame.lazy(), frame.lazy().select(["s", "u"])]))
        plans.append(
            concat(
                [
                    frame.lazy().select_exprs([col("s"), col("u")]),
                    frame.lazy().with_columns([col("u").alias("u")]),
                ]
            )
        )
        plans.append(
            concat([frame.lazy(), frame.lazy().filter(col("u") > 0)]).select(
                ["s"]
            )
        )
        for k in range(len(plans)):
            var result = plans[k].collect()
            assert_true(Bool(result.column("s")._field_metadata), String(k))
            if k < 2:
                check(plans[k], expected)


def test_join_over_a_union_of_joins_pruned_apart() raises:
    """A join reads a union whose two inputs are joins narrowed to
    different columns (a filter above the second reads a right column
    that nothing above the union reads). The union is aligned before the
    join above it reads the union's columns, over files and over frames,
    on one worker and on eight."""
    var directory = getenv("COVERAGE_TEST_TMP")
    assert_true(Bool(directory))
    directory += "/join-over-union-of-joins"
    makedirs(directory, exist_ok=True)
    var left_path = directory + "/left.csv"
    var right_path = directory + "/right.csv"
    var left_text = String("k,b,c\n")
    var right_text = String("k,b,t\n")
    for row in range(30):
        left_text += String(row % 9) + "," + String(row % 2) + ","
        left_text += String(row * 3) + "\n"
        right_text += String(row % 11) + "," + String(row % 3 // 2) + ","
        right_text += String(row % 9) + "\n"
    with open(left_path, "w") as file:
        file.write(left_text)
    with open(right_path, "w") as file:
        file.write(right_text)
    var left_schema = CsvSchema(
        [CsvField.int64("k"), CsvField.int64("b"), CsvField.int64("c")]
    )
    var right_schema = CsvSchema(
        [CsvField.int64("k"), CsvField.int64("b"), CsvField.int64("t")]
    )
    var left = read_csv(left_path, left_schema)
    var right = read_csv(right_path, right_schema)
    var eager = left.filter(col("b") != lit(Int64(1))).join(
        right, on=["k"], how="full", coalesce=True
    )
    var expected = (
        concat([eager.copy(), eager.filter(col("k") == col("t"))])
        .join(right, on=["b"], how="inner")
        .group_by(["b_right"])
        .agg([col("c").null_count().alias("n"), col("k").count().alias("m")])
        .sort(["b_right"], nulls_last=True)
    )
    var lefts = List[LazyFrame]()
    var rights = List[LazyFrame]()
    lefts.append(scan_csv(left_path, left_schema))
    rights.append(scan_csv(right_path, right_schema))
    lefts.append(left.lazy())
    rights.append(right.lazy())
    for threads in ["1", "8"]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
        for s in range(2):
            var joined = (
                lefts[s]
                .filter(col("b") != lit(Int64(1)))
                .join(rights[s], on=["k"], how="full", coalesce=True)
            )
            check(
                concat([joined.copy(), joined.filter(col("k") == col("t"))])
                .join(rights[s], on=["b"], how="inner")
                .group_by(["b_right"])
                .agg(
                    [
                        col("c").null_count().alias("n"),
                        col("k").count().alias("m"),
                    ]
                )
                .sort(["b_right"], nulls_last=True),
                expected,
            )
        _ = workers^


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
    assert_true("SCAN frame [project s]" in text, text)
    assert_true("WITH_COLUMNS twice\n" in text, text)
    assert_true(line_of(text, "FILTER") < line_of(text, "HCONCAT"), text)
    # Heights must match and names must not repeat.
    check_fails(concat([left.lazy(), right.lazy().head(3)], "horizontal"))
    check_fails(concat([left.lazy(), left.lazy()], "horizontal"))
