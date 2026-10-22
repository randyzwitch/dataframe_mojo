"""Lazy tail, reverse, shift and with_row_index give the eager DataFrame
methods' results under every executor setting (optimized or not, streaming
or materializing, one batch or many), including nulls, empty inputs, counts
past the height and negative counts. Filters and slices written above them
must not move below them: each composition is checked against literal rows
that a moved filter or slice would change."""
# contract-group: lazy.row_order
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
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.joins import lower_join_dag
from dataframe_metal.accelerator.metadata import prune_projections, schema_frame
from dataframe_metal.accelerator.rows import lower_rows
from tests.support.coverage_compare import assert_frame

# Stands for a null Int64 in literal expectations.
comptime NULL = Int64.MIN


def source() raises -> DataFrame:
    """Seven rows with nulls in every column."""
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
        ]
    )


def numbers(rows: Int) raises -> DataFrame:
    """`rows` rows: a key with nulls, a group with few values, a payload."""
    var keys = List[Int64]()
    var valid = List[Bool]()
    var groups = List[Int64]()
    var payload = List[Float64]()
    for i in range(rows):
        keys.append(Int64(i))
        valid.append(i % 7 != 3)
        groups.append(Int64((i * 5) % 11))
        payload.append(Float64(i) * 0.25)
    return DataFrame(
        [
            Series("k", Column[Int64](keys^, valid)),
            Series("g", Column[Int64](groups^)),
            Series("v", Column[Float64](payload^)),
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
    # The accelerator's metadata run, node by node, and its pruning.
    var metadata = schema_frame(plan, len(plan._nodes) - 1)
    assert_equal(metadata.columns(), expected.columns())
    for name in expected.columns():
        assert_true(
            metadata.column(name).dtype() == expected.column(name).dtype()
        )
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


def assert_strings(
    frame: DataFrame, name: String, expected: List[String]
) raises:
    """Column `name` holds `expected`, "<null>" meaning a null."""
    assert_equal(frame.height(), len(expected))
    for row in range(len(expected)):
        var value = frame.item(row, name)
        if expected[row] == "<null>":
            assert_true(value.is_null(), name + " row " + String(row))
        else:
            assert_equal(value.string(), expected[row])


def test_tail_counts() raises:
    var frame = source()
    for n in [-100, -8, -7, -3, -1, 0, 1, 3, 7, 8, 100]:
        check(frame.lazy().tail(n), frame.tail(n))
    var last = frame.lazy().tail(2).collect()
    assert_ints(last, "k", [6, 7])
    assert_strings(last, "s", ["f", "<null>"])
    assert_ints(frame.lazy().tail(-5).collect(), "k", [6, 7])
    assert_ints(frame.lazy().tail().collect(), "k", [3, 4, NULL, 6, 7])
    assert_equal(frame.lazy().tail(0).collect().height(), 0)
    assert_equal(frame.lazy().tail(100).collect().height(), 7)


def test_reverse_rows() raises:
    var frame = source()
    check(frame.lazy().reverse(), frame.reverse())
    check(frame.lazy().reverse().reverse(), frame)
    var reversed = frame.lazy().reverse().collect()
    assert_ints(reversed, "k", [7, 6, NULL, 4, 3, NULL, 1])
    assert_strings(reversed, "s", ["<null>", "f", "e", "d", "<null>", "b", "a"])


def test_shift_nulls_and_counts() raises:
    var frame = source()
    for n in [-100, -8, -7, -3, -1, 0, 1, 3, 7, 8, 100]:
        check(frame.lazy().shift(n), frame.shift(n))
    var later = frame.lazy().shift(2).collect()
    assert_ints(later, "k", [NULL, NULL, 1, NULL, 3, 4, NULL])
    assert_strings(
        later, "s", ["<null>", "<null>", "a", "b", "<null>", "d", "e"]
    )
    var earlier = frame.lazy().shift(-2).collect()
    assert_ints(earlier, "k", [3, 4, NULL, 6, 7, NULL, NULL])
    assert_ints(
        frame.lazy().shift().collect(), "k", [NULL, 1, NULL, 3, 4, NULL, 6]
    )
    assert_ints(
        frame.lazy().shift(100).collect(),
        "k",
        [NULL, NULL, NULL, NULL, NULL, NULL, NULL],
    )


def test_shift_fill_values() raises:
    var numeric = source().select(["k", "f"])
    for n in [-100, -3, -1, 0, 1, 3, 100]:
        check(
            numeric.lazy().shift(n, fill_value=lit(Int64(-1))),
            numeric.shift(n, fill_value=lit(Int64(-1))),
        )
    var filled = numeric.lazy().shift(2, fill_value=lit(Int64(-1))).collect()
    # Only the vacated rows take the fill; the input's own nulls stay.
    assert_ints(filled, "k", [-1, -1, 1, NULL, 3, 4, NULL])
    assert_equal(filled.item(0, "f").float64(), -1.0)
    assert_true(filled.item(5, "f").is_null())
    assert_true(filled.column("f").dtype() == DataType.FLOAT64)
    var up = numeric.lazy().shift(-3, fill_value=lit(Int64(-1))).collect()
    assert_ints(up, "k", [4, NULL, 6, 7, -1, -1, -1])
    # A reduction of the input fills too.
    var keys = source().select(["k"])
    var by_max = keys.lazy().shift(1, fill_value=col("k").max()).collect()
    assert_ints(by_max, "k", [7, 1, NULL, 3, 4, NULL, 6])
    check(
        keys.lazy().shift(1, fill_value=col("k").max()),
        keys.shift(1, fill_value=col("k").max()),
    )
    var text = source().select(["s"])
    var strings = text.lazy().shift(1, fill_value=lit("z")).collect()
    assert_strings(strings, "s", ["z", "a", "b", "<null>", "d", "e", "f"])
    # A fill that cannot take a column's type, or that is not one value.
    check_fails(numeric.lazy().shift(1, fill_value=lit("z")))
    check_fails(keys.lazy().shift(1, fill_value=col("k")))


def test_with_row_index() raises:
    var frame = source()
    check(frame.lazy().with_row_index(), frame.with_row_index())
    check(
        frame.lazy().with_row_index("row", 10),
        frame.with_row_index("row", 10),
    )
    var indexed = frame.lazy().with_row_index().collect()
    assert_equal(indexed.columns(), ["index", "k", "s", "f"])
    assert_true(indexed.column("index").dtype() == DataType.INT64)
    assert_ints(indexed, "index", [0, 1, 2, 3, 4, 5, 6])
    assert_ints(
        frame.lazy().with_row_index("row", -2).collect(),
        "row",
        [-2, -1, 0, 1, 2, 3, 4],
    )
    # A name already in use raises, even when the column is not selected.
    check_fails(frame.lazy().with_row_index("k"))
    check_fails(frame.lazy().with_row_index("k").select(["s"]))
    # Only the index selected: the input's columns can be pruned.
    var only = frame.lazy().with_row_index("i").select(["i"])
    check(only, frame.with_row_index("i").select(["i"]))


def test_empty_inputs() raises:
    var empty = source().clear()
    check(empty.lazy().tail(3), empty)
    check(empty.lazy().tail(-3), empty)
    check(empty.lazy().reverse(), empty)
    check(empty.lazy().shift(2), empty)
    check(
        empty.lazy().shift(-2, fill_value=lit(Int64(0))).select(["k"]),
        empty.select(["k"]),
    )
    check(empty.lazy().with_row_index(), empty.with_row_index())
    check(
        source().lazy().filter(col("k") > 100).with_row_index(),
        empty.with_row_index(),
    )


def test_filters_stay_above() raises:
    var frame = source()
    # A filter below the operator, then one above it.
    check(
        frame.lazy().filter(col("k") > 1).tail(2),
        frame.filter(col("k") > 1).tail(2),
    )
    var after_tail = frame.lazy().tail(3).filter(col("k").is_not_null())
    check(after_tail, frame.tail(3).filter(col("k").is_not_null()))
    # Filtering first would keep the last three non-null keys: 4, 6, 7.
    assert_ints(after_tail.collect(), "k", [6, 7])
    var indexed = frame.lazy().with_row_index().filter(col("k").is_null())
    check(indexed, frame.with_row_index().filter(col("k").is_null()))
    # Filtering first would number the null rows 0 and 1.
    assert_ints(indexed.collect(), "index", [1, 4])
    var shifted = frame.lazy().shift(1).filter(col("k").is_not_null())
    check(shifted, frame.shift(1).filter(col("k").is_not_null()))
    assert_ints(shifted.collect(), "k", [1, 3, 4, 6])
    var reversed = frame.lazy().reverse().filter(col("f") > 2.0)
    check(reversed, frame.reverse().filter(col("f") > 2.0))
    assert_ints(reversed.collect(), "k", [7, 6, NULL, 3])
    # Through a computed column above the index.
    var twice = (
        frame.lazy()
        .with_row_index()
        .with_columns((col("index") * 2).alias("twice"))
        .filter(col("twice") >= 8)
    )
    check(
        twice,
        frame.with_row_index()
        .with_columns((col("index") * 2).alias("twice"))
        .filter(col("twice") >= 8),
    )
    assert_ints(twice.collect(), "twice", [8, 10, 12])


def test_slices_stay_above() raises:
    var frame = source()
    var head_of_tail = frame.lazy().tail(4).head(2)
    check(head_of_tail, frame.tail(4).head(2))
    assert_ints(head_of_tail.collect(), "k", [4, NULL])
    var tail_of_head = frame.lazy().head(5).tail(2)
    check(tail_of_head, frame.head(5).tail(2))
    assert_ints(tail_of_head.collect(), "k", [4, NULL])
    var head_of_reverse = frame.lazy().reverse().head(2)
    check(head_of_reverse, frame.reverse().head(2))
    assert_ints(head_of_reverse.collect(), "k", [7, 6])
    check(frame.lazy().head(3).reverse(), frame.head(3).reverse())
    var sliced = frame.lazy().with_row_index().slice(2, 3)
    check(sliced, frame.with_row_index().slice(2, 3))
    assert_ints(sliced.collect(), "index", [2, 3, 4])
    check(frame.lazy().shift(2).head(3), frame.shift(2).head(3))
    check(frame.lazy().reverse().slice(-3, 2), frame.reverse().slice(-3, 2))
    # A projection between the slice and the operator.
    check(
        frame.lazy().tail(3).select(["k"]).head(1),
        frame.tail(3).select(["k"]).head(1),
    )
    # Top-k of a sort on a column without ties, then the last of those.
    check(
        frame.lazy().sort("f").head(5).tail(2),
        frame.sort("f").head(5).tail(2),
    )
    check(
        frame.lazy().sort("f").with_row_index().head(3),
        frame.sort("f").with_row_index().head(3),
    )


def test_joins_and_aggregates() raises:
    var frame = source()
    var names = DataFrame(
        [
            Series("k", Column[Int64]([7, 1, 4, 6, 3])),
            Series(
                "name", Column[String](["seven", "one", "four", "six", "three"])
            ),
        ]
    )
    check(
        frame.lazy().join(names.lazy(), on=["k"]).tail(2),
        frame.join(names, on=["k"]).tail(2),
    )
    check(
        frame.lazy().join(names.lazy(), on=["k"]).with_row_index(),
        frame.join(names, on=["k"]).with_row_index(),
    )
    check(
        frame.lazy().with_row_index().join(names.lazy().reverse(), on=["k"]),
        frame.with_row_index().join(names.reverse(), on=["k"]),
    )
    check(
        names.lazy()
        .with_row_index("rank")
        .join(frame.lazy(), on=["k"], how="left"),
        names.with_row_index("rank").join(frame, on=["k"], how="left"),
    )
    check(
        frame.lazy().shift(1).join(names.lazy(), on=["k"], how="semi"),
        frame.shift(1).join(names, on=["k"], how="semi"),
    )
    # Aggregates whose answer depends on the order the operator made.
    var many = numbers(1000)
    check(
        many.lazy()
        .reverse()
        .group_by("g", maintain_order=True)
        .agg([col("k").first().alias("first"), col("v").sum().alias("sum")]),
        many.reverse()
        .group_by("g", maintain_order=True)
        .agg([col("k").first().alias("first"), col("v").sum().alias("sum")]),
    )
    check(
        many.lazy().shift(3).select_exprs([col("k").sum().alias("total")]),
        many.shift(3).select_exprs([col("k").sum().alias("total")]),
    )
    check(
        many.lazy().filter(col("g") < 4).with_row_index().tail(5),
        many.filter(col("g") < 4).with_row_index().tail(5),
    )
    check(
        many.lazy().with_row_index().filter(col("g") == 2).head(4),
        many.with_row_index().filter(col("g") == 2).head(4),
    )
    check(
        many.lazy().concat(many.lazy().filter(col("g") == 1)).tail(3),
        many.vstack(many.filter(col("g") == 1)).tail(3),
    )


def test_explain_and_accelerator_rejection() raises:
    var frame = source()
    var plan = (
        frame.lazy()
        .with_row_index()
        .shift(2, fill_value=lit(Int64(0)))
        .reverse()
        .tail(3)
    )
    var text = plan.explain()
    assert_true(text.find("TAIL 3 [materialize]") >= 0, text)
    assert_true(text.find("REVERSE [materialize]") >= 0, text)
    assert_true(text.find("SHIFT 2") >= 0, text)
    assert_true(text.find("WITH_ROW_INDEX index from 0") >= 0, text)
    # The accelerator lowers none of them; each raises its ordinary
    # unsupported-plan error before any device work.
    for query in [
        frame.lazy().tail(2),
        frame.lazy().reverse(),
        frame.lazy().shift(1),
        frame.lazy().with_row_index(),
        frame.lazy().filter(col("k") > 1).with_row_index().select(["index"]),
    ]:
        for joined in [False, True]:
            var failed = False
            try:
                if joined:
                    _ = lower_join_dag(query, RowCapabilities())
                else:
                    _ = lower_rows(query)
            except error:
                assert_true(
                    String(error).find("unsupported") >= 0, String(error)
                )
                failed = True
            assert_true(failed, "row-order operators must not lower")
