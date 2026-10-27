"""Join order changes preserve coalesced names and position-sensitive barriers."""
# contract-group: planner.relational
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col
from dataframe.lazy import JOIN, SELECT, PlanNode, _late_key_build


def _inputs() raises -> Tuple[DataFrame, DataFrame]:
    var small = DataFrame(
        [
            Series("a", Column[Int64]([2, 0, 1])),
            Series("label", Column[String](["two", "zero", "one"])),
        ]
    )
    var keys = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(299):
        keys.append(Int64(row * 17 % 7))
        values.append(Int64(row - 149))
        valid.append(row % 11 != 3)
    return (
        small^,
        DataFrame(
            [
                Series("b", Column[Int64](keys^, valid)),
                Series("value", Column[Int64](values^)),
            ]
        ),
    )


def _check_groups(actual: DataFrame, first: Bool, ordered: Bool) raises:
    assert_equal(actual.height(), 3)
    var result = actual.copy() if ordered else actual.sort("a")
    for position in range(3):
        var key = [2, 0, 1][position] if ordered else position
        var total = Int128(0)
        var earliest = Int64(0)
        var count = 0
        for row in range(299):
            if row * 17 % 7 == key and row % 11 != 3:
                if count == 0:
                    earliest = Int64(row - 149)
                total += Int128(row - 149)
                count += 1
        assert_equal(result.item(position, "a").int64(), Int64(key))
        assert_equal(
            result.item(position, "answer").int64(),
            earliest if first else Int64(total),
        )
        assert_equal(result.item(position, "n").int64(), Int64(count))


def test_join_rerooting_restores_coalesced_key_names_and_respects_order_sensitive_reductions() raises:
    var inputs = _inputs()
    for first in [False, True]:
        for ordered in [False, True]:
            var joined = (
                inputs[0]
                .lazy()
                .select(["a"])
                .join(inputs[1].lazy(), left_on=["a"], right_on=["b"])
            )
            var query = joined.group_by("a", maintain_order=ordered).agg(
                [
                    (
                        col("value").first() if first else col("value").sum()
                    ).alias("answer"),
                    col("value").count().alias("n"),
                ]
            )
            for streaming in [False, True]:
                var planned = query._optimized()
                var root = len(planned._plan._nodes) - 1
                assert_equal(
                    planned._probe_largest_inputs(root, streaming, 7),
                    not first and not ordered,
                )
                var actual = planned._execute(root, False, streaming, 7)
                assert_equal(actual.columns(), ["a", "answer", "n"])
                _check_groups(actual, first, ordered)
                for optimize in [False, True]:
                    _check_groups(
                        query.collect(
                            optimize=optimize,
                            streaming=streaming,
                            batch_size=7,
                            engine="cpu",
                        ),
                        first,
                        ordered,
                    )


def test_a_renamed_progression_key_is_still_looked_up_by_position() raises:
    # A key a projection renames is the same column of the same table: a
    # progression there needs no hash table, so a join builds on it (and
    # keeps building on it) rather than hashing the other side. A key a
    # projection computes is no table's column.
    var keys = List[Int64]()
    var values = List[Int64]()
    for row in range(500):
        keys.append(Int64(row + 1))
        values.append(Int64(row % 7))
    var table = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^)),
        ]
    )
    var plan = table.lazy().select_exprs(
        [
            col("k").alias("key"),
            (col("k") * 2).alias("doubled"),
            col("v"),
        ]
    )
    var top = len(plan._plan._nodes) - 1
    assert_true(plan._progression_key(top, ["key"]))
    assert_true(not plan._progression_key(top, ["doubled"]))
    assert_true(not plan._progression_key(top, ["v"]))


def test_join_rerooting_keeps_payload_sources_and_progression_lookup_plans_in_place() raises:
    var inputs = _inputs()
    var query = (
        inputs[0]
        .lazy()
        .join(inputs[1].lazy(), left_on=["a"], right_on=["b"])
        .group_by(["a", "label"])
        .agg(
            [
                col("value").sum().alias("answer"),
                col("value").count().alias("n"),
            ]
        )
    )
    var planned = query._optimized()
    assert_true(
        not planned._probe_largest_inputs(
            len(planned._plan._nodes) - 1, True, 7
        )
    )
    var actual = query.collect(engine="cpu")
    _check_groups(actual, False, False)
    var ordered = actual.sort("a")
    for row in range(3):
        assert_equal(
            ordered.item(row, "label").string(), ["zero", "one", "two"][row]
        )
    var keys = List[Int64]()
    var values = List[Int64]()
    for row in range(299):
        keys.append(Int64(row))
        values.append(Int64(row - 149))
    var progression = DataFrame(
        [
            Series("b", Column[Int64](keys^)),
            Series("value", Column[Int64](values^)),
        ]
    )
    var direct = (
        inputs[0]
        .lazy()
        .select(["a"])
        .join(progression.lazy(), left_on=["a"], right_on=["b"])
        .group_by("a")
        .agg(col("value").sum().alias("answer"))
    )
    planned = direct._optimized()
    assert_true(
        not planned._probe_largest_inputs(
            len(planned._plan._nodes) - 1, True, 7
        )
    )
    actual = direct.collect(engine="cpu").sort("a")
    assert_equal(actual.height(), 3)
    for row in range(3):
        assert_equal(actual.item(row, "a").int64(), Int64(row))
        assert_equal(actual.item(row, "answer").int64(), Int64(row - 149))


def test_late_materialized_join_string_keys_are_eligible_only_when_no_expression_reads_or_replaces_them() raises:
    var source = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 1, 3])),
            Series("value", Column[Int64]([4, 7, -1, 9])),
        ]
    )
    var build = DataFrame(
        [
            Series("k", Column[Int64]([1, 2])),
            Series(
                "label",
                Column[String](
                    ["long external group alpha", "long external group beta"]
                ),
            ),
        ]
    )
    var joined = source.lazy().join(build.lazy(), "k", how="left")
    var operations = List[PlanNode]()
    var join_index = -1
    for index in range(len(joined._plan._nodes)):
        if joined._plan._nodes[index].kind == JOIN:
            join_index = index
    assert_true(join_index >= 0)
    var operation = joined._plan._nodes[join_index].copy()
    operation.offset = 0
    operations.append(operation^)
    var builds: List[DataFrame] = [build.copy()]
    var aggregates: List[Expr] = [col("value").sum().alias("total")]
    assert_equal(
        _late_key_build(
            "label", source.columns(), builds, operations, aggregates
        ),
        0,
    )
    assert_equal(
        _late_key_build(
            "label",
            source.columns(),
            builds,
            operations,
            [col("label").first()],
        ),
        -1,
    )
    assert_equal(
        _late_key_build(
            "label", ["k", "label"], builds, operations, aggregates
        ),
        -1,
    )
    var result = (
        joined.group_by("label", maintain_order=True)
        .agg(col("value").sum().alias("total"))
        .collect(optimize=False, streaming=True, batch_size=1, engine="cpu")
    )
    assert_equal(result.height(), 3)
    for row in range(3):
        assert_equal(result.item(row, "label").is_null(), row == 2)
        if row != 2:
            assert_equal(
                result.item(row, "label").string(),
                ["long external group alpha", "long external group beta"][row],
            )
        assert_equal(result.item(row, "total").int64(), Int64([3, 7, 9][row]))


def _join_below(plan: LazyFrame) raises -> Tuple[Int, List[Int]]:
    """The topmost join of `plan` and every node's parent."""
    var root = len(plan._plan._nodes) - 1
    var parents = plan._live_parents(root)
    var top = -1
    for i in range(len(plan._plan._nodes)):
        if parents[i] != -2 and plan._plan._nodes[i].kind == JOIN:
            if top < 0 or i > top:
                top = i
    return (top, parents^)


def test_a_sort_leaves_the_joins_below_it_free_to_reorder_unless_ties_keep_order() raises:
    # Equal sort keys owe no order (Polars and DuckDB promise none), so the
    # rows a join below a sort makes may arrive in any order; with
    # `maintain_order` their input order shows in the ties.
    var inputs = _inputs()
    var joined = (
        inputs[0].lazy().join(inputs[1].lazy(), left_on=["a"], right_on=["b"])
    )
    var free = joined.sort("label")
    var found = _join_below(free)
    assert_true(free._order_free_above(found[0], found[1]))
    var kept = joined.sort("label", maintain_order=True)
    found = _join_below(kept)
    assert_true(not kept._order_free_above(found[0], found[1]))
    # Aggregates over partitions read no position; a running sum does.
    var partitioned = joined.with_columns(
        [col("value").mean().over("a").alias("m")]
    ).sort("label")
    found = _join_below(partitioned)
    assert_true(partitioned._order_free_above(found[0], found[1]))
    var running = joined.with_columns([col("value").cum_sum().alias("c")]).sort(
        "label"
    )
    found = _join_below(running)
    assert_true(not running._order_free_above(found[0], found[1]))


def test_maintain_order_keeps_input_order_among_equal_sort_keys() raises:
    var frame = DataFrame(
        [
            Series("k", Column[Int64]([2, 1, 2, 1, 2, 1])),
            Series("row", Column[Int64]([0, 1, 2, 3, 4, 5])),
        ]
    )
    var expected: List[Int64] = [1, 3, 5, 0, 2, 4]
    var eager = frame.sort("k", maintain_order=True)
    var lazy = frame.lazy().sort("k", maintain_order=True).collect()
    for row in range(6):
        assert_equal(eager.item(row, "row").int64(), expected[row])
        assert_equal(lazy.item(row, "row").int64(), expected[row])
    # Without it the same rows come out, ordered by key.
    var free = frame.lazy().sort("k").collect()
    var seen = List[Bool](length=6, fill=False)
    for row in range(6):
        assert_equal(free.item(row, "k").int64(), Int64(1 if row < 3 else 2))
        seen[Int(free.item(row, "row").int64())] = True
    for row in range(6):
        assert_true(seen[row])
