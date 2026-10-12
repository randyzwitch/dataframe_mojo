"""Branch rewrites preserve Boolean implications and each union input's meaning."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col


def test_implied_join_filters_preserve_or_branches_and_nullable_side_values() raises:
    var facts = DataFrame(
        [
            Series("id", Column[Int64]([Int64(row) for row in range(31)])),
            Series("a", Column[Int64]([Int64(row % 3) for row in range(31)])),
            Series("b", Column[Int64]([Int64(row % 4) for row in range(31)])),
        ]
    )
    var left = DataFrame(
        [
            Series("a", Column[Int64]([0, 1, 2])),
            Series(
                "label_a",
                Column[String](["a0", "hidden", "a2"], [True, False, True]),
            ),
        ]
    )
    var right = DataFrame(
        [
            Series("b", Column[Int64]([0, 1, 2, 3])),
            Series(
                "label_b",
                Column[String](
                    ["b0", "b1", "b2", "hidden"], [True, True, True, False]
                ),
            ),
        ]
    )
    for nullable_branch in [False, True]:
        var predicate = (
            (col("label_a") == "a0") & (col("label_b") == "b1")
        ) | ((col("label_a") == "a2") & (col("label_b") == "b0"))
        if nullable_branch:
            predicate = (
                predicate
                | (col("label_a").is_null() & (col("id") < 12))
                | ((col("id") > 20) & (col("label_b") == "b2"))
            )
        var query = (
            facts.lazy()
            .join(left.lazy(), "a")
            .join(right.lazy(), "b")
            .filter(predicate)
        )
        if not nullable_branch:
            var text = query.explain(engine="cpu", streaming=False)
            var filters = 0
            var at = text.find("FILTER")
            while at >= 0:
                filters += 1
                at = text.find("FILTER", at + 1)
            assert_true(
                filters >= 3, "both dimension implications must be exercised"
            )
        for optimize in [False, True]:
            for streaming in [False, True]:
                var actual = query.collect(
                    optimize=optimize, streaming=streaming, batch_size=7
                ).sort("id")
                assert_equal(
                    actual.columns(),
                    [String("id"), "a", "b", "label_a", "label_b"],
                )
                var position = 0
                for row in range(31):
                    var a = row % 3
                    var b = row % 4
                    var keep = a == 0 and b == 1 or a == 2 and b == 0
                    if nullable_branch:
                        keep = (
                            keep or a == 1 and row < 12 or row > 20 and b == 2
                        )
                    if not keep:
                        continue
                    assert_equal(
                        actual.item(position, "id").int64(), Int64(row)
                    )
                    assert_equal(actual.item(position, "a").int64(), Int64(a))
                    assert_equal(actual.item(position, "b").int64(), Int64(b))
                    assert_equal(
                        actual.item(position, "label_a").is_null(), a == 1
                    )
                    assert_equal(
                        actual.item(position, "label_b").is_null(), b == 3
                    )
                    if a != 1:
                        assert_equal(
                            actual.item(position, "label_a").string(),
                            "a" + String(a),
                        )
                    if b != 3:
                        assert_equal(
                            actual.item(position, "label_b").string(),
                            "b" + String(b),
                        )
                    position += 1
                assert_equal(actual.height(), position)


def test_union_inputs_keep_their_own_replacements_filters_and_column_order_before_outer_slice() raises:
    var left = DataFrame(
        [
            Series("id", Column[Int64]([Int64(row) for row in range(13)])),
            Series("v", Column[Int64]([Int64(row * 3) for row in range(13)])),
        ]
    )
    var right = DataFrame(
        [
            Series(
                "v", Column[Int64]([Int64(row * 3) for row in range(10, 29)])
            ),
            Series("id", Column[Int64]([Int64(row) for row in range(10, 29)])),
        ]
    )
    var ids = List[Int64]()
    var values = List[Int64]()
    for side in range(2):
        for row in range(0 if side == 0 else 10, 13 if side == 0 else 29):
            var keep = row % 3 != 1 if side == 0 else row % 4 == 2
            var value = row * (3 if side == 0 else 6)
            if keep and value > 20:
                ids.append(Int64(row))
                values.append(Int64(value))
    var query = (
        left.lazy()
        .filter(col("id") % 3 != 1)
        .concat(
            right.lazy()
            .with_columns((col("v") * 2).alias("v"))
            .filter(col("id") % 4 == 2)
        )
        .filter(col("v") > 20)
        .select(["v", "id"])
        .slice(1, 13)
    )
    for optimize in [False, True]:
        for streaming in [False, True]:
            for batch in [1, 7, 64]:
                var actual = query.collect(
                    optimize=optimize, streaming=streaming, batch_size=batch
                )
                assert_equal(actual.columns(), [String("v"), "id"])
                assert_equal(actual.height(), min(13, max(0, len(ids) - 1)))
                for row in range(actual.height()):
                    assert_equal(actual.item(row, "id").int64(), ids[row + 1])
                    assert_equal(actual.item(row, "v").int64(), values[row + 1])
    with assert_raises():
        _ = (
            left.lazy()
            .concat(
                right.lazy().select_exprs([col("v").alias("other"), col("id")])
            )
            .collect_schema()
        )
