"""Joins on key expressions: keys computed per side, inputs' columns kept."""
# contract-group: table.join
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, Expr, Series, StringColumn, col
from tests.support.group_strategy_reference import ScopedEnvironment


def _emails() raises -> Tuple[DataFrame, DataFrame]:
    var a = DataFrame(
        [
            Series("email", StringColumn(["A@x", "b@x", "c@x", "e@x"])),
            Series("x", Column[Int64]([1, 2, 3, 4])),
        ]
    )
    var b = DataFrame(
        [
            Series("email", StringColumn(["a@x", "B@X", "d@x"])),
            Series("y", Column[Int64]([10, 20, 30])),
        ]
    )
    return (a^, b^)


def _expr_join(
    a: DataFrame, b: DataFrame, how: String, lazy: Bool
) raises -> DataFrame:
    var lkey: List[Expr] = [col("email").str().to_lowercase()]
    var rkey: List[Expr] = [col("email").str().to_lowercase()]
    if lazy:
        return (
            a.lazy()
            .join(b.lazy(), left_on=lkey, right_on=rkey, how=how)
            .collect()
        )
    return a.join(b, left_on=lkey, right_on=rkey, how=how)


def _expected_height(how: String) -> Int:
    if how == "left":
        return 4
    if how == "right":
        return 3
    if how == "full":
        return 5
    return 2


def test_computed_keys_match_and_keep_both_sides_columns() raises:
    """Rows pair as joining on the computed keys would, and the output is
    the inputs' columns: both email columns, no computed key."""
    var frames = _emails()
    ref a = frames[0]
    ref b = frames[1]
    for threads in ["1", "8"]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
        for lazy in [False, True]:
            for how in ["inner", "left", "right", "full", "semi", "anti"]:
                var out = _expr_join(a, b, how, lazy)
                assert_equal(out.height(), _expected_height(how), how)
                if how == "semi" or how == "anti":
                    assert_equal(out.columns(), a.columns(), how)
                    continue
                assert_equal(
                    out.columns(),
                    ["email", "x", "email_right", "y"],
                    how,
                )
            var inner = _expr_join(a, b, "inner", lazy).sort(["x"])
            assert_equal(inner.column("email_right").get(0).string(), "a@x")
            assert_equal(inner.column("email_right").get(1).string(), "B@X")
            assert_equal(inner.column("y").get(1).int64(), 20)
        _ = workers^


def test_mixed_and_plain_expression_keys() raises:
    var c = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 3])),
            Series("x", Column[Int64]([1, 2, 3])),
        ]
    )
    var d = DataFrame(
        [
            Series("k", Column[Int64]([2, 4, 6])),
            Series("y", Column[Int64]([1, 2, 3])),
        ]
    )
    for lazy in [False, True]:
        # A computed left key against a plain right column.
        var doubled = (
            c.lazy()
            .join(d.lazy(), left_on=[col("k") * 2], right_on=[col("k")])
            .collect() if lazy else c.join(
                d, left_on=[col("k") * 2], right_on=[col("k")]
            )
        )
        assert_equal(doubled.columns(), ["k", "x", "k_right", "y"])
        assert_equal(doubled.height(), 3)
        # Keys that are all plain columns join as key names do, coalesced.
        var plain = (
            c.lazy()
            .join(d.lazy(), left_on=[col("k")], right_on=[col("k")], how="left")
            .collect() if lazy else c.join(
                d, left_on=[col("k")], right_on=[col("k")], how="left"
            )
        )
        var named = c.join(d, on="k", how="left")
        assert_equal(plain.columns(), named.columns())
        assert_true(plain.sort(["x"]).equals(named.sort(["x"])))


def test_expression_keys_with_validate_filters_and_taken_names() raises:
    """validate checks the computed keys; a filter above reaches the right
    rows; an input column with a hidden key's name does not collide."""
    var a = DataFrame(
        [
            Series("__join_left_key_0", Column[Int64]([1, 2, 3, 4])),
            Series("s", StringColumn(["A", "a", "B", "c"])),
        ]
    )
    var b = DataFrame(
        [
            Series("s", StringColumn(["a", "b"])),
            Series("y", Column[Int64]([1, 2])),
        ]
    )
    var lkey: List[Expr] = [col("s").str().to_lowercase()]
    var rkey: List[Expr] = [col("s")]
    for lazy in [False, True]:
        var out = (
            a.lazy()
            .join(b.lazy(), left_on=lkey, right_on=rkey, validate="m:1")
            .collect() if lazy else a.join(
                b, left_on=lkey, right_on=rkey, validate="m:1"
            )
        )
        assert_equal(out.columns(), ["__join_left_key_0", "s", "s_right", "y"])
        assert_equal(out.height(), 3)
        with assert_raises(contains="1:m validation: the left input has key"):
            if lazy:
                _ = (
                    a.lazy()
                    .join(b.lazy(), left_on=lkey, right_on=rkey, validate="1:m")
                    .collect()
                )
            else:
                _ = a.join(b, left_on=lkey, right_on=rkey, validate="1:m")
    var filtered = (
        a.lazy()
        .join(b.lazy(), left_on=lkey, right_on=rkey)
        .filter(col("y") == 2)
        .collect()
    )
    assert_equal(filtered.height(), 1)
    assert_equal(filtered.column("__join_left_key_0").get(0).int64(), 3)
    with assert_raises(contains="same nonzero number"):
        _ = a.join(b, left_on=lkey, right_on=[col("s"), col("y")])
