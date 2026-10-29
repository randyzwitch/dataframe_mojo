"""Unsorted as-of inputs give the presorted join's matches in left row order."""
# contract-group: table.join
from std.collections import Optional
from std.testing import assert_equal, assert_true
from dataframe import AnyValue, Column, DataFrame, Series
from tests.support.coverage_compare import assert_frame


def _left() raises -> DataFrame:
    """Keys out of order within each group, with duplicates and nulls."""
    var k: List[Int64] = [5, 1, 9, 0, 3, 0, 7, 2, 8, 6, 0, 3]
    var valid: List[Bool] = [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
    ]
    return DataFrame(
        [
            Series("k", Column[Int64](k^, valid)),
            Series(
                "g",
                Column[String](
                    ["a", "b", "a", "b", "a", "a", "b", "a", "b", "b", "a", "b"]
                ),
            ),
            Series("li", Column[Int64]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])),
        ]
    )


def _right() raises -> DataFrame:
    var k: List[Int64] = [4, 2, 2, 8, 0, 0, 9, 5, 0, 7, 3, 6, 2]
    var valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    ]
    return DataFrame(
        [
            Series("k", Column[Int64](k^, valid)),
            Series(
                "g",
                Column[String](
                    [
                        "b",
                        "a",
                        "b",
                        "a",
                        "a",
                        "b",
                        "a",
                        "b",
                        "b",
                        "a",
                        "a",
                        "b",
                        "a",
                    ]
                ),
            ),
            Series(
                "ri",
                Column[Int64]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            ),
        ]
    )


def _presorted(frame: DataFrame, by: List[String]) raises -> DataFrame:
    """A stable sort by the groups and then the key, nulls last."""
    var keys = by.copy()
    keys.append("k")
    return frame.sort(keys, nulls_last=True, maintain_order=True)


def _in_order_of(
    result: DataFrame, frame: DataFrame, index: String
) raises -> DataFrame:
    """`result` rows reordered so `index` follows its order in `frame`."""
    var position = List[Int](length=result.height(), fill=-1)
    for row in range(result.height()):
        position[Int(result.item(row, index).to_physical())] = row
    var order = List[Int]()
    for row in range(frame.height()):
        order.append(position[Int(frame.item(row, index).to_physical())])
    return result.take(order)


def _expected(
    left: DataFrame,
    right: DataFrame,
    by: List[String],
    strategy: String,
    exact: Bool,
    tolerance: Optional[AnyValue],
) raises -> DataFrame:
    """The join of presorted inputs, put back in `left`'s row order."""
    var joined = _presorted(left, by).join_asof(
        _presorted(right, by),
        on="k",
        by=by,
        strategy=strategy,
        allow_exact_matches=exact,
        tolerance=tolerance,
    )
    return _in_order_of(joined, left, "li")


def _check_all(left: DataFrame, right: DataFrame) raises:
    """Every strategy, exact setting, tolerance and grouping, eager and lazy."""
    for grouped in [False, True]:
        var by: List[String] = ["g"] if grouped else List[String]()
        for strategy in ["backward", "forward", "nearest"]:
            for exact in [True, False]:
                for limited in [False, True]:
                    var tolerance = Optional[AnyValue]()
                    if limited:
                        tolerance = AnyValue(Int64(2))
                    var expected = _expected(
                        left, right, by, strategy, exact, tolerance
                    )
                    var actual = left.join_asof(
                        right,
                        on="k",
                        by=by,
                        strategy=strategy,
                        allow_exact_matches=exact,
                        tolerance=tolerance,
                    )
                    assert_frame(actual, expected)
                    var plan = left.lazy().join_asof(
                        right.lazy(),
                        on="k",
                        by=by,
                        strategy=strategy,
                        allow_exact_matches=exact,
                        tolerance=tolerance,
                    )
                    for streaming in [False, True]:
                        assert_frame(
                            plan.collect(
                                streaming=streaming,
                                batch_size=3,
                                engine="cpu",
                            ),
                            expected,
                        )


def test_asof_unsorted_left_only() raises:
    var left = _left()
    var right = _right()
    _check_all(left, _presorted(right, ["g"]))
    _check_all(left, _presorted(right, List[String]()))


def test_asof_unsorted_right_only() raises:
    var left = _left()
    var right = _right()
    _check_all(_presorted(left, ["g"]), right)
    _check_all(_presorted(left, List[String]()), right)


def test_asof_both_unsorted() raises:
    _check_all(_left(), _right())
    # Reversed inputs order every group descending.
    var left = _left()
    var right = _right()
    var back = List[Int]()
    for row in range(left.height() - 1, -1, -1):
        back.append(row)
    var right_back = List[Int]()
    for row in range(right.height() - 1, -1, -1):
        right_back.append(row)
    _check_all(left.take(back), right.take(right_back))


def test_asof_unsorted_literal_matches() raises:
    # Left keys 3, 1, 2 against right keys 2, null, 0 (no groups). Backward:
    # 3 -> right row 0 (key 2), 1 -> right row 2 (key 0), 2 -> right row 0.
    # Forward: 3 -> none, 1 -> row 0, 2 -> row 0. Output keeps left order.
    var left = DataFrame([Series("k", Column[Int64]([3, 1, 2]))])
    var rk: List[Int64] = [2, 0, 0]
    var rv: List[Bool] = [True, False, True]
    var right = DataFrame(
        [
            Series("k", Column[Int64](rk^, rv)),
            Series("q", Column[Int64]([10, 11, 12])),
        ]
    )
    var backward = left.join_asof(right, on="k")
    var keys: List[Int64] = [3, 1, 2]
    for row in range(3):
        assert_equal(backward.item(row, "k").to_physical(), keys[row])
    assert_true(backward.item(0, "q").to_physical() == 10)
    assert_true(backward.item(1, "q").to_physical() == 12)
    assert_true(backward.item(2, "q").to_physical() == 10)
    var forward = left.join_asof(right, on="k", strategy="forward")
    assert_true(forward.item(0, "q").is_null())
    assert_true(forward.item(1, "q").to_physical() == 10)
    assert_true(forward.item(2, "q").to_physical() == 10)


def test_asof_unsorted_float_keys_place_nan_last() raises:
    var nan = Float64("nan")
    var left = DataFrame(
        [
            Series("k", Column[Float64]([2.5, nan, 0.5, 1.5])),
            Series("li", Column[Int64]([0, 1, 2, 3])),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Float64]([nan, 1.0, 3.0, 0.0])),
            Series("ri", Column[Int64]([0, 1, 2, 3])),
        ]
    )
    for strategy in ["backward", "forward", "nearest"]:
        var expected = _expected(
            left,
            right,
            List[String](),
            strategy,
            True,
            Optional[AnyValue](),
        )
        assert_frame(left.join_asof(right, on="k", strategy=strategy), expected)
