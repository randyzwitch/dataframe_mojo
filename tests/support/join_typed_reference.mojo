"""Nested-loop join row selection independent of production key encoding."""
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    ListColumn,
    Series,
    StructColumn,
)
from tests.support.coverage_compare import assert_value
from tests.support.ordered_reduction_fixtures import typed_ordered_columns


def typed_join_values() raises -> List[Series]:
    var result = typed_ordered_columns([True, True, True, True, True])
    result.append(
        Series(
            "x",
            ListColumn(
                [0, 0, 2, 3, 5, 6],
                Series(
                    "child",
                    Column[Int64](
                        [1, 999, 2, 1, 2, 9],
                        [True, False, True, True, True, True],
                    ),
                ),
            ),
        )
    )
    result.append(
        Series(
            "x",
            StructColumn(
                [
                    Series("n", Column[Int64]([0, 1, 2, 3, 4])),
                    Series(
                        "s",
                        Column[String](
                            ["", "é", "abcdefghijklmnop", "z", "hidden"],
                            [True, True, True, True, False],
                        ),
                    ),
                ]
            ),
        )
    )
    return result^


def fixture_join_ranks(dtype: DataType) -> List[Int]:
    # Reference identity is chosen independently of storage hashes/codes.
    # Float +/-0 and unsigned min=0 duplicate row4; all NaNs match.
    # Booleans have two values.
    if dtype == DataType.BOOL:
        return [0, 1, 1, 0, 1]
    if dtype.is_float() or dtype.is_unsigned():
        return [0, 1, 2, 3, 0]
    return [0, 1, 2, 3, 4]


def selected_ranks(ranks: List[Int], indices: List[Int]) -> List[Int]:
    var result = List[Int]()
    for row in indices:
        result.append(-1 if row < 0 else ranks[row])
    return result^


def nested_loop_pairs(
    left: List[Int], right: List[Int], how: String
) -> List[Tuple[Int, Int]]:
    var result = List[Tuple[Int, Int]]()
    if how == "cross":
        for i in range(len(left)):
            for j in range(len(right)):
                result.append((i, j))
        return result^
    if how == "right":
        for j in range(len(right)):
            var matched = False
            for i in range(len(left)):
                if left[i] >= 0 and left[i] == right[j]:
                    result.append((i, j))
                    matched = True
            if not matched:
                result.append((-1, j))
        return result^
    var right_seen = List[Bool](length=len(right), fill=False)
    for i in range(len(left)):
        var matched = False
        for j in range(len(right)):
            if left[i] >= 0 and left[i] == right[j]:
                matched = True
                right_seen[j] = True
                if how != "semi" and how != "anti":
                    result.append((i, j))
        if how == "semi" and matched:
            result.append((i, -1))
        if (how == "anti" or how == "left" or how == "full") and not matched:
            result.append((i, -1))
    if how == "full":
        for j in range(len(right)):
            if not right_seen[j]:
                result.append((-1, j))
    return result^


def _cell(
    actual: DataFrame, output: Int, name: String, source: Series, row: Int
) raises:
    if row < 0:
        assert_true(actual.item(output, name).is_null())
    else:
        assert_value(actual.item(output, name), source.get(row))


def assert_typed_join(
    actual: DataFrame,
    left: DataFrame,
    right: DataFrame,
    pairs: List[Tuple[Int, Int]],
    how: String,
    coalesce: Bool = True,
    dictionary_union: Bool = False,
) raises:
    var left_only = how == "semi" or how == "anti"
    var keep_right_key = how == "cross" or (not coalesce and not left_only)
    var names: List[String] = ["k", "p"]
    if keep_right_key:
        names.append("k_right")
    if not left_only:
        names.append("p_right")
    assert_equal(actual.columns(), names)
    assert_true(
        actual.height() == len(pairs),
        "typed join cardinality mismatch: "
        + left.column("k").dtype().name()
        + ", kind="
        + how
        + ", coalesce="
        + String(coalesce)
        + ", actual="
        + String(actual.height())
        + ", expected="
        + String(len(pairs)),
    )
    if dictionary_union:
        # Independently encoded categoricals may use a new shared dictionary.
        # Its ordering is not a public guarantee; decoded cells remain exact.
        assert_true(left.column("k").dtype().is_categorical())
        assert_true(right.column("k").dtype().is_categorical())
        assert_true(actual.column("k").dtype().is_categorical())
    else:
        assert_true(actual.column("k").dtype() == left.column("k").dtype())
    assert_true(actual.column("p").dtype() == left.column("p").dtype())
    if keep_right_key:
        if dictionary_union:
            assert_true(actual.column("k_right").dtype().is_categorical())
        else:
            assert_true(
                actual.column("k_right").dtype() == right.column("k").dtype()
            )
    if not left_only:
        assert_true(
            actual.column("p_right").dtype() == right.column("p").dtype()
        )
    for output in range(len(pairs)):
        var i = pairs[output][0]
        var j = pairs[output][1]
        if coalesce and (how == "right" or (how == "full" and i < 0)):
            _cell(actual, output, "k", right.column("k"), j)
        else:
            _cell(actual, output, "k", left.column("k"), i)
        _cell(actual, output, "p", left.column("p"), i)
        if keep_right_key:
            _cell(actual, output, "k_right", right.column("k"), j)
        if not left_only:
            _cell(actual, output, "p_right", right.column("p"), j)
