"""Quadratic as-of selection by independent ordering and exact distances.

Nearest reference requires nonnull finite right keys; null-position scan cases
are independent literal fixtures verified against pinned Polars 1.44.2.
"""
from std.math import abs, isnan
from std.testing import assert_equal, assert_true
from dataframe import DataFrame
from tests.support.coverage_compare import assert_value


def _less[D: DType](a: Scalar[D], b: Scalar[D]) -> Bool:
    comptime if D.is_floating_point():
        return a < b or (not isnan(a) and isnan(b))
    else:
        return a < b


def reference_asof[
    D: DType
](
    left: List[Scalar[D]],
    right: List[Scalar[D]],
    lv: List[Bool],
    rv: List[Bool],
    lg: List[Int],
    rg: List[Int],
    strategy: String,
    exact: Bool,
    limited: Bool = False,
    integer_bound: Int128 = 0,
    float_bound: Float64 = 0,
) -> List[Int]:
    var result = List[Int]()
    for i in range(len(left)):
        var best = -1
        if lv[i] and lg[i] >= 0:
            for j in range(len(right)):
                if not rv[j] or rg[j] != lg[i] or rg[j] < 0:
                    continue
                var less = _less[D](right[j], left[i])
                var greater = _less[D](left[i], right[j])
                var equal = not less and not greater
                if equal and not exact:
                    continue
                if (strategy == "backward" and greater) or (
                    strategy == "forward" and less
                ):
                    continue
                if limited:
                    comptime if D.is_floating_point():
                        if not (
                            abs(Float64(left[i]) - Float64(right[j]))
                            <= float_bound
                        ):
                            continue
                    else:
                        if (
                            abs(Int128(left[i]) - Int128(right[j]))
                            > integer_bound
                        ):
                            continue
                if best < 0:
                    best = j
                    continue
                if strategy == "backward":
                    if not _less[D](right[j], right[best]):
                        best = j
                elif strategy == "forward":
                    if _less[D](right[j], right[best]):
                        best = j
                else:
                    var closer: Bool
                    var tied: Bool
                    comptime if D.is_floating_point():
                        var candidate = abs(
                            Float64(right[j]) - Float64(left[i])
                        )
                        var previous = abs(
                            Float64(right[best]) - Float64(left[i])
                        )
                        closer = candidate < previous
                        tied = candidate == previous
                    else:
                        var candidate = abs(Int128(right[j]) - Int128(left[i]))
                        var previous = abs(
                            Int128(right[best]) - Int128(left[i])
                        )
                        closer = candidate < previous
                        tied = candidate == previous
                    if closer or (tied and not _less[D](right[j], right[best])):
                        best = j
        result.append(best)
    return result^


def assert_asof_rows(
    actual: DataFrame,
    left: DataFrame,
    right: DataFrame,
    selected: List[Int],
    names: List[String],
    right_names: List[String],
) raises:
    assert_equal(actual.columns(), names)
    assert_equal(actual.height(), left.height())
    assert_equal(len(selected), left.height())
    for name in left.columns():
        assert_true(actual.column(name).dtype() == left.column(name).dtype())
        for row in range(left.height()):
            assert_value(actual.item(row, name), left.item(row, name))
    for index in range(len(right_names)):
        var name = right_names[index]
        var output_name = names[len(left.columns()) + index]
        assert_true(
            actual.column(output_name).dtype() == right.column(name).dtype()
        )
        for row in range(left.height()):
            if selected[row] < 0:
                assert_true(actual.item(row, output_name).is_null())
            else:
                assert_value(
                    actual.item(row, output_name),
                    right.item(selected[row], name),
                )
