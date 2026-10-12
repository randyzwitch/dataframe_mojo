"""As-of matching is a barrier to later filtering, projection and row limits."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, col
from tests.support.asof_reference import reference_asof
from tests.support.coverage_generators import offset_layout, chunk_layout


def _assert_rows(
    actual: DataFrame,
    rows: List[Int],
    selected: List[Int],
    payloads: List[Int64],
) raises:
    assert_equal(actual.columns(), ["id", "value"])
    assert_equal(actual.dtypes(), [DataType.INT64, DataType.INT64])
    assert_equal(actual.height(), len(rows))
    for position in range(len(rows)):
        var row = rows[position]
        assert_equal(actual.item(position, "id").int64(), Int64(row))
        assert_equal(
            actual.item(position, "value").int64(), payloads[selected[row]]
        )


def test_asof_post_match_filter_projection_and_limits_preserve_independent_pairs_and_profile() raises:
    var left_keys: List[Int64] = [-3, 0, 2, 2, 7, 10]
    var right_keys: List[Int64] = [-2, 0, 2, 2, 5, 9, 11]
    var payloads: List[Int64] = [4, 12, 3, 20, 1, 30, 2]
    for empty_left in [False, True]:
        var keys = List[Int64]() if empty_left else left_keys.copy()
        for mode in range(3):
            var lv = List[Bool](length=len(keys), fill=mode != 2)
            var rv = List[Bool](length=len(right_keys), fill=True)
            if mode == 1 and not empty_left:
                lv[1] = False
                lv[4] = False
            var ids = List[Int64]()
            for row in range(len(keys)):
                ids.append(Int64(row))
            var left = DataFrame(
                [
                    Series("k", Column[Int64](keys.copy(), lv.copy())),
                    Series("id", Column[Int64](ids^)),
                ]
            )
            var right = DataFrame(
                [
                    Series("k", Column[Int64](right_keys.copy())),
                    Series("value", Column[Int64](payloads.copy())),
                ]
            )
            for layout in range(3):
                var l = (
                    left.copy() if layout
                    == 0 else offset_layout(left) if layout
                    == 1 else chunk_layout(left)
                )
                var r = (
                    right.copy() if layout
                    == 0 else offset_layout(right) if layout
                    == 1 else chunk_layout(right)
                )
                for strategy in ["backward", "forward", "nearest"]:
                    for exact in [False, True]:
                        var selected = reference_asof[DType.int64](
                            keys,
                            right_keys,
                            lv,
                            rv,
                            List[Int](length=len(keys), fill=0),
                            List[Int](length=len(right_keys), fill=0),
                            strategy,
                            exact,
                        )
                        var survivors = List[Int]()
                        for row in range(len(selected)):
                            if (
                                selected[row] >= 0
                                and payloads[selected[row]] > 10
                            ):
                                survivors.append(row)
                        for limit in [0, 1, 3, 9]:
                            var rows = List[Int]()
                            for position in range(min(limit, len(survivors))):
                                rows.append(survivors[position])
                            var plan = (
                                l.lazy()
                                .join_asof(
                                    r.lazy(),
                                    on="k",
                                    strategy=strategy,
                                    allow_exact_matches=exact,
                                )
                                .filter(col("value") > 10)
                                .select(["id", "value"])
                                .head(limit)
                            )
                            assert_true("ASOF" in plan.explain())
                            for optimize in [False, True]:
                                for streaming in [False, True]:
                                    _assert_rows(
                                        plan.collect(
                                            optimize=optimize,
                                            streaming=streaming,
                                            batch_size=1,
                                            engine="cpu",
                                        ),
                                        rows,
                                        selected,
                                        payloads,
                                    )
                                    var observed = plan.profile(
                                        optimize=optimize,
                                        streaming=streaming,
                                        batch_size=7,
                                        engine="cpu",
                                    )
                                    _assert_rows(
                                        observed[0], rows, selected, payloads
                                    )
                                    var saw_asof = False
                                    for row in range(observed[1].height()):
                                        saw_asof = (
                                            saw_asof
                                            or "ASOF"
                                            in observed[1]
                                            .item(row, "operator")
                                            .string()
                                        )
                                    assert_true(saw_asof)
