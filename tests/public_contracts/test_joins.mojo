"""Full join rows and schema against a nested-loop oracle; duplicates are pairs."""
# contract-group: table.join
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, Series
from tests.support.reference_model import (
    generated_model,
    ReferenceRow,
    ReferenceTable,
)
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import assert_join_model
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def test_all_equality_join_kinds_match_nested_loops() raises:
    var shapes: List[Tuple[Int, Int]] = [
        (0, 0),
        (0, 3),
        (3, 0),
        (1, 1),
        (9, 7),
        (35, 33),
    ]
    var key_sets: List[List[String]] = [["i"], ["s"], ["f"], ["b"], ["s", "i"]]
    for shape in shapes:
        for validity in range(3):
            var left_model = generated_model(shape[0], 3, validity)
            var right_model = generated_model(shape[1], 5, validity)
            for layout in range(3):
                var left = frame_from_model(left_model, 7 if layout == 1 else 0)
                var right = frame_from_model(
                    right_model, 1 if layout == 1 else 0
                )
                if layout == 2:
                    left = chunk_layout(left)
                    right = chunk_layout(right)
                assert_join_model(
                    left.join(right, how="cross"),
                    left_model,
                    right_model,
                    [],
                    "cross",
                )
                for keys in key_sets:
                    for how in [
                        "inner",
                        "left",
                        "right",
                        "full",
                        "semi",
                        "anti",
                    ]:
                        var result = left.join(right, keys, how=how)
                        assert_join_model(
                            result, left_model, right_model, keys, how
                        )
                        if how == "full":
                            assert_join_model(
                                left.join(right, keys, how=how, coalesce=False),
                                left_model,
                                right_model,
                                keys,
                                how,
                                coalesce=False,
                            )


def test_join_float_specials_and_validation() raises:
    var rows = List[ReferenceRow]()
    var values: List[Float64] = [
        0.0,
        -0.0,
        Float64(0) / Float64(0),
        Float64(0) / Float64(0),
        9.0,
    ]
    for index in range(len(values)):
        rows.append(
            ReferenceRow(
                Int64(index),
                values[index],
                String(index),
                False,
                True,
                index != 4,
                True,
                True,
            )
        )
    var model = ReferenceTable(["i", "f", "s", "b"], rows^)
    var frame = frame_from_model(model, 7)
    for how in ["inner", "left", "right", "full", "semi", "anti"]:
        assert_join_model(
            frame.join(frame, "f", how=how), model, model, ["f"], how
        )
    var invalid_keys: List[List[String]] = [[], ["missing"], ["i", "i"]]
    for keys in invalid_keys:
        with assert_raises():
            _ = frame.join(frame, keys)
    with assert_raises():
        _ = frame.join(frame, "i", how="unknown")
    with assert_raises():
        _ = frame.join(frame, left_on=["i"], right_on=["s"])
    with assert_raises():
        _ = frame.join(frame, left_on=["i", "s"], right_on=["i"])
    with assert_raises():
        _ = frame.join(frame, "i", suffix="")
    var right = frame.rename({"i": "key"})
    var different = frame.join(right, left_on=["i"], right_on=["key"])
    var same = frame.join(frame, "i")
    assert_equal(different.columns(), same.columns())
    # Full expected data comes from the oracle, rather than the sibling join.
    assert_join_model(different, model, model, ["i"], "inner")


def test_a_coalesced_right_join_keeps_the_right_key_names() raises:
    # As Polars (a left join with the sides swapped): the left's non-key
    # columns, then every right column, keys included under their own
    # names; a right name the left's other columns hold takes the suffix.
    var left = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 3])),
            Series("bk", Column[Int64]([7, 8, 9])),
            Series("x", Column[Int64]([10, 20, 30])),
        ]
    )
    var right = DataFrame(
        [
            Series("x", Column[String](["a", "b", "c"])),
            Series("bk", Column[Int64]([4, 3, 2])),
            Series("k", Column[Int64]([0, 0, 0])),
        ]
    )
    var names: List[String] = ["bk", "x", "x_right", "bk_right", "k"]
    for lazy in [False, True]:
        var out = (
            left.lazy()
            .join(right.lazy(), left_on=["k"], right_on=["bk"], how="right")
            .collect() if lazy else left.join(
                right, left_on=["k"], right_on=["bk"], how="right"
            )
        )
        assert_equal(out.columns(), names)
        out = out.sort("bk_right")
        assert_equal(out.item(0, "bk_right").int64(), 2)
        assert_equal(out.item(0, "x").int64(), 20)
        assert_equal(out.item(0, "x_right").string(), "c")
        assert_true(out.item(2, "x").is_null())
        assert_equal(out.item(2, "bk_right").int64(), 4)
        # Selecting the right key reads it from the right input.
        var picked = (
            left.lazy()
            .join(right.lazy(), left_on=["k"], right_on=["bk"], how="right")
            .select(["bk_right", "k"])
            .collect()
        )
        assert_equal(picked.columns(), List[String](["bk_right", "k"]))
        assert_equal(picked.height(), 3)
        # The lazy join runs as the right join keeping both keys, which
        # streams, with the left keys dropped.
        var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
        var observed = StrategyTrace()
        var traced = (
            left.lazy()
            .join(right.lazy(), left_on=["k"], right_on=["bk"], how="right")
            .collect()
        )
        assert_true(
            "dataframe-path: lazy.lower_right_join\n" in observed.text(),
            "the right join was not lowered",
        )
        _ = observed^
        _ = trace^
        assert_equal(traced.columns(), names)
        # Keys named alike come out once, after the left's other columns.
        var same = left.drop(["bk"]).join(
            right.drop(["k"]).rename({"bk": "k"}), on="k", how="right"
        )
        assert_equal(same.columns(), List[String](["x", "x_right", "k"]))
