"""As-of sortedness is a validated input contract, including zero-row results."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_empty_left_asof_still_validates_right_sortedness() raises:
    for grouped in [False, True]:
        var left = DataFrame(
            [Series("k", Column[Int64]([])), Series("g", Column[Int64]([]))]
        )
        # Hidden -999 does not break ordering. In grouped mode, the 100 in
        # another group does not either; the final 2 after 3 is the violation.
        var keys: List[Int64] = [1, -999, 100, 3, 2] if grouped else [
            1,
            -999,
            2,
            3,
            2,
        ]
        var raw = DataFrame(
            [
                Series(
                    "k", Column[Int64](keys^, [True, False, True, True, True])
                ),
                Series("g", Column[Int64]([0, 0, 1, 0, 0])),
            ]
        )
        var by: List[String] = ["g"] if grouped else List[String]()
        for layout in range(3):
            var right = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            for strategy in ["backward", "forward", "nearest"]:
                for execution in range(5):
                    var failed = False
                    try:
                        if execution == 0:
                            _ = left.join_asof(
                                right, on="k", by=by, strategy=strategy
                            )
                        else:
                            _ = (
                                left.lazy()
                                .join_asof(
                                    right.lazy(),
                                    on="k",
                                    by=by,
                                    strategy=strategy,
                                )
                                .collect(
                                    optimize=execution > 2,
                                    streaming=execution % 2 == 0,
                                    batch_size=2,
                                    engine="cpu",
                                )
                            )
                    except error:
                        assert_true(String(error).find("right column 'k'") >= 0)
                        assert_true(String(error).find("row 4") >= 0)
                        failed = True
                    assert_true(
                        failed,
                        "empty left must not bypass right sortedness validation",
                    )
                var control = left.join_asof(
                    right.head(4), on="k", by=by, strategy=strategy
                )
                assert_equal(control.height(), 0)
                var names: List[String] = ["k", "g"]
                if not grouped:
                    names.append("g_right")
                assert_equal(control.columns(), names)
