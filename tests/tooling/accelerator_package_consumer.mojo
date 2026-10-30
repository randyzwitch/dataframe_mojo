"""Provider-neutral package checks: identity, CPU cells, forced GPU runs."""
from std.sys import argv
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, col, lit
from dataframe import DataType
from dataframe._build_identity import CPU_BUILD_ID


def assert_literal_cells(frame: DataFrame) raises:
    assert_equal(frame.columns(), ["y"])
    assert_true(frame.column("y").dtype() == DataType.FLOAT32)
    assert_equal(frame.height(), 4)
    for row in range(4):
        assert_equal(frame.item(row, "y").is_null(), row == 2)
        if row != 2:
            assert_equal(
                frame.item(row, "y").float32(),
                Float32(6 if row == 0 else 4 if row == 1 else 0),
            )


def main() raises:
    var args = argv()
    var mode = String(args[1])
    if mode == "identity":
        print(CPU_BUILD_ID)
        return
    var frame = DataFrame(
        [Series("x", Column[Float32]([2, 1, 2, -1], [True, True, False, True]))]
    )
    var plan = frame.lazy().select(
        ((col("x") + lit(Float32(1))) * lit(Float32(2))).alias("y")
    )
    var expected = plan.collect(engine="cpu")
    assert_literal_cells(expected)
    assert_true(plan.collect(engine="auto").equals(expected))
    var measured = plan.profile(engine="auto")
    assert_literal_cells(measured[0])
    assert_true(measured[0].equals(expected))
    assert_true("selection_reason" not in measured[1].columns())
    assert_true(plan.explain(engine="auto").startswith("ENGINE auto:"))
    if mode == "gpu":
        # The installed provider runs it when forced; results match the CPU.
        assert_true(plan.collect(engine="accel").equals(expected))
        var forced = plan.profile(engine="accel")
        assert_literal_cells(forced[0])
    else:
        assert_true(measured[1].item(0, "executor").string() != "metal")
        assert_true(measured[1].item(0, "executor").string() != "nvidia")
    assert_true(
        frame.equals(
            DataFrame(
                [
                    Series(
                        "x",
                        Column[Float32](
                            [2, 1, 2, -1], [True, True, False, True]
                        ),
                    )
                ]
            )
        )
    )
    print("Shared packaged backend checks passed: " + mode)
