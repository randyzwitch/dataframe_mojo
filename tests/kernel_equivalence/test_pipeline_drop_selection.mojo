"""Dropping columns preserves mixed source/compact pipeline coordinates."""
# contract-group: table.join
# contract-group: kernel.row_ops
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, StringColumn, col, lit
from dataframe.pipeline import Morsel


def source(n: Int, offset: Int) raises -> DataFrame:
    var x = List[Int64]()
    var y = List[Int64]()
    var y_valid = List[Bool]()
    var text = List[String]()
    var text_valid = List[Bool]()
    for at in range(offset + n):
        var row = at - offset
        x.append(Int64(row))
        y.append(Int64(100 + row))
        y_valid.append(row >= 0 and row % 7 != 2)
        text.append("λ-" + String(row) + "\x00tail")
        text_valid.append(row >= 0 and row % 5 != 1)
    return DataFrame(
        [
            Series("x", Column[Int64](x^)),
            Series("y", Column[Int64](y^, y_valid)),
            Series("s", StringColumn(text, text_valid)),
        ]
    ).slice(offset, n)


def test_drop_keeps_deferred_mixed_columns_and_empty_schema_row_counts() raises:
    for n in [0, 1, 9, 65]:
        for offset in [0, 3]:
            for keep in [0, 1, 3, 4]:
                for drop_all in [False, True]:
                    var input = source(n, offset)
                    var morsel = Morsel(input^, 0)
                    morsel.filter(
                        col("x") % lit(Int64(4)) < lit(Int64(keep)), 8192
                    )
                    morsel.gather(["y"])
                    var expected = List[Int]()
                    for row in range(n):
                        if row % 4 < keep:
                            expected.append(row)
                    var dropped: List[String] = [
                        "x",
                        "y",
                        "s",
                    ] if drop_all else ["x"]
                    morsel.drop(dropped)
                    assert_equal(morsel.height(), len(expected))
                    assert_equal(morsel.frame.height(), n)
                    if not drop_all and len(expected) < n:
                        # Dropping a name does not gather unrelated text.
                        assert_equal(len(morsel.frame.column("s")), n)
                        assert_equal(
                            len(morsel.frame.column("y")), len(expected)
                        )
                    var result = morsel.materialized()
                    _ = morsel^
                    assert_equal(result.height(), len(expected))
                    if drop_all:
                        assert_equal(result.width(), 0)
                        continue
                    var names: List[String] = ["y", "s"]
                    assert_equal(result.columns(), names)
                    for at in range(len(expected)):
                        var row = expected[at]
                        assert_equal(
                            result.item(at, "y").is_null(), row % 7 == 2
                        )
                        if row % 7 != 2:
                            assert_equal(
                                result.item(at, "y").int64(), Int64(100 + row)
                            )
                        assert_equal(
                            result.item(at, "s").is_null(), row % 5 == 1
                        )
                        if row % 5 != 1:
                            assert_equal(
                                String(result.item(at, "s").string()),
                                "λ-" + String(row) + "\x00tail",
                            )


def test_drop_after_selected_join_preserves_complete_public_results() raises:
    var x = List[Int64]()
    var k = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(9):
        x.append(Int64(row))
        k.append(Int64(row % 3))
        values.append(Int64(100 + row))
        valid.append(row % 4 != 0)
    var fact = DataFrame(
        [
            Series("x", Column[Int64](x^)),
            Series("k", Column[Int64](k^)),
            Series("value", Column[Int64](values^, valid)),
        ]
    )
    var dimension = DataFrame(
        [
            Series("dk", Column[Int64]([0, 1, 2])),
            Series("dval", Column[Int64]([0, 10, 20])),
        ]
    )
    var right = DataFrame(
        [
            Series("rk", Column[Int64]([0, 10])),
            Series("rval", Column[Int64]([500, 600])),
        ]
    )
    for how in ["inner", "left"]:
        var query = (
            fact.lazy()
            .filter(col("x") % lit(Int64(2)) == lit(Int64(0)))
            .join(dimension.lazy(), left_on=["k"], right_on=["dk"])
            .drop(["k"])
            .join(right.lazy(), left_on=["dval"], right_on=["rk"], how=how)
            .filter(
                col("rval").is_null()
                | ((col("x") + col("rval")) >= lit(Int64(0)))
            )
        )
        for optimize in [False, True]:
            for streaming in [False, True]:
                for batch in [1, 3, 64]:
                    var result = query.collect(
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=batch,
                        engine="cpu",
                    )
                    var expected = List[Int]()
                    for row in range(9):
                        if row % 2 == 0 and (how == "left" or row % 3 != 2):
                            expected.append(row)
                    assert_equal(result.height(), len(expected))
                    var names: List[String] = ["x", "value", "dval", "rval"]
                    assert_equal(result.columns(), names)
                    for at in range(len(expected)):
                        var row = expected[at]
                        assert_equal(result.item(at, "x").int64(), Int64(row))
                        assert_equal(
                            result.item(at, "value").is_null(), row % 4 == 0
                        )
                        if row % 4 != 0:
                            assert_equal(
                                result.item(at, "value").int64(),
                                Int64(100 + row),
                            )
                        assert_equal(
                            result.item(at, "dval").int64(),
                            Int64(10 * (row % 3)),
                        )
                        assert_equal(
                            result.item(at, "rval").is_null(), row % 3 == 2
                        )
                        if row % 3 != 2:
                            assert_equal(
                                result.item(at, "rval").int64(),
                                Int64(500 + 100 * (row % 3)),
                            )
