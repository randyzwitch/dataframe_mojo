"""Value selection preserves selected validity and resolves logical result types."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    coalesce,
    col,
    concat_str,
    lit,
    null,
    when,
)
from tests.support.coverage_generators import offset_layout, chunk_layout


def _layout(frame: DataFrame, layout: Int) raises -> DataFrame:
    return (
        frame.copy() if layout
        == 0 else offset_layout(frame) if layout
        == 1 else chunk_layout(frame)
    )


def test_decimal_branch_fill_and_coalesce_resolve_common_type_without_rounding() raises:
    var x: List[Int128] = [125, -250, 399, 750, 0]
    var y: List[Int128] = [-100, 75, 20, 100, 5]
    var xv: List[Bool] = [True, False, True, True, False]
    var yv: List[Bool] = [True, True, False, True, False]
    var condition: List[Bool] = [True, True, False, False, True]
    var cv: List[Bool] = [True, True, True, False, True]
    for left_width in [32, 64, 128]:
        for right_width in [32, 64, 128]:
            var a = DataType.decimal(7, 2, left_width)
            var b = DataType.decimal(7, 2, right_width)
            var common = (
                a.copy() if left_width
                == right_width else DataType.decimal(38, 2)
            )
            var raw = DataFrame(
                [
                    Series("x", Column[Int128](x.copy(), xv.copy()))
                    .with_dtype(DataType.decimal(7, 2))
                    .cast(a),
                    Series("y", Column[Int128](y.copy(), yv.copy()))
                    .with_dtype(DataType.decimal(7, 2))
                    .cast(b),
                    Series("c", Column[Bool](condition.copy(), cv.copy())),
                    Series("g", Column[Int64]([1, 0, 1, 0, 0])),
                ]
            )
            var expressions: List[Expr] = [
                when(col("c"))
                .then(col("x"))
                .otherwise(col("y"))
                .alias("branch"),
                when(col("c")).then(col("x")).end().alias("no_else"),
                col("x").fill_null(col("y")).alias("fill"),
                coalesce([col("x"), col("y")]).alias("coalesce"),
            ]
            for layout in range(3):
                var frame = _layout(raw, layout)
                var actual = frame.select_exprs(expressions)
                assert_equal(
                    actual.columns(), ["branch", "no_else", "fill", "coalesce"]
                )
                assert_equal(actual.height(), 5)
                var branch_totals = List[Int128](length=2, fill=0)
                for operation in range(4):
                    var name = ["branch", "no_else", "fill", "coalesce"][
                        operation
                    ]
                    assert_true(
                        actual.column(name).dtype()
                        == (a if operation == 1 else common)
                    )
                    for row in range(5):
                        var choose_x = (
                            cv[row]
                            and condition[row] if operation < 2 else xv[row]
                        )
                        var present = (
                            xv[row] if choose_x else yv[row] if operation
                            != 1 else False
                        )
                        assert_equal(
                            actual.item(row, name).is_null(), not present
                        )
                        if present:
                            var expected = x[row] if choose_x else y[row]
                            assert_equal(
                                actual.column(name)._decimal_raw(row), expected
                            )
                            if operation == 0:
                                branch_totals[
                                    1 if row == 0 or row == 2 else 0
                                ] += expected
                var grouped = frame.group_by("g", maintain_order=True).agg(
                    [expressions[0].copy().sum().alias("sum")]
                )
                assert_equal(grouped.columns(), ["g", "sum"])
                assert_equal(grouped.height(), 2)
                assert_true(
                    grouped.column("sum").dtype()
                    == (
                        DataType.decimal(38, 2) if common.decimal_width()
                        < 128 else common
                    )
                )
                for row in range(2):
                    var group = 1 - row
                    assert_equal(grouped.item(row, "g").int64(), Int64(group))
                    assert_equal(
                        grouped.column("sum")._decimal_raw(row),
                        branch_totals[group],
                    )
                for optimize in [False, True]:
                    for streaming in [False, True]:
                        var lazy = (
                            frame.lazy()
                            .select_exprs(expressions)
                            .collect(
                                optimize=optimize,
                                streaming=streaming,
                                batch_size=2,
                                engine="cpu",
                            )
                        )
                        for operation in range(4):
                            var name = [
                                "branch",
                                "no_else",
                                "fill",
                                "coalesce",
                            ][operation]
                            assert_true(
                                lazy.column(name).dtype()
                                == (a if operation == 1 else common)
                            )
                            for row in range(5):
                                var choose_x = (
                                    cv[row]
                                    and condition[row] if operation
                                    < 2 else xv[row]
                                )
                                var present = (
                                    xv[row] if choose_x else yv[
                                        row
                                    ] if operation
                                    != 1 else False
                                )
                                assert_equal(
                                    lazy.item(row, name).is_null(),
                                    not present,
                                )
                                if present:
                                    assert_equal(
                                        lazy.column(name)._decimal_raw(row),
                                        x[row] if choose_x else y[row],
                                    )
                for current in [frame.copy(), frame.clear()]:
                    var bad = lit("0").cast(DataType.decimal(7, 3))
                    for expression in [
                        when(col("c")).then(col("x")).otherwise(bad.copy()),
                        col("x").fill_null(bad.copy()),
                        coalesce([col("x"), bad.copy()]),
                    ]:
                        with assert_raises(
                            contains="requires decimals of one scale"
                        ):
                            _ = current.select(expression)


def test_categorical_expressions_read_decoded_strings_and_derived_casts_reencode_values() raises:
    var strings: List[String] = ["z", "é", "", "z", "ignored"]
    var valid: List[Bool] = [True, True, True, True, False]
    var raw = DataFrame(
        [
            Series("x", Column[String](strings.copy(), valid.copy())).cast(
                DataType.CATEGORICAL
            )
        ]
    )
    for layout in range(3):
        var frame = _layout(raw, layout)
        assert_true(
            frame.select(col("x")).column("x").dtype()
            == frame.column("x").dtype()
        )
        var actual = frame.select_exprs(
            [
                col("x").str().to_uppercase().alias("upper"),
                col("x").str().contains("é").alias("contains"),
                col("x").is_in(["z", ""]).alias("member"),
                col("x").fill_null("missing").alias("fill"),
                concat_str([col("x"), lit("!")]).alias("concat"),
                when(col("x") == "z")
                .then(col("x"))
                .otherwise("other")
                .alias("branch"),
                col("x")
                .str()
                .slice(0, 1)
                .cast(DataType.CATEGORICAL)
                .alias("encoded"),
            ]
        )
        for name in ["upper", "fill", "concat", "branch"]:
            assert_true(actual.column(name).dtype() == DataType.STRING)
        for name in ["contains", "member"]:
            assert_true(actual.column(name).dtype() == DataType.BOOL)
        assert_true(actual.column("encoded").dtype().is_categorical())
        var uppercase: List[String] = ["Z", "É", "", "Z", ""]
        for row in range(5):
            for name in ["upper", "contains", "member", "concat", "encoded"]:
                assert_equal(actual.item(row, name).is_null(), not valid[row])
            assert_equal(
                actual.item(row, "fill").string(),
                strings[row] if valid[row] else "missing",
            )
            assert_equal(
                actual.item(row, "branch").string(),
                "z" if valid[row] and strings[row] == "z" else "other",
            )
            if valid[row]:
                assert_equal(actual.item(row, "upper").string(), uppercase[row])
                assert_equal(actual.item(row, "contains").bool(), row == 1)
                assert_equal(actual.item(row, "member").bool(), row != 1)
                assert_equal(
                    actual.item(row, "concat").string(), strings[row] + "!"
                )
                assert_equal(actual.item(row, "encoded").string(), strings[row])


def test_membership_and_interval_endpoints_use_kleene_null_rules() raises:
    var raw = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [0, 1, 2, 3, 4, 999], [True, True, True, True, True, False]
                ),
            )
        ]
    )
    for layout in range(3):
        var frame = _layout(raw, layout)
        var candidates: List[Expr] = [
            null(DataType.INT64),
            lit(Int64(1)),
            lit(Int64(3)),
            lit(Int64(1)),
        ]
        for empty in [False, True]:
            var actual = frame.select(
                col("x").is_in(List[Expr]() if empty else candidates.copy())
            )
            for row in range(6):
                assert_equal(actual.item(row, "x").is_null(), row == 5)
                if row != 5:
                    assert_equal(
                        actual.item(row, "x").bool(),
                        not empty and (row == 1 or row == 3),
                    )
        for closed in ["both", "left", "right", "none"]:
            var actual = frame.select(col("x").is_between(1, 3, closed=closed))
            for row in range(6):
                assert_equal(actual.item(row, "x").is_null(), row == 5)
                if row != 5:
                    var lower = row > 1 or (
                        row == 1 and (closed == "both" or closed == "left")
                    )
                    var upper = row < 3 or (
                        row == 3 and (closed == "both" or closed == "right")
                    )
                    assert_equal(actual.item(row, "x").bool(), lower and upper)
        var unknown = frame.select(
            col("x").is_between(null(DataType.INT64), lit(Int64(3)))
        )
        for row in range(6):
            assert_equal(unknown.item(row, "x").is_null(), row != 4)
        assert_true(not unknown.item(4, "x").bool())
    with assert_raises(contains="closed must be"):
        _ = col("x").is_between(1, 3, closed="invalid")
    # Explicit typed literals retain their width inside membership, just
    # as in scalar equality; an untyped literal adopts the column width.
    with assert_raises(contains="eq requires matching dtypes"):
        _ = raw.select(col("x").is_in([lit(Int32(1))]))
    var narrow = DataFrame([Series("x", Column[Int32]([1]))])
    with assert_raises(contains="eq requires matching dtypes"):
        _ = narrow.select(col("x").is_in([lit(Int64(1))]))


def test_standalone_untyped_expressions_default_once_and_keep_scalar_shapes() raises:
    for height in [0, 3]:
        var frame = DataFrame([], height=height)
        var actual = frame.select_exprs(
            [
                Expr(7).alias("i"),
                Expr(0.5).alias("f"),
                (Expr(1) + 2.5).alias("mixed"),
                (Expr(3) > 2).alias("comparison"),
            ]
        )
        assert_equal(actual.shape(), (1, 4))
        assert_equal(
            actual.dtypes(),
            [DataType.INT64, DataType.FLOAT64, DataType.FLOAT64, DataType.BOOL],
        )
        assert_equal(actual.item(0, "i").int64(), Int64(7))
        assert_equal(actual.item(0, "f").float64(), Float64(0.5))
        assert_equal(actual.item(0, "mixed").float64(), Float64(3.5))
        assert_true(actual.item(0, "comparison").bool())


def test_boolean_comparisons_order_false_before_true_and_propagate_either_null() raises:
    var a = List[Bool]()
    var b = List[Bool]()
    var av = List[Bool]()
    var bv = List[Bool]()
    for left in range(3):
        for right in range(3):
            a.append(left == 1)
            b.append(right == 1)
            av.append(left != 2)
            bv.append(right != 2)
    var raw = DataFrame(
        [
            Series("a", Column[Bool](a.copy(), av.copy())),
            Series("b", Column[Bool](b.copy(), bv.copy())),
        ]
    )
    var expressions: List[Expr] = [
        col("a") < col("b"),
        col("a") <= col("b"),
        col("a") > col("b"),
        col("a") >= col("b"),
        col("a").eq(col("b")),
        col("a").ne(col("b")),
    ]
    for layout in range(3):
        var frame = _layout(raw, layout)
        for operation in range(6):
            var actual = frame.select(expressions[operation].copy().alias("r"))
            assert_true(actual.column("r").dtype() == DataType.BOOL)
            assert_equal(actual.height(), 9)
            for row in range(9):
                assert_equal(
                    actual.item(row, "r").is_null(), not (av[row] and bv[row])
                )
                if av[row] and bv[row]:
                    var left = Int(a[row])
                    var right = Int(b[row])
                    var expected = (
                        left
                        < right if operation
                        == 0 else left
                        <= right if operation
                        == 1 else left
                        > right if operation
                        == 2 else left
                        >= right if operation
                        == 3 else left
                        == right if operation
                        == 4 else left
                        != right
                    )
                    assert_equal(actual.item(row, "r").bool(), expected)
