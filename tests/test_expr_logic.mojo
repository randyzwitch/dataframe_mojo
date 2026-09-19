"""Kleene Boolean logic, null/NaN predicates, fills, and any/all reductions."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import (
    AnyValue,
    Column,
    DataFrame,
    Expr,
    Series,
    coalesce,
    col,
    lit,
    null,
)
from dataframe.binding import bind
from dataframe.execution import evaluate
from dataframe.reductions import LogicState


def nan() -> Float64:
    return Float64(0) / Float64(0)


def inf() -> Float64:
    return Float64(1) / Float64(0)


def tristate() raises -> DataFrame:
    """Every (p, q) pair over {true, false, null}."""
    var p = List[Bool]()
    var pv = List[Bool]()
    var q = List[Bool]()
    var qv = List[Bool]()
    for a in range(3):
        for b in range(3):
            p.append(a == 0)
            pv.append(a != 2)
            q.append(b == 0)
            qv.append(b != 2)
    return DataFrame(
        [
            Series("p", Column[Bool](p^, pv^)),
            Series("q", Column[Bool](q^, qv^)),
        ]
    )


def run(frame: DataFrame, expr: Expr, batch_size: Int = 4) raises -> Series:
    return evaluate(
        bind(expr, frame._columns),
        frame._columns,
        frame.height(),
        batch_size=batch_size,
    )


def cells(series: Series) raises -> String:
    """Compact T/F/N string for truth-table comparisons."""
    var out = String()
    for i in range(len(series)):
        var value = series.get(i)
        if value.is_null():
            out += "N"
        else:
            out += "T" if value.bool() else "F"
    return out^


def test_kleene_truth_tables() raises:
    var frame = tristate()
    # Rows: (T,T) (T,F) (T,N) (F,T) (F,F) (F,N) (N,T) (N,F) (N,N)
    assert_equal(cells(run(frame, col("p") & col("q"))), "TFNFFFNFN")
    assert_equal(cells(run(frame, col("p") | col("q"))), "TTTTFNTNN")
    assert_equal(cells(run(frame, col("p") ^ col("q"))), "FTNTFNNNN")
    assert_equal(cells(run(frame, ~col("p"))), "FFFTTTNNN")
    assert_equal(cells(run(frame, col("p").and_(col("q")))), "TFNFFFNFN")
    assert_equal(cells(run(frame, col("p").or_(col("q")))), "TTTTFNTNN")
    assert_equal(cells(run(frame, col("p").not_())), "FFFTTTNNN")
    # Scalar broadcasting, including a typed null literal.
    assert_equal(cells(run(frame, col("q") & lit(False))), "FFFFFFFFF")
    assert_equal(cells(run(frame, col("q") | lit(True))), "TTTTTTTTT")
    assert_equal(cells(run(frame, col("q") & null("bool"))), "NFNNFNNFN")
    assert_equal(cells(run(frame, null("bool") | col("q"))), "TNNTNNTNN")
    # Results do not depend on batch boundaries.
    for size in range(1, 10):
        assert_true(
            run(frame, (col("p") & col("q")) | ~col("q"), size).equals(
                run(frame, (col("p") & col("q")) | ~col("q"), 9)
            )
        )


def test_filter_keeps_only_true() raises:
    var frame = tristate().with_row_index()
    var kept = frame.filter(col("p") | col("q"))
    var rows = kept.column("index").int64()
    var expected: List[Int64] = [0, 1, 2, 3, 6]
    assert_equal(len(rows), len(expected))
    for i in range(len(expected)):
        assert_equal(rows.value(i), expected[i])


def test_null_and_nan_predicates() raises:
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [1.5, nan(), inf(), -inf(), 0],
                    [True, True, True, True, False],
                ),
            ),
            Series(
                "s",
                Column[String](
                    ["a", "", "b", "c", "d"], [True, True, False, True, True]
                ),
            ),
        ]
    )
    assert_equal(cells(run(frame, col("x").is_null())), "FFFFT")
    assert_equal(cells(run(frame, col("x").is_not_null())), "TTTTF")
    assert_equal(cells(run(frame, col("s").is_null())), "FFTFF")
    assert_equal(cells(run(frame, col("x").is_nan())), "FTFFN")
    assert_equal(cells(run(frame, col("x").is_not_nan())), "TFTTN")
    assert_equal(cells(run(frame, col("x").is_finite())), "TFFFN")
    assert_equal(cells(run(frame, col("x").is_infinite())), "FFTTN")
    assert_equal(run(frame, lit(Int64(1)).is_null()).bool().value(0), False)
    assert_true(run(frame, null("string").is_null()).bool().value(0))
    var ints = DataFrame([Series("i", Column[Int64]([1]))])
    with assert_raises(contains="is_nan requires a float operand, found int64"):
        _ = ints.select(col("i").is_nan())
    with assert_raises(contains="not requires a bool operand"):
        _ = ints.select(~col("i"))
    with assert_raises(contains="and requires bool operands, found int64"):
        _ = ints.select(col("i") & col("i"))
    with assert_raises(contains="Unknown null literal dtype"):
        _ = ints.select(null("int128"))


def test_fill_null_fill_nan_and_coalesce() raises:
    var frame = DataFrame(
        [
            Series(
                "a", Column[Int64]([1, 0, 0, 4], [True, False, False, True])
            ),
            Series("b", Column[Int64]([9, 8, 0, 6], [True, True, False, True])),
            Series("c", Column[Int64]([7, 7, 7, 7])),
            Series(
                "x",
                Column[Float64](
                    [nan(), 2, 0, nan()], [True, True, False, True]
                ),
            ),
            Series(
                "s",
                Column[String](["p", "", "r", "s"], [False, True, True, False]),
            ),
        ]
    )
    var filled = run(frame, col("a").fill_null(lit(Int64(-1))))
    assert_true(filled.equals(Series("a", Column[Int64]([1, -1, -1, 4]))))
    var from_column = run(frame, col("a").fill_null(col("b")))
    assert_true(
        from_column.equals(
            Series("a", Column[Int64]([1, 8, 0, 4], [True, True, False, True]))
        )
    )
    assert_equal(run(frame, col("a").fill_null(null("int64"))).null_count(), 2)
    var strings = run(frame, col("s").fill_null(lit(String("?"))))
    assert_equal(strings.get(0).string(), "?")
    assert_equal(strings.get(1).string(), "")
    var no_nan = run(frame, col("x").fill_nan(lit(Float64(0)))).float64()
    assert_equal(no_nan.value(0), Float64(0))
    assert_equal(no_nan.value(1), Float64(2))
    assert_true(no_nan.is_null(2))
    assert_equal(no_nan.value(3), Float64(0))
    assert_equal(run(frame, col("x").fill_nan(null("float64"))).null_count(), 3)
    var first = run(frame, coalesce([col("a"), col("b"), col("c")]))
    assert_true(first.equals(Series("a", Column[Int64]([1, 8, 7, 4]))))
    assert_equal(frame.select(coalesce([col("b")])).column("b").null_count(), 1)
    with assert_raises(contains="at least one"):
        _ = coalesce(List[Expr]())
    with assert_raises(contains="fill_null requires matching dtypes"):
        _ = frame.select(col("a").fill_null(lit(Float64(0))))
    with assert_raises(contains="fill_nan requires float operands"):
        _ = frame.select(col("a").fill_nan(lit(Int64(0))))


def test_is_in_and_is_between() raises:
    var frame = DataFrame(
        [
            Series(
                "n",
                Column[Int64]([1, 2, 3, 4, 5], [True, True, True, True, False]),
            ),
            Series("s", Column[String](["a", "b", "", "d", "e"])),
        ]
    )
    assert_equal(
        cells(run(frame, col("n").is_in([lit(Int64(2)), lit(Int64(4))]))),
        "FTFTN",
    )
    # A null candidate never matches and never makes the result null.
    assert_equal(
        cells(run(frame, col("n").is_in([null("int64"), lit(Int64(1))]))),
        "TFFFN",
    )
    assert_equal(cells(run(frame, col("n").is_in(List[Expr]()))), "FFFFN")
    assert_equal(
        cells(run(frame, col("s").is_in([lit(String("")), lit(String("e"))]))),
        "FFTFT",
    )
    assert_equal(
        frame.select(col("n").is_in([lit(Int64(1))])).columns()[0], "n"
    )
    with assert_raises(contains="eq requires matching dtypes"):
        _ = frame.select(col("n").is_in([lit(String("1"))]))
    var lo = lit(Int64(2))
    var hi = lit(Int64(4))
    assert_equal(cells(run(frame, col("n").is_between(lo, hi))), "FTTTN")
    assert_equal(
        cells(run(frame, col("n").is_between(lo, hi, "left"))), "FTTFN"
    )
    assert_equal(
        cells(run(frame, col("n").is_between(lo, hi, "right"))), "FFTTN"
    )
    assert_equal(
        cells(run(frame, col("n").is_between(lo, hi, "none"))), "FFTFN"
    )
    assert_equal(
        cells(run(frame, col("n").is_between(null("int64"), lit(Int64(3))))),
        "NNNFN",
    )
    with assert_raises(contains="closed must be"):
        _ = col("n").is_between(lo, hi, "open")


def test_any_all_and_null_count() raises:
    var frame = tristate()
    var result = frame.select_exprs(
        [
            col("q").any().alias("any"),
            col("q").all().alias("all"),
            col("q").null_count().alias("nulls"),
            col("q").count().alias("count"),
        ]
    )
    assert_true(result.item(0, "any").bool())
    assert_false(result.item(0, "all").bool())
    assert_equal(result.item(0, "nulls").int64(), Int64(3))
    assert_equal(result.item(0, "count").int64(), Int64(6))
    var only_true = frame.filter(col("q").is_null() | col("q"))
    var kleene = only_true.select_exprs(
        [
            col("q").all().alias("ignoring"),
            col("q").all(ignore_nulls=False).alias("kleene"),
            (~col("q")).any(ignore_nulls=False).alias("any_kleene"),
            (~col("q")).any().alias("any_ignoring"),
        ]
    )
    assert_true(kleene.item(0, "ignoring").bool())
    assert_true(kleene.item(0, "kleene").is_null())
    assert_true(kleene.item(0, "any_kleene").is_null())
    assert_false(kleene.item(0, "any_ignoring").bool())
    var empty = frame.clear().select_exprs(
        [
            col("q").any().alias("any"),
            col("q").all().alias("all"),
            col("q").null_count().alias("nulls"),
        ]
    )
    assert_false(empty.item(0, "any").bool())
    assert_true(empty.item(0, "all").bool())
    assert_equal(empty.item(0, "nulls").int64(), Int64(0))
    with assert_raises(contains="any requires a bool expression"):
        _ = frame.with_row_index().select(col("index").any())


def test_grouped_logic_reductions() raises:
    var frame = DataFrame(
        [
            Series("k", Column[String](["a", "a", "b", "b", "c", "c"])),
            Series(
                "v",
                Column[Bool](
                    [True, False, True, False, False, False],
                    [True, True, True, False, True, False],
                ),
            ),
        ]
    )
    var result = frame.group_by("k", maintain_order=True).agg(
        [
            col("v").any().alias("any"),
            col("v").all().alias("all"),
            col("v").all(ignore_nulls=False).alias("all_kleene"),
            col("v").any(ignore_nulls=False).alias("any_kleene"),
            col("v").null_count().alias("nulls"),
        ]
    )
    assert_equal(cells(result.column("any")), "TTF")
    assert_equal(cells(result.column("all")), "FTF")
    assert_equal(cells(result.column("all_kleene")), "FNF")
    assert_equal(cells(result.column("any_kleene")), "TTN")
    var nulls = result.column("nulls").int64()
    assert_equal(nulls.value(0), Int64(0))
    assert_equal(nulls.value(1), Int64(1))
    assert_equal(nulls.value(2), Int64(1))


def test_logic_state_merge_is_partition_independent() raises:
    var valid: List[Bool] = [True, True, False, True, False, True, True]
    var value: List[Bool] = [False, True, False, False, True, True, False]
    var whole = LogicState()
    for i in range(len(valid)):
        whole.add(valid[i], value[i])
    for split in range(len(valid) + 1):
        var left = LogicState()
        var right = LogicState()
        for i in range(len(valid)):
            if i < split:
                left.add(valid[i], value[i])
            else:
                right.add(valid[i], value[i])
        right.merge(left)
        for ignore in range(2):
            var a = right.any(ignore == 1)
            var b = whole.any(ignore == 1)
            assert_equal(Bool(a), Bool(b))
            var c = right.all(ignore == 1)
            var d = whole.all(ignore == 1)
            assert_equal(Bool(c), Bool(d))
            if c and d:
                assert_equal(c.value(), d.value())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
