"""join_where: inequality, range and mixed joins against a filtered cross join."""
# contract-group: table.join
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.group_strategy_reference import ScopedEnvironment


def _ints(
    name: String, n: Int, seed: Int, span: Int, nulls: Bool = True
) raises -> Series:
    """`n` integers in [0, span), every seventh null when `nulls`."""
    var values = List[Int64](capacity=n)
    var valid = List[Bool](capacity=n)
    var state = seed * 2654435761 + 12345
    for i in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 62)
        values.append(Int64((state >> 17) % span))
        valid.append(not nulls or i % 7 != 3)
    return Series(name, Column[Int64](values^, valid^))


def _floats(name: String, n: Int, seed: Int, span: Int) raises -> Series:
    """Like `_ints` as Float64 halves, with a NaN every eleventh row."""
    var ints = _ints(name, n, seed, span)
    var values = List[Float64](capacity=n)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        var v = ints.get(i)
        valid.append(not v.is_null())
        if i % 11 == 5:
            var zero = Float64(i - i)
            values.append(zero / zero)
        else:
            values.append(0.0 if v.is_null() else Float64(v.int64()) / 2.0)
    return Series(name, Column[Float64](values^, valid^))


def _row_ids(name: String, n: Int) raises -> Series:
    var values = List[Int64](capacity=n)
    for i in range(n):
        values.append(Int64(i))
    return Series(name, Column[Int64](values^))


def _expected(
    left: DataFrame, right: DataFrame, predicates: List[Expr]
) raises -> DataFrame:
    """Every pair, filtered: the cross join made first, as its own plan."""
    var crossed = left.join(right, how="cross")
    var predicate = predicates[0].copy()
    for k in range(1, len(predicates)):
        predicate = predicate & predicates[k]
    return crossed.filter(predicate)


def _check(
    left: DataFrame, right: DataFrame, predicates: List[Expr], label: String
) raises:
    var expected = _expected(left, right, predicates)
    var names = expected.columns()
    expected = expected.sort(names)
    for threads in ["1", "8"]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
        var eager = left.join_where(right, predicates)
        var lazy = left.lazy().join_where(right.lazy(), predicates).collect()
        assert_equal(eager.columns(), names, label)
        assert_equal(lazy.columns(), names, label)
        assert_equal(eager.height(), expected.height(), label + " eager")
        assert_equal(lazy.height(), expected.height(), label + " lazy")
        assert_true(eager.sort(names).equals(expected), label + " eager")
        assert_true(lazy.sort(names).equals(expected), label + " lazy")
        _ = workers^


def _ops(a: Expr, b: Expr) -> List[Expr]:
    return [a < b, a <= b, a > b, a >= b]


def test_one_inequality_every_comparison_and_point_side() raises:
    """One comparison of integer keys with repeats and nulls, sorting the
    smaller side either way round."""
    for shape in [(40, 300), (300, 40), (257, 257)]:
        var left = DataFrame(
            [_row_ids("id", shape[0]), _ints("x", shape[0], 1, 50)]
        )
        var right = DataFrame(
            [_row_ids("id", shape[1]), _ints("x", shape[1], 2, 50)]
        )
        var preds = _ops(col("x"), col("x_right"))
        for k in range(4):
            _check(left, right, [preds[k].copy()], "one " + String(k))


def test_range_join_with_point_on_either_side() raises:
    """Two comparisons bounding one key of a side: events within sessions
    (left point) and sessions containing a point (right point), with float
    keys holding NaN and nulls."""
    var events = DataFrame([_row_ids("e", 500), _floats("ts", 500, 3, 400)])
    var sessions = DataFrame(
        [
            _row_ids("s", 120),
            _floats("start", 120, 4, 400),
            _floats("end", 120, 5, 400),
        ]
    )
    var lower: List[Expr] = [
        col("ts") >= col("start"),
        col("ts") > col("start"),
    ]
    var upper: List[Expr] = [col("ts") < col("end"), col("ts") <= col("end")]
    for i in range(2):
        for j in range(2):
            _check(
                events,
                sessions,
                [lower[i].copy(), upper[j].copy()],
                "left point",
            )
            # The same, written from the sessions' side.
            _check(
                sessions,
                events,
                [col("start") <= col("ts"), col("end") > col("ts")],
                "right point",
            )


def test_two_key_inequality_join_every_comparison_pair() raises:
    """Different keys in each comparison on both sides, with many ties:
    every pair of comparison kinds, on enough rows that several blocks
    pair up on eight workers."""
    var left = DataFrame(
        [_row_ids("id", 260), _ints("x", 260, 6, 30), _ints("y", 260, 7, 30)]
    )
    var right = DataFrame(
        [_row_ids("id", 230), _ints("x", 230, 8, 30), _ints("y", 230, 9, 30)]
    )
    var first = _ops(col("x"), col("x_right"))
    var second = _ops(col("y"), col("y_right"))
    for i in range(4):
        for j in range(4):
            _check(
                left,
                right,
                [first[i].copy(), second[j].copy()],
                "two keys " + String(i) + String(j),
            )
    var big_left = DataFrame(
        [
            _row_ids("id", 3000),
            _ints("x", 3000, 10, 2000),
            _ints("y", 3000, 11, 2000),
        ]
    )
    var big_right = DataFrame(
        [
            _row_ids("id", 2500),
            _ints("x", 2500, 12, 2000),
            _ints("y", 2500, 13, 2000),
        ]
    )
    _check(
        big_left,
        big_right,
        [col("x") < col("x_right"), col("y") > col("y_right") + 1900],
        "two keys, blocks",
    )


def test_equalities_use_the_hash_join_and_filter_the_rest() raises:
    var events = DataFrame(
        [
            _ints("user", 400, 14, 9),
            _ints("ts", 400, 15, 100),
            _row_ids("e", 400),
        ]
    )
    var sessions = DataFrame(
        [
            _ints("user", 60, 16, 9),
            _ints("start", 60, 17, 100),
            _ints("end", 60, 18, 100),
        ]
    )
    _check(
        events,
        sessions,
        [
            col("ts") >= col("start"),
            col("ts") < col("end"),
            col("user") == col("user_right"),
        ],
        "mixed",
    )
    # Two equalities on one left column: one is the hash key, the other
    # filters.
    _check(
        events,
        sessions,
        [col("user") == col("user_right"), col("user") == col("start")],
        "repeated key",
    )
    # Float equality keys stay in the filter too: NaN matches nothing.
    var a = DataFrame([_floats("f", 200, 19, 20), _row_ids("i", 200)])
    var b = DataFrame([_floats("f", 150, 20, 20), _row_ids("j", 150)])
    _check(a, b, [col("f") == col("f_right")], "float equality")


def test_expressions_extra_predicates_dates_and_empty_inputs() raises:
    var left = DataFrame(
        [_row_ids("id", 300), _ints("x", 300, 21, 80), _ints("y", 300, 22, 80)]
    )
    var right = DataFrame(
        [
            _row_ids("id", 200),
            _ints("lo", 200, 23, 80),
            _ints("hi", 200, 24, 80),
        ]
    )
    # Expression operands, a third comparison and a predicate that is no
    # comparison of the two sides stay correct.
    _check(
        left,
        right,
        [
            col("x") + 5 > col("lo"),
            col("x") < col("hi") * 2,
            col("y") >= col("lo"),
            (col("id") + col("id_right")) % 3 != 0,
        ],
        "expressions",
    )
    _check(left, right, [col("x") != col("lo")], "no inequality")
    # Dates compare as dates.
    var dl = DataFrame(
        [_row_ids("id", 100), _ints("d", 100, 25, 60).with_dtype(DataType.DATE)]
    )
    var dr = DataFrame(
        [_row_ids("k", 80), _ints("d", 80, 26, 60).with_dtype(DataType.DATE)]
    )
    _check(dl, dr, [col("d") < col("d_right")], "dates")
    # A date comparison and a float one: one inequality join takes keys of
    # one kind, the other comparison filters.
    var dfl = dl.with_column(_floats("f", 100, 30, 40))
    var dfr = dr.with_column(_floats("f", 80, 31, 40))
    _check(
        dfl,
        dfr,
        [col("d") >= col("d_right"), col("f") >= col("f_right")],
        "date and float",
    )
    # No rows on one side: the joined schema.
    var empty = right.clear()
    var out = left.join_where(empty, [col("x") < col("lo")])
    assert_equal(out.height(), 0)
    assert_equal(out.columns(), ["id", "x", "y", "id_right", "lo", "hi"])
    with assert_raises(contains="at least one predicate"):
        _ = left.join_where(right, List[Expr]())


def test_lazy_filters_above_and_plan_shape() raises:
    """Filters above a join_where reach its inputs, and the plan is an
    inequality join, not a cross join."""
    var events = DataFrame([_row_ids("e", 300), _ints("ts", 300, 27, 200)])
    var sessions = DataFrame(
        [
            _row_ids("s", 90),
            _ints("start", 90, 28, 200),
            _ints("end", 90, 29, 200),
        ]
    )
    var preds: List[Expr] = [col("ts") >= col("start"), col("ts") < col("end")]
    var plan = (
        events.lazy()
        .join_where(sessions.lazy(), preds)
        .filter(col("e") % 2 == 0)
        .filter(col("s") < 50)
    )
    var explained = plan.explain()
    assert_true("RANGE_JOIN" in explained, explained)
    # Against one row (a scalar's), filtering every pair is less work than
    # sorting: the plan keeps the filtered cross join, with the same rows.
    var one = sessions.head(1)
    var scalar = events.lazy().join_where(one.lazy(), preds)
    assert_true("RANGE_JOIN" not in scalar.explain(), scalar.explain())
    _check(events, one, preds, "one row")
    var expected = (
        _expected(events, sessions, preds)
        .filter((col("e") % 2 == 0) & (col("s") < 50))
        .sort(["e", "s"])
    )
    for threads in ["1", "8"]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
        assert_true(plan.collect().sort(["e", "s"]).equals(expected))
        _ = workers^
