"""Actual reduction-state APIs under independent partition/association laws."""
from std.math import abs, isnan, sqrt
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series
from dataframe.reductions import (
    IntSumState,
    FloatSumState,
    LogicState,
    VarState,
    MomentState,
    CoMomentState,
)
from dataframe.aggregate import Reducer
from dataframe.expr import (
    FIRST,
    LAST,
    MIN,
    MAX,
    SUM,
    MEAN,
    COUNT,
    NULL_COUNT,
    LEN,
    N_UNIQUE,
    ARG_MIN,
    ARG_MAX,
)
from tests.support.state_merge_reference import ExactMoments, exact_comoment
from tests.support.ordered_reduction_fixtures import typed_ordered_columns
from tests.support.coverage_compare import assert_float, assert_value
from tests.support.coverage_generators import offset_layout, chunk_layout


def _near(actual: Float64, expected: Float64, budget: Float64) raises:
    if isnan(expected):
        assert_true(isnan(actual))
    else:
        assert_true(not isnan(actual))
        assert_true(
            abs(actual - expected) <= budget,
            "state merge error exceeds declared numeric budget",
        )


struct _Statistics(Copyable):
    var total: FloatSumState
    var variance: VarState
    var moment: MomentState
    var pair: CoMomentState

    def __init__(
        out self,
        xs: List[Int64],
        ys: List[Int64],
        xv: List[Bool],
        yv: List[Bool],
        start: Int,
        end: Int,
    ):
        self.total = FloatSumState()
        self.variance = VarState()
        self.moment = MomentState()
        self.pair = CoMomentState()
        for row in range(start, end):
            if xv[row]:
                self.total.add(Float64(xs[row]))
                self.variance.add(Float64(xs[row]))
                self.moment.add(Float64(xs[row]))
            if xv[row] and yv[row]:
                self.pair.add(Float64(xs[row]), Float64(ys[row]))

    def merge(mut self, other: Self):
        self.total.merge(other.total)
        self.variance.merge(other.variance)
        self.moment.merge(other.moment)
        self.pair.merge(other.pair)


def _assert_statistics(
    state: _Statistics,
    model: ExactMoments,
    pairs_x: ExactMoments,
    pairs_y: ExactMoments,
    cxy: Float64,
) raises:
    assert_equal(state.total.count, Int64(model.count))
    assert_equal(state.variance.count, Int64(model.count))
    assert_equal(state.moment.count, Int64(model.count))
    assert_equal(state.pair.count, Int64(pairs_x.count))
    _near(state.total.total, Float64(model.total()), model.sum_budget())
    var mean_budget = (
        Float64(32 * max(1, model.count))
        * 2.220446049250313e-16
        * (abs(model.mean()) + model.span + 1)
    )
    _near(state.variance.mean, model.mean(), mean_budget)
    _near(state.moment.mean, model.mean(), mean_budget)
    _near(state.variance.m2, model.central(2), model.moment_budget(2))
    _near(state.moment.m2, model.central(2), model.moment_budget(2))
    _near(state.moment.m3, model.central(3), model.moment_budget(3))
    _near(state.moment.m4, model.central(4), model.moment_budget(4))
    _near(state.pair.mean_x, pairs_x.mean(), mean_budget)
    _near(state.pair.mean_y, pairs_y.mean(), mean_budget)
    _near(state.pair.m2x, pairs_x.central(2), pairs_x.moment_budget(2))
    _near(state.pair.m2y, pairs_y.central(2), pairs_y.moment_budget(2))
    var pair_budget = pairs_x.moment_budget(2) + pairs_y.moment_budget(2)
    _near(state.pair.cxy, cxy, pair_budget)
    for ddof in [0, 1, 2, 8]:
        var variance = state.variance.variance(ddof)
        assert_equal(Bool(variance), model.count > ddof)
        if variance:
            _near(
                variance.value(),
                model.central(2) / Float64(model.count - ddof),
                model.moment_budget(2) / Float64(model.count - ddof),
            )
        var covariance = state.pair.covariance(ddof)
        assert_equal(Bool(covariance), pairs_x.count > 0)
        if covariance:
            var expected = cxy / Float64(
                pairs_x.count - ddof
            ) if pairs_x.count > ddof else Float64(0)
            _near(covariance.value(), expected, pair_budget)
    if pairs_x.count < 2 or pairs_x.central(2) == 0 or pairs_y.central(2) == 0:
        assert_true(isnan(state.pair.correlation()))
    else:
        _near(
            state.pair.correlation(),
            cxy / sqrt(pairs_x.central(2) * pairs_y.central(2)),
            max(Float64(1e-10), pair_budget),
        )
    for bias in [False, True]:
        var skew = state.moment.skew(bias)
        assert_equal(Bool(skew), model.count > 0 and (bias or model.count >= 3))
        if skew:
            var n = Float64(model.count)
            var m2 = model.central(2)
            var expected = sqrt(n) * model.central(3) / (m2 * sqrt(m2))
            if not bias:
                expected *= sqrt(n * (n - 1)) / (n - 2)
            _near(
                skew.value(),
                expected,
                max(Float64(1e-10), model.moment_budget(3)),
            )
        for fisher in [False, True]:
            var kurtosis = state.moment.kurtosis(fisher, bias)
            assert_equal(
                Bool(kurtosis), model.count > 0 and (bias or model.count >= 4)
            )
            if kurtosis:
                var n = Float64(model.count)
                var m2 = model.central(2)
                var excess = n * model.central(4) / (m2 * m2) - 3
                if not bias:
                    excess = (
                        ((n + 1) * excess + 6) * (n - 1) / ((n - 2) * (n - 3))
                    )
                _near(
                    kurtosis.value(),
                    excess if fisher else excess + 3,
                    max(Float64(1e-10), model.moment_budget(4)),
                )


def test_statistical_states_match_exact_reference_at_every_split_and_association() raises:
    for scenario in range(3):
        var offset = Int64(1_000_000_000 if scenario == 1 else 0)
        var xs: List[Int64] = [
            offset + 3,
            offset + 1,
            offset + 9,
            offset + 2,
            offset + 6,
            offset + 5,
            offset + 10,
        ]
        var ys: List[Int64] = [
            offset + 8,
            offset + 4,
            offset + 3,
            offset + 12,
            offset + 7,
            offset + 1,
            offset + 9,
        ]
        if scenario == 2:
            xs = List[Int64](length=len(xs), fill=3)
        for prefix in [0, 1, 2, 3, 7]:
            for mode in range(2):
                var xv = List[Bool]()
                var yv = List[Bool]()
                var pairs = List[Bool]()
                for row in range(len(xs)):
                    xv.append(row < prefix and (mode == 0 or row % 3 != 0))
                    yv.append(row < prefix and (mode == 0 or row % 3 != 1))
                    pairs.append(xv[row] and yv[row])
                var model = ExactMoments(xs, xv)
                var pair_x = ExactMoments(xs, pairs)
                var pair_y = ExactMoments(ys, pairs)
                var cxy = exact_comoment(xs, ys, pairs)
                for a in range(len(xs) + 1):
                    for b in range(a, len(xs) + 1):
                        var first = _Statistics(xs, ys, xv, yv, 0, a)
                        var middle = _Statistics(xs, ys, xv, yv, a, b)
                        var last = _Statistics(xs, ys, xv, yv, b, len(xs))
                        var left = first.copy()
                        left.merge(middle)
                        left.merge(last)
                        _assert_statistics(left, model, pair_x, pair_y, cxy)
                        var tail = middle.copy()
                        tail.merge(last)
                        var right = first.copy()
                        right.merge(tail)
                        _assert_statistics(right, model, pair_x, pair_y, cxy)
                        var reordered = last.copy()
                        reordered.merge(middle)
                        reordered.merge(first)
                        _assert_statistics(
                            reordered, model, pair_x, pair_y, cxy
                        )


def test_state_reference_and_error_budgets_have_literal_and_mutation_anchors() raises:
    var values: List[Int64] = [1, 2, 4, 5]
    var valid: List[Bool] = [True, True, True, True]
    var model = ExactMoments(values, valid)
    assert_equal(model.total(), Int128(12))
    assert_equal(model.mean(), Float64(3))
    assert_equal(model.central(2), Float64(10))
    assert_equal(model.central(3), Float64(0))
    assert_equal(model.central(4), Float64(34))
    assert_equal(exact_comoment(values, [2, 4, 8, 10], valid), Float64(20))
    _near(model.central(2), model.central(2), model.moment_budget(2))
    with assert_raises():
        _near(model.central(2) + 1, model.central(2), model.moment_budget(2))
    with assert_raises():
        _near(Float64(0) / Float64(0), model.central(2), model.moment_budget(2))


def _int_partition(
    values: List[Int64], valid: List[Bool], start: Int, end: Int
) -> IntSumState:
    var state = IntSumState()
    for row in range(start, end):
        if valid[row]:
            state.add(values[row])
    return state^


def test_integer_state_merge_defers_overflow_until_exact_finalization() raises:
    var cases: List[List[Int64]] = [
        [Int64.MAX, Int64.MAX, -Int64.MAX, -Int64.MAX, 7, Int64.MIN],
        [Int64.MIN, Int64.MIN, Int64.MAX, Int64.MAX, 2, Int64.MAX],
        [Int64.MAX, 1],
        [Int64.MIN, -1],
        [],
    ]
    for values in cases:
        for mode in range(3):
            var valid = List[Bool]()
            var total = Int128(0)
            var count = Int64(0)
            for row in range(len(values)):
                var yes = mode == 0 or (mode == 1 and row != len(values) - 1)
                valid.append(yes)
                if yes:
                    total += Int128(values[row])
                    count += 1
            for a in range(len(values) + 1):
                for b in range(a, len(values) + 1):
                    var first = _int_partition(values, valid, 0, a)
                    var middle = _int_partition(values, valid, a, b)
                    var last = _int_partition(values, valid, b, len(values))
                    var left = first.copy()
                    left.merge(middle)
                    left.merge(last)
                    var tail = middle.copy()
                    tail.merge(last)
                    var right = first.copy()
                    right.merge(tail)
                    var reverse = last.copy()
                    reverse.merge(middle)
                    reverse.merge(first)
                    for state in [left.copy(), right.copy(), reverse.copy()]:
                        assert_equal(state.total, total)
                        assert_equal(state.count, count)
                        if total > Int128(Int64.MAX) or total < Int128(
                            Int64.MIN
                        ):
                            with assert_raises():
                                _ = state.value()
                        else:
                            assert_equal(state.value(), Int64(total))
    var wide = IntSumState()
    wide.add_wide(Int128(UInt64.MAX))
    wide.add_wide(Int128(UInt64.MAX))
    assert_equal(wide.total, Int128(UInt64.MAX) * 2)
    assert_equal(wide.count, Int64(2))
    with assert_raises():
        _ = wide.value()


def _logic_partition(states: List[Int], start: Int, end: Int) -> LogicState:
    var state = LogicState()
    for row in range(start, end):
        # Null rows deliberately carry true as a poison Boolean payload.
        state.add(states[row] >= 0, states[row] != 0)
    return state^


def _assert_logic(state: LogicState, values: List[Int]) raises:
    var trues = 0
    var falses = 0
    var nulls = 0
    for value in values:
        trues += 1 if value == 1 else 0
        falses += 1 if value == 0 else 0
        nulls += 1 if value < 0 else 0
    assert_equal(state.saw_true, trues > 0)
    assert_equal(state.saw_false, falses > 0)
    assert_equal(state.saw_null, nulls > 0)
    for ignore in [False, True]:
        var any = state.any(ignore)
        var all = state.all(ignore)
        assert_equal(Bool(any), trues > 0 or ignore or nulls == 0)
        assert_equal(Bool(all), falses > 0 or ignore or nulls == 0)
        if any:
            assert_equal(any.value(), trues > 0)
        if all:
            assert_equal(all.value(), falses == 0)


def test_logic_state_merge_obeys_kleene_truth_tables_and_empty_identities() raises:
    _assert_logic(LogicState(), [])
    for encoded in range(81):
        var digits = encoded
        var values = List[Int]()
        for _ in range(4):
            values.append(digits % 3 - 1)
            digits //= 3
        for a in range(5):
            for b in range(a, 5):
                var first = _logic_partition(values, 0, a)
                var middle = _logic_partition(values, a, b)
                var last = _logic_partition(values, b, 4)
                var left = first.copy()
                left.merge(middle)
                left.merge(last)
                _assert_logic(left, values)
                var tail = middle.copy()
                tail.merge(last)
                var right = first.copy()
                right.merge(tail)
                _assert_logic(right, values)
                var reordered = last.copy()
                reordered.merge(first)
                reordered.merge(middle)
                _assert_logic(reordered, values)


def test_float_sum_state_merge_has_a_forward_error_budget_and_nonfinite_policy() raises:
    var values: List[Int64] = [
        10_000_000_000_000_000,
        1,
        -10_000_000_000_000_000,
        3,
        -2,
        7,
    ]
    var exact = Int128(0)
    var magnitude = Float64(0)
    for value in values:
        exact += Int128(value)
        magnitude += abs(Float64(value))
    var steps = Float64(2 * len(values) + 1)
    var epsilon = Float64(2.220446049250313e-16)
    var budget = steps * epsilon / (1 - steps * epsilon) * magnitude
    for a in range(len(values) + 1):
        for b in range(a, len(values) + 1):
            var first = FloatSumState()
            var middle = FloatSumState()
            var last = FloatSumState()
            for row in range(len(values)):
                if row < a:
                    first.add(Float64(values[row]))
                elif row < b:
                    middle.add(Float64(values[row]))
                else:
                    last.add(Float64(values[row]))
            var left = first.copy()
            left.merge(middle)
            left.merge(last)
            var tail = middle.copy()
            tail.merge(last)
            var right = first.copy()
            right.merge(tail)
            for state in [left.copy(), right.copy()]:
                assert_equal(state.count, Int64(len(values)))
                _near(state.total, Float64(exact), budget)
    with assert_raises():
        _near(Float64(exact) + budget * 2, Float64(exact), budget)
    var inf = Float64(1) / Float64(0)
    var nan = Float64(0) / Float64(0)
    var cases: List[List[Float64]] = [
        [inf, 1, 2],
        [-inf, 1],
        [inf, -inf],
        [nan, 1],
    ]
    var expected: List[Float64] = [inf, -inf, nan, nan]
    for example in range(len(cases)):
        for cut in range(len(cases[example]) + 1):
            var first = FloatSumState()
            var last = FloatSumState()
            for row in range(len(cases[example])):
                if row < cut:
                    first.add(cases[example][row])
                else:
                    last.add(cases[example][row])
            first.merge(last)
            assert_equal(first.count, Int64(len(cases[example])))
            assert_float(first.total, expected[example])


def _new_reducer(source: Series, op: Int, groups: Int) -> Reducer:
    return Reducer(
        op,
        source.dtype().physical(),
        groups,
        0,
        0,
        logical=Optional(source.dtype()),
    )


def _ordered_partition(
    source: Series, op: Int, start: Int, end: Int, groups: List[Int], count: Int
) raises -> Reducer:
    var state = _new_reducer(source, op, count)
    state.update(source.slice(start, end - start), start, True, groups)
    return state^


def _assert_ordered(
    state: Reducer,
    source: Series,
    op: Int,
    groups: List[Int],
    valid: List[Bool],
    group_count: Int,
) raises:
    var actual = state.finish()
    # Reducer returns physical cells; execution restores logical tags. Decimal
    # finish already restores its declared physical width and logical metadata.
    assert_true(
        actual.dtype()
        == (
            source.dtype() if source.dtype()
            .is_decimal() else source.dtype()
            .physical()
        )
    )
    assert_equal(len(actual), group_count)
    actual = actual.with_dtype(source.dtype())
    assert_true(actual.dtype() == source.dtype())
    for group in range(group_count):
        var chosen = -1
        for row in range(len(source)):
            if groups[row] == group and (chosen < 0 or op == LAST):
                chosen = row
        assert_equal(state.seen[group], chosen >= 0)
        assert_equal(
            actual.get(group).is_null(), chosen < 0 or not valid[chosen]
        )
        if chosen >= 0:
            assert_value(actual.get(group), source.get(chosen))


def test_ordered_first_last_states_keep_terminal_nulls_and_logical_types() raises:
    var groups: List[Int] = [1, 0, 1, 2, 0]
    for mode in range(4):
        var valid = List[Bool](length=5, fill=mode != 3)
        if mode == 1:
            valid[0] = False
        if mode == 2:
            valid[4] = False
        var columns = typed_ordered_columns(valid)
        for original in columns:
            for layout in range(3):
                var frame = DataFrame([original.copy()])
                frame = (
                    frame.copy() if layout
                    == 0 else offset_layout(frame) if layout
                    == 1 else chunk_layout(frame)
                )
                var source = frame.column("x")
                for op in [FIRST, LAST]:
                    for a in range(6):
                        for b in range(a, 6):
                            var first = _ordered_partition(
                                source, op, 0, a, groups, 4
                            )
                            var middle = _ordered_partition(
                                source, op, a, b, groups, 4
                            )
                            var last = _ordered_partition(
                                source, op, b, 5, groups, 4
                            )
                            var left = _new_reducer(source, op, 4)
                            left.merge(first)
                            left.merge(middle)
                            left.merge(last)
                            _assert_ordered(left, source, op, groups, valid, 4)
                            # Preserve partition order while changing association.
                            middle.merge(last)
                            var right = _new_reducer(source, op, 4)
                            right.merge(first)
                            right.merge(middle)
                            _assert_ordered(right, source, op, groups, valid, 4)


def _mapped_reducer(
    source: Series, op: Int, begin: Int, end: Int, groups: List[Int]
) raises -> Reducer:
    var state = _new_reducer(source, op, 3)
    state.update(source.slice(begin, end - begin), begin, True, groups)
    return state^


def _mapped_expected(op: Int) raises -> Series:
    if op == SUM:
        return Series("", Column[Int64]([11, 2, 0]))
    if op == COUNT:
        return Series("", Column[Int64]([3, 2, 0]))
    if op == NULL_COUNT:
        return Series("", Column[Int64]([0, 2, 0]))
    if op == LEN:
        return Series("", Column[Int64]([3, 4, 0]))
    if op == N_UNIQUE:
        return Series("", Column[Int64]([2, 2, 0]))
    if op == MIN or op == MAX:
        return Series(
            "",
            Column[Int64](
                [Int64(3 if op == MIN else 5), 1, 0], [True, True, False]
            ),
        )
    if op == ARG_MIN or op == ARG_MAX:
        return Series(
            "",
            Column[UInt32](
                [UInt32(0 if op == ARG_MIN else 2), 1, 0], [True, True, False]
            ),
        )
    return Series(
        "", Column[Float64]([Float64(11) / 3, 1, 0], [True, True, False])
    )


def test_reducer_group_remapping_and_source_selection_preserve_complete_results() raises:
    var source = Series(
        "x",
        Column[Int64](
            [3, 99, 3, 1, 5, 99, 1],
            [True, False, True, True, True, False, True],
        ),
    )
    var groups: List[Int] = [0, 1, 0, 1, 0, 1, 1]
    var reversed: List[Int] = [1, 0, 1, 0, 1, 0, 0]
    var mapping: List[Int] = [1, 0, 2]
    for op in [
        SUM,
        MEAN,
        COUNT,
        NULL_COUNT,
        LEN,
        N_UNIQUE,
        MIN,
        MAX,
        ARG_MIN,
        ARG_MAX,
    ]:
        var expected = _mapped_expected(op)
        for split in range(len(source) + 1):
            var left = _mapped_reducer(source, op, 0, split, groups)
            var right = _mapped_reducer(
                source, op, split, len(source), reversed
            )
            left.merge(right, groups=mapping)
            var actual = left.finish()
            assert_true(actual.dtype() == expected.dtype())
            assert_equal(len(actual), 3)
            for row in range(3):
                assert_value(actual.get(row), expected.get(row))
            var selected = _new_reducer(source, op, 3)
            selected.merge(left, sources=[1, 0, 2])
            var reordered = selected.finish()
            assert_true(reordered.dtype() == expected.dtype())
            for row in range(3):
                assert_value(reordered.get(row), expected.get([1, 0, 2][row]))


def test_float_extrema_state_merge_preserves_first_zero_ties_and_nan_order() raises:
    var nan = Float64(0) / Float64(0)
    var values: List[List[Float64]] = [
        [-0.0, 0.0, nan, 3],
        [0.0, -0.0, nan, 3],
        [nan, nan, nan, nan],
        [nan, -0.0, 0.0, nan],
    ]
    var minima: List[Float64] = [-0.0, 0.0, nan, -0.0]
    for example in range(len(values)):
        var source = Series("x", Column[Float64](values[example].copy()))
        for op in [MIN, MAX]:
            for a in range(len(source) + 1):
                for b in range(a, len(source) + 1):
                    var groups = List[Int](length=len(source), fill=0)
                    var first = _ordered_partition(source, op, 0, a, groups, 1)
                    var middle = _ordered_partition(source, op, a, b, groups, 1)
                    var last = _ordered_partition(
                        source, op, b, len(source), groups, 1
                    )
                    var left = _new_reducer(source, op, 1)
                    left.merge(first)
                    left.merge(middle)
                    left.merge(last)
                    middle.merge(last)
                    var right = _new_reducer(source, op, 1)
                    right.merge(first)
                    right.merge(middle)
                    var expected = minima[example] if op == MIN else nan
                    assert_float(left.finish().get(0).float64(), expected)
                    assert_float(right.finish().get(0).float64(), expected)
