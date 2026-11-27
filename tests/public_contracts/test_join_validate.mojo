"""Join validate: repeated keys on a side the option names raise, eager and lazy."""
# contract-group: table.join
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, Series, StringColumn, col
from tests.support.group_strategy_reference import ScopedEnvironment


def _ints(name: String, values: List[Int64]) raises -> Series:
    return Series(name, Column[Int64](values.copy()))


def _frame(keys: List[Int64], name: String) raises -> DataFrame:
    var payload = List[Int64]()
    for i in range(len(keys)):
        payload.append(Int64(i))
    return DataFrame([_ints("k", keys), _ints(name, payload)])


def _join(
    left: DataFrame,
    right: DataFrame,
    how: String,
    validate: String,
    lazy: Bool,
) raises -> DataFrame:
    if lazy:
        return (
            left.lazy()
            .join(right.lazy(), on="k", how=how, validate=validate)
            .collect()
        )
    return left.join(right, on="k", how=how, validate=validate)


def test_validate_passes_unique_sides_and_keeps_the_join_result() raises:
    """A side with unique keys passes, and the result is the join's own;
    repeats on a side the option does not name are fine."""
    var many = _frame([1, 1, 2, 3], "x")
    var one = _frame([1, 2, 4], "y")
    for threads in ["1", "8"]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
        for lazy in [False, True]:
            for how in ["inner", "left", "right", "full"]:
                var plain = many.join(one, on="k", how=how).sort(
                    ["x", "y"], nulls_last=True
                )
                var checked = _join(many, one, how, "m:1", lazy).sort(
                    ["x", "y"], nulls_last=True
                )
                assert_true(checked.equals(plain), how)
                var flipped = _join(one, many, how, "1:m", lazy)
                assert_equal(
                    flipped.height(),
                    one.join(many, on="k", how=how).height(),
                    how,
                )
                _ = _join(one, one, how, "1:1", lazy)
                _ = _join(many, many, how, "m:m", lazy)
        _ = workers^


def test_validate_raises_naming_the_side_and_a_repeated_key() raises:
    var many = _frame([5, 1, 2, 1], "x")
    var one = _frame([1, 2, 4], "y")
    for threads in ["1", "8"]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
        for lazy in [False, True]:
            for how in ["inner", "left", "right", "full"]:
                with assert_raises(
                    contains=(
                        "join keys did not fulfill m:1 validation: the right"
                        " input has key (k=1) more than once"
                    )
                ):
                    _ = _join(one, many, how, "m:1", lazy)
                with assert_raises(
                    contains=(
                        "join keys did not fulfill 1:m validation: the left"
                        " input has key (k=1) more than once"
                    )
                ):
                    _ = _join(many, one, how, "1:m", lazy)
                with assert_raises(contains="1:1 validation: the left"):
                    _ = _join(many, one, how, "1:1", lazy)
                with assert_raises(contains="1:1 validation: the right"):
                    _ = _join(one, many, how, "1:1", lazy)
        _ = workers^


def test_null_keys_never_count_as_repeats() raises:
    var left = DataFrame(
        [
            Series(
                "k", Column[Int64]([1, 0, 0, 2], [True, False, False, True])
            ),
            _ints("x", [1, 2, 3, 4]),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64]([0, 2, 0], [False, True, False])),
            _ints("y", [1, 2, 3]),
        ]
    )
    for lazy in [False, True]:
        for how in ["inner", "left", "right", "full"]:
            var joined = _join(left, right, how, "1:1", lazy)
            assert_equal(
                joined.height(), left.join(right, on="k", how=how).height()
            )


def test_multiple_and_string_keys_report_every_key() raises:
    var left = DataFrame(
        [
            Series("s", StringColumn(["a", "b", "a"])),
            _ints("n", [1, 1, 1]),
            _ints("x", [1, 2, 3]),
        ]
    )
    var right = DataFrame(
        [
            Series("s", StringColumn(["a", "b"])),
            _ints("n", [1, 1]),
            _ints("y", [1, 2]),
        ]
    )
    for lazy in [False, True]:
        var plan = left.lazy().join(
            right.lazy(), on=["s", "n"], how="left", validate="m:1"
        )
        var joined = plan.collect() if lazy else left.join(
            right, on=["s", "n"], how="left", validate="m:1"
        )
        assert_equal(joined.height(), 3)
        with assert_raises(contains="the left input has key (s=a, n=1)"):
            if lazy:
                _ = (
                    left.lazy()
                    .join(right.lazy(), on=["s", "n"], validate="1:1")
                    .collect()
                )
            else:
                _ = left.join(right, on=["s", "n"], validate="1:1")


def test_large_inputs_find_one_repeat_on_many_workers() raises:
    """One repeated key among many rows, on the side each option names,
    built and probed by several workers; the key's place does not
    matter."""
    var n = 70_000
    var unique = List[Int64](capacity=n)
    for i in range(n):
        unique.append(Int64(i * 3))
    var repeated = unique.copy()
    repeated[n - 7] = repeated[11]
    var u = _frame(unique, "x")
    var r = _frame(repeated, "y")
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    for lazy in [False, True]:
        assert_equal(_join(u, u, "inner", "1:1", lazy).height(), n)
        with assert_raises(contains="the right input has key (k=33)"):
            _ = _join(u, r, "left", "m:1", lazy)
        with assert_raises(contains="the left input has key (k=33)"):
            _ = _join(r, u, "inner", "1:m", lazy)
    _ = workers^


def test_validate_in_a_larger_plan_checks_the_join_inputs() raises:
    """A validated join among other joins and a filter checks its own
    inputs: the plan keeps it where it was written."""
    var facts = _frame([1, 2, 2, 3], "x")
    var dims = _frame([1, 2, 3], "y")
    var more = _frame([1, 1, 2, 3], "z")
    for threads in ["1", "8"]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
        var ok = (
            facts.lazy()
            .join(dims.lazy(), on="k", validate="m:1")
            .join(more.lazy(), on="k")
            .filter(col("x") >= 0)
            .collect()
        )
        assert_equal(ok.height(), 5)
        with assert_raises(contains="m:1 validation"):
            _ = (
                facts.lazy()
                .join(dims.lazy(), on="k")
                .join(more.lazy(), on="k", validate="m:1")
                .collect()
            )
        _ = workers^


def test_validate_rejects_semi_anti_and_unknown_names() raises:
    var a = _frame([1, 2], "x")
    var b = _frame([1, 2], "y")
    for how in ["semi", "anti"]:
        with assert_raises(contains="does not apply to a " + how + " join"):
            _ = a.join(b, on="k", how=how, validate="1:1")
        with assert_raises(contains="does not apply to a " + how + " join"):
            _ = (
                a.lazy()
                .join(b.lazy(), on="k", how=how, validate="m:1")
                .collect()
            )
        # The default checks nothing, so it is allowed.
        _ = a.join(b, on="k", how=how, validate="m:m")
    with assert_raises(contains="validate must be"):
        _ = a.join(b, on="k", validate="one_to_one")
    with assert_raises(contains="validate must be"):
        _ = a.lazy().join(b.lazy(), on="k", validate="1:n").collect()
