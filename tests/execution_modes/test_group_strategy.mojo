"""Streaming aggregation decisions preserve scalar groups across budget boundaries."""
# contract-group: execution.modes
from std.math import abs
from std.os import getenv
from std.sys import stderr
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, LazyFrame, Series, col
from dataframe.lazy import (
    _stream_split_groups,
    _stream_collect_bytes,
)
from dataframe.frame import _stream_reductions
from tests.support.group_strategy_reference import (
    GroupSample,
    ScalarGroup,
    ScopedEnvironment,
    StrategyTrace,
)
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_float


def _source(
    rows: Int, groups: Int, stride: Int = 1, phase: Int = 0
) raises -> DataFrame:
    var k = List[Int64]()
    var kv = List[Bool]()
    var s = List[String]()
    var sv = List[Bool]()
    var dimension = List[Int64]()
    var integers = List[Int64]()
    var iv = List[Bool]()
    var floats = List[Float64]()
    var fv = List[Bool]()
    var words = List[String]()
    var wv = List[Bool]()
    for row in range(rows):
        var sample = GroupSample(row, groups, stride, phase)
        k.append(sample.key)
        kv.append(sample.key_valid)
        s.append(sample.string_key.copy())
        sv.append(sample.string_key_valid)
        dimension.append(sample.dimension)
        integers.append(sample.integer)
        iv.append(sample.integer_valid)
        floats.append(sample.floating)
        fv.append(sample.float_valid)
        words.append(sample.text.copy())
        wv.append(sample.text_valid)
    return DataFrame(
        [
            Series("s", Column[String](s^, sv)),
            Series("k", Column[Int64](k^, kv)),
            Series("d", Column[Int64](dimension^)),
            Series("i", Column[Int64](integers^, iv)),
            Series("f", Column[Float64](floats^, fv)),
            Series("w", Column[String](words^, wv)),
        ]
    )


def _expressions(median: Bool = False) -> List[Expr]:
    var expressions: List[Expr] = [
        col("i").sum().alias("sum"),
        col("i").count().alias("count"),
        col("i").mean().alias("mean"),
        col("i").n_unique().alias("i_unique"),
        col("f").n_unique().alias("f_unique"),
        col("w").n_unique().alias("w_unique"),
        col("i").len().alias("rows"),
        col("i").first().alias("first"),
        col("i").last().alias("last"),
        col("f").first().alias("f_first"),
        col("f").last().alias("f_last"),
    ]
    if median:
        expressions.insert(3, col("i").median().alias("median"))
    return expressions^


def _plan(
    frame: DataFrame, ordered: Bool, median: Bool = False
) raises -> LazyFrame:
    # The identity/selective join is a real public plan operation. Distinct
    # aggregates exclude the in-memory pipeline sink; the join excludes its
    # row-step-only eager shortcut. This reaches the streamed state strategy.
    var dimension = DataFrame([Series("d", Column[Int64]([0]))])
    return (
        frame.lazy()
        .join(dimension.lazy(), left_on=["d"], right_on=["d"])
        .group_by(["s", "k"], maintain_order=ordered)
        .agg(_expressions(median))
    )


def _assert_groups(
    actual: DataFrame,
    rows: Int,
    groups: Int,
    ordered: Bool,
    stride: Int = 1,
    phase: Int = 0,
    median: Bool = False,
) raises:
    var expected = List[ScalarGroup](length=groups, fill=ScalarGroup())
    var order = List[Int]()
    for row in range(rows):
        var sample = GroupSample(row, groups, stride, phase)
        if sample.dimension != 0:
            continue
        if expected[sample.group].rows == 0:
            order.append(sample.group)
        expected[sample.group].add(sample)
    var names: List[String] = [
        "s",
        "k",
        "sum",
        "count",
        "mean",
        "i_unique",
        "f_unique",
        "w_unique",
        "rows",
        "first",
        "last",
        "f_first",
        "f_last",
    ]
    if median:
        names.insert(5, "median")
    assert_equal(actual.columns(), names)
    assert_equal(actual.height(), len(order))
    assert_true(actual.column("s").dtype() == DataType.STRING)
    for name in [
        "k",
        "sum",
        "count",
        "i_unique",
        "f_unique",
        "w_unique",
        "rows",
        "first",
        "last",
    ]:
        assert_true(actual.column(name).dtype() == DataType.INT64)
    var float_names: List[String] = ["mean", "f_first", "f_last"]
    if median:
        float_names.append("median")
    for name in float_names:
        assert_true(actual.column(name).dtype() == DataType.FLOAT64)
    var seen = List[Bool](length=groups, fill=False)
    for row in range(actual.height()):
        var group = Int(actual.item(row, "k").int64()) if not actual.item(
            row, "k"
        ).is_null() else 0
        assert_true(0 <= group and group < groups)
        assert_true(not seen[group])
        seen[group] = True
        if ordered:
            assert_equal(group, order[row])
        ref model = expected[group]
        assert_true(model.rows > 0)
        assert_equal(actual.item(row, "s").is_null(), group % 11 == 0)
        if group % 11 != 0:
            assert_equal(
                actual.item(row, "s").string(), "é:" + String(group % 7)
            )
        assert_equal(actual.item(row, "sum").int64(), Int64(model.total))
        assert_equal(actual.item(row, "count").int64(), Int64(model.count))
        assert_equal(actual.item(row, "rows").int64(), Int64(model.rows))
        assert_equal(
            actual.item(row, "i_unique").int64(),
            Int64(len(model.integers) + Int(model.integer_null)),
        )
        assert_equal(
            actual.item(row, "f_unique").int64(),
            Int64(len(model.floats) + Int(model.float_null)),
        )
        assert_equal(
            actual.item(row, "w_unique").int64(),
            Int64(len(model.words) + Int(model.word_null)),
        )
        assert_equal(actual.item(row, "mean").is_null(), model.count == 0)
        if median:
            assert_equal(actual.item(row, "median").is_null(), model.count == 0)
        if model.count:
            var mean = Float64(model.total) / Float64(model.count)
            assert_true(
                abs(actual.item(row, "mean").float64() - mean)
                <= 2e-12 * (1 + abs(mean))
            )
            if median:
                assert_equal(
                    actual.item(row, "median").float64(), model.median()
                )
        for which in range(2):
            var name = "first" if which == 0 else "last"
            var valid = model.first_valid if which == 0 else model.last_valid
            assert_equal(actual.item(row, name).is_null(), not valid)
            if valid:
                assert_equal(
                    actual.item(row, name).int64(),
                    model.first if which == 0 else model.last,
                )
            name = "f_first" if which == 0 else "f_last"
            valid = (
                model.float_first_valid if which
                == 0 else model.float_last_valid
            )
            assert_equal(actual.item(row, name).is_null(), not valid)
            if valid:
                assert_float(
                    actual.item(row, name).float64(),
                    model.float_first if which == 0 else model.float_last,
                )
    for group in order:
        assert_true(seen[group])


def _assert_profile(
    report: DataFrame, inputs: Int, outputs: Int, executor: String = "streaming"
) raises:
    var found = 0
    for row in range(report.height()):
        if report.item(row, "operator").string().startswith("GROUP_BY"):
            found += 1
            assert_equal(report.item(row, "executor").string(), executor)
            assert_equal(report.item(row, "executions").int64(), Int64(1))
            assert_equal(report.item(row, "input_rows").int64(), Int64(inputs))
            assert_equal(
                report.item(row, "output_rows").int64(), Int64(outputs)
            )
    assert_equal(found, 1)


def test_group_strategy_scalar_models_anchor_distinct_classes_and_median() raises:
    var sample = GroupSample(0, 7)
    assert_equal(sample.group, 4)
    assert_equal(sample.key, Int64(4))
    assert_true(not sample.integer_valid)
    sample = GroupSample(3, 7)
    assert_equal(sample.group, 0)
    assert_true(not sample.key_valid and not sample.string_key_valid)
    sample = GroupSample(127, 32, 128, 1)
    assert_equal(sample.group, 29)
    assert_equal(sample.dimension, Int64(0))
    sample = GroupSample(126, 32, 128, 1)
    assert_equal(sample.dimension, Int64(127))
    var model = ScalarGroup()
    # Scalar reference anchor is authored independently of dataframe output.
    for row in range(5):
        var sample = GroupSample(row, 7)
        sample.integer = Int64([-3, 1, 1, 3, 999][row])
        sample.integer_valid = row != 4
        sample.floating = [
            Float64(0) / Float64(0),
            Float64(0) / Float64(0),
            -0.0,
            0.0,
            999.0,
        ][row]
        sample.float_valid = row != 4
        sample.text = ["é", "é", "", "a\x00b", "hidden"][row]
        sample.text_valid = row != 4
        model.add(sample)
    assert_equal(model.total, Int128(2))
    assert_equal(model.count, 4)
    assert_equal(model.median(), 1.0)
    assert_equal(len(model.integers) + Int(model.integer_null), 4)
    assert_equal(len(model.floats) + Int(model.float_null), 3)
    assert_equal(len(model.words) + Int(model.word_null), 4)
    assert_equal(model.first, Int64(-3))
    assert_true(not model.last_valid)


def test_group_settings_and_scoped_environment_restore() raises:
    var captured = StrategyTrace()
    print("literal trace anchor", file=stderr)
    assert_equal(captured.text(), "literal trace anchor\n")
    _ = captured^
    var previous = getenv(
        "DATAFRAME_TEST_GROUP_ENV", "\x00missing group setting"
    )
    var outer = ScopedEnvironment("DATAFRAME_TEST_GROUP_ENV", "sentinel")
    var previously_existed = outer.existed
    var empty = ScopedEnvironment("DATAFRAME_TEST_GROUP_ENV", "")
    assert_equal(
        getenv("DATAFRAME_TEST_GROUP_ENV", "\x00missing group setting"), ""
    )
    with assert_raises(contains="deliberate environment unwind"):
        var inner = ScopedEnvironment("DATAFRAME_TEST_GROUP_ENV", "changed")
        _fail_with_live_setting(inner)
        _ = inner^
    assert_equal(
        getenv("DATAFRAME_TEST_GROUP_ENV", "\x00missing group setting"), ""
    )
    _ = empty^
    assert_equal(
        getenv("DATAFRAME_TEST_GROUP_ENV", "\x00missing group setting"),
        "sentinel",
    )
    var split = ScopedEnvironment("DATAFRAME_STREAM_SPLIT_GROUPS", "32")
    assert_equal(_stream_split_groups(), 32)
    var budget = ScopedEnvironment("DATAFRAME_STREAM_COLLECT_BYTES", "1")
    assert_equal(_stream_collect_bytes(), 1)
    _ = budget^
    _ = split^
    _ = outer^
    assert_equal(
        getenv("DATAFRAME_TEST_GROUP_ENV", "\x00missing group setting"),
        previous,
    )
    var restored = ScopedEnvironment(
        "DATAFRAME_TEST_GROUP_ENV", "restore probe"
    )
    assert_equal(restored.existed, previously_existed)
    assert_equal(restored.previous, previous)
    _ = restored^


def _fail_with_live_setting(setting: ScopedEnvironment) raises:
    assert_equal(
        getenv("DATAFRAME_TEST_GROUP_ENV", "\x00missing group setting"),
        "changed",
    )
    raise Error("deliberate environment unwind")


def test_nonmergeable_median_materializes_without_claiming_stream_state_coverage() raises:
    assert_equal(_stream_reductions(_expressions()), True)
    assert_equal(_stream_reductions(_expressions(True)), False)
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var split = ScopedEnvironment("DATAFRAME_STREAM_SPLIT_GROUPS", "32")
    var budget = ScopedEnvironment("DATAFRAME_STREAM_COLLECT_BYTES", "1")
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "1")
    var observed = StrategyTrace()
    var frame = _source(517, 32)
    var profiled = _plan(frame, True, True).profile(
        optimize=False, batch_size=33, engine="cpu"
    )
    _assert_groups(profiled[0], 517, 32, True, median=True)
    _assert_profile(profiled[1], 517, 32, "eager")
    var paths = observed.text()
    assert_equal("dataframe-path: lazy.collect_group_by" in paths, False)
    _ = observed^
    _ = workers^
    _ = budget^
    _ = split^
    _ = trace^


def test_streamed_collect_budget_boundaries_preserve_complete_composite_groups() raises:
    var split = ScopedEnvironment("DATAFRAME_STREAM_SPLIT_GROUPS", "32")
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "1")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var raw = _source(517, 32)
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        # Small collection thresholds and a comfortably large threshold must
        # preserve the same groups across independent physical layouts.
        for bytes in [1, 4095, 4096, 4097, 65536, 1 << 30]:
            var budget = ScopedEnvironment(
                "DATAFRAME_STREAM_COLLECT_BYTES", String(bytes)
            )
            var observed = StrategyTrace()
            for optimize in [False, True]:
                var plan = _plan(frame, True)
                _assert_groups(
                    plan.collect(
                        optimize=optimize,
                        streaming=True,
                        batch_size=33,
                        engine="cpu",
                    ),
                    517,
                    32,
                    True,
                )
            var paths = observed.text()
            # Backing capacities and sharing vary by layout. Only the minimum
            # and comfortably large thresholds guarantee a particular route.
            # Every intermediate threshold must still produce complete groups.
            if bytes == 1 or bytes == 1 << 30:
                assert_equal(
                    "dataframe-path: lazy.collect_group_by_bounded\n" in paths,
                    bytes == 1,
                )
                assert_equal(
                    "dataframe-path: lazy.collect_group_by\n" in paths,
                    bytes == 1 << 30,
                )
            _ = observed^
            _ = budget^
    var unlimited = ScopedEnvironment(
        "DATAFRAME_STREAM_COLLECT_BYTES", String(1 << 30)
    )
    var profiled = _plan(raw, True).profile(
        optimize=False, batch_size=33, engine="cpu"
    )
    _assert_groups(profiled[0], 517, 32, True)
    _assert_profile(profiled[1], 517, 32)
    _ = unlimited^
    _ = trace^
    _ = workers^
    _ = split^


def test_collect_threshold_counts_long_payloads_beyond_row_width_estimates() raises:
    var split = ScopedEnvironment("DATAFRAME_STREAM_SPLIT_GROUPS", "32")
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "1")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var raw = _source(517, 32)
    var words = List[String]()
    for i in range(raw.height()):
        words.append(String("x") * 4096 + String(i))
    var frame = raw.with_column(Series("w", Column[String](words^)))
    var plan = _plan(frame, True)
    var expected = plan.collect(streaming=False, optimize=False, engine="cpu")
    # The former 88-byte row estimate totaled under 64 KiB for this input;
    # its retained text alone exceeds 2 MiB.
    for bytes in [65536, 1 << 28]:
        var budget = ScopedEnvironment(
            "DATAFRAME_STREAM_COLLECT_BYTES", String(bytes)
        )
        var observed = StrategyTrace()
        var actual = plan.collect(
            optimize=False, streaming=True, batch_size=33, engine="cpu"
        )
        assert_true(actual.equals(expected))
        assert_equal(
            "dataframe-path: lazy.collect_group_by_bounded\n"
            in observed.text(),
            bytes == 65536,
        )
        _ = observed^
        _ = budget^
    _ = trace^
    _ = workers^
    _ = split^


def test_selective_joins_collect_their_rows_for_the_eager_group_by() raises:
    var split = ScopedEnvironment("DATAFRAME_STREAM_SPLIT_GROUPS", "32")
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "1")
    var budget = ScopedEnvironment(
        "DATAFRAME_STREAM_COLLECT_BYTES", String(1 << 30)
    )
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    for groups in [7, 32]:
        var observed = StrategyTrace()
        var raw = _source(4096, groups, 128, 1)
        # One row per 128 survives; only 32 rows reach the reducer. The
        # first kept row is 127: initial batches can therefore be empty.
        for batch in [16, 127, 128, 257]:
            for ordered in [False, True]:
                var actual = _plan(raw, ordered).collect(
                    optimize=False,
                    streaming=True,
                    batch_size=batch,
                    engine="cpu",
                )
                _assert_groups(actual, 4096, groups, ordered, 128, 1)
        var profiled = _plan(raw, True).profile(
            optimize=False, batch_size=127, engine="cpu"
        )
        _assert_groups(profiled[0], 4096, groups, True, 128, 1)
        _assert_profile(profiled[1], 32, groups)
        var paths = observed.text()
        # The ordered grouping over a join runs on the batches, which
        # collect their rows for one eager group-by.
        assert_true("dataframe-path: lazy.collect_group_by\n" in paths)
        assert_equal(
            "dataframe-path: lazy.collect_group_by_bounded\n" in paths,
            False,
        )
        _ = observed^
    _ = trace^
    _ = budget^
    _ = split^
    _ = workers^


def test_default_split_threshold_bounded_and_unordered_results_match_full_scalar_rows() raises:
    var split = ScopedEnvironment("DATAFRAME_STREAM_SPLIT_GROUPS", "4096")
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "2")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    assert_equal(_stream_split_groups(), 4096)
    var raw = _source(8193, 4097)
    for bytes in [1, 1 << 30]:
        var budget = ScopedEnvironment(
            "DATAFRAME_STREAM_COLLECT_BYTES", String(bytes)
        )
        var observed = StrategyTrace()
        for ordered in [False, True]:
            for batch in [4095, 4096, 4097]:
                _assert_groups(
                    _plan(raw, ordered).collect(
                        optimize=False,
                        streaming=True,
                        batch_size=batch,
                        engine="cpu",
                    ),
                    8193,
                    4097,
                    ordered,
                )
        var paths = observed.text()
        assert_equal(
            "dataframe-path: lazy.collect_group_by_bounded\n" in paths,
            bytes == 1,
        )
        assert_equal(
            "dataframe-path: lazy.collect_group_by\n" in paths,
            bytes != 1,
        )
        _ = observed^
        _ = budget^
    _ = trace^
    _ = split^
    _ = workers^
