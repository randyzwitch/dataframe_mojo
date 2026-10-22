"""Whole duration units retain signed storage semantics across native consumers."""
# contract-group: execution.backends
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.accelerator.gpu.runtime import (
    capabilities as _capabilities,
)
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    Series,
    Column,
    Expr,
    LazyFrame,
    DataType,
    col,
    lit,
    when,
)
from dataframe.expr import DT_TOTAL
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal import require_metal
from tests.support.coverage_compare import assert_frame


def units() -> List[String]:
    return [
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ]


def expressions() -> List[Expr]:
    var result = List[Expr]()
    for unit in units():
        result.append(col("d").dt().total(unit).alias(unit))
    return result^


def fixture(unit: String) raises -> DataFrame:
    var raw: List[Int64] = [
        0,
        Int64.MIN,
        Int64.MIN + 1,
        -1000000001,
        -1000000,
        -1001,
        -1000,
        -999,
        -1,
        0,
        1,
        999,
        1000,
        1001,
        1000000,
        1000000001,
        Int64.MAX - 1,
        Int64.MAX,
    ]
    var seed = UInt64(0x4455524154494F4E)
    for _ in range(512):
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        raw.append(bitcast[DType.int64](seed))
    var valid = List[Bool]()
    var keys = List[Int32]()
    for i in range(len(raw)):
        valid.append(i < 18 or i % 11 != 0)
        keys.append(Int32(i % 3))
    return DataFrame(
        [
            Series("d", Column[Int64](raw^, valid)).with_dtype(
                DataType.duration(unit)
            ),
            Series("k", Column[Int32](keys^)),
        ]
    ).slice(1, 528)


def expected(frame: DataFrame) raises -> DataFrame:
    var source = frame.column("d").int64()
    var nanoseconds_per_tick = Int128(1000000000) // Int128(
        frame.column("d").dtype().per_second()
    )
    var denominators: List[Int128] = [
        86400000000000,
        3600000000000,
        60000000000,
        1000000000,
        1000000,
        1000,
        1,
    ]
    var names = units()
    var columns = List[Series]()
    for unit in range(len(names)):
        var values = List[Int64]()
        var valid = List[Bool]()
        for row in range(frame.height()):
            valid.append(source.is_valid(row))
            var exact = Int128(source._get(row)) * nanoseconds_per_tick
            exact = exact // denominators[unit] if exact >= 0 else -(
                (-exact) // denominators[unit]
            )
            var low = UInt64(exact & Int128(UInt64.MAX))
            values.append(bitcast[DType.int64](low))
        columns.append(Series(names[unit], Column[Int64](values^, valid)))
    return DataFrame(columns^)


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    assert_frame(
        query.collect(accelerator=runtime),
        query.collect(engine="cpu", streaming=False),
    )


def test_duration_totals_capability_metadata_units_and_minimum_cpu_sign() raises:
    assert_true(not RowCapabilities().duration_totals)
    for unit in List[String](["ms", "us", "ns"]):
        var frame = fixture(unit)
        var query = frame.lazy().select_exprs(expressions())
        var caps = _capabilities()
        caps.duration_totals = False
        with assert_raises(contains="resident expression operation"):
            _ = lower_rows(query, caps)
        var plan = lower_rows(query, _capabilities())
        var conversions = 0
        for i in range(0, len(plan.code), 4):
            if plan.code[i] == DT_TOTAL:
                conversions += 1
                assert_true(plan.code[i + 2] > 0)
                assert_true(plan.code[i + 3] == 0 or plan.code[i + 3] == 1)
        assert_equal(conversions, 7)
        _ = Program(plan)
        assert_true(
            "unsupported" not in MetalRuntime(device_id=999999).describe(query)
        )
        assert_frame(
            query.collect(engine="cpu", streaming=False), expected(frame)
        )
    with assert_raises(contains="unknown total unit"):
        _ = lower_rows(
            fixture("ns").lazy().select(col("d").dt().total("weeks")),
            _capabilities(),
        )
    var numbers = DataFrame([Series("d", Column[Int64]([1]))])
    with assert_raises(contains="duration expression"):
        _ = lower_rows(
            numbers.lazy().select(col("d").dt().total_seconds()),
            _capabilities(),
        )


# requires: metal
def test_native_metal_duration_totals_all_storage_units_full_domain_and_nulls() raises:
    var runtime = require_metal()
    for unit in List[String](["ms", "us", "ns"]):
        var frame = fixture(unit)
        var query = frame.lazy().select_exprs(expressions())
        assert_frame(query.collect(accelerator=runtime), expected(frame))
        for rows in List[Int]([0, 1, 17, 257]):
            var selected = frame.head(rows)
            assert_frame(
                selected.lazy()
                .select_exprs(expressions())
                .collect(accelerator=runtime),
                expected(selected),
            )
        var nulls = DataFrame(
            [
                Series(
                    "d",
                    Column[Int64](
                        [Int64.MIN, 1, Int64.MAX], [False, False, False]
                    ),
                ).with_dtype(DataType.duration(unit))
            ]
        )
        assert_frame(
            nulls.lazy()
            .select_exprs(expressions())
            .collect(accelerator=runtime),
            expected(nulls),
        )


# requires: metal
def test_native_metal_duration_totals_conditions_filters_groups_windows_and_joins() raises:
    var runtime = require_metal()
    var frame = fixture("ns")
    check(
        frame.lazy().select(
            when(col("k") == lit(Int32(1)))
            .then(col("d").dt().total_seconds())
            .otherwise(lit(Int64(7)))
            .alias("v")
        ),
        runtime,
    )
    check(
        frame.lazy()
        .filter(col("d").dt().total_seconds() > lit(Int64(0)))
        .select_exprs(expressions())
        .head(17),
        runtime,
    )
    check(
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(col("d").dt().total_seconds().min().alias("m")),
        runtime,
    )
    check(
        frame.lazy().select(col("d").dt().total_milliseconds().max().over("k")),
        runtime,
    )
    check(
        frame.lazy().select(
            col("d").shift(1).dt().total_seconds().alias("shifted")
        ),
        runtime,
    )
    check(
        frame.lazy()
        .join(frame.head(3).lazy(), ["k"])
        .select(col("d").dt().total_seconds().alias("v")),
        runtime,
    )


# requires: metal
def test_native_metal_duration_totals_profiles_cost_budget_and_runtime_reuse() raises:
    var runtime = require_metal()
    var query = fixture("ms").lazy().select_exprs(expressions())
    var profiled = query.profile(accelerator=runtime)
    assert_frame(profiled[0], query.collect(engine="cpu", streaming=False))
    assert_true(profiled[1].height() > 0)
    var estimate = runtime.estimate_auto(query)
    assert_true(";duration-totals-v1" in estimate.family)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    check(query, runtime)
