"""Certified private IMPLODE row producers preserve checked AST fault order."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe_metal.accelerator import implode_ordered
from dataframe_metal.accelerator.lists import lower_terminal_implode
from dataframe_metal.accelerator.rows import RowPlan
from dataframe_metal.accelerator.joins import DAG_ROWS
from dataframe_metal.accelerator.gpu.dag_runtime import (
    preflight as dag_preflight,
)
from dataframe_metal.accelerator.gpu.runtime import (
    _preflight,
    capabilities,
    estimate_resources,
    execute_plan,
)
from dataframe_metal.accelerator.gpu.residency import Scope, Table, HostMatrix
from dataframe_metal.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout


def _cast() -> Expr:
    return col("x").cast(DataType.INT8).cast(DataType.INT64)


def _time() raises -> Expr:
    return col("t").cast(DataType.duration("ns")).cast(DataType.INT64)


def _frame(ticks: List[Int64], values: List[Int64]) raises -> DataFrame:
    return DataFrame(
        [
            Series("x", Column[Int64](values.copy())),
            Series("t", Column[Int64](ticks.copy())).with_dtype(
                DataType.duration("ms")
            ),
        ]
    )


def _plan(query: LazyFrame) raises -> RowPlan:
    var flattened = lower_terminal_implode(query, ordered_source=True)
    var dag = dag_preflight(flattened.query, capabilities())
    for task in dag.tasks:
        if task.kind == DAG_ROWS:
            var plan = _preflight(task.query, terminal_boundary=task.terminal)
            if plan.ordered_mode == 3:
                return plan^
    raise Error("expected a certified private row producer")


def _fails(query: LazyFrame, cpu: String, gpu: String) raises:
    assert_equal(_plan(query).ordered_mode, 3)
    with assert_raises(contains=cpu):
        _ = query.collect(engine="cpu", streaming=False, optimize=False)
    with assert_raises(contains=gpu):
        _ = query.collect(accelerator=MetalRuntime(), optimize=False)


# requires: metal
def test_mojo_gpu_ordered_implode_checked_categories_batches_and_occurrences() raises:
    var numbers = DataFrame(
        [
            Series("x", Column[Int64]([128])),
            Series("base", Column[Int64]([2])),
            Series("exp", Column[Int64]([-1])),
        ]
    )
    var power = col("base").pow(col("exp"))
    _fails(
        numbers.lazy().select((_cast() + power).implode()),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )
    _fails(
        numbers.lazy().select((power + _cast()).implode()),
        "nonnegative exponent",
        "nonnegative exponent",
    )
    var huge = Int128(9)
    for _ in range(37):
        huge *= Int128(10)
    var dtype = DataType.decimal(38, 0, 128)
    var decimals = DataFrame(
        [
            Series("x", Column[Int64]([128])),
            Series("d", Column[Int128]([huge])).with_dtype(dtype),
            Series("two", Column[Int128]([Int128(2)])).with_dtype(dtype),
        ]
    )
    var product = (col("d") * col("two")).cast(DataType.INT64)
    _fails(
        decimals.lazy().select((_cast() + product).implode()),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )
    _fails(
        decimals.lazy().select((product + _cast()).implode()),
        "decimal multiplication overflow",
        "decimal multiplication overflow",
    )
    var original = _frame([0, Int64.MAX], [128, 0])
    for layout in range(3):
        var input = (
            original.copy() if layout
            == 0 else offset_layout(original) if layout
            == 1 else chunk_layout(original)
        )
        _fails(
            input.lazy().select((_time() + _cast()).implode()),
            "temporal value overflows Int64",
            "temporal value overflows Int64",
        )
        _fails(
            input.lazy().select((_cast() + _time()).implode()),
            "cast from int64 to int8 failed",
            "strict cast failed",
        )
    var ticks = List[Int64](length=8193, fill=0)
    var values = List[Int64](length=8193, fill=0)
    ticks[8192] = Int64.MAX
    values[8191] = 128
    var frame = _frame(ticks, values)
    _fails(
        frame.lazy().select((_time() + _cast()).implode()),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )


# requires: metal
def test_mojo_gpu_ordered_implode_masks_empty_and_later_eager_states() raises:
    var frame = _frame([0, Int64.MAX], [128, 0]).with_column(
        Series("keep", Column[Bool]([False, True]))
    )
    _fails(
        frame.lazy().select(
            when(col("keep")).then(_time()).otherwise(_cast()).implode()
        ),
        "temporal value overflows Int64",
        "temporal value overflows Int64",
    )
    var predicate = col("x").cast(DataType.INT8) > lit(Int8(0))
    _fails(
        frame.lazy().select(
            when(predicate).then(_time()).otherwise(col("x")).implode()
        ),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )
    frame = frame.with_column(Series("keep", Column[Bool]([False, False])))
    var expression = (
        when(col("keep"))
        .then(_time() + _cast())
        .otherwise(col("x"))
        .implode()
        .alias("v")
    )
    var query = frame.lazy().select(expression.copy())
    var actual = query.collect(accelerator=MetalRuntime(), optimize=False)
    var child = actual.item().list()
    assert_equal(child.int64()._get(0), Int64(128))
    assert_equal(child.int64()._get(1), Int64(0))
    assert_true(
        actual.equals(
            query.collect(engine="cpu", streaming=False, optimize=False)
        )
    )
    var empty_query = frame.clear().lazy().select(expression.copy())
    assert_equal(_plan(empty_query).ordered_mode, 3)
    var empty = empty_query.collect(
        accelerator=MetalRuntime(device_id=999999), optimize=False
    )
    assert_equal(empty.height(), 1)
    assert_equal(len(empty.item().list()), 0)
    var nulls = (
        DataFrame(
            [
                Series("x", Column[Int64]([128, 128], [False, False])),
                Series(
                    "t", Column[Int64]([Int64.MAX, Int64.MAX], [False, False])
                ).with_dtype(DataType.duration("ms")),
            ]
        )
        .lazy()
        .select((_time() + _cast()).implode())
    )
    var null_result = nulls.collect(accelerator=MetalRuntime(), optimize=False)
    assert_equal(len(null_result.item().list()), 2)
    assert_equal(null_result.item().list().null_count(), 2)
    assert_true(
        null_result.equals(
            nulls.collect(engine="cpu", streaming=False, optimize=False)
        )
    )
    var decimal_type = DataType.decimal(3, 0, 128)
    var decimals: List[Int128] = [999, 1]
    var zeros: List[Int128] = [0, 0]
    frame = (
        frame.with_column(
            Series("d", Column[Int128](decimals^)).with_dtype(decimal_type)
        )
        .with_column(
            Series("zero", Column[Int128](zeros^)).with_dtype(decimal_type)
        )
        .with_column(Series("k", Column[Int32]([0, 0])))
    )
    for windowed in [False, True]:
        var state = col("d").sum().over("k") if windowed else col("d").sum()
        var later = (col("zero") + state).implode().alias("later")
        var first = (_cast() + _time()).implode().alias("first")
        for reverse_order in [False, True]:
            var fields = List[Expr]()
            fields.append(later.copy() if reverse_order else first.copy())
            fields.append(first.copy() if reverse_order else later.copy())
            var checked = frame.lazy().select_exprs(fields^).head(0)
            if not windowed:
                # Shared binding rejects nested aggregates before either
                # producer executes, independently of projection order.
                for gpu in [False, True]:
                    with assert_raises(
                        contains="Aggregate input must be row-valued without nested aggregates"
                    ):
                        if gpu:
                            _ = checked.collect(
                                accelerator=MetalRuntime(), optimize=False
                            )
                        else:
                            _ = checked.collect(
                                engine="cpu", streaming=False, optimize=False
                            )
                continue
            _fails(
                checked,
                "decimal precision" if reverse_order else "cast from int64 to int8 failed",
                "decimal precision" if reverse_order else "strict cast failed",
            )


# requires: metal
def test_mojo_gpu_ordered_implode_resources_provenance_and_foreign_flags() raises:
    var frame = _frame([-1, 0, 1], [-2, 0, 2])
    var query = frame.lazy().select((_time() + _cast()).implode().alias("v"))
    var plan = _plan(query)
    var facts = estimate_resources(plan)
    assert_equal(plan.ordered_mode, 3)
    assert_equal(len(plan.steps), 1)
    assert_equal(facts.kernel_launches, 5)
    var scope = Scope()
    var executed = execute_plan(plan, scope)
    assert_equal(executed[1].item(0, "kernel_launches").int64(), Int64(5))
    assert_equal(executed[1].item(0, "download_bytes").int64(), Int64(16))
    assert_true(
        executed[1].item(0, "upload_bytes").int64() <= Int64(facts.upload_bytes)
    )
    assert_true(
        executed[1].item(0, "synchronizations").int64()
        <= Int64(facts.synchronizations)
    )
    var measured = query.profile(accelerator=MetalRuntime(), optimize=False)
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    assert_true(
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), optimize=False
        ).equals(measured[0])
    )
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            optimize=False,
        )
    with assert_raises(contains="no analytical calibration"):
        _ = MetalRuntime().estimate_auto(query)
    # The external helper has conservative provenance by default. Even an
    # otherwise identical synthetic raw scan cannot opt itself into mode 3.
    with assert_raises(contains="source provenance"):
        _ = lower_terminal_implode(query)
    assert_true(
        "source provenance"
        in MetalRuntime(device_id=999999)._describe(query, False)
    )
    with assert_raises(contains="source provenance"):
        _ = MetalRuntime(device_id=999999)._run(
            query, False, original_entry=False
        )
    with assert_raises(contains="source provenance"):
        _ = lower_terminal_implode(
            frame.lazy()
            .with_columns(col("x").alias("alias"))
            .select(
                (
                    _time()
                    + col("alias").cast(DataType.INT8).cast(DataType.INT64)
                ).implode()
            ),
            ordered_source=True,
        )
    var staged = query.copy()
    staged._nodes[0].names2 = ["metal_staged_file_source"]
    with assert_raises(contains="source provenance"):
        _ = lower_terminal_implode(staged, ordered_source=True)
    var lowered = lower_terminal_implode(query, ordered_source=True)
    for at in range(len(lowered.query._nodes)):
        if implode_ordered.tagged(lowered.query._nodes[at]):
            var altered = lowered.query.copy()
            altered._nodes[at].exprs[0] = (_time() - _cast()).alias(
                altered._nodes[at].exprs[0]._name
            )
            with assert_raises(contains="AST or logical metadata changed"):
                _ = dag_preflight(altered, capabilities())
    # An explicitly injected foreign flag must be rejected as an unobserved
    # input, never interpreted as a rank or silently hidden by a later WHEN.
    var input_scope = Scope()
    var cells = frame.height() * plan.source.width()
    var errors = List[UInt32](length=cells, fill=0)
    errors[cells - 1] = 32
    var foreign = Table.identity(
        input_scope,
        HostMatrix(
            plan.source.clear(),
            frame.height(),
            List[UInt64](length=2 * cells, fill=0),
            List[UInt8](length=cells, fill=1),
            errors^,
        ),
    )
    with assert_raises(contains="already observed successful inputs"):
        _ = execute_plan(plan, input_scope, resident_input=foreign.copy())
