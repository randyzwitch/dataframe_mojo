"""Prepared OVER states precede certified IMPLODE row AST evaluation."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe_metal.accelerator.lists import lower_terminal_implode
from dataframe_metal.accelerator.joins import DAG_ROWS
from dataframe_metal.accelerator.gpu.dag_runtime import (
    preflight as dag_preflight,
)
from dataframe_metal.accelerator.gpu.runtime import _preflight, capabilities
from dataframe_metal.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout


def _frame() raises -> DataFrame:
    var decimals: List[Int128] = [999, 1]
    return DataFrame(
        [
            Series("x", Column[Int64]([128, 0])),
            Series("base", Column[Int64]([2, 2])),
            Series("exp", Column[Int64]([-1, -1])),
            Series("v", Column[Int64]([3, 4])),
            Series("k", Column[Int32]([0, 0])),
            Series("d", Column[Int128](decimals^)).with_dtype(
                DataType.decimal(3, 0, 128)
            ),
            Series("t", Column[Int64]([0, Int64.MAX])).with_dtype(
                DataType.duration("ms")
            ),
            Series("keep", Column[Bool]([False, False])),
        ]
    )


def _cast() -> Expr:
    return col("x").cast(DataType.INT8).cast(DataType.INT64)


def _power() -> Expr:
    return col("base").pow(col("exp"))


def _time() raises -> Expr:
    return col("t").cast(DataType.duration("ns")).cast(DataType.INT64)


def _state() -> Expr:
    return col("v").sum().over("k")


def _certificate_count(query: LazyFrame) raises -> Int:
    var lowered = lower_terminal_implode(query, ordered_source=True)
    var dag = dag_preflight(lowered.query, capabilities())
    var count = 0
    for task in dag.tasks:
        if task.kind != DAG_ROWS:
            continue
        var plan = _preflight(task.query, terminal_boundary=task.terminal)
        if plan.ordered_mode == 3:
            assert_true(task.terminal)
            assert_true(task.source >= 0)
            assert_equal(len(plan.steps), 1)
            assert_equal(len(plan.outputs), plan.source.width() + 1)
            count += 1
    return count


def _fails(query: LazyFrame, cpu: String, gpu: String) raises:
    assert_true(_certificate_count(query) > 0)
    with assert_raises(contains=cpu):
        _ = query.collect(engine="cpu", optimize=False, streaming=False)
    with assert_raises(contains=gpu):
        _ = query.collect(accelerator=MetalRuntime(), optimize=False)


# requires: metal
def test_mojo_gpu_ordered_implode_post_state_ast_categories_and_batches() raises:
    var original = _frame()
    for layout in range(3):
        var frame = (
            original.copy() if layout
            == 0 else offset_layout(original) if layout
            == 1 else chunk_layout(original)
        )
        _fails(
            frame.lazy().select((_state() + (_cast() + _power())).implode()),
            "cast from int64 to int8 failed",
            "strict cast failed",
        )
        _fails(
            frame.lazy().select((_state() + (_power() + _cast())).implode()),
            "nonnegative exponent",
            "nonnegative exponent",
        )
        _fails(
            frame.lazy().select((_state() + (_time() + _cast())).implode()),
            "temporal value overflows Int64",
            "temporal value overflows Int64",
        )
        _fails(
            frame.lazy().select((_state() + (_cast() + _time())).implode()),
            "cast from int64 to int8 failed",
            "strict cast failed",
        )
    var huge = Int128(9)
    for _ in range(37):
        huge *= Int128(10)
    var dtype = DataType.decimal(38, 0, 128)
    var wide = original.with_column(
        Series("wide", Column[Int128]([huge, 0])).with_dtype(dtype)
    ).with_column(
        Series("two", Column[Int128]([Int128(2), Int128(2)])).with_dtype(dtype)
    )
    var product = (col("wide") * col("two")).cast(DataType.INT64)
    _fails(
        wide.lazy().select((_state() + (_cast() + product)).implode()),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )
    _fails(
        wide.lazy().select((_state() + (product + _cast())).implode()),
        "decimal multiplication overflow",
        "decimal multiplication overflow",
    )
    # CPU prepares the safe state, then _full observes batch 0 before batch 1,
    # even though the temporal AST occurrence comes before the strict cast.
    var ticks = List[Int64](length=8193, fill=0)
    var values = List[Int64](length=8193, fill=0)
    ticks[8192] = Int64.MAX
    values[8191] = 128
    var boundary = DataFrame(
        [
            Series("x", Column[Int64](values^)),
            Series("t", Column[Int64](ticks^)).with_dtype(
                DataType.duration("ms")
            ),
            Series("v", Column[Int64](List[Int64](length=8193, fill=0))),
            Series("k", Column[Int32](List[Int32](length=8193, fill=0))),
        ]
    )
    _fails(
        boundary.lazy().select((_state() + (_time() + _cast())).implode()),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )


# requires: metal
def test_mojo_gpu_ordered_implode_prepared_state_order_masks_and_projections() raises:
    var frame = _frame()
    var bad_state = col("d").sum().over("k").cast(DataType.INT64)
    for state_first in [False, True]:
        var expression = (
            bad_state
            + (_cast() + _power()) if state_first else (_cast() + _power())
            + bad_state
        )
        # Prepared state overflow precedes every outer row fault, regardless
        # of its location in the row expression's AST.
        _fails(
            frame.lazy().select(expression.implode()).head(0),
            "decimal precision",
            "decimal precision",
        )
    var strict_state = col("x").cast(DataType.INT8).sum().over("k")
    for strict_first in [False, True]:
        var states = (
            strict_state
            + bad_state if strict_first else bad_state
            + strict_state
        )
        _fails(
            frame.lazy().select((states + (_cast() + _power())).implode()),
            "cast from int64 to int8 failed" if strict_first else "decimal precision",
            "strict cast failed" if strict_first else "decimal precision",
        )
    _fails(
        frame.lazy().select(
            when(col("keep"))
            .then(bad_state + (_cast() + _power()))
            .otherwise(col("v"))
            .implode()
        ),
        "decimal precision",
        "decimal precision",
    )
    var safe = frame.lazy().select(
        when(col("keep"))
        .then(_state() + (_cast() + _power()))
        .otherwise(col("v"))
        .implode()
    )
    assert_equal(_certificate_count(safe), 1)
    var result = safe.collect(accelerator=MetalRuntime(), optimize=False)
    assert_true(
        result.item().list().equals(Series("item", Column[Int64]([3, 4])))
    )
    assert_true(
        result.equals(
            safe.collect(engine="cpu", optimize=False, streaming=False)
        )
    )
    var first = (_state() + (_cast() + _power())).implode().alias("first")
    var later = bad_state.implode().alias("later")
    for later_first in [False, True]:
        var fields = List[Expr]()
        fields.append(later.copy() if later_first else first.copy())
        fields.append(first.copy() if later_first else later.copy())
        _fails(
            frame.lazy().select_exprs(fields^).head(0),
            "decimal precision" if later_first else "cast from int64 to int8 failed",
            "decimal precision" if later_first else "strict cast failed",
        )


# requires: metal
def test_mojo_gpu_ordered_implode_state_compositions_resources_empty_provenance() raises:
    var frame = (
        _frame()
        .with_column(Series("x", Column[Int64]([1, 2])))
        .with_column(
            Series("t", Column[Int64]([-1, 1])).with_dtype(
                DataType.duration("ms")
            )
        )
    )
    # Two eager reduction leaves inside OVER are already observed before the
    # ranked row consumer. SUM(v)=7, MIN(v)=3, hence the state is exactly 10.
    var state = (col("v").sum() + col("v").min()).over("k")
    var expression = (state + (_cast() + _time())).implode().alias("values")
    var query = frame.lazy().select(expression.copy())
    assert_equal(_certificate_count(query), 1)
    var measured = query.profile(accelerator=MetalRuntime(), optimize=False)
    var expected = Series("item", Column[Int64]([-999989, 1000012]))
    assert_true(measured[0].item().list().equals(expected))
    assert_true(
        measured[0].equals(
            query.collect(engine="cpu", optimize=False, streaming=False)
        )
    )
    var ordered_regions = 0
    for row in range(measured[1].height()):
        if (
            measured[1].item(row, "algorithm").string()
            != "mojo_gpu_ordered_implode_rows"
        ):
            continue
        ordered_regions += 1
        # One resident copy + one program + four bounded observation kernels.
        assert_equal(measured[1].item(row, "kernel_launches").int64(), Int64(6))
        # Input/output token controls (16 bytes) plus legacy zero-error
        # verification of the complete successful producer table.
        assert_true(
            measured[1].item(row, "download_bytes").int64() >= Int64(16)
        )
        assert_equal(
            measured[1].item(row, "host_result_bytes").int64(), Int64(0)
        )
    assert_equal(ordered_regions, 1)
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
    var empty = (
        frame.clear()
        .lazy()
        .select(expression.copy())
        .collect(accelerator=MetalRuntime(device_id=999999), optimize=False)
    )
    assert_equal(empty.height(), 1)
    assert_equal(len(empty.item().list()), 0)
    with assert_raises(contains="source provenance"):
        _ = lower_terminal_implode(query)
    assert_true(
        "source provenance"
        in MetalRuntime(device_id=999999)._describe(query, False)
    )
    var staged = query.copy()
    staged._nodes[0].names2 = ["metal_staged_file_source"]
    with assert_raises(contains="source provenance"):
        _ = lower_terminal_implode(staged, ordered_source=True)
    var synthetic = (
        frame.lazy()
        .with_columns(col("v").alias("alias"))
        .select(expression.copy())
    )
    with assert_raises(contains="source provenance"):
        _ = lower_terminal_implode(synthetic, ordered_source=True)
