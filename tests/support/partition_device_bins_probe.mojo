"""Literal partition controls + private ranked operand + final resident output.

All source/result value transfers are ordinary explicit probe boundaries. The
production operation downloads only edge/fault controls and keeps its output
resident until this probe explicitly materializes it.
"""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.categorical import decode
from dataframe_metal.accelerator.partition_operands import certificate, prepare
from dataframe.plan_types import WindowKey
from dataframe_metal.accelerator.gpu.residency import Scope, Table
from dataframe_metal.accelerator.gpu.runtime import (
    _preflight,
    execute_plan,
    materialize,
)
from dataframe_metal.accelerator.gpu.partition_layout import KeyInput, build
from dataframe_metal.accelerator.gpu.partition_bins import (
    execute_default,
    estimate,
)
from dataframe_metal.accelerator.gpu.ranked_rows import evaluate
from tests.support.lifecycle_jobs import WorkerSetting


def _stage(frame: DataFrame, scope: Scope) raises -> Table:
    var columns = List[Expr]()
    for field in frame._columns:
        columns.append(col(field.name()))
    var plan = _preflight(frame.lazy().select_exprs(columns))
    return execute_plan(plan, scope, export_values=False)[0].copy()


def _controls(values: List[Float64]) -> List[UInt64]:
    var result = List[UInt64](capacity=len(values))
    for value in values:
        result.append(bitcast[DType.uint64](value))
    return result^


def _successful(workers: Int, closed: Bool, qcut: Bool) raises:
    var setting = WorkerSetting(workers)
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000042))
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [0, 10, 2, 14, 99, nan, 4, 8],
                    [True, True, True, True, False, True, True, True],
                ),
            ),
            Series(
                "k",
                Column[Int32](
                    [0, 1, 0, 1, 0, 1, 7, 7],
                    [True, True, True, True, True, True, False, False],
                ),
            ),
        ]
    )
    var scope = Scope()
    var source = _stage(frame, scope)
    var keys: List[KeyInput] = [
        KeyInput("k", WindowKey("k", DataType.INT32, False))
    ]
    var layout = build(source, keys, scope, workers)
    var expression = col("x")
    var operand_plan = prepare(
        expression,
        frame,
        frame.height(),
        certificate(expression, frame),
        source_proved=True,
    )
    var operand = evaluate(operand_plan, source, scope)
    # Keep independently typed literal controls alive across CPU reference and
    # asynchronous device execution. Assert their exact payloads before either
    # path so an empty/changed fixture cannot make CPU/device agreement pass.
    var quantiles: List[Float64] = [0.5]
    var breaks: List[Float64] = [2.0]
    var specifications = quantiles.copy() if qcut else breaks.copy()
    assert_equal(len(specifications), 1)
    assert_equal(
        bitcast[DType.uint64](specifications[0]),
        UInt64(0x3FE0000000000000) if qcut else UInt64(0x4000000000000000),
    )
    var node = expression.qcut(
        quantiles, left_closed=closed, include_breaks=True
    ) if qcut else expression.cut(
        breaks, left_closed=closed, include_breaks=True
    )
    assert_true(node._nodes[len(node._nodes) - 1].text.byte_length() > 0)
    if qcut:
        assert_equal(node._nodes[len(node._nodes) - 1].text, "0.5")
    var cpu = (
        frame.lazy()
        .select(node.over("k").alias("cut"))
        .collect(engine="cpu", optimize=False, streaming=False)
    )
    var before = scope.transfers()
    var live = scope.info()[0]
    var output = execute_default(
        operand,
        layout,
        _controls(specifications),
        scope,
        qcut=qcut,
        left_closed=closed,
        include_breaks=True,
        operand_proved=True,
    )
    var after = scope.transfers()
    var resources = estimate(
        frame.height(), layout[].groups, len(specifications), True
    )
    assert_equal(after[0] - before[0], resources.upload_bytes)
    assert_equal(after[1] - before[1], resources.download_bytes)
    assert_equal(after[2] - before[2], resources.kernel_launches)
    assert_equal(after[3] - before[3], resources.synchronizations)
    assert_equal(output.rows, 8)
    assert_equal(output.schema.width(), 2)
    assert_equal(
        scope.info()[0] - live,
        8 * 2 * 21 + output.text[0].value().domain[].retained,
    )
    assert_equal(output.text[0].value().logical, DataType.STRING)
    assert_false(Bool(output.text[1]))
    var actual = materialize(output)
    var labels = decode(
        actual.column("category").with_dtype(
            output.text[0].value().domain[].dtype
        )
    )
    var expected: List[String] = [
        "(-inf, 1]",
        "(-inf, 12]",
        "(1, inf]",
        "(12, inf]",
        "",
        "",
        "(-inf, 6]",
        "(6, inf]",
    ]
    var literal_edges: List[UInt64] = [
        0x3FF0000000000000,
        0x4028000000000000,
        0x7FF0000000000000,
        0x7FF0000000000000,
        0,
        0,
        0x4018000000000000,
        0x7FF0000000000000,
    ]
    for row in range(8):
        var wanted = cpu.item(row, "cut")
        if row == 4 or row == 5:
            assert_true(labels.get(row).is_null())
            assert_true(actual.item(row, "breakpoint").is_null())
        else:
            assert_equal(
                labels.get(row).string(),
                wanted.struct_field("category").string(),
            )
            assert_equal(
                bitcast[DType.uint64](actual.item(row, "breakpoint").float64()),
                bitcast[DType.uint64](
                    wanted.struct_field("breakpoint").float64()
                ),
            )
            if qcut and not closed:
                assert_equal(labels.get(row).string(), expected[row])
                assert_equal(
                    bitcast[DType.uint64](
                        actual.item(row, "breakpoint").float64()
                    ),
                    literal_edges[row],
                )
    _ = output^
    assert_equal(scope.info()[0], live)
    _ = operand^
    _ = layout^
    _ = source^
    _ = setting^
    print("PASS resident partition bins", workers, closed, qcut)


def _fault_order(workers: Int) raises:
    var setting = WorkerSetting(workers)
    var values = List[Int64](length=8192, fill=0)
    values[4096] = Int64.MAX
    var keys = List[Int32](length=8192, fill=0)
    for row in range(4096, 8192):
        keys[row] = 1
    var frame = DataFrame(
        [Series("x", Column[Int64](values^)), Series("k", Column[Int32](keys^))]
    )
    var expression = col("x") + lit(Int64(1))
    var public_expression = expression.qcut([0.0, 0.5]).over("k")
    var expected = "breaks must be unique" if workers >= 2 else "overflow"
    with assert_raises(contains=expected):
        _ = (
            frame.lazy()
            .select(public_expression.copy())
            .head(0)
            .collect(engine="cpu", optimize=False, streaming=False)
        )
    with assert_raises(contains=expected):
        _ = (
            frame.lazy()
            .select(
                when(lit(False))
                .then(public_expression.copy())
                .otherwise(lit("safe"))
            )
            .collect(engine="cpu", optimize=False, streaming=False)
        )
    var scope = Scope()
    var source = _stage(frame, scope)
    var key_inputs: List[KeyInput] = [
        KeyInput("k", WindowKey("k", DataType.INT32, False))
    ]
    var layout = build(source, key_inputs, scope, workers)
    var plan = prepare(
        expression,
        frame,
        frame.height(),
        certificate(expression, frame),
        source_proved=True,
    )
    var operand = evaluate(plan, source, scope)
    var specs: List[Float64] = [0.0, 0.5]
    var live = scope.info()[0]
    var before = scope.transfers()
    with assert_raises(contains=expected):
        _ = execute_default(
            operand,
            layout,
            _controls(specs),
            scope,
            qcut=True,
            operand_proved=True,
        )
    assert_equal(scope.info()[0], live)
    var after = scope.transfers()
    assert_equal(after[2] - before[2], 5)
    # Failure observes only edge/fault controls and publishes no result/domain.
    assert_equal(
        after[1] - before[1], estimate(8192, 2, 2, False).download_bytes
    )
    var foreign = Scope(99999)
    with assert_raises(contains="different query scope"):
        _ = execute_default(
            operand,
            layout,
            _controls(specs),
            foreign,
            qcut=True,
            operand_proved=True,
        )
    assert_false(foreign.initialized())
    with assert_raises(contains="certificate is missing"):
        _ = execute_default(operand, layout, _controls(specs), scope, qcut=True)
    _ = operand^
    _ = layout^
    _ = source^
    _ = setting^
    print("PASS exact run control/operand priority", workers, expected)


def _scalar_faults() raises:
    var frame = DataFrame(
        [Series("x", Column[Int64]([1, 2])), Series("k", Column[Int32]([0, 0]))]
    )
    var scope = Scope()
    var source = _stage(frame, scope)
    var keys: List[KeyInput] = [
        KeyInput("k", WindowKey("k", DataType.INT32, False))
    ]
    var layout = build(source, keys, scope, 1)
    var cast = lit(Int64(128)).cast(DataType.INT8).cast(DataType.INT64)
    var power = lit(Int64(2)).pow(lit(Int64(-1)))
    for direction in range(2):
        var expression = col("x") + (
            cast.copy() + power.copy() if direction
            == 0 else power.copy() + cast.copy()
        )
        with assert_raises(
            contains="cast from int64 to int8 failed" if direction
            == 0 else "nonnegative exponent"
        ):
            _ = (
                frame.lazy()
                .select(expression.qcut([0.5]).over("k"))
                .collect(engine="cpu", optimize=False, streaming=False)
            )
        var plan = prepare(
            expression,
            frame,
            2,
            certificate(expression, frame),
            source_proved=True,
        )
        var operand = evaluate(plan, source, scope)
        var specs: List[Float64] = [0.5]
        with assert_raises(
            contains="strict cast failed" if direction
            == 0 else "nonnegative exponent"
        ):
            _ = execute_default(
                operand,
                layout,
                _controls(specs),
                scope,
                qcut=True,
                operand_proved=True,
            )
    var expression = (
        when(lit(False)).then(col("x") + cast.copy()).otherwise(lit(Int64(7)))
    )
    var plan = prepare(
        expression, frame, 2, certificate(expression, frame), source_proved=True
    )
    var operand = evaluate(plan, source, scope)
    var specs: List[Float64] = [0.5]
    var output = execute_default(
        operand, layout, _controls(specs), scope, qcut=True, operand_proved=True
    )
    var actual = materialize(output)
    var labels = decode(
        actual.column("category").with_dtype(
            output.text[0].value().domain[].dtype
        )
    )
    assert_equal(labels.get(0).string(), "(-inf, 7]")
    assert_equal(labels.get(1).string(), "(-inf, 7]")
    print("PASS private scalar occurrence order and WHEN suppression")


def run_probe() raises:
    for workers in [1, 4]:
        for closed in [False, True]:
            for qcut in [False, True]:
                _successful(workers, closed, qcut)
        _fault_order(workers)
    _scalar_faults()
    print(
        "PASS partition bins:8literal outputs +2run fault priorities +scalar/mask cases"
    )
