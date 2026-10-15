"""Public partition CUT/QCUT state execution through resident private owners."""
from std.memory import ArcPointer
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, Series
from dataframe.lazy import LazyFrame
from dataframe.expr import COL
from dataframe.parse import parse_int64
from dataframe.parallel import configured_workers
from dataframe.partition import _hash_bytes
from dataframe.accelerator.partition_cuts import (
    is_marker,
    marker_capture,
    Capture,
)
from dataframe.accelerator.partition_operands import prepare
from dataframe.accelerator.code_domains import TextField
from .residency import (
    Scope,
    Table,
    Matrix,
    TextDomain,
    CELL_BYTES,
    checked_mul,
    checked_add,
)
from .ranked_rows import evaluate, _Reservation
from .partition_layout import build, KeyInput
from .partition_bins import execute_default
from .partition_labels import build as build_labels
from .qcut_runtime import _report as _base_report
from . import qcuts, floating as f


def _report(scope: Scope, input: Table, result: Table, before: Tuple[Int, Int, Int, Int], started: Int, qcut: Bool) raises -> DataFrame:
    var report = _base_report(scope, input, result, before, started, scope.info()[1])
    report = report.with_column(Series("operator", Column[String](["PARTITION QCUT" if qcut else "PARTITION CUT"])))
    return report.with_column(Series("algorithm", Column[String](["mojo_gpu_partition_bins_ranked_faults"])))


def preflight(query: LazyFrame) raises -> Capture:
    if not is_marker(query) or len(query._nodes) != 2:
        raise Error("Partition CUT requires an isolated resident state marker")
    var saved = marker_capture(query)
    ref operation = query._nodes[1]
    if (
        operation.left != 0
        or len(operation.exprs) != (2 if saved.include_breaks else 1)
        or not operation.names2[2]
        or (saved.include_breaks and not operation.names2[3])
    ):
        raise Error("Partition CUT private output geometry changed")
    # Full original binding occurs during public lowering. Recheck its numeric
    # operand proof now, before any context initialization or submission.
    var source = DataFrame(operation.partition_columns.copy())
    _ = prepare(
        saved.operand, source, 0, saved.operand_signature, source_proved=True
    )
    return saved^


def _label_codes(
    saved: Capture, domain: ArcPointer[TextDomain]
) raises -> List[UInt32]:
    var codes = List[UInt32](capacity=len(saved.labels))
    ref dictionary = domain[].dtype.dictionary()[]
    for label in saved.labels:
        var found = -1
        for code in range(len(dictionary)):
            if dictionary.get(code) == label:
                found = code
                break
        if found < 0:
            raise Error(
                "Partition CUT custom literal is absent from its text domain"
            )
        codes.append(UInt32(found))
    return codes^


def _append(input: Table, output: Table, scope: Scope) raises -> Table:
    var width = checked_add(input.schema.width(), output.schema.width())
    var rows = input.rows
    if output.rows != rows or width > 256:
        raise Error("Partition CUT resident output alignment changed")
    var columns = input.schema._columns.copy()
    columns.extend(output.schema._columns.copy())
    var schema = DataFrame(columns^)
    if not rows:
        var result = Table.empty(schema^)
        result.text = input.text.copy()
        result.text.extend(output.text.copy())
        return result^
    var cells = checked_mul(rows, width)
    # Charge both immutable owners and the complete append/copy descriptors
    # before allocating; source/result payloads remain device resident.
    var bytes = checked_mul(cells, CELL_BYTES)
    var reservation = _Reservation(
        scope, checked_add(bytes, checked_mul(width, 56))
    )
    var context = scope.context()
    var words = context.enqueue_create_buffer[DType.uint64](2 * cells)
    var validity = context.enqueue_create_buffer[DType.uint8](cells)
    var errors = context.enqueue_create_buffer[DType.uint32](cells)
    var inputs = List[Int64](capacity=input.schema.width())
    var outputs = List[Int64](capacity=output.schema.width())
    var mapping = List[Int64](capacity=width)
    for at in range(input.schema.width()):
        inputs.append(Int64(at))
        mapping.append(Int64(at))
    for at in range(output.schema.width()):
        outputs.append(Int64(at))
        mapping.append(Int64(input.schema.width() + at))
    try:
        _ = input.copy_to(scope, words, validity, errors, rows, width, inputs)
        _ = output.copy_to(
            scope,
            words,
            validity,
            errors,
            rows,
            width,
            outputs,
            input.schema.width(),
        )
        scope.synchronize()
    except error:
        scope.synchronize()
        raise error^
    reservation.release(bytes)
    var result = Table(
        Matrix(scope, rows, width, words^, validity^, errors^),
        schema^,
        mapping^,
        rows,
    )
    result.text = input.text.copy()
    result.text.extend(output.text.copy())
    return result^


def execute_plan(
    query: LazyFrame,
    input: Table,
    scope: Scope,
    *,
    export_values: Bool = True,
    text_domain: Optional[ArcPointer[TextDomain]] = None,
) raises -> Tuple[Table, DataFrame]:
    var started = Int(perf_counter_ns())
    var before = scope.transfers()
    var saved = preflight(query)
    ref operation = query._nodes[1]
    var source = DataFrame(operation.partition_columns.copy())
    var category = operation.names2[2]
    var breakpoint = operation.names2[3]
    var custom = Optional[ArcPointer[TextDomain]]()
    var codes = List[UInt32]()
    if len(saved.labels):
        if not text_domain:
            raise Error(
                "Partition CUT custom labels require a query text domain"
            )
        custom = text_domain.copy()
        codes = _label_codes(saved, custom.value())
    if not input.rows:
        # CPU evaluates the virtual empty partition's controls even head(0).
        # No numeric input values are evaluated or downloaded.
        if saved.qcut:
            var controls = qcuts.empty_controls(saved.specifications)
            if controls[1]:
                raise Error(qcuts.failure_message(controls[1]))
        else:
            for at in range(1, len(saved.specifications)):
                if not f.less(
                    saved.specifications[at - 1], saved.specifications[at]
                ):
                    raise Error("breaks must be unique and strictly increasing")
        var columns: List[Series] = [
            Series.full_null(category, DataType.UINT32, 0)
        ]
        if saved.include_breaks:
            columns.append(Series.full_null(breakpoint, DataType.FLOAT64, 0))
        var output = Table.empty(DataFrame(columns^))
        if custom:
            output.attach_text(
                [TextField(category, DataType.STRING)], custom.value()
            )
        else:
            var edges = saved.specifications.copy() if not saved.qcut else List[
                UInt64
            ](length=max(1, len(saved.specifications)), fill=0)
            var used: List[UInt32] = [
                UInt32(0 if saved.qcut else len(saved.specifications))
            ]
            var labels = build_labels(
                scope, edges, used, len(saved.specifications), saved.left_closed
            )
            output.attach_text(
                [TextField(category, DataType.STRING)], labels.domain
            )
        var result = _append(input, output, scope)
        return (
            result.copy(),
            _report(scope, input, result, before, started, saved.qcut),
        )
    var keys = List[KeyInput](capacity=len(operation.window_keys))
    for key in operation.window_keys:
        keys.append(
            KeyInput(
                key.name,
                key,
                original_codes=key.dtype.is_categorical()
                and not key.string_hash,
            )
        )
    var layout = build(
        input,
        keys,
        scope,
        configured_workers(),
        cached_layout=Bool(parse_int64(operation.names2[7]))
        and min(configured_workers(), input.rows // 2048) > 1,
    )
    var plan = prepare(
        saved.operand,
        source,
        input.rows,
        saved.operand_signature,
        source_proved=True,
    )
    var operand = evaluate(plan, input, scope)
    var output = execute_default(
        operand,
        layout,
        saved.specifications,
        scope,
        qcut=saved.qcut,
        allow_duplicates=saved.allow_duplicates,
        left_closed=saved.left_closed,
        include_breaks=saved.include_breaks,
        operand_proved=True,
        category_name=category,
        breakpoint_name=breakpoint,
        custom_codes=codes,
        custom_domain=custom,
    )
    var result = _append(input, output, scope)
    return (
        result.copy(),
        _report(scope, input, result, before, started, saved.qcut),
    )
