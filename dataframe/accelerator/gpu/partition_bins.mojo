"""Private partition CUT/QCUT publication after exact run fault observation.

The caller certifies original binding, Float64 operand conversion and CPU
OVER dispatch. Ranked operands are never materialized or observed globally.
Only O(G*K) edge controls reach the shared label formatter after the exact
run/state observer succeeds; the row classifier and values remain on device.
"""
from std.memory import ArcPointer
from dataframe import DataFrame, DataType, Series
from dataframe.accelerator.code_domains import TextField
from .residency import (
    Scope,
    TextDomain,
    Table,
    Matrix,
    CELL_BYTES,
    checked_add,
    checked_mul,
    raise_value_error,
)
from .ranked_rows import RankedRows, _Reservation
from .partition_layout import PartitionOwner
from .partition_labels import build as build_labels, Labels
from .ordered_faults import category, rank
from . import partition_faults as faults, partition_qcuts as bins, qcuts


@fieldwise_init
struct Resources(Copyable):
    var scratch_bytes: Int
    var result_bytes: Int
    var upload_bytes: Int
    var download_bytes: Int
    var kernel_launches: Int
    var synchronizations: Int
    var row_slices: Int
    var group_slices: Int


def estimate(
    rows: Int, groups: Int, count: Int, include_breaks: Bool
) raises -> Resources:
    """Additional fixed bytes above retained source/layout/ranked owners.

    Label domain construction is data-dependent and separately checked by
    partition_labels.build. Its host map and immutable dictionary stay charged
    while the device map/output owner are created. No input value readback.
    """
    if (
        rows <= 0
        or rows > Int(Int32.MAX)
        or groups <= 0
        or groups > rows
        or count < 0
        or count > Int(Int32.MAX)
    ):
        raise Error(
            "Partition bin geometry requires positive Int32 rows/groups"
        )
    var row_slices = min(1024, (rows + 255) // 256)
    var group_slices = min(1024, (groups + 255) // 256)
    var edge_count = checked_mul(groups, max(1, count))
    var maps = checked_mul(groups, count + 1)
    # mask + selection scratch + host/device specs + host/device edges +
    # original custom-label indices + host/device group controls + fault
    # partials + host/device final fault + device default-label code map.
    var bytes = checked_add(rows, checked_mul(rows, 4))
    bytes = checked_add(bytes, checked_mul(max(1, count), 16))
    bytes = checked_add(bytes, checked_mul(edge_count, 16))
    bytes = checked_add(bytes, checked_mul(maps, 8))
    bytes = checked_add(bytes, checked_mul(groups, 20))
    bytes = checked_add(bytes, checked_mul(row_slices + group_slices, 32))
    bytes = checked_add(bytes, 64)
    var outputs = checked_mul(
        checked_mul(rows, 2 if include_breaks else 1), CELL_BYTES
    )
    return Resources(
        bytes,
        outputs,
        checked_add(checked_mul(count, 8), checked_mul(maps, 4)),
        checked_add(
            32, checked_add(checked_mul(edge_count, 8), checked_mul(groups, 8))
        ),
        6,
        2,
        row_slices,
        group_slices,
    )


def execute_default(
    operand: RankedRows,
    layout: ArcPointer[PartitionOwner],
    specifications: List[UInt64],
    scope: Scope,
    *,
    qcut: Bool,
    allow_duplicates: Bool = False,
    left_closed: Bool = False,
    include_breaks: Bool = False,
    operand_proved: Bool = False,
    batch_rows: Int = 8192,
    operand_stage: Int = 0,
    category_name: String = "category",
    breakpoint_name: String = "breakpoint",
    custom_codes: List[UInt32] = List[UInt32](),
    custom_domain: Optional[ArcPointer[TextDomain]] = None,
) raises -> Table:
    if not operand_proved:
        raise Error(
            "Partition bin original Float64 operand certificate is missing"
        )
    if not operand.scope.same_query(scope) or not layout[].scope.same_query(
        scope
    ):
        raise Error("Partition bin owners belong to a different query scope")
    if (
        operand.rows != layout[].rows
        or batch_rows <= 0
        or batch_rows > Int(UInt32.MAX)
        or operand_stage < 0
        or operand_stage >= Int(UInt32.MAX)
    ):
        raise Error("Partition bin operand/run geometry or phase is invalid")
    if (
        operand.slots <= 0
        or operand.slots > 256
        or operand.target < 0
        or operand.target >= operand.slots
        or operand.cells != checked_mul(operand.rows, operand.slots)
        or len(operand.words) != checked_mul(operand.cells, 2)
        or len(operand.validity) != operand.cells
        or len(operand.tokens) != operand.cells
    ):
        raise Error("Partition bin ranked matrix shape changed")
    if not category_name or (
        include_breaks
        and (not breakpoint_name or breakpoint_name == category_name)
    ):
        raise Error("Partition bin output names must be distinct and nonempty")
    if Bool(custom_domain) != Bool(len(custom_codes)):
        raise Error("Partition custom label codes require their text domain")
    if custom_domain and (
        not custom_domain.value()[].scope.same_query(scope)
        or len(custom_codes) != len(specifications) + 1
    ):
        raise Error("Partition custom label domain or count changed")
    if custom_domain:
        for code in custom_codes:
            if UInt64(code) >= UInt64(
                len(custom_domain.value()[].dtype.dictionary()[])
            ):
                raise Error("Partition custom label code is outside its domain")
    var rows = operand.rows
    var groups = layout[].groups
    var count = len(specifications)
    var resources = estimate(rows, groups, count, include_breaks)
    var reservation = _Reservation(
        scope, checked_add(resources.scratch_bytes, resources.result_bytes)
    )
    var entries = checked_mul(groups, max(1, count))
    var mapping = checked_mul(groups, count + 1)
    var host_specs = List[UInt64](length=max(1, count), fill=0)
    for index in range(count):
        host_specs[index] = specifications[index]
    var host_edges = List[UInt64](length=entries, fill=0)
    var host_controls = List[UInt32](length=2 * groups, fill=0)
    var host_fault = List[UInt32](length=8, fill=0)
    var used = List[UInt32](capacity=groups)
    var ctx = scope.context()
    var specs = ctx.enqueue_create_buffer[DType.uint64](max(1, count))
    var mask = ctx.enqueue_create_buffer[DType.uint8](rows)
    var permutation = ctx.enqueue_create_buffer[DType.uint32](rows)
    var edges = ctx.enqueue_create_buffer[DType.uint64](entries)
    var original_labels = ctx.enqueue_create_buffer[DType.uint32](mapping)
    var controls = ctx.enqueue_create_buffer[DType.uint32](2 * groups)
    var partials = ctx.enqueue_create_buffer[DType.uint32](
        8 * (resources.row_slices + resources.group_slices)
    )
    var winner = ctx.enqueue_create_buffer[DType.uint32](8)
    var codes = ctx.enqueue_create_buffer[DType.uint32](mapping)
    var slots = 2 if include_breaks else 1
    var cells = checked_mul(rows, slots)
    var words = ctx.enqueue_create_buffer[DType.uint64](2 * cells)
    var validity = ctx.enqueue_create_buffer[DType.uint8](cells)
    var errors = ctx.enqueue_create_buffer[DType.uint32](cells)
    var held_layout = layout.copy()
    var held_input_words = operand.words.copy()
    var held_input_validity = operand.validity.copy()
    var held_input_tokens = operand.tokens.copy()
    try:
        if count:
            ctx.enqueue_copy(specs, host_specs.unsafe_ptr())
        # Copy uploads max(1,K) words for nonempty controls; K=0 needs no read
        # of the uninitialized device placeholder. No source values are copied.
        scope.record_upload(checked_mul(count, 8))
        ctx.enqueue_function[qcuts.sample_mask](
            operand.words.unsafe_ptr(),
            operand.validity.unsafe_ptr(),
            mask.unsafe_ptr(),
            UInt64(operand.target * rows),
            UInt32(rows),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[bins.segmented_controls](
            operand.words.unsafe_ptr(),
            mask.unsafe_ptr(),
            layout[].index.unsafe_ptr(),
            layout[].starts.unsafe_ptr(),
            permutation.unsafe_ptr(),
            specs.unsafe_ptr(),
            edges.unsafe_ptr(),
            original_labels.unsafe_ptr(),
            controls.unsafe_ptr(),
            UInt64(operand.target * rows),
            UInt32(groups),
            UInt32(count),
            UInt32(qcut),
            UInt32(allow_duplicates),
            grid_dim=(groups + 63) // 64,
            block_dim=64,
        )
        ctx.enqueue_function[faults.operand_partials](
            operand.tokens.unsafe_ptr(),
            layout[].phase_by_row.unsafe_ptr(),
            layout[].ordinal_by_row.unsafe_ptr(),
            partials.unsafe_ptr(),
            UInt32(operand.target),
            UInt32(rows),
            UInt32(rows),
            UInt32(resources.row_slices),
            UInt32(operand_stage),
            UInt32(batch_rows),
            grid_dim=(resources.row_slices + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[faults.control_partials](
            controls.unsafe_ptr(),
            layout[].phase_by_group.unsafe_ptr(),
            layout[].rank_by_group.unsafe_ptr(),
            partials.unsafe_ptr(),
            UInt32(groups),
            UInt32(resources.group_slices),
            UInt32(resources.row_slices),
            UInt32(operand_stage + 1),
            grid_dim=(resources.group_slices + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[faults.finish](
            partials.unsafe_ptr(),
            winner.unsafe_ptr(),
            UInt32(resources.row_slices + resources.group_slices),
            grid_dim=1,
            block_dim=1,
        )
        scope.record_launches(5)
        ctx.enqueue_copy(host_fault.unsafe_ptr(), winner)
        ctx.enqueue_copy(host_edges.unsafe_ptr(), edges)
        ctx.enqueue_copy(host_controls.unsafe_ptr(), controls)
        scope.synchronize()
        scope.record_download(resources.download_bytes)
        if host_fault[0] == faults.OPERAND:
            if not rank(host_fault[7]) or not category(host_fault[7]):
                raise Error(
                    "Partition operand contains an invalid occurrence token"
                )
            raise_value_error(
                category(host_fault[7]),
                "Mojo GPU partition operand strict cast failed or arithmetic overflow",
            )
        elif host_fault[0] == faults.CONTROL:
            raise Error(qcuts.failure_message(host_fault[7]))
        elif host_fault[0]:
            raise Error("Partition bin observer returned an invalid fault kind")
        for group in range(groups):
            used.append(host_controls[group * 2])
        var labels: Labels
        if custom_domain:
            # Fixed-stride allocation matches default-label resource bounds;
            # custom classification addresses the original K+1 label indices.
            var label_map = List[UInt32](length=mapping, fill=0)
            for at in range(len(custom_codes)):
                label_map[at] = custom_codes[at]
            labels = Labels(
                scope, custom_domain.value(), label_map^, scope.info()[1]
            )
        else:
            labels = build_labels(scope, host_edges, used, count, left_closed)
        # Formatting scratch has finished. Its immutable domain and host code
        # map remain owned through upload/classification and adoption.
        var held_domain = labels.domain.copy()
        try:
            ctx.enqueue_copy(codes, labels.codes.unsafe_ptr())
            scope.record_upload(checked_mul(mapping, 4))
            ctx.enqueue_function[bins.segmented_classify](
                operand.words.unsafe_ptr(),
                operand.tokens.unsafe_ptr(),
                mask.unsafe_ptr(),
                layout[].group_by_row.unsafe_ptr(),
                edges.unsafe_ptr(),
                original_labels.unsafe_ptr(),
                codes.unsafe_ptr(),
                controls.unsafe_ptr(),
                words.unsafe_ptr(),
                validity.unsafe_ptr(),
                errors.unsafe_ptr(),
                UInt64(operand.target * rows),
                UInt64(0),
                UInt64(rows),
                UInt64(cells),
                UInt32(rows),
                UInt32(count),
                UInt32(left_closed),
                UInt32(include_breaks),
                UInt32(not custom_domain),
                grid_dim=(rows + 255) // 256,
                block_dim=256,
            )
            scope.record_launches(1)
            scope.synchronize()
        except error:
            scope.synchronize()
            _ = labels^
            _ = held_domain^
            raise error^
        var schema_columns = List[Series]()
        schema_columns.append(
            Series.full_null(category_name, DataType.UINT32, 0)
        )
        if include_breaks:
            schema_columns.append(
                Series.full_null(breakpoint_name, DataType.FLOAT64, 0)
            )
        var schema = DataFrame(schema_columns^)
        var columns: List[Int64] = [0]
        if include_breaks:
            columns.append(Int64(1))
        reservation.release(resources.result_bytes)
        var matrix = Matrix(
            scope, rows, slots, words.copy(), validity.copy(), errors.copy()
        )
        var result = Table(matrix^, schema^, columns^, rows)
        result.attach_text(
            [TextField(category_name, DataType.STRING)], held_domain
        )
        _ = labels^
        _ = held_domain^
        _ = specs^
        _ = mask^
        _ = permutation^
        _ = edges^
        _ = original_labels^
        _ = controls^
        _ = partials^
        _ = winner^
        _ = codes^
        _ = words^
        _ = validity^
        _ = errors^
        _ = host_specs^
        _ = host_edges^
        _ = host_controls^
        _ = host_fault^
        _ = used^
        _ = held_layout^
        _ = held_input_words^
        _ = held_input_validity^
        _ = held_input_tokens^
        return result^
    except error:
        scope.synchronize()
        _ = specs^
        _ = mask^
        _ = permutation^
        _ = edges^
        _ = original_labels^
        _ = controls^
        _ = partials^
        _ = winner^
        _ = codes^
        _ = words^
        _ = validity^
        _ = errors^
        _ = host_specs^
        _ = host_edges^
        _ = host_controls^
        _ = host_fault^
        _ = used^
        _ = held_layout^
        _ = held_input_words^
        _ = held_input_validity^
        _ = held_input_tokens^
        raise error^
