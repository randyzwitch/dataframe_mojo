"""Coalesce retained batches by source owner without gathering their values."""
from std.collections import Dict
from std.memory import Pointer
from std.ffi import external_call
from std.sys import size_of
from max.gpu.host import DeviceBuffer
from dataframe.dtype import DataType
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.vector.selection_batches import SelectionBatches
from ...selection_plan import SelectionReduction
from .filter_sum import BLOCK_SIZE
from .reduction_inputs import MAX_PARTIALS
from .selection_reduce import SELECTION_WORDS
from .numeric_view import NumericView
from .query_scope import Scope, checked_add, checked_mul
from .typed_storage import _NativeReservation


struct SelectionGroup(Movable):
    var field: Int
    var owner: Int
    # Before packing: triples of count, source start, selection start.
    var segments: List[UInt64]
    var rows: Int
    var first: Int
    var count: Int
    var grid: Int
    var partial: Int

    def __init__(out self, field: Int, owner: Int):
        self.field = field
        self.owner = owner
        self.segments = List[UInt64]()
        self.rows = 0
        self.first = 0
        self.count = 0
        self.grid = 0
        self.partial = 0


struct SelectionInputs(Movable):
    var owners: List[Series]
    var tags: List[Int]
    var groups: List[SelectionGroup]
    var selections: List[UInt32]
    var descriptors: List[UInt64]
    var partials: Int

    def __init__(out self):
        self.owners = List[Series]()
        self.tags = List[Int]()
        self.groups = List[SelectionGroup]()
        self.selections = List[UInt32]()
        self.descriptors = List[UInt64]()
        self.partials = 0


def _owner[
    D: DType
](
    series: Series,
    mut inputs: SelectionInputs,
    mut known: Dict[Tuple[Int, Int, Int], Int],
    tag: Int,
) raises -> Tuple[Int, Int, Bool]:
    if series.is_chunked() or not series._data.isa[Column[Scalar[D]]]():
        raise Error("Metal selection input requires native primitive storage")
    ref column = series._data[Column[Scalar[D]]]
    var constant = column.is_constant()
    if not constant and len(column._data[]) >= Int(Int32.MAX):
        raise Error(
            "Metal selection source exceeds the native bitmap index range"
        )
    var key = (Int(column._data.ptr()), Int(column._bits.ptr()), tag)
    var found = known.get(key)
    var owner: Int
    if found:
        owner = found.value()
    else:
        owner = len(inputs.owners)
        known[key] = owner
        var whole: Column[Scalar[D]]
        if constant:
            # One value for any number of selected copies. Asking the source
            # for a pointer would expand its entire repeated buffer.
            whole = Column[Scalar[D]](
                [column.constant_value()], [column.constant_valid()]
            )
        else:
            whole = Column[Scalar[D]]._from_buffers(
                column._data, column._bits, 0, len(column._data[])
            )
        inputs.owners.append(Series(series.name(), whole^))
        inputs.tags.append(tag)
    return (owner, 0 if constant else column._offset, constant)


def _supported_owner[D: DType](series: Series) -> Bool:
    if not series._data.isa[Column[Scalar[D]]]():
        return False
    ref column = series._data[Column[Scalar[D]]]
    # The kernel indexes the whole owner, including a slice's offset. Its
    # UInt32 validity addressing needs the same bound as ordinary handoffs.
    if column.is_constant():
        return True
    return (
        len(column._data[]) < Int(Int32.MAX)
        and Int(column.unsafe_values()) % size_of[Scalar[D]]() == 0
    )


def selection_inputs_supported(
    plan: SelectionReduction, batches: SelectionBatches
) -> Bool:
    if batches.rows >= Int(Int32.MAX):
        return False
    for batch_index in range(len(batches.batches)):
        ref batch = batches.batches[batch_index]
        for field in plan.fields:
            ref series = batch.columns[field.column]
            if series.is_chunked():
                return False
            if field.dtype == DataType.INT32:
                if not _supported_owner[DType.int32](series):
                    return False
            elif field.dtype == DataType.INT64:
                if not _supported_owner[DType.int64](series):
                    return False
            elif not _supported_owner[DType.float32](series):
                return False
    return True


def selection_inputs(
    plan: SelectionReduction, batches: SelectionBatches
) raises -> SelectionInputs:
    batches.validate()
    return _validated_selection_inputs(plan, batches)


def _validated_selection_inputs(
    plan: SelectionReduction, batches: SelectionBatches
) raises -> SelectionInputs:
    # Callers validate before inspecting or packing any source mapping.
    if batches.rows >= Int(Int32.MAX):
        raise Error("Metal selection reduction requires fewer than 2**31 rows")
    var result = SelectionInputs()
    var owners = Dict[Tuple[Int, Int, Int], Int]()
    var groups = Dict[Tuple[Int, Int], Int]()
    for batch_index in range(len(batches.batches)):
        ref batch = batches.batches[batch_index]
        if batch.count == 0:
            continue
        var selections = List[Int](length=len(batch.rows), fill=-1)
        for field in range(len(plan.fields)):
            var c = plan.fields[field].column
            ref series = batch.columns[c]
            var source: Tuple[Int, Int, Bool]
            if plan.fields[field].dtype == DataType.INT32:
                source = _owner[DType.int32](series, result, owners, 0)
            elif plan.fields[field].dtype == DataType.INT64:
                source = _owner[DType.int64](series, result, owners, 1)
            else:
                source = _owner[DType.float32](series, result, owners, 2)
            var list = batch.lists[c]
            var selection = UInt64.MAX
            if source[2]:
                selection = UInt64.MAX - 1
            elif list >= 0:
                if selections[list] < 0:
                    selections[list] = len(result.selections)
                    for row in batch.rows[list]:
                        result.selections.append(row)
                selection = UInt64(selections[list])
            var key = (field, source[0])
            var found = groups.get(key)
            var group: Int
            if found:
                group = found.value()
            else:
                group = len(result.groups)
                groups[key] = group
                result.groups.append(SelectionGroup(field, source[0]))
            result.groups[group].segments.append(UInt64(batch.count))
            result.groups[group].segments.append(
                UInt64(checked_add(source[1], batch.starts[c]))
            )
            result.groups[group].segments.append(selection)
            result.groups[group].rows = checked_add(
                result.groups[group].rows, batch.count
            )
    for g in range(len(result.groups)):
        ref group = result.groups[g]
        group.first = len(result.descriptors) // SELECTION_WORDS
        group.count = len(group.segments) // 3
        group.partial = result.partials
        for i in range(group.count):
            var rows = Int(group.segments[i * 3])
            # The grid budget belongs to the whole submission. A segment
            # needs one workgroup at least, not a separate launch's floor.
            var partials = min(
                (rows + BLOCK_SIZE - 1) // BLOCK_SIZE,
                max(1, (rows * MAX_PARTIALS + group.rows - 1) // group.rows),
            )
            result.descriptors.append(UInt64(group.grid))
            result.descriptors.append(UInt64(partials))
            result.descriptors.append(UInt64(rows))
            result.descriptors.append(group.segments[i * 3 + 1])
            result.descriptors.append(group.segments[i * 3 + 2])
            result.descriptors.append(UInt64(result.partials))
            group.grid = checked_add(group.grid, partials)
            result.partials = checked_add(result.partials, partials)
        if group.first > Int(UInt32.MAX) or group.count > Int(UInt32.MAX):
            raise Error("Metal selection segment table is too large")
        group.segments = List[UInt64]()
    return result^


def map_selection_owner[
    D: DType
](
    series: Series,
    scope: Scope,
    empty: DeviceBuffer[DType.uint8],
    mut reservations: List[_NativeReservation],
) raises -> NumericView:
    ref column = series._data[Column[Scalar[D]]]
    var page = Int(external_call["getpagesize", Int32]())
    var count = checked_mul(len(column), size_of[Scalar[D]]())
    var address = Int(column.unsafe_values())
    if address % size_of[Scalar[D]]():
        raise Error("Metal selection source is not naturally aligned")
    var base = address - address % page
    var length = (
        checked_add(checked_add(address - base, count), page - 1) // page * page
    )
    var bit_length = len(column._bits[])
    var bit_address = 0
    var bit_base = 0
    var bit_size = 0
    if bit_length:
        bit_address = Int(column.unsafe_validity())
        bit_base = bit_address - bit_address % page
        bit_size = (
            checked_add(
                checked_add(bit_address - bit_base, bit_length), page - 1
            )
            // page
            * page
        )
    reservations.append(
        _NativeReservation(scope, checked_add(length, bit_size))
    )
    var ctx = scope.context()
    var mapped = ctx.wrap_host_memory[DType.uint8](
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base), length
    )
    var bits = empty.copy()
    if bit_length:
        bits = ctx.wrap_host_memory[DType.uint8](
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bit_base), bit_size
        )
    var tag = 0 if D == DType.int32 else (1 if D == DType.int64 else 2)
    return NumericView(
        mapped.create_sub_buffer[DType.uint8](address - base, count),
        bits^,
        (bit_address - bit_base) * 8,
        len(column),
        tag,
        bit_length != 0,
    )
