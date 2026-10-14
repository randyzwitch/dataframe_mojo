"""Immutable text dictionary unions and resident code remapping.

Only dictionary metadata and GPU-produced edge controls reach host functions.
Input row codes/values are consumed by remap_columns directly on the device.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer, Pointer
from std.sys import size_of
from dataframe.dtype import CategoricalDictionary, DataType
from dataframe.frame import DataFrame
from dataframe.accelerator.code_domains import (
    code,
    dictionary_bytes,
    same_dictionary,
    text_less,
    TextField,
)
from .target import METAL_TARGET
from .residency import (
    Scope,
    TextDomain,
    TextColumn,
    Table,
    checked_add,
    checked_mul,
)


def _same_values(left: DataType, right: DataType) -> Bool:
    return left.dictionary()[].same_values(right.dictionary()[])


def _heap_less(
    domains: List[DataType], positions: List[Int], a: Int, b: Int
) -> Bool:
    return text_less(
        domains[a].dictionary()[].get(positions[a]),
        domains[b].dictionary()[].get(positions[b]),
    )


def _sift(
    domains: List[DataType],
    positions: List[Int],
    mut heap: List[Int],
    root: Int,
):
    var at = root
    while at * 2 + 1 < len(heap):
        var chosen = at * 2 + 1
        if chosen + 1 < len(heap) and _heap_less(
            domains, positions, heap[chosen + 1], heap[chosen]
        ):
            chosen += 1
        if not _heap_less(domains, positions, heap[chosen], heap[at]):
            return
        var held = heap[at]
        heap[at] = heap[chosen]
        heap[chosen] = held
        at = chosen


def _merge_pass[
    Write: Bool
](
    domains: List[DataType], mut dictionary: CategoricalDictionary
) raises -> Tuple[Int, Int]:
    var positions = List[Int](length=len(domains), fill=0)
    var heap = List[Int](capacity=len(domains))
    for index in range(len(domains)):
        if len(domains[index].dictionary()[]):
            heap.append(index)
    for index in range(len(heap) // 2 - 1, -1, -1):
        _sift(domains, positions, heap, index)
    var previous_domain = -1
    var previous_index = -1
    var entries = 0
    var bytes = 0
    while len(heap):
        var source = heap[0]
        var index = positions[source]
        var value = domains[source].dictionary()[].get(index)
        if previous_domain < 0 or value != domains[
            previous_domain
        ].dictionary()[].get(previous_index):
            entries = checked_add(entries, 1)
            bytes = checked_add(bytes, value.byte_length())
            comptime if Write:
                dictionary.append(value)
            previous_domain = source
            previous_index = index
        positions[source] += 1
        if positions[source] == len(domains[source].dictionary()[]):
            var last = heap.pop()
            if len(heap):
                heap[0] = last
        if len(heap):
            _sift(domains, positions, heap, 0)
    return (entries, bytes)


def union_domain(
    scope: Scope,
    base_domain: ArcPointer[TextDomain],
    inputs: List[Optional[TextColumn]],
    workspace: Int = 0,
) raises -> ArcPointer[TextDomain]:
    if not base_domain[].scope.same_query(scope):
        raise Error("Private text domain belongs to a different query scope")
    var metadata = checked_add(
        workspace, checked_mul(len(inputs) + 1, size_of[DataType]())
    )
    _ = scope.check(metadata)
    var domains = List[DataType](capacity=len(inputs) + 1)
    domains.append(base_domain[].dtype)
    for field in inputs:
        if not field:
            continue
        var owner = field.value().domain
        if not owner[].scope.same_query(scope):
            raise Error(
                "Private text domain belongs to a different query scope"
            )
        var found = False
        for old in domains:
            found |= same_dictionary(old, owner[].dtype)
        if not found:
            domains.append(owner[].dtype)
    if len(domains) == 1:
        return base_domain.copy()
    # Two heap/position arrays of Int, with explicit capacity, in each pass.
    var temporary = checked_add(metadata, checked_mul(len(domains), 16))
    _ = scope.check(temporary)
    var ignored = CategoricalDictionary(List[UInt8](), List[Int64]())
    var shape = _merge_pass[False](domains, ignored)
    if shape[0] > Int(UInt32.MAX):
        raise Error("Private text union exceeds UInt32 code capacity")
    if shape[0] == len(base_domain[].dtype.dictionary()[]) and shape[1] == len(
        base_domain[].dtype.dictionary()[].bytes
    ):
        return base_domain.copy()
    for field in inputs:
        if not field:
            continue
        var owner = field.value().domain
        if shape[0] == len(owner[].dtype.dictionary()[]) and shape[1] == len(
            owner[].dtype.dictionary()[].bytes
        ):
            # Every input is included in this union. Equal entry/byte counts
            # therefore prove an existing immutable superset is the union.
            return owner.copy()
    var payload = checked_add(shape[1], checked_mul(shape[0] + 1, 8))
    _ = scope.check(checked_add(temporary, payload))
    var bytes = List[UInt8](capacity=shape[1])
    var offsets = List[Int64](capacity=shape[0] + 1)
    offsets.append(0)
    var dictionary = CategoricalDictionary(bytes^, offsets^)
    _ = _merge_pass[True](domains, dictionary)
    var dtype = DataType.categorical(dictionary^)
    return ArcPointer(TextDomain(scope, dtype, True))


@fieldwise_init
struct ColumnMap(Copyable):
    var slot: Int
    var rows: Int
    var source: ArcPointer[TextDomain]


struct MapPlan(Movable):
    var words: List[UInt32]
    var descriptors: List[Int64]
    var target: ArcPointer[TextDomain]
    var largest_slot: Int
    var largest_rows: Int

    def __init__(
        out self, target: ArcPointer[TextDomain], columns: List[ColumnMap]
    ) raises:
        self.target = target.copy()
        self.largest_slot = -1
        self.largest_rows = 0
        var map_words = 0
        var mapped = 0
        var unique = 0
        for index in range(len(columns)):
            ref column = columns[index]
            if (
                column.slot < 0
                or column.rows < 0
                or column.slot > Int(UInt32.MAX)
                or column.rows > Int(UInt32.MAX)
                or not column.source[].scope.same_query(target[].scope)
            ):
                raise Error(
                    "Private text remap descriptor/query scope is invalid"
                )
            self.largest_slot = max(self.largest_slot, column.slot)
            self.largest_rows = max(self.largest_rows, column.rows)
            if _same_values(column.source[].dtype, target[].dtype):
                continue
            mapped += 1
            var duplicate = False
            for before in range(index):
                duplicate |= same_dictionary(
                    columns[before].source[].dtype, column.source[].dtype
                )
            if not duplicate:
                map_words = checked_add(
                    map_words, len(column.source[].dtype.dictionary()[])
                )
                unique += 1
        # Constructor control owners coexist, but die before device maps.
        # Include the borrowed ColumnMap list explicitly, even for unchanged
        # fields that need no device remap descriptor.
        var phase = checked_add(
            checked_mul(map_words, 8), checked_mul(mapped, 64)
        )
        var construction = checked_add(
            checked_mul(columns.capacity(), size_of[ColumnMap]()),
            checked_add(
                checked_mul(unique, size_of[ArcPointer[TextDomain]]() + 8),
                checked_add(checked_mul(map_words, 4), checked_mul(mapped, 32)),
            ),
        )
        _ = target[].scope.check(max(phase, construction))
        var owners = List[ArcPointer[TextDomain]](capacity=unique)
        for column in columns:
            if _same_values(column.source[].dtype, target[].dtype):
                continue
            var duplicate = False
            for old in owners:
                duplicate |= same_dictionary(old[].dtype, column.source[].dtype)
            if not duplicate:
                owners.append(column.source.copy())
        self.words = List[UInt32](capacity=map_words)
        self.descriptors = List[Int64](capacity=mapped * 4)
        var offsets = List[Int](capacity=len(owners))
        for old in owners:
            offsets.append(len(self.words))
            var dictionary = old[].dtype.dictionary()
            for index in range(len(dictionary[])):
                self.words.append(code(target[].dtype, dictionary[].get(index)))
        for column in columns:
            if _same_values(column.source[].dtype, target[].dtype):
                continue
            var offset = -1
            for index in range(len(owners)):
                if same_dictionary(
                    owners[index][].dtype, column.source[].dtype
                ):
                    offset = offsets[index]
                    break
            if offset < 0 or column.slot < 0 or column.rows < 0:
                raise Error("Private text remap descriptor is invalid")
            self.descriptors.extend(
                [
                    Int64(column.slot),
                    Int64(offset),
                    Int64(len(column.source[].dtype.dictionary()[])),
                    Int64(column.rows),
                ]
            )

    def validate_shape(
        self, scope: Scope, stride: Int, words: Int, validity: Int, errors: Int
    ) raises:
        """Validate raw buffer addressing before any context or submission."""
        if not self.target[].scope.same_query(scope):
            raise Error("Private text remap belongs to a different query scope")
        if (
            stride < 0
            or stride > Int(UInt32.MAX)
            or words < 0
            or validity < 0
            or errors < 0
            or words != checked_mul(validity, 2)
            or errors != validity
            or len(self.descriptors) % 4
            or len(self.descriptors) // 4 > Int(UInt32.MAX)
            or (stride == 0 and validity != 0)
            or (stride > 0 and validity % stride)
        ):
            raise Error("Private text remap matrix shape is invalid")
        if self.largest_rows > stride or (
            stride > 0 and self.largest_slot >= validity // stride
        ):
            raise Error("Private text remap descriptor is outside its matrix")
        _ = checked_mul(stride, len(self.descriptors) // 4)
        for at in range(0, len(self.descriptors), 4):
            var slot = Int(self.descriptors[at])
            var offset = Int(self.descriptors[at + 1])
            var count = Int(self.descriptors[at + 2])
            var rows = Int(self.descriptors[at + 3])
            if (
                slot < 0
                or slot > Int(UInt32.MAX)
                or rows < 0
                or rows > stride
                or offset < 0
                or count < 0
                or checked_add(offset, count) > len(self.words)
                or (stride > 0 and slot >= validity // stride)
            ):
                raise Error(
                    "Private text remap descriptor is outside its matrix"
                )

    def bytes(self) raises -> Int:
        return checked_add(
            checked_mul(len(self.words), 8),
            checked_mul(len(self.descriptors), 16),
        )


struct Submission(Movable):
    var host: MapPlan
    var maps: Optional[DeviceBuffer[DType.uint32]]
    var descriptors: Optional[DeviceBuffer[DType.int64]]

    def __init__(out self, var plan: MapPlan):
        self.host = plan^
        self.maps = None
        self.descriptors = None

    def enqueue(
        mut self,
        scope: Scope,
        words: DeviceBuffer[DType.uint64],
        validity: DeviceBuffer[DType.uint8],
        errors: DeviceBuffer[DType.uint32],
        stride: Int,
    ) raises:
        self.host.validate_shape(
            scope, stride, len(words), len(validity), len(errors)
        )
        if len(self.host.descriptors) == 0 or stride == 0:
            return
        comptime if not METAL_TARGET:
            raise Error(
                "Private text remap requires an Apple Metal compilation target"
            )
        else:
            var context = scope.context()
            if len(self.host.words):
                self.maps = context.enqueue_create_buffer[DType.uint32](
                    len(self.host.words)
                )
            self.descriptors = context.enqueue_create_buffer[DType.int64](
                len(self.host.descriptors)
            )
            if len(self.host.words):
                context.enqueue_copy(
                    self.maps.value(), self.host.words.unsafe_ptr()
                )
            context.enqueue_copy(
                self.descriptors.value(), self.host.descriptors.unsafe_ptr()
            )
            scope.record_upload(
                len(self.host.words) * 4 + len(self.host.descriptors) * 8
            )
            var columns = len(self.host.descriptors) // 4
            var map_pointer = (
                words.unsafe_ptr()
                .unsafe_bitcast[UInt32]()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
            )
            if self.maps:
                map_pointer = (
                    self.maps.value()
                    .unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin]()
                )
            context.enqueue_function[remap_columns](
                words.unsafe_ptr(),
                validity.unsafe_ptr(),
                errors.unsafe_ptr(),
                map_pointer,
                self.descriptors.value().unsafe_ptr(),
                UInt32(stride),
                UInt32(columns),
                grid_dim=(stride * columns + 255) // 256,
                block_dim=256,
            )
            scope.record_launches()
            # The caller must retain this complete owner until its existing
            # synchronize on BOTH success/catch paths. No additional wait here.


def remap_columns(
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    maps: Pointer[UInt32, ImmutAnyOrigin],
    descriptors: Pointer[Int64, ImmutAnyOrigin],
    stride: UInt32,
    columns: UInt32,
):
    var work = global_idx.x
    if work >= Int(stride) * Int(columns):
        return
    var column = work // Int(stride)
    var row = work % Int(stride)
    var descriptor = column * 4
    if row >= Int(descriptors[unsafe_offset=descriptor + 3]):
        return
    var at = Int(descriptors[unsafe_offset=descriptor]) * Int(stride) + row
    if validity[unsafe_offset=at] == 0:
        return
    var value = words[unsafe_offset=at]
    if value >= UInt64(descriptors[unsafe_offset=descriptor + 2]):
        errors[unsafe_offset=at] |= 1
        return
    words[unsafe_offset=at] = UInt64(
        maps[
            unsafe_offset=Int(descriptors[unsafe_offset=descriptor + 1])
            + Int(value)
        ]
    )


def submit(
    scope: Scope,
    var plan: MapPlan,
    words: DeviceBuffer[DType.uint64],
    validity: DeviceBuffer[DType.uint8],
    errors: DeviceBuffer[DType.uint32],
    stride: Int,
) raises -> Submission:
    var result = Submission(plan^)
    try:
        result.enqueue(scope, words, validity, errors, stride)
        return result^
    except error:
        scope.synchronize()
        _ = result^
        raise error^


def source_maps(
    schema: DataFrame,
    fields: List[TextField],
    base_domain: ArcPointer[TextDomain],
    resident: Optional[Table],
    rows: Int,
    target_start: Int = 0,
) raises -> List[ColumnMap]:
    _ = base_domain[].scope.check(
        checked_mul(len(fields), size_of[ColumnMap]())
    )
    var result = List[ColumnMap](capacity=len(fields))
    for index in range(schema.width()):
        ref column = schema._columns[index]
        var logical = Optional[DataType]()
        for field in fields:
            if field.name == column.name():
                logical = field.dtype
                break
        if not logical:
            continue
        if column.dtype() != DataType.UINT32:
            raise Error("Private text input requires UInt32 transport storage")
        var source = base_domain.copy()
        if resident:
            var found = resident.value().schema._index(column.name())
            if (
                not resident.value().text[found]
                or resident.value().text[found].value().logical
                != logical.value()
            ):
                raise Error(
                    "Private resident text input lost its original logical provenance"
                )
            source = resident.value().text[found].value().domain.copy()
        result.append(ColumnMap(index + target_start, rows, source^))
    return result^
