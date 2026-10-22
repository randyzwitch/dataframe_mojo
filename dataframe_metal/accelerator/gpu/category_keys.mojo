"""Explicit CPU categorical key identity over private resident text transport.

Matching temporarily uses unified CPU dictionary codes in LOW. Restoration
reinstates ordered label codes, and only JOIN changes categorical HIGH codes.
ASOF's comparison unification never changes its original output owners.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer, Pointer
from std.collections import Dict
from dataframe.column import Column
from dataframe.dtype import CategoricalDictionary, DataType
from dataframe.categorical import union_of
from dataframe_metal.accelerator.code_domains import (
    CategoryField,
    CategoryOwner,
    TextField,
    code,
)
from dataframe_metal.accelerator.joins import DagTask
from .residency import (
    Scope,
    TextDomain,
    Table,
    CategoryLease,
    checked_add,
    checked_mul,
)
from std.sys import size_of
from dataframe_metal.accelerator.code_domains import dictionary_bytes
from .target import METAL_TARGET


# Only dictionary-sized first-occurrence controls leave the device. Each
# control is a row position, never a row code or decoded row value.
def first_occurrence(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    output: Pointer[UInt32, MutAnyOrigin],
    base: UInt32,
    rows: UInt32,
    entries: UInt32,
):
    var entry = global_idx.x
    if entry >= Int(entries):
        return
    var first = UInt32.MAX
    for row in range(Int(rows)):
        var at = Int(base) + row
        if validity[unsafe_offset=at] and words[unsafe_offset=at] == UInt64(
            entry
        ):
            first = UInt32(row)
            break
    output[unsafe_offset=entry] = first


def _field(fields: List[TextField], slot: Int) -> Optional[TextField]:
    for field in fields:
        if field.name == String(slot):
            return field.copy()
    return None


def _dictionary(
    scope: Scope,
    field: TextField,
    domain: ArcPointer[TextDomain],
    words: DeviceBuffer[DType.uint64],
    validity: DeviceBuffer[DType.uint8],
    base: Int,
    rows: Int,
    workspace: Int,
) raises -> DataType:
    if field.category and not field.category.value().string_codes:
        return field.category.value().owner[].dtype
    if field.dtype != DataType.STRING:
        raise Error("Categorical comparison requires text provenance")
    var entries = len(domain[].dtype.dictionary()[])
    _ = scope.check(checked_add(workspace, checked_mul(entries, 24)))
    var controls: List[UInt32]
    if entries and rows:
        comptime if not METAL_TARGET:
            raise Error(
                "Categorical key controls require an Apple Metal target"
            )
        else:
            var ctx = scope.context()
            var device = ctx.enqueue_create_buffer[DType.uint32](entries)
            ctx.enqueue_function[first_occurrence](
                words.unsafe_ptr(),
                validity.unsafe_ptr(),
                device.unsafe_ptr(),
                UInt32(base),
                UInt32(rows),
                UInt32(entries),
                grid_dim=(entries + 255) // 256,
                block_dim=256,
            )
            scope.record_launches()
            controls = scope.read_control(device)
            scope.record_download(entries * 4)
    else:
        controls = List[UInt32](length=entries, fill=UInt32.MAX)
    # Sort dictionary indices by their first resident occurrence. The host
    # receives O(D) controls even when the source has many more than D rows.
    var positions = List[Tuple[UInt32, UInt32]](capacity=entries)
    for entry in range(entries):
        if controls[entry] != UInt32.MAX:
            positions.append((controls[entry], UInt32(entry)))
    sort(positions)
    var dictionary = CategoricalDictionary()
    for position in positions:
        dictionary.append(domain[].dtype.dictionary()[].get(Int(position[1])))
    return DataType.categorical(dictionary^)


struct Submission(Movable):
    var maps: List[UInt64]
    # base, rows, map offset, map length, source-is-category, write-category
    var descriptors: List[Int64]
    var device_maps: Optional[DeviceBuffer[DType.uint64]]
    var device_descriptors: Optional[DeviceBuffer[DType.int64]]
    var output_categories: List[Optional[CategoryField]]
    var cells: Int
    var leases: List[ArcPointer[CategoryLease]]

    def __init__(out self, task: DagTask):
        self.maps = List[UInt64]()
        self.descriptors = List[Int64]()
        self.device_maps = None
        self.device_descriptors = None
        self.output_categories = List[Optional[CategoryField]](
            length=len(task.fields), fill=None
        )
        self.cells = 0
        self.leases = List[ArcPointer[CategoryLease]]()

    def bytes(self) raises -> Int:
        return checked_add(
            checked_mul(self.maps.capacity(), 16),
            checked_mul(self.descriptors.capacity(), 16),
        )

    def prepare(
        mut self,
        scope: Scope,
        task: DagTask,
        fields: List[TextField],
        domain: ArcPointer[TextDomain],
        columns: List[Int64],
        words: DeviceBuffer[DType.uint64],
        validity: DeviceBuffer[DType.uint8],
        asof: Bool,
        workspace: Int,
    ) raises:
        self.cells = len(validity)
        # Upper bound includes maps for both original dictionaries, text-code
        # maps for String sides, output owner metadata, and union UTF8 bytes.
        var dictionary_payload = len(domain[].dtype.dictionary()[].bytes)
        var entries = len(domain[].dtype.dictionary()[])
        var original_entries = 0
        for field in fields:
            if field.category:
                original_entries = checked_add(
                    original_entries,
                    len(field.category.value().owner[].dtype.dictionary()[]),
                )
        var map_bound = checked_mul(
            checked_add(original_entries, checked_mul(entries, len(task.keys))),
            64,
        )
        _ = scope.check(
            checked_add(
                workspace,
                checked_add(
                    map_bound,
                    checked_mul(
                        dictionary_payload + entries * 8, len(task.keys)
                    ),
                ),
            )
        )
        for key in range(0, len(task.keys), 2):
            var left = Int(task.keys[key])
            var right = Int(task.keys[key + 1])
            var l = _field(fields, left)
            var r = _field(fields, right)
            if (
                not l
                or not r
                or (
                    not (
                        l.value().category
                        and not l.value().category.value().string_codes
                    )
                    and not (
                        r.value().category
                        and not r.value().category.value().string_codes
                    )
                )
            ):
                continue
            var ld = _dictionary(
                scope,
                l.value(),
                domain,
                words,
                validity,
                Int(columns[left * 3]),
                Int(columns[left * 3 + 1]),
                checked_add(workspace, self.bytes()),
            )
            var rd = _dictionary(
                scope,
                r.value(),
                domain,
                words,
                validity,
                Int(columns[right * 3]),
                Int(columns[right * 3 + 1]),
                checked_add(workspace, self.bytes()),
            )
            # Duplicate labels may occupy large repeated byte payloads in
            # the original dictionary. Charge the original dictionaries,
            # not merely the deduplicated label domain, BEFORE union copies.
            var payload_bound = checked_add(
                dictionary_bytes(ld), dictionary_bytes(rd)
            )
            _ = scope.check(
                checked_add(
                    checked_add(workspace, self.bytes()),
                    checked_add(
                        checked_mul(payload_bound, 3),
                        checked_mul(
                            len(ld.dictionary()[]) + len(rd.dictionary()[]), 64
                        ),
                    ),
                )
            )
            var same = (
                l.value().category
                and not l.value().category.value().string_codes
                and r.value().category
                and not r.value().category.value().string_codes
                and ld == rd
            )
            var unified = ld if same else DataType.categorical(
                union_of(ld.dictionary()[], rd.dictionary()[])
            )
            var lookup = Dict[String, UInt32]()
            for entry in range(len(unified.dictionary()[])):
                lookup[String(unified.dictionary()[].get(entry))] = UInt32(
                    entry
                )
            var owner = CategoryField(
                ArcPointer(CategoryOwner(unified, List[Column[UInt32]]()))
            )
            if not asof and not same:
                self.leases.append(
                    ArcPointer(
                        CategoryLease(
                            scope,
                            checked_add(
                                dictionary_bytes(unified),
                                size_of[CategoryOwner]()
                                + size_of[CategoryLease](),
                            ),
                        )
                    )
                )
            for side in range(2):
                var slot = right if side else left
                var field = r.value().copy() if side else l.value().copy()
                var source = rd if side else ld
                var categorical = Bool(field.category)
                var mapping = (
                    field.category.value()
                    .owner[]
                    .dtype if categorical else domain[]
                    .dtype
                )
                var offset = len(self.maps)
                for entry in range(len(mapping.dictionary()[])):
                    var label = String(mapping.dictionary()[].get(entry))
                    var selected = (
                        UInt32(entry) if not side
                        and categorical
                        and not field.category.value().string_codes else (
                            UInt32(entry) if same else (
                                lookup[label] if label in lookup else UInt32(0)
                            )
                        )
                    )
                    self.maps.append(UInt64(selected))
                    self.maps.append(UInt64(code(domain[].dtype, label)))
                self.descriptors.extend(
                    [
                        columns[slot * 3],
                        columns[slot * 3 + 1],
                        Int64(offset),
                        Int64(len(mapping.dictionary()[])),
                        Int64(categorical),
                        Int64(not asof),
                    ]
                )
                if not asof:
                    for output in range(len(task.fields)):
                        ref result = task.fields[output]
                        var selected_source = (
                            result.right if result.prefer_right
                            or result.left < 0 else result.left
                        )
                        if selected_source == slot:
                            self.output_categories[
                                output
                            ] = field.category.copy() if same else Optional[
                                CategoryField
                            ](
                                owner.copy()
                            )
        _ = scope.check(checked_add(workspace, self.bytes()))
        if not len(self.descriptors):
            return
        var ctx = scope.context()
        self.device_maps = ctx.enqueue_create_buffer[DType.uint64](
            len(self.maps)
        )
        self.device_descriptors = ctx.enqueue_create_buffer[DType.int64](
            len(self.descriptors)
        )
        ctx.enqueue_copy(self.device_maps.value(), self.maps.unsafe_ptr())
        ctx.enqueue_copy(
            self.device_descriptors.value(), self.descriptors.unsafe_ptr()
        )
        scope.record_upload(len(self.maps) * 8 + len(self.descriptors) * 8)
        self.enqueue(scope, words, validity, False)

    def enqueue(
        self,
        scope: Scope,
        words: DeviceBuffer[DType.uint64],
        validity: DeviceBuffer[DType.uint8],
        restore: Bool,
    ) raises:
        if not len(self.descriptors) or not self.cells:
            return
        comptime if not METAL_TARGET:
            raise Error(
                "Categorical key transport requires an Apple Metal target"
            )
        else:
            var ctx = scope.context()
            ctx.enqueue_function[transport](
                words.unsafe_ptr(),
                validity.unsafe_ptr(),
                self.device_maps.value().unsafe_ptr(),
                self.device_descriptors.value().unsafe_ptr(),
                UInt32(self.cells),
                UInt32(len(self.descriptors) // 6),
                UInt32(restore),
                grid_dim=(self.cells * (len(self.descriptors) // 6) + 255)
                // 256,
                block_dim=256,
            )
            scope.record_launches()

    def attach(self, mut result: Table):
        result.category_leases.extend(self.leases.copy())
        for output in range(len(self.output_categories)):
            if self.output_categories[output]:
                result.categories[output] = self.output_categories[
                    output
                ].copy()


def transport(
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    maps: Pointer[UInt64, ImmutAnyOrigin],
    descriptors: Pointer[Int64, ImmutAnyOrigin],
    cells: UInt32,
    columns: UInt32,
    restore: UInt32,
):
    var work = global_idx.x
    if work >= Int(cells) * Int(columns):
        return
    var descriptor = (work // Int(cells)) * 6
    var row = work % Int(cells)
    if row >= Int(descriptors[unsafe_offset=descriptor + 1]):
        return
    var at = Int(descriptors[unsafe_offset=descriptor]) + row
    if not validity[unsafe_offset=at]:
        return
    var original = (
        words[unsafe_offset=Int(cells) + at]
        - 1 if descriptors[unsafe_offset=descriptor + 4] else words[
            unsafe_offset=at
        ]
    )
    if restore:
        # LOW currently holds the unified code. Categorical HIGH still holds
        # the original index; String's original label code was saved in HIGH.
        original = words[unsafe_offset=Int(cells) + at] - 1
    if original >= UInt64(descriptors[unsafe_offset=descriptor + 3]):
        return
    var mapping = (
        Int(descriptors[unsafe_offset=descriptor + 2]) + Int(original) * 2
    )
    if not restore:
        if not descriptors[unsafe_offset=descriptor + 4]:
            words[unsafe_offset=Int(cells) + at] = original + 1
        words[unsafe_offset=at] = maps[unsafe_offset=mapping]
    else:
        words[unsafe_offset=at] = maps[unsafe_offset=mapping + 1]
        if descriptors[unsafe_offset=descriptor + 5]:
            words[unsafe_offset=Int(cells) + at] = (
                maps[unsafe_offset=mapping] + 1
            )
        elif not descriptors[unsafe_offset=descriptor + 4]:
            words[unsafe_offset=Int(cells) + at] = 0
