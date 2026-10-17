"""Mojo-owned immutable device tables and query allocation accounting.

Matrices store low words, high words, validity and deferred errors. A table
shares one matrix owner and only changes metadata when copied or sliced.
Consumers copy resident columns on the device; only readback exports values.
The MAX context is created lazily and always selects the Metal device API.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.atomic import Atomic
from std.memory import ArcPointer, Pointer
from std.sys import size_of
from dataframe import DataFrame, DataType, Series
from dataframe.dtype import NUMERIC_DTYPES
from .target import METAL_TARGET
from dataframe.accelerator.code_domains import (
    TextField,
    CategoryField,
    dictionary_bytes,
    same_dictionary,
    text_less,
)

from .grouped_key_owner import GroupedKeyField
from dataframe.accelerator.faults import data_fault

comptime CELL_BYTES = 21


def raise_value_error(code: UInt32, message: String) raises:
    """Report a deferred value fault without changing its failure category."""
    if code & 4:
        raise data_fault("Mojo GPU decimal multiplication overflow")
    if code & 8:
        raise data_fault("Mojo GPU decimal precision overflow")
    if code & 16:
        raise data_fault("Mojo GPU integer overflow")
    if code & 32:
        raise data_fault(
            "Mojo GPU integer power requires a nonnegative exponent"
        )
    if code & 64:
        raise data_fault("Mojo GPU temporal value overflows Int64")
    if code:
        raise data_fault(message)


def checked_add(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or a > Int.MAX - b:
        raise Error("Mojo GPU resident allocation size overflow")
    return a + b


def checked_mul(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or (b and a > Int.MAX // b):
        raise Error("Mojo GPU resident allocation size overflow")
    return a * b


# Metadata-only targets retain accounting and lease ownership without
# instantiating an unavailable DeviceContext destructor. Real Metal keeps
# the exact context owner type, including its queue lifetime and teardown.
comptime _ContextCell = Optional[DeviceContext if METAL_TARGET else Int]


struct _ScopeState(Movable):
    var context: _ContextCell
    var device: Int
    var budget: Int
    var live: Atomic[Int64]
    var peak: Atomic[Int64]
    var uploads: Atomic[Int64]
    var downloads: Atomic[Int64]
    var launches: Atomic[Int64]
    var waits: Atomic[Int64]
    var context_lock: Atomic[Int64]
    var context_ready: Atomic[Int64]

    def __init__(out self, device: Int, budget: Int):
        self.context = None
        self.device = device
        self.budget = budget
        self.live = Atomic[Int64](0)
        self.peak = Atomic[Int64](0)
        self.uploads = Atomic[Int64](0)
        self.downloads = Atomic[Int64](0)
        self.launches = Atomic[Int64](0)
        self.waits = Atomic[Int64](0)
        self.context_lock = Atomic[Int64](0)
        self.context_ready = Atomic[Int64](0)


struct Scope(Copyable):
    var _state: ArcPointer[_ScopeState]

    def __init__(out self, device_id: Int = 0, budget: Int = -1) raises:
        if device_id < 0 or budget < -1:
            raise Error("Mojo GPU resident device or budget is invalid")
        self._state = ArcPointer(_ScopeState(device_id, budget))

    @staticmethod
    def device_count() raises -> Int:
        return DeviceContext.number_of_devices(api="metal")

    def initialized(self) -> Bool:
        return self._state[].context_ready.load() != 0

    def context(self) raises -> DeviceContext:
        comptime if not METAL_TARGET:
            raise Error(
                "Mojo GPU resident context requires a compiled Metal target"
            )
        else:
            # Arc ownership is immutable externally. This one protected cell is
            # interior mutable so copied scopes initialize exactly the same queue.
            var unlocked = Int64(0)
            while not self._state[].context_lock.compare_exchange(unlocked, 1):
                unlocked = 0
            var state = Pointer(to=self._state[]).unsafe_origin_cast[
                MutAnyOrigin
            ]()
            try:
                if not state[].context:
                    state[].context = _ContextCell(
                        rebind[DeviceContext if METAL_TARGET else Int](
                            DeviceContext(state[].device, api="metal")
                        )
                    )
                    state[].context_ready.store(1)
                var result = rebind[DeviceContext](
                    state[].context.value()
                ).copy()
                state[].context_lock.store(0)
                return result^
            except error:
                state[].context_lock.store(0)
                raise error^

    def same_query(self, other: Self) -> Bool:
        return Int(Pointer(to=self._state[])) == Int(Pointer(to=other._state[]))

    def info(self) -> Tuple[Int, Int, Int]:
        return (
            Int(self._state[].live.load()),
            Int(self._state[].peak.load()),
            self._state[].budget,
        )

    def transfers(self) -> Tuple[Int, Int, Int, Int]:
        return (
            Int(self._state[].uploads.load()),
            Int(self._state[].downloads.load()),
            Int(self._state[].launches.load()),
            Int(self._state[].waits.load()),
        )

    def _record_peak(self, total: Int):
        var previous = self._state[].peak.load()
        while Int64(total) > previous:
            if self._state[].peak.compare_exchange(previous, Int64(total)):
                break

    def check(self, additional: Int) raises -> Int:
        var total = checked_add(Int(self._state[].live.load()), additional)
        if self._state[].budget >= 0 and total > self._state[].budget:
            raise Error(
                "Mojo GPU resident query exceeds memory budget before submission"
            )
        self._record_peak(total)
        return total

    def _retain(self, bytes: Int) raises:
        var previous = self._state[].live.load()
        while True:
            var total = checked_add(Int(previous), bytes)
            if self._state[].budget >= 0 and total > self._state[].budget:
                raise Error(
                    "Mojo GPU resident query exceeds memory budget before submission"
                )
            if self._state[].live.compare_exchange(previous, Int64(total)):
                self._record_peak(total)
                return

    def _release(self, bytes: Int):
        _ = self._state[].live.fetch_sub(Int64(bytes))

    def record_upload(self, bytes: Int):
        _ = self._state[].uploads.fetch_add(Int64(bytes))

    def record_download(self, bytes: Int):
        _ = self._state[].downloads.fetch_add(Int64(bytes))

    def record_launches(self, count: Int = 1):
        _ = self._state[].launches.fetch_add(Int64(count))

    def synchronize(self) raises:
        if self.initialized():
            self.context().synchronize()
            _ = self._state[].waits.fetch_add(1)

    def read_control_into[
        dtype: DType
    ](
        self, buffer: DeviceBuffer[dtype], mut values: List[Scalar[dtype]]
    ) raises:
        """Complete a control copy before its caller's host owner can unwind."""
        if len(values) != len(buffer):
            raise Error("Mojo GPU control readback shape mismatch")
        var ctx = self.context()
        try:
            ctx.enqueue_copy(values.unsafe_ptr(), buffer)
            self.synchronize()
        except error:
            self.synchronize()
            raise error^

    def read_control[
        dtype: DType
    ](self, buffer: DeviceBuffer[dtype]) raises -> List[Scalar[dtype]]:
        """Read shape/error metadata while keeping its host owner alive.

        Preserve the caller's existing synchronization point. Byte accounting
        remains with the executor, which knows this control record's meaning.
        """
        var values = List[Scalar[dtype]](length=len(buffer), fill=0)
        self.read_control_into(buffer, values)
        return values^

    def device_info(self) raises -> Tuple[Int, String, String]:
        var ctx = self.context()
        return (self._state[].device, ctx.name(), ctx.api())


struct Matrix(Movable):
    var scope: Scope
    var capacity: Int
    var slots: Int
    var cells: Int
    var retained: Int
    var words: DeviceBuffer[DType.uint64]
    var validity: DeviceBuffer[DType.uint8]
    var errors: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        scope: Scope,
        capacity: Int,
        slots: Int,
        var words: DeviceBuffer[DType.uint64],
        var validity: DeviceBuffer[DType.uint8],
        var errors: DeviceBuffer[DType.uint32],
    ) raises:
        var cells = checked_mul(capacity, slots)
        var retained = checked_mul(cells, CELL_BYTES)
        if (
            capacity < 0
            or capacity > Int(Int32.MAX)
            or slots < 0
            or slots > 256
            or len(words) != checked_mul(cells, 2)
            or len(validity) != cells
            or len(errors) != cells
        ):
            raise Error(
                "Mojo GPU resident matrix shape or buffer length mismatch"
            )
        # The shared executor checked live + complete region peak before it
        # allocated these buffers. Adoption charges only the retained owner.
        scope._retain(retained)
        self.scope = scope.copy()
        self.capacity = capacity
        self.slots = slots
        self.cells = cells
        self.retained = retained
        self.words = words^
        self.validity = validity^
        self.errors = errors^

    def __deinit__(deinit self):
        self.scope._release(self.retained)


@fieldwise_init
struct HostMatrix(Copyable):
    var schema: DataFrame
    var rows: Int
    var words: List[UInt64]
    var validity: List[UInt8]
    var errors: List[UInt32]


struct _Identity(Movable):
    """Owned structural finalizer values; never a CPU-evaluated region."""

    var scope: Scope
    var value: HostMatrix
    var retained: Int

    def __init__(out self, scope: Scope, var value: HostMatrix) raises:
        var cells = checked_mul(value.rows, value.schema.width())
        if (
            value.schema.height() != 0
            or value.rows < 0
            or value.rows > Int(Int32.MAX)
            or len(value.words) != cells * 2
            or len(value.validity) != cells
            or len(value.errors) != cells
        ):
            raise Error("Mojo GPU structural identity shape mismatch")
        var bytes = checked_mul(cells, CELL_BYTES)
        scope._retain(bytes)
        self.scope = scope.copy()
        self.value = value^
        self.retained = bytes

    def __deinit__(deinit self):
        self.scope._release(self.retained)


def _output_bytes(schema: DataFrame, rows: Int) raises -> Int:
    var bytes = 0
    for field in schema._columns:
        var width = 0
        if field.dtype().is_decimal() and field.dtype().decimal_width() == 128:
            width = 16
        elif field.dtype() == DataType.BOOL:
            width = 1
        else:
            comptime for k in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[k]
                if field.dtype().physical() == DataType.of(D):
                    width = size_of[Scalar[D]]()
        if width == 0:
            raise Error("Mojo GPU resident output storage is unsupported")
        bytes = checked_add(
            bytes, checked_add(checked_mul(rows, width), (rows + 7) // 8)
        )
    return checked_add(bytes, checked_mul(rows, 4 if schema.width() else 0))


struct TextDomain(Movable):
    """One scoped immutable ordered UTF8 dictionary, shared across columns."""

    var scope: Scope
    var dtype: DataType
    var retained: Int

    def __init__(
        out self, scope: Scope, dtype: DataType, charge: Bool = True
    ) raises:
        if not dtype.is_categorical():
            raise Error("Private text domain requires an ordered dictionary")
        var dictionary = dtype.dictionary()
        for index in range(1, len(dictionary[])):
            if not text_less(
                dictionary[].get(index - 1), dictionary[].get(index)
            ):
                raise Error(
                    "Private text domain dictionary must be strictly ordered"
                )
        var retained = dictionary_bytes(dtype) if charge else 0
        # Charge before self becomes fully initialized. A rejected allocation
        # must not run a destructor that releases an unretained owner.
        scope._retain(retained)
        self.scope = scope.copy()
        self.dtype = dtype
        self.retained = retained
        # Static query dictionaries are already charged by outer staging.

    def __deinit__(deinit self):
        self.scope._release(self.retained)


@fieldwise_init
struct TextColumn(Copyable):
    var logical: DataType
    var domain: ArcPointer[TextDomain]


struct CategoryLease(Movable):
    var scope: Scope
    var retained: Int

    def __init__(out self, scope: Scope, bytes: Int) raises:
        scope._retain(bytes)
        self.scope = scope.copy()
        self.retained = bytes

    def __deinit__(deinit self):
        self.scope._release(self.retained)


struct Table(Copyable):
    var storage: Optional[ArcPointer[Matrix]]
    var identity_storage: Optional[ArcPointer[_Identity]]
    var schema: DataFrame
    var columns: List[Int64]
    var rows: Int
    var offset: Int
    var text: List[Optional[TextColumn]]
    var categories: List[Optional[CategoryField]]
    var category_leases: List[ArcPointer[CategoryLease]]
    var grouped_keys: List[Optional[GroupedKeyField]]

    def __init__(
        out self,
        var matrix: Matrix,
        var schema: DataFrame,
        var columns: List[Int64],
        rows: Int,
        offset: Int = 0,
    ) raises:
        if (
            schema.height() != 0
            or schema.width() != len(columns)
            or rows < 0
            or offset < 0
            or checked_add(rows, offset) > matrix.capacity
        ):
            raise Error("Mojo GPU resident table schema or row range mismatch")
        for slot in columns:
            if slot < 0 or slot >= Int64(matrix.slots):
                raise Error(
                    "Mojo GPU resident table column is outside its matrix"
                )
        self.storage = ArcPointer(matrix^)
        self.identity_storage = None
        self.schema = schema^
        self.columns = columns^
        self.rows = rows
        self.offset = offset
        self.text = List[Optional[TextColumn]](capacity=self.schema.width())
        self.categories = List[Optional[CategoryField]](
            capacity=self.schema.width()
        )
        self.category_leases = List[ArcPointer[CategoryLease]]()
        self.grouped_keys = List[Optional[GroupedKeyField]](
            length=self.schema.width(), fill=None
        )
        for _ in range(self.schema.width()):
            self.text.append(None)
            self.categories.append(None)

    def __init__(out self, var schema: DataFrame) raises:
        if schema.height() != 0:
            raise Error(
                "Mojo GPU resident empty table requires metadata-only schema"
            )
        self.storage = None
        self.identity_storage = None
        self.schema = schema^
        self.columns = List[Int64]()
        self.rows = 0
        self.offset = 0
        self.text = List[Optional[TextColumn]](capacity=self.schema.width())
        self.categories = List[Optional[CategoryField]](
            capacity=self.schema.width()
        )
        self.category_leases = List[ArcPointer[CategoryLease]]()
        self.grouped_keys = List[Optional[GroupedKeyField]](
            length=self.schema.width(), fill=None
        )
        for _ in range(self.schema.width()):
            self.text.append(None)
            self.categories.append(None)

    @staticmethod
    def empty(var schema: DataFrame) raises -> Self:
        return Self(schema^)

    @staticmethod
    def rows_only(scope: Scope, rows: Int) raises -> Self:
        """Preserve zero-column cardinality without allocating a device."""
        return Self.identity(
            scope,
            HostMatrix(
                DataFrame(List[Series]()),
                rows,
                List[UInt64](),
                List[UInt8](),
                List[UInt32](),
            ),
        )

    @staticmethod
    def identity(scope: Scope, var host: HostMatrix) raises -> Self:
        """Retain empty-input finalizer output without initializing a GPU."""
        var result = Self(host.schema.copy())
        result.rows = host.rows
        result.columns = List[Int64](capacity=host.schema.width())
        for column in range(host.schema.width()):
            result.columns.append(Int64(column))
        result.identity_storage = ArcPointer(_Identity(scope, host^))
        return result^

    def attach_text(
        mut self, fields: List[TextField], domain: ArcPointer[TextDomain]
    ) raises:
        """Intersect original logical text metadata with the retained schema.

        Projection pruning may leave a node's pre-pruning text-field superset.
        Present fields retain strict dtype/scope checks; removed fields own no
        result column and must not create provenance or a dictionary reference.
        """
        if (
            self.storage
            and not self.storage.value()[].scope.same_query(domain[].scope)
        ) or (
            self.identity_storage
            and not self.identity_storage.value()[].scope.same_query(
                domain[].scope
            )
        ):
            raise Error(
                "Private text domain belongs to a different query scope"
            )
        for field in fields:
            if (
                field.dtype != DataType.STRING
                and not field.dtype.is_categorical()
            ):
                raise Error(
                    "Private text field must retain original String/categorical dtype"
                )
            var index = -1
            for column in range(self.schema.width()):
                if self.schema._columns[column].name() == field.name:
                    index = column
                    break
            if index < 0:
                continue
            if self.schema._columns[index].dtype() != DataType.UINT32:
                raise Error(
                    "Private text field does not have UInt32 transport storage"
                )
            self.text[index] = TextColumn(field.dtype, domain.copy())
            if not self.categories[index]:
                self.categories[index] = field.category.copy()

    def attach_categories(mut self, fields: List[TextField]):
        """Certify exact categorical identity; UInt32 alone proves nothing."""
        for field in fields:
            for index in range(self.schema.width()):
                if self.schema._columns[index].name() == field.name:
                    self.categories[index] = field.category.copy()

    def _text_bytes(self) -> Int:
        var total = 0
        var seen = List[DataType]()
        for field in self.text:
            if not field:
                continue
            var domain = field.value().domain
            var duplicate = False
            for old in seen:
                duplicate |= same_dictionary(old, domain[].dtype)
            if not duplicate:
                total += domain[].retained
                seen.append(domain[].dtype)
        return total

    def matrix(self) raises -> ArcPointer[Matrix]:
        if not self.storage:
            raise Error("Mojo GPU resident empty table has no device matrix")
        return self.storage.value().copy()

    def retained_bytes(self) -> Int:
        var payload = self.storage.value()[].retained if self.storage else (
            self.identity_storage.value()[].retained if self.identity_storage else 0
        )
        # Domains are already charged exactly once by their scoped Arc owner.
        # This per-table report includes each retained dictionary once.
        var categories = 0
        for index in range(len(self.category_leases)):
            var owner = self.category_leases[index]
            var duplicate = False
            for before in range(index):
                duplicate |= Int(
                    Pointer(to=self.category_leases[before][])
                ) == Int(Pointer(to=owner[]))
            if not duplicate:
                categories += owner[].retained
        var grouped = 0
        var seen = List[Int]()
        for field in self.grouped_keys:
            if not field:
                continue
            var owner = field.value().owner
            var identity = Int(Pointer(to=owner[]))
            var duplicate = False
            for old in seen:
                duplicate |= old == identity
            if not duplicate:
                seen.append(identity)
                grouped += (
                    owner[].staging[].bytes + owner[].source.retained_bytes()
                )
        return payload + self._text_bytes() + categories + grouped

    def same_storage(self, other: Self) -> Bool:
        if self.identity_storage and other.identity_storage:
            return Int(Pointer(to=self.identity_storage.value()[])) == Int(
                Pointer(to=other.identity_storage.value()[])
            )
        return (
            self.storage
            and other.storage
            and Int(Pointer(to=self.storage.value()[]))
            == Int(Pointer(to=other.storage.value()[]))
        )

    def head(self, count: Int) raises -> Self:
        if count < 0:
            raise Error("Head length must be nonnegative")
        var result = self.copy()
        result.rows = min(result.rows, count)
        return result^

    def observe_errors(
        self,
        message: String = "Mojo GPU arithmetic overflow or strict cast failed",
    ) raises:
        """Observe deferred failures without exporting values or validity."""
        var cells = checked_mul(self.rows, self.schema.width())
        if not cells:
            return
        if self.identity_storage:
            var owner = self.identity_storage.value().copy()
            for column in range(self.schema.width()):
                for row in range(self.rows):
                    var source = (
                        Int(self.columns[column]) * owner[].value.rows
                        + self.offset
                        + row
                    )
                    if owner[].value.errors[source]:
                        raise_value_error(owner[].value.errors[source], message)
            return
        if not self.storage:
            return
        var owner = self.matrix()
        _ = owner[].scope.check(checked_mul(cells, 4))
        var errors = List[UInt32](length=cells, fill=0)
        var ctx = owner[].scope.context()
        try:
            for column in range(self.schema.width()):
                var source = (
                    Int(self.columns[column]) * owner[].capacity + self.offset
                )
                var buffer = owner[].errors.create_sub_buffer[DType.uint32](
                    source, self.rows
                )
                ctx.enqueue_copy(
                    errors.unsafe_ptr().unsafe_offset(column * self.rows),
                    buffer,
                )
            owner[].scope.synchronize()
        except error:
            owner[].scope.synchronize()
            _ = errors^
            raise error^
        owner[].scope.record_download(cells * 4)
        for failure in errors:
            if failure:
                raise_value_error(failure, message)

    def readback(self) raises -> HostMatrix:
        var cells = checked_mul(self.rows, self.schema.width())
        if self.identity_storage:
            var owner = self.identity_storage.value().copy()
            _ = owner[].scope.check(
                checked_add(
                    checked_mul(cells, CELL_BYTES),
                    _output_bytes(self.schema, self.rows),
                )
            )
            var words = List[UInt64](length=cells * 2, fill=0)
            var validity = List[UInt8](length=cells, fill=0)
            var errors = List[UInt32](length=cells, fill=0)
            var source_cells = owner[].value.rows * owner[].value.schema.width()
            for column in range(self.schema.width()):
                for row in range(self.rows):
                    var source = (
                        Int(self.columns[column]) * owner[].value.rows
                        + self.offset
                        + row
                    )
                    var target = column * self.rows + row
                    words[target] = owner[].value.words[source]
                    words[cells + target] = owner[].value.words[
                        source_cells + source
                    ]
                    validity[target] = owner[].value.validity[source]
                    errors[target] = owner[].value.errors[source]
                    if errors[target]:
                        raise_value_error(
                            errors[target],
                            "Mojo GPU resident arithmetic overflow, strict cast failed or input expression failure",
                        )
            return HostMatrix(
                self.schema.copy(), self.rows, words^, validity^, errors^
            )
        if not self.storage:
            return HostMatrix(
                self.schema.copy(),
                0,
                List[UInt64](),
                List[UInt8](),
                List[UInt32](),
            )
        var owner = self.matrix()
        var host_bytes = checked_mul(cells, CELL_BYTES)
        _ = owner[].scope.check(
            checked_add(host_bytes, _output_bytes(self.schema, self.rows))
        )
        var words = List[UInt64](length=cells * 2, fill=0)
        var validity = List[UInt8](length=cells, fill=0)
        var errors = List[UInt32](length=cells, fill=0)
        var ctx = owner[].scope.context()
        try:
            for column in range(self.schema.width()):
                var source = (
                    Int(self.columns[column]) * owner[].capacity + self.offset
                )
                var target = column * self.rows
                if not self.rows:
                    continue
                var low = owner[].words.create_sub_buffer[DType.uint64](
                    source, self.rows
                )
                var high = owner[].words.create_sub_buffer[DType.uint64](
                    owner[].cells + source, self.rows
                )
                var mask = owner[].validity.create_sub_buffer[DType.uint8](
                    source, self.rows
                )
                var failure = owner[].errors.create_sub_buffer[DType.uint32](
                    source, self.rows
                )
                ctx.enqueue_copy(words.unsafe_ptr().unsafe_offset(target), low)
                ctx.enqueue_copy(
                    words.unsafe_ptr().unsafe_offset(cells + target), high
                )
                ctx.enqueue_copy(
                    validity.unsafe_ptr().unsafe_offset(target), mask
                )
                ctx.enqueue_copy(
                    errors.unsafe_ptr().unsafe_offset(target), failure
                )
            owner[].scope.synchronize()
        except error:
            owner[].scope.synchronize()
            _ = words^
            _ = validity^
            _ = errors^
            raise error^
        owner[].scope.record_download(host_bytes)
        for error in errors:
            if error:
                raise_value_error(
                    error,
                    "Mojo GPU resident arithmetic overflow, strict cast failed or input expression failure",
                )
        return HostMatrix(
            self.schema.copy(), self.rows, words^, validity^, errors^
        )

    def copy_to(
        self,
        scope: Scope,
        words: DeviceBuffer[DType.uint64],
        validity: DeviceBuffer[DType.uint8],
        errors: DeviceBuffer[DType.uint32],
        capacity: Int,
        slots: Int,
        mapping: List[Int64],
        target_start: Int = 0,
    ) raises -> Int:
        var target_cells = checked_mul(capacity, slots)
        if (
            capacity < self.rows
            or target_start < 0
            or checked_add(len(mapping), target_start) > slots
            or len(words) != target_cells * 2
            or len(validity) != target_cells
            or len(errors) != target_cells
        ):
            raise Error("Mojo GPU resident input target shape mismatch")
        for column in mapping:
            if column < 0 or column >= Int64(self.schema.width()):
                raise Error("Mojo GPU resident input column mapping is invalid")
        if not self.rows or not len(mapping):
            return 0
        if self.identity_storage:
            var owner = self.identity_storage.value().copy()
            if not owner[].scope.same_query(scope):
                raise Error(
                    "Mojo GPU resident input belongs to another query context"
                )
            var ctx = scope.context()
            var source_cells = owner[].value.rows * owner[].value.schema.width()
            try:
                for target in range(len(mapping)):
                    var source = (
                        Int(self.columns[Int(mapping[target])])
                        * owner[].value.rows
                        + self.offset
                    )
                    var at = (target_start + target) * capacity
                    var low = words.create_sub_buffer[DType.uint64](
                        at, self.rows
                    )
                    var high = words.create_sub_buffer[DType.uint64](
                        target_cells + at, self.rows
                    )
                    var mask = validity.create_sub_buffer[DType.uint8](
                        at, self.rows
                    )
                    var failure = errors.create_sub_buffer[DType.uint32](
                        at, self.rows
                    )
                    ctx.enqueue_copy(
                        low,
                        owner[].value.words.unsafe_ptr().unsafe_offset(source),
                    )
                    ctx.enqueue_copy(
                        high,
                        owner[]
                        .value.words.unsafe_ptr()
                        .unsafe_offset(source_cells + source),
                    )
                    ctx.enqueue_copy(
                        mask,
                        owner[]
                        .value.validity.unsafe_ptr()
                        .unsafe_offset(source),
                    )
                    ctx.enqueue_copy(
                        failure,
                        owner[].value.errors.unsafe_ptr().unsafe_offset(source),
                    )
                scope.record_upload(self.rows * len(mapping) * CELL_BYTES)
                scope.synchronize()
            except error:
                scope.synchronize()
                _ = owner^
                raise error^
            _ = owner^
            return 0
        var owner = self.matrix()
        if not owner[].scope.same_query(scope):
            raise Error(
                "Mojo GPU resident input belongs to another query context"
            )
        var descriptors = List[Int64](capacity=len(mapping) * 3)
        for target in range(len(mapping)):
            var column = Int(mapping[target])
            ref dtype = self.schema._columns[column]._dtype
            descriptors.extend(
                [
                    self.columns[column],
                    Int64(target_start + target),
                    Int64(
                        (dtype.is_decimal() and dtype.decimal_width() == 128)
                        or Bool(self.categories[column])
                    ),
                ]
            )
        _ = scope.check(checked_mul(len(descriptors), 16))
        var ctx = scope.context()
        try:
            var device_descriptors = ctx.enqueue_create_buffer[DType.int64](
                len(descriptors)
            )
            ctx.enqueue_copy(device_descriptors, descriptors.unsafe_ptr())
            scope.record_upload(len(descriptors) * 8)
            var output_words = words.copy()
            var output_validity = validity.copy()
            var output_errors = errors.copy()
            ctx.enqueue_function[copy_columns](
                owner[].words.unsafe_ptr(),
                owner[].validity.unsafe_ptr(),
                owner[].errors.unsafe_ptr(),
                output_words.unsafe_ptr(),
                output_validity.unsafe_ptr(),
                output_errors.unsafe_ptr(),
                device_descriptors.unsafe_ptr(),
                UInt32(len(mapping)),
                UInt32(self.rows),
                UInt32(self.offset),
                UInt32(owner[].capacity),
                UInt32(owner[].slots),
                UInt32(capacity),
                UInt32(slots),
                grid_dim=(self.rows * len(mapping) + 255) // 256,
                block_dim=256,
            )
            # Copy descriptors and host source owners survive their queued reads.
            scope.record_launches()
            scope.synchronize()
        except error:
            scope.synchronize()
            _ = descriptors^
            raise error^
        # Raw copy pointers do not retain the source List across queued work.
        _ = descriptors^
        return 1


def copy_columns(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    descriptors: Pointer[Int64, ImmutAnyOrigin],
    columns: UInt32,
    rows: UInt32,
    offset: UInt32,
    source_capacity: UInt32,
    source_slots: UInt32,
    capacity: UInt32,
    slots: UInt32,
):
    var work = global_idx.x
    if work >= Int(rows) * Int(columns):
        return
    var column = work // Int(rows)
    var row = work % Int(rows)
    var source = (
        Int(descriptors[unsafe_offset=column * 3]) * Int(source_capacity)
        + Int(offset)
        + row
    )
    var target = (
        Int(descriptors[unsafe_offset=column * 3 + 1]) * Int(capacity) + row
    )
    output_words[unsafe_offset=target] = words[unsafe_offset=source]
    output_words[unsafe_offset=Int(capacity) * Int(slots) + target] = words[
        unsafe_offset=Int(source_capacity) * Int(source_slots) + source
    ] if descriptors[unsafe_offset=column * 3 + 2] else UInt64(0)
    output_validity[unsafe_offset=target] = validity[unsafe_offset=source]
    output_errors[unsafe_offset=target] = errors[unsafe_offset=source]
