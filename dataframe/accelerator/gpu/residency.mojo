"""Mojo-owned immutable device tables and query allocation accounting.

Matrices store low words, high words, validity and deferred errors. A table
shares one matrix owner and only changes metadata when copied or sliced.
Consumers copy resident columns on the device; only readback exports values.
The MAX context is created lazily and always selects the Metal device API.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.atomic import Atomic
from std.memory import ArcPointer, Pointer, bitcast
from std.sys import size_of
from dataframe import DataFrame, DataType, Series
from dataframe.dtype import NUMERIC_DTYPES
from .target import METAL_TARGET
from .query_scope import DeviceService, Scope, checked_add, checked_mul
from .typed_storage import (
    NativeColumn,
    NativeColumnAny,
    _NativeReservation,
    allocate_native_column,
)
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
comptime NATIVE_MISSING = UInt32.MAX


@fieldwise_init
struct NativeJoinField(Copyable):
    """Output sources for a pair-map native gather.

    `left` and `right` are source schema column indices, or -1 when absent.
    `prefer_right` implements JoinField's coalescing precedence. The pair-map
    certificate determines whether a single-sided output needs null extension.
    The schema passed to `Table.gather_join` owns output names and metadata.
    """

    var left: Int
    var right: Int
    var prefer_right: Bool


struct _NativeRowMapReservation(Movable):
    """Keep a row map's query-memory charge shared with copied map handles."""

    var owner: _NativeReservation

    def __init__(out self, var owner: _NativeReservation):
        self.owner = owner^


struct NativeRowMap(Copyable):
    """A retained device row map plus its producer's range certificate.

    `source_rows` and `may_be_missing` are shape facts from pair generation;
    values are still range-checked on device before publishing the result.
    """

    var scope: Scope
    var values: DeviceBuffer[DType.uint32]
    var rows: Int
    var source_rows: Int
    var may_be_missing: Bool
    var _reservation: ArcPointer[_NativeRowMapReservation]

    def __init__(
        out self,
        scope: Scope,
        var values: DeviceBuffer[DType.uint32],
        rows: Int,
        source_rows: Int,
        may_be_missing: Bool,
        var reservation: Optional[_NativeReservation] = None,
    ) raises:
        if (
            rows < 0
            or rows > Int(Int32.MAX)
            or source_rows < 0
            or source_rows > Int(Int32.MAX)
            or len(values) < rows
        ):
            raise Error("Mojo GPU native row map shape is invalid")
        var bytes = checked_mul(len(values), size_of[UInt32]())
        if reservation:
            if reservation.value().bytes != bytes or not scope.same_query(
                reservation.value().scope
            ):
                raise Error("Mojo GPU native row map reservation mismatch")
        else:
            _ = scope.check(bytes)
            reservation = Optional[_NativeReservation](
                _NativeReservation(scope, bytes)
            )
        self.scope = scope.copy()
        self.values = values^
        self.rows = rows
        self.source_rows = source_rows
        self.may_be_missing = may_be_missing
        self._reservation = ArcPointer(
            _NativeRowMapReservation(reservation.take())
        )


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


def _native_join_output_bytes(
    logical: DataType, rows: Int, validity: Bool, errors: Bool
) raises -> Int:
    if (
        logical.is_decimal()
        or logical.is_categorical()
        or logical.is_nested()
        or logical == DataType.BOOL
    ):
        raise Error("Mojo GPU native join payload dtype is unsupported")
    var width = 0
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if logical.physical() == DataType.of(D):
            width = size_of[Scalar[D]]()
    if not width:
        raise Error("Mojo GPU native join payload dtype is unsupported")
    var bytes = checked_mul(rows, width)
    if validity:
        bytes = checked_add(bytes, checked_add(rows, 7) // 8)
    if errors:
        bytes = checked_add(bytes, checked_mul(rows, size_of[UInt32]()))
    return bytes


def _validate_native_join_source(
    text: Bool,
    category: Bool,
    grouped_key: Bool,
    column: NativeColumnAny,
    logical: DataType,
    offset: Int,
    rows: Int,
) raises:
    if text or category or grouped_key:
        raise Error("Mojo GPU native join cannot carry text metadata")
    if (
        column.logical != logical
        or logical.is_decimal()
        or logical.is_categorical()
        or logical.is_nested()
        or logical == DataType.BOOL
        or not (logical.is_numeric() or logical.is_temporal())
    ):
        raise Error(
            "Mojo GPU native join source dtype differs or is unsupported"
        )
    if offset > column.length or rows > column.length - offset:
        raise Error("Mojo GPU native join source view is outside its owner")


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
    var native_columns: List[NativeColumnAny]
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
        self.native_columns = List[NativeColumnAny]()
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
        self.native_columns = List[NativeColumnAny]()
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
    def native(
        var schema: DataFrame,
        var columns: List[NativeColumnAny],
    ) raises -> Self:
        if not len(columns) or schema.width() != len(columns):
            raise Error("Mojo GPU native table requires matching columns")
        var rows = columns[0].length
        var scope = columns[0].scope()
        for i in range(len(columns)):
            if (
                columns[i].length != rows
                or schema._columns[i].dtype() != columns[i].logical
                or not scope.same_query(columns[i].scope())
            ):
                raise Error("Mojo GPU native table column mismatch")
        var result = Self(schema^)
        result.rows = rows
        result.native_columns = columns^
        for i in range(len(result.native_columns)):
            result.columns.append(Int64(i))
        return result^

    @staticmethod
    def native_float32(
        var schema: DataFrame,
        var columns: List[NativeColumn[DType.float32]],
    ) raises -> Self:
        """Compatibility constructor for the original Float32 row route."""
        var erased = List[NativeColumnAny](capacity=len(columns))
        for column in columns:
            erased.append(NativeColumnAny(column))
        return Self.native(schema^, erased^)

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
                    "Private text field must retain original String/categorical"
                    " dtype"
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
        for i in range(len(self.native_columns)):
            var duplicate = False
            for j in range(i):
                duplicate |= self.native_columns[i].owner_identity() == (
                    self.native_columns[j].owner_identity()
                )
            if not duplicate:
                payload += self.native_columns[i].retained_bytes()
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
        if len(self.native_columns) or len(other.native_columns):
            if len(self.native_columns) != len(other.native_columns):
                return False
            for i in range(len(self.native_columns)):
                if self.native_columns[i].owner_identity() != (
                    other.native_columns[i].owner_identity()
                ):
                    return False
            return True
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

    def project_view(
        self, scope: Scope, schema: DataFrame, mapping: List[Int], rows: Int
    ) raises -> Self:
        """Rename/reorder immutable resident columns without copying a matrix.

        Storage, deferred errors and provenance retain their original owners.
        The caller must establish that no expression or text remap is needed.
        """
        if (
            schema.height()
            or schema.width() != len(mapping)
            or rows < 0
            or rows > self.rows
        ):
            raise Error("Mojo GPU resident projection shape mismatch")
        if len(self.native_columns):
            if not self.native_columns[0].scope().same_query(scope):
                raise Error(
                    "Mojo GPU typed input belongs to another query context"
                )
        elif self.storage:
            if not self.storage.value()[].scope.same_query(scope):
                raise Error(
                    "Mojo GPU resident input belongs to another query context"
                )
        elif self.identity_storage:
            if not self.identity_storage.value()[].scope.same_query(scope):
                raise Error(
                    "Mojo GPU resident input belongs to another query context"
                )
        else:
            raise Error("Mojo GPU resident projection requires a storage owner")
        var result = self.copy()
        result.schema = schema.copy()
        result.rows = rows
        result.columns = List[Int64](capacity=len(mapping))
        result.text = List[Optional[TextColumn]](capacity=len(mapping))
        result.categories = List[Optional[CategoryField]](capacity=len(mapping))
        result.grouped_keys = List[Optional[GroupedKeyField]](
            capacity=len(mapping)
        )
        for output in range(len(mapping)):
            var input = mapping[output]
            if input < 0 or input >= self.schema.width():
                raise Error(
                    "Mojo GPU resident projection column is outside its schema"
                )
            if (
                schema._columns[output].dtype()
                != self.schema._columns[input].dtype()
            ):
                raise Error(
                    "Mojo GPU resident projection cannot change a column dtype"
                )
            result.columns.append(self.columns[input])
            result.text.append(self.text[input].copy())
            result.categories.append(self.categories[input].copy())
            result.grouped_keys.append(self.grouped_keys[input].copy())
        return result^

    def gather(
        self,
        scope: Scope,
        indices: List[Int],
        *,
        allow_missing: Bool = False,
    ) raises -> Self:
        """Gather native columns with one shared uploaded row map.

        This materializes each output column once. It is not a lazy selection:
        the result owns fresh typed buffers, validity and error sidecars. The
        shared index map and all source owners remain live until the one final
        scope drain.
        """
        if (
            not len(self.native_columns)
            or len(self.columns) != self.schema.width()
            or self.rows > Int(Int32.MAX)
            or len(indices) > Int(Int32.MAX)
        ):
            raise Error("Mojo GPU native gather requires a fixed-width table")
        if not self.native_columns[0].scope().same_query(scope):
            raise Error("Mojo GPU native gather belongs to another query")
        for row in indices:
            if row == -1 and allow_missing:
                continue
            if row < 0 or row >= self.rows:
                raise Error("Mojo GPU native gather index is outside its table")

        var index_bytes = checked_mul(len(indices), size_of[Int32]())
        var payload = checked_add(index_bytes, index_bytes)
        for output in range(self.schema.width()):
            var source = Int(self.columns[output])
            if source < 0 or source >= len(self.native_columns):
                raise Error("Mojo GPU native gather owner mapping is invalid")
            ref column = self.native_columns[source]
            if column.logical == DataType.BOOL:
                raise Error("Native Bool gather is not implemented")
            if (
                self.text[output]
                or self.categories[output]
                or self.grouped_keys[output]
            ):
                raise Error("Mojo GPU native gather cannot carry text metadata")
            payload = checked_add(
                payload, column.gather_bytes(len(indices), allow_missing)
            )
        _ = scope.check(payload)

        var host_reservation = _NativeReservation(scope, index_bytes)
        var device_reservation = _NativeReservation(scope, index_bytes)
        var host_indices = List[Int32](capacity=len(indices))
        for row in indices:
            host_indices.append(Int32(row))
        var device_indices = Optional[DeviceBuffer[DType.int32]]()
        if len(indices):
            device_indices = Optional[DeviceBuffer[DType.int32]](
                scope.context().enqueue_create_buffer[DType.int32](len(indices))
            )
        var outputs = List[NativeColumnAny](capacity=self.schema.width())
        var queued = False
        try:
            if len(indices):
                scope.context().enqueue_copy(
                    device_indices.value(), host_indices.unsafe_ptr()
                )
                scope.record_upload(index_bytes)
                queued = True
                for output in range(self.schema.width()):
                    var source = Int(self.columns[output])
                    ref column = self.native_columns[source]
                    var destination = _gather_native_column(
                        scope,
                        column,
                        self.offset,
                        self.rows,
                        device_indices.value(),
                        len(indices),
                        allow_missing,
                    )
                    outputs.append(destination^)
                scope.synchronize()
            else:
                for output in range(self.schema.width()):
                    ref column = self.native_columns[Int(self.columns[output])]
                    var destination = _empty_native_column(scope, column)
                    outputs.append(destination^)
        except error:
            if queued:
                scope.synchronize()
            _ = outputs^
            _ = device_indices^
            _ = host_indices^
            _ = device_reservation^
            _ = host_reservation^
            raise error^
        _ = device_indices^
        _ = host_indices^
        _ = device_reservation^
        _ = host_reservation^
        return Self.native(self.schema.copy(), outputs^)

    def gather_join(
        self,
        scope: Scope,
        right: Self,
        schema: DataFrame,
        fields: List[NativeJoinField],
        left_map: NativeRowMap,
        right_map: NativeRowMap,
    ) raises -> Self:
        """Gather two resident inputs through validated device-side pair maps.

        Pair maps remain UInt32 device owners throughout. All field gathers
        enqueue against both source tables and maps, then share one small
        status read and one final drain. `schema` is the caller's fully formed
        output schema, including field metadata and coalesced-key choices.
        """
        if (
            not len(self.native_columns)
            or not len(right.native_columns)
            or len(self.columns) != self.schema.width()
            or len(right.columns) != right.schema.width()
            or schema.height() != 0
            or schema.width() == 0
            or schema.width() != len(fields)
            or self.rows > Int(Int32.MAX)
            or right.rows > Int(Int32.MAX)
            or left_map.rows != right_map.rows
            or left_map.rows > Int(Int32.MAX)
            or left_map.source_rows != self.rows
            or right_map.source_rows != right.rows
            or (left_map.rows > 0 and self.rows == 0 and right.rows == 0)
        ):
            raise Error("Mojo GPU native join gather shape is invalid")
        if (
            not self.native_columns[0].scope().same_query(scope)
            or not right.native_columns[0].scope().same_query(scope)
            or not left_map.scope.same_query(scope)
            or not right_map.scope.same_query(scope)
        ):
            raise Error("Mojo GPU native join gather belongs to another query")

        # Validate every source and calculate the exact output sidecars before
        # the first output or status buffer is allocated.
        var bytes = 0
        for output in range(len(fields)):
            ref field = fields[output]
            ref output_field = schema._columns[output]
            if (
                field.left < -1
                or field.left >= self.schema.width()
                or field.right < -1
                or field.right >= right.schema.width()
                or (field.left < 0 and field.right < 0)
            ):
                raise Error("Mojo GPU native join output source is invalid")
            if (
                left_map.rows
                and field.left >= 0
                and field.right < 0
                and (self.rows == 0 and not left_map.may_be_missing)
            ):
                raise Error("Mojo GPU native join left source cannot be empty")
            if (
                left_map.rows
                and field.right >= 0
                and field.left < 0
                and (right.rows == 0 and not right_map.may_be_missing)
            ):
                raise Error("Mojo GPU native join right source cannot be empty")
            var left_possible = field.left >= 0 and (
                self.rows > 0
                and (
                    not field.prefer_right
                    or field.right < 0
                    or right_map.may_be_missing
                )
            )
            var right_possible = field.right >= 0 and (
                right.rows > 0
                and (
                    field.prefer_right
                    or field.left < 0
                    or left_map.may_be_missing
                )
            )
            var valid_sidecar = False
            var error_sidecar = False
            if field.left >= 0:
                var source = Int(self.columns[field.left])
                if source < 0 or source >= len(self.native_columns):
                    raise Error("Mojo GPU native join left owner is invalid")
                ref column = self.native_columns[source]
                _validate_native_join_source(
                    Bool(self.text[field.left]),
                    Bool(self.categories[field.left]),
                    Bool(self.grouped_keys[field.left]),
                    column,
                    output_field.dtype(),
                    self.offset,
                    self.rows,
                )
                if left_possible and self.rows > 0:
                    valid_sidecar |= column.has_validity()
                    error_sidecar |= column.has_errors()
            if field.right >= 0:
                var source = Int(right.columns[field.right])
                if source < 0 or source >= len(right.native_columns):
                    raise Error("Mojo GPU native join right owner is invalid")
                ref column = right.native_columns[source]
                _validate_native_join_source(
                    Bool(right.text[field.right]),
                    Bool(right.categories[field.right]),
                    Bool(right.grouped_keys[field.right]),
                    column,
                    output_field.dtype(),
                    right.offset,
                    right.rows,
                )
                if right_possible and right.rows > 0:
                    valid_sidecar |= column.has_validity()
                    error_sidecar |= column.has_errors()
            var nullable = (
                field.left >= 0 and field.right < 0 and left_map.may_be_missing
            ) or (
                field.right >= 0 and field.left < 0 and right_map.may_be_missing
            )
            bytes = checked_add(
                bytes,
                _native_join_output_bytes(
                    output_field.dtype(),
                    left_map.rows,
                    valid_sidecar or nullable,
                    error_sidecar,
                ),
            )
        _ = scope.check(checked_add(bytes, 8 if left_map.rows else 0))

        if not left_map.rows:
            var empty = List[NativeColumnAny](capacity=len(fields))
            for output in range(len(fields)):
                ref field = fields[output]
                var selected = (
                    self.native_columns[
                        Int(self.columns[field.left])
                    ].copy() if field.left
                    >= 0 else right.native_columns[
                        Int(right.columns[field.right])
                    ].copy()
                )
                empty.append(_empty_native_column(scope, selected))
            return Self.native(schema.copy(), empty^)

        # Four device bytes and four host control bytes live through the
        # shared final status drain.
        var status_reservation = _NativeReservation(scope, 8)
        var ctx = scope.context()
        var status = ctx.enqueue_create_buffer[DType.uint32](1)
        var outputs = List[NativeColumnAny](capacity=len(fields))
        var host_status = List[UInt32](length=1, fill=0)
        var queued = True
        try:
            ctx.enqueue_memset(status, UInt32(0))
            ctx.enqueue_function[validate_native_join_maps](
                left_map.values.unsafe_ptr(),
                right_map.values.unsafe_ptr(),
                status.unsafe_ptr(),
                UInt32(self.rows),
                UInt32(right.rows),
                UInt32(left_map.rows),
                UInt8(left_map.may_be_missing),
                UInt8(right_map.may_be_missing),
                grid_dim=(left_map.rows + 255) // 256,
                block_dim=256,
            )
            scope.record_launches()
            for output in range(len(fields)):
                ref field = fields[output]
                var left_column = Optional[NativeColumnAny]()
                var right_column = Optional[NativeColumnAny]()
                if field.left >= 0:
                    left_column = Optional[NativeColumnAny](
                        self.native_columns[
                            Int(self.columns[field.left])
                        ].copy()
                    )
                if field.right >= 0:
                    right_column = Optional[NativeColumnAny](
                        right.native_columns[
                            Int(right.columns[field.right])
                        ].copy()
                    )
                var left_possible = field.left >= 0 and (
                    self.rows > 0
                    and (
                        not field.prefer_right
                        or field.right < 0
                        or right_map.may_be_missing
                    )
                )
                var right_possible = field.right >= 0 and (
                    right.rows > 0
                    and (
                        field.prefer_right
                        or field.left < 0
                        or left_map.may_be_missing
                    )
                )
                var nullable = (
                    field.left >= 0
                    and field.right < 0
                    and left_map.may_be_missing
                ) or (
                    field.right >= 0
                    and field.left < 0
                    and right_map.may_be_missing
                )
                outputs.append(
                    _native_join_gather_column(
                        scope,
                        left_column,
                        right_column,
                        field.prefer_right,
                        left_possible,
                        right_possible,
                        nullable,
                        self.offset,
                        self.rows,
                        right.offset,
                        right.rows,
                        left_map.values,
                        right_map.values,
                        left_map.rows,
                        status,
                        schema._columns[output].dtype(),
                        queued,
                    )
                    ^
                )
            ctx.enqueue_copy(host_status.unsafe_ptr(), status)
            scope.record_download(4)
            scope.synchronize()
            queued = False
            if host_status[0] != 0:
                raise Error(
                    "Mojo GPU native join row map is outside its bounds"
                )
        except error:
            if queued:
                scope.synchronize()
            _ = outputs^
            _ = status^
            _ = status_reservation^
            _ = host_status^
            raise error^
        _ = host_status^
        _ = status^
        _ = status_reservation^
        return Self.native(schema.copy(), outputs^)

    def observe_errors(
        self,
        message: String = "Mojo GPU arithmetic overflow or strict cast failed",
    ) raises:
        """Observe deferred failures without exporting values or validity."""
        var cells = checked_mul(self.rows, self.schema.width())
        if not cells:
            return
        if len(self.native_columns):
            for column in self.columns:
                var view = self.native_columns[Int(column)].view(
                    self.offset, self.rows
                )
                for failure in view.read_error_codes():
                    if failure:
                        raise_value_error(failure, message)
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
        if len(self.native_columns):
            raise Error("Mojo GPU typed table requires native materialization")
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
                            (
                                "Mojo GPU resident arithmetic overflow, strict"
                                " cast failed or input expression failure"
                            ),
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
                    (
                        "Mojo GPU resident arithmetic overflow, strict cast"
                        " failed or input expression failure"
                    ),
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
        if len(self.native_columns):
            if self.schema.width() != len(self.columns):
                raise Error("Mojo GPU native table column mapping is invalid")
            var target_words = words.copy()
            try:
                # One table call shares the row selection and drains once. The
                # table keeps every native source owner alive across all
                # queued column conversions.
                for output in range(len(mapping)):
                    var source = Int(mapping[output])
                    if source < 0 or source >= self.schema.width():
                        raise Error(
                            "Mojo GPU native input column mapping is invalid"
                        )
                    var index = Int(self.columns[source])
                    if index < 0 or index >= len(self.native_columns):
                        raise Error("Mojo GPU native owner mapping is invalid")
                    ref column = self.native_columns[index]
                    if (
                        self.text[source]
                        or self.categories[source]
                        or self.grouped_keys[source]
                    ):
                        raise Error(
                            "Mojo GPU native input cannot carry text metadata"
                        )
                    if column._storage.isa[NativeColumn[DType.bool]]():
                        raise Error("Native Bool transport is not implemented")
                    if column._storage.isa[NativeColumn[DType.int64]]():
                        _copy_native_to_matrix[DType.int64](
                            scope,
                            column._storage[NativeColumn[DType.int64]],
                            self.offset,
                            self.rows,
                            target_start + output,
                            capacity,
                            slots,
                            target_cells,
                            target_words,
                            validity,
                            errors,
                        )
                    elif column._storage.isa[NativeColumn[DType.float64]]():
                        _copy_native_to_matrix[DType.float64](
                            scope,
                            column._storage[NativeColumn[DType.float64]],
                            self.offset,
                            self.rows,
                            target_start + output,
                            capacity,
                            slots,
                            target_cells,
                            target_words,
                            validity,
                            errors,
                        )
                    elif column._storage.isa[NativeColumn[DType.int8]]():
                        _copy_native_to_matrix[DType.int8](
                            scope,
                            column._storage[NativeColumn[DType.int8]],
                            self.offset,
                            self.rows,
                            target_start + output,
                            capacity,
                            slots,
                            target_cells,
                            target_words,
                            validity,
                            errors,
                        )
                    elif column._storage.isa[NativeColumn[DType.int16]]():
                        _copy_native_to_matrix[DType.int16](
                            scope,
                            column._storage[NativeColumn[DType.int16]],
                            self.offset,
                            self.rows,
                            target_start + output,
                            capacity,
                            slots,
                            target_cells,
                            target_words,
                            validity,
                            errors,
                        )
                    elif column._storage.isa[NativeColumn[DType.int32]]():
                        _copy_native_to_matrix[DType.int32](
                            scope,
                            column._storage[NativeColumn[DType.int32]],
                            self.offset,
                            self.rows,
                            target_start + output,
                            capacity,
                            slots,
                            target_cells,
                            target_words,
                            validity,
                            errors,
                        )
                    elif column._storage.isa[NativeColumn[DType.uint8]]():
                        _copy_native_to_matrix[DType.uint8](
                            scope,
                            column._storage[NativeColumn[DType.uint8]],
                            self.offset,
                            self.rows,
                            target_start + output,
                            capacity,
                            slots,
                            target_cells,
                            target_words,
                            validity,
                            errors,
                        )
                    elif column._storage.isa[NativeColumn[DType.uint16]]():
                        _copy_native_to_matrix[DType.uint16](
                            scope,
                            column._storage[NativeColumn[DType.uint16]],
                            self.offset,
                            self.rows,
                            target_start + output,
                            capacity,
                            slots,
                            target_cells,
                            target_words,
                            validity,
                            errors,
                        )
                    elif column._storage.isa[NativeColumn[DType.uint32]]():
                        _copy_native_to_matrix[DType.uint32](
                            scope,
                            column._storage[NativeColumn[DType.uint32]],
                            self.offset,
                            self.rows,
                            target_start + output,
                            capacity,
                            slots,
                            target_cells,
                            target_words,
                            validity,
                            errors,
                        )
                    elif column._storage.isa[NativeColumn[DType.uint64]]():
                        _copy_native_to_matrix[DType.uint64](
                            scope,
                            column._storage[NativeColumn[DType.uint64]],
                            self.offset,
                            self.rows,
                            target_start + output,
                            capacity,
                            slots,
                            target_cells,
                            target_words,
                            validity,
                            errors,
                        )
                    else:
                        _copy_native_to_matrix[DType.float32](
                            scope,
                            column._storage[NativeColumn[DType.float32]],
                            self.offset,
                            self.rows,
                            target_start + output,
                            capacity,
                            slots,
                            target_cells,
                            target_words,
                            validity,
                            errors,
                        )
                scope.synchronize()
            except error:
                scope.synchronize()
                _ = target_words^
                raise error^
            _ = target_words^
            return checked_mul(checked_mul(self.rows, len(mapping)), CELL_BYTES)
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


def copy_native_column[
    dtype: DType, nullable: Bool, has_errors: Bool
](
    values: Pointer[Scalar[dtype], ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    error_offset: UInt32,
    validity_offset: UInt32,
    rows: UInt32,
    low: Pointer[UInt64, MutAnyOrigin],
    high: Pointer[UInt64, MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var present = True
    comptime if nullable:
        var bit = Int(validity_offset) + row
        present = (validity[unsafe_offset=bit // 8] >> UInt8(bit % 8)) & 1 != 0
    var word = UInt64(0)
    if present:
        comptime if dtype == DType.float64:
            word = bitcast[DType.uint64](values[unsafe_offset=row])
        elif dtype == DType.float32:
            word = UInt64(bitcast[DType.uint32](values[unsafe_offset=row]))
        else:
            word = values[unsafe_offset=row].cast[DType.uint64]()
    low[unsafe_offset=row] = word
    high[unsafe_offset=row] = 0
    output_validity[unsafe_offset=row] = UInt8(present)
    comptime if has_errors:
        output_errors[unsafe_offset=row] = errors[
            unsafe_offset=Int(error_offset) + row
        ]
    else:
        output_errors[unsafe_offset=row] = 0


def gather_native_column[
    dtype: DType
](
    source: Pointer[Scalar[dtype], ImmutAnyOrigin],
    source_validity: Pointer[UInt8, ImmutAnyOrigin],
    source_errors: Pointer[UInt32, ImmutAnyOrigin],
    indices: Pointer[Int32, ImmutAnyOrigin],
    source_offset: UInt32,
    validity_offset: UInt32,
    error_offset: UInt32,
    source_has_validity: UInt8,
    source_has_errors: UInt8,
    output_has_validity: UInt8,
    rows: UInt32,
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
):
    """Gather eight output rows per lane, packing output validity once."""
    var byte = global_idx.x
    var bytes = (Int(rows) + 7) // 8
    if byte >= bytes:
        return
    var mask = UInt8(0)
    for bit in range(8):
        var row = byte * 8 + bit
        if row >= Int(rows):
            break
        var selected = indices[unsafe_offset=row]
        var present = selected >= 0
        if present and source_has_validity != 0:
            var source_bit = Int(validity_offset) + Int(selected)
            present = (
                (
                    source_validity[unsafe_offset=source_bit // 8]
                    >> UInt8(source_bit % 8)
                )
                & 1
            ) != 0
        if selected >= 0:
            output[unsafe_offset=row] = source[
                unsafe_offset=Int(source_offset) + Int(selected)
            ] if present else Scalar[dtype](0)
        else:
            output[unsafe_offset=row] = Scalar[dtype](0)
        if present:
            mask |= UInt8(1) << UInt8(bit)
        if source_has_errors != 0:
            output_errors[unsafe_offset=row] = source_errors[
                unsafe_offset=Int(error_offset) + Int(selected)
            ] if selected >= 0 else UInt32(0)
    if output_has_validity != 0:
        output_validity[unsafe_offset=byte] = mask


def _empty_native_column(
    scope: Scope, column: NativeColumnAny
) raises -> NativeColumnAny:
    var result: NativeColumnAny
    if column._storage.isa[NativeColumn[DType.int64]]():
        result = NativeColumnAny(
            allocate_native_column[DType.int64](scope, column.logical, 0)
        )
    elif column._storage.isa[NativeColumn[DType.float64]]():
        result = NativeColumnAny(
            allocate_native_column[DType.float64](scope, column.logical, 0)
        )
    elif column._storage.isa[NativeColumn[DType.bool]]():
        result = NativeColumnAny(
            allocate_native_column[DType.bool](scope, column.logical, 0)
        )
    elif column._storage.isa[NativeColumn[DType.int8]]():
        result = NativeColumnAny(
            allocate_native_column[DType.int8](scope, column.logical, 0)
        )
    elif column._storage.isa[NativeColumn[DType.int16]]():
        result = NativeColumnAny(
            allocate_native_column[DType.int16](scope, column.logical, 0)
        )
    elif column._storage.isa[NativeColumn[DType.int32]]():
        result = NativeColumnAny(
            allocate_native_column[DType.int32](scope, column.logical, 0)
        )
    elif column._storage.isa[NativeColumn[DType.uint8]]():
        result = NativeColumnAny(
            allocate_native_column[DType.uint8](scope, column.logical, 0)
        )
    elif column._storage.isa[NativeColumn[DType.uint16]]():
        result = NativeColumnAny(
            allocate_native_column[DType.uint16](scope, column.logical, 0)
        )
    elif column._storage.isa[NativeColumn[DType.uint32]]():
        result = NativeColumnAny(
            allocate_native_column[DType.uint32](scope, column.logical, 0)
        )
    elif column._storage.isa[NativeColumn[DType.uint64]]():
        result = NativeColumnAny(
            allocate_native_column[DType.uint64](scope, column.logical, 0)
        )
    else:
        result = NativeColumnAny(
            allocate_native_column[DType.float32](scope, column.logical, 0)
        )
    return result^


def validate_native_join_maps(
    left: Pointer[UInt32, ImmutAnyOrigin],
    right: Pointer[UInt32, ImmutAnyOrigin],
    status: Pointer[UInt32, MutAnyOrigin],
    left_rows: UInt32,
    right_rows: UInt32,
    rows: UInt32,
    left_missing_allowed: UInt8,
    right_missing_allowed: UInt8,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var l = left[unsafe_offset=row]
    var r = right[unsafe_offset=row]
    var fault = UInt32(0)
    if l == NATIVE_MISSING:
        if left_missing_allowed == 0:
            fault = 1
    elif l >= left_rows:
        fault = 1
    if r == NATIVE_MISSING:
        if right_missing_allowed == 0:
            fault = 1
    elif r >= right_rows:
        fault = 1
    if l == NATIVE_MISSING and r == NATIVE_MISSING:
        fault = 2
    if fault:
        Atomic[UInt32].max(status, fault)


def native_join_gather_column[
    dtype: DType
](
    left_values: Pointer[Scalar[dtype], ImmutAnyOrigin],
    left_validity: Pointer[UInt8, ImmutAnyOrigin],
    left_errors: Pointer[UInt32, ImmutAnyOrigin],
    right_values: Pointer[Scalar[dtype], ImmutAnyOrigin],
    right_validity: Pointer[UInt8, ImmutAnyOrigin],
    right_errors: Pointer[UInt32, ImmutAnyOrigin],
    left_indices: Pointer[UInt32, ImmutAnyOrigin],
    right_indices: Pointer[UInt32, ImmutAnyOrigin],
    status: Pointer[UInt32, MutAnyOrigin],
    left_value_offset: UInt32,
    left_validity_offset: UInt32,
    left_error_offset: UInt32,
    right_value_offset: UInt32,
    right_validity_offset: UInt32,
    right_error_offset: UInt32,
    left_rows: UInt32,
    right_rows: UInt32,
    rows: UInt32,
    has_left: UInt8,
    has_right: UInt8,
    left_possible: UInt8,
    right_possible: UInt8,
    left_has_validity: UInt8,
    right_has_validity: UInt8,
    left_has_errors: UInt8,
    right_has_errors: UInt8,
    prefer_right: UInt8,
    output_has_validity: UInt8,
    output_has_errors: UInt8,
    output_nullable: UInt8,
    output_values: Pointer[Scalar[dtype], MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
):
    """Coalesced typed gather; one lane packs one validity byte."""
    var byte = global_idx.x
    var bytes = (Int(rows) + 7) // 8
    if byte >= bytes:
        return
    var mask = UInt8(0)
    for bit in range(8):
        var row = byte * 8 + bit
        if row >= Int(rows):
            break
        var l = left_indices[unsafe_offset=row]
        var r = right_indices[unsafe_offset=row]
        var selected = NATIVE_MISSING
        var use_right = False
        if prefer_right != 0 and has_right != 0 and r != NATIVE_MISSING:
            selected = r
            use_right = True
        elif has_left != 0 and l != NATIVE_MISSING:
            selected = l
        elif has_right != 0 and r != NATIVE_MISSING:
            selected = r
            use_right = True
        if selected == NATIVE_MISSING:
            if output_nullable == 0:
                Atomic[UInt32].max(status, 2)
        else:
            var candidate_possible = (
                right_possible != 0 if use_right else left_possible != 0
            )
            var limit = right_rows if use_right else left_rows
            if not candidate_possible or selected >= limit:
                Atomic[UInt32].max(status, 1)
                selected = NATIVE_MISSING
        var present = selected != NATIVE_MISSING
        var valid = present
        var error = UInt32(0)
        var value = Scalar[dtype](0)
        if present and use_right:
            if right_has_validity != 0:
                var source_bit = Int(right_validity_offset) + Int(selected)
                valid = (
                    (
                        right_validity[unsafe_offset=source_bit // 8]
                        >> UInt8(source_bit % 8)
                    )
                    & 1
                ) != 0
            if valid:
                value = right_values[
                    unsafe_offset=Int(right_value_offset) + Int(selected)
                ]
            if right_has_errors != 0:
                error = right_errors[
                    unsafe_offset=Int(right_error_offset) + Int(selected)
                ]
        elif present:
            if left_has_validity != 0:
                var source_bit = Int(left_validity_offset) + Int(selected)
                valid = (
                    (
                        left_validity[unsafe_offset=source_bit // 8]
                        >> UInt8(source_bit % 8)
                    )
                    & 1
                ) != 0
            if valid:
                value = left_values[
                    unsafe_offset=Int(left_value_offset) + Int(selected)
                ]
            if left_has_errors != 0:
                error = left_errors[
                    unsafe_offset=Int(left_error_offset) + Int(selected)
                ]
        output_values[unsafe_offset=row] = value
        if output_has_errors != 0:
            output_errors[unsafe_offset=row] = error
        if valid:
            mask |= UInt8(1) << UInt8(bit)
    if output_has_validity != 0:
        output_validity[unsafe_offset=byte] = mask


def _native_join_gather_typed[
    dtype: DType
](
    scope: Scope,
    left: NativeColumn[dtype],
    right: NativeColumn[dtype],
    has_left: Bool,
    has_right: Bool,
    left_possible: Bool,
    right_possible: Bool,
    prefer_right: Bool,
    nullable: Bool,
    left_offset: Int,
    left_rows: Int,
    right_offset: Int,
    right_rows: Int,
    left_indices: DeviceBuffer[DType.uint32],
    right_indices: DeviceBuffer[DType.uint32],
    rows: Int,
    status: DeviceBuffer[DType.uint32],
    mut queued: Bool,
) raises -> NativeColumn[dtype]:
    var left_view = left.view(left_offset, left_rows)
    var right_view = right.view(right_offset, right_rows)
    var output_has_validity = (
        nullable
        or (left_possible and left_view.has_validity())
        or (right_possible and right_view.has_validity())
    )
    var output_has_errors = (left_possible and left_view.has_errors()) or (
        right_possible and right_view.has_errors()
    )
    var output = allocate_native_column[dtype](
        scope,
        left.logical,
        rows,
        with_validity=output_has_validity,
        with_errors=output_has_errors,
    )
    var left_values = (
        left_view.values() if left_view.length else right_view.values()
    )
    var right_values = (
        right_view.values() if right_view.length else left_view.values()
    )
    var left_validity = left_values.create_sub_buffer[DType.uint8](0, 1)
    var left_validity_buffer = Optional[DeviceBuffer[DType.uint8]]()
    if left_view.has_validity():
        left_validity_buffer = left_view.validity()
        left_validity = left_validity_buffer.value().copy()
    var right_validity = right_values.create_sub_buffer[DType.uint8](0, 1)
    var right_validity_buffer = Optional[DeviceBuffer[DType.uint8]]()
    if right_view.has_validity():
        right_validity_buffer = right_view.validity()
        right_validity = right_validity_buffer.value().copy()
    var left_errors = status.create_sub_buffer[DType.uint32](0, 1)
    var left_errors_buffer = Optional[DeviceBuffer[DType.uint32]]()
    if left_view.has_errors():
        left_errors_buffer = left_view.errors()
        left_errors = left_errors_buffer.value().copy()
    var right_errors = status.create_sub_buffer[DType.uint32](0, 1)
    var right_errors_buffer = Optional[DeviceBuffer[DType.uint32]]()
    if right_view.has_errors():
        right_errors_buffer = right_view.errors()
        right_errors = right_errors_buffer.value().copy()
    var output_values = output.values()
    var output_validity = output_values.create_sub_buffer[DType.uint8](0, 1)
    var output_validity_buffer = output.validity()
    if output_validity_buffer:
        output_validity = output_validity_buffer.value().copy()
    var output_errors = status.create_sub_buffer[DType.uint32](0, 1)
    var output_errors_buffer = output.errors()
    if output_errors_buffer:
        output_errors = output_errors_buffer.value().copy()
    var ctx = scope.context()
    try:
        # Float64 payloads move as bits; Metal cannot load native doubles.
        comptime transport_dtype = DType.uint64 if dtype == DType.float64 else dtype
        ctx.enqueue_function[native_join_gather_column[transport_dtype]](
            left_values.create_sub_buffer[transport_dtype](0, len(left_values)),
            left_validity,
            left_errors,
            right_values.create_sub_buffer[transport_dtype](
                0, len(right_values)
            ),
            right_validity,
            right_errors,
            left_indices,
            right_indices,
            status,
            UInt32(left_view.value_offset),
            UInt32(left_view.validity_bit_offset),
            UInt32(left_view.value_offset),
            UInt32(right_view.value_offset),
            UInt32(right_view.validity_bit_offset),
            UInt32(right_view.value_offset),
            UInt32(left_rows),
            UInt32(right_rows),
            UInt32(rows),
            UInt8(has_left),
            UInt8(has_right),
            UInt8(left_possible),
            UInt8(right_possible),
            UInt8(left_view.has_validity()),
            UInt8(right_view.has_validity()),
            UInt8(left_view.has_errors()),
            UInt8(right_view.has_errors()),
            UInt8(prefer_right),
            UInt8(Bool(output_validity_buffer)),
            UInt8(Bool(output_errors_buffer)),
            UInt8(nullable),
            output_values.create_sub_buffer[transport_dtype](
                0, len(output_values)
            ),
            output_validity,
            output_errors,
            grid_dim=((rows + 7) // 8 + 127) // 128,
            block_dim=128,
        )
        scope.record_launches()
    except error:
        # Keep the output and all source/map/status handles alive until any
        # command submitted by this launch attempt has drained. The caller
        # then knows this failure path already performed the one cleanup wait.
        scope.synchronize()
        queued = False
        _ = output^
        raise error^
    return output^


def _native_join_typed_source[
    dtype: DType
](column: NativeColumnAny,) raises -> NativeColumn[dtype]:
    if column._storage.isa[NativeColumn[dtype]]():
        return column._storage[NativeColumn[dtype]].copy()
    raise Error("Mojo GPU native join physical dtype differs")


def _native_null_join_typed[
    dtype: DType
](
    scope: Scope, logical: DataType, rows: Int, mut queued: Bool
) raises -> NativeColumn[dtype]:
    var output = allocate_native_column[dtype](
        scope, logical, rows, with_validity=True
    )
    var values = output.values()
    var validity = output.validity().value()
    var ctx = scope.context()
    try:
        comptime transport_dtype = DType.uint64 if dtype == DType.float64 else dtype
        ctx.enqueue_memset(
            values.create_sub_buffer[transport_dtype](0, len(values)),
            Scalar[transport_dtype](0),
        )
        ctx.enqueue_memset(validity, UInt8(0))
    except error:
        # A partially submitted clear needs its owner alive through the drain.
        # Tell the enclosing gather that this path already performed that wait.
        scope.synchronize()
        queued = False
        _ = output^
        raise error^
    return output^


def _native_null_join_column(
    scope: Scope, logical: DataType, rows: Int, mut queued: Bool
) raises -> NativeColumnAny:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if logical.physical() == DataType.of(D):
            return NativeColumnAny(
                _native_null_join_typed[D](scope, logical, rows, queued)^
            )
    raise Error("Mojo GPU native join payload dtype is unsupported")


def _native_join_gather_column(
    scope: Scope,
    left: Optional[NativeColumnAny],
    right: Optional[NativeColumnAny],
    prefer_right: Bool,
    left_possible: Bool,
    right_possible: Bool,
    nullable: Bool,
    left_offset: Int,
    left_rows: Int,
    right_offset: Int,
    right_rows: Int,
    left_indices: DeviceBuffer[DType.uint32],
    right_indices: DeviceBuffer[DType.uint32],
    rows: Int,
    status: DeviceBuffer[DType.uint32],
    logical: DataType,
    mut queued: Bool,
) raises -> NativeColumnAny:
    var has_left = Bool(left)
    var has_right = Bool(right)
    if has_left and left_rows == 0 and not has_right:
        return _native_null_join_column(scope, logical, rows, queued)
    if has_right and right_rows == 0 and not has_left:
        return _native_null_join_column(scope, logical, rows, queued)
    var left_view_offset = left_offset if has_left else right_offset
    var left_view_rows = left_rows if has_left else right_rows
    var right_view_offset = right_offset if has_right else left_offset
    var right_view_rows = right_rows if has_right else left_rows
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if logical.physical() == DataType.of(D):
            var fallback = _native_join_typed_source[D](
                left.value().copy() if left else right.value().copy()
            )
            var l = _native_join_typed_source[D](
                left.value()
            ) if left else fallback.copy()
            var r = _native_join_typed_source[D](
                right.value()
            ) if right else fallback.copy()
            return NativeColumnAny(
                _native_join_gather_typed[D](
                    scope,
                    l,
                    r,
                    has_left,
                    has_right,
                    left_possible,
                    right_possible,
                    prefer_right,
                    nullable,
                    left_view_offset,
                    left_view_rows,
                    right_view_offset,
                    right_view_rows,
                    left_indices,
                    right_indices,
                    rows,
                    status,
                    queued,
                )
                ^
            )
    raise Error("Mojo GPU native join payload dtype is unsupported")


def _gather_native_typed[
    dtype: DType
](
    scope: Scope,
    source: NativeColumn[dtype],
    offset: Int,
    source_rows: Int,
    indices: DeviceBuffer[DType.int32],
    rows: Int,
    allow_missing: Bool,
) raises -> NativeColumn[dtype]:
    var input = source.view(offset, source_rows)
    var output_has_validity = input.has_validity() or allow_missing
    var output = allocate_native_column[dtype](
        scope,
        input.logical,
        rows,
        with_validity=output_has_validity,
        with_errors=input.has_errors(),
    )
    if not rows:
        return output^
    var source_values = input.values()
    var source_validity = indices.create_sub_buffer[DType.uint8](0, 1)
    var source_validity_buffer = Optional[DeviceBuffer[DType.uint8]]()
    if input.has_validity():
        source_validity_buffer = input.validity()
        source_validity = source_validity_buffer.value().copy()
    var source_errors = indices.create_sub_buffer[DType.uint32](0, 1)
    var source_errors_buffer = Optional[DeviceBuffer[DType.uint32]]()
    if input.has_errors():
        source_errors_buffer = input.errors()
        source_errors = source_errors_buffer.value().copy()
    var output_values = output.values()
    var output_validity = indices.create_sub_buffer[DType.uint8](0, 1)
    var output_validity_buffer = output.validity()
    if output_validity_buffer:
        output_validity = output_validity_buffer.value().copy()
    var output_errors = indices.create_sub_buffer[DType.uint32](0, 1)
    var output_errors_buffer = output.errors()
    if output_errors_buffer:
        output_errors = output_errors_buffer.value().copy()
    var ctx = scope.context()
    try:
        # Float64 payloads move as bits; Metal cannot load native doubles.
        comptime transport_dtype = DType.uint64 if dtype == DType.float64 else dtype
        ctx.enqueue_function[gather_native_column[transport_dtype]](
            source_values.create_sub_buffer[transport_dtype](
                0, len(source_values)
            ),
            source_validity,
            source_errors,
            indices,
            UInt32(input.value_offset),
            UInt32(input.validity_bit_offset),
            UInt32(input.value_offset),
            UInt8(input.has_validity()),
            UInt8(input.has_errors()),
            UInt8(Bool(output_validity_buffer)),
            UInt32(rows),
            output_values.create_sub_buffer[transport_dtype](
                0, len(output_values)
            ),
            output_validity,
            output_errors,
            grid_dim=((rows + 7) // 8 + 127) // 128,
            block_dim=128,
        )
        scope.record_launches()
    except error:
        # The destination and all sub-buffer handles are local to this helper.
        # Drain while they and the source/index owners still exist; only then
        # unwind to Table.gather's aggregate cleanup path.
        scope.synchronize()
        _ = output^
        _ = input^
        raise error^
    return output^


def _gather_native_column(
    scope: Scope,
    column: NativeColumnAny,
    offset: Int,
    source_rows: Int,
    indices: DeviceBuffer[DType.int32],
    rows: Int,
    allow_missing: Bool,
) raises -> NativeColumnAny:
    if column._storage.isa[NativeColumn[DType.int64]]():
        return NativeColumnAny(
            _gather_native_typed[DType.int64](
                scope,
                column._storage[NativeColumn[DType.int64]],
                offset,
                source_rows,
                indices,
                rows,
                allow_missing,
            )
            ^
        )
    if column._storage.isa[NativeColumn[DType.float64]]():
        return NativeColumnAny(
            _gather_native_typed[DType.float64](
                scope,
                column._storage[NativeColumn[DType.float64]],
                offset,
                source_rows,
                indices,
                rows,
                allow_missing,
            )
            ^
        )
    if column._storage.isa[NativeColumn[DType.int8]]():
        return NativeColumnAny(
            _gather_native_typed[DType.int8](
                scope,
                column._storage[NativeColumn[DType.int8]],
                offset,
                source_rows,
                indices,
                rows,
                allow_missing,
            )
            ^
        )
    if column._storage.isa[NativeColumn[DType.int16]]():
        return NativeColumnAny(
            _gather_native_typed[DType.int16](
                scope,
                column._storage[NativeColumn[DType.int16]],
                offset,
                source_rows,
                indices,
                rows,
                allow_missing,
            )
            ^
        )
    if column._storage.isa[NativeColumn[DType.int32]]():
        return NativeColumnAny(
            _gather_native_typed[DType.int32](
                scope,
                column._storage[NativeColumn[DType.int32]],
                offset,
                source_rows,
                indices,
                rows,
                allow_missing,
            )
            ^
        )
    if column._storage.isa[NativeColumn[DType.uint8]]():
        return NativeColumnAny(
            _gather_native_typed[DType.uint8](
                scope,
                column._storage[NativeColumn[DType.uint8]],
                offset,
                source_rows,
                indices,
                rows,
                allow_missing,
            )
            ^
        )
    if column._storage.isa[NativeColumn[DType.uint16]]():
        return NativeColumnAny(
            _gather_native_typed[DType.uint16](
                scope,
                column._storage[NativeColumn[DType.uint16]],
                offset,
                source_rows,
                indices,
                rows,
                allow_missing,
            )
            ^
        )
    if column._storage.isa[NativeColumn[DType.uint32]]():
        return NativeColumnAny(
            _gather_native_typed[DType.uint32](
                scope,
                column._storage[NativeColumn[DType.uint32]],
                offset,
                source_rows,
                indices,
                rows,
                allow_missing,
            )
            ^
        )
    if column._storage.isa[NativeColumn[DType.uint64]]():
        return NativeColumnAny(
            _gather_native_typed[DType.uint64](
                scope,
                column._storage[NativeColumn[DType.uint64]],
                offset,
                source_rows,
                indices,
                rows,
                allow_missing,
            )
            ^
        )
    return NativeColumnAny(
        _gather_native_typed[DType.float32](
            scope,
            column._storage[NativeColumn[DType.float32]],
            offset,
            source_rows,
            indices,
            rows,
            allow_missing,
        )
        ^
    )


def _copy_native_to_matrix[
    dtype: DType
](
    scope: Scope,
    column: NativeColumn[dtype],
    offset: Int,
    rows: Int,
    output_column: Int,
    capacity: Int,
    slots: Int,
    target_cells: Int,
    target_words: DeviceBuffer[DType.uint64],
    target_validity: DeviceBuffer[DType.uint8],
    target_errors: DeviceBuffer[DType.uint32],
) raises:
    var native = column.view(offset, rows)
    if not native.scope().same_query(scope):
        raise Error("Mojo GPU native input belongs to another query context")
    var values = native.values()
    var source_values = values.create_sub_buffer[dtype](
        native.value_offset, rows
    )
    var source_validity = values.create_sub_buffer[DType.uint8](0, 1)
    var validity = Optional[DeviceBuffer[DType.uint8]]()
    if native.has_validity():
        validity = native.validity()
        source_validity = validity.value().copy()
    var source_errors = target_errors
    var errors = Optional[DeviceBuffer[DType.uint32]]()
    if native.has_errors():
        errors = native.errors()
        source_errors = errors.value().copy()
    var target = checked_mul(output_column, capacity)
    var high_target = checked_add(target_cells, target)
    var low = target_words.create_sub_buffer[DType.uint64](target, rows)
    var high = target_words.create_sub_buffer[DType.uint64](high_target, rows)
    var valid_out = target_validity.create_sub_buffer[DType.uint8](target, rows)
    var error_out = target_errors.create_sub_buffer[DType.uint32](target, rows)
    var ctx = scope.context()
    try:
        if native.has_validity():
            if native.has_errors():
                ctx.enqueue_function[copy_native_column[dtype, True, True]](
                    source_values,
                    source_validity,
                    source_errors,
                    UInt32(native.value_offset),
                    UInt32(native.validity_bit_offset),
                    UInt32(rows),
                    low,
                    high,
                    valid_out,
                    error_out,
                    grid_dim=(rows + 255) // 256,
                    block_dim=256,
                )
            else:
                ctx.enqueue_function[copy_native_column[dtype, True, False]](
                    source_values,
                    source_validity,
                    source_errors,
                    UInt32(native.value_offset),
                    UInt32(native.validity_bit_offset),
                    UInt32(rows),
                    low,
                    high,
                    valid_out,
                    error_out,
                    grid_dim=(rows + 255) // 256,
                    block_dim=256,
                )
        elif native.has_errors():
            ctx.enqueue_function[copy_native_column[dtype, False, True]](
                source_values,
                source_validity,
                source_errors,
                UInt32(native.value_offset),
                UInt32(0),
                UInt32(rows),
                low,
                high,
                valid_out,
                error_out,
                grid_dim=(rows + 255) // 256,
                block_dim=256,
            )
        else:
            ctx.enqueue_function[copy_native_column[dtype, False, False]](
                source_values,
                source_validity,
                source_errors,
                UInt32(native.value_offset),
                UInt32(0),
                UInt32(rows),
                low,
                high,
                valid_out,
                error_out,
                grid_dim=(rows + 255) // 256,
                block_dim=256,
            )
        scope.record_launches()
    except error:
        # Keep the source owner, destination and all local sub-buffer handles
        # alive until partially submitted work has drained.
        scope.synchronize()
        raise error^
    # The caller drains once after all source columns have been enqueued.
    # `column` and its underlying owner stay alive in the Table until then.


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
