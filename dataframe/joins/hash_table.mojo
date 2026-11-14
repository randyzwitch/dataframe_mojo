"""Partitioned hash join storage and bounded probe output.

Build rows contain validity, keys, payload columns, build row numbers (for
the eager join, which reads its build frame by them), match flags and
hashes. Long strings stay in the build columns' own buffers,
which the table keeps, or else are copied into its heap. Workers append to
private radix partitions, then independent tasks assemble one
pointer-table region per partition, avoiding shared inserts.

Each eight-byte pointer entry packs a 16-bit hash salt and a 48-bit row
address. Linear probing locates an empty slot or a chain with equal keys.
The salt rejects most unrelated entries before their key bytes are read.
Each region holds at least twice as many entries as its rows, keeping
probes short. Duplicate keys form chains in ascending build-row order;
semi, anti and mark joins need only one row per key.

Probe chunks contain at most 2,048 rows. Large tables prefetch entries in
blocks of 256 probe rows so memory requests can overlap. A probe compares
keys once per chain and emits at most one chunk per call to next. Null
keys never match. Right/full joins record matched build rows for a later
scan; build-side semi/anti joins emit flagged/unflagged rows respectively.
"""
from std.atomic import Atomic
from std.bit import next_power_of_two
from std.collections import Optional
from std.memory import (
    ArcPointer,
    Pointer,
    bitcast,
    unsafe_memcpy,
    unsafe_memcmp,
    unsafe_memset_zero,
)
from std.sys.intrinsics import prefetch

from dataframe.dtype import NUMERIC_DTYPES, DataType
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_MARK,
    JOIN_RIGHT,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
    join_emits_build_rows,
)
from dataframe.kernels.partition import _hash_column
from dataframe.kernels.runtime_filter import (
    KeyBloom,
    bloom_blocks_for,
    bloom_set_bits,
    can_use_bloom_filter,
)
from dataframe.parallel import (
    Job,
    configured_workers,
    partitions,
    run_jobs,
    worker_count,
)
from dataframe.series import Series
from dataframe.storage.allocation import huge_uninit
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.buffer import _Buffer
from dataframe.storage.column import Column, _validity_at
from dataframe.storage.string import StringColumn
from dataframe.storage.string_view import StringView, _mapped
from dataframe.trace import trace_path
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    decode_values,
    flat_over,
    to_series,
)
from dataframe.vector.sink import Sink
from dataframe.vector.partitioned_tuple_data import (
    PartitionedTupleDataAppendState,
    RadixPartitionedTupleData,
    number_of_partitions,
    radix_partition,
)
from dataframe.vector.tuple_data_collection import TupleDataCollection
from dataframe.vector.tuple_data_layout import (
    TupleDataLayout,
    physical_of_dtype,
)
from dataframe.vector.tuple_data_scatter_gather import (
    TupleDataHeaps,
    NumberedGather,
    gather_rows_series,
)
from dataframe.vector.tuple_data_states import TupleDataChunkState
from dataframe.vector.types import (
    PHYSICAL_BOOL,
    PHYSICAL_INT16,
    PHYSICAL_INT32,
    PHYSICAL_INT64,
    PHYSICAL_INT8,
    PHYSICAL_UINT16,
    PHYSICAL_UINT32,
    PHYSICAL_UINT64,
    PHYSICAL_UINT8,
    PHYSICAL_VARCHAR,
    STANDARD_VECTOR_SIZE,
    SelectionVector,
    physical_of,
    physical_size,
)

# A table entry is one 64-bit word: the upper 16 bits are the salt (bits of the
# key's hash), the lower 48 the row's address. User-space addresses fit in 48
# bits, so the salt rides along for free and most non-matching entries are
# rejected without reading the row.
comptime SALT_MASK = UInt64(0xFFFF000000000000)
comptime POINTER_MASK = UInt64(0x0000FFFFFFFFFFFF)
# Salts are compared on probes only once the table no longer fits the CPU cache:
# in cache, reading the row to compare the key is cheap, and the salt test is
# extra work. 8,192 entries are 64 KB, about a core's L2 share; the value is an
# empirical choice.
comptime USE_SALT_THRESHOLD = 8192
# The table has twice as many slots as rows, so at most half are full and
# linear-probing chains stay short. Small builds still get 16,384 slots (128
# KB): the memory is trivial and the chains are shorter. Both are empirical
# choices.
comptime DEFAULT_LOAD_FACTOR = 2.0
comptime MINIMUM_CAPACITY = 16384
# Each sink thread splits its rows into 16 partitions by hash: enough to
# finalize partitions in parallel, few enough that every partition's write
# position stays in cache. An empirical choice.
comptime INITIAL_RADIX_BITS = 4
# Hash and prefetch BLOCK_SIZE probe keys together once the table exceeds
# MIN_PREFETCH_BUCKETS; smaller tables are expected to remain in cache.
comptime BLOCK_SIZE = 256
comptime MIN_PREFETCH_BUCKETS = 32768
# The widest key range a perfect hash join addresses directly: one slot per key
# value, so a wider range costs memory for keys that may not exist. Bounds the
# slot array at about 8 MB; an empirical choice.
comptime PERFECT_MAX_BUILD_SIZE = 1048576
# Dictionary emission gathers every build row's output once, then emits probe
# results as row numbers into it. That one gather pays off only when the probe
# side is much larger than the build side (100 times, and at least 262,144 rows)
# and the gathered columns are small enough (8 MB) to stay in cache while probe
# results index them. Empirical choices.
comptime _DICT_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
comptime _DICT_MIN_PROBE_ROWS = 262144
comptime _DICT_PROBE_BUILD_RATIO = 100

# How a key field compares: by its bytes, as a float (every NaN equal,
# -0.0 equal to 0.0), or as a 16-byte string record.
comptime _FIX = 0
comptime _F32 = 1
comptime _F64 = 2
comptime _TEXT = 3
# A 16-byte string record's inline capacity: 16 bytes less the 4-byte length.
comptime _INLINE = 12


@always_inline
def _load64(address: Int) -> UInt64:
    return Pointer[UInt64, MutAnyOrigin](
        unsafe_from_address=address
    ).unsafe_load[alignment=1]()


@always_inline
def _store64(address: Int, value: UInt64):
    Pointer[UInt64, MutAnyOrigin](unsafe_from_address=address).unsafe_store[
        alignment=1
    ](value)


@always_inline
def _load8(address: Int) -> UInt8:
    return Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)[]


@always_inline
def _store8(address: Int, value: UInt8):
    Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)[] = value


@always_inline
def _load32(address: Int) -> UInt32:
    return Pointer[UInt32, MutAnyOrigin](
        unsafe_from_address=address
    ).unsafe_load[alignment=1]()


@always_inline
def _load_as[D: DType](address: Int) -> Scalar[D]:
    return Pointer[Scalar[D], MutAnyOrigin](
        unsafe_from_address=address
    ).unsafe_load[alignment=1]()


@always_inline
def _store_as[D: DType](address: Int, value: Scalar[D]):
    Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=address).unsafe_store[
        alignment=1
    ](value)


struct _StringBytes:
    """Where a string column's rows start and how long they are, from
    pointers taken once: offsets and a byte buffer, or Arrow views (inline
    bytes, or a buffer and an offset)."""

    var views: Int
    var offsets: Int
    var base: Int
    var buffers: Int
    # Views that select their descriptors (`StringViewStorage._row_map`).
    var rows: Int
    var shift: Int

    def __init__(out self, column: StringColumn):
        self.rows = 0
        self.shift = 0
        if column._is_view_storage():
            ref storage = column._view_storage.value()
            var map = storage._row_map(column._offset)
            self.views = map[0]
            self.rows = map[1]
            self.shift = map[2]
            self.buffers = Int(storage._buffers[].unsafe_ptr())
            self.offsets = 0
            self.base = 0
        else:
            self.views = 0
            self.buffers = 0
            self.offsets = Int(
                column._offsets[].unsafe_ptr().unsafe_offset(column._offset)
            )
            self.base = Int(column._base())

    @always_inline
    def at(self, i: Int) -> Tuple[Int, Int]:
        """Row i's first byte's address and its length."""
        if self.views == 0:
            var offsets = Pointer[Int64, MutAnyOrigin](
                unsafe_from_address=self.offsets
            )
            var start = Int(offsets[unsafe_offset=i])
            return (
                self.base + start,
                Int(offsets[unsafe_offset=i + 1]) - start,
            )
        var view = Pointer[StringView, MutAnyOrigin](
            unsafe_from_address=self.views
        ).unsafe_offset(_mapped(self.rows, self.shift, i))
        var length = Int(view[].length)
        if length <= 12:
            return (Int(view) + 4, length)
        var buffers = Pointer[_Buffer[UInt8], MutAnyOrigin](
            unsafe_from_address=self.buffers
        )
        return (
            Int(buffers[unsafe_offset=Int(view[].buffer_index)][].unsafe_ptr())
            + Int(view[].offset),
            length,
        )


def _row_type(key: Series) raises -> DataType:
    """The row column type of a key: its storage (a decimal keeps its own
    type, whose storage is 128 bits)."""
    if key._data.isa[StringColumn]():
        return DataType.STRING
    if key._data.isa[BoolColumn]():
        return DataType.BOOL
    if key._data.isa[Column[Int128]]():
        return key.dtype()
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if key._data.isa[Column[Scalar[D]]]():
            return DataType(D)
    raise Error("unsupported join key type: " + key.dtype().name())


def _field_kind(key: Series) -> Int:
    """How a key field compares."""
    if key._data.isa[StringColumn]():
        return _TEXT
    if key._data.isa[Column[Float64]]():
        return _F64
    if key._data.isa[Column[Float32]]():
        return _F32
    return _FIX


def _write_native(
    key: Series,
    start: Int,
    count: Int,
    base: Int,
    stride: Int,
    valid: Int,
    valid_stride: Int,
) raises:
    """Rows [start, start + count) of an unchunked key as the row field
    TupleDataCollection's scatter writes: row i's field at `base + i *
    stride`, its validity ANDed into the byte at `valid + i *
    valid_stride`. A null row's field is left undefined."""
    var bits = _validity(key)
    if bits[0] != 0:
        var bitmap = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bits[0])
        for i in range(count):
            if not _validity_at(bitmap, bits[1] + start + i):
                _store8(valid + i * valid_stride, 0)
    if key._data.isa[StringColumn]():
        var strings = _StringBytes(key._data[StringColumn])
        for i in range(count):
            var at = base + i * stride
            var found = strings.at(start + i)
            var length = found[1]
            _store64(at, UInt64(length))
            _store64(at + 8, 0)
            if length <= _INLINE:
                if length > 0:
                    unsafe_memcpy(
                        dest=Pointer[UInt8, MutAnyOrigin](
                            unsafe_from_address=at + 4
                        ),
                        src=Pointer[UInt8, MutAnyOrigin](
                            unsafe_from_address=found[0]
                        ),
                        count=length,
                    )
            else:
                unsafe_memcpy(
                    dest=Pointer[UInt8, MutAnyOrigin](
                        unsafe_from_address=at + 4
                    ),
                    src=Pointer[UInt8, MutAnyOrigin](
                        unsafe_from_address=found[0]
                    ),
                    count=4,
                )
                _store64(at + 8, UInt64(found[0]))
        return
    if key._data.isa[Column[Int128]]():
        var values = key._data[Column[Int128]]._ptr()
        for i in range(count):
            _store_as[DType.int128](
                base + i * stride, values[unsafe_offset=start + i]
            )
        return
    if key._data.isa[BoolColumn]():
        ref bools = key._data[BoolColumn]
        for i in range(count):
            _store8(
                base + i * stride,
                UInt8(1) if bools._get(start + i) else UInt8(0),
            )
        return
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if key._data.isa[Column[Scalar[D]]]():
            var values = key._data[Column[Scalar[D]]]._ptr()
            for i in range(count):
                _store_as[D](base + i * stride, values[unsafe_offset=start + i])
            return
    raise Error("unsupported join key type: " + key.dtype().name())


@fieldwise_init
struct KeyLayout(Copyable, Movable):
    """Where a join row's columns are, and how its key fields compare (one
    comparison kind per column)."""

    var kinds: List[Int]
    # Each key field's offset from the first key field, and its width.
    var offsets: List[Int]
    var widths: List[Int]
    var key_bytes: Int
    # Every field compares by its bytes (no float, no string).
    var plain: Bool
    var key_offset: Int
    var index_offset: Int
    var found_offset: Int
    var pointer_offset: Int
    var width: Int

    @staticmethod
    def of(
        layout: TupleDataLayout,
        keys: List[Series],
        index_column: Int,
        found_column: Int,
    ) -> KeyLayout:
        var at = layout.offsets()
        var fields = len(keys)
        var kinds = List[Int](capacity=fields)
        var offsets = List[Int](capacity=fields)
        var widths = List[Int](capacity=fields)
        var plain = True
        for k in range(fields):
            var kind = _field_kind(keys[k])
            kinds.append(kind)
            offsets.append(at[k] - at[0])
            widths.append(at[k + 1] - at[k])
            plain = plain and kind == _FIX
        return KeyLayout(
            kinds^,
            offsets^,
            widths^,
            at[fields] - at[0],
            plain,
            at[0],
            at[index_column] if index_column >= 0 else -1,
            at[found_column] if found_column >= 0 else -1,
            at[len(layout.types()) - 1],
            layout.row_width(),
        )


@always_inline
def _same_bytes(a: Int, b: Int, count: Int) -> Bool:
    return (
        count == 0
        or unsafe_memcmp(
            Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=a),
            Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=b),
            count,
        )
        == 0
    )


@always_inline
def _same_text(a: Int, b: Int) -> Bool:
    """Whether two string_t fields hold the same string (bytes past an
    inlined string's length are not compared)."""
    var length = _load32(a)
    if length != _load32(b):
        return False
    var n = Int(length)
    if n == 0:
        return True
    if n <= 4:
        var mask = UInt32((UInt64(1) << UInt64(8 * n)) - 1)
        return (_load32(a + 4) ^ _load32(b + 4)) & mask == 0
    if _load32(a + 4) != _load32(b + 4):
        return False
    if n <= _INLINE:
        var rest = n - 4
        var mask = UInt64.MAX if rest == 8 else (
            (UInt64(1) << UInt64(8 * rest)) - 1
        )
        return (_load64(a + 8) ^ _load64(b + 8)) & mask == 0
    return (
        unsafe_memcmp(
            Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=Int(_load64(a + 8))
            ),
            Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=Int(_load64(b + 8))
            ),
            n,
        )
        == 0
    )


@always_inline
def _same_keys(
    kinds: Int,
    offsets: Int,
    widths: Int,
    fields: Int,
    key_bytes: Int,
    plain: Bool,
    a: Int,
    b: Int,
) -> Bool:
    """Whether the key fields at `a` and `b` are equal, over every key column.
    `kinds`, `offsets` and `widths` address the layout's lists."""
    if plain:
        return _same_bytes(a, b, key_bytes)
    var kind = Pointer[Int, MutAnyOrigin](unsafe_from_address=kinds)
    var offset = Pointer[Int, MutAnyOrigin](unsafe_from_address=offsets)
    var width = Pointer[Int, MutAnyOrigin](unsafe_from_address=widths)
    for f in range(fields):
        var at = offset[unsafe_offset=f]
        var k = kind[unsafe_offset=f]
        if k == _TEXT:
            if not _same_text(a + at, b + at):
                return False
        elif k == _F64:
            var x = _load_as[DType.float64](a + at)
            var y = _load_as[DType.float64](b + at)
            if not (x == y or (x != x and y != y)):
                return False
        elif k == _F32:
            var x = _load_as[DType.float32](a + at)
            var y = _load_as[DType.float32](b + at)
            if not (x == y or (x != x and y != y)):
                return False
        elif not _same_bytes(a + at, b + at, width[unsafe_offset=f]):
            return False
    return True


struct _TableData(Movable):
    """What a table owns: its rows, its pointer table and the build keys."""

    var collection: TupleDataCollection
    var entries: List[UInt64]
    var keys: List[Series]
    # A perfect hash join's table: per key slot its build row (the payload
    # stand-in, unset where the slot's bit is clear) and one bit per slot, set
    # when a build row has the slot's key.
    var slots: List[UInt32]
    var slot_bits: List[UInt8]
    # Each partition's region of `entries` (one region unless the table
    # is built per partition): its first entry and its bitmask.
    var bases: List[UInt64]
    var masks: List[UInt64]
    # Retain payload columns indexed by build-row order or perfect-table slot.
    # Keep next pointers separate when dictionary emission replaces row links
    # with row numbers.
    var dictionary: List[Series]
    # Per dictionary column, its values decoded to their vector layout once
    # (`decode_values`; empty for a fixed-width column), which every
    # probing worker reads.
    var decoded: List[List[UInt8]]
    var slot_rows: List[UInt64]
    var aux_next: List[UInt64]
    # All zeros: a row with every column null (its validity bits clear).
    var null_row: List[UInt8]
    # The rows' heap blocks, which gathered string columns reference.
    var heaps: TupleDataHeaps
    # The finalize tasks not yet run, one per build sink thread.
    var pending: List[_FinalizeJob]
    # While a perfect hash join's finalize tasks run, two rounds of them.
    # The first: per task, the run of chunks it reads and the
    # collection's row its run starts at; each task gathers its rows into
    # every output column (`gathers`) and sorts their slots into
    # `buckets` (per task, per range of slots: slot and number in one
    # word, then the row's pointer). The second: per range of
    # `1 << slot_shift` slots, a task fills its slots from every task's
    # bucket for it. `repeated`: a word a task sets on finding a repeated
    # key. `phase`: the round under way (0 when none).
    var gathers: List[NumberedGather]
    var repeated: List[UInt64]
    var slot_runs: List[List[Tuple[Int, Int]]]
    var slot_firsts: List[Int]
    var buckets: List[List[UInt64]]
    var slot_shift: Int
    var slot_ranges: Int
    var slot_physical: Int
    var phase: Int
    # What a pointer table needs if a repeated key turns a perfect hash
    # join back into a hash join: each partition's rows, its first
    # segment (one more entry for the end), the build's sink threads,
    # whether the chains keep duplicate keys and whether the inserts skip
    # null keys.
    var partition_rows: List[Int]
    var segment_starts: List[Int]
    var sink_threads: Int
    var duplicates: Bool
    var skip_null_keys: Bool

    def __init__(
        out self, var keys: List[Series], layout: TupleDataLayout
    ) raises:
        self.collection = TupleDataCollection(layout)
        self.entries = List[UInt64]()
        self.keys = keys^
        self.slots = List[UInt32]()
        self.slot_bits = List[UInt8]()
        self.bases = List[UInt64]()
        self.masks = List[UInt64]()
        self.dictionary = List[Series]()
        self.decoded = List[List[UInt8]]()
        self.slot_rows = List[UInt64]()
        self.aux_next = List[UInt64]()
        self.null_row = List[UInt8](length=layout.row_width(), fill=0)
        self.heaps = TupleDataHeaps()
        self.pending = List[_FinalizeJob]()
        self.gathers = List[NumberedGather]()
        self.repeated = List[UInt64](length=1, fill=0)
        self.slot_runs = List[List[Tuple[Int, Int]]]()
        self.slot_firsts = List[Int]()
        self.buckets = List[List[UInt64]]()
        self.slot_shift = 0
        self.slot_ranges = 0
        self.slot_physical = 0
        self.phase = 0
        self.partition_rows = List[Int]()
        self.segment_starts = List[Int]()
        self.sink_threads = 1
        self.duplicates = False
        self.skip_null_keys = False


def _integer_physical(physical: Int) -> Bool:
    """Whether row values of `physical` are integers, whose least and
    greatest key a build records."""
    return (
        physical == PHYSICAL_UINT8
        or physical == PHYSICAL_INT8
        or physical == PHYSICAL_UINT16
        or physical == PHYSICAL_INT16
        or physical == PHYSICAL_UINT32
        or physical == PHYSICAL_INT32
        or physical == PHYSICAL_UINT64
        or physical == PHYSICAL_INT64
    )


struct JoinBuildLayout(Copyable, Movable):
    """What every thread sinking rows into one join's build shares: the
    row layout [keys, payload, build row number, found flag, hash] (the
    number only when the table is numbered, the flag only when the join
    reads it), which column of the sink's input chunks each key and
    payload field comes from, and the build side's columns the join
    outputs from the rows."""

    var row_layout: TupleDataLayout
    var key_layout: KeyLayout
    var key_columns: List[Int]
    var payload_columns: List[Int]
    var key_types: List[DataType]
    # Whether each key's least and greatest values are recorded (integer
    # keys).
    var bounded: List[Bool]
    var numbered: Bool
    var found: Bool
    var nested: Bool
    var output_columns: List[Int]
    var output_names: List[String]
    var output_types: List[DataType]
    var perfect: Bool
    var keys_only: Bool
    var probe_estimate: Int
    # Whether the build side filters its rows (the bloom filter rule).
    var build_filtered: Bool
    # The build side's columns in order: names, types, and the row field
    # holding each (-1 for a column the rows do not keep).
    var build_names: List[String]
    var build_types: List[DataType]
    var build_fields: List[Int]

    def __init__(
        out self,
        keys: List[Series],
        payload: List[Series],
        key_columns: List[Int],
        payload_columns: List[Int],
        numbered: Bool,
        found: Bool,
        output_columns: List[Int],
        output_names: List[String],
        output_types: List[DataType],
        perfect: Bool,
        keys_only: Bool,
        probe_estimate: Int,
        build_names: List[String],
        build_types: List[DataType],
        build_fields: List[Int],
        build_filtered: Bool = False,
    ) raises:
        """`keys` and `payload` are columns (of any length) of the key and
        payload types, which the row fields are typed from."""
        var types = List[DataType](capacity=len(keys) + len(payload) + 3)
        self.key_types = List[DataType](capacity=len(keys))
        for key in keys:
            if key.dtype().is_nested():
                raise Error(
                    "list and struct columns cannot be join keys: " + key.name()
                )
            types.append(_row_type(key))
            self.key_types.append(key.dtype())
        self.nested = False
        for column in payload:
            if column.dtype().is_nested():
                types.append(column.dtype())
                self.nested = True
            else:
                types.append(_row_type(column))
        var index_column = -1
        if numbered:
            index_column = len(types)
            types.append(DataType.UINT32)
        var found_column = -1
        if found:
            found_column = len(types)
            types.append(DataType.BOOL)
        types.append(DataType.UINT64)
        self.row_layout = TupleDataLayout(types)
        self.key_layout = KeyLayout.of(
            self.row_layout, keys, index_column, found_column
        )
        self.bounded = List[Bool](capacity=len(keys))
        for k in range(len(keys)):
            self.bounded.append(
                _integer_physical(self.row_layout.nodes[][0].physicals[k])
            )
        self.key_columns = key_columns.copy()
        self.payload_columns = payload_columns.copy()
        self.numbered = numbered
        self.found = found
        self.output_columns = output_columns.copy()
        self.output_names = output_names.copy()
        self.output_types = output_types.copy()
        self.perfect = perfect
        self.keys_only = keys_only
        self.probe_estimate = probe_estimate
        self.build_filtered = build_filtered
        self.build_names = build_names.copy()
        self.build_types = build_types.copy()
        self.build_fields = build_fields.copy()

    def width(self) -> Int:
        """The key and payload fields."""
        return len(self.key_columns) + len(self.payload_columns)

    def hash_column(self) -> Int:
        """The row chunk's last column: each row's hash."""
        return (
            self.width()
            + (1 if self.numbered else 0)
            + (1 if self.found else 0)
        )


struct JoinBuildSink(Sink):
    """One thread's build state: the input chunks' rows appended to its
    own radix-partitioned rows (keys, payload and hash; a row with a null
    key is left out unless the join reads found flags, since it never
    matches), with the count of rows seen, those with a null key, and
    each integer key's least and greatest value. Threads never share
    one, so sinking takes no lock; the table combines them."""

    var layout: ArcPointer[JoinBuildLayout]
    var rows: RadixPartitionedTupleData
    var state: PartitionedTupleDataAppendState
    # The row chunk: the input's key and payload vectors plus the columns
    # the sink writes.
    var chunk: DataChunk
    var kept: List[UInt32]
    var count: Int
    var invalid: Int
    # A numbered table's next build row number.
    var next_row: Int
    # The buffers this sink's rows point into for strings left where they
    # are, each held once (by the address of its bytes).
    var strings: List[_Buffer[UInt8]]
    var held: List[Int]
    var least: List[Int64]
    var most: List[Int64]
    # Per key: whether a valid value was seen, and whether every value
    # fits the 64-bit signed bounds.
    var seen: List[Bool]
    var exact: List[Bool]

    def __init__(
        out self, layout: ArcPointer[JoinBuildLayout], first_row: Int = 0
    ) raises:
        self.layout = layout
        self.rows = RadixPartitionedTupleData(
            layout[].row_layout, INITIAL_RADIX_BITS, layout[].hash_column()
        )
        self.state = PartitionedTupleDataAppendState()
        self.rows.initialize_append_state(self.state)
        self.chunk = DataChunk(layout[].hash_column() + 1)
        self.kept = List[UInt32](unsafe_uninit_length=STANDARD_VECTOR_SIZE)
        self.count = 0
        self.invalid = 0
        self.next_row = first_row
        self.strings = List[_Buffer[UInt8]]()
        self.held = List[Int]()
        var keys = len(layout[].key_columns)
        self.least = List[Int64](length=keys, fill=Int64.MAX)
        self.most = List[Int64](length=keys, fill=Int64.MIN)
        self.seen = List[Bool](length=keys, fill=False)
        self.exact = layout[].bounded.copy()

    def begin(mut self, morsel: Int):
        pass

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        var n = chunk.count
        if n == 0:
            return
        var shared = self.layout
        ref layout = shared[]
        var fields = len(layout.key_columns)
        var columns = layout.width()
        self.chunk.reset()
        for k in range(fields):
            self.chunk.data[k] = chunk.data[layout.key_columns[k]]
        for p in range(len(layout.payload_columns)):
            self.chunk.data[fields + p] = chunk.data[layout.payload_columns[p]]
        var at = columns
        if layout.numbered:
            var index = self.chunk.writable(at, PHYSICAL_UINT32)
            var numbers = Pointer[UInt32, MutAnyOrigin](
                unsafe_from_address=index.data
            )
            for i in range(n):
                numbers[unsafe_offset=i] = UInt32(self.next_row + i)
            at += 1
        if layout.found:
            # Every row starts not found.
            var found = self.chunk.writable(at, PHYSICAL_BOOL)
            unsafe_memset_zero(
                Pointer[UInt8, MutAnyOrigin](unsafe_from_address=found.data),
                n,
            )
        var hashes = self.chunk.writable(layout.hash_column(), PHYSICAL_UINT64)
        for k in range(fields):
            var key = to_series(
                chunk, layout.key_columns[k], arena, "", layout.key_types[k]
            )
            _hash_column(key, 0, n, hashes.data, k == 0)
            if self.exact[k]:
                self._bound(k, key)
        self.chunk.count = n
        self.rows.partitions[0].to_unified_format(
            self.state.chunk_state, self.chunk, arena
        )
        for k in range(columns):
            var referenced = False
            if layout.row_layout.nodes[][0].physicals[k] == PHYSICAL_VARCHAR:
                # A vector read from a column whose bytes the table can
                # keep (a window of it, or its rows a filter kept) is not
                # copied into the rows' heap: the rows point at its bytes,
                # and the table holds the buffers they are in.
                var v = self.chunk.data[k]
                if v.source >= 0:
                    var buffers = _string_buffers(arena.columns[v.source])
                    if buffers:
                        referenced = True
                        self._hold(buffers.value())
            self.state.chunk_state.vector_data[k].referenced = referenced
        # The rows whose keys are all valid.
        var sel = self.kept.unsafe_ptr()
        var added = n
        var append_sel = SelectionVector()
        for k in range(fields):
            ref format = self.state.chunk_state.vector_data[k].unified
            if format.validity.all_valid():
                continue
            var kept_count = 0
            for i in range(added):
                var row = append_sel.get_index(i)
                if format.validity.row_is_valid(format.sel.get_index(row)):
                    sel[unsafe_offset=kept_count] = UInt32(row)
                    kept_count += 1
            added = kept_count
            append_sel = SelectionVector(Int(sel))
        self.invalid += n - added
        if layout.found:
            # A join that reads the found flags keeps its rows with a null
            # key (never matched, so its scan emits them); only the insert
            # skips them.
            self.rows.append_unified(
                self.state, self.chunk, SelectionVector(), n
            )
        else:
            self.rows.append_unified(self.state, self.chunk, append_sel, added)
        self.count += n
        self.next_row += n

    def _hold(mut self, buffers: List[_Buffer[UInt8]]):
        for buffer in buffers:
            var address = Int(buffer.unsafe_ptr())
            var known = False
            for h in self.held:
                if h == address:
                    known = True
                    break
            if not known:
                self.held.append(address)
                self.strings.append(buffer.copy())

    def _bound(mut self, k: Int, key: Series):
        """Widen key `k`'s least and greatest value by `key`'s valid
        values."""
        var valid = _validity(key)
        var bitmap = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=valid[0])
        var least = self.least[k]
        var most = self.most[k]
        var seen = self.seen[k]
        comptime for d in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[d]
            comptime if not D.is_floating_point():
                if key._data.isa[Column[Scalar[D]]]():
                    var values = key._data[Column[Scalar[D]]]._ptr()
                    for i in range(len(key)):
                        if valid[0] != 0 and not _validity_at(
                            bitmap, valid[1] + i
                        ):
                            continue
                        var value = values[unsafe_offset=i]
                        comptime if D == DType.uint64:
                            if UInt64(value) > UInt64(Int64.MAX):
                                self.exact[k] = False
                                return
                        least = min(least, Int64(value))
                        most = max(most, Int64(value))
                        seen = True
                    self.least[k] = least
                    self.most[k] = most
                    self.seen[k] = seen
                    return
        self.exact[k] = False

    def release(mut self):
        self.rows.flush_append_state(self.state)

    def combine(var self, mut combined: List[JoinBuildSink]) raises:
        """The thread's rows into the global state, every thread's, which
        the table merges partition by partition."""
        combined.append(self^)


struct _FrameSinkJob(Job):
    """One build thread over a frame: rows [start, start + count) of the
    key and payload columns into its own sink, a vector at a time."""

    var columns: List[Series]
    var start: Int
    var count: Int
    # The thread's sink (one), taken by the table once the job has run.
    var sink: List[JoinBuildSink]

    def __init__(
        out self,
        columns: List[Series],
        layout: ArcPointer[JoinBuildLayout],
        start: Int,
        count: Int,
    ) raises:
        self.columns = List[Series](capacity=len(columns))
        for column in columns:
            self.columns.append(column.slice(start, count))
        self.start = start
        self.count = count
        self.sink = List[JoinBuildSink]()
        self.sink.append(JoinBuildSink(layout, start))

    def run(mut self) raises:
        var arena = ColumnArena()
        var handles = List[Int](capacity=len(self.columns))
        for column in self.columns:
            handles.append(arena.add(column))
        var chunk = DataChunk(len(self.columns))
        var low = 0
        while low < self.count:
            var n = min(STANDARD_VECTOR_SIZE, self.count - low)
            chunk.reset()
            for c in range(len(handles)):
                chunk.data[c] = flat_over(arena, handles[c], low)
            chunk.count = n
            self.sink[0].sink(chunk, arena)
            low += n
        self.sink[0].release()


def join_build_layout(
    columns: List[Series],
    keys: List[String],
    how: Int,
    coalesce: Bool,
    probe_estimate: Int,
    from_chunks: Bool,
    numbered: Bool = False,
    build_filtered: Bool = False,
) raises -> Optional[JoinBuildLayout]:
    """The rows a hash join of type `how` on the build keys `keys` keeps
    of a build side with `columns` (columns of any length, typing the
    rows): the keys, then every other column unless the join is a semi or
    anti join (it emits no build column), the outputs the join reads from
    them (a key only where the join does not coalesce it), and found
    flags when the join emits build rows after its probe. `probe_estimate`
    is the probe side's estimated rows, `build_filtered` whether the build
    side filters its rows. None when the build cannot be a
    table: a nested key, or, for rows sunk from pipeline chunks
    (`from_chunks`), a categorical key (the probe unifies dictionaries
    against the build frame) or a nested column (a chunk's nested vector
    may have no column to read)."""
    if len(keys) == 0:
        return None
    var key_columns = List[Int](capacity=len(keys))
    for name in keys:
        var at = -1
        for c in range(len(columns)):
            if columns[c].name() == name:
                at = c
        if at < 0:
            raise Error("join build: no key column " + name)
        ref dtype = columns[at].dtype()
        if dtype.is_nested() or (from_chunks and dtype.is_categorical()):
            return None
        key_columns.append(at)
    var emits = how != JOIN_SEMI and how != JOIN_ANTI
    var payload_columns = List[Int]()
    var output_columns = List[Int]()
    var output_names = List[String]()
    var output_types = List[DataType]()
    var build_names = List[String]()
    var build_types = List[DataType]()
    var build_fields = List[Int]()
    for i in range(len(columns)):
        ref column = columns[i]
        if from_chunks and column.dtype().is_nested():
            return None
        build_names.append(column.name())
        build_types.append(column.dtype())
        var key_at = -1
        for k in range(len(keys)):
            if keys[k] == column.name():
                key_at = k
        if key_at < 0:
            # A payload column: in the rows, strings and nested values in
            # their heap.
            if emits:
                payload_columns.append(i)
                var field = len(keys) + len(payload_columns) - 1
                build_fields.append(field)
                output_columns.append(field)
                output_names.append(column.name())
                output_types.append(column.dtype())
            else:
                build_fields.append(-1)
            continue
        # A key column: read from its own field where the join outputs it.
        build_fields.append(key_at)
        if not emits:
            continue
        if coalesce and how != JOIN_RIGHT_SEMI and how != JOIN_RIGHT_ANTI:
            continue
        output_columns.append(key_at)
        output_names.append(column.name())
        output_types.append(column.dtype())
    var key_series = List[Series](capacity=len(key_columns))
    for c in key_columns:
        key_series.append(columns[c].copy())
    var payload_series = List[Series](capacity=len(payload_columns))
    for c in payload_columns:
        payload_series.append(columns[c].copy())
    var perfect = how == JOIN_INNER
    var keys_only = keys_only_join(how)
    return JoinBuildLayout(
        key_series,
        payload_series,
        key_columns,
        payload_columns,
        numbered,
        join_emits_build_rows(how) and not perfect and not keys_only,
        output_columns,
        output_names,
        output_types,
        perfect,
        keys_only,
        probe_estimate,
        build_names,
        build_types,
        build_fields,
        build_filtered,
    )


def build_from_columns(
    layout: ArcPointer[JoinBuildLayout],
    columns: List[Series],
    defer_finalize: Bool,
) raises -> JoinHashTable:
    """The table over a build side's whole columns (each one chunk, all of
    one length), which `layout` reads: the rows are fed as vectors of
    STANDARD_VECTOR_SIZE, and every thread that can take at least one
    vector sinks a contiguous run of them into its own partitioned rows.
    A sink's work per row is the same whatever the table's size, so a
    small build spreads over the threads just as a large one does instead
    of leaving them idle. The runs are contiguous so that combining the
    threads' partitions in thread order keeps every partition's rows in
    build order. The table keeps the key columns, which its key filters
    read."""
    var count = len(columns[0]) if len(columns) > 0 else 0
    if count > Int(Int32.MAX):
        raise Error("A join build exceeds 2^31 - 1 rows")
    var vectors = (count + STANDARD_VECTOR_SIZE - 1) // STANDARD_VECTOR_SIZE
    var workers = max(1, min(configured_workers(), vectors))
    var bounds = partitions(count, workers, STANDARD_VECTOR_SIZE)
    var jobs = List[_FrameSinkJob](capacity=workers)
    for w in range(workers):
        if bounds[w] < bounds[w + 1]:
            jobs.append(
                _FrameSinkJob(
                    columns, layout, bounds[w], bounds[w + 1] - bounds[w]
                )
            )
    if len(jobs) == 1:
        jobs[0].run()
    elif len(jobs) > 1:
        run_jobs(jobs)
    var sinks = List[JoinBuildSink](capacity=len(jobs))
    for j in range(len(jobs)):
        sinks.append(jobs[j].sink.pop())
    _ = jobs^
    var keys = List[Series](capacity=len(layout[].key_columns))
    for c in layout[].key_columns:
        keys.append(columns[c].copy())
    return JoinHashTable(layout, sinks^, defer_finalize, keys^)


def _string_buffers(series: Series) -> Optional[List[_Buffer[UInt8]]]:
    """The buffers holding every long string of `series`, when its
    decoded string_t values point into them: an Arrow string column's
    bytes, or a view column's data buffers when its views are its own
    (not read through a row list). None for any other column."""
    if not series._data.isa[StringColumn]():
        return None
    ref column = series._data[StringColumn]
    if column._is_view_storage():
        ref storage = column._view_storage.value()
        if storage._selects():
            return None
        return storage._buffers[].copy()
    var buffers = List[_Buffer[UInt8]]()
    buffers.append(column._bytes.copy())
    return buffers^


struct _FinalizeJob(Job):
    """One finalize thread's partitions, filled one after another."""

    var regions: List[_InsertJob]
    var chained: Bool

    def __init__(out self):
        self.regions = List[_InsertJob]()
        self.chained = False

    def run(mut self) raises:
        var state = TupleDataChunkState()
        for r in range(len(self.regions)):
            self.regions[r].fill(state)
            self.chained = self.chained or self.regions[r].chained


struct _TableTask(Job):
    """Finalize task `task` of a table, for finalizing on the caller's
    threads rather than as tasks of a schedule."""

    var table: JoinHashTable
    var task: Int

    def __init__(out self, table: JoinHashTable, task: Int):
        self.table = table.copy()
        self.task = task

    def run(mut self) raises:
        self.table.finalize_task(self.task)


def _bucket_slots(
    physical: Int,
    rows: Int,
    count: Int,
    key_offset: Int,
    index_offset: Int,
    place: Int,
    base: UInt64,
    span: UInt64,
    shift: Int,
    numbers: Int,
    mut buckets: List[List[UInt64]],
) -> Bool:
    """Sort the `count` rows at the pointers at `rows` (the collection's
    rows from `place` on) into `buckets` by their keys' slots, a bucket
    per range of `1 << shift` slots: per row its slot and number in one
    word (the slot above), then its pointer. A row's number is its build
    row when the rows carry one (`index_offset` >= 0), else its place in
    the collection; the numbers also go to `numbers` (UInt32s). False,
    sorting no further, if a key's slot is past `span`."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if not D.is_floating_point():
            if physical == physical_of[D]():
                return _bucket_slots_of[D](
                    rows,
                    count,
                    key_offset,
                    index_offset,
                    place,
                    base,
                    span,
                    shift,
                    numbers,
                    buckets,
                )
    return False


def _bucket_slots_of[
    D: DType
](
    rows: Int,
    count: Int,
    key_offset: Int,
    index_offset: Int,
    place: Int,
    base: UInt64,
    span: UInt64,
    shift: Int,
    numbers: Int,
    mut buckets: List[List[UInt64]],
) -> Bool:
    var at = Pointer[Int, MutAnyOrigin](unsafe_from_address=rows)
    var numbered = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=numbers)
    for i in range(count):
        var row = at[unsafe_offset=i]
        var slot = _word[D](_load_as[D](row + key_offset)) - base
        if slot > span:
            return False
        var number = UInt32(place + i)
        if index_offset >= 0:
            number = _load32(row + index_offset)
        numbered[unsafe_offset=i] = number
        ref bucket = buckets[Int(slot >> UInt64(shift))]
        bucket.append((slot << 32) | UInt64(number))
        bucket.append(UInt64(row))
    return True


def _fill_slot_range(
    buckets: List[List[UInt64]],
    tasks: Int,
    ranges: Int,
    range_index: Int,
    slots: Int,
    bits: Int,
    slot_rows: Int,
) -> Bool:
    """Fill range `range_index`'s slots from every task's bucket for it
    (bucket `t * ranges + range_index` of task t): each row's number and
    pointer at its slot, its bit set. No other range shares a byte of
    bits or a cache line of slots, so no write needs an atomic. False,
    filling no further, on a slot set already: a repeated key."""
    var table = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=slots)
    var set = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bits)
    var pointers = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=slot_rows)
    for t in range(tasks):
        ref bucket = buckets[t * ranges + range_index]
        var words = bucket.unsafe_ptr()
        for j in range(0, len(bucket), 2):
            var word = words[unsafe_offset=j]
            var slot = Int(word >> 32)
            var byte = set[unsafe_offset=slot >> 3]
            var bit = UInt8(1) << UInt8(slot & 7)
            if byte & bit != 0:
                return False
            set[unsafe_offset=slot >> 3] = byte | bit
            table[unsafe_offset=slot] = UInt32(word & 0xFFFFFFFF)
            pointers[unsafe_offset=slot] = words[unsafe_offset=j + 1]
    return True


struct _InsertJob(Copyable, Job):
    """The finish task for one partition's region of the pointer table (at
    `base`, with `bitmask`): zero it, then insert the rows of the partition's
    chunks, last row first, so each chain lists its rows in build order."""

    var layout: KeyLayout
    var collection: Int
    # (segment, first chunk, end chunk) per run of chunks, in row order.
    var chunks: List[Tuple[Int, Int, Int]]
    var entries: Int
    var base: Int
    var bitmask: UInt64
    var duplicates: Bool
    var skip_null_keys: Bool
    # The bloom filter's words and their count (0: no filter), and whether
    # other tasks set bits in it at the same time.
    var bloom_words: Int
    var bloom_blocks: Int
    var bloom_shared: Bool
    var chained: Bool
    var kinds_at: Int
    var offsets_at: Int
    var widths_at: Int

    def __init__(
        out self,
        layout: KeyLayout,
        collection: Int,
        var chunks: List[Tuple[Int, Int, Int]],
        entries: Int,
        base: Int,
        bitmask: UInt64,
        duplicates: Bool,
        skip_null_keys: Bool,
        bloom_words: Int = 0,
        bloom_blocks: Int = 0,
        bloom_shared: Bool = False,
    ):
        self.layout = layout.copy()
        self.collection = collection
        self.chunks = chunks^
        self.entries = entries
        self.base = base
        self.bitmask = bitmask
        self.duplicates = duplicates
        self.skip_null_keys = skip_null_keys
        self.bloom_words = bloom_words
        self.bloom_blocks = bloom_blocks
        self.bloom_shared = bloom_shared
        self.chained = False
        self.kinds_at = 0
        self.offsets_at = 0
        self.widths_at = 0

    def run(mut self) raises:
        var state = TupleDataChunkState()
        self.fill(state)

    def fill(mut self, mut state: TupleDataChunkState):
        """Zero the region and insert the rows, reading their pointers
        through `state`."""
        self.kinds_at = Int(self.layout.kinds.unsafe_ptr())
        self.offsets_at = Int(self.layout.offsets.unsafe_ptr())
        self.widths_at = Int(self.layout.widths.unsafe_ptr())
        var region = self.entries + 8 * self.base
        unsafe_memset_zero(
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=region),
            8 * (Int(self.bitmask) + 1),
        )
        ref rows = Pointer[TupleDataCollection, MutAnyOrigin](
            unsafe_from_address=self.collection
        )[]
        var r = len(self.chunks) - 1
        while r >= 0:
            var segment = self.chunks[r][0]
            var c = self.chunks[r][2] - 1
            while c >= self.chunks[r][1]:
                var n = rows.segments[segment].initialize_chunk_state(
                    state, c, False
                )
                var at = state.row_locations.unsafe_ptr()
                if self.bloom_words != 0:
                    # The bloom filter from the hashes the rows store, so
                    # no key is read or hashed again.
                    var hash_at = self.layout.pointer_offset
                    for i in range(n):
                        bloom_set_bits(
                            self.bloom_words,
                            self.bloom_blocks,
                            _load64(at[unsafe_offset=i] + hash_at),
                            self.bloom_shared,
                        )
                var i = n - 1
                while i >= 0:
                    if not self.skip_null_keys or self._keys_valid(
                        at[unsafe_offset=i]
                    ):
                        self._insert(region, at[unsafe_offset=i])
                    i -= 1
                c -= 1
            r -= 1

    @always_inline
    def _keys_valid(self, row: Int) -> Bool:
        """Whether the row's key columns are all valid (the row's validity
        bytes, one bit a column; the keys are the layout's first columns)."""
        for k in range(len(self.layout.kinds)):
            if (_load8(row + (k >> 3)) >> UInt8(k & 7)) & 1 == 0:
                return False
        return True

    @always_inline
    def _match(self, a: Int, b: Int) -> Bool:
        return _same_keys(
            self.kinds_at,
            self.offsets_at,
            self.widths_at,
            len(self.layout.kinds),
            self.layout.key_bytes,
            self.layout.plain,
            a,
            b,
        )

    @always_inline
    def _insert(mut self, region: Int, row: Int):
        """The row joins the chain of its keys, or starts one in the first empty
        entry (linear probing)."""
        var table = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=region)
        var next = row + self.layout.pointer_offset
        var hash = _load64(next)
        var offset = Int(hash & self.bitmask)
        var salt = hash | POINTER_MASK
        var desired = (hash & SALT_MASK) | UInt64(row)
        while True:
            var entry = table[unsafe_offset=offset]
            if entry == 0:
                _store64(next, 0)
                table[unsafe_offset=offset] = desired
                return
            if (entry | POINTER_MASK) == salt:
                var other = Int(entry & POINTER_MASK)
                if self._match(
                    other + self.layout.key_offset, row + self.layout.key_offset
                ):
                    self.chained = True
                    if self.duplicates:
                        _store64(next, UInt64(other))
                        table[unsafe_offset=offset] = desired
                    return
            offset = Int((UInt64(offset) + 1) & self.bitmask)


@always_inline
def _word[D: DType](value: Scalar[D]) -> UInt64:
    """An integer as a 64-bit word: unsigned values zero-extended, signed
    ones sign-extended (two's complement), so the unsigned difference of
    two words is their distance."""
    comptime if D.is_unsigned():
        return UInt64(value)
    else:
        return bitcast[DType.uint64](Int64(value))


def _perfect_span(
    physical: Int, least: Int64, most: Int64, inserted: Int
) -> Optional[Tuple[UInt64, Int]]:
    """The base and span of a direct-address table for unique integer keys
    whose row values are `physical`, the least and greatest of them `least`
    and `most`: one slot per value from least to most, so none when the
    range is wider than PERFECT_MAX_BUILD_SIZE or narrower than the keys
    inserted (a repeated key, certainly). Whether the keys are unique is
    learned while filling the slots (`_fill_slots`)."""
    var integer = False
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if not D.is_floating_point():
            if physical == physical_of[D]():
                integer = True
    if not integer:
        return None
    # Both bounds fit 64-bit signed values, so their two's complement
    # words differ by the range's width.
    var base = bitcast[DType.uint64](least)
    var span = bitcast[DType.uint64](most) - base
    if span > UInt64(PERFECT_MAX_BUILD_SIZE):
        return None
    if UInt64(inserted) > span + 1:
        return None
    return (base, Int(span))


def _perfect_heads(
    key: Series,
    start: Int,
    count: Int,
    base: UInt64,
    span: Int,
    slots: Int,
    slot_bits: Int,
    heads: Int,
):
    """Each probe row's build row plus one (0 for none), looked up by `key -
    base`: the key's offset in the build range is its slot, so there is no
    hashing and no key comparison."""
    var valid = _validity(key)
    var bitmap = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=valid[0])
    var set = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=slot_bits)
    var out = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=heads)
    var limit = UInt64(span)
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if not D.is_floating_point():
            if key._data.isa[Column[Scalar[D]]]():
                var values = key._data[Column[Scalar[D]]]._ptr()
                for i in range(count):
                    var at = _word[D](values[unsafe_offset=start + i]) - base
                    if at > limit:
                        continue
                    if valid[0] != 0 and not _validity_at(
                        bitmap, valid[1] + start + i
                    ):
                        continue
                    var slot = Int(at)
                    if (set[unsafe_offset=slot >> 3] >> UInt8(slot & 7)) & 1:
                        out[unsafe_offset=i] = UInt64(slot) + 1
                return


def keys_only_join(how: Int) -> Bool:
    """Whether this join observes only key membership: semi, anti and mark
    joins need a single build row per key rather than every duplicate match."""
    return how == JOIN_SEMI or how == JOIN_ANTI or how == JOIN_MARK


def _validity(key: Series) -> Tuple[Int, Int]:
    """A key's validity bitmap address (0 when every row is valid) and the
    bit of its row 0."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if key._data.isa[Column[Scalar[D]]]():
            ref column = key._data[Column[Scalar[D]]]
            if len(column._bits[]) == 0 or column.null_count() == 0:
                return (0, 0)
            return (Int(column._bits[].unsafe_ptr()), column._offset)
    if key._data.isa[Column[Int128]]():
        ref column = key._data[Column[Int128]]
        if len(column._bits[]) == 0 or column.null_count() == 0:
            return (0, 0)
        return (Int(column._bits[].unsafe_ptr()), column._offset)
    if key._data.isa[BoolColumn]():
        ref column = key._data[BoolColumn]
        if len(column._bits[]) == 0 or key.null_count() == 0:
            return (0, 0)
        return (Int(column._bits[].unsafe_ptr()), column._offset)
    ref column = key._data[StringColumn]
    if len(column._bits[]) == 0 or key.null_count() == 0:
        return (0, 0)
    return (Int(column._bits[].unsafe_ptr()), column._offset)


struct JoinHashTable(Copyable, Movable):
    """A finalized join hash table over a build side's key columns, shared
    by every probing thread (copies share the same rows and entries)."""

    var data: ArcPointer[_TableData]
    var layout: KeyLayout
    # For an inner join with unique bounded integer build keys, look up key -
    # perfect_base in data.slots instead of hashing. Zero indicates an empty
    # slot; otherwise it identifies the build row.
    var perfect: Bool
    var perfect_base: UInt64
    var perfect_span: Int
    # Build rows, and those whose keys are all valid (in the table).
    var count: Int
    var inserted: Int
    # Entries over every region, the regions (one, or one per radix
    # partition), and the radix bits that pick a row's region.
    var capacity: Int
    var parts: Int
    var radix_bits: Int
    var chains_longer_than_one: Bool
    # Whether a build key is null (a mark join's unknown results).
    var has_null: Bool
    # Whether the build's subplan filters its tables, which the join's bloom
    # filter rule reads.
    var build_filtered: Bool
    # The probe's bloom filter over the single key's hashes, filled while
    # the table finalizes when the join's bloom rule holds.
    var bloom: Optional[ArcPointer[KeyBloom]]
    # The row layout, and the output columns gathered from it.
    var row_layout: TupleDataLayout
    var output_columns: List[Int]
    var output_names: List[String]
    var output_types: List[DataType]
    # Whether a probe emits `data.dictionary` (perfect or dictionary
    # emission) rather than gathering from the rows.
    var use_dictionary: Bool
    # The rows the collection holds: `inserted`, plus a right or full
    # join's rows with a null key.
    var rows_stored: Int
    # The probe side's estimated rows (dictionary emission at finish).
    var probe_estimate: Int
    # The key types, and per key its least and greatest valid value, known
    # for an integer key with a valid value (`bounded`).
    var key_types: List[DataType]
    var least: List[Int64]
    var most: List[Int64]
    var bounded: List[Bool]
    # The build side's columns in order (names, types, and the row field
    # holding each, -1 where the rows do not keep it), from which
    # `rows_frame` makes a frame of the rows.
    var build_names: List[String]
    var build_types: List[DataType]
    var build_fields: List[Int]

    def __init__(
        out self,
        keys: List[Series],
        perfect: Bool = False,
        keys_only: Bool = False,
        propagate: Bool = True,
    ) raises:
        """Build and finalize the eager join's table over `keys` (one
        column per equality condition, all of one length): the rows hold
        the keys and each row's build row number (`build_side_row`), which
        the eager join reads its build frame by. `perfect`: the join is an
        inner join, which may run as a perfect hash join. `keys_only`: the
        join is a semi, anti or mark join (`keys_only_join`), which keeps
        one row per key. `propagate`: the join reads the rows' found flags
        (it emits build rows after the probe)."""
        var columns = List[Series](capacity=len(keys))
        var key_columns = List[Int](capacity=len(keys))
        var build_names = List[String](capacity=len(keys))
        var build_types = List[DataType](capacity=len(keys))
        for key in keys:
            if key.dtype().is_nested():
                raise Error(
                    "list and struct columns cannot be join keys: " + key.name()
                )
            key_columns.append(len(columns))
            build_names.append(key.name())
            build_types.append(key.dtype())
            columns.append(key.rechunk() if key.is_chunked() else key.copy())
        var layout = ArcPointer(
            JoinBuildLayout(
                columns,
                List[Series](),
                key_columns,
                List[Int](),
                True,
                propagate and not perfect and not keys_only,
                List[Int](),
                List[String](),
                List[DataType](),
                perfect,
                keys_only,
                -1,
                build_names,
                build_types,
                key_columns,
            )
        )
        self = build_from_columns(layout, columns, False)

    def __init__(
        out self,
        layout: ArcPointer[JoinBuildLayout],
        var sinks: List[JoinBuildSink],
        defer_finalize: Bool,
        var keys: List[Series] = List[Series](),
    ) raises:
        """The table over the rows every build thread sank (`sinks`, in
        thread order; Combine, then Finalize): their partitions merged, the
        pointer table's regions planned, and the finalize tasks run here
        unless `defer_finalize`. `keys`: the build's key columns when they
        exist as columns (a build from a frame), which `key_column` then
        reads instead of the rows."""
        ref shape = layout[]
        self.layout = shape.key_layout.copy()
        self.row_layout = shape.row_layout.copy()
        self.output_columns = shape.output_columns.copy()
        self.output_names = shape.output_names.copy()
        self.output_types = shape.output_types.copy()
        self.key_types = shape.key_types.copy()
        self.build_names = shape.build_names.copy()
        self.build_types = shape.build_types.copy()
        self.build_fields = shape.build_fields.copy()
        self.use_dictionary = False
        self.rows_stored = 0
        self.probe_estimate = shape.probe_estimate
        self.data = ArcPointer(_TableData(keys^, shape.row_layout))
        # Combine: the threads' partitions merged, thread by thread, and
        # their key bounds with them.
        var fields = len(shape.key_columns)
        var sink = RadixPartitionedTupleData(
            shape.row_layout, INITIAL_RADIX_BITS, shape.hash_column()
        )
        var invalid = 0
        var count = 0
        var strings = List[_Buffer[UInt8]]()
        self.least = List[Int64](length=fields, fill=Int64.MAX)
        self.most = List[Int64](length=fields, fill=Int64.MIN)
        self.bounded = shape.bounded.copy()
        var seen = List[Bool](length=fields, fill=False)
        for w in range(len(sinks)):
            invalid += sinks[w].invalid
            count += sinks[w].count
            sink.combine(sinks[w].rows)
            for buffer in sinks[w].strings:
                strings.append(buffer.copy())
            for k in range(fields):
                self.bounded[k] = self.bounded[k] and sinks[w].exact[k]
                if sinks[w].seen[k]:
                    seen[k] = True
                    self.least[k] = min(self.least[k], sinks[w].least[k])
                    self.most[k] = max(self.most[k], sinks[w].most[k])
        for k in range(fields):
            self.bounded[k] = self.bounded[k] and seen[k]
        if count > Int(Int32.MAX):
            raise Error("A join build exceeds 2^31 - 1 rows")
        var sink_threads = max(len(sinks), 1)
        _ = sinks^
        self.count = count
        self.has_null = invalid > 0
        self.build_filtered = shape.build_filtered
        self.bloom = None
        self.inserted = self.count - invalid
        self.radix_bits = INITIAL_RADIX_BITS
        self.perfect = False
        self.perfect_base = 0
        self.perfect_span = 0
        self.chains_longer_than_one = False
        self.capacity = 0
        self.parts = 1
        # Each partition's rows and segments, then Unpartition: the
        # collection lists partition 0's segments first.
        var partition_count = sink.partition_count()
        var partition_rows = List[Int](capacity=partition_count)
        var segment_starts = List[Int](capacity=partition_count + 1)
        var first_segment = len(self.data[].collection.segments)
        for p in range(partition_count):
            partition_rows.append(sink.partitions[p].count())
            segment_starts.append(first_segment)
            if sink.partitions[p].count() > 0:
                first_segment += len(sink.partitions[p].segments)
        segment_starts.append(first_segment)
        self.data[].collection = sink.get_unpartitioned()
        self.data[].heaps = self.data[].collection.heaps()
        self.data[].heaps.add_buffers(strings)
        self.rows_stored = self.data[].collection.count()
        self.data[].partition_rows = partition_rows^
        self.data[].segment_starts = segment_starts^
        self.data[].sink_threads = sink_threads
        self.data[].duplicates = not shape.keys_only
        self.data[].skip_null_keys = self.has_null and shape.found
        if self.inserted == 0:
            return
        var span = Optional[Tuple[UInt64, Int]]()
        if (
            shape.perfect
            and not shape.keys_only
            and fields == 1
            and not shape.nested
            and self.bounded[0]
        ):
            # Unique integer keys over a narrow range index their slots
            # directly. Nested build columns are not supported there (a
            # slot array holds fixed-width values or string_t only), and a
            # repeated key or a wide range falls back to the hash table.
            span = _perfect_span(
                shape.row_layout.nodes[][0].physicals[0],
                self.least[0],
                self.most[0],
                self.inserted,
            )
        if span:
            self._plan_slots(
                shape.row_layout.nodes[][0].physicals[0],
                span.value()[0],
                span.value()[1],
            )
        else:
            self._plan_pointer_table()
        if not defer_finalize:
            self._finalize_here()

    def _finalize_here(mut self) raises:
        """Run the finalize tasks on this thread and the crew's, round by
        round, each round's step after it."""
        while True:
            var tasks = self.finalize_tasks()
            if tasks == 0:
                return
            if tasks == 1:
                self.finalize_task(0)
            else:
                var jobs = List[_TableTask](capacity=tasks)
                for t in range(tasks):
                    jobs.append(_TableTask(self, t))
                run_jobs(jobs)
            self.finish_finalize()

    def _plan_slots(mut self, physical: Int, base: UInt64, span: Int) raises:
        """Plan a perfect hash join's finalize tasks for keys `base` to
        `base + span`, two rounds of them. The first: as many tasks as the
        build had sink threads, each reading a run of whole chunks of the
        rows (balanced by rows): it gathers its rows into every output
        column, and sorts their slots into a bucket per range of slots.
        Each output column is a dictionary over the build rows, each row
        at its number (its build row, else its place in the collection),
        so the tasks write disjoint entries of columns allocated here. The
        second: a task per range of slots, filling its slots from every
        first-round task's bucket for it. A range covers whole cache
        lines of every slot array (at least 512 slots, a 64-byte line of
        bits), so no two tasks write near each other, and a repeated key
        is found exactly, without atomics."""
        self.perfect = True
        self.perfect_base = base
        self.perfect_span = span
        var size = span + 1
        self.data[].slots = List[UInt32](unsafe_uninit_length=size)
        self.data[].slot_rows = List[UInt64](unsafe_uninit_length=size)
        self.data[].slot_bits = List[UInt8](
            unsafe_uninit_length=(size + 7) // 8
        )
        self.data[].repeated[0] = 0
        self.data[].slot_physical = physical
        ref rows = self.data[].collection
        var chunks = List[Tuple[Int, Int]]()
        var counts = List[Int]()
        for s in range(len(rows.segments)):
            ref segment = rows.segments[s]
            for c in range(segment.chunk_count()):
                chunks.append((s, c))
                counts.append(segment.chunks[c].count)
        var tasks = max(1, min(self.data[].sink_threads, len(chunks)))
        var runs = List[List[Tuple[Int, Int]]](capacity=tasks)
        var firsts = List[Int](capacity=tasks)
        var at = 0
        var place = 0
        for t in range(tasks):
            var end_rows = (t + 1) * self.rows_stored // tasks
            var run = List[Tuple[Int, Int]]()
            firsts.append(place)
            # Each later task keeps at least one chunk.
            var limit = len(chunks) - (tasks - t - 1)
            while at < limit and (
                len(run) == 0 or place < end_rows or t == tasks - 1
            ):
                run.append(chunks[at])
                place += counts[at]
                at += 1
            runs.append(run^)
        self.data[].slot_runs = runs^
        self.data[].slot_firsts = firsts^
        # Ranges of a power of two slots, as many as the first round's
        # tasks unless that would make them narrower than 512 slots.
        var width = max(512, next_power_of_two((size + tasks - 1) // tasks))
        var shift = 0
        while (1 << shift) < width:
            shift += 1
        self.data[].slot_shift = shift
        self.data[].slot_ranges = (size + width - 1) // width
        self.data[].buckets = List[List[UInt64]](
            length=tasks * self.data[].slot_ranges, fill=List[UInt64]()
        )
        # Every entry gets a row when every build row is in the table;
        # numbered rows without one (null keys) leave theirs empty.
        var gathers = List[NumberedGather](capacity=len(self.output_columns))
        for c in range(len(self.output_columns)):
            gathers.append(
                NumberedGather(
                    self.row_layout,
                    self.output_columns[c],
                    self.output_names[c],
                    self.output_types[c],
                    self.count,
                    tasks,
                    self.count == self.rows_stored,
                )
            )
        self.data[].gathers = gathers^
        self.data[].phase = 1

    def _plan_pointer_table(mut self) raises:
        """Plan a hash join's finalize tasks: the pointer table, each
        partition its own region (together at least MINIMUM_CAPACITY), and
        as many tasks as the build had sink threads, each filling a run of
        partitions' regions."""
        ref partition_rows = self.data[].partition_rows
        ref segment_starts = self.data[].segment_starts
        var sink_threads = self.data[].sink_threads
        ref rows = self.data[].collection
        var collection = Int(Pointer(to=self.data[].collection))
        # The probe's bloom filter, when the join's rule holds: a single
        # integer key, whose row hash is the filter's hash of the key, so
        # the finalize tasks fill it from the hashes the rows store.
        var bloom_words = 0
        var bloom_blocks = 0
        if (
            len(self.key_types) == 1
            and (
                self.key_types[0].physical() == DataType.INT64
                or self.key_types[0].physical() == DataType.INT32
            )
            and can_use_bloom_filter(
                1, self.inserted, self.probe_estimate, self.build_filtered
            )
        ):
            bloom_blocks = bloom_blocks_for(self.inserted)
            if bloom_blocks > 0:
                self.bloom = ArcPointer(KeyBloom(bloom_blocks))
                bloom_words = Int(self.bloom.value()[].words.unsafe_ptr())
        self.parts = len(partition_rows)
        self.capacity = 0
        self.data[].bases = List[UInt64]()
        self.data[].masks = List[UInt64]()
        var floor = next_power_of_two(
            (MINIMUM_CAPACITY + self.parts - 1) // self.parts
        )
        for p in range(self.parts):
            var size = max(
                next_power_of_two(
                    Int(Float64(partition_rows[p]) * DEFAULT_LOAD_FACTOR)
                ),
                floor,
            )
            self.data[].bases.append(UInt64(self.capacity))
            self.data[].masks.append(UInt64(size - 1))
            self.capacity += size
        self.data[].entries = huge_uninit[UInt64](self.capacity)
        var entries = Int(self.data[].entries.unsafe_ptr())
        var inserts = List[_InsertJob](capacity=self.parts)
        for p in range(self.parts):
            var runs = List[Tuple[Int, Int, Int]]()
            for s in range(segment_starts[p], segment_starts[p + 1]):
                var chunks = rows.segments[s].chunk_count()
                if chunks > 0:
                    runs.append((s, 0, chunks))
            inserts.append(
                _InsertJob(
                    self.layout,
                    collection,
                    runs^,
                    entries,
                    Int(self.data[].bases[p]),
                    self.data[].masks[p],
                    self.data[].duplicates,
                    self.data[].skip_null_keys,
                    bloom_words,
                    bloom_blocks,
                    sink_threads > 1,
                )
            )
        # As many tasks as the build had sink threads, each filling a run
        # of partitions' regions.
        var tasks = List[_FinalizeJob](capacity=sink_threads)
        for _ in range(sink_threads):
            tasks.append(_FinalizeJob())
        for p in range(self.parts):
            tasks[p * sink_threads // self.parts].regions.append(
                inserts[p].copy()
            )
        self.data[].pending = tasks^

    def key_bounds(self, k: Int) -> Optional[Tuple[Int64, Int64]]:
        """Key `k`'s least and greatest valid value, when it is an integer
        key with a valid value."""
        if not self.bounded[k]:
            return None
        return (self.least[k], self.most[k])

    def _row_pointers(self) raises -> List[UInt64]:
        """Every row's address, in the collection's order."""
        var pointers = List[UInt64](capacity=self.rows_stored)
        var state = TupleDataChunkState()
        ref rows = self.data[].collection
        for s in range(len(rows.segments)):
            ref segment = rows.segments[s]
            for c in range(segment.chunk_count()):
                var n = segment.initialize_chunk_state(state, c, False)
                var at = state.row_locations.unsafe_ptr()
                for i in range(n):
                    pointers.append(UInt64(at[unsafe_offset=i]))
        return pointers^

    def key_column(self, k: Int, var name: String) raises -> Series:
        """Key `k` as a column named `name`: the build's own key column
        when the table was built from one, else gathered from the rows
        (every row the table keeps, in no particular order)."""
        if k < len(self.data[].keys):
            return self.data[].keys[k].renamed(name^)
        var pointers = self._row_pointers()
        var column = self.gather_field(
            k,
            Int(pointers.unsafe_ptr()),
            len(pointers),
            name^,
            self.key_types[k],
        )
        _ = pointers^
        return column^

    def rows_frame(self) raises -> List[Series]:
        """The build side's columns over the rows the table keeps, for an
        operator that reads the build's columns rather than the table: a
        row with a null key is missing unless the join reads found flags
        (no such row ever matches), and a column the rows do not keep (the
        payload of a join that outputs no build column) is null."""
        var pointers = self._row_pointers()
        var n = len(pointers)
        var columns = List[Series](capacity=len(self.build_names))
        for c in range(len(self.build_names)):
            var field = self.build_fields[c]
            if field < 0:
                columns.append(
                    Series.full_null(
                        self.build_names[c], self.build_types[c], n
                    )
                )
            else:
                columns.append(
                    self.gather_field(
                        field,
                        Int(pointers.unsafe_ptr()),
                        n,
                        self.build_names[c],
                        self.build_types[c],
                    )
                )
        _ = pointers^
        return columns^

    def finalize_tasks(self) -> Int:
        """The finalize tasks of the round to run next, 0 once the table is
        done. A hash join's: one per build sink thread, each filling a run
        of partitions' regions of the pointer table, so the tasks never
        write the same entries and need no atomics. A perfect hash join's
        two rounds (`_plan_slots`): one per build sink thread, then one per
        range of slots."""
        if self.data[].phase == 1:
            return len(self.data[].slot_runs)
        if self.data[].phase == 2:
            return self.data[].slot_ranges
        return len(self.data[].pending)

    def finalize_task(self, task: Int) raises:
        """Run finalize task `task` of the round under way; tasks run at
        the same time, each on its own regions, rows or slots."""
        if self.data[].phase == 1:
            self._perfect_rows(task)
        elif self.data[].phase == 2:
            self._perfect_slots(task)
        else:
            self.data[].pending[task].run()

    def _repeat_found(self) -> Bool:
        return (
            Pointer[Atomic[UInt64], MutAnyOrigin](
                unsafe_from_address=Int(self.data[].repeated.unsafe_ptr())
            )[].load()
            != 0
        )

    def _record_repeat(self):
        Pointer[Atomic[UInt64], MutAnyOrigin](
            unsafe_from_address=Int(self.data[].repeated.unsafe_ptr())
        )[].store(1)

    def _perfect_rows(self, task: Int) raises:
        """A perfect hash join's first-round task `task`: for each chunk
        of its run, sort the rows' slots into its buckets, then gather the
        rows into every output column."""
        var ranges = self.data[].slot_ranges
        var buckets = List[List[UInt64]](capacity=ranges)
        ref run = self.data[].slot_runs[task]
        var rows_in_run = 0
        for r in range(len(run)):
            rows_in_run += (
                self.data[]
                .collection.segments[run[r][0]]
                .chunks[run[r][1]]
                .count
            )
        # Keys spread over the ranges about evenly when they are dense.
        var expected = 2 * (rows_in_run // ranges + rows_in_run // 8 + 8)
        for _ in range(ranges):
            buckets.append(List[UInt64](capacity=expected))
        var numbers = List[UInt32](unsafe_uninit_length=STANDARD_VECTOR_SIZE)
        var state = TupleDataChunkState()
        ref rows = self.data[].collection
        var place = self.data[].slot_firsts[task]
        var span = UInt64(self.perfect_span)
        for r in range(len(run)):
            if self._repeat_found():
                return
            var count = rows.segments[run[r][0]].initialize_chunk_state(
                state, run[r][1], False
            )
            var at = Int(state.row_locations.unsafe_ptr())
            if not _bucket_slots(
                self.data[].slot_physical,
                at,
                count,
                self.layout.key_offset,
                self.layout.index_offset,
                place,
                self.perfect_base,
                span,
                self.data[].slot_shift,
                Int(numbers.unsafe_ptr()),
                buckets,
            ):
                self._record_repeat()
                return
            for c in range(len(self.data[].gathers)):
                self.data[].gathers[c].gather(
                    at,
                    Int(numbers.unsafe_ptr()),
                    count,
                    task,
                    self.data[].heaps,
                )
            place += count
        for r in range(ranges):
            swap(self.data[].buckets[task * ranges + r], buckets[r])
        _ = numbers^

    def _perfect_slots(self, task: Int) raises:
        """A perfect hash join's second-round task `task`: clear its range
        of slot bits, then fill the range's slots from every first-round
        task's bucket for it."""
        var shift = self.data[].slot_shift
        var size = self.perfect_span + 1
        var low = task << shift
        var high = min(size, (task + 1) << shift)
        var bits = Int(self.data[].slot_bits.unsafe_ptr())
        # `low` is a multiple of 512, so the range's bits start a byte.
        unsafe_memset_zero(
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bits + (low >> 3)),
            (high + 7) // 8 - (low >> 3),
        )
        if self._repeat_found():
            return
        if not _fill_slot_range(
            self.data[].buckets,
            len(self.data[].slot_runs),
            self.data[].slot_ranges,
            task,
            Int(self.data[].slots.unsafe_ptr()),
            bits,
            Int(self.data[].slot_rows.unsafe_ptr()),
        ):
            self._record_repeat()

    def finish_finalize(mut self) raises:
        """After a round of finalize tasks. A perfect hash join's first
        round arms its second; after the second its gathered columns
        become the dictionary the probe emits from. A repeated key found
        in either turns the join back into a hash join, whose pointer
        table is the next round. After a hash join's round: whether any
        chain is longer than one, then dictionary emission's arrays when
        they apply."""
        if self.data[].phase == 1:
            self.data[].phase = 2
            if self.data[].repeated[0] == 0:
                return
        if self.data[].phase == 2:
            self.data[].phase = 0
            self.data[].slot_runs = List[List[Tuple[Int, Int]]]()
            self.data[].slot_firsts = List[Int]()
            self.data[].buckets = List[List[UInt64]]()
            if self.data[].repeated[0] == 0:
                trace_path("join.perfect_hash")
                if len(self.output_columns) > 0:
                    var columns = List[Series](
                        capacity=len(self.output_columns)
                    )
                    var decoded = List[List[UInt8]](
                        capacity=len(self.output_columns)
                    )
                    for c in range(len(self.output_columns)):
                        var values = List[UInt8]()
                        columns.append(
                            self.data[]
                            .gathers[c]
                            .finish(self.data[].heaps, values)
                        )
                        decoded.append(values^)
                    self._set_dictionary(columns^, decoded^)
                self.data[].gathers = List[NumberedGather]()
                return
            # A repeated key: the hash table instead, its pointer table the
            # next round.
            self.perfect = False
            self.perfect_base = 0
            self.perfect_span = 0
            self.data[].slots = List[UInt32]()
            self.data[].slot_bits = List[UInt8]()
            self.data[].slot_rows = List[UInt64]()
            self.data[].gathers = List[NumberedGather]()
            self._plan_pointer_table()
            return
        if len(self.data[].pending) == 0:
            return
        for t in range(len(self.data[].pending)):
            self.chains_longer_than_one = (
                self.chains_longer_than_one or self.data[].pending[t].chained
            )
        self.data[].pending = List[_FinalizeJob]()
        if self._can_use_dictionary_emission(self.probe_estimate):
            self._build_dictionary_arrays()

    def _can_use_dictionary_emission(self, probe_cardinality: Int) -> Bool:
        """Whether to gather the build side once and emit probe results as row
        numbers into it (see `_DICT_MAX_PAYLOAD_BYTES`)."""
        if len(self.output_columns) == 0 or self.rows_stored == 0:
            return False
        if probe_cardinality < _DICT_MIN_PROBE_ROWS:
            return False
        if probe_cardinality < _DICT_PROBE_BUILD_RATIO * self.rows_stored:
            return False
        var bytes_per_row = 0
        for t in self.output_types:
            if t.is_nested():
                return False
            try:
                bytes_per_row += physical_size(physical_of_dtype(t))
            except:
                return False
        return self.rows_stored * bytes_per_row <= _DICT_MAX_PAYLOAD_BYTES

    def _gather_dictionary(mut self, rows: List[UInt64]) raises:
        """Gather every output column at the row pointers `rows`."""
        var out = List[Series](capacity=len(self.output_columns))
        for c in range(len(self.output_columns)):
            out.append(
                self.gather(
                    c, Int(rows.unsafe_ptr()), len(rows), self.output_names[c]
                )
            )
        self._set_dictionary(out^)

    def _set_dictionary(
        mut self,
        var columns: List[Series],
        var decoded: List[List[UInt8]] = List[List[UInt8]](),
    ):
        """The probe emits `columns`, each decoded once here unless
        `decoded` holds its values decoded already."""
        for c in range(len(decoded), len(columns)):
            decoded.append(List[UInt8]())
        for c in range(len(columns)):
            if len(decoded[c]) == 0:
                decoded[c] = decode_values(columns[c])
        self.data[].decoded = decoded^
        self.data[].dictionary = columns^
        self.use_dictionary = True

    def dictionary_values(self, column: Int) -> Int:
        """The address of dictionary column `column`'s decoded values, 0 for
        a fixed-width column."""
        if len(self.data[].decoded[column]) == 0:
            return 0
        return Int(self.data[].decoded[column].unsafe_ptr())

    def gather(
        self, output: Int, rows: Int, count: Int, var name: String
    ) raises -> Series:
        """Output column `output` of the `count` rows at the pointers at
        `rows`, named `name`: the build side's values of the matched
        rows; strings reference the rows' heap."""
        return self.gather_field(
            self.output_columns[output],
            rows,
            count,
            name^,
            self.output_types[output],
        )

    def gather_field(
        self,
        column: Int,
        rows: Int,
        count: Int,
        var name: String,
        dtype: DataType,
    ) raises -> Series:
        """Row layout column `column` (a key, or a payload column) of the
        `count` rows at the pointers at `rows`, typed `dtype`."""
        return gather_rows_series(
            self.row_layout,
            rows,
            count,
            column,
            name^,
            dtype,
            self.data[].heaps,
        )

    def _build_dictionary_arrays(mut self) raises:
        """Gather every output column into a dictionary over the rows in
        collection order (one extra null row last for a left join's unmatched
        probe rows), each row's number stored where its next pointer was, and
        the next pointers kept in `aux_next` (a chain ends at the extra
        slot)."""
        var pointers = self._row_pointers()
        pointers.append(UInt64(self.null_row_address()))
        self._gather_dictionary(pointers)
        var next_offset = self.layout.pointer_offset
        self.data[].aux_next = List[UInt64](
            unsafe_uninit_length=self.rows_stored + 1
        )
        var aux = self.data[].aux_next.unsafe_ptr()
        aux[unsafe_offset=self.rows_stored] = 0
        for i in range(self.rows_stored):
            var row = Int(pointers[i])
            aux[unsafe_offset=i] = _load64(row + next_offset)
            _store64(row + next_offset, UInt64(i))

    def null_row_address(self) -> Int:
        """A row whose every column is null, for a probe row with no match."""
        return Int(self.data[].null_row.unsafe_ptr())

    def build_side_row(self, pointer: UInt64) -> Int:
        """The build row of the row at `pointer`, -1 for the null row."""
        if Int(pointer) == self.null_row_address():
            return -1
        return Int(_load32(Int(pointer) + self.layout.index_offset))

    def entries_address(self) -> Int:
        return Int(self.data[].entries.unsafe_ptr())

    def reset_found(self):
        """Clear every row's found flag before a new probe phase."""
        var found_offset = self.layout.found_offset
        if found_offset < 0:
            return
        var state = TupleDataChunkState()
        ref rows = self.data[].collection
        for s in range(len(rows.segments)):
            ref segment = rows.segments[s]
            for c in range(segment.chunk_count()):
                var n = segment.initialize_chunk_state(state, c, False)
                var at = state.row_locations.unsafe_ptr()
                for i in range(n):
                    _store8(at[unsafe_offset=i] + found_offset, 0)

    def output_index(self, name: String) -> Int:
        """The output column `name`'s place among the outputs gathered from
        the rows, -1 when the join does not output it."""
        for c in range(len(self.output_names)):
            if self.output_names[c] == name:
                return c
        return -1

    def use_salt(self) -> Bool:
        """Whether probes compare salts (see `USE_SALT_THRESHOLD`)."""
        return self.capacity > USE_SALT_THRESHOLD

    def scan_chunk(
        self, chunk: Int, want: UInt8, mut rows: List[UInt64]
    ) raises:
        """Scan one chunk of the rows for their found flags (each worker takes
        its own): append the pointers of the rows whose found flag is `want` (0:
        no probe matched them; 1: a right-semi join's matches)."""
        var state = TupleDataChunkState()
        var found_offset = self.layout.found_offset
        var count = self.data[].collection.fetch_chunk(state, chunk, False)
        var at = state.row_locations.unsafe_ptr()
        for i in range(count):
            var row = at[unsafe_offset=i]
            if _load8(row + found_offset) == want:
                rows.append(UInt64(row))

    def chunk_count(self) -> Int:
        """The rows' chunks (one scan morsel each)."""
        return self.data[].collection.chunk_count()

    def full_outer_rows(self, want: UInt8) raises -> List[Int]:
        """Every build row whose found flag is `want`."""
        var rows = List[Int]()
        var pointers = List[UInt64]()
        for c in range(self.chunk_count()):
            self.scan_chunk(c, want, pointers)
        for r in range(len(pointers)):
            rows.append(self.build_side_row(pointers[r]))
        # In build order, as the eager join's output lists them.
        sort(rows)
        return rows^


struct ScanStructure(Movable):
    """One thread's probe of a JoinHashTable, a chunk of at most
    STANDARD_VECTOR_SIZE probe rows at a time.

    `probe` hashes the chunk's keys and finds each row's chain; `next` then
    emits the join's rows for the chunk, at most STANDARD_VECTOR_SIZE per call,
    into `probe_rows` (the chunk's row, from 0) and `build_rows` (the build row,
    -1 for none), until it returns 0. A mark join emits every probe row, its
    result in `marks` (1 matched, 0 not, 2 unknown)."""

    var table: JoinHashTable
    var how: Int
    var count: Int
    var hashes: List[UInt64]
    var valid: List[UInt8]
    var records: List[UInt8]
    var heads: List[UInt64]
    var offsets: List[UInt64]
    # Each row's partition region: its first entry and bitmask.
    var region: List[UInt64]
    var masks: List[UInt64]
    var todo: List[Int32]
    # The partition split's scratch: each pending row's partition and each
    # partition's first place.
    var part_of: List[Int32]
    var part_starts: List[Int]
    var compare: List[Int32]
    var no_match: List[Int32]
    # Emission state: the probe row emitting, and its chain's next row.
    var row: Int
    var cursor: UInt64
    var started: Bool
    var finished: Bool
    var probe_rows: List[Int32]
    var build_rows: List[Int32]
    # The matched rows' pointers (the table's null row for none).
    var row_pointers: List[UInt64]
    var marks: List[UInt8]
    # The join flags matched build rows (right, full, right-semi/anti).
    var propagate: Bool

    def __init__(out self, table: JoinHashTable, how: Int):
        self.table = table.copy()
        self.how = how
        self.count = 0
        var n = STANDARD_VECTOR_SIZE
        self.hashes = List[UInt64](unsafe_uninit_length=n)
        self.valid = List[UInt8](unsafe_uninit_length=n)
        self.records = List[UInt8](
            unsafe_uninit_length=n * max(table.layout.key_bytes, 8)
        )
        self.heads = List[UInt64](length=n, fill=0)
        self.offsets = List[UInt64](unsafe_uninit_length=n)
        self.region = List[UInt64](unsafe_uninit_length=n)
        self.masks = List[UInt64](unsafe_uninit_length=n)
        self.todo = List[Int32](unsafe_uninit_length=n)
        self.part_of = List[Int32](unsafe_uninit_length=n)
        self.part_starts = List[Int](length=table.parts + 1, fill=0)
        self.compare = List[Int32](unsafe_uninit_length=n)
        self.no_match = List[Int32](unsafe_uninit_length=n)
        self.row = 0
        self.cursor = 0
        self.started = False
        self.finished = True
        self.probe_rows = List[Int32](unsafe_uninit_length=n)
        self.build_rows = List[Int32](unsafe_uninit_length=n)
        self.row_pointers = List[UInt64](unsafe_uninit_length=n)
        self.marks = List[UInt8](unsafe_uninit_length=n)
        self.propagate = (
            how == JOIN_RIGHT
            or how == JOIN_FULL
            or how == JOIN_RIGHT_SEMI
            or how == JOIN_RIGHT_ANTI
        )

    def probe(mut self, keys: List[Series], start: Int, count: Int) raises:
        """Probe rows [start, start + count) of the unchunked `keys` (at most
        STANDARD_VECTOR_SIZE rows): each row's first candidate entry, followed
        by `next` to find its matches."""
        self.count = count
        self.row = 0
        self.cursor = 0
        self.started = False
        self.finished = False
        ref layout = self.table.layout
        var heads = self.heads.unsafe_ptr()
        for i in range(count):
            heads[unsafe_offset=i] = 0
        if self.table.perfect:
            # The key's slot holds its build row, emitted by `next` with no
            # chain to walk.
            _perfect_heads(
                keys[0],
                start,
                count,
                self.table.perfect_base,
                self.table.perfect_span,
                Int(self.table.data[].slots.unsafe_ptr()),
                Int(self.table.data[].slot_bits.unsafe_ptr()),
                Int(heads),
            )
            return
        var valid = self.valid.unsafe_ptr()
        for i in range(count):
            valid[unsafe_offset=i] = 1
        var records = Int(self.records.unsafe_ptr())
        var key_bytes = layout.key_bytes
        for k in range(len(keys)):
            _write_native(
                keys[k],
                start,
                count,
                records + layout.offsets[k],
                key_bytes,
                Int(valid),
                1,
            )
            _hash_column(
                keys[k],
                start,
                start + count,
                Int(self.hashes.unsafe_ptr()),
                k == 0,
                output_offset=start,
            )
        if self.table.inserted == 0:
            return
        # Rows with a null key match nothing.
        var todo = self.todo.unsafe_ptr()
        var pending = 0
        for i in range(count):
            todo[unsafe_offset=pending] = Int32(i)
            pending += Int(valid[unsafe_offset=i])
        self._row_pointers(pending)

    def _row_pointers(mut self, var pending: Int):
        """Each pending row's chain, by linear probing to an empty entry or one
        whose salt matches and whose keys are equal."""
        var table = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.table.entries_address()
        )
        var parts = self.table.parts
        var radix_bits = self.table.radix_bits
        var bases = self.table.data[].bases.unsafe_ptr()
        var masks = self.table.data[].masks.unsafe_ptr()
        var salted = self.table.use_salt()
        var hashes = self.hashes.unsafe_ptr()
        var offsets = self.offsets.unsafe_ptr()
        var heads = self.heads.unsafe_ptr()
        var todo = self.todo.unsafe_ptr()
        var compare = self.compare.unsafe_ptr()
        var no_match = self.no_match.unsafe_ptr()
        ref layout = self.table.layout
        var kinds = Int(layout.kinds.unsafe_ptr())
        var key_offsets = Int(layout.offsets.unsafe_ptr())
        var widths = Int(layout.widths.unsafe_ptr())
        var fields = len(layout.kinds)
        var key_bytes = layout.key_bytes
        var plain = layout.plain
        var key_offset = layout.key_offset
        var records = Int(self.records.unsafe_ptr())
        var region = self.region.unsafe_ptr()
        var bitmasks = self.masks.unsafe_ptr()
        # Split probe row indices by hash partition with a counting pass and a
        # placing pass. Keep indices ascending within each partition so
        # lookups reuse one table region while preserving the required per-
        # partition order.
        if parts > 1:
            var starts = self.part_starts.unsafe_ptr()
            var part_of = self.part_of.unsafe_ptr()
            for p in range(parts + 1):
                starts[unsafe_offset=p] = 0
            for t in range(pending):
                var i = Int(todo[unsafe_offset=t])
                var p = radix_partition(hashes[unsafe_offset=i], radix_bits)
                part_of[unsafe_offset=t] = Int32(p)
                starts[unsafe_offset=p + 1] += 1
            for p in range(parts):
                starts[unsafe_offset=p + 1] += starts[unsafe_offset=p]
            for t in range(pending):
                var p = Int(part_of[unsafe_offset=t])
                var at = starts[unsafe_offset=p]
                compare[unsafe_offset=at] = todo[unsafe_offset=t]
                starts[unsafe_offset=p] = at + 1
            var placed = 0
            for p in range(parts):
                var base = bases[unsafe_offset=p]
                var mask = masks[unsafe_offset=p]
                var end = starts[unsafe_offset=p]
                while placed < end:
                    var i = Int(compare[unsafe_offset=placed])
                    todo[unsafe_offset=placed] = Int32(i)
                    region[unsafe_offset=i] = base
                    bitmasks[unsafe_offset=i] = mask
                    offsets[unsafe_offset=i] = base + (
                        hashes[unsafe_offset=i] & mask
                    )
                    placed += 1
        else:
            var base = bases[unsafe_offset=0]
            var mask = masks[unsafe_offset=0]
            for t in range(pending):
                var i = Int(todo[unsafe_offset=t])
                region[unsafe_offset=i] = base
                bitmasks[unsafe_offset=i] = mask
                offsets[unsafe_offset=i] = base + (
                    hashes[unsafe_offset=i] & mask
                )
        # Prefetch all buckets in a probe block before reading any candidate
        # rows, overlapping memory requests once the table is too large to
        # stay cached.
        var fetch = self.table.capacity >= MIN_PREFETCH_BUCKETS
        while pending > 0:
            # Walk to an empty entry or a salt match.
            var candidates = 0
            for block in range(0, pending, BLOCK_SIZE):
                var block_end = min(block + BLOCK_SIZE, pending)
                if fetch:
                    for t in range(block, block_end):
                        var i = Int(todo[unsafe_offset=t])
                        prefetch(
                            table.unsafe_offset(Int(offsets[unsafe_offset=i]))
                        )
                for t in range(block, block_end):
                    var i = Int(todo[unsafe_offset=t])
                    var hash = hashes[unsafe_offset=i]
                    var salt = hash | POINTER_MASK
                    var offset = offsets[unsafe_offset=i]
                    var base = region[unsafe_offset=i]
                    var mask = bitmasks[unsafe_offset=i]
                    while True:
                        var entry = table[unsafe_offset=Int(offset)]
                        if entry == 0:
                            break
                        if not salted or (entry | POINTER_MASK) == salt:
                            var head = entry & POINTER_MASK
                            heads[unsafe_offset=i] = head
                            compare[unsafe_offset=candidates] = Int32(i)
                            candidates += 1
                            if fetch:
                                # The candidate's row, for the key comparison
                                # below.
                                prefetch(
                                    Pointer[UInt64, MutAnyOrigin](
                                        unsafe_from_address=Int(head)
                                        + key_offset
                                    )
                                )
                            break
                        offset = base + ((offset - base + 1) & mask)
                    offsets[unsafe_offset=i] = offset
            # Compare the candidates' keys; a mismatch goes on probing from the
            # next entry.
            var misses = 0
            if plain and key_bytes == 8:
                for c in range(candidates):
                    var i = Int(compare[unsafe_offset=c])
                    var head = Int(heads[unsafe_offset=i])
                    if _load64(head + key_offset) != _load64(records + i * 8):
                        heads[unsafe_offset=i] = 0
                        no_match[unsafe_offset=misses] = Int32(i)
                        misses += 1
            elif plain and key_bytes == 4:
                for c in range(candidates):
                    var i = Int(compare[unsafe_offset=c])
                    var head = Int(heads[unsafe_offset=i])
                    if _load32(head + key_offset) != _load32(records + i * 4):
                        heads[unsafe_offset=i] = 0
                        no_match[unsafe_offset=misses] = Int32(i)
                        misses += 1
            else:
                for c in range(candidates):
                    var i = Int(compare[unsafe_offset=c])
                    var head = Int(heads[unsafe_offset=i])
                    if not _same_keys(
                        kinds,
                        key_offsets,
                        widths,
                        fields,
                        key_bytes,
                        plain,
                        head + key_offset,
                        records + i * key_bytes,
                    ):
                        heads[unsafe_offset=i] = 0
                        no_match[unsafe_offset=misses] = Int32(i)
                        misses += 1
            for m in range(misses):
                var i = Int(no_match[unsafe_offset=m])
                var base = region[unsafe_offset=i]
                offsets[unsafe_offset=i] = base + (
                    (offsets[unsafe_offset=i] - base + 1)
                    & bitmasks[unsafe_offset=i]
                )
                todo[unsafe_offset=m] = Int32(i)
            pending = misses

    def matched(self, i: Int) -> Bool:
        """Whether chunk row i found a chain."""
        return self.heads[i] != 0

    def next(mut self) -> Int:
        """The next at most STANDARD_VECTOR_SIZE output rows of the probed
        chunk, or 0 once it is done."""
        if self.finished:
            return 0
        var heads = self.heads.unsafe_ptr()
        var out_probe = self.probe_rows.unsafe_ptr()
        var out_build = self.build_rows.unsafe_ptr()
        var out_row = self.row_pointers.unsafe_ptr()
        var null_row = UInt64(self.table.null_row_address())
        var dict_emission = len(self.table.data[].aux_next) > 0
        var aux = self.table.data[].aux_next.unsafe_ptr()
        var slot_rows = self.table.data[].slot_rows.unsafe_ptr()
        var slots = self.table.data[].slots.unsafe_ptr()
        var index_offset = self.table.layout.index_offset
        var next_offset = self.table.layout.pointer_offset
        var found_offset = self.table.layout.found_offset
        var out = 0
        if (
            self.how == JOIN_SEMI
            or self.how == JOIN_ANTI
            or self.how == JOIN_MARK
        ):
            # Semi, anti and mark joins: one pass, since a row's output depends
            # only on whether it matched.
            var keep = self.how == JOIN_SEMI
            var unknown = self.table.has_null
            var valid = self.valid.unsafe_ptr()
            var marks = self.marks.unsafe_ptr()
            for i in range(self.count):
                var hit = heads[unsafe_offset=i] != 0
                if self.how == JOIN_MARK:
                    out_probe[unsafe_offset=out] = Int32(i)
                    out_build[unsafe_offset=out] = -1
                    marks[unsafe_offset=out] = UInt8(1) if hit else (
                        UInt8(2) if (
                            unknown or valid[unsafe_offset=i] == 0
                        ) else UInt8(0)
                    )
                    out += 1
                elif hit == keep:
                    out_probe[unsafe_offset=out] = Int32(i)
                    out_build[unsafe_offset=out] = -1
                    out += 1
            self.finished = True
            return out
        if self.how == JOIN_RIGHT_SEMI or self.how == JOIN_RIGHT_ANTI:
            # Right semi and anti joins: flag every row of each matched chain; a
            # chain whose head is flagged was flagged whole before.
            for i in range(self.count):
                var p = heads[unsafe_offset=i]
                if p == 0:
                    continue
                if _load8(Int(p) + found_offset) != 0:
                    continue
                while p != 0:
                    _store8(Int(p) + found_offset, 1)
                    if dict_emission:
                        p = aux[
                            unsafe_offset=Int(_load64(Int(p) + next_offset))
                        ]
                    else:
                        p = _load64(Int(p) + next_offset)
            self.finished = True
            return 0
        if self.table.perfect:
            # A perfect hash probe: each key at most one build row,
            # `heads` holding it plus one.
            var outer_rows = self.how == JOIN_LEFT or self.how == JOIN_FULL
            for i in range(self.count):
                var h = heads[unsafe_offset=i]
                if h != 0 or outer_rows:
                    out_probe[unsafe_offset=out] = Int32(i)
                    var slot = Int(h) - 1
                    # The slot's build row: also its dictionary entry.
                    out_build[unsafe_offset=out] = -1 if h == 0 else Int32(
                        slots[unsafe_offset=slot]
                    )
                    if h != 0:
                        out_row[unsafe_offset=out] = slot_rows[
                            unsafe_offset=Int(h) - 1
                        ]
                    else:
                        out_row[unsafe_offset=out] = null_row
                    out += 1
            self.finished = True
            return out
        # Inner and left joins, one probe row's chain at a time.
        var outer = self.how == JOIN_LEFT or self.how == JOIN_FULL
        var flag = self.propagate
        while self.row < self.count:
            if out == STANDARD_VECTOR_SIZE:
                return out
            var i = self.row
            if not self.started:
                self.started = True
                self.cursor = heads[unsafe_offset=i]
                if self.cursor == 0:
                    if outer:
                        out_probe[unsafe_offset=out] = Int32(i)
                        out_build[unsafe_offset=out] = Int32(
                            self.table.rows_stored
                        ) if dict_emission else -1
                        out_row[unsafe_offset=out] = null_row
                        out += 1
                    self.row += 1
                    self.started = False
                    continue
            var p = self.cursor
            while p != 0 and out < STANDARD_VECTOR_SIZE:
                # The build row number, in rows that keep one.
                var b = Int32(-1)
                if index_offset >= 0:
                    b = Int32(_load32(Int(p) + index_offset))
                out_probe[unsafe_offset=out] = Int32(i)
                out_row[unsafe_offset=out] = p
                if flag:
                    _store8(Int(p) + found_offset, 1)
                out += 1
                if dict_emission:
                    # The row number sits where the next pointer was, the next
                    # pointer in `aux_next`.
                    var number = _load64(Int(p) + next_offset)
                    out_build[unsafe_offset=out - 1] = Int32(number)
                    p = aux[unsafe_offset=Int(number)]
                else:
                    out_build[unsafe_offset=out - 1] = b
                    p = _load64(Int(p) + next_offset)
            self.cursor = p
            if p == 0:
                self.row += 1
                self.started = False
        self.finished = True
        return out


struct _ProbeJob(Job):
    """One thread's probe of rows [start, end), chunk by chunk, keeping the
    pairs it emits (absolute probe rows; build rows, -1 for none)."""

    var table: JoinHashTable
    var keys: List[Series]
    var how: Int
    var start: Int
    var end: Int
    var both: Bool
    var probe: List[Int]
    var build: List[Int]

    def __init__(
        out self,
        table: JoinHashTable,
        keys: List[Series],
        how: Int,
        start: Int,
        end: Int,
    ):
        self.table = table.copy()
        self.keys = keys.copy()
        self.how = how
        self.start = start
        self.end = end
        self.both = not (
            how == JOIN_SEMI or how == JOIN_ANTI or how == JOIN_MARK
        )
        self.probe = List[Int]()
        self.build = List[Int]()

    def run(mut self) raises:
        var scan = ScanStructure(self.table, self.how)
        var used = 0
        var low = self.start
        while low < self.end:
            var count = min(STANDARD_VECTOR_SIZE, self.end - low)
            scan.probe(self.keys, low, count)
            while True:
                var n = scan.next()
                if n == 0:
                    break
                if len(self.probe) < used + n:
                    # Doubled, written through pointers, cut to size last.
                    var size = max(used + n, 2 * len(self.probe))
                    self.probe = _grown(self.probe, used, size)
                    if self.both:
                        self.build = _grown(self.build, used, size)
                var rows = scan.probe_rows.unsafe_ptr()
                var to = self.probe.unsafe_ptr()
                for k in range(n):
                    to[unsafe_offset=used + k] = low + Int(
                        rows[unsafe_offset=k]
                    )
                if self.both:
                    var builds = scan.build_rows.unsafe_ptr()
                    var into = self.build.unsafe_ptr()
                    for k in range(n):
                        into[unsafe_offset=used + k] = Int(
                            builds[unsafe_offset=k]
                        )
                used += n
            low += count
        self.probe.shrink(used)
        if self.both:
            self.build.shrink(used)


def _grown(rows: List[Int], used: Int, size: Int) -> List[Int]:
    """`size` rows, the first `used` of them `rows`'."""
    var grown = List[Int](unsafe_uninit_length=size)
    if used > 0:
        unsafe_memcpy(
            dest=grown.unsafe_ptr(), src=rows.unsafe_ptr(), count=used
        )
    return grown^


struct _AppendJob(Job):
    """Copy one probe job's pairs to their place in the joined lists."""

    var source: Int
    var target: Int
    var at: Int
    var count: Int

    def __init__(out self, source: Int, target: Int, at: Int, count: Int):
        self.source = source
        self.target = target
        self.at = at
        self.count = count

    def run(mut self) raises:
        if self.count == 0:
            return
        unsafe_memcpy(
            dest=Pointer[Int, MutAnyOrigin](
                unsafe_from_address=self.target
            ).unsafe_offset(self.at),
            src=Pointer[Int, MutAnyOrigin](unsafe_from_address=self.source),
            count=self.count,
        )


def probe_pairs(
    table: JoinHashTable,
    keys: List[Series],
    how: Int,
    workers: Int,
) raises -> Tuple[List[Int], List[Int]]:
    """Every row `keys` (the probe side's key columns) joins against
    `table` with physical join type `how`: the probe rows and the build
    rows (-1 for none), in probe order, on `workers` threads (0: by the
    row count). A semi, anti or mark join gives probe rows only. A right
    or full join then appends the build rows no probe matched (probe row
    -1), and a right-semi or right-anti join gives its build rows only
    (`scan_full_outer`)."""
    var unchunked = List[Series](capacity=len(keys))
    for key in keys:
        unchunked.append(key.rechunk() if key.is_chunked() else key.copy())
    var n = len(unchunked[0]) if len(unchunked) > 0 else 0
    var lanes = workers if workers > 0 else worker_count(n)
    var bounds = partitions(n, lanes, STANDARD_VECTOR_SIZE)
    var jobs = List[_ProbeJob](capacity=lanes)
    for w in range(lanes):
        if bounds[w] < bounds[w + 1]:
            jobs.append(
                _ProbeJob(table, unchunked, how, bounds[w], bounds[w + 1])
            )
    if len(jobs) == 1:
        jobs[0].run()
    elif len(jobs) > 1:
        run_jobs(jobs)
    var probe = List[Int]()
    var build = List[Int]()
    if len(jobs) == 1:
        swap(probe, jobs[0].probe)
        swap(build, jobs[0].build)
    elif len(jobs) > 1:
        var total = 0
        for w in range(len(jobs)):
            total += len(jobs[w].probe)
        probe = huge_uninit[Int](total)
        var both = jobs[0].both
        if both:
            build = huge_uninit[Int](total)
        var copies = List[_AppendJob](capacity=2 * len(jobs))
        var at = 0
        for w in range(len(jobs)):
            ref job = jobs[w]
            copies.append(
                _AppendJob(
                    Int(job.probe.unsafe_ptr()),
                    Int(probe.unsafe_ptr()),
                    at,
                    len(job.probe),
                )
            )
            if both:
                copies.append(
                    _AppendJob(
                        Int(job.build.unsafe_ptr()),
                        Int(build.unsafe_ptr()),
                        at,
                        len(job.build),
                    )
                )
            at += len(job.probe)
        run_jobs(copies)
    _ = jobs^
    if how == JOIN_RIGHT or how == JOIN_FULL:
        for row in table.full_outer_rows(0):
            build.append(row)
        while len(probe) < len(build):
            probe.append(-1)
    elif how == JOIN_RIGHT_SEMI or how == JOIN_RIGHT_ANTI:
        var want = UInt8(1) if how == JOIN_RIGHT_SEMI else UInt8(0)
        return (List[Int](), table.full_outer_rows(want))
    return (probe^, build^)
