"""Recoding values: `replace` and `replace_strict` (#789).

Recoding matches each input against the old values, with null matching
null, then picks the new value or the fallback per row. Here the work is
two passes over row ranges: each worker probes a table built once from the
old values (a direct table for integers in a small span, otherwise open
addressing; every NaN is one value and -0.0 equals 0.0) and records each
match's old position. The result is written from those positions: the new
value, else the input (`replace`), the default, or null. String results are
views that share the new values'
and the fallback's bytes. Over a categorical, the whole expression runs
once per dictionary value (`_on_dictionary` in
execution), so a recode costs a lookup per distinct value and a gather of
the codes.
"""
from std.collections import Optional
from std.memory import ArcPointer, Pointer, bitcast

from .bool_column import BoolColumn
from .cast import cast_series
from .categorical import decode
from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .expr import (
    Node,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_STRING,
    LIT_NULL,
    REPLACE_STRICT,
    replace_literals,
)
from dataframe.aggregation.reducer import float_key
from .column import _validity_at
from .expr_kernels import choose
from .gather import take_parallel
from .hashing import encode_rows
from .join_hash import _perfect_slots
from .parallel import Job, partitions, run_jobs, worker_count
from .partition import _hash_bytes, _mix
from .series import Series
from .string_bytes import _Bytes, _same_bytes
from .string_column import StringBuilder, StringColumn
from .string_view import (
    STRING_VIEW_INLINE_BYTES,
    StringView,
    StringViewStorage,
    _external_view,
    _inline_view,
)


def _literal_dtype(node: Node) raises -> DataType:
    if len(node.dtypes) > 0:
        if not node.dtypes[0]:
            raise Error("Unknown literal dtype: " + node.text)
        return node.dtypes[0].value()
    if node.op == LIT_INT:
        return DataType.INT64
    return DataType.FLOAT64


def values_dtype(values: List[Node]) raises -> Optional[DataType]:
    """The dtype a list of literals has as one column: strings, Bools or
    numbers. Integers keep the type
    they share, else Int64; any float makes the list Float64 (Float32 when
    every value is one). Nulls take the others' type, and a list of nulls
    its first null's. None for no values, or for anything but literals."""
    var strings = 0
    var bools = 0
    var numbers = 0
    var floats = 0
    var float32 = 0
    var shared = Optional[DataType]()
    var mixed = False
    var nulls = Optional[DataType]()
    for node in values:
        if node.op == LIT_NULL:
            if not nulls:
                nulls = _literal_dtype(node)
            continue
        if node.op == LIT_STRING:
            strings += 1
        elif node.op == LIT_BOOL:
            bools += 1
        elif node.op == LIT_INT or node.op == LIT_FLOAT:
            numbers += 1
            var dtype = _literal_dtype(node)
            if node.op == LIT_FLOAT:
                floats += 1
                if dtype == DataType.FLOAT32:
                    float32 += 1
            if shared and shared.value() != dtype:
                mixed = True
            shared = dtype
        else:
            return None
    var kinds = Int(strings > 0) + Int(bools > 0) + Int(numbers > 0)
    if kinds > 1:
        raise Error(
            "replace values must share one type: found"
            + (" strings" if strings > 0 else "")
            + (" Bools" if bools > 0 else "")
            + (" numbers" if numbers > 0 else "")
        )
    if strings > 0:
        return DataType.STRING
    if bools > 0:
        return DataType.BOOL
    if numbers == 0:
        return nulls
    if floats > 0:
        if float32 == numbers:
            return DataType.FLOAT32
        return DataType.FLOAT64
    if mixed:
        if not shared.value().is_integer():
            raise Error(
                "replace values must share one type: found "
                + shared.value().name()
                + " among other numbers"
            )
        return DataType.INT64
    return shared


def replace_supertype(first: DataType, second: DataType) raises -> DataType:
    """The dtype that holds values of both: strings hold anything,
    integers widen (signed when either is), and floats hold integers."""
    if first == second:
        return first
    if first == DataType.STRING or second == DataType.STRING:
        return DataType.STRING
    var a = DataType.INT64 if first == DataType.BOOL else first
    var b = DataType.INT64 if second == DataType.BOOL else second
    if a == b:
        return a
    if (a.is_float() or b.is_float()) and a.is_numeric() and b.is_numeric():
        # Float32 holds integers of up to 16 bits exactly.
        if (a == DataType.FLOAT32 or a.bit_width() <= 16) and (
            b == DataType.FLOAT32 or b.bit_width() <= 16
        ):
            return DataType.FLOAT32
        return DataType.FLOAT64
    if a.is_integer() and b.is_integer():
        var a_signed = a.storage().value().is_signed()
        var b_signed = b.storage().value().is_signed()
        var bits = max(a.bit_width(), b.bit_width())
        if a_signed != b_signed:
            var unsigned_bits = b.bit_width() if a_signed else a.bit_width()
            var signed_bits = a.bit_width() if a_signed else b.bit_width()
            if unsigned_bits >= 64:
                return DataType.FLOAT64
            bits = max(signed_bits, unsigned_bits * 2)
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            comptime if D.is_integral():
                if DataType.of(D).bit_width() == bits and D.is_signed() == (
                    a_signed or b_signed
                ):
                    return DataType.of(D)
    raise Error(
        "replace_strict: new values of type "
        + first.name()
        + " and a default of type "
        + second.name()
        + " have no common type; set return_dtype"
    )


def _literal_column(values: List[Node], dtype: DataType) raises -> Series:
    """The literals as one column of `dtype` (see `values_dtype`), nulls
    where they are null."""
    var n = len(values)
    if dtype == DataType.STRING:
        var builder = StringBuilder(n)
        for node in values:
            if node.op == LIT_NULL:
                builder.append_null()
            else:
                builder.append(node.text)
        return Series("", builder^.finish())
    var valid = List[Bool](capacity=n)
    for node in values:
        valid.append(node.op != LIT_NULL)
    if dtype == DataType.BOOL:
        var bools = List[Bool](capacity=n)
        for node in values:
            bools.append(node.op == LIT_BOOL and node.integer != 0)
        return Series("", Column[Bool](bools^, valid))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if dtype.physical() == DataType.of(D):
            var numbers = List[Scalar[D]](capacity=n)
            for node in values:
                comptime if D.is_floating_point():
                    if node.op == LIT_FLOAT:
                        numbers.append(node.floating.cast[D]())
                    else:
                        numbers.append(Float64(node.integer).cast[D]())
                else:
                    numbers.append(node.integer.cast[D]())
            var column = Series("", Column[Scalar[D]](numbers^, valid))
            if column.dtype() != dtype:
                return column.with_dtype(dtype)
            return column^
    for node in values:
        if node.op != LIT_NULL:
            raise Error("replace does not support literals of " + dtype.name())
    return Series.full_null("", dtype, n)


def _as(values: Series, dtype: DataType, what: String) raises -> Series:
    """`values` cast strictly to `dtype`, naming which values failed."""
    if values.dtype() == dtype:
        return values.copy()
    try:
        return cast_series(values, dtype, True, 0, List[Bool]())
    except e:
        raise Error(
            what + " cannot be cast to " + dtype.name() + ": " + String(e)
        )


def _validity(values: Series) -> List[Bool]:
    """Whether each row of a contiguous column is valid."""
    var n = len(values)
    var out = List[Bool](length=n, fill=True)
    if values.null_count() == 0:
        return out^
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if values._data.isa[Column[Scalar[D]]]():
            ref column = values._data[Column[Scalar[D]]]
            for i in range(n):
                out[i] = column._valid(i)
            return out^
    if values._data.isa[Column[Int128]]():
        ref column = values._data[Column[Int128]]
        for i in range(n):
            out[i] = column._valid(i)
    elif values._data.isa[BoolColumn]():
        ref column = values._data[BoolColumn]
        for i in range(n):
            out[i] = column._valid(i)
    elif values._data.isa[StringColumn]():
        ref column = values._data[StringColumn]
        for i in range(n):
            out[i] = column._valid(i)
    return out^


# What a key's word holds: the value itself (integers of up to 64 bits,
# Bools, and floats as `float_key`), or a hash of a string or a 128-bit
# integer, whose candidates are then compared in full.
comptime _EXACT = 0
comptime _TEXT = 1
comptime _WIDE = 2


@always_inline
def _signed_word[D: DType](value: Scalar[D]) -> UInt64:
    """An integer's word: signed values sign-extended, unsigned ones
    zero-extended, so equal values of one type give equal words."""
    comptime if D.is_unsigned():
        return UInt64(value)
    else:
        return bitcast[DType.uint64](Int64(value))


@always_inline
def _wide_word(value: Int128) -> UInt64:
    return _mix(UInt64(value & 0xFFFFFFFFFFFFFFFF) ^ _mix(UInt64(value >> 64)))


struct _Lookup(Copyable, Movable):
    """The old values' positions by key, built once and probed by every
    worker. Integers within a small span index a table directly; anything
    else goes to an open-addressing table of key words."""

    var kind: Int
    var old: Series
    var null_at: Int
    # Perfect hash: the old position of `key - base`, -1 for none.
    var base: Int64
    var direct: List[Int32]
    # Open addressing: per slot the key word and its old position.
    var words: List[UInt64]
    var slots: List[Int32]
    var mask: Int

    def __init__(out self, old: Series) raises:
        self.old = old.copy()
        self.kind = _EXACT
        self.null_at = -1
        self.base = 0
        self.direct = List[Int32]()
        self.words = List[UInt64]()
        self.slots = List[Int32]()
        self.mask = 0
        var k = len(old)
        var valid = _validity(old)
        var keys = List[UInt64](capacity=k)
        var positions = List[Int](capacity=k)
        var integers = False
        var narrow = True
        for j in range(k):
            if not valid[j]:
                if self.null_at < 0:
                    self.null_at = j
                continue
            positions.append(j)
            keys.append(0)
        if old._data.isa[StringColumn]():
            self.kind = _TEXT
            var text = _Bytes(old._data[StringColumn])
            for p in range(len(positions)):
                keys[p] = _mix(_hash_bytes(text.get(positions[p])))
        elif old._data.isa[Column[Int128]]():
            self.kind = _WIDE
            ref column = old._data[Column[Int128]]
            for p in range(len(positions)):
                keys[p] = _wide_word(column._get(positions[p]))
        elif old._data.isa[BoolColumn]():
            ref column = old._data[BoolColumn]
            for p in range(len(positions)):
                keys[p] = UInt64(Int(column._get(positions[p])))
        else:
            comptime for d in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[d]
                if old._data.isa[Column[Scalar[D]]]():
                    ref column = old._data[Column[Scalar[D]]]
                    for p in range(len(positions)):
                        var value = column._get(positions[p])
                        comptime if D.is_floating_point():
                            keys[p] = float_key(Float64(value))
                        else:
                            keys[p] = _signed_word[D](value)
                    comptime if not D.is_floating_point():
                        integers = True
                        narrow = D != DType.uint64
        if integers and narrow and len(positions) > 0:
            var words = List[Int64](capacity=len(positions))
            for key in keys:
                words.append(bitcast[DType.int64](key))
            var perfect = _perfect_slots(Series("", Column[Int64](words^)))
            if perfect:
                var made = perfect.take()
                self.base = made[0]
                self.direct = made[1].copy()
                for at in range(len(self.direct)):
                    var slot = Int(self.direct[at])
                    if slot >= 0:
                        self.direct[at] = Int32(positions[slot])
                return
        var capacity = 16
        while capacity < 2 * len(positions):
            capacity *= 2
        self.mask = capacity - 1
        self.words = List[UInt64](length=capacity, fill=0)
        self.slots = List[Int32](length=capacity, fill=-1)
        for p in range(len(positions)):
            var slot = Int(_mix(keys[p])) & self.mask
            while self.slots[slot] >= 0:
                slot = (slot + 1) & self.mask
            self.words[slot] = keys[p]
            self.slots[slot] = Int32(positions[p])


struct _ProbeJob(Job):
    """Rows [start, end): each row's old position, or -1, into `rows`;
    how many matched, and the first valid row that did not."""

    var values: Series
    var lookup: ArcPointer[_Lookup]
    var rows: Int
    var start: Int
    var end: Int
    var hits: Int
    var missed: Int

    def __init__(
        out self,
        values: Series,
        lookup: ArcPointer[_Lookup],
        rows: Int,
        start: Int,
        end: Int,
    ):
        self.values = values.copy()
        self.lookup = lookup.copy()
        self.rows = rows
        self.start = start
        self.end = end
        self.hits = 0
        self.missed = -1

    @always_inline
    def _hashed(self, key: UInt64) -> Int:
        """The slot holding `key`'s word, or -1."""
        ref table = self.lookup[]
        var words = table.words.unsafe_ptr()
        var slots = table.slots.unsafe_ptr()
        var slot = Int(_mix(key)) & table.mask
        while True:
            var at = Int(slots[unsafe_offset=slot])
            if at < 0:
                return -1
            if words[unsafe_offset=slot] == key:
                return slot
            slot = (slot + 1) & table.mask

    def run(mut self) raises:
        ref table = self.lookup[]
        var out = Pointer[Int32, MutAnyOrigin](unsafe_from_address=self.rows)
        var bits = _bitmap(self.values)
        var null_at = Int32(table.null_at)
        var hits = 0
        var missed = -1
        if self.values._data.isa[StringColumn]():
            var text = _Bytes(self.values._data[StringColumn])
            var old = _Bytes(table.old._data[StringColumn])
            var slots = table.slots.unsafe_ptr()
            var words = table.words.unsafe_ptr()
            for i in range(self.start, self.end):
                var found = Int32(-1)
                if bits[0] != 0 and not _valid_at(bits, i):
                    found = null_at
                else:
                    var row = text.get(i)
                    var key = _mix(_hash_bytes(row))
                    var slot = Int(_mix(key)) & table.mask
                    while slots[unsafe_offset=slot] >= 0:
                        var at = slots[unsafe_offset=slot]
                        if words[unsafe_offset=slot] == key and _same_bytes(
                            old.get(Int(at)), row
                        ):
                            found = at
                            break
                        slot = (slot + 1) & table.mask
                    if found < 0 and missed < 0:
                        missed = i
                out[unsafe_offset=i] = found
                hits += Int(found >= 0)
            self.hits = hits
            self.missed = missed
            return
        if self.values._data.isa[Column[Int128]]():
            ref column = self.values._data[Column[Int128]]
            ref old = table.old._data[Column[Int128]]
            for i in range(self.start, self.end):
                var found = Int32(-1)
                if bits[0] != 0 and not _valid_at(bits, i):
                    found = null_at
                else:
                    var value = column._get(i)
                    var key = _wide_word(value)
                    var slot = Int(_mix(key)) & table.mask
                    while table.slots[slot] >= 0:
                        var at = table.slots[slot]
                        if table.words[slot] == key and old._get(Int(at)) == (
                            value
                        ):
                            found = at
                            break
                        slot = (slot + 1) & table.mask
                    if found < 0 and missed < 0:
                        missed = i
                out[unsafe_offset=i] = found
                hits += Int(found >= 0)
            self.hits = hits
            self.missed = missed
            return
        if self.values._data.isa[BoolColumn]():
            ref column = self.values._data[BoolColumn]
            for i in range(self.start, self.end):
                var found = Int32(-1)
                if bits[0] != 0 and not _valid_at(bits, i):
                    found = null_at
                else:
                    var slot = self._hashed(UInt64(Int(column._get(i))))
                    if slot >= 0:
                        found = table.slots[slot]
                    elif missed < 0:
                        missed = i
                out[unsafe_offset=i] = found
                hits += Int(found >= 0)
            self.hits = hits
            self.missed = missed
            return
        comptime for d in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[d]
            if self.values._data.isa[Column[Scalar[D]]]():
                var data = self.values._data[Column[Scalar[D]]]._ptr()
                var direct = table.direct.unsafe_ptr()
                var span = UInt64(len(table.direct))
                var base = UInt64(bitcast[DType.uint64](table.base))
                var slots = table.slots.unsafe_ptr()
                for i in range(self.start, self.end):
                    var found = Int32(-1)
                    if bits[0] != 0 and not _valid_at(bits, i):
                        found = null_at
                    else:
                        var value = data[unsafe_offset=i]
                        var key: UInt64
                        comptime if D.is_floating_point():
                            key = float_key(Float64(value))
                        else:
                            key = _signed_word[D](value)
                        if span > 0:
                            var offset = key - base
                            if offset < span:
                                found = direct[unsafe_offset=Int(offset)]
                        else:
                            var slot = self._hashed(key)
                            if slot >= 0:
                                found = slots[unsafe_offset=slot]
                        if found < 0 and missed < 0:
                            missed = i
                    out[unsafe_offset=i] = found
                    hits += Int(found >= 0)
                self.hits = hits
                self.missed = missed
                return
        raise Error("replace does not support " + self.values.dtype().name())


@always_inline
def _valid_at(bits: Tuple[Int, Int], i: Int) -> Bool:
    return _validity_at(
        Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=bits[0]),
        bits[1] + i,
    )


def _bitmap(values: Series) -> Tuple[Int, Int]:
    """A contiguous column's validity bitmap address and row 0's bit, or
    (0, 0) when every row is valid."""
    if values.null_count() == 0:
        return (0, 0)
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if values._data.isa[Column[Scalar[D]]]():
            ref column = values._data[Column[Scalar[D]]]
            return (Int(column._bits[].unsafe_ptr()), column._offset)
    if values._data.isa[Column[Int128]]():
        ref column = values._data[Column[Int128]]
        return (Int(column._bits[].unsafe_ptr()), column._offset)
    if values._data.isa[BoolColumn]():
        ref column = values._data[BoolColumn]
        return (Int(column._bits[].unsafe_ptr()), column._offset)
    ref column = values._data[StringColumn]
    return (Int(column._bits[].unsafe_ptr()), column._offset)


struct _WriteJob[D: DType](Job):
    """Rows [start, end) of a fixed-width result: a matched row's new
    value, else the fallback's (one per row, or one for all), else null. Bits are written a byte at a time, so ranges start at multiples
    of 8 rows."""

    var rows: Int
    var new: Series
    var fallback: Series
    var single: Bool
    var values: Int
    var bits: Int
    var start: Int
    var end: Int
    var nulls: Int

    def __init__(
        out self,
        rows: Int,
        new: Series,
        fallback: Series,
        single: Bool,
        values: Int,
        bits: Int,
        start: Int,
        end: Int,
    ):
        self.rows = rows
        self.new = new.copy()
        self.fallback = fallback.copy()
        self.single = single
        self.values = values
        self.bits = bits
        self.start = start
        self.end = end
        self.nulls = 0

    def run(mut self) raises:
        var rows = Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=self.rows)
        var out = Pointer[Scalar[Self.D], MutAnyOrigin](
            unsafe_from_address=self.values
        )
        var bits = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.bits)
        var new = self.new._data[Column[Scalar[Self.D]]]._ptr()
        var new_bits = _bitmap(self.new)
        var count = len(self.fallback)
        var address = 0
        var fallback_bits = (0, 0)
        if count > 0:
            address = Int(self.fallback._data[Column[Scalar[Self.D]]]._ptr())
            fallback_bits = _bitmap(self.fallback)
        var from_rows = Pointer[Scalar[Self.D], ImmutAnyOrigin](
            unsafe_from_address=address
        )
        var fallback = Scalar[Self.D](0)
        var fallback_ok = False
        if count == 1:
            fallback = from_rows[unsafe_offset=0]
            fallback_ok = fallback_bits[0] == 0 or _valid_at(fallback_bits, 0)
        var nulls = 0
        var i = self.start
        while i < self.end:
            var stop = min(i + 8, self.end)
            var byte = UInt8(0)
            for row in range(i, stop):
                var at = Int(rows[unsafe_offset=row])
                var value = Scalar[Self.D](0)
                var ok = False
                if at >= 0:
                    var j = 0 if self.single else at
                    value = new[unsafe_offset=j]
                    ok = new_bits[0] == 0 or _valid_at(new_bits, j)
                elif count > 1:
                    value = from_rows[unsafe_offset=row]
                    ok = fallback_bits[0] == 0 or _valid_at(fallback_bits, row)
                elif count == 1:
                    value = fallback
                    ok = fallback_ok
                out[unsafe_offset=row] = value
                byte |= UInt8(Int(ok)) << UInt8(row - i)
                nulls += Int(not ok)
            bits[unsafe_offset=i // 8] = byte
            i = stop
        self.nulls = nulls


struct _ViewJob(Job):
    """Rows [start, end) of a string result as views, sharing the bytes of
    the new values and of the fallback instead of copying them: a matched
    row takes its new value's view, any other the fallback's, whose long
    values point into the fallback's buffers, listed after the new
    values' (`shift`)."""

    var rows: Int
    var new: List[StringView]
    var new_valid: List[Bool]
    var fallback: StringColumn
    var shift: UInt32
    var single: Bool
    var views: Int
    var bits: Int
    var start: Int
    var end: Int
    var nulls: Int

    def __init__(
        out self,
        rows: Int,
        new: List[StringView],
        new_valid: List[Bool],
        fallback: StringColumn,
        shift: UInt32,
        single: Bool,
        views: Int,
        bits: Int,
        start: Int,
        end: Int,
    ):
        self.rows = rows
        self.new = new.copy()
        self.new_valid = new_valid.copy()
        self.fallback = fallback.copy()
        self.shift = shift
        self.single = single
        self.views = views
        self.bits = bits
        self.start = start
        self.end = end
        self.nulls = 0

    def run(mut self) raises:
        var rows = Pointer[Int32, ImmutAnyOrigin](unsafe_from_address=self.rows)
        var out = Pointer[StringView, MutAnyOrigin](
            unsafe_from_address=self.views
        )
        var bits = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.bits)
        var count = len(self.fallback)
        var text = _Bytes(self.fallback)
        var nulls = 0
        var i = self.start
        while i < self.end:
            var stop = min(i + 8, self.end)
            var byte = UInt8(0)
            for row in range(i, stop):
                var at = Int(rows[unsafe_offset=row])
                var ok = False
                if at >= 0:
                    var j = 0 if self.single else at
                    out[unsafe_offset=row] = self.new[j].copy()
                    ok = self.new_valid[j]
                elif count > 0:
                    var source = row if count > 1 else 0
                    ok = self.fallback._valid(source)
                    out[unsafe_offset=row] = self._fallback_view(
                        text, source
                    ) if ok else StringView(0, 0, 0, 0)
                else:
                    out[unsafe_offset=row] = StringView(0, 0, 0, 0)
                byte |= UInt8(Int(ok)) << UInt8(row - i)
                nulls += Int(not ok)
            bits[unsafe_offset=i // 8] = byte
            i = stop
        self.nulls = nulls

    @always_inline
    def _fallback_view(self, text: _Bytes, row: Int) -> StringView:
        if text.is_views():
            var view = Pointer[StringView, ImmutAnyOrigin](
                unsafe_from_address=text.views
            )[unsafe_offset=row].copy()
            if not view.is_inline():
                view.buffer_index += self.shift
            return view^
        var offsets = Pointer[Int64, ImmutAnyOrigin](
            unsafe_from_address=text.offsets
        )
        var start = Int(offsets[unsafe_offset=row])
        var bytes = Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=text.data + start
            ),
            length=Int(offsets[unsafe_offset=row + 1]) - start,
        )
        if len(bytes) <= STRING_VIEW_INLINE_BYTES:
            return _inline_view(bytes)
        return _external_view(bytes, self.shift, UInt32(start))


def _write_views(
    rows: List[Int32],
    new: Series,
    fallback: Optional[Series],
    single: Bool,
    lanes: Int,
) raises -> Optional[Series]:
    """A string result as views over the new values' and the fallback's
    bytes; None when the fallback's bytes are past what a view addresses."""
    var n = len(rows)
    # The new values as views (a few rows): their own buffers come first.
    var identity = List[Int](capacity=len(new))
    for j in range(len(new)):
        identity.append(j)
    var as_views = new._data[StringColumn].take(identity)
    if not as_views._is_view_storage():
        return None
    var storage = as_views._view_storage_unchecked()
    var new_views = List[StringView](capacity=len(new))
    var new_valid = List[Bool](capacity=len(new))
    for j in range(len(new)):
        new_views.append(storage._view_unchecked(j))
        new_valid.append(as_views._valid(j))
    var buffers = storage._buffers[].copy()
    var shift = UInt32(len(buffers))
    var source = StringColumn(List[String]())
    if fallback:
        source = fallback.value()._data[StringColumn].copy()
        if source._is_view_storage():
            for buffer in source._view_storage_unchecked()._buffers[]:
                buffers.append(buffer.copy())
        else:
            if len(source._bytes[]) > 4_294_967_295:
                return None
            buffers.append(source._bytes.copy())
    var total = 0
    for buffer in buffers:
        total += len(buffer[])
    var views = List[StringView](unsafe_uninit_length=n)
    var bits = List[UInt8](length=(n + 7) // 8, fill=0)
    var bounds = partitions(n, lanes, 8)
    var jobs = List[_ViewJob](capacity=lanes)
    for w in range(lanes):
        jobs.append(
            _ViewJob(
                Int(rows.unsafe_ptr()),
                new_views,
                new_valid,
                source,
                shift,
                single,
                Int(views.unsafe_ptr()),
                Int(bits.unsafe_ptr()),
                bounds[w],
                bounds[w + 1],
            )
        )
    if lanes == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    var nulls = 0
    for w in range(len(jobs)):
        nulls += jobs[w].nulls
    if nulls == 0:
        bits = List[UInt8]()
    _ = rows
    return Series(
        "",
        StringColumn(StringViewStorage(views^, buffers^, bits^, n, total)),
    )


def _write_numeric[
    D: DType
](
    rows: List[Int32],
    new: Series,
    fallback: Optional[Series],
    single: Bool,
    lanes: Int,
) raises -> Series:
    var n = len(rows)
    var values = List[Scalar[D]](unsafe_uninit_length=n)
    var bits = List[UInt8](length=(n + 7) // 8, fill=0)
    var given = fallback.value().copy() if fallback else Series(
        "", Column[Scalar[D]](List[Scalar[D]]())
    )
    var bounds = partitions(n, lanes, 8)
    var jobs = List[_WriteJob[D]](capacity=lanes)
    for w in range(lanes):
        jobs.append(
            _WriteJob[D](
                Int(rows.unsafe_ptr()),
                new,
                given,
                single,
                Int(values.unsafe_ptr()),
                Int(bits.unsafe_ptr()),
                bounds[w],
                bounds[w + 1],
            )
        )
    if lanes == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    var nulls = 0
    for w in range(len(jobs)):
        nulls += jobs[w].nulls
    if nulls == 0:
        bits = List[UInt8]()
    _ = rows
    return Series("", Column[Scalar[D]](values=values^, bits=bits^))


def replace_values(
    node: Node,
    input: Series,
    input_dtype: DataType,
    result_dtype: DataType,
    default: Optional[Series],
    workers: Int,
) raises -> Series:
    """A REPLACE or REPLACE_STRICT node over `input`. `default` is the
    default's values, one row or one per input row. `workers` is 0 to
    choose by the row count, or 1 inside a batch another worker runs."""
    var strict = node.op == REPLACE_STRICT
    # Inside an expression a categorical is carried as its strings (#106).
    var source = DataType.STRING if input_dtype.is_categorical() else (
        input_dtype
    )
    var target = DataType.STRING if result_dtype.is_categorical() else (
        result_dtype
    )
    var values = input.rechunk() if input.is_chunked() else input.copy()
    if values.dtype().is_categorical():
        values = decode(values)
    elif values.dtype() != source:
        values = values.with_dtype(source)
    var n = len(values)
    var old_nodes = replace_literals(node.text)
    var new_nodes = replace_literals(node.text2)
    var old_type = values_dtype(old_nodes)
    var new_type = values_dtype(new_nodes)
    if (
        strict
        and old_type
        and old_type.value() == DataType.STRING
        and (source.is_numeric() or source == DataType.BOOL)
    ):
        raise Error(
            "cannot use values of type string to replace values in a"
            " column of type " + source.name()
        )
    var old = _as(
        _literal_column(old_nodes, old_type.value() if old_type else source),
        source,
        "old values",
    )
    var new = _as(
        _literal_column(new_nodes, new_type.value() if new_type else target),
        target,
        "new values",
    )
    var fallback = Optional[Series]()
    if not strict:
        fallback = values.copy()
    elif default:
        var given = default.value().copy()
        if given.is_chunked():
            given = given.rechunk()
        if given.dtype().is_categorical():
            given = decode(given)
        fallback = _as(given, target, "default")
    if len(old) == 0:
        if fallback:
            var kept = fallback.take()
            return kept._broadcast(n) if len(kept) == 1 and n != 1 else kept^
        if values.null_count() != n:
            raise Error("replace_strict: must specify which values to replace")
        return Series.full_null("", target, n)
    if len(new) != len(old) and len(new) != 1:
        raise Error(
            "new values for replace must have the same length as old or"
            " length 1"
        )
    if encode_rows([old.copy()], True).count() != len(old):
        raise Error("old values for replace must not contain duplicates")
    var lanes = workers if workers > 0 else worker_count(n)
    lanes = max(1, min(lanes, n // 8))
    var lookup = ArcPointer(_Lookup(old))
    var rows = List[Int32](unsafe_uninit_length=n)
    var bounds = partitions(n, lanes, 8)
    var probes = List[_ProbeJob](capacity=lanes)
    for w in range(lanes):
        probes.append(
            _ProbeJob(
                values, lookup, Int(rows.unsafe_ptr()), bounds[w], bounds[w + 1]
            )
        )
    if lanes == 1:
        probes[0].run()
    else:
        run_jobs(probes)
    var hits = 0
    var missed = -1
    for w in range(len(probes)):
        hits += probes[w].hits
        if missed < 0:
            missed = probes[w].missed
    if strict and not default and missed >= 0:
        raise Error(
            "incomplete mapping specified for replace_strict: no value for "
            + String(values.get(missed))
            + "; pass a default to set unmapped values"
        )
    var single = len(new) == 1
    var result = Optional[Series]()
    if hits == 0 and fallback:
        var kept = fallback.take()
        result = kept._broadcast(n) if len(kept) == 1 and n != 1 else kept^
    elif new._data.isa[StringColumn]():
        result = _write_views(rows, new, fallback, single, lanes)
    else:
        comptime for d in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[d]
            if new._data.isa[Column[Scalar[D]]]():
                result = _write_numeric[D](rows, new, fallback, single, lanes)
    if not result:
        # Bools and 128-bit decimals: gather the new values, then pick.
        var take = List[Int](capacity=n)
        var matched = List[Bool](capacity=n)
        for i in range(n):
            var at = Int(rows[i])
            take.append(0 if single and at >= 0 else at)
            matched.append(at >= 0)
        var gathered = take_parallel([new.copy()], take^, lanes, or_null=True)
        var picked = gathered.pop()
        if hits != n and fallback:
            picked = choose(matched, picked, fallback.value())
        result = picked^
    var done = result.take()
    if done.dtype() != target and not target.is_categorical():
        done = done.with_dtype(target)
    return done^
