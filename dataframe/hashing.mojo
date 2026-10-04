"""Composite row keys: dense ids for grouping, joins, and distinct rows.

Each key column is first mapped to dense per-column codes. Codes are then
combined one column at a time and re-densified, so ids never depend on
string concatenation or hash collisions: equality is exact per column.
Float64 keys treat every NaN as one value and -0.0 as equal to 0.0.
"""
from std.collections import Dict
from std.memory import Pointer, bitcast, unsafe_memcpy
from .aggregate import float_key
from .bool_column import BoolColumn
from .column import Column, _validity_at
from .dtype import DataType, NUMERIC_DTYPES
from .string_column import StringColumn, StringBuilder
from .series import Series
from .parallel import Job, partitions, run_jobs


@fieldwise_init
struct RowKeys(Movable):
    """Dense key ids in first-occurrence order.

    `ids[i]` is row i's key id, or -1 when the row has a null key and nulls
    do not match. `representatives[g]` is the first row with id g.
    """

    var ids: List[Int]
    var representatives: List[Int]

    def count(self) -> Int:
        return len(self.representatives)


def _codes_by_value[
    T: Copyable & Deinitable & Hashable & Equatable
](column: Column[T], mut codes: List[Int], mut nulls: List[Bool]) -> Int:
    var lookup = Dict[T, Int]()
    for i in range(len(column)):
        if not column._valid(i):
            nulls[i] = True
            continue
        ref value = column._get(i)
        var code = lookup.get(value, -1)
        if code < 0:
            code = len(lookup)
            lookup[value.copy()] = code
        codes[i] = code
    return len(lookup)


struct _U64Codes(Movable):
    """Dense codes for 64-bit keys in first-insertion order: an
    open-addressing table with a multiplicative hash, growing at half load
    (#380). A standard-library `Dict` lookup per row was most of a
    low-cardinality group-by's key numbering."""

    var keys: List[UInt64]
    var slots: List[Int32]
    var count: Int
    var shift: UInt64

    def __init__(out self):
        self.keys = List[UInt64]()
        self.slots = List[Int32](length=1024, fill=-1)
        self.count = 0
        self.shift = 64 - 10

    @always_inline
    def code(mut self, key: UInt64) -> Int:
        """`key`'s code, assigning the next one to a new key."""
        var mask = len(self.slots) - 1
        var slot = Int((key * 0x9E3779B97F4A7C15) >> self.shift)
        while True:
            var at = Int(self.slots[slot])
            if at < 0:
                self.slots[slot] = Int32(self.count)
                self.keys.append(key)
                self.count += 1
                if 2 * self.count > len(self.slots):
                    self._grow()
                return self.count - 1
            if self.keys[at] == key:
                return at
            slot = (slot + 1) & mask

    def _grow(mut self):
        var size = 2 * len(self.slots)
        self.shift -= 1
        self.slots = List[Int32](length=size, fill=-1)
        for at in range(self.count):
            var slot = Int((self.keys[at] * 0x9E3779B97F4A7C15) >> self.shift)
            while self.slots[slot] >= 0:
                slot = (slot + 1) & (size - 1)
            self.slots[slot] = Int32(at)


def column_codes(
    series: Series,
    mut codes: List[Int],
    mut nulls: List[Bool],
    *,
    dense_int64: Bool = True,
) raises -> Int:
    """Replace codes with owned per-row codes and fill null flags.

    Value encoders number valid values by first occurrence. BoolColumn
    retains its fixed 0/1 codes. Return the valid-value code-domain size.
    """
    if series._data.isa[StringColumn]():
        var keys = _encode_string_rows(series, False)
        var count = keys.count()
        if series.null_count() > 0:
            for i in range(len(keys.ids)):
                nulls[i] = keys.ids[i] < 0
        codes = keys.ids^
        keys.ids = List[Int]()
        return count
    if (
        series.dtype().is_categorical()
        and series.dtype().has_dictionary()
        and len(series.dtype().dictionary()[]) <= max(len(series), 4096)
    ):
        # Composite keys can use the same bounded dictionary lookup as a
        # single categorical key. Number the full source, including chunks,
        # and validate codes before directly indexing the dictionary domain.
        var source = series.rechunk() if series.is_chunked() else series.copy()
        var keys = _encode_categorical[checked=True](
            source._data[Column[UInt32]],
            len(source.dtype().dictionary()[]),
            False,
        )
        var count = keys.count()
        if source.null_count() > 0:
            for i in range(len(keys.ids)):
                nulls[i] = keys.ids[i] < 0
        codes = keys.ids^
        keys.ids = List[Int]()
        return count
    if dense_int64 and series._data.isa[Column[Int64]]():
        var dense = _encode_dense_int64(series._data[Column[Int64]], False)
        if dense:
            var keys = dense.take()
            var count = keys.count()
            if series.null_count() > 0:
                for i in range(len(keys.ids)):
                    nulls[i] = keys.ids[i] < 0
            codes = keys.ids^
            keys.ids = List[Int]()
            return count
    codes = List[Int](unsafe_uninit_length=len(series))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            ref column = series._data[Column[Scalar[D]]]
            # Values of up to 64 bits are their own keys: every NaN is one
            # key and -0.0 equals 0.0 (float_key), integers by bit pattern.
            var table = _U64Codes()
            var has_nulls = column.null_count() > 0
            for i in range(len(column)):
                if has_nulls and not column._valid(i):
                    nulls[i] = True
                    continue
                comptime if D.is_floating_point():
                    codes[i] = table.code(float_key(Float64(column._get(i))))
                else:
                    codes[i] = table.code(
                        bitcast[DType.uint64](
                            column._get(i).cast[DType.int64]()
                        ) if D.is_signed() else column._get(i).cast[
                            DType.uint64
                        ]()
                    )
            return table.count
    if series._data.isa[Column[Int128]]():
        return _codes_by_value(series._data[Column[Int128]], codes, nulls)
    if series._data.isa[BoolColumn]():
        ref column = series._data[BoolColumn]
        for i in range(len(column)):
            if not column._valid(i):
                nulls[i] = True
            else:
                codes[i] = Int(column._get(i))
        return 2
    raise Error("unsupported key column storage")


struct _InlineStringCodes(Movable):
    """Exact codes for inline view keys with a general high-cardinality fallback."""

    var keys: List[UInt128]
    var codes: List[Int]
    var count: Int
    var fallback: Dict[UInt128, Int]
    var use_fallback: Bool

    def __init__(out self):
        self.keys = List[UInt128](length=512, fill=0)
        self.codes = List[Int](length=512, fill=-1)
        self.count = 0
        self.fallback = Dict[UInt128, Int]()
        self.use_fallback = False

    @always_inline
    def get_or_insert(mut self, key: UInt128, next_id: Int) -> Int:
        if self.use_fallback:
            var code = self.fallback.get(key, -1)
            if code < 0:
                self.fallback[key] = next_id
                return next_id
            return code
        var hash_value = UInt64(key) ^ UInt64(key >> 64)
        hash_value = (hash_value ^ (hash_value >> 30)) * 0xBF58476D1CE4E5B9
        hash_value = (hash_value ^ (hash_value >> 27)) * 0x94D049BB133111EB
        var slot = Int((hash_value ^ (hash_value >> 31)) & 511)
        var keys = self.keys.unsafe_ptr()
        var codes = self.codes.unsafe_ptr()
        while codes[unsafe_offset=slot] >= 0:
            if keys[unsafe_offset=slot] == key:
                return codes[unsafe_offset=slot]
            slot = (slot + 1) & 511
        keys[unsafe_offset=slot] = key
        codes[unsafe_offset=slot] = next_id
        self.count += 1
        if self.count == 256:
            for i in range(512):
                if self.codes[i] >= 0:
                    self.fallback[self.keys[i]] = self.codes[i]
            self.use_fallback = True
        return next_id


def _inline_view_key(value: StringSlice[ImmutAnyOrigin]) -> UInt128:
    """Pack legacy short strings into the same key as inline views."""
    var bytes = value.as_bytes()
    var key = UInt128(len(bytes))
    for i in range(len(bytes)):
        key |= UInt128(bytes[i]) << UInt128((i + 4) * 8)
    return key


def _encode_string_rows(series: Series, nulls_equal: Bool) -> RowKeys:
    """Number one string key directly across its physical chunks."""
    var ids = List[Int](capacity=len(series))
    var representatives = List[Int]()
    var inline = _InlineStringCodes()
    var long_lookup = Dict[StringSlice[ImmutAnyOrigin], Int]()
    var null_code = -1
    var row = 0
    for chunk in series.chunks():
        ref column = chunk._data[StringColumn]
        if column._is_view_storage():
            var storage = column._view_storage_unchecked()
            for i in range(len(column)):
                if not column._valid(i):
                    if nulls_equal:
                        if null_code < 0:
                            null_code = len(representatives)
                            representatives.append(row)
                        ids.append(null_code)
                    else:
                        ids.append(-1)
                    row += 1
                    continue
                var view = storage._view_unchecked(column._offset + i)
                var code: Int
                if view.is_inline():
                    var key = (
                        UInt128(view.length)
                        | (UInt128(view.prefix) << 32)
                        | (UInt128(view.buffer_index) << 64)
                        | (UInt128(view.offset) << 96)
                    )
                    code = inline.get_or_insert(key, len(representatives))
                else:
                    var value = storage._get_unchecked(column._offset + i)
                    code = long_lookup.get(value, -1)
                    if code < 0:
                        code = len(representatives)
                        long_lookup[value] = code
                if code == len(representatives):
                    representatives.append(row)
                ids.append(code)
                row += 1
        else:
            # Offsets and bytes are read in place: a short value's key is
            # one copy into the inline-view layout (length, then bytes from
            # byte 4), not a loop over its bytes, and a row equal to the one
            # before reuses its code.
            var m = len(column)
            var start_len = len(ids)
            ids.resize(start_len + m, -1)
            var out = ids.unsafe_ptr().unsafe_offset(start_len)
            var offsets = (
                column._offsets[].unsafe_ptr().unsafe_offset(column._offset)
            )
            var bytes = column._base()
            # A missing bitmap means no nulls; counting them is O(n).
            var nulls = len(column._bits[]) != 0
            var last_key = UInt128(0)
            var last_code = -1
            for i in range(m):
                if nulls and not column._valid(i):
                    if nulls_equal:
                        if null_code < 0:
                            null_code = len(representatives)
                            representatives.append(row + i)
                        out[unsafe_offset=i] = null_code
                    continue
                var start = Int(offsets[unsafe_offset=i])
                var length = Int(offsets[unsafe_offset=i + 1]) - start
                var code: Int
                if length <= 12:
                    var key = UInt128(length)
                    unsafe_memcpy(
                        dest=Pointer(to=key)
                        .unsafe_bitcast[UInt8]()
                        .unsafe_offset(4),
                        src=bytes.unsafe_offset(start),
                        count=length,
                    )
                    if key == last_key and last_code >= 0:
                        out[unsafe_offset=i] = last_code
                        continue
                    code = inline.get_or_insert(key, len(representatives))
                    last_key = key
                    last_code = code
                else:
                    var value = column._get(i)
                    code = long_lookup.get(value, -1)
                    if code < 0:
                        code = len(representatives)
                        long_lookup[value] = code
                if code == len(representatives):
                    representatives.append(row + i)
                out[unsafe_offset=i] = code
            row += m
    return RowKeys(ids^, representatives^)


def encode_string_rows_parallel(
    series: Series, nulls_equal: Bool, workers: Int
) raises -> RowKeys:
    """Encode contiguous string row ranges in parallel, then merge local ids.

    The merge numbers each range's distinct values with the string encoder
    and renumbers the ranges on every worker (`encode_rows_parallel`). It
    used to read each distinct value into a `String` and look it up in a
    `Dict` on one thread, which on 835K distinct ClickBench search phrases
    took most of a 900 ms group-by.
    """
    return encode_rows_parallel([series.copy()], nulls_equal, workers)


struct _RangeEncodeJob(Job):
    """encode_rows over rows [first, last) of every key."""

    var keys: List[Series]
    var first: Int
    var last: Int
    var nulls_equal: Bool
    var ids: List[Int]
    var representatives: List[Int]

    def __init__(
        out self, keys: List[Series], first: Int, last: Int, nulls_equal: Bool
    ):
        self.keys = keys.copy()
        self.first = first
        self.last = last
        self.nulls_equal = nulls_equal
        self.ids = List[Int]()
        self.representatives = List[Int]()

    def run(mut self) raises:
        var window = List[Series](capacity=len(self.keys))
        for key in self.keys:
            window.append(key.slice(self.first, self.last - self.first))
        var local = encode_rows(window, self.nulls_equal)
        self.ids = local.ids.copy()
        self.representatives = local.representatives.copy()


struct _RenumberJob(Job):
    """Replace one range's local ids by global ones."""

    var ids: Int
    var local: Int
    var mapping: List[Int]
    var first: Int

    def __init__(
        out self, ids: Int, local: Int, var mapping: List[Int], first: Int
    ):
        self.ids = ids
        self.local = local
        self.mapping = mapping^
        self.first = first

    def run(mut self) raises:
        var out = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.ids
        )[].unsafe_ptr()
        ref local = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.local
        )[]
        var mapping = self.mapping.unsafe_ptr()
        for i in range(len(local)):
            var code = local[i]
            out[unsafe_offset=self.first + i] = (
                mapping[unsafe_offset=code] if code >= 0 else -1
            )


def encode_rows_parallel(
    keys: List[Series], nulls_equal: Bool, workers: Int
) raises -> RowKeys:
    """`encode_rows` for keys with few distinct rows, on every worker.

    Each worker encodes a contiguous range of rows. The ranges' distinct
    rows, taken in range order and each range's first-occurrence order, are
    then encoded once more, which numbers them in first-occurrence order
    over the whole input, and each range's ids are renumbered in parallel.
    The merge encodes one row per distinct key per range, so this suits the
    few distinct keys a whole-frame group_by is chosen for; many distinct
    keys belong to the hash-partitioned encoder.
    """
    var n = len(keys[0]) if len(keys) > 0 else 0
    if workers <= 1 or n < 2 * workers:
        return encode_rows(keys, nulls_equal)
    var bounds = partitions(n, workers, 1)
    var jobs = List[_RangeEncodeJob](capacity=workers)
    for w in range(workers):
        if bounds[w + 1] > bounds[w]:
            jobs.append(
                _RangeEncodeJob(keys, bounds[w], bounds[w + 1], nulls_equal)
            )
    run_jobs(jobs)
    var rows = List[Int]()
    for job in range(len(jobs)):
        for local in jobs[job].representatives:
            rows.append(jobs[job].first + local)
    var firsts = List[Series](capacity=len(keys))
    for key in keys:
        firsts.append(key.take(rows))
    var merged = encode_rows(firsts, nulls_equal)
    var representatives = List[Int](capacity=merged.count())
    for r in merged.representatives:
        representatives.append(rows[r])
    var ids = List[Int](length=n, fill=-1)
    var renumber = List[_RenumberJob](capacity=len(jobs))
    var at = 0
    for job in range(len(jobs)):
        var count = len(jobs[job].representatives)
        var mapping = List[Int](capacity=count)
        for k in range(count):
            mapping.append(merged.ids[at + k])
        at += count
        renumber.append(
            _RenumberJob(
                Int(Pointer(to=ids)),
                Int(Pointer(to=jobs[job].ids)),
                mapping^,
                jobs[job].first,
            )
        )
    run_jobs(renumber)
    _ = jobs^
    return RowKeys(ids^, representatives^)


def _encode_dense_int64(
    column: Column[Int64], nulls_equal: Bool
) raises -> Optional[RowKeys]:
    """Ids by direct lookup when the key's values span fewer than 4,096
    integers, avoiding a hash-table probe and a renumbering pass per row.
    Values and ids go through pointers, and a column without nulls reads no
    validity: H2O's 100-value id4 spent most of a group-by here."""
    var n = len(column)
    var values = column._ptr()
    var nulls = column.null_count() > 0
    var bits = column._bits[].unsafe_ptr()
    var bit_offset = column._offset
    var first = True
    var low = Int64(0)
    var high = Int64(0)
    if not nulls and n > 0:
        low = values[unsafe_offset=0]
        high = low
        first = False
        for i in range(1, n):
            var value = values[unsafe_offset=i]
            low = min(low, value)
            high = max(high, value)
    else:
        for i in range(n):
            if not _validity_at(bits, bit_offset + i):
                continue
            var value = values[unsafe_offset=i]
            if first:
                low = value
                high = value
                first = False
            else:
                low = min(low, value)
                high = max(high, value)
    if not first and UInt64(high) - UInt64(low) >= 4096:
        return None
    var span = 1 if first else Int(UInt64(high) - UInt64(low)) + 1
    var slots = List[Int](length=span, fill=-1)
    var table = slots.unsafe_ptr()
    var ids = List[Int](unsafe_uninit_length=n)
    var out = ids.unsafe_ptr()
    var representatives = List[Int]()
    if not nulls:
        for i in range(n):
            var slot = Int(UInt64(values[unsafe_offset=i]) - UInt64(low))
            var id = table[unsafe_offset=slot]
            if id < 0:
                id = len(representatives)
                table[unsafe_offset=slot] = id
                representatives.append(i)
            out[unsafe_offset=i] = id
        return RowKeys(ids^, representatives^)
    var null_id = -1
    for i in range(n):
        if not _validity_at(bits, bit_offset + i):
            if nulls_equal:
                if null_id < 0:
                    null_id = len(representatives)
                    representatives.append(i)
                out[unsafe_offset=i] = null_id
            else:
                out[unsafe_offset=i] = -1
            continue
        var slot = Int(UInt64(values[unsafe_offset=i]) - UInt64(low))
        var id = table[unsafe_offset=slot]
        if id < 0:
            id = len(representatives)
            table[unsafe_offset=slot] = id
            representatives.append(i)
        out[unsafe_offset=i] = id
    return RowKeys(ids^, representatives^)


def _encode_categorical[
    checked: Bool = False
](column: Column[UInt32], domain: Int, nulls_equal: Bool) raises -> RowKeys:
    """Ids by direct lookup on a categorical's codes, which already lie in
    [0, dictionary size): no hash and no range scan, as `_encode_dense_int64`
    does for small Int64 domains. Sorted H2O q1 grouped its coded string key
    through the general path 27% slower than the strings."""
    var n = len(column)
    var codes = column._ptr()
    var nulls = column.null_count() > 0
    var bits = column._bits[].unsafe_ptr()
    var bit_offset = column._offset
    var slots = List[Int](length=max(domain, 1), fill=-1)
    var table = slots.unsafe_ptr()
    var ids = List[Int](unsafe_uninit_length=n)
    var out = ids.unsafe_ptr()
    var representatives = List[Int]()
    var null_id = -1
    for i in range(n):
        if nulls and not _validity_at(bits, bit_offset + i):
            if nulls_equal:
                if null_id < 0:
                    null_id = len(representatives)
                    representatives.append(i)
                out[unsafe_offset=i] = null_id
            else:
                out[unsafe_offset=i] = -1
            continue
        var slot = Int(codes[unsafe_offset=i])
        comptime if checked:
            if slot >= domain:
                raise Error("Categorical code exceeds its dictionary")
        var id = table[unsafe_offset=slot]
        if id < 0:
            id = len(representatives)
            table[unsafe_offset=slot] = id
            representatives.append(i)
        out[unsafe_offset=i] = id
    return RowKeys(ids^, representatives^)


def encode_rows(keys: List[Series], nulls_equal: Bool) raises -> RowKeys:
    """Assign dense ids to distinct key rows, in first-occurrence order.

    With nulls_equal=True a null is an ordinary key value, equal only to
    other nulls in the same column. Otherwise any null key yields id -1.
    """
    for key in keys:
        if key.dtype().is_nested():
            raise Error(
                "list and struct columns cannot be keys yet: " + key.name()
            )
    if len(keys) == 0:
        raise Error("Row keys require at least one column")
    if len(keys) == 1 and keys[0]._data.isa[StringColumn]():
        return _encode_string_rows(keys[0], nulls_equal)
    for key in keys:
        if key.is_chunked():
            var contiguous = List[Series](capacity=len(keys))
            for item in keys:
                contiguous.append(item.rechunk())
            return encode_rows(contiguous^, nulls_equal)
    var n = len(keys[0])
    for key in keys:
        if len(key) != n:
            raise Error("Key columns must have equal lengths")
    if n >= 2147483647:
        raise Error("Row key encoding supports fewer than 2**31 rows")
    if (
        len(keys) == 1
        and keys[0].dtype().is_categorical()
        and keys[0].dtype().has_dictionary()
        and len(keys[0].dtype().dictionary()[]) <= max(n, 4096)
    ):
        return _encode_categorical(
            keys[0]._data[Column[UInt32]],
            len(keys[0].dtype().dictionary()[]),
            nulls_equal,
        )
    if len(keys) == 1 and keys[0]._data.isa[Column[Int64]]():
        var dense = _encode_dense_int64(
            keys[0]._data[Column[Int64]], nulls_equal
        )
        if dense:
            return dense.take()
    # Every row gets an id (or -1) below before any is read (#388).
    var ids = List[Int]()
    # Null flags are kept only for columns that have nulls, and exclusion
    # only when nulls drop rows, so a null-free key fills nothing (#388).
    var any_nulls = False
    for key in keys:
        any_nulls = any_nulls or key.null_count() > 0
    var exclude = any_nulls and not nulls_equal
    var excluded = List[Bool](length=n if exclude else 0, fill=False)
    var representatives = List[Int]()

    if len(keys) == 1:
        # Value encoders already assign first-occurrence codes. A null-free
        # non-Boolean key can return those ids directly; otherwise inserting
        # the null group or renumbering fixed Boolean codes requires the
        # ordinary mapping below. The dense Int64 route was tried above, so
        # avoid repeating its range scan here.
        var codes = List[Int]()
        var has_nulls = keys[0].null_count() > 0
        var nulls = List[Bool](length=n if has_nulls else 0, fill=False)
        var distinct = column_codes(keys[0], codes, nulls, dense_int64=False)
        if not has_nulls and not keys[0]._data.isa[BoolColumn]():
            for i in range(n):
                if codes[i] == len(representatives):
                    representatives.append(i)
            return RowKeys(codes^, representatives^)
        ids = List[Int](unsafe_uninit_length=n)
        var renumber = List[Int](length=distinct + 1, fill=-1)
        var next_id = 0
        for i in range(n):
            if has_nulls and nulls[i]:
                if not nulls_equal:
                    ids[i] = -1
                    continue
                codes[i] = distinct
            var code = codes[i]
            if renumber[code] < 0:
                renumber[code] = next_id
                next_id += 1
                representatives.append(i)
            ids[i] = renumber[code]
        return RowKeys(ids^, representatives^)

    for j in range(len(keys)):
        # column_codes writes every valid row; null rows are given their
        # code below before it is read, so nothing is filled first (#388).
        var codes = List[Int]()
        var has_nulls = keys[j].null_count() > 0
        var nulls = List[Bool](length=n if has_nulls else 0, fill=False)
        var distinct = column_codes(keys[j], codes, nulls)
        # Value encoders assign first-occurrence ids already. Without nulls
        # there is no null id to insert; BoolColumn alone uses fixed 0/1
        # codes and still needs the ordinary renumbering below.
        if j == 0:
            if not has_nulls and not keys[j]._data.isa[BoolColumn]():
                for i in range(n):
                    if codes[i] == len(representatives):
                        representatives.append(i)
                ids = codes^
                continue
            ids = List[Int](unsafe_uninit_length=n)
        # Reserve code `distinct` for null so it is one ordinary value.
        var radix = distinct + 1
        # Combined codes index a direct array when there are few enough of
        # them, else a 64-bit table; both replace a `Dict` lookup (#380).
        var combinations = (len(representatives) if j > 0 else 1) * radix
        var direct = combinations <= max(4 * n, 1 << 20)
        var dense = List[Int](length=combinations if direct else 0, fill=-1)
        var table = _U64Codes()
        representatives = List[Int]()
        for i in range(n):
            if has_nulls and nulls[i]:
                if not nulls_equal:
                    excluded[i] = True
                codes[i] = distinct
            if exclude and excluded[i]:
                ids[i] = -1
                continue
            var combined = ids[i] * radix + codes[i] if j > 0 else codes[i]
            var id: Int
            if direct:
                id = dense[combined]
                if id < 0:
                    id = len(representatives)
                    dense[combined] = id
                    representatives.append(i)
            else:
                var before = table.count
                id = table.code(UInt64(combined))
                if table.count > before:
                    representatives.append(i)
            ids[i] = id
    return RowKeys(ids^, representatives^)
