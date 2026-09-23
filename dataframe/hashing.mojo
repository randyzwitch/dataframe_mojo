"""Composite row keys: dense ids for grouping, joins, and distinct rows.

Each key column is first mapped to dense per-column codes. Codes are then
combined one column at a time and re-densified, so ids never depend on
string concatenation or hash collisions: equality is exact per column.
Float64 keys treat every NaN as one value and -0.0 as equal to 0.0.
"""
from std.collections import Dict
from .aggregate import float_key
from .bool_column import BoolColumn
from .column import Column
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


def column_codes(
    series: Series, mut codes: List[Int], mut nulls: List[Bool]
) -> Int:
    """Fill per-row dense codes and null flags; return the distinct count."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            ref column = series._data[Column[Scalar[D]]]
            comptime if D.is_floating_point():
                # Every NaN is one key and -0.0 equals 0.0 (exact widening).
                var lookup = Dict[UInt64, Int]()
                for i in range(len(column)):
                    if not column._valid(i):
                        nulls[i] = True
                        continue
                    var key = float_key(Float64(column._get(i)))
                    var code = lookup.get(key, -1)
                    if code < 0:
                        code = len(lookup)
                        lookup[key] = code
                    codes[i] = code
                return len(lookup)
            else:
                return _codes_by_value(column, codes, nulls)
    if series._data.isa[BoolColumn]():
        ref column = series._data[BoolColumn]
        for i in range(len(column)):
            if not column._valid(i):
                nulls[i] = True
            else:
                codes[i] = Int(column._get(i))
        return 2
    # Keys borrow the column's UTF-8 buffer; no String is allocated per row.
    ref column = series._data[StringColumn]
    var lookup = Dict[StringSlice[ImmutAnyOrigin], Int]()
    for i in range(len(column)):
        if not column._valid(i):
            nulls[i] = True
            continue
        var value = column._get(i)
        var code = lookup.get(value, -1)
        if code < 0:
            code = len(lookup)
            lookup[value] = code
        codes[i] = code
    return len(lookup)


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
        while self.codes[slot] >= 0:
            if self.keys[slot] == key:
                return self.codes[slot]
            slot = (slot + 1) & 511
        self.keys[slot] = key
        self.codes[slot] = next_id
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
                var value = column._get(i)
                var code = -1
                if value.byte_length() <= 12:
                    code = inline.get_or_insert(
                        _inline_view_key(value), len(representatives)
                    )
                else:
                    code = long_lookup.get(value, -1)
                    if code < 0:
                        code = len(representatives)
                        long_lookup[value] = code
                if code == len(representatives):
                    representatives.append(row)
                ids.append(code)
                row += 1
    return RowKeys(ids^, representatives^)


struct _StringEncodeJob(Job):
    var source: Series
    var start: Int
    var end: Int
    var nulls_equal: Bool
    var result: RowKeys

    def __init__(
        out self, source: Series, start: Int, end: Int, nulls_equal: Bool
    ):
        self.source = source.copy()
        self.start = start
        self.end = end
        self.nulls_equal = nulls_equal
        self.result = RowKeys(List[Int](), List[Int]())

    def run(mut self) raises:
        self.result = _encode_string_rows(
            self.source.slice(self.start, self.end - self.start),
            self.nulls_equal,
        )

    def into_result(deinit self) -> RowKeys:
        return self.result^


def encode_string_rows_parallel(
    series: Series, nulls_equal: Bool, workers: Int
) raises -> RowKeys:
    """Encode contiguous string row ranges in parallel, then merge local ids."""
    if workers <= 1:
        return _encode_string_rows(series, nulls_equal)
    var bounds = partitions(len(series), workers, 1)
    var jobs = List[_StringEncodeJob](capacity=workers)
    for w in range(workers):
        jobs.append(
            _StringEncodeJob(series, bounds[w], bounds[w + 1], nulls_equal)
        )
    run_jobs(jobs)
    var ids = List[Int](length=len(series), fill=-1)
    var representatives = List[Int]()
    var lookup = Dict[String, Int]()
    var null_code = -1
    for w in range(workers):
        var local = jobs.pop(0).into_result()
        var mapping = List[Int](length=local.count(), fill=-1)
        for code in range(local.count()):
            var row = bounds[w] + local.representatives[code]
            var value = series.get(row)
            var global_code: Int
            if value.is_null():
                if null_code < 0:
                    null_code = len(representatives)
                    representatives.append(row)
                global_code = null_code
            else:
                var key = value.string()
                global_code = lookup.get(key, -1)
                if global_code < 0:
                    global_code = len(representatives)
                    lookup[key] = global_code
                    representatives.append(row)
            mapping[code] = global_code
        for i in range(len(local.ids)):
            var code = local.ids[i]
            if code >= 0:
                ids[bounds[w] + i] = mapping[code]
    return RowKeys(ids^, representatives^)


def encode_rows(keys: List[Series], nulls_equal: Bool) raises -> RowKeys:
    """Assign dense ids to distinct key rows, in first-occurrence order.

    With nulls_equal=True a null is an ordinary key value, equal only to
    other nulls in the same column. Otherwise any null key yields id -1.
    """
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
    if len(keys) == 1 and keys[0]._data.isa[Column[Int64]]():
        ref column = keys[0]._data[Column[Int64]]
        var first = True
        var low = Int64(0)
        var high = Int64(0)
        for i in range(n):
            if not column._valid(i):
                continue
            var value = column._get(i)
            if first:
                low = value
                high = value
                first = False
            else:
                low = min(low, value)
                high = max(high, value)
        # A compact integer domain can be encoded by direct lookup, avoiding
        # a hash-table probe and a second renumbering pass for every row.
        if first or UInt64(high) - UInt64(low) < 4096:
            var slots = List[Int](
                length=1 if first else Int(UInt64(high) - UInt64(low)) + 1,
                fill=-1,
            )
            var ids = List[Int](length=n, fill=-1)
            var representatives = List[Int]()
            var null_id = -1
            for i in range(n):
                if not column._valid(i):
                    if nulls_equal:
                        if null_id < 0:
                            null_id = len(representatives)
                            representatives.append(i)
                        ids[i] = null_id
                    continue
                var slot = Int(UInt64(column._get(i)) - UInt64(low))
                if slots[slot] < 0:
                    slots[slot] = len(representatives)
                    representatives.append(i)
                ids[i] = slots[slot]
            return RowKeys(ids^, representatives^)
    var ids = List[Int](length=n, fill=0)
    var excluded = List[Bool](length=n, fill=False)
    var representatives = List[Int]()

    if len(keys) == 1:
        # One key column needs no hash map to combine columns, because
        # there is nothing to combine. `column_codes` has already numbered
        # the distinct values; all that remains is to renumber them in the
        # order the rows meet them, which an array indexed by code does.
        # The general path below uses a Dict for this, which on a
        # high-cardinality key is a lookup per row over as many entries as
        # there are distinct keys.
        #
        # The renumbering is not skippable even though most dtypes already
        # code in first-occurrence order: a null takes an id in row order
        # too, so one null early in the column shifts every id after it.
        # Booleans code by value rather than by order, and this renumbers
        # them correctly as well.
        var codes = List[Int](length=n, fill=0)
        var nulls = List[Bool](length=n, fill=False)
        var distinct = column_codes(keys[0], codes, nulls)
        var renumber = List[Int](length=distinct + 1, fill=-1)
        var next_id = 0
        for i in range(n):
            if nulls[i]:
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
        var codes = List[Int](length=n, fill=0)
        var nulls = List[Bool](length=n, fill=False)
        var distinct = column_codes(keys[j], codes, nulls)
        # Reserve code `distinct` for null so it is one ordinary value.
        var radix = distinct + 1
        var lookup = Dict[Int, Int]()
        representatives = List[Int]()
        for i in range(n):
            if nulls[i]:
                if not nulls_equal:
                    excluded[i] = True
                codes[i] = distinct
            if excluded[i]:
                ids[i] = -1
                continue
            var combined = ids[i] * radix + codes[i] if j > 0 else codes[i]
            var id = lookup.get(combined, -1)
            if id < 0:
                id = len(lookup)
                lookup[combined] = id
                representatives.append(i)
            ids[i] = id
    return RowKeys(ids^, representatives^)
