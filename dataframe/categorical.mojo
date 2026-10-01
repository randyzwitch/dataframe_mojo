"""Dictionary-encoded strings: the Categorical type (#106).

A categorical column stores one UInt32 code per row, in an ordinary
`Column[UInt32]`, and its DataType carries the dictionary of distinct values
the codes index, as temporal types ride on Int64 storage. This is Arrow's
dictionary layout (`dictionary<uint32, large_utf8>`), so it exports and
imports without conversion. Operations that only need equality (grouping,
joins, unique, value counts) work on the codes as integers; sorting orders
by the values through a rank per code; anything reading the text decodes
it here. A row-wise expression over a categorical alone runs once per
dictionary value, and `gather` spreads its results over the rows by code.

Two categoricals index the same values only if their dictionaries match.
Joining or concatenating columns with different dictionaries first moves
them onto the union of both (`unify`), in which the first dictionary's codes
are unchanged.
"""
from std.collections import Dict

from .bool_column import BoolColumn
from .column import Column
from .dtype import CategoricalDictionary, DataType, NUMERIC_DTYPES
from .gather import take_parallel
from .hashing import encode_string_rows_parallel
from .parallel import Job, run_jobs, worker_count
from .series import Series
from .string_column import StringBuilder, StringColumn


def encode(values: Series) raises -> Series:
    """A String column as a categorical: the distinct values in the order
    they first appear, nulls kept as nulls."""
    if not values.dtype().physical() == DataType.STRING:
        raise Error(
            "only a string column can be encoded as categorical, not "
            + values.dtype().name()
        )
    var strings = values.rechunk() if values.is_chunked() else values.copy()
    var n = len(strings)
    var keys = encode_string_rows_parallel(strings, False, worker_count(n))
    ref column = strings._data[StringColumn]
    var dictionary = CategoricalDictionary()
    for row in keys.representatives:
        dictionary.append(column._get(row))
    var codes = List[UInt32](length=n, fill=0)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        var id = keys.ids[i]
        if id >= 0:
            codes[i] = UInt32(id)
            valid[i] = True
    return Series(values.name(), Column[UInt32](codes^, valid)).with_dtype(
        DataType.categorical(dictionary^)
    )


def decode(values: Series) raises -> Series:
    """A categorical column's values as a String column."""
    if not values.dtype().is_categorical():
        return values.copy()
    var codes = values.rechunk() if values.is_chunked() else values.copy()
    ref column = codes._data[Column[UInt32]]
    var dictionary = values.dtype().dictionary()
    var out = StringBuilder(len(column))
    for i in range(len(column)):
        if column._valid(i):
            out.append(dictionary[].get(Int(column._get(i))))
        else:
            out.append_null()
    return Series(values.name(), out^.finish())


def union_of(
    first: CategoricalDictionary, second: CategoricalDictionary
) -> CategoricalDictionary:
    """`first`'s values, then those of `second` it lacks, in order."""
    var merged = CategoricalDictionary(first.bytes.copy(), first.offsets.copy())
    var seen = Dict[String, Int]()
    for code in range(len(first)):
        seen[String(first.get(code))] = code
    for code in range(len(second)):
        var value = String(second.get(code))
        if value not in seen:
            seen[value] = len(merged)
            merged.append(value)
    return merged^


def recode(values: Series, target: DataType) raises -> Series:
    """`values` (categorical or String) on `target`'s dictionary, which must
    hold every value (see `union_of`)."""
    var strings = decode(values)
    var dictionary = target.dictionary()
    var lookup = Dict[String, Int]()
    for code in range(len(dictionary[])):
        lookup[String(dictionary[].get(code))] = code
    var flat = strings.rechunk() if strings.is_chunked() else strings.copy()
    ref column = flat._data[StringColumn]
    var n = len(column)
    var codes = List[UInt32](length=n, fill=0)
    var valid = List[Bool](length=n, fill=False)
    # One lookup per value of the source dictionary, not per row, when the
    # source is categorical.
    if values.dtype().is_categorical() and values.dtype().has_dictionary():
        var source = values.dtype().dictionary()
        var mapping = List[UInt32](capacity=len(source[]))
        for code in range(len(source[])):
            mapping.append(UInt32(lookup[String(source[].get(code))]))
        var original = (
            values.rechunk() if values.is_chunked() else values.copy()
        )
        ref from_codes = original._data[Column[UInt32]]
        for i in range(n):
            if from_codes._valid(i):
                codes[i] = mapping[Int(from_codes._get(i))]
                valid[i] = True
    else:
        for i in range(n):
            if column._valid(i):
                codes[i] = UInt32(lookup[String(column._get(i))])
                valid[i] = True
    return Series(values.name(), Column[UInt32](codes^, valid)).with_dtype(
        target
    )


def _values_of(column: Series) raises -> CategoricalDictionary:
    """A copy of a categorical's dictionary, or a String column's distinct
    values in first-occurrence order."""
    if column.dtype().is_categorical():
        return column.dtype().dictionary()[].copy_values()
    return encode(column).dtype().dictionary()[].copy_values()


def unify(mut first: Series, mut second: Series) raises:
    """Put two key columns, at least one categorical, on one dictionary so
    their codes compare: the union of both, with `first`'s codes unchanged.
    A String side is encoded into it."""
    if (
        first.dtype().is_categorical()
        and second.dtype().is_categorical()
        and first.dtype() == second.dtype()
    ):
        return
    var target = DataType.categorical(
        union_of(_values_of(first), _values_of(second))
    )
    if first.dtype().is_categorical():
        first = first.with_dtype(target)
    else:
        first = recode(first, target)
    second = recode(second, target)


def sort_ranks(values: Series) raises -> Series:
    """Codes replaced by each value's rank in sorted order, as a plain
    UInt32 column: sorting by it sorts by the values."""
    var dictionary = values.dtype().dictionary()
    var count = len(dictionary[])
    var order = List[Int](capacity=count)
    for code in range(count):
        order.append(code)

    def less(a: Int, b: Int) {imm dictionary} -> Bool:
        return dictionary[].get(a) < dictionary[].get(b)

    sort(order, less)
    var rank = List[UInt32](length=count, fill=0)
    for position in range(count):
        rank[order[position]] = UInt32(position)
    var flat = values.rechunk() if values.is_chunked() else values.copy()
    ref column = flat._data[Column[UInt32]]
    var n = len(column)
    var ranks = List[UInt32](length=n, fill=0)
    var valid = List[Bool](length=n, fill=True)
    for i in range(n):
        if column._valid(i):
            ranks[i] = rank[Int(column._get(i))]
        else:
            valid[i] = False
    return Series(values.name(), Column[UInt32](ranks^, valid))


struct _GatherJob[D: DType, packed: Bool](Job):
    """Rows [first, last) of a gather by code: each row takes its code's
    entry of a small per-value table, or entry `null_index` when its code
    is null. `first` is a multiple of 8, so every job writes whole bytes of
    the validity bitmap, and of the values bitmap when `packed` (Booleans,
    whose table holds 0 or 1)."""

    var codes: Column[UInt32]
    var nulls: Bool
    var table: List[Scalar[Self.D]]
    var table_valid: List[UInt8]
    var null_index: Int
    var values: Int
    var bits: Int
    var first: Int
    var last: Int

    def __init__(
        out self,
        codes: Column[UInt32],
        nulls: Bool,
        table: List[Scalar[Self.D]],
        table_valid: List[UInt8],
        values: Int,
        bits: Int,
        first: Int,
        last: Int,
    ):
        self.codes = codes.copy()
        self.nulls = nulls
        self.table = table.copy()
        self.table_valid = table_valid.copy()
        self.null_index = len(table) - 1
        self.values = values
        self.bits = bits
        self.first = first
        self.last = last

    def run(mut self) raises:
        var codes = self.codes.unsafe_values()
        var table = self.table.unsafe_ptr()
        var table_valid = self.table_valid.unsafe_ptr()
        var bits = Pointer[List[UInt8], MutAnyOrigin](
            unsafe_from_address=self.bits
        )[].unsafe_ptr()
        var row = self.first
        while row < self.last:
            var end = min(row + 8, self.last)
            var valid: UInt8 = 0
            var packed_values: UInt8 = 0
            for i in range(row, end):
                var index = Int(codes.unsafe_offset(i)[])
                if self.nulls and not self.codes._valid(i):
                    index = self.null_index
                var shift = UInt8(i - row)
                valid |= table_valid.unsafe_offset(index)[] << shift
                comptime if Self.packed:
                    packed_values |= (
                        rebind[UInt8](table.unsafe_offset(index)[]) << shift
                    )
                else:
                    Pointer[List[Scalar[Self.D]], MutAnyOrigin](
                        unsafe_from_address=self.values
                    )[].unsafe_ptr().unsafe_offset(i)[] = table.unsafe_offset(
                        index
                    )[]
            bits.unsafe_offset(row // 8)[] = valid
            comptime if Self.packed:
                Pointer[List[UInt8], MutAnyOrigin](
                    unsafe_from_address=self.values
                )[].unsafe_ptr().unsafe_offset(row // 8)[] = packed_values
            row = end


def _gather[
    D: DType, packed: Bool
](
    codes: Column[UInt32],
    table: List[Scalar[D]],
    table_valid: List[UInt8],
    mut values: List[Scalar[D]],
    mut bits: List[UInt8],
) raises:
    var n = len(codes)
    var nulls = codes.null_count() > 0
    var workers = worker_count(n)
    var jobs = List[_GatherJob[D, packed]](capacity=workers)
    for w in range(workers):
        var first = (n * w // workers) // 8 * 8
        var last = n if w == workers - 1 else (n * (w + 1) // workers) // 8 * 8
        jobs.append(
            _GatherJob[D, packed](
                codes,
                nulls,
                table,
                table_valid,
                Int(Pointer(to=values)),
                Int(Pointer(to=bits)),
                first,
                last,
            )
        )
    run_jobs(jobs)


def gather(per_value: Series, codes: Series) raises -> Series:
    """Each row of a categorical `codes` column given its value's entry of
    `per_value`, which holds one row per dictionary value and then one for
    null: an expression evaluated on the dictionary, spread over the rows.
    """
    var flat_codes = codes.rechunk() if codes.is_chunked() else codes.copy()
    var table = (
        per_value.rechunk() if per_value.is_chunked() else per_value.copy()
    )
    ref column = flat_codes._data[Column[UInt32]]
    var n = len(column)
    var entries = len(table)
    var table_valid = List[UInt8](capacity=entries)
    var all_valid = True
    for i in range(entries):
        var ok = not table.get(i).is_null()
        table_valid.append(UInt8(1) if ok else UInt8(0))
        all_valid = all_valid and ok
    var bits = List[UInt8](length=(n + 7) // 8, fill=0)
    if table._data.isa[BoolColumn]():
        ref source = table._data[BoolColumn]
        var lookup = List[UInt8](capacity=entries)
        for i in range(entries):
            lookup.append(UInt8(1) if source._get(i) else UInt8(0))
        var values = List[UInt8](length=(n + 7) // 8, fill=0)
        _gather[DType.uint8, True](column, lookup, table_valid, values, bits)
        if all_valid:
            bits = List[UInt8]()
        var result = Series(
            per_value.name(), BoolColumn(values=values^, bits=bits^, length=n)
        )
        result._dtype = per_value._dtype
        return result^
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if table._data.isa[Column[Scalar[D]]]():
            ref source = table._data[Column[Scalar[D]]]
            var lookup = List[Scalar[D]](capacity=entries)
            for i in range(entries):
                lookup.append(source._get(i))
            var values = List[Scalar[D]](unsafe_uninit_length=n)
            _gather[D, False](column, lookup, table_valid, values, bits)
            if all_valid:
                bits = List[UInt8]()
            var result = Series(
                per_value.name(), Column[Scalar[D]](values=values^, bits=bits^)
            )
            result._dtype = per_value._dtype
            return result^
    # Strings and other values: a row gather.
    var rows = List[Int](length=n, fill=entries - 1)
    for i in range(n):
        if column._valid(i):
            rows[i] = Int(column._get(i))
    var gathered = take_parallel([table^], rows^, worker_count(n))
    return gathered.pop()
