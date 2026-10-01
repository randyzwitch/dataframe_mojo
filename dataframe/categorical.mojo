"""Dictionary-encoded strings: the Categorical type (#106).

A categorical column stores one UInt32 code per row, in an ordinary
`Column[UInt32]`, and its DataType carries the dictionary of distinct values
the codes index, as temporal types ride on Int64 storage. This is Arrow's
dictionary layout (`dictionary<uint32, large_utf8>`), so it exports and
imports without conversion. Operations that only need equality (grouping,
joins, unique, value counts) work on the codes as integers; sorting orders
by the values through a rank per code; anything reading the text decodes
it here.

Two categoricals index the same values only if their dictionaries match.
Joining or concatenating columns with different dictionaries first moves
them onto the union of both (`unify`), in which the first dictionary's codes
are unchanged.
"""
from std.collections import Dict

from .column import Column
from .dtype import CategoricalDictionary, DataType
from .hashing import encode_string_rows_parallel
from .parallel import worker_count
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
