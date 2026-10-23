"""Equal labels are equal categorical group keys (#943).

Polars' categories never repeat a string, so a categorical key there groups by
label. Arrow and Parquet dictionaries may repeat one, and this library keeps
the dictionary it was given (codes round-trip). Grouping therefore replaces
each code by the first code of its label before it numbers the keys.
"""
from .column import Column
from .distinct_statistics import _hash_bytes
from .dtype import CategoricalDictionary
from .series import Series


def canonical_code_map(
    dictionary: CategoricalDictionary,
) -> Optional[List[UInt32]]:
    """For each code, the first code holding the same label (#943). Equal
    labels are equal group keys, as in Polars, whose categories never repeat
    a string. None when no label repeats, which is every dictionary this
    library builds itself; Arrow and Parquet dictionaries may repeat one."""
    var n = len(dictionary)
    var size = 16
    while size < 2 * n:
        size *= 2
    var table = List[Int32](length=size, fill=-1)
    var result = List[UInt32](capacity=n)
    var repeated = False
    for code in range(n):
        var label = dictionary.get(code)
        var slot = Int(_hash_bytes(label.as_bytes())) & (size - 1)
        var found = -1
        while table[slot] >= 0:
            if dictionary.get(Int(table[slot])) == label:
                found = Int(table[slot])
                break
            slot = (slot + 1) & (size - 1)
        if found < 0:
            table[slot] = Int32(code)
            result.append(UInt32(code))
        else:
            result.append(UInt32(found))
            repeated = True
    if not repeated:
        return None
    return result^


def label_map(values: Series) -> Optional[List[UInt32]]:
    """`canonical_code_map` of a categorical key's dictionary; None for any
    other key."""
    if (
        not values.dtype().is_categorical()
        or not values.dtype().has_dictionary()
    ):
        return None
    return canonical_code_map(values.dtype().dictionary()[])


def canonical_labels(values: Series) raises -> Series:
    """A categorical key with every code replaced by the first code of its
    label, so equal labels group together (#943); a copy of `values` when its
    dictionary repeats no label, or when it is not a categorical."""
    var map = label_map(values)
    if not map:
        return values.copy()
    return remap_codes(values, map.value())


def remap_codes(values: Series, map: List[UInt32]) raises -> Series:
    """`values` with each code replaced by `map[code]`, same dictionary."""
    var flat = values.rechunk() if values.is_chunked() else values.copy()
    ref column = flat._data[Column[UInt32]]
    var n = len(column)
    var codes = List[UInt32](length=n, fill=0)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        if column._valid(i):
            var code = Int(column._get(i))
            if code >= len(map):
                raise Error("Categorical code exceeds its dictionary")
            codes[i] = map[code]
            valid[i] = True
    var result = Series(
        values.name(), Column[UInt32](codes^, valid)
    ).with_dtype(values.dtype())
    result._field_metadata = values._field_metadata
    return result^
