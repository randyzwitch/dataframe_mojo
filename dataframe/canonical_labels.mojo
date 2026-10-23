"""Equal labels are equal categorical group keys (#943).

Polars' categories never repeat a string, so a categorical key there groups by
label. Arrow and Parquet dictionaries may repeat one, and this library keeps
the dictionary it was given (codes round-trip). Grouping therefore replaces
each code by the first code of its label before it numbers the keys.
"""
from .column import Column
from .dtype import CategoricalDictionary
from .series import Series


def canonical_code_map(
    dictionary: CategoricalDictionary,
) -> Optional[List[UInt32]]:
    """For each code, the first code holding the same label (#943); None
    when no label repeats. Read from the dictionary, where
    `DataType.categorical` computed it once."""
    if len(dictionary.canonical) == 0:
        return None
    return dictionary.canonical.copy()


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
