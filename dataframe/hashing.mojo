"""Composite row keys: dense ids for grouping, joins, and distinct rows.

Each key column is first mapped to dense per-column codes. Codes are then
combined one column at a time and re-densified, so ids never depend on
string concatenation or hash collisions: equality is exact per column.
Float64 keys treat every NaN as one value and -0.0 as equal to 0.0.
"""
from std.collections import Dict
from .aggregate import float_key
from .column import Column
from .string_column import StringColumn, StringBuilder
from .series import Series


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
    if series._data.isa[Column[Int64]]():
        return _codes_by_value(series._data[Column[Int64]], codes, nulls)
    if series._data.isa[Column[Float64]]():
        ref column = series._data[Column[Float64]]
        var lookup = Dict[UInt64, Int]()
        for i in range(len(column)):
            if not column._valid(i):
                nulls[i] = True
                continue
            var key = float_key(column._get(i))
            var code = lookup.get(key, -1)
            if code < 0:
                code = len(lookup)
                lookup[key] = code
            codes[i] = code
        return len(lookup)
    if series._data.isa[Column[Bool]]():
        ref column = series._data[Column[Bool]]
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


def encode_rows(keys: List[Series], nulls_equal: Bool) raises -> RowKeys:
    """Assign dense ids to distinct key rows, in first-occurrence order.

    With nulls_equal=True a null is an ordinary key value, equal only to
    other nulls in the same column. Otherwise any null key yields id -1.
    """
    if len(keys) == 0:
        raise Error("Row keys require at least one column")
    var n = len(keys[0])
    for key in keys:
        if len(key) != n:
            raise Error("Key columns must have equal lengths")
    if n >= 2147483647:
        raise Error("Row key encoding supports fewer than 2**31 rows")
    var ids = List[Int](length=n, fill=0)
    var excluded = List[Bool](length=n, fill=False)
    var representatives = List[Int]()
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
