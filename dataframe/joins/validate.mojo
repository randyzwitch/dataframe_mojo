"""Check a join's `validate` option: that the named inputs have unique keys.

A hash join's build side finds repeated keys while inserting its rows into
the table (a row whose keys equal a chain's joins that chain), so checking
that side costs nothing more. The other side is checked with a table of its
own keys, built only when the option asks for it. Rows with a null key never
match, so they never count as repeats.
"""
from dataframe.frame import DataFrame
from dataframe.join_type import (
    join_validate_checks,
    join_validate_name,
    join_validate_side_name,
)
from dataframe.joins.hash_table import JoinHashTable
from dataframe.series import Series
from std.collections import Optional


def keys_repeat(keys: List[Series]) raises -> Bool:
    """Whether two rows of `keys` (columns of one length) with no null key
    have equal keys."""
    if len(keys) == 0 or len(keys[0]) < 2:
        return False
    var table = JoinHashTable(keys, keys_only=True, propagate=False)
    var known = table.has_duplicate_keys()
    return Bool(known) and known.value()


def _repeat_error(
    keys: List[Series], names: List[String], code: Int, right_side: Bool
) raises -> Error:
    """The error for a side whose keys repeat, naming one repeated key."""
    var columns = List[Series](capacity=len(keys))
    var hidden = List[String](capacity=len(keys))
    for k in range(len(keys)):
        hidden.append("k" + String(k))
        columns.append(keys[k].renamed(hidden[k]))
    var frame = DataFrame(columns^).drop_nulls(hidden)
    var text = String("")
    var counted = frame._key_counts(hidden)
    ref ids = counted[0]
    ref counts = counted[1]
    for i in range(frame.height()):
        if counts[ids[i]] > 1:
            var row = frame.row(i)
            for k in range(len(names)):
                if k > 0:
                    text += ", "
                text += names[k] + "=" + String(row[k])
            break
    return Error(
        "join keys did not fulfill "
        + join_validate_name(code)
        + " validation: the "
        + join_validate_side_name(code, right_side)
        + " input has key ("
        + text
        + ") more than once"
    )


def key_columns(frame: DataFrame, names: List[String]) raises -> List[Series]:
    """The key columns `names` of `frame`, in order."""
    var keys = List[Series](capacity=len(names))
    for name in names:
        keys.append(frame.column(name))
    return keys^


def check_side_unique(
    keys: List[Series],
    names: List[String],
    code: Int,
    right_side: Bool,
    table: Optional[JoinHashTable] = None,
) raises:
    """Raise when the bits `code` ask for unique keys on the plan's right
    input (`right_side`) or its left, and that side's keys repeat. `keys`
    are its key columns (empty when only `table` holds its rows), named
    `names` as the join was written. `table`: a hash table over these keys,
    whose inserts already found whether they repeat."""
    if not join_validate_checks(code, right_side):
        return
    var columns = keys.copy()
    if len(columns) == 0 and table:
        for k in range(len(names)):
            columns.append(table.value().key_column(k, names[k]))
    var repeat = Optional[Bool]()
    if table:
        repeat = table.value().has_duplicate_keys()
    if not repeat:
        repeat = keys_repeat(columns)
    if repeat.value():
        raise _repeat_error(columns, names, code, right_side)
