"""Dictionary-code join reconciliation experiment; no public categorical dtype.

Usage: dictionary_join_experiment ROWS DISTINCT

`ROWS` is the fact-table size (at most one million). `DISTINCT` is the
dimension dictionary cardinality; use 100 and 100000. The right dictionary
uses a reversed independently assigned code order and reserves one quarter of
its entries for right-only keys. Both sides also contain null keys. This is a
dimension join, deliberately avoiding the quadratic output of joining two
repeated low-cardinality fact tables.

The experiment reports the one-time string encoding/dictionary reconciliation
separately from repeated string-key and reconciled-Int32-key joins. It checks
row order and payload pairs against the string join before timing.

Arrow dictionary import/export is outside this prototype: dataframe/arrow.mojo
currently rejects either non-null Arrow dictionary pointer with "Arrow
dictionary arrays are not supported". A real categorical dtype would need an
Arrow mapping and dictionary lifetime/reconciliation rules as well.
"""
from std.collections import Dict
from std.sys import argv
from std.time import monotonic

from dataframe import Column, DataFrame, Series, StringColumn
from dataframe.hashing import column_codes


comptime REPEATS = 8


def key_name(index: Int) -> String:
    return "key_" + String(index)


def string_side(rows: Int, distinct: Int, left: Bool) raises -> DataFrame:
    """Build one fact or dimension side with different dictionary discovery order."""
    var keys = List[String](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var payload = List[Int64](capacity=rows)
    for row in range(rows):
        # One null is enough on the dimension side; periodic nulls exercise
        # null exclusion on the fact side without suppressing a whole key.
        # The first occurrence of every left dictionary value must be valid;
        # the following row is a guaranteed null. Later periodic nulls retain
        # that coverage without changing the encoded cardinality.
        var is_null = row == 0 if not left else row == distinct or (
            row > distinct and row % 97 == 0
        )
        valid.append(not is_null)
        if is_null:
            keys.append("")
        elif left:
            keys.append(key_name(row % distinct))
        else:
            # Shared strings appear in reverse first-occurrence order. The
            # final quarter exists only on the right, so remapping must not
            # accidentally make its local code collide with a left code.
            var shared = distinct * 3 // 4
            var code = row - 1
            keys.append(
                key_name(shared - 1 - code) if code
                < shared else key_name(distinct + code - shared)
            )
        payload.append(Int64(row))
    return DataFrame(
        [
            Series("key", StringColumn(keys^, valid^)),
            Series(
                "left_row" if left else "right_row", Column[Int64](payload^)
            ),
        ]
    )


def dictionary_for(
    key: Series, codes: List[Int], nulls: List[Bool], count: Int
) raises -> List[String]:
    """Materialize code -> string once, using each first representative."""
    var dictionary = List[String](length=count, fill="")
    var filled = List[Bool](length=count, fill=False)
    ref strings = key._data[StringColumn]
    for row in range(len(codes)):
        var code = codes[row]
        if not nulls[row] and not filled[code]:
            dictionary[code] = String(strings._get(row))
            filled[code] = True
    for code in range(count):
        if not filled[code]:
            raise Error("dictionary code has no representative")
    return dictionary^


def reconcile(left: List[String], right: List[String]) -> List[Int]:
    """Map right-local code to left-local code, or -1 for a right-only key."""
    var lookup = Dict[String, Int]()
    for code in range(len(left)):
        lookup[left[code]] = code
    var result = List[Int](capacity=len(right))
    for value in right:
        result.append(lookup.get(value, -1))
    return result^


def encoded_side(
    codes: List[Int],
    nulls: List[Bool],
    remap: List[Int],
    left_codes: Int,
    left: Bool,
) raises -> DataFrame:
    """Build Int32 join keys; unmapped right keys get non-colliding codes."""
    var values = List[Int32](capacity=len(codes))
    var valid = List[Bool](capacity=len(codes))
    for row in range(len(codes)):
        if nulls[row]:
            values.append(0)
            valid.append(False)
        elif left:
            values.append(Int32(codes[row]))
            valid.append(True)
        else:
            var mapped = remap[codes[row]]
            # Values outside left's id space remain valid unmatched keys.
            values.append(
                Int32(mapped if mapped >= 0 else left_codes + codes[row])
            )
            valid.append(True)
    var payload = List[Int64](capacity=len(codes))
    for row in range(len(codes)):
        payload.append(Int64(row))
    return DataFrame(
        [
            Series("code", Column[Int32](values^, valid^)),
            Series(
                "left_row" if left else "right_row", Column[Int64](payload^)
            ),
        ]
    )


def check_same_pairs(strings: DataFrame, codes: DataFrame) raises:
    if strings.height() != codes.height():
        raise Error("string and reconciled-code joins have different heights")
    for name in ["left_row", "right_row"]:
        if not strings.column(name).equals(codes.column(name)):
            raise Error("string and reconciled-code joins differ in " + name)


def best_join(left: DataFrame, right: DataFrame, key: String) raises -> Int:
    var best = Int.MAX
    for iteration in range(REPEATS):
        var started = monotonic()
        var result = left.join(right, key)
        var elapsed = monotonic() - started
        if iteration > 0:
            best = min(best, elapsed)
        # Keep the result live through the measured interval and guard against
        # a future join shortcut that silently changes the output.
        if result.height() < 0:
            raise Error("unreachable")
    return best


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: dictionary_join_experiment ROWS DISTINCT")
    var rows = Int(String(args[1]))
    var distinct = Int(String(args[2]))
    if rows < 1 or rows > 1_000_000:
        raise Error("ROWS must be in 1..1000000")
    if distinct < 4 or distinct >= rows:
        raise Error("DISTINCT must be in 4..ROWS-1")

    var left_strings = string_side(rows, distinct, True)
    # A dimension table has one row for every independently coded value plus
    # one null row. This keeps output O(ROWS) at cardinality 100 as well.
    var right_strings = string_side(distinct + 1, distinct, False)

    var started = monotonic()
    var left_key = left_strings.column("key")
    var right_key = right_strings.column("key")
    var left_codes = List[Int](length=rows, fill=0)
    var left_nulls = List[Bool](length=rows, fill=False)
    var right_codes = List[Int](length=distinct + 1, fill=0)
    var right_nulls = List[Bool](length=distinct + 1, fill=False)
    var left_count = column_codes(left_key, left_codes, left_nulls)
    var right_count = column_codes(right_key, right_codes, right_nulls)
    var left_dictionary = dictionary_for(
        left_key, left_codes, left_nulls, left_count
    )
    var right_dictionary = dictionary_for(
        right_key, right_codes, right_nulls, right_count
    )
    var right_to_left = reconcile(left_dictionary, right_dictionary)
    var left_encoded = encoded_side(
        left_codes, left_nulls, List[Int](), left_count, True
    )
    var right_encoded = encoded_side(
        right_codes,
        right_nulls,
        right_to_left,
        left_count,
        False,
    )
    var encode_remap_ns = monotonic() - started

    var remapped = 0
    var unmatched = 0
    for code in right_to_left:
        if code >= 0:
            remapped += 1
        else:
            unmatched += 1
    if remapped != distinct * 3 // 4 or unmatched != distinct - remapped:
        raise Error("unexpected dictionary reconciliation")
    var string_join = left_strings.join(right_strings, "key")
    var code_join = left_encoded.join(right_encoded, "code")
    check_same_pairs(string_join, code_join)

    print("rows", rows, sep=",")
    print("distinct", distinct, sep=",")
    print("left_dictionary", left_count, sep=",")
    print("right_dictionary", right_count, sep=",")
    print("right_remapped", remapped, sep=",")
    print("right_unmatched", unmatched, sep=",")
    print("join_rows", string_join.height(), sep=",")
    print("encode_remap_ns", encode_remap_ns, sep=",")
    print(
        "string_join_ns", best_join(left_strings, right_strings, "key"), sep=","
    )
    print(
        "reconciled_int32_join_ns",
        best_join(left_encoded, right_encoded, "code"),
        sep=",",
    )
