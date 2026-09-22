"""Isolated #106 code-reuse experiment; no categorical public API.

Usage: dictionary_reuse_experiment ROWS DISTINCT

The code column is a one-time lexical remapping of a nullable String key.  It
therefore has the correct String sort order; sorting first-occurrence ids would
be incorrect.  Before timing, every operation is compared with the String
operation: stable sort row order, first-occurrence unique rows, n_unique,
value-count order/counts, and nullable elementwise equality.
"""
from std.collections import Dict
from std.sys import argv
from std.time import monotonic

from dataframe import Column, DataFrame, Series, StringColumn


comptime REPEATS = 5


def key_value(code: Int) -> String:
    # Same width makes lexical order differ materially from encounter order.
    var suffix = String(code)
    var value = "key_"
    for _ in range(8 - suffix.byte_length()):
        value += "0"
    return value + suffix


def make_key(rows: Int, distinct: Int, phase: Int = 0) raises -> Series:
    var values = List[String](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for row in range(rows):
        # Seed every dictionary value before introducing nulls, so both
        # nullable input columns are guaranteed to use the same codebook.
        var is_null = row >= distinct and (
            row % 127 == 0 or (phase != 0 and row % 113 == 0)
        )
        valid.append(not is_null)
        var code = (row * 37 + 11 + phase) % distinct
        values.append(key_value(code) if not is_null else "")
    return Series("key", StringColumn(values^, valid^))


def lexical_codebook(key: Series) raises -> Dict[String, Int32]:
    # This is one-time dictionary construction/remapping, intentionally
    # excluded from repeated operation timings below.
    var ordered = key.unique(maintain_order=True).sort()
    var result = Dict[String, Int32]()
    var rank = 0
    for row in range(len(ordered)):
        if not ordered[row].is_null():
            result[String(ordered[row])] = Int32(rank)
            rank += 1
    return result^


def encode_with(key: Series, codes: Dict[String, Int32]) raises -> Series:
    ref strings = key._data[StringColumn]
    var values = List[Int32](capacity=len(key))
    var valid = List[Bool](capacity=len(key))
    for row in range(len(key)):
        var is_valid = not strings.is_null(row)
        valid.append(is_valid)
        values.append(codes[String(strings._get(row))] if is_valid else 0)
    return Series("code", Column[Int32](values^, valid^))


def rows(rows: Int) -> Series:
    var values = List[Int64](capacity=rows)
    for row in range(rows):
        values.append(Int64(row))
    return Series("row", Column[Int64](values^))


def require(equal: Bool, message: String) raises:
    if not equal:
        raise Error(message)


def verify(key: Series, other: Series, code: Series, other_code: Series) raises:
    var payload = rows(len(key))
    var strings = DataFrame([key.copy(), payload.copy()])
    var encoded = DataFrame([code.copy(), payload.copy()])

    # Stable row order proves the numeric rank remap preserves lexical String
    # order, including duplicate ties and null-last placement.
    var string_sorted = strings.sort(["key"])
    var code_sorted = encoded.sort(["code"])
    require(
        string_sorted.column("row").equals(code_sorted.column("row")),
        "lexical code sort differs from nullable String sort",
    )

    # First surviving source rows prove code uniqueness is not merely the same
    # cardinality under a different key ordering.
    var string_unique = strings.unique(
        ["key"], keep="first", maintain_order=True
    )
    var code_unique = encoded.unique(
        ["code"], keep="first", maintain_order=True
    )
    require(
        string_unique.column("row").equals(code_unique.column("row")),
        "code unique differs from nullable String unique",
    )
    require(
        key.n_unique() == code.n_unique(),
        "code n_unique differs from String",
    )

    # Both value_counts calls retain first-occurrence tie order.  Counts and
    # representative-row order must agree after using the same codebook.
    var string_counts = key.value_counts(sort=True)
    var code_counts = code.value_counts(sort=True)
    require(
        string_counts.column("count").equals(code_counts.column("count")),
        "code value_counts differs from String value_counts",
    )
    require(
        string_counts.height() == code_counts.height(),
        "code value_counts has wrong cardinality",
    )

    require(
        key.eq(other).equals(code.eq(other_code)),
        "code equality differs from nullable String equality",
    )


def best_sort(frame: DataFrame, key: String) raises -> Int:
    var best = Int.MAX
    for iteration in range(REPEATS):
        var start = monotonic()
        var result = frame.sort([key])
        var elapsed = monotonic() - start
        if result.height() != frame.height():
            raise Error("sort lost rows")
        if iteration > 0:
            best = min(best, elapsed)
    return best


def best_unique(frame: DataFrame, key: String, expected: Int) raises -> Int:
    var best = Int.MAX
    for iteration in range(REPEATS):
        var start = monotonic()
        var result = frame.unique([key], keep="first", maintain_order=True)
        var elapsed = monotonic() - start
        if result.height() != expected:
            raise Error("unique cardinality changed")
        if iteration > 0:
            best = min(best, elapsed)
    return best


def best_n_unique(key: Series, expected: Int) raises -> Int:
    var best = Int.MAX
    for iteration in range(REPEATS):
        var start = monotonic()
        var count = key.n_unique()
        var elapsed = monotonic() - start
        if count != expected:
            raise Error("n_unique cardinality changed")
        if iteration > 0:
            best = min(best, elapsed)
    return best


def best_value_counts(key: Series, expected: Int) raises -> Int:
    var best = Int.MAX
    for iteration in range(REPEATS):
        var start = monotonic()
        var result = key.value_counts(sort=True)
        var elapsed = monotonic() - start
        if result.height() != expected:
            raise Error("value_counts cardinality changed")
        if iteration > 0:
            best = min(best, elapsed)
    return best


def best_eq(left: Series, right: Series) raises -> Int:
    var best = Int.MAX
    for iteration in range(REPEATS):
        var start = monotonic()
        var result = left.eq(right)
        var elapsed = monotonic() - start
        if len(result) != len(left):
            raise Error("equality lost rows")
        if iteration > 0:
            best = min(best, elapsed)
    return best


def main() raises:
    var args = argv()
    if len(args) == 2:
        if String(args[1]) != "check":
            raise Error("usage: dictionary_reuse_experiment ROWS DISTINCT")
        # Correctness-only mode deliberately performs no measured operation.
        var checked_key = make_key(10_000, 100)
        var checked_other = make_key(10_000, 100, 1)
        var checked_book = lexical_codebook(checked_key)
        verify(
            checked_key,
            checked_other,
            encode_with(checked_key, checked_book),
            encode_with(checked_other, checked_book),
        )
        print("dictionary_reuse_check,pass")
        return
    if len(args) != 3:
        raise Error("usage: dictionary_reuse_experiment ROWS DISTINCT")
    var count = Int(String(args[1]))
    var distinct = Int(String(args[2]))
    if count < 1 or count > 1_000_000 or distinct < 2 or distinct > count:
        raise Error("ROWS must be 1..1000000 and DISTINCT must be 2..ROWS")

    var key = make_key(count, distinct)
    var other = make_key(count, distinct, 1)
    var started = monotonic()
    var codebook = lexical_codebook(key)
    var code = encode_with(key, codebook)
    var other_code = encode_with(other, codebook)
    var encode_remap_ns = monotonic() - started
    verify(key, other, code, other_code)

    var payload = rows(count)
    var string_frame = DataFrame([key.copy(), payload.copy()])
    var code_frame = DataFrame([code.copy(), payload.copy()])
    var expected = key.n_unique()
    print("rows", count, sep=",")
    print("distinct", distinct, sep=",")
    print("encode_lexical_remap_ns", encode_remap_ns, sep=",")
    print("string_sort_ns", best_sort(string_frame, "key"), sep=",")
    print("code_sort_ns", best_sort(code_frame, "code"), sep=",")
    print(
        "string_unique_ns", best_unique(string_frame, "key", expected), sep=","
    )
    print("code_unique_ns", best_unique(code_frame, "code", expected), sep=",")
    print("string_n_unique_ns", best_n_unique(key, expected), sep=",")
    print("code_n_unique_ns", best_n_unique(code, expected), sep=",")
    print("string_value_counts_ns", best_value_counts(key, expected), sep=",")
    print("code_value_counts_ns", best_value_counts(code, expected), sep=",")
    print("string_eq_ns", best_eq(key, other), sep=",")
    print("code_eq_ns", best_eq(code, other_code), sep=",")
