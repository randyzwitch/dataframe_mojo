"""Dictionary-code grouping experiment, not a new public categorical dtype.

Usage: dictionary_experiment ROWS DISTINCT
Reports encoding separately from repeated grouping on string and Int32 keys.
Decodes every code to verify row correspondence before timing grouping.
"""
from std.sys import argv
from std.time import monotonic
from dataframe import Column, DataFrame, Series, StringColumn, col
from dataframe.hashing import column_codes


def group_time(frame: DataFrame, expected: Int) raises -> Int:
    var best = Int.MAX
    for iteration in range(8):
        var start = monotonic()
        var result = frame.group_by("key").agg(col("x").sum().alias("sum"))
        var elapsed = monotonic() - start
        if result.height() != expected:
            raise Error("wrong group count")
        if iteration > 0:
            best = min(best, elapsed)
    return best


def main() raises:
    var args = argv()
    var rows = Int(String(args[1]))
    var distinct = Int(String(args[2]))
    var dictionary = List[String](capacity=distinct)
    for k in range(distinct):
        var suffix = String(k)
        var text = String("")
        for _ in range(max(0, 20 - suffix.byte_length())):
            text += "k"
        dictionary.append(text + suffix)
    var strings = List[String](capacity=rows)
    var values = List[Int64](capacity=rows)
    for i in range(rows):
        strings.append(dictionary[i % distinct])
        values.append(Int64(i % 100))
    var key = Series("key", StringColumn(strings^))
    var x = Series("x", Column[Int64](values^))
    var frame = DataFrame([key.copy(), x.copy()])
    var started = monotonic()
    var codes = List[Int](length=rows, fill=0)
    var nulls = List[Bool](length=rows, fill=False)
    var found = column_codes(key, codes, nulls)
    var packed = List[Int32](capacity=rows)
    for code in codes:
        packed.append(Int32(code))
    var encoded = DataFrame([Series("key", Column[Int32](packed^)), x.copy()])
    var encode_ns = monotonic() - started
    if found != distinct:
        raise Error("dictionary cardinality mismatch")
    for i in range(rows):
        if nulls[i] or dictionary[codes[i]] != dictionary[i % distinct]:
            raise Error("dictionary decode mismatch")
    # Compare corresponding group sums in first-occurrence order; integer
    # codes alone would not prove that a different grouping is equivalent.
    var strings_result = frame.group_by("key", maintain_order=True).agg(
        col("x").sum().alias("sum")
    )
    var codes_result = encoded.group_by("key", maintain_order=True).agg(
        col("x").sum().alias("sum")
    )
    if not strings_result.column("sum").equals(codes_result.column("sum")):
        raise Error("group sums differ")
    print("encode_ns", encode_ns, sep=",")
    print("string_group_ns", group_time(frame, distinct), sep=",")
    print("code_group_ns", group_time(encoded, distinct), sep=",")
    print(
        "string_payload_bytes",
        rows * 20 + (rows + 1) * 8 + (rows + 7) // 8,
        sep=",",
    )
    print(
        "dictionary_payload_bytes",
        rows * 4
        + distinct * 20
        + (distinct + 1) * 8
        + (rows + 7) // 8
        + (distinct + 7) // 8,
        sep=",",
    )
