"""CPU-only diagnostics over completed query answers, never a pass tolerance.

Row alignment requires declared output keys whose uniqueness and exact key
membership are checked on both answers. Otherwise column bags are reported
without claiming that corresponding rows identify the same group.
"""
from std.memory import bitcast
from std.os import getenv
from dataframe import DataFrame, DataType


def clean(value: String) -> String:
    return value.replace("\n", " ").replace("\t", " ")


def environment_keys(name: String) -> List[String]:
    var result = List[String]()
    var value = getenv(name)
    if value:
        for key in value.split("\t"):
            result.append(String(key))
    return result^


def schema_equal(left: DataFrame, right: DataFrame) -> Bool:
    return left.columns() == right.columns() and left.dtypes() == right.dtypes()


@fieldwise_init
struct Alignment(Movable):
    var available: Bool
    var reason: String
    var left: DataFrame
    var right: DataFrame


def align(
    left: DataFrame, right: DataFrame, keys: List[String]
) raises -> Alignment:
    if not schema_equal(left, right) or left.height() != right.height():
        return Alignment(
            False, "schema_or_row_count_differs", left.copy(), right.copy()
        )
    if left.height() <= 1:
        return Alignment(True, "at_most_one_row", left.copy(), right.copy())
    if not len(keys):
        return Alignment(
            False, "no_declared_unique_output_keys", left.copy(), right.copy()
        )
    for name in keys:
        if name not in left.columns():
            return Alignment(
                False, "declared_key_missing", left.copy(), right.copy()
            )
    var a = left.select(keys)
    var b = right.select(keys)
    if (
        a.unique().height() != left.height()
        or b.unique().height() != right.height()
    ):
        return Alignment(
            False, "declared_keys_not_unique", left.copy(), right.copy()
        )
    if not a.sort(keys).equals(b.sort(keys)):
        return Alignment(
            False, "key_membership_differs", left.copy(), right.copy()
        )
    return Alignment(
        True, "verified_unique_keys", left.sort(keys), right.sort(keys)
    )


@fieldwise_init
struct ColumnMetrics(Copyable):
    var mismatches: Int
    var null_mismatches: Int
    var finite_float_mismatches: Int
    var nonfinite_float_mismatches: Int
    var max_ulp: UInt64
    var max_absolute: Float64
    var max_relative: Float64


def _finite(value: Float64) -> Bool:
    return bitcast[DType.uint64](value) & UInt64(0x7FF0000000000000) != UInt64(
        0x7FF0000000000000
    )


def _ordered_word(value: Float64, width: Int) -> UInt64:
    var word = bitcast[DType.uint64](value)
    var sign = UInt64(0x8000000000000000)
    var mask = UInt64.MAX
    if width == 32:
        word = UInt64(bitcast[DType.uint32](Float32(value)))
        sign = UInt64(0x80000000)
        mask = UInt64(0xFFFFFFFF)
    return (~word) & mask if word & sign else word | sign


def column_metrics(
    left: DataFrame, right: DataFrame, name: String
) raises -> ColumnMetrics:
    # Only call after schema, cardinality and row alignment were established.
    var result = ColumnMetrics(0, 0, 0, 0, 0, 0, 0)
    var dtype = left.column(name).dtype()
    for row in range(left.height()):
        var a = left.item(row, name)
        var b = right.item(row, name)
        if a == b:
            continue
        result.mismatches += 1
        if a.is_null() != b.is_null():
            result.null_mismatches += 1
        if a.is_null() or b.is_null() or not dtype.is_float():
            continue
        var x = (
            Float64(a.float32()) if dtype == DataType.FLOAT32 else a.float64()
        )
        var y = (
            Float64(b.float32()) if dtype == DataType.FLOAT32 else b.float64()
        )
        if not _finite(x) or not _finite(y):
            result.nonfinite_float_mismatches += 1
            continue
        result.finite_float_mismatches += 1
        var width = 32 if dtype == DataType.FLOAT32 else 64
        var xword = _ordered_word(x, width)
        var yword = _ordered_word(y, width)
        var distance = xword - yword if xword >= yword else yword - xword
        result.max_ulp = max(result.max_ulp, distance)
        var difference = abs(x - y)
        result.max_absolute = max(result.max_absolute, difference)
        var scale = max(abs(x), abs(y))
        var relative = difference / scale if _finite(difference) else abs(
            x / scale - y / scale
        )
        result.max_relative = max(result.max_relative, relative)
    return result^


def _bool(prefix: String, name: String, value: Bool):
    print("diagnostic\t" + prefix + "." + name + "\t" + String(value))


def _text(prefix: String, name: String, value: String):
    print("diagnostic\t" + prefix + "." + name + "\t" + clean(value))


def emit_comparison(
    left: DataFrame,
    right: DataFrame,
    prefix: String,
    alignment_keys: List[String],
    order_keys: List[String],
) raises:
    var same_schema = schema_equal(left, right)
    var same_rows = left.height() == right.height()
    _bool(prefix, "schema_equal", same_schema)
    _bool(prefix, "row_count_equal", same_rows)
    _bool(prefix, "ordered_equal", left.equals(right))
    if not same_schema or not same_rows:
        _text(
            prefix,
            "classification",
            "schema_difference" if not same_schema else "row_count_difference",
        )
        _text(prefix, "alignment", "unavailable")
        return
    var same_bag = left.sort(left.columns()).equals(right.sort(right.columns()))
    _bool(prefix, "row_multiset_equal", same_bag)
    _text(
        prefix,
        "classification",
        "row_multiset_difference" if not same_bag else "equal" if left.equals(
            right
        ) else "order_only",
    )
    for name in left.columns():
        var a = left.select([name]).sort([name])
        var b = right.select([name]).sort([name])
        print(
            "column_bag\t"
            + prefix
            + "\t"
            + clean(name)
            + "\t"
            + String(a.equals(b))
        )
    var aligned = align(left, right, alignment_keys)
    _text(prefix, "alignment", aligned.reason)
    if aligned.available:
        for name in left.columns():
            var metrics = column_metrics(aligned.left, aligned.right, name)
            var finite = metrics.finite_float_mismatches > 0
            print(
                "column_diagnostic\t"
                + prefix
                + "\t"
                + clean(name)
                + "\t"
                + clean(left.column(name).dtype().name())
                + "\t"
                + String(metrics.mismatches)
                + "\t"
                + String(metrics.null_mismatches)
                + "\t"
                + String(metrics.finite_float_mismatches)
                + "\t"
                + String(metrics.nonfinite_float_mismatches)
                + "\t"
                + (String(metrics.max_ulp) if finite else "unavailable")
                + "\t"
                + (String(metrics.max_absolute) if finite else "unavailable")
                + "\t"
                + (String(metrics.max_relative) if finite else "unavailable")
            )
    var valid_order_keys = len(order_keys) > 0
    for name in order_keys:
        valid_order_keys &= name in left.columns()
    if valid_order_keys:
        _bool(
            prefix,
            "left_output_order_key_ties",
            left.select(order_keys).unique().height() < left.height(),
        )
        _bool(
            prefix,
            "right_output_order_key_ties",
            right.select(order_keys).unique().height() < right.height(),
        )
        _text(
            prefix,
            "ties_cause",
            "unproved; output ties do not prove a LIMIT-boundary cause",
        )
    else:
        _text(prefix, "ties_cause", "unassessed; no declared output sort keys")
