"""`n_unique` counted by hash partition (#336), overall and per group.

Every result is compared with sets built here from the values themselves:
nulls count once, every NaN is one value, -0.0 equals 0.0. Sizes reach the
parallel passes and many partitions, values repeat enough for the workers'
duplicate filters to drop most rows, strings come in offset and view
storage and share long prefixes, and groups are skewed so that one holds
most rows. Grouped counts take both group_by paths: per-range sets for
values that repeat, partitions for values that do not.
"""
from std.collections import Dict
from std.math import isnan
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    concat,
    lit,
)
from dataframe.string_view import StringViewBuilder


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def canonical(column: Series, row: Int) raises -> String:
    """The equality key n_unique uses, as text; nulls are handled apart."""
    var value = column.get(row)
    if column.dtype().is_float():
        var x = (
            Float64(value.float32()) if column.dtype()
            == DataType.FLOAT32 else value.float64()
        )
        if isnan(x):
            return "nan"
        if x == 0:
            return "0"
    return String(value)


def expected_counts(keys: Series, values: Series) raises -> Dict[String, Int]:
    """Distinct values per group key (keyed by the key's text)."""
    var sets = Dict[String, Dict[String, Bool]]()
    for row in range(len(values)):
        var group = String(keys.get(row))
        if group not in sets:
            sets[group] = Dict[String, Bool]()
        var key = "<null>" if values.get(row).is_null() else "=" + canonical(
            values, row
        )
        sets[group][key] = True
    var counts = Dict[String, Int]()
    for item in sets.items():
        counts[item.key] = len(item.value)
    return counts^


def check(values: Series, label: String) raises:
    """Overall and grouped n_unique of `values` against sets."""
    var n = len(values)
    var rng = Lcg(UInt64(n) + 17)
    # A skewed key: most rows in one group, the rest over a few dozen.
    var groups = List[Int64](capacity=n)
    for _ in range(n):
        groups.append(0 if rng.next(10) < 7 else Int64(rng.next(40)))
    var frame = DataFrame([Series("g", Column[Int64](groups^)), values.copy()])
    var name = values.name()
    var whole = expected_counts(
        Series("all", Column[Int64](List[Int64](length=n, fill=0))), values
    )
    var total = whole.get("0", 0)
    assert_equal(
        frame.select(col(name).n_unique()).item().int64(),
        Int64(total),
        label,
    )
    var expected = expected_counts(frame.column("g"), values)
    var grouped = frame.group_by("g").agg([col(name).n_unique().alias("u")])
    assert_equal(grouped.height(), len(expected), label)
    for row in range(grouped.height()):
        var key = String(grouped.column("g").get(row))
        assert_equal(
            grouped.column("u").get(row).int64(),
            Int64(expected[key]),
            label + " group " + key,
        )


def numbers[
    D: DType
](rows: Int, distinct: Int, nulls: Bool, seed: UInt64) raises -> Series:
    var rng = Lcg(seed)
    var values = List[Scalar[D]](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for _ in range(rows):
        comptime if D.is_floating_point():
            var pick = rng.next(distinct + 3)
            var nan = Scalar[D](0) / Scalar[D](0)
            values.append(
                nan if pick
                == 0 else (
                    Scalar[D](-0.0) if pick == 1 else Scalar[D](pick) / 4
                )
            )
        else:
            values.append(Scalar[D](rng.next(distinct)))
        valid.append(not nulls or rng.next(9) != 0)
    return Series("v", Column[Scalar[D]](values^, valid))


def test_numbers_every_width() raises:
    for rows in [0, 1, 50, 3000, 200_000]:
        for distinct in [1, 7, 5000]:
            for nulls in [False, True]:
                var label = String(rows) + "/" + String(distinct)
                check(
                    numbers[DType.int8](rows, min(distinct, 100), nulls, 1),
                    label,
                )
                check(numbers[DType.int16](rows, distinct, nulls, 2), label)
                check(numbers[DType.int32](rows, distinct, nulls, 3), label)
                check(numbers[DType.int64](rows, distinct, nulls, 4), label)
                check(
                    numbers[DType.uint8](rows, min(distinct, 200), nulls, 5),
                    label,
                )
                check(numbers[DType.uint32](rows, distinct, nulls, 6), label)
                check(numbers[DType.uint64](rows, distinct, nulls, 7), label)
                check(numbers[DType.float32](rows, distinct, nulls, 8), label)
                check(numbers[DType.float64](rows, distinct, nulls, 9), label)


def test_all_null_bool_and_temporal() raises:
    check(
        Series(
            "v",
            Column[Int64](
                List[Int64](length=5000, fill=0),
                List[Bool](length=5000, fill=False),
            ),
        ),
        "all null",
    )
    var rng = Lcg(3)
    var flags = List[Bool]()
    var valid = List[Bool]()
    for _ in range(100_000):
        flags.append(rng.next(2) == 0)
        valid.append(rng.next(5) != 0)
    check(Series("v", Column[Bool](flags^, valid^)), "bool")
    var days = numbers[DType.int64](100_000, 900, True, 11)
    var frame = DataFrame([days^])
    check(frame.select_exprs([col("v").cast("date")]).column("v"), "date")
    check(
        frame.select_exprs([col("v").cast("datetime[us]")]).column("v"),
        "datetime",
    )


def strings(rows: Int, distinct: Int, seed: UInt64) raises -> Series:
    """Offset storage, with long values that differ only near their end."""
    var rng = Lcg(seed)
    var values = List[String](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for _ in range(rows):
        var pick = rng.next(distinct)
        values.append(
            "" if pick
            == 0 else (
                "https://example.org/a/very/long/shared/prefix/"
                + String(pick) if pick % 2
                == 0 else String(pick)
            )
        )
        valid.append(rng.next(11) != 0)
    return Series("v", StringColumn(values, valid))


def test_strings_in_both_storages_and_chunks() raises:
    for rows in [10, 5000, 150_000]:
        for distinct in [3, 400, 100_000]:
            check(strings(rows, distinct, UInt64(rows + distinct)), "strings")
    # View storage, including inline and out-of-line values, in chunks.
    var first = StringViewBuilder()
    var second = StringViewBuilder()
    var rng = Lcg(5)
    for i in range(90_000):
        var pick = rng.next(3000)
        var text = String(
            pick
        ) if pick % 3 else "a long value past twelve " + String(pick)
        if i % 13 == 0:
            first.append_null() if i < 45_000 else second.append_null()
        elif i < 45_000:
            first.append(StringSlice(text))
        else:
            second.append(StringSlice(text))
    var views = Series._from_chunks(
        [
            Series("v", StringColumn(first^.finish())),
            Series("v", StringColumn(second^.finish())),
            strings(30_000, 3000, 9),
        ]
    )
    assert_true(views.is_chunked())
    check(views, "views")
    check(views.rechunk(), "views rechunked")
    # Binary shares the string layout.
    var frame = DataFrame([strings(20_000, 500, 21)])
    check(frame.select_exprs([col("v").cast("binary")]).column("v"), "binary")


def test_expression_inputs_and_chunked_numbers() raises:
    var a = numbers[DType.int64](120_000, 70_000, True, 31)
    var b = numbers[DType.int64](80_000, 70_000, False, 32)
    var stacked = concat([DataFrame([a^]), DataFrame([b^])])
    assert_true(stacked.column("v").is_chunked())
    check(stacked.column("v"), "chunked")
    var shifted = stacked.select_exprs(
        [(col("v") % lit(Int64(1000))).alias("v")]
    )
    assert_equal(
        stacked.select((col("v") % lit(Int64(1000))).n_unique()).item().int64(),
        shifted.select(col("v").n_unique()).item().int64(),
    )


def test_lazy_streaming_matches_eager() raises:
    var values = numbers[DType.int32](150_000, 20_000, True, 41)
    var rng = Lcg(42)
    var keys = List[Int64]()
    for _ in range(len(values)):
        keys.append(Int64(rng.next(5)))
    var frame = DataFrame([Series("g", Column[Int64](keys^)), values^])
    var eager = frame.group_by("g", maintain_order=True).agg(
        [col("v").n_unique().alias("u")]
    )
    var lazy = (
        frame.lazy()
        .group_by("g", maintain_order=True)
        .agg([col("v").n_unique().alias("u")])
        .collect(batch_size=4096)
    )
    assert_true(lazy.equals(eager))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
