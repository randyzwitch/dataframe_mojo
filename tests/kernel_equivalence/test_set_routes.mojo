"""Whole-row parallel deduplication preserves every scalar equality class."""
# contract-group: table.sets_reshape
# contract-group: kernel.order_keys
from std.math import isnan
from std.testing import assert_equal, assert_true
from dataframe import BoolColumn, Column, DataFrame, DataType, Series
from dataframe.parallel import worker_count
from dataframe.partition import low_cardinality
from tests.support.lifecycle_jobs import WorkerSetting


def set_integer_class(row: Int) -> Int:
    return 17 if row % 11 == 2 else row % 17


def set_float_class(row: Int) -> Int:
    var pick = row % 7
    return 0 if pick < 2 else pick - 1


def set_text_class(row: Int) -> Int:
    return 4 if row % 13 == 3 else row % 4


def set_literal_text(code: Int) -> String:
    var words: List[String] = ["", "a\0", "a long external string é", "é"]
    return words[code].copy()


def set_signature(i: Int, f: Int, s: Int, b: Int) -> Int:
    return ((i * 6 + f) * 5 + s) * 3 + b


def test_parallel_whole_row_unique_no_aggregate_route_preserves_complete_null_nan_zero_text_classes() raises:
    var setting = WorkerSetting(4)
    var n = 196609
    assert_true(worker_count(n) > 1)
    var ints = List[Int64]()
    var iv = List[Bool]()
    var floats = List[Float64]()
    var fv = List[Bool]()
    var words = List[String]()
    var sv = List[Bool]()
    var bools = List[Bool]()
    var bv = List[Bool]()
    var expected = List[Bool](length=1620, fill=False)
    for row in range(n + 7):
        var ic = set_integer_class(row)
        var fc = set_float_class(row)
        var sc = set_text_class(row)
        var bc = row % 3
        ints.append(Int64(row % 17) if ic < 17 else Int64.MAX)
        iv.append(ic < 17)
        floats.append(
            (-Float64(0) if row % 7 == 0 else Float64(0)) if fc
            == 0 else Float64(1) if fc
            == 1 else Float64(-1) if fc
            == 2 else Float64(0) / Float64(0) if fc
            == 3 else Float64(1) / Float64(0) if fc
            == 4 else Float64.MAX
        )
        fv.append(fc < 5)
        words.append(set_literal_text(row % 4))
        sv.append(sc < 4)
        bools.append(bc != 0)
        bv.append(bc < 2)
        if row >= 7:
            expected[set_signature(ic, fc, sc, bc)] = True
    var source = DataFrame(
        [
            Series("i", Column[Int64](ints^, iv^)).slice(7, n),
            Series("f", Column[Float64](floats^, fv^)).slice(7, n),
            Series("s", Column[String](words^, sv^)).slice(7, n),
            Series("b", Column[Bool](bools^, bv^)).slice(7, n),
        ]
    )
    for chunked in [False, True]:
        var frame = source.copy()
        if chunked:
            var columns = List[Series]()
            for c in range(4):
                var value = source.column(source.columns()[c])
                var cut = 65535 + c
                columns.append(
                    Series._from_chunks(
                        [value.slice(0, cut), value.slice(cut, n - cut)]
                    )
                )
            frame = DataFrame(columns^)
        for explicit_subset in [False, True]:
            # All columns + keep any + unordered + multiple workers are the
            # exact public dispatch condition to group-by with no aggregates.
            var actual = frame.unique(
                frame.columns() if explicit_subset else List[String](),
                keep="any",
                maintain_order=False,
            )
            assert_equal(actual.columns(), List[String](["i", "f", "s", "b"]))
            var dtypes: List[DataType] = [
                DataType.INT64,
                DataType.FLOAT64,
                DataType.STRING,
                DataType.BOOL,
            ]
            for index in range(4):
                assert_true(
                    actual.column(actual.columns()[index]).dtype()
                    == dtypes[index]
                )
            var seen = List[Bool](length=1620, fill=False)
            var count = 0
            for row in range(actual.height()):
                var i = actual.item(row, "i")
                var f = actual.item(row, "f")
                var s = actual.item(row, "s")
                var b = actual.item(row, "b")
                var ic = 17 if i.is_null() else Int(i.int64())
                assert_true(ic >= 0 and ic <= 17)
                var fc = 5
                if not f.is_null():
                    var value = f.float64()
                    fc = (
                        3 if isnan(value) else 0 if value
                        == 0 else 1 if value
                        == 1 else 2 if value
                        == -1 else 4 if value
                        == Float64(1) / Float64(0) else -1
                    )
                assert_true(fc >= 0)
                var sc = 4
                if not s.is_null():
                    sc = -1
                    for code in range(4):
                        if s.string() == set_literal_text(code):
                            sc = code
                assert_true(sc >= 0)
                var bc = 2 if b.is_null() else Int(b.bool())
                var signature = set_signature(ic, fc, sc, bc)
                assert_true(
                    expected[signature],
                    "deduplication invented a composite row",
                )
                assert_true(
                    not seen[signature], "deduplication repeated a complete row"
                )
                seen[signature] = True
            for code in range(1620):
                assert_equal(seen[code], expected[code])
                count += Int(expected[code])
            assert_equal(actual.height(), count)
    _ = setting^


def test_parallel_subset_unique_keeps_input_order_and_exact_key_counts() raises:
    # A high-cardinality subset on several workers encodes one hash bucket
    # at a time. Every keep mode, n_unique and is_duplicated must still
    # match a scalar reference, with surviving rows in input order. Rows
    # r and r + 150,000 repeat a key; one key column has nulls.
    var setting = WorkerSetting(4)
    var n = 196609
    assert_true(worker_count(n) > 1)
    var a = List[Int64]()
    var av = List[Bool]()
    var b = List[String]()
    var ids = List[Int64]()
    var keys = List[String]()
    for row in range(n):
        var r = row % 150000
        var null = r % 1009 == 0
        a.append(Int64(r * 7 % 150001))
        av.append(not null)
        b.append(String(r % 5))
        ids.append(Int64(row))
        keys.append(
            ("null" if null else String(r * 7 % 150001)) + "|" + String(r % 5)
        )
    var frame = DataFrame(
        [
            Series("a", Column[Int64](a^, av^)),
            Series("b", Column[String](b^)),
            Series("row", Column[Int64](ids^)),
        ]
    )
    var counts = Dict[String, Int]()
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    var first = List[Int]()
    var last = List[Int]()
    var single = List[Int]()
    var seen = Dict[String, Int]()
    for row in range(n):
        if keys[row] not in seen:
            first.append(row)
        seen[keys[row]] = seen.get(keys[row], 0) + 1
        if seen[keys[row]] == counts[keys[row]]:
            last.append(row)
        if counts[keys[row]] == 1:
            single.append(row)
    var subset: List[String] = ["a", "b"]
    assert_true(not low_cardinality([frame.column("a"), frame.column("b")]))
    var modes: List[String] = ["first", "last", "none"]
    var wanted: List[List[Int]] = [first^, last^, single^]
    for m in range(len(modes)):
        ref expected = wanted[m]
        for maintain_order in [False, True]:
            var actual = frame.unique(
                subset.copy(), keep=modes[m], maintain_order=maintain_order
            )
            assert_equal(actual.height(), len(expected))
            ref rows = actual.column("row")._data[Column[Int64]]
            for i in range(len(expected)):
                assert_equal(Int(rows._get(i)), expected[i])
    assert_equal(frame.n_unique(subset.copy()), len(counts))
    var duplicated = frame.is_duplicated(subset.copy())
    for row in range(n):
        assert_equal(
            duplicated._data[BoolColumn]._get(row), counts[keys[row]] > 1
        )
    _ = setting^
