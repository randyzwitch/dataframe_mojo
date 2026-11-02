"""A join's build-side string columns, read through the matched rows.

A pipeline join step does not copy the view descriptors of a build-side
string column: the joined column selects the build column's descriptors
through the matched rows (`StringColumn._select_rows`), and carries its
dictionary codes the same way. Every reader then sees the joined rows: row
reads, gathers of the selection, flat copies, concatenation, grouping (by
the codes and by the bytes), comparisons, string kernels and sorting are
checked against a scalar model, with and without nulls."""
# contract-group: table.join
from std.memory import ArcPointer
from std.testing import assert_equal, assert_false, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
)
from dataframe.categorical import decode
from dataframe.dtype import CategoricalDictionary
from dataframe.string_column import StringCodes
from dataframe.value import AnyValue
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 20_000
comptime DIMENSIONS = 500
comptime LABELS = 37


def integer(value: AnyValue) raises -> Int64:
    if value.dtype() == DataType.INT32:
        return Int64(value.int32())
    return value.int64()


def label(code: Int) -> String:
    # Labels past twelve bytes are stored out of line in their views.
    if code % 3 == 0:
        return "n" + String(code)
    return "dimension payload " + String(code)


def name_valid(k: Int, nulls: Bool) -> Bool:
    return not nulls or k % 11 != 4


def key_of(row: Int) -> Int:
    return (row * 7919) % (DIMENSIONS + 50)


def names(nulls: Bool) raises -> Series:
    """The dimension's name column as Parquet hands it over: Arrow views
    beside the dictionary codes they were read from."""
    var dictionary = CategoricalDictionary()
    for code in range(LABELS):
        dictionary.append(label(code))
    var codes = List[UInt32]()
    var valid = List[Bool]()
    for k in range(DIMENSIONS):
        codes.append(UInt32(k % LABELS))
        valid.append(name_valid(k, nulls))
    var categorical = Series(
        "name", Column[UInt32](codes.copy(), valid^)
    ).with_dtype(DataType.categorical(dictionary^))
    var text = decode(categorical)
    var strings = text._data[StringColumn].copy()
    strings._codes = ArcPointer(
        StringCodes(ArcPointer(codes^), categorical.dtype())
    )
    var views = strings._as_views()
    assert_true(views._is_view_storage())
    assert_true(Bool(views._codes))
    return Series("name", views^)


def dimension(nulls: Bool) raises -> DataFrame:
    var keys = List[Int64]()
    var b = List[Int64]()
    for k in range(DIMENSIONS):
        keys.append(Int64(k))
        b.append(Int64(3 * k))
    return DataFrame(
        [
            Series("dk", Column[Int64](keys^)),
            names(nulls),
            Series("b", Column[Int64](b^)),
        ]
    )


def probe() raises -> DataFrame:
    var keys = List[Int64]()
    var a = List[Int64]()
    for row in range(ROWS):
        keys.append(Int64(key_of(row)))
        a.append(Int64(row))
    return DataFrame(
        [Series("k", Column[Int64](keys^)), Series("a", Column[Int64](a^))]
    )


def test_selected_rows_read_as_the_rows_they_select() raises:
    for nulls in [False, True]:
        var source = names(nulls)
        ref strings = source._data[StringColumn]
        var picks = List[Int]()
        for i in range(3_000):
            picks.append((i * 131) % DIMENSIONS)
        var through = strings._select_rows(ArcPointer(picks.copy()))
        assert_true(Bool(through))
        var selected = through.take()
        assert_equal(len(selected), len(picks))
        assert_true(selected._view_storage.value()._selects())
        assert_false(strings._shares_buffers_with(selected))
        var codes = List[UInt32](length=len(selected), fill=0)
        selected._write_codes(Int(codes.unsafe_ptr()))
        var flat = selected._flat_views()
        assert_false(flat._view_storage.value()._selects())
        var offsets = selected.to_large_utf8()
        var nulls_seen = 0
        for i in range(len(picks)):
            var k = picks[i]
            assert_equal(selected._valid(i), name_valid(k, nulls))
            if not name_valid(k, nulls):
                nulls_seen += 1
                continue
            assert_equal(String(selected._get(i)), label(k % LABELS))
            assert_equal(String(flat._get(i)), label(k % LABELS))
            assert_equal(String(offsets._get(i)), label(k % LABELS))
            assert_equal(
                selected._byte_length(i), label(k % LABELS).byte_length()
            )
            assert_equal(Int(codes[i]), k % LABELS)
        assert_equal(selected.null_count(), nulls_seen)
        # A gather of the selection, with a missing row, and a window.
        var rows = List[Int]()
        for i in range(0, len(picks), 7):
            rows.append(i)
        rows.append(-1)
        var gathered = selected.take_or_null(rows, "")
        for j in range(len(rows) - 1):
            var k = picks[rows[j]]
            assert_equal(gathered._valid(j), name_valid(k, nulls))
            if name_valid(k, nulls):
                assert_equal(String(gathered._get(j)), label(k % LABELS))
        assert_false(gathered._valid(len(rows) - 1))
        var window = selected.slice(100, 50)
        for j in range(50):
            var k = picks[100 + j]
            if name_valid(k, nulls):
                assert_equal(String(window._get(j)), label(k % LABELS))
        # Concatenated, as morsel results are.
        var combined = selected.copy()
        combined._append_column(window)
        assert_equal(len(combined), len(picks) + 50)
        for j in range(50):
            var k = picks[100 + j]
            if name_valid(k, nulls):
                assert_equal(
                    String(combined._get(len(picks) + j)), label(k % LABELS)
                )
        var chunked = Series._from_chunks(
            [Series("name", selected.copy()), Series("name", window^)]
        ).rechunk()
        assert_equal(len(chunked), len(picks) + 50)
        var k0 = picks[0]
        if name_valid(k0, nulls):
            assert_equal(
                String(chunked._data[StringColumn]._get(0)),
                label(k0 % LABELS),
            )


def test_joined_build_strings_match_the_scalar_model() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    for nulls in [False, True]:
        for threads in [1, 8]:
            var workers = ScopedEnvironment(
                "DATAFRAME_THREADS", String(threads)
            )
            var observed = StrategyTrace()
            var plan = (
                probe()
                .lazy()
                .filter(col("a") % lit(Int64(3)) != lit(Int64(1)))
                .join(dimension(nulls).lazy(), left_on=["k"], right_on=["dk"])
            )
            var joined = plan.collect(batch_size=4096)
            assert_true(
                "dataframe-path: pipeline.join_view_selection\n"
                in observed.text(),
                "the join gathered the build strings",
            )
            _ = observed^
            # Every joined row, and the per-label model of the rest.
            var sums = Dict[String, Int64]()
            var counts = Dict[String, Int]()
            var r = 0
            for row in range(ROWS):
                var k = key_of(row)
                if row % 3 == 1 or k >= DIMENSIONS:
                    continue
                var a = integer(joined.item(r, "a"))
                assert_equal(a, Int64(row))
                var cell = joined.item(r, "name")
                var key = String("<null>")
                if name_valid(k, nulls):
                    key = label(k % LABELS)
                    assert_equal(cell.string(), key)
                else:
                    assert_true(cell.is_null())
                sums[key] = sums.get(key, 0) + a
                counts[key] = counts.get(key, 0) + 1
                r += 1
            assert_equal(joined.height(), r)

            # Grouped by the name: by its codes in the aggregate table.
            var grouped = (
                plan.group_by("name")
                .agg([col("a").sum().alias("s"), col("a").count().alias("n")])
                .collect(batch_size=4096)
            )
            assert_equal(grouped.height(), len(sums))
            for g in range(grouped.height()):
                var cell = grouped.item(g, "name")
                var key = String("<null>") if cell.is_null() else cell.string()
                assert_equal(integer(grouped.item(g, "s")), sums[key])
                assert_equal(Int(integer(grouped.item(g, "n"))), counts[key])

            # Grouped by the name with a second key and a reduction of it.
            var by_two = (
                plan.group_by(["name", "b"])
                .agg([col("name").max().alias("top")])
                .collect(batch_size=4096)
            )
            for g in range(by_two.height()):
                var cell = by_two.item(g, "name")
                var top = by_two.item(g, "top")
                assert_equal(cell.is_null(), top.is_null())
                if not cell.is_null():
                    assert_equal(cell.string(), top.string())

            # A comparison with a literal, and string kernels on the rows.
            var wanted = label(5)
            var equal = plan.filter(col("name") == lit(wanted)).collect(
                batch_size=4096
            )
            assert_equal(equal.height(), counts.get(wanted, 0))
            var measured = plan.with_columns(
                [
                    col("name").str().len_bytes().alias("bytes"),
                    col("name")
                    .str()
                    .contains("load 1", literal=True)
                    .alias("has"),
                    col("name").str().slice(2, 5).alias("part"),
                ]
            ).collect(batch_size=4096)
            for i in range(0, measured.height(), 17):
                var cell = measured.item(i, "name")
                if cell.is_null():
                    assert_true(measured.item(i, "bytes").is_null())
                    continue
                var text = cell.string()
                assert_equal(
                    Int(integer(measured.item(i, "bytes"))), text.byte_length()
                )
                assert_equal(measured.item(i, "has").bool(), "load 1" in text)
                # ASCII labels: five bytes from the third.
                assert_equal(
                    measured.item(i, "part").string().byte_length(),
                    max(0, min(text.byte_length(), 7) - 2),
                )

            # Sorted by the name, then the row.
            var ordered = plan.sort(["name", "a"]).collect(batch_size=4096)
            assert_equal(ordered.height(), r)
            for i in range(1, ordered.height()):
                var left = ordered.item(i - 1, "name")
                var right = ordered.item(i, "name")
                if left.is_null() or right.is_null():
                    continue
                assert_true(left.string() <= right.string())
            _ = workers^
    _ = trace^
