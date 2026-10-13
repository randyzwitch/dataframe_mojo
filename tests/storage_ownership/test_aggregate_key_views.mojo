"""The aggregate table's string key columns are Arrow views built from each
group's stored words: a key up to twelve bytes inline in its view, a longer
one pointing into the input's own byte buffer, whether the input stores
offsets or views, so no key's bytes are copied. Keys whose bytes live in a
dictionary (Parquet dictionary codes) are copied into a buffer the result
owns, and stay correct after the input is gone."""
from std.testing import assert_equal, assert_true

from dataframe import DataFrame, Expr, Series, StringColumn, col
from dataframe.string_view import StringViewBuilder
from tests.support.aggregate_table_reference import (
    aggregate_frame,
    assert_groups,
    coded_strings,
    group_models,
)
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 72_000
comptime GROUPS = 6_000


def reductions() -> List[Expr]:
    return [
        col("x").sum(min_count=1).alias("x_sum"),
        col("x").mean().alias("x_mean"),
        col("x").min().alias("x_min"),
        col("x").max().alias("x_max"),
        col("x").count().alias("x_count"),
        col("w").sum(min_count=1).alias("w_sum"),
        col("w").null_count().alias("w_nulls"),
        col("x").len().alias("rows"),
        col("y").mean().alias("y_mean"),
        col("y").std().alias("y_std"),
    ]


def grouped(frame: DataFrame) raises -> DataFrame:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var result = frame.group_by(["s", "i", "f", "b"]).agg(reductions())
    assert_true(
        "dataframe-path: group_by.aggregate_table\n" in observed.text(),
        "the group-by did not reach the aggregate table",
    )
    _ = observed^
    _ = trace^
    return result^


def as_views(strings: Series) raises -> Series:
    ref column = strings._data[StringColumn]
    var builder = StringViewBuilder(len(strings))
    for row in range(len(strings)):
        if column._valid(row):
            builder.append(column._get(row))
        else:
            builder.append_null()
    return Series(strings.name(), StringColumn(builder^.finish()))


def input_buffers(strings: Series) -> List[Int]:
    """The addresses of the byte buffers `strings` reads from."""
    var addresses = List[Int]()
    ref column = strings._data[StringColumn]
    if column._is_view_storage():
        ref storage = column._view_storage.value()
        for b in range(len(storage._buffers[])):
            addresses.append(Int(storage._buffers[][b][].unsafe_ptr()))
    else:
        addresses.append(Int(column._bytes[].unsafe_ptr()))
    return addresses^


def assert_views_into(result: DataFrame, buffers: List[Int]) raises:
    """Every key is a view; every long key's buffer is one of `buffers`."""
    var long_keys = 0
    for part in result.column("s").chunks():
        ref column = part._data[StringColumn]
        assert_true(column._is_view_storage(), "keys must be views")
        ref storage = column._view_storage.value()
        for row in range(len(part)):
            if not column._valid(row):
                continue
            var view = storage._views[][column._offset + row].copy()
            if view.is_inline():
                continue
            long_keys += 1
            var address = Int(
                storage._buffers[][Int(view.buffer_index)][].unsafe_ptr()
            )
            assert_true(address in buffers, "a long key's bytes were copied")
    assert_true(long_keys > 0)


def test_long_keys_point_into_the_inputs_offsets_and_view_buffers() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var models = group_models(ROWS, GROUPS)
    var raw = aggregate_frame(ROWS, GROUPS)
    for views in [False, True]:
        var columns = raw._columns.copy()
        if views:
            columns[0] = as_views(raw.column("s"))
        var frame = DataFrame(columns^, height=ROWS)
        var buffers = input_buffers(frame.column("s"))
        var result = grouped(frame)
        assert_views_into(result, buffers)
        assert_groups(result, models, True)
    _ = workers^


def grouped_coded() raises -> DataFrame:
    """Grouped keys carrying dictionary codes; the input dies here."""
    var raw = aggregate_frame(ROWS, GROUPS)
    var columns = raw._columns.copy()
    columns[0] = coded_strings(raw.column("s"))
    return grouped(DataFrame(columns^, height=ROWS))


def test_dictionary_keys_are_copied_and_outlive_their_input() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var result = grouped_coded()
    # Allocate and drop as much again, so freed input buffers are reused.
    for _ in range(3):
        var other = aggregate_frame(ROWS, GROUPS)
        assert_equal(other.height(), ROWS)
    assert_groups(result, group_models(ROWS, GROUPS), True)
    _ = workers^
