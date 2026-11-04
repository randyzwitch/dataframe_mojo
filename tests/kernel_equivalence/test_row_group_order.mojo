"""A top-N over a stored table reads its row groups in statistics order
and drops those that cannot reach the limit (RowGroupPruner); the rows
match a sort of the same rows without statistics."""
# contract-group: kernel.order_keys
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series
from dataframe.sorting.row_group_order import ROW_GROUP_ROWS, row_group_order
from dataframe.storage.string import StringColumn

comptime GROUPS = 5
comptime ROWS = GROUPS * ROW_GROUP_ROWS + 1000


def _lcg(mut state: UInt64) -> Int:
    state = state * 6364136223846793005 + 1442695040888963407
    return Int(state >> 33)


def _stored(var columns: List[Series]) raises -> DataFrame:
    return DataFrame(columns^)._with_storage_statistics()


def _plain(frame: DataFrame) raises -> DataFrame:
    """The frame's rows in new buffers with no statistics."""
    var columns = List[Series]()
    var half = frame.height() // 2
    for column in frame._columns:
        columns.append(
            Series._from_chunks(
                [column.slice(0, half), column.slice(half, len(column) - half)]
            ).rechunk()
        )
    return DataFrame(columns^)


def _ids() -> Series:
    var ids = List[Int64](capacity=ROWS)
    for i in range(ROWS):
        ids.append(Int64(i))
    return Series("id", Column[Int64](ids^))


def _check(frame: DataFrame, key: String, label: String) raises:
    """Every direction, null placement, limit and offset: the stored
    table's top-N against the plain rows' sort, ties in input order."""
    var plain = _plain(frame)
    var limits = [10, 100, 5, 50, 3]
    var offsets = [0, 0, 200000, 130000, ROWS - 2]
    for descending in [False, True]:
        for nulls_last in [True, False]:
            var expected_all = plain.sort(
                [key],
                descending=[descending],
                nulls_last=[nulls_last],
                maintain_order=True,
            )
            for c in range(len(limits)):
                var limit = limits[c]
                var offset = offsets[c]
                var got = (
                    frame.lazy()
                    .sort(
                        [key],
                        descending=[descending],
                        nulls_last=[nulls_last],
                        maintain_order=True,
                    )
                    .slice(offset, limit)
                    .collect()
                )
                var expected = expected_all.slice(
                    offset, max(0, min(limit, ROWS - offset))
                )
                assert_true(
                    got.equals(expected),
                    label
                    + " desc="
                    + String(descending)
                    + " nulls_last="
                    + String(nulls_last)
                    + " limit="
                    + String(limit)
                    + " offset="
                    + String(offset),
                )


def test_blocks_in_reverse_order_prune_to_one_row_group() raises:
    var values = List[Int64](capacity=ROWS)
    for i in range(ROWS):
        var group = i // ROW_GROUP_ROWS
        values.append(Int64((GROUPS - group) * 1_000_000 + i % 977))
    var frame = _stored([Series("k", Column[Int64](values^)), _ids()])
    var order = row_group_order(frame.column("k"), False, True, 10, 0, True)
    assert_true(Bool(order), "stored row groups are ordered")
    # The last (short) row group has the smallest keys and holds the
    # limit; every other one is dropped.
    assert_equal(len(order.value().ranges), 2)
    assert_equal(order.value().ranges[0], GROUPS * ROW_GROUP_ROWS)
    var offset_order = row_group_order(
        frame.column("k"), False, True, 10, 1000 + ROW_GROUP_ROWS, True
    )
    # The offset covers the two best row groups: both are dropped.
    assert_equal(offset_order.value().remaining_offset, 0)
    assert_equal(offset_order.value().ranges[0], (GROUPS - 2) * ROW_GROUP_ROWS)
    _check(frame, "k", "blocks")


def test_ties_and_nulls() raises:
    var state = UInt64(7)
    var values = List[Int64](capacity=ROWS)
    var valid = List[Bool](capacity=ROWS)
    for i in range(ROWS):
        var group = i // ROW_GROUP_ROWS
        values.append(Int64(_lcg(state) % 1000))
        # Row group 1 is all null, row group 3 partly null.
        valid.append(not (group == 1 or (group == 3 and i % 7 == 0)))
    var frame = _stored([Series("k", Column[Int64](values^, valid)), _ids()])
    _check(frame, "k", "ties and nulls")


def test_floats_with_nan() raises:
    var state = UInt64(11)
    var values = List[Float64](capacity=ROWS)
    for i in range(ROWS):
        var group = i // ROW_GROUP_ROWS
        if group == 2 and i % 5 == 0:
            values.append(Float64(0) / Float64(0))
        else:
            values.append(Float64(group * 10 + _lcg(state) % 10))
    var frame = _stored([Series("k", Column[Float64](values^)), _ids()])
    _check(frame, "k", "floats")


def test_long_strings_with_shared_prefixes() raises:
    var state = UInt64(13)
    var values = List[String](capacity=ROWS)
    for i in range(ROWS):
        var group = i // ROW_GROUP_ROWS
        # Past the 12 stored bytes the row groups differ only in a suffix.
        values.append(
            "shared-prefix-"
            + String(GROUPS - group)
            + "-"
            + String(_lcg(state) % 100000)
        )
    var frame = _stored([Series("k", StringColumn(values)), _ids()])
    _check(frame, "k", "strings")
