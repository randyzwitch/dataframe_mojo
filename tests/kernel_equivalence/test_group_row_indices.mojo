"""Group row lists hold the same groups with 32-bit and 64-bit row indices."""
# contract-group: kernel.window_mappings
from std.testing import assert_equal, assert_false, assert_true
from dataframe import Column, Series, StringColumn
from dataframe.kernels.group_rows import (
    GroupRows,
    WIDE_ROW_INDEX,
    _group_rows_from_ids,
    group_ids,
    group_rows_from_ids,
    wide_row_indices,
    write_by_groups,
)
from dataframe.window_sort import _partition_groups, partition_groups


def _lists(groups: GroupRows, height: Int) -> List[List[Int]]:
    """Each group's rows, groups ordered by their first row."""
    var by_first = List[Int](length=height, fill=-1)
    for g in range(groups.count()):
        by_first[_row(groups, groups.offset(g))] = g
    var lists = List[List[Int]]()
    for first in range(height):
        var g = by_first[first]
        if g < 0:
            continue
        var rows = List[Int]()
        for k in range(groups.offset(g), groups.offset(g + 1)):
            rows.append(_row(groups, k))
        lists.append(rows^)
    return lists^


def _row(groups: GroupRows, k: Int) -> Int:
    if groups.wide:
        return Int(groups.wide_rows[k])
    return Int(groups.rows[k])


def _assert_same(narrow: GroupRows, wide: GroupRows, height: Int) raises:
    assert_false(narrow.wide)
    assert_true(wide.wide)
    var a = _lists(narrow, height)
    var b = _lists(wide, height)
    assert_equal(len(a), len(b))
    for g in range(len(a)):
        assert_equal(len(a[g]), len(b[g]))
        for k in range(len(a[g])):
            assert_equal(a[g][k], b[g][k])
            if k > 0:
                assert_true(a[g][k - 1] < a[g][k])


def _int_keys(n: Int) -> List[Series]:
    var values = List[Int64](capacity=n)
    var valid = List[Bool](capacity=n)
    for row in range(n):
        values.append(Int64((row * 7919) % 37 - 18))
        valid.append(row % 11 != 3)
    return [Series("k", Column[Int64](values^, valid^))]


def _string_keys(n: Int) -> List[Series]:
    var values = List[String](capacity=n)
    for row in range(n):
        values.append("key" + String((row * 31) % 23))
    return [Series("k", StringColumn(values))]


def test_index_width_follows_the_row_count() raises:
    assert_false(wide_row_indices(0))
    assert_false(wide_row_indices(4294967295))
    assert_true(wide_row_indices(4294967296))


def test_partition_groups_match_at_both_index_widths() raises:
    var n = 5000
    for keys in [_int_keys(n), _string_keys(n)]:
        var narrow = partition_groups(keys, n)
        var wide = _partition_groups[WIDE_ROW_INDEX](keys, n)
        _assert_same(narrow, wide, n)
        var total = 0
        for group in _lists(narrow, n):
            total += len(group)
        assert_equal(total, n)


def test_null_keys_form_one_group() raises:
    var n = 3000
    var keys = _int_keys(n)
    var groups = partition_groups(keys, n)
    var ids = group_ids(groups, n, 4)
    var null_group = -1
    for row in range(n):
        if row % 11 == 3:
            if null_group < 0:
                null_group = ids[row]
            assert_equal(ids[row], null_group)
        else:
            assert_true(ids[row] != null_group or null_group < 0)


def test_writes_and_ids_match_at_both_index_widths() raises:
    var n = 2000
    var ids = List[Int](capacity=n)
    for row in range(n):
        ids.append((row * 13) % 17)
    var narrow = group_rows_from_ids(ids, 17)
    var wide = _group_rows_from_ids[WIDE_ROW_INDEX](ids, 17)
    _assert_same(narrow, wide, n)
    var values = List[Int64](capacity=17)
    for g in range(17):
        values.append(Int64(100 + g))
    var per_group = Series("v", Column[Int64](values^))
    var a = write_by_groups(per_group, narrow, n, 4).value()
    var b = write_by_groups(per_group, wide, n, 4).value()
    var a_ids = group_ids(narrow, n, 4)
    var b_ids = group_ids(wide, n, 4)
    for row in range(n):
        assert_equal(a.get(row).int64(), Int64(100 + ids[row]))
        assert_equal(b.get(row).int64(), Int64(100 + ids[row]))
        assert_equal(a_ids[row], ids[row])
        assert_equal(b_ids[row], ids[row])
