"""Prepared duplicate groups retain exact equality, order and null semantics."""
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import Column, Series
from dataframe.join_hash import (
    prepare_hash_index,
    prepared_hash_join_rows,
    prepared_hash_semi_anti_rows,
)


def test_heavy_groups_keep_integer_string_and_compound_order() raises:
    var numbers = List[Int64]()
    var text = List[String]()
    var valid = List[Bool]()
    for i in range(12000):
        numbers.append(Int64(i % 4))
        text.append("v" + String(i % 4))
        valid.append(i % 17 != 0)
    var number_key = Series("n", Column[Int64](numbers^, valid))
    var text_key = Series("s", Column[String](text^, valid))
    var number_probe = Series(
        "n", Column[Int64]([0, 2, 9, 0], [True, True, True, False])
    )
    var text_probe = Series(
        "s",
        Column[String](
            ["v0", "v2", "missing", "v0"], [True, True, True, False]
        ),
    )
    var expected_left = List[Int]()
    var expected_right = List[Int]()
    for probe in range(2):
        for row in range(12000):
            if row % 4 == 2 * probe and row % 17 != 0:
                expected_left.append(probe)
                expected_right.append(row)
    expected_left.append(2)
    expected_left.append(3)
    expected_right.append(-1)
    expected_right.append(-1)
    for kind in range(3):
        var build = List[Series]()
        var probe = List[Series]()
        if kind != 1:
            build.append(number_key.copy())
            probe.append(number_probe.copy())
        if kind != 0:
            build.append(text_key.copy())
            probe.append(text_probe.copy())
        var index = prepare_hash_index(build)
        var group_rows = 0
        var slots = 0
        for bucket in index.indexes[]:
            group_rows += len(bucket.groups)
            slots += len(bucket.slots)
        assert_true(group_rows > 0)
        assert_true(slots < 6000)
        for _ in range(2):
            var rows = prepared_hash_join_rows(
                probe, index, True, omit_identity=True
            )
            assert_equal(rows[0], expected_left)
            assert_equal(rows[1], expected_right)
            assert_equal(rows[2], False)
            assert_equal(
                prepared_hash_semi_anti_rows(probe, index, True), [0, 1]
            )
            assert_equal(
                prepared_hash_semi_anti_rows(probe, index, False), [2, 3]
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
