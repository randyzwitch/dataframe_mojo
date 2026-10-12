"""Identifier encoding preserves identity through dependencies and shared plans."""
from std.testing import assert_equal
from dataframe import Column, DataFrame, Series, as_struct, col
from tests.support.name_cases import opaque_names


def test_opaque_names_survive_selectors_windows_struct_fields_and_group_keys() raises:
    for name in opaque_names():
        var second = "partition:" + name
        var frame = DataFrame(
            [
                Series(name, Column[Int64]([2, 1, 2, 1, 2, 1])),
                Series(second, Column[Int64]([0, 0, 1, 0, 0, 1])),
                Series("value", Column[Int64]([5, 7, 11, 13, 17, 19])),
            ]
        )
        var picked = frame.select(col([second, name]).name_prefix("p\x1f"))
        assert_equal(
            picked.columns(), [String("p\x1f") + second, String("p\x1f") + name]
        )
        for row in range(6):
            assert_equal(
                picked.item(row, "p\x1f" + second).int64(),
                Int64([0, 0, 1, 0, 0, 1][row]),
            )
            assert_equal(
                picked.item(row, "p\x1f" + name).int64(),
                Int64([2, 1, 2, 1, 2, 1][row]),
            )
        var sums = frame.select(col("value").sum().over([name, second]))
        for row in range(6):
            assert_equal(
                sums.item(row, "value").int64(),
                Int64([22, 20, 11, 20, 22, 19][row]),
            )
        var packed = frame.with_columns(
            as_struct([col(second).alias(name)], "key")
        )
        var grouped = packed.group_by([name, "key"], maintain_order=True).agg(
            col("value").sum()
        )
        assert_equal(grouped.columns(), [name.copy(), "key", "value"])
        assert_equal(grouped["key"].struct_column().field(0).name(), name)
        assert_equal(grouped.height(), 4)
        for row in range(4):
            assert_equal(
                grouped.item(row, "value").int64(), Int64([22, 20, 11, 19][row])
            )


def test_similar_name_lists_keep_distinct_shared_subplans_and_complete_join_cells() raises:
    for delimiter in [String(", "), "\x1f", "\x1e"]:
        var combined = "a" + delimiter + "b"
        var frame = DataFrame(
            [
                Series("id", Column[Int64]([9, 4, 9])),
                Series("a", Column[Int64]([10, 20, 10])),
                Series("b", Column[Int64]([30, 40, 30])),
                Series(combined, Column[Int64]([50, 60, 50])),
            ]
        )
        for optimize in [False, True]:
            for streaming in [False, True]:
                var actual = (
                    frame.lazy()
                    .drop([combined])
                    .unique()
                    .join(frame.lazy().drop(["a", "b"]).unique(), "id")
                    .collect(
                        optimize=optimize, streaming=streaming, batch_size=1
                    )
                    .sort("id")
                )
                assert_equal(
                    actual.columns(), [String("id"), "a", "b", combined.copy()]
                )
                assert_equal(actual.height(), 2)
                for row in range(2):
                    assert_equal(
                        actual.item(row, "id").int64(), Int64([4, 9][row])
                    )
                    assert_equal(
                        actual.item(row, "a").int64(), Int64([20, 10][row])
                    )
                    assert_equal(
                        actual.item(row, "b").int64(), Int64([40, 30][row])
                    )
                    assert_equal(
                        actual.item(row, combined).int64(), Int64([60, 50][row])
                    )
