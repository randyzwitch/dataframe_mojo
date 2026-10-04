"""String ranks preserve ties, nulls and row mapping in every method."""
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, Series, col
from dataframe.window import _rank
from dataframe.string_column import StringColumn
from dataframe.string_view import StringViewBuilder


def test_string_rank_methods_directions_and_chunks() raises:
    var values = Series(
        "s",
        Column[String](
            [
                "β",
                "a\x00tail",
                "β",
                "",
                "long_common_prefix_x",
                "long_common_prefix_y",
                "a\x00tail",
                "",
                "",
            ],
            [True, True, True, False, True, True, True, True, True],
        ),
    )
    var key = Series("g", Column[Int64]([1, 1, 1, 1, 2, 2, 1, 2, 2]))
    var ascending: List[List[Int64]] = [
        [3, 1, 3, 0, 3, 4, 1, 1, 1],
        [4, 2, 4, 0, 3, 4, 2, 2, 2],
        [2, 1, 2, 0, 2, 3, 1, 1, 1],
        [3, 1, 4, 0, 3, 4, 2, 1, 2],
    ]
    var descending: List[List[Int64]] = [
        [1, 3, 1, 0, 2, 1, 3, 3, 3],
        [2, 4, 2, 0, 2, 1, 4, 4, 4],
        [1, 2, 1, 0, 2, 1, 2, 3, 3],
        [1, 3, 2, 0, 2, 1, 4, 3, 4],
    ]
    var methods: List[String] = ["min", "max", "dense", "ordinal", "average"]
    for chunks in [False, True]:
        var input = Series._from_chunks(
            [values.slice(0, 4), values.slice(4, 5)]
        ) if chunks else values.copy()
        var frame = DataFrame([key.copy(), input^])
        for reverse in [False, True]:
            var expected = descending.copy() if reverse else ascending.copy()
            for method in range(len(methods)):
                var result = frame.select(
                    col("s")
                    .rank(methods[method], descending=reverse)
                    .over("g")
                    .alias("r")
                ).column("r")
                for i in range(9):
                    assert_equal(result.get(i).is_null(), i == 3)
                    if i == 3:
                        continue
                    if method == 4:
                        assert_equal(
                            result.get(i).float64(),
                            Float64(expected[0][i] + expected[1][i]) / 2,
                        )
                    else:
                        assert_equal(result.get(i).int64(), expected[method][i])


def test_string_rank_empty_singletons_and_all_null() raises:
    var empty = Series("s", Column[String](List[String]()))
    for method in ["average", "min", "max", "dense", "ordinal"]:
        assert_equal(len(_rank(empty, List[List[Int]](), method, False)), 0)
        var nulls = Series(
            "s", Column[String](["", "", ""], [False, False, False])
        )
        var all_null = _rank(nulls, [[0, 1, 2]], method, True)
        assert_equal(all_null.null_count(), 3)
        var single = Series("s", Column[String](["a", "a", "b"]))
        var ranks = _rank(single, [[0], [1], [2]], method, False)
        for i in range(3):
            assert_true(not ranks.get(i).is_null())
            if method == "average":
                assert_equal(ranks.get(i).float64(), 1.0)
            else:
                assert_equal(ranks.get(i).int64(), 1)


def test_string_rank_borrowed_views_and_offset_slices() raises:
    var text: List[String] = [
        "skip",
        "abcdefg-long_tail_z",
        "abcdefg-long_tail_a",
        "",
        "多字节é",
        "a\x00tail",
        "abcdefg-long_tail_a",
        "z",
    ]
    var valid: List[Bool] = [True, True, True, False, True, True, True, True]
    var builder = StringViewBuilder(len(text))
    for i in range(len(text)):
        if valid[i]:
            builder.append(StringSlice(text[i]))
        else:
            builder.append_null()
    var source = Series("s", StringColumn(text.copy(), valid.copy())).slice(
        1, 7
    )
    var view = Series("s", StringColumn(builder^.finish())).slice(1, 7)
    var groups: List[List[Int]] = [[0, 2, 4, 6], [1, 3, 5]]
    for method in ["average", "min", "max", "dense", "ordinal"]:
        for descending in [False, True]:
            var expected = _rank(source, groups, method, descending)
            var result = _rank(view, groups, method, descending)
            assert_true(result.equals(expected))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
