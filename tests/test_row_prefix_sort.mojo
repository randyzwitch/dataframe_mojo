"""Full comparisons resolve prefix ties before comparing later keys."""
from std.testing import TestSuite, assert_equal
from dataframe import DataType, Column, DataFrame, Series, col
from dataframe.row_sort import row_arg_sort
from test_sort import reference_order


def test_prefix_boundaries_nuls_directions_and_stability() raises:
    var words: List[String] = [
        "",
        "a",
        "a\x00",
        "abcdefghijklmnopqrstuvwx",
        "abcdefghijklmnopqrstuvwx\x00",
        "abcdefghijklmnopqrstuvwxz",
        "abcdefghijklmnopqrstuvwxa-longer",
        "abcdefghijklmnopqrstuvwxb",
        "abcdefghijklmnopqrstuvwx\x00-more",
        "é共同前缀abcdefghijklmnopqrstuvwxyz",
    ]
    var strings = List[String]()
    var ints = List[Int64]()
    var floats = List[Float64]()
    var valid = List[Bool]()
    for i in range(143):
        strings.append(words[(i * 7) % len(words)])
        ints.append(Int64((i * 13) % 5))
        floats.append(
            Float64(0) / Float64(0) if i % 7
            == 0 else (-0.0 if i % 3 == 0 else Float64(i % 4))
        )
        valid.append(i % 17 != 0)
    var frame = DataFrame(
        [
            Series("s", Column[String](strings^, valid.copy())),
            Series("i", Column[Int64](ints^, valid.copy())),
            Series("f", Column[Float64](floats^, valid^)),
        ]
    )
    var by: List[String] = ["s", "i", "f"]
    for bits in range(64):
        var desc: List[Bool] = [Bool(bits & 1), Bool(bits & 2), Bool(bits & 4)]
        var nulls: List[Bool] = [
            Bool(bits & 8),
            Bool(bits & 16),
            Bool(bits & 32),
        ]
        assert_equal(
            frame.arg_sort(by, descending=desc, nulls_last=nulls),
            reference_order(frame, by, desc, nulls),
        )


def test_parallel_prefix_ties_and_decimal_words() raises:
    var strings = List[String]()
    var ints = List[Int64]()
    for i in range(40_001):
        strings.append("abcdefghijklmnopqrstuvwx" + String((i * 71) % 503))
        ints.append(Int64((i * 19) % 37 - 18))
    var frame = DataFrame(
        [
            Series("s", Column[String](strings^)),
            Series("d", Column[Int64](ints^)),
        ]
    ).with_columns(col("d").cast(DataType.decimal(38, 2)))
    var by: List[String] = ["s", "d"]
    var desc: List[Bool] = [False, True]
    var nulls: List[Bool] = [True, False]
    # Existing dense ranks provide an independent scalable reference.
    from dataframe.series import sort_indices

    var expected = sort_indices(frame._sort_ranks(by, desc, nulls))
    assert_equal(
        frame.arg_sort(by, descending=desc, nulls_last=nulls), expected
    )
    assert_equal(
        row_arg_sort([frame.column("d"), frame.column("s")], desc, nulls),
        sort_indices(frame._sort_ranks(["d", "s"], desc, nulls)),
    )


def test_leading_prefix_buckets_match_dense_reference() raises:
    from dataframe.series import sort_indices

    var integers = List[Int64]()
    var strings = List[String]()
    for i in range(40_003):
        integers.append(Int64((i * 37) % 997))
        strings.append("abcdefghijklmnopqrstuvwx" + String((i * 19) % 503))
    var frame = DataFrame(
        [
            Series("i", Column[Int64](integers^)),
            Series("s", Column[String](strings^)),
        ]
    )
    var by: List[String] = ["i", "s"]
    var nulls: List[Bool] = [True, False]
    for flip in [False, True]:
        var desc: List[Bool] = [flip, not flip]
        assert_equal(
            frame.arg_sort(by, descending=desc, nulls_last=nulls),
            sort_indices(frame._sort_ranks(by, desc, nulls)),
        )


def test_decimal_full_width_and_sliced_chunked_outlier() raises:
    var huge = Int128(1) << 100
    var low = Int128(1) << 63
    var decimal = Series(
        "d", Column[Int128]([huge, huge + low, -huge, -huge + low, 0, huge])
    ).with_dtype(DataType.decimal(38, 0))
    var frame = DataFrame([decimal^])
    assert_equal(frame.arg_sort(["d"]), List[Int]([2, 3, 4, 0, 5, 1]))
    assert_equal(
        frame.arg_sort(["d"], descending=True), List[Int]([1, 0, 5, 4, 3, 2])
    )
    var source = Series(
        "s",
        Column[String](
            [
                "ignored",
                "b",
                "a",
                "abcdefghijklmnopqrstuvwxy",
                "b",
                "a",
                "ignored",
            ]
        ),
    )
    var sliced = source.slice(1, 5)
    var chunks = Series._from_chunks([sliced.slice(0, 2), sliced.slice(2, 3)])
    var chunked = DataFrame(
        [chunks^, Series("i", Column[Int32]([1, 0, 2, 0, 0]))]
    )
    assert_equal(chunked.arg_sort(["s", "i"]), List[Int]([1, 4, 2, 3, 0]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
