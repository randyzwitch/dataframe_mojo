"""Native StringView rank consumers match independent scalar typed ranks."""
# contract-group: kernel.order_keys
from std.testing import assert_equal, assert_true
from dataframe import DataFrame, DataType, Series, StringColumn, col
from dataframe.string_view import StringViewBuilder
from dataframe.window import _rank_strings, _rank
from tests.support.selection_types_reference import (
    selection_column,
    selection_text,
    selection_valid,
)
from tests.support.extended_window_reference import (
    extended_partitions,
    extended_ranks,
)


def test_native_inline_external_string_rank_specialization_and_partition_fallback_match_scalar_ranks() raises:
    for n in [0, 1, 9, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var builder = StringViewBuilder(n + offset)
                for row in range(n + offset):
                    if selection_valid(row, mode):
                        var text = selection_text(row)
                        builder.append(StringSlice(text))
                    else:
                        builder.append_null()
                var native = Series("x", StringColumn(builder^.finish())).slice(
                    offset, n
                )
                assert_true(Bool(native._data[StringColumn]._view_storage))
                var arrow = selection_column(11, n, offset, mode)
                assert_true(not Bool(arrow._data[StringColumn]._view_storage))
                for partitioned in [False, True]:
                    var groups = extended_partitions(n, offset, partitioned)
                    for descending in [False, True]:
                        for method in [
                            "average",
                            "min",
                            "max",
                            "dense",
                            "ordinal",
                        ]:
                            var expected = extended_ranks(
                                11,
                                n,
                                offset,
                                mode,
                                partitioned,
                                method,
                                descending,
                            )
                            for source in [native.copy(), arrow.copy()]:
                                var fast = _rank_strings(
                                    source._data[StringColumn],
                                    groups,
                                    method,
                                    descending,
                                )
                                assert_true(
                                    Bool(fast),
                                    "complete StringView partitions must take actual packed string rank route",
                                )
                                var actual = fast.take()
                                assert_true(
                                    actual.dtype()
                                    == (
                                        DataType.FLOAT64 if method
                                        == "average" else DataType.INT64
                                    )
                                )
                                assert_equal(len(actual), n)
                                for row in range(n):
                                    var present = selection_valid(
                                        row + offset, mode
                                    )
                                    assert_equal(
                                        actual.get(row).is_null(), not present
                                    )
                                    if present:
                                        if method == "average":
                                            assert_equal(
                                                actual.get(row).float64(),
                                                expected[row],
                                            )
                                        else:
                                            assert_equal(
                                                actual.get(row).int64(),
                                                Int64(expected[row]),
                                            )
                                # Public rank is also required to consume the
                                # native storage, independently of direct
                                # specialization eligibility assertions.
                                if not partitioned:
                                    var public = (
                                        DataFrame([source.copy()])
                                        .select(
                                            col("x").rank(method, descending)
                                        )
                                        .column("x")
                                    )
                                    for row in range(n):
                                        assert_equal(
                                            public.get(row).is_null(),
                                            not selection_valid(
                                                row + offset, mode
                                            ),
                                        )
                                        if selection_valid(row + offset, mode):
                                            if method == "average":
                                                assert_equal(
                                                    public.get(row).float64(),
                                                    expected[row],
                                                )
                                            else:
                                                assert_equal(
                                                    public.get(row).int64(),
                                                    Int64(expected[row]),
                                                )
                if n > 1:
                    # Incomplete coverage must decline the packed path and
                    # rank only explicit partition members through fallback.
                    var partial: List[List[Int]] = [[0, n - 1]]
                    assert_true(
                        not Bool(
                            _rank_strings(
                                native._data[StringColumn],
                                partial,
                                "ordinal",
                                False,
                            )
                        )
                    )
                    var fallback = _rank(native, partial, "ordinal", False)
                    var positions: List[Int] = [0, n - 1]
                    for row in range(n):
                        var present = row in positions and selection_valid(
                            row + offset, mode
                        )
                        assert_equal(fallback.get(row).is_null(), not present)
                        if present:
                            var value = selection_text(row + offset)
                            var rank = Int64(1)
                            for other in positions:
                                if not selection_valid(other + offset, mode):
                                    continue
                                var text = selection_text(other + offset)
                                rank += Int64(
                                    text < value
                                    or (text == value and other < row)
                                )
                            assert_equal(fallback.get(row).int64(), rank)
