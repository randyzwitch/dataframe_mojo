"""Indexed group reductions retain the requested null policy at source rows."""
# contract-group: kernel.distinct
from std.testing import assert_equal
from dataframe import Column, Series, DataType, col
from dataframe.binding import bind
from dataframe.indexed_reduce import reduce_indexed
from tests.support.numeric_model import INTEGER_DTYPES, FLOAT_DTYPES
from tests.support.reduction_types_reference import (
    typed_reduction_columns,
    reduction_validity,
)


def _check[D: DType]() raises:
    var values = Series(
        "x",
        Column[Scalar[D]](
            [3, 1, 3, 2, 1, 9],
            [True, False, True, True, True, False],
        ),
    )
    var columns: List[Series] = [values^]
    var rows: List[Int] = [5, 0, 1, 4, 2, 3]
    var ids: List[Int] = [0, 1, 1, 0, 2, 2]
    for ignore in [False, True]:
        var expression = bind(col("x").n_unique(ignore_nulls=ignore), columns)
        var result = reduce_indexed(
            expression, columns, rows.unsafe_ptr(), ids, 3
        )
        assert_equal(result.dtype(), DataType.INT64)
        assert_equal(len(result), 3)
        assert_equal(result.null_count(), 0)
        for g in range(3):
            assert_equal(
                result.get(g).int64(), Int64(1 if ignore and g < 2 else 2)
            )


def test_indexed_distinct_counts_ignore_nulls_without_gathering_values() raises:
    comptime for i in range(len(INTEGER_DTYPES)):
        _check[INTEGER_DTYPES[i]]()
    comptime for i in range(len(FLOAT_DTYPES)):
        _check[FLOAT_DTYPES[i]]()


def test_series_distinct_counts_forward_both_null_policies() raises:
    for mode in [1, 2]:
        for series in typed_reduction_columns(reduction_validity(mode)):
            var count = 0 if mode == 2 else (
                2 if series.dtype() == DataType.BOOL else 3
            )
            assert_equal(series.n_unique(), count + 1)
            assert_equal(series.n_unique(ignore_nulls=False), count + 1)
            assert_equal(series.n_unique(ignore_nulls=True), count)
            var empty = series.slice(0, 0)
            assert_equal(empty.n_unique(), 0)
            assert_equal(empty.n_unique(ignore_nulls=True), 0)
