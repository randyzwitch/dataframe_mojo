"""Partition top-k and parallel stable ordering exercise their actual routes."""
from std.math import isnan
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, Series, col, lit
from dataframe.binding import bind
from dataframe.bool_column import BoolColumn
from dataframe.top_k import top_k_mask, _pattern
from dataframe.aggregate_table import TOP_ROWS_INLINE
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)
from dataframe.parallel import worker_count, configured_workers, Pool
from dataframe.comparison_sort import (
    _co_rank,
    _merge_slice,
    _prefix_buckets,
    comparison_arg_sort,
)
from dataframe.packed_sort import packed_arg_sort, _sort_heavy
from dataframe.row_sort import row_arg_sort
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.parallel_key_reference import (
    ScalarParallelOrder,
    expected_partition_mask,
    parallel_group,
    parallel_present,
    parallel_primary,
    parallel_secondary,
    parallel_string,
    parallel_value,
    prefix_remainder_string,
    scalar_parallel_rows,
)


def topk_frame(n: Int, groups: Int, offset: Int = 7) raises -> DataFrame:
    var values = List[Float64]()
    var valid = List[Bool]()
    var keys = List[Int64]()
    var key_valid = List[Bool]()
    var ids = List[Int64]()
    for row in range(n + offset):
        var present = parallel_present(row)
        values.append(parallel_value(row) if present else Float64.MAX)
        valid.append(present)
        keys.append(Int64(row % groups))
        key_valid.append(row % 97 != 3)
        ids.append(Int64(row - offset))
    var value = Series("value", Column[Float64](values^, valid^)).slice(
        offset, n
    )
    if n > 65536:
        value = Series._from_chunks(
            [
                value.slice(0, 65535),
                value.slice(65535, 1),
                value.slice(65536, n - 65536),
            ]
        )
    return DataFrame(
        [
            Series("group", Column[Int64](keys^, key_valid^)).slice(offset, n),
            value^,
            Series("row", Column[Int64](ids^)).slice(offset, n),
        ]
    )


def check_topk(
    source: DataFrame, groups: Int, limit: Int64, strict: Bool, descending: Bool
) raises:
    var rank = col("value").rank("ordinal", descending=descending).over("group")
    var predicate = rank < lit(limit) if strict else rank <= lit(limit)
    var columns: List[Series] = [
        source.column("group"),
        source.column("value"),
        source.column("row"),
    ]
    var bound = bind(predicate, columns)
    var shape = _pattern(bound)
    assert_true(
        shape[0] >= 0, "ordinal partition limit must match specialized shape"
    )
    var k = max(Int64(0), limit)
    if strict and k > 0:
        k -= 1
    assert_equal(shape[2], Int(k))
    var result = top_k_mask(bound, columns, source.height())
    assert_true(Bool(result), "specialized numeric top-k must produce a mask")
    var mask = result.take()
    var expected = expected_partition_mask(
        source.height(), 7, groups, limit, strict, descending
    )
    for row in range(source.height()):
        assert_true(mask._valid(row))
        assert_equal(mask._get(row), expected[row])
    var actual = source.filter(predicate)
    var destination = 0
    for row in range(source.height()):
        if not expected[row]:
            continue
        assert_equal(actual.item(destination, "row").int64(), Int64(row))
        assert_equal(
            actual.item(destination, "group").is_null(), (row + 7) % 97 == 3
        )
        if (row + 7) % 97 != 3:
            assert_equal(
                actual.item(destination, "group").int64(),
                Int64((row + 7) % groups),
            )
        var value = actual.item(destination, "value")
        assert_true(not value.is_null())
        var want = parallel_value(row + 7)
        if isnan(want):
            assert_true(isnan(value.float64()))
        else:
            assert_equal(value.float64(), want)
        destination += 1
    assert_equal(actual.height(), destination)


def test_partition_top_k_replicated_owned_and_bucketed_routes_match_complete_scalar_memberships() raises:
    var setting = WorkerSetting(4)
    var n = 196609
    var workers = worker_count(n)
    assert_true(workers > 1)
    for route in range(3):
        var groups = 4 if route == 0 else 4096 if route == 1 else 32768
        # More pairs than a group row holds inline, so the partition
        # routes run rather than the aggregate table's.
        var k = 17 if route == 0 else 20 if route == 1 else 17
        assert_true(k > TOP_ROWS_INLINE)
        # Fixture visits every non-null key and the null key. These exact
        # inequalities are the production allocator's dispatch conditions.
        var count = groups + 1
        if route == 0:
            assert_true(k <= n // count // workers)
        elif route == 1:
            assert_true(k <= n // count and k > n // count // workers)
        else:
            assert_true(k > n // count)
        var source = topk_frame(n, groups)
        for descending in [False, True]:
            for strict in [False, True]:
                check_topk(
                    source, groups, Int64(k + Int(strict)), strict, descending
                )
    _ = setting^


def test_partition_top_k_through_the_aggregate_table_matches_scalar_memberships() raises:
    # Up to TOP_ROWS_INLINE pairs a group: each worker's table keeps every
    # group's best (value, row) pairs in the group's row, merged at the
    # finish (DuckDB's arg_max(row, value, k)). Few groups and many, a
    # null key, NaN and null values, ties kept by row, chunked values.
    var setting = WorkerSetting(4)
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var n = 196609
    for shape in range(4):
        var groups = 4 if shape == 0 else 4096 if shape == 1 else 32768
        var k = (
            1 if shape
            == 0 else TOP_ROWS_INLINE if shape
            == 1 else (7 if shape == 2 else 2)
        )
        if shape == 3:
            groups = 150000
        var source = topk_frame(n, groups)
        for descending in [False, True]:
            for strict in [False, True]:
                var observed = StrategyTrace()
                check_topk(
                    source, groups, Int64(k + Int(strict)), strict, descending
                )
                assert_true(
                    "dataframe-path: filter.top_k_table\n" in observed.text(),
                    "the top-k filter did not run through the table",
                )
                _ = observed^
    _ = trace^
    _ = setting^


def test_partition_top_k_limit_extremes_and_shape_fallbacks_are_explicit() raises:
    var setting = WorkerSetting(1)
    var source = topk_frame(65, 4)
    for limit in [Int64.MIN, Int64(-1), Int64(0), Int64(1), Int64.MAX]:
        for strict in [False, True]:
            for descending in [False, True]:
                check_topk(source, 4, limit, strict, descending)
    var columns: List[Series] = [
        source.column("group"),
        source.column("value"),
        source.column("row"),
    ]
    var misses: List[Expr] = [
        col("value").rank("dense").over("group") <= lit(Int64(1)),
        col("value").rank("ordinal") <= lit(Int64(1)),
        col("value").rank("ordinal").over("group") == lit(Int64(1)),
    ]
    for predicate in misses:
        assert_true(not top_k_mask(bind(predicate, columns), columns, 65))
    _ = setting^


def test_merge_corank_and_every_output_slice_match_independent_stable_literal_order() raises:
    for left in [0, 1, 7, 8, 9, 31, 64]:
        for right in [0, 1, 7, 9, 32, 65]:
            var n = left + right
            var order = ScalarParallelOrder(7, False, False)
            var source = List[Int]()
            var l = scalar_parallel_rows(left, order)
            var r = List[Int]()
            for row in range(left, n):
                r.append(row)

            def before(a: Int, b: Int) {imm order} -> Bool:
                return order.less(a, b)

            sort(r, before)
            source.extend(l^)
            source.extend(r^)
            var expected = scalar_parallel_rows(n, order)
            for first in range(n + 1):
                var expected_left = 0
                for p in range(first):
                    expected_left += Int(expected[p] < left)
                assert_equal(
                    _co_rank(order, source, 0, left, n, first), expected_left
                )
                for last in range(first, n + 1):
                    var actual = List[Int](length=n + 2, fill=-777)
                    _merge_slice(order, source, actual, 0, left, n, first, last)
                    for p in range(n + 2):
                        assert_equal(
                            actual[p],
                            expected[p] if p >= first and p < last else -777,
                        )


def test_parallel_prefix_bucket_eligibility_fallback_and_full_string_ties_match_scalar_order() raises:
    var setting = WorkerSetting(4)
    assert_equal(configured_workers(), 4)
    var n = 32769
    for skew in [False, True]:
        for descending in [False, True]:
            var order = ScalarParallelOrder(257, skew, descending, True)
            var first = List[Int]()
            var primary = List[Int64]()
            var strings = List[String]()
            for row in range(n):
                var value = parallel_primary(row, 257, skew)
                first.append(-value - 1 if descending else value)
                primary.append(Int64(value))
                strings.append(parallel_string(row))
            var expected = scalar_parallel_rows(n, order)
            var bucketed = _prefix_buckets(ArcPointer(order.copy()), first, 4)
            assert_equal(Bool(bucketed), not skew)
            if bucketed:
                assert_equal(bucketed.take(), expected)
            assert_equal(comparison_arg_sort(order.copy(), n, first), expected)
            var columns: List[Series] = [
                Series("k", Column[Int64](primary^)),
                Series("s", Column[String](strings^)),
            ]
            assert_equal(
                row_arg_sort(columns, [descending, False], [True, True]),
                expected,
            )
    _ = setting^


def test_parallel_packed_heavy_buckets_and_string_settling_preserve_complete_stable_permutations() raises:
    var setting = WorkerSetting(4)
    # Invoke the heavy-bucket run/merge kernel itself. Public large examples
    # alone cannot prove that a particular bucket crossed its threshold.
    var data = List[UInt64]()
    var expected_words = List[UInt64]()
    data.append(UInt64.MAX)
    for row in range(65537):
        var value = (UInt64((row * 7919) % 31) << 32) | UInt64(row)
        data.append(value)
        expected_words.append(value)
    data.append(UInt64.MAX)
    sort(expected_words)
    var scratch = List[UInt64](length=len(data), fill=0)
    var pool = Pool(4)
    _sort_heavy[DType.uint64](pool, data, scratch, 1, 65538, 4)
    pool.release()
    assert_equal(data[0], UInt64.MAX)
    assert_equal(data[65538], UInt64.MAX)
    for row in range(65537):
        assert_equal(data[row + 1], expected_words[row])
    var n = 196609
    assert_true(worker_count(n) > 1)
    for skew in [False, True]:
        for descending in [False, True]:
            var primary = List[Int64]()
            var secondary = List[Int64]()
            var strings = List[String]()
            for row in range(n):
                primary.append(Int64(parallel_primary(row, 257, skew)))
                secondary.append(Int64(parallel_secondary(row)))
                strings.append(parallel_string(row))
            for string_key in [False, True]:
                var columns: List[Series] = [
                    Series("k", Column[Int64](primary.copy()))
                ]
                columns.append(
                    Series(
                        "s", Column[String](strings.copy())
                    ) if string_key else Series(
                        "s", Column[Int64](secondary.copy())
                    )
                )
                var actual = packed_arg_sort(
                    columns, [descending, False], [True, True]
                )
                assert_true(
                    Bool(actual),
                    "bounded integer and final string keys must pack",
                )
                var expected = scalar_parallel_rows(
                    n, ScalarParallelOrder(257, skew, descending, string_key)
                )
                assert_equal(actual.take(), expected)
    _ = setting^


def test_parallel_numeric_rank_large_partition_bucketed_and_sparse_one_sort_routes_reassociate_every_row() raises:
    from dataframe.rank import rank_numeric
    from tests.support.parallel_key_reference import expected_parallel_ranks

    var setting = WorkerSetting(4)
    var n = 131073
    assert_true(worker_count(n) > 1)
    var source = topk_frame(n, 257).column("value").rechunk()
    for route in range(3):
        var ids = List[Int]()
        if route > 0:
            for row in range(n):
                ids.append(row % 257 if route == 1 else row * 3 + 7)
        var partition_count = (
            1 if route == 0 else 257 if route == 1 else (n - 1) * 3 + 8
        )
        var threads = min(configured_workers(), n // 65536)
        if route < 2:
            assert_true(partition_count * threads <= n)
        else:
            assert_true(partition_count * threads > n)
        for method in ["ordinal", "min", "max", "dense", "average"]:
            for descending in [False, True]:
                var actual = rank_numeric(source, ids, method, descending)
                assert_true(Bool(actual))
                var got = actual.take()
                var expected = expected_parallel_ranks(
                    n, ids, method, descending
                )
                assert_equal(len(got), n)
                for row in range(n):
                    var cell = got.get(row)
                    assert_equal(cell.is_null(), not parallel_present(row + 7))
                    if parallel_present(row + 7):
                        assert_equal(
                            cell.float64() if method
                            == "average" else Float64(cell.int64()),
                            expected[row],
                        )
    _ = setting^


def test_parallel_rank_word_buckets_validate_domain_cutoff_skew_fallback_and_radix_stability() raises:
    from dataframe.series import _low_card_first_sort, sort_indices

    var setting = WorkerSetting(4)
    var n = 65537
    for distinct in [64, 65, 256, 257]:
        for skew in [False, True]:
            var primary = List[Int]()
            var secondary = List[Int]()
            for row in range(n):
                primary.append(parallel_primary(row, distinct, skew) - 128)
                secondary.append(parallel_secondary(row))
            var ranks: List[List[Int]] = [primary^, secondary^]
            var expected = scalar_parallel_rows(
                n, ScalarParallelOrder(distinct, skew, False)
            )
            var bucketed = _low_card_first_sort(ranks, 4)
            assert_equal(len(bucketed) > 0, distinct <= 256 and not skew)
            if len(bucketed) > 0:
                assert_equal(bucketed, expected)
            assert_equal(sort_indices(ranks), expected)
    _ = setting^


def test_row_prefix_long_string_remainders_precede_later_numeric_keys_in_parallel_merge_fallback() raises:
    var setting = WorkerSetting(4)
    var n = 32769
    var text = List[String]()
    var numbers = List[Int64]()
    var expected = List[Int]()
    for row in range(n):
        text.append(prefix_remainder_string(row))
        numbers.append(Int64(parallel_primary(row, 257, False)))
        expected.append(row)

    def before(a: Int, b: Int) -> Bool:
        var x = prefix_remainder_string(a)
        var y = prefix_remainder_string(b)
        if x != y:
            return x < y
        var u = parallel_primary(a, 257, False)
        var v = parallel_primary(b, 257, False)
        return u > v if u != v else a < b

    sort(expected, before)
    var columns: List[Series] = [
        Series("s", Column[String](text^)),
        Series("k", Column[Int64](numbers^)),
    ]
    assert_true(
        not packed_arg_sort(columns, [False, True], [True, True]),
        "nonfinal over-prefix string must reject packed encoding",
    )
    assert_equal(row_arg_sort(columns, [False, True], [True, True]), expected)
    _ = setting^
