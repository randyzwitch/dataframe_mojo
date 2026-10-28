"""Actual numeric conjunction, sorted-window and selective filter contracts."""
# contract-group: kernel.row_ops
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, Series, col, lit, concat
from dataframe.fused_compare import (
    fused_comparisons,
    fused_mask,
    selection_tree,
)
from dataframe.sorted_range import sorted_window
from dataframe.parallel import worker_count
from tests.support.lifecycle_jobs import WorkerSetting

comptime FILTER_DTYPES = (
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float32,
    DType.float64,
)


def scalar_predicate(value: Int, op: Int) -> Bool:
    if op == 0:
        return value == 5
    if op == 1:
        return value != 5
    if op == 2:
        return value < 5
    if op == 3:
        return value <= 5
    if op == 4:
        return value > 5
    return value >= 5


def fused_dtype[D: DType]() raises:
    for n in [0, 1, 9, 1023, 1024, 1025]:
        for offset in range(8):
            for mode in range(3):
                var values = List[Scalar[D]]()
                var valid = List[Bool]()
                for row in range(n + offset):
                    var present = mode == 0 or (mode == 1 and row % 5 != 2)
                    values.append(
                        Scalar[D](row % 11) if present else Scalar[D].MAX
                    )
                    valid.append(present)
                var column = Series(
                    "x", Column[Scalar[D]](values^, valid)
                ).slice(offset, n)
                var columns: List[Series] = [column.copy()]
                var constant = lit(Scalar[D](5))
                var parts: List[Expr] = [
                    col("x") == constant.copy(),
                    col("x") != constant.copy(),
                    col("x") < constant.copy(),
                    constant.copy() >= col("x"),
                    col("x") > constant.copy(),
                    constant.copy() <= col("x"),
                ]
                for op in range(6):
                    var result = fused_comparisons(
                        columns, n, [parts[op].copy()]
                    )
                    assert_equal(result.used, List[Bool]([n > 0]))
                    if n == 0:
                        assert_equal(len(result.mask), 0)
                        continue
                    var mask = fused_mask(result.mask, n)
                    for row in range(n):
                        var physical = row + offset
                        var present = mode == 0 or (
                            mode == 1 and physical % 5 != 2
                        )
                        assert_true(mask._valid(row))
                        assert_equal(
                            mask._get(row),
                            present and scalar_predicate(physical % 11, op),
                        )
                if n > 1:
                    var chunked = Series._from_chunks(
                        [column.slice(0, 1), column.slice(1, n - 1)]
                    )
                    var miss = fused_comparisons(
                        [chunked^], n, [parts[0].copy()]
                    )
                    assert_equal(miss.used, List[Bool]([False]))
                    assert_equal(len(miss.mask), 0)


def test_fused_comparison_dtype_operator_bit_offset_and_tail_routes_match_independent_scalars() raises:
    comptime for i in range(len(FILTER_DTYPES)):
        fused_dtype[FILTER_DTYPES[i]]()


def test_selective_numeric_first_string_last_narrowing_and_fused_fallback_keep_complete_rows() raises:
    var setting = WorkerSetting(4)
    var n = 131073
    assert_true(worker_count(n) > 1)
    var numbers = List[Int32]()
    var others = List[Int32]()
    var valid = List[Bool]()
    var strings = List[String]()
    var string_valid = List[Bool]()
    for row in range(n + 7):
        numbers.append(Int32(row % 100))
        others.append(Int32(row % 31))
        valid.append(row % 13 != 2)
        strings.append("" if row % 5 == 0 else "value" + String(row % 7))
        string_valid.append(row % 17 != 3)
    var source = DataFrame(
        [
            Series("x", Column[Int32](numbers^, valid.copy())).slice(7, n),
            Series("y", Column[Int32](others^, valid)).slice(7, n),
            Series("s", Column[String](strings^, string_valid)).slice(7, n),
        ]
    )
    for chunked in [False, True]:
        var frame = source.copy()
        if chunked:
            var columns = List[Series]()
            for name in source.columns():
                var value = source.column(name)
                columns.append(
                    Series._from_chunks(
                        [value.slice(0, 65535), value.slice(65535, n - 65535)]
                    )
                )
            frame = DataFrame(columns^)
        for selective in [False, True]:
            var threshold = Int32(2 if selective else 90)
            var parts: List[Expr] = [
                col("s").ne(""),
                col("x") < lit(threshold),
                col("x") > col("y"),
            ]
            var columns: List[Series] = [
                frame.column("x"),
                frame.column("y"),
                frame.column("s"),
            ]
            var fused = fused_comparisons(columns, n, parts)
            assert_equal(fused.used, List[Bool]([False, not chunked, False]))
            var mask = frame._selective_mask(parts, 1025)
            var actual = frame._filter_selective(parts, 1025, frame.columns())
            var output = 0
            for row in range(n):
                var physical = row + 7
                var keep = (
                    physical % 13 != 2
                    and physical % 17 != 3
                    and physical % 5 != 0
                    and physical % 100 < Int(threshold)
                    and physical % 100 > physical % 31
                )
                assert_equal(mask._valid(row), True)
                assert_equal(mask._get(row), keep)
                if not keep:
                    continue
                assert_equal(
                    actual.item(output, "x").int32(), Int32(physical % 100)
                )
                assert_equal(
                    actual.item(output, "y").int32(), Int32(physical % 31)
                )
                assert_equal(
                    actual.item(output, "s").string(),
                    "value" + String(physical % 7),
                )
                output += 1
            assert_equal(actual.height(), output)
    _ = setting^


def test_selection_trees_narrow_kept_rows_with_and_or_comparisons_and_null_tests() raises:
    # After a selective first part, later parts made of AND/OR comparisons
    # and null tests read the source columns at the kept rows. Columns are
    # sliced so validity bitmaps start mid-byte, and nulls sit in every
    # column the trees read.
    var setting = WorkerSetting(4)
    var n = 131073
    var quantities = List[Int64]()
    var quantity_valid = List[Bool]()
    var prices = List[Float64]()
    var price_valid = List[Bool]()
    var codes = List[Int32]()
    var code_valid = List[Bool]()
    for row in range(n + 7):
        quantities.append(Int64(row % 100))
        quantity_valid.append(row % 13 != 2)
        prices.append(Float64(row % 40))
        price_valid.append(row % 9 != 4)
        codes.append(Int32(row % 97))
        code_valid.append(row % 11 != 5)
    var frame = DataFrame(
        [
            Series("q", Column[Int64](quantities^, quantity_valid)).slice(7, n),
            Series("a", Column[Float64](prices^, price_valid)).slice(7, n),
            Series("b", Column[Int32](codes^, code_valid)).slice(7, n),
        ]
    )
    var band = (
        (col("a") >= lit(Float64(10))) & (col("a") <= lit(Float64(20)))
        | (col("b") > lit(Int32(90)))
        | col("a").is_null()
    )
    for selective in [True, False]:
        var limit = 5 if selective else 90
        var parts: List[Expr] = [
            col("q") < lit(Int64(limit)),
            band.copy(),
            col("b").is_not_null(),
        ]
        var columns = frame._columns.copy()
        assert_true(Bool(selection_tree(parts[1], columns)))
        assert_true(Bool(selection_tree(parts[2], columns)))
        assert_true(not selection_tree(col("a") > col("b"), columns))
        var mask = frame._selective_mask(parts, 1025)
        var actual = frame._filter_selective(parts, 1025, frame.columns())
        var output = 0
        for row in range(n):
            var p = row + 7
            var a_valid = p % 9 != 4
            var b_valid = p % 11 != 5
            var a = p % 40
            var b = p % 97
            var keep = (
                p % 13 != 2
                and p % 100 < limit
                and (
                    (a_valid and a >= 10 and a <= 20)
                    or (b_valid and b > 90)
                    or not a_valid
                )
                and b_valid
            )
            assert_equal(mask._get(row), keep)
            if keep:
                assert_equal(actual.item(output, "q").int64(), Int64(p % 100))
                output += 1
        assert_equal(actual.height(), output)
    _ = setting^


def ordered_series(n: Int, base: Int, nullable: Bool = False) raises -> Series:
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(n):
        values.append(Int64(base + row // 3))
        valid.append(not nullable or row % 11 != 2)
    return Series("x", Column[Int64](values^, valid))


def test_sorted_window_order_cache_is_shared_by_views_and_invalidated_by_descending_boundary_reassembly() raises:
    var setting = WorkerSetting(4)
    var n = 65537
    var source = ordered_series(n, 0)
    var parts: List[Expr] = [
        col("x") >= lit(Int64(3)),
        col("x") <= lit(Int64(9)),
    ]
    assert_equal(source._data[Column[Int64]]._facts[].order_state(), 0)
    var first = sorted_window([source.copy()], n, parts)
    assert_equal(first.used, List[Bool]([True, True]))
    assert_equal(first.low, 9)
    assert_equal(first.high, 30)
    assert_equal(source._data[Column[Int64]]._facts[].order_state(), 1)
    var view = source.slice(7, 65)
    assert_equal(
        Int(view._data[Column[Int64]]._facts.ptr()),
        Int(source._data[Column[Int64]]._facts.ptr()),
    )
    var window = sorted_window([view^], 65, parts)
    assert_equal(window.low, 2)
    assert_equal(window.high, 23)
    var left = DataFrame([source.copy()])
    var right = DataFrame([ordered_series(n, -100)])
    for rechunked in [False, True]:
        var joined = concat([left.copy(), right.copy()])
        if rechunked:
            joined = joined.rechunk()
            assert_equal(
                joined.column("x")._data[Column[Int64]]._facts[].order_state(),
                0,
            )
        var found = sorted_window([joined.column("x")], 2 * n, parts)
        assert_equal(found.used, List[Bool]([False, False]))
        assert_equal(found.low, 0)
        assert_equal(found.high, 2 * n)
        var actual = joined.filter(parts[0].copy() & parts[1])
        assert_equal(actual.height(), 42)
        for row in range(42):
            assert_equal(
                actual.item(row, "x").int64(), Int64(3 + (row % 21) // 3)
            )
    var nulls = ordered_series(65, 0, True)
    var miss = sorted_window([nulls^], 65, parts)
    assert_equal(miss.used, List[Bool]([False, False]))
    _ = setting^
