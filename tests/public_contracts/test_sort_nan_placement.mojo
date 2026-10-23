"""NaN and null placement in sort, top_k, bottom_k and rank (#808).

Expected orders cover `sort(..., maintain_order=True)`, `top_k`, `bottom_k`
and `rank` on the same input. NaN is the largest value:
it follows the numbers ascending and precedes them descending. Nulls go first
or last as asked, in either direction.
"""
# contract-group: table.sort
# contract-group: expr.order_windows
from std.math import isnan
from std.testing import assert_equal, assert_true
from dataframe import AnyValue, Column, DataFrame, DataType, Series, col

# Above _MIN_PARALLEL_ROWS (16,384) in packed_sort, and large enough that a
# top_k of a few rows selects instead of sorting everything.
comptime _REPEAT = 3000


def _x_classes() -> List[Int]:
    """Pattern index -> equality class for the one-key pattern
    [2.0, NaN, null, 1.0, NaN, -0.0, 3.0]: the two NaNs are one value."""
    return [0, 1, 2, 3, 1, 4, 5]


def _ab_classes() -> List[Int]:
    """The same for the two-key pattern, where rows 6 and 9 are both
    (1.0, NaN)."""
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 6]


def _frame[
    D: DType
](
    values: List[Float64], valid: List[Bool], repeat: Int, name: String = "x"
) raises -> DataFrame:
    """One float column, the pattern repeated, and its row number "i"."""
    var column = List[Scalar[D]]()
    var present = List[Bool]()
    var ids = List[Int64]()
    for r in range(repeat):
        for j in range(len(values)):
            column.append(Scalar[D](values[j]))
            present.append(valid[j])
            ids.append(Int64(r * len(values) + j))
    return DataFrame(
        [
            Series(name, Column[Scalar[D]](column^, present)),
            Series("i", Column[Int64](ids^)),
        ]
    )


def _x_frame[D: DType](repeat: Int) raises -> DataFrame:
    var nan = Float64(0) / Float64(0)
    return _frame[D](
        [2.0, nan, 0.0, 1.0, nan, -0.0, 3.0],
        [True, True, False, True, True, True, True],
        repeat,
    )


def _ab_frame[D: DType](repeat: Int) raises -> DataFrame:
    var nan = Float64(0) / Float64(0)
    var a = _frame[D](
        [-1.0, nan, 1.0, -1.0, nan, 0.0, 1.0, nan, -1.0, 1.0],
        [True, True, True, True, True, False, True, True, True, True],
        repeat,
        "a",
    )
    var b = _frame[D](
        [nan, 2.0, -2.0, 0.0, -3.0, 1.0, nan, nan, -2.0, nan],
        [True, True, True, False, True, True, True, True, True, True],
        repeat,
        "b",
    )
    return DataFrame([a.column("a"), b.column("b"), a.column("i")])


def _repeated(small: List[Int], classes: List[Int], repeat: Int) -> List[Int64]:
    """The stable order of the pattern repeated `repeat` times, given its
    order once: equal rows keep row order, so each class's rows come out
    together, in the order the classes first appear in `small`."""
    var width = len(classes)
    var done = List[Bool](length=width, fill=False)
    var order = List[Int64]()
    for j in small:
        var c = classes[j]
        if done[c]:
            continue
        done[c] = True
        for row in range(repeat * width):
            if classes[row % width] == c:
                order.append(Int64(row))
    return order^


def _ids(frame: DataFrame) raises -> List[Int64]:
    var ids = List[Int64]()
    for row in range(frame.height()):
        ids.append(frame.item(row, "i").int64())
    return ids^


def _float(cell: AnyValue) raises -> Float64:
    if cell.dtype() == DataType.FLOAT32:
        return Float64(cell.float32())
    return cell.float64()


def _prefix(order: List[Int64], k: Int) -> List[Int64]:
    var head = List[Int64]()
    for j in range(min(k, len(order))):
        head.append(order[j])
    return head^


def _check_one_key(
    frame: DataFrame,
    descending: Bool,
    nulls_last: Bool,
    small: List[Int],
    repeat: Int,
) raises:
    var expected = _repeated(small, _x_classes(), repeat)
    assert_equal(_ids(frame.sort("x", descending, nulls_last)), expected)
    var source = frame.column("x")
    var order = source.argsort(descending, nulls_last)
    var as_ids = List[Int64]()
    for row in order:
        as_ids.append(Int64(row))
    assert_equal(as_ids, expected)
    var lazy = frame.lazy().sort(["x"], descending, nulls_last)
    assert_equal(_ids(lazy.collect()), expected)
    assert_equal(_ids(lazy.collect(streaming=False)), expected)
    # A sort followed by a slice runs as a top-k.
    var head = lazy.slice(0, 3)
    assert_equal(_ids(head.collect()), _prefix(expected, 3))
    assert_equal(_ids(head.collect(streaming=False)), _prefix(expected, 3))
    # The sorted values themselves, NaN included.
    var sorted = source.sort(descending, nulls_last)
    for row in range(len(expected)):
        var want = source.get(Int(expected[row]))
        var got = sorted.get(row)
        assert_equal(got.is_null(), want.is_null())
        if not want.is_null():
            if isnan(_float(want)):
                assert_true(isnan(_float(got)))
            else:
                assert_equal(_float(got), _float(want))


def _one_key_orders[D: DType]() raises:
    for repeat in [1, _REPEAT]:
        var frame = _x_frame[D](repeat)
        # (descending, nulls_last) and the Polars / DuckDB row order.
        _check_one_key(frame, False, False, [2, 5, 3, 0, 6, 1, 4], repeat)
        _check_one_key(frame, False, True, [5, 3, 0, 6, 1, 4, 2], repeat)
        _check_one_key(frame, True, False, [2, 1, 4, 6, 0, 3, 5], repeat)
        _check_one_key(frame, True, True, [1, 4, 6, 0, 3, 5, 2], repeat)


def test_one_float_key_puts_nan_first_descending_and_last_ascending() raises:
    _one_key_orders[DType.float64]()
    _one_key_orders[DType.float32]()


def _check_keys(
    frame: DataFrame,
    by: List[String],
    descending: List[Bool],
    nulls_last: List[Bool],
    small: List[Int],
    classes: List[Int],
    repeat: Int,
) raises:
    var expected = _repeated(small, classes, repeat)
    assert_equal(
        _ids(frame.sort(by, descending=descending, nulls_last=nulls_last)),
        expected,
    )
    var lazy = frame.lazy().sort(
        by, descending=descending, nulls_last=nulls_last
    )
    assert_equal(_ids(lazy.collect()), expected)
    assert_equal(_ids(lazy.collect(streaming=False)), expected)


def _multi_key_orders[D: DType]() raises:
    var nan = Float64(0) / Float64(0)
    var g = Series(
        "g",
        Column[Int64](
            [1, 2, 1, 2, 1, 2, 0, 1, 2, 1],
            [True, True, True, True, True, True, False, True, True, True],
        ),
    )
    var x = _frame[D](
        [nan, 1.0, 0.0, nan, 3.0, -1.0, nan, 1.0, 0.0, nan],
        [True, True, False, True, True, True, True, True, False, True],
        1,
    )
    var gx = DataFrame([g^, x.column("x"), x.column("i")])
    var distinct: List[Int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    var by: List[String] = ["g", "x"]
    _check_keys(
        gx,
        by,
        [False, True],
        [True, True],
        [0, 9, 4, 7, 2, 3, 1, 5, 8, 6],
        distinct,
        1,
    )
    _check_keys(
        gx,
        by,
        [True, False],
        [False, True],
        [6, 5, 1, 3, 8, 7, 4, 0, 9, 2],
        distinct,
        1,
    )
    _check_keys(
        gx,
        by,
        [True, True],
        [True, False],
        [8, 3, 1, 5, 2, 0, 9, 4, 7, 6],
        distinct,
        1,
    )
    _check_keys(
        gx,
        by,
        [False, True],
        [False, False],
        [6, 2, 0, 9, 4, 7, 8, 3, 1, 5],
        distinct,
        1,
    )
    # Two float keys that span both signs: too wide to pack into one
    # integer, so these take the word-encoded sort.
    var ab_by: List[String] = ["a", "b"]
    for repeat in [1, _REPEAT]:
        var ab = _ab_frame[D](repeat)
        _check_keys(
            ab,
            ab_by,
            [True, True],
            [True, True],
            [7, 1, 4, 6, 9, 2, 0, 8, 3, 5],
            _ab_classes(),
            repeat,
        )
        _check_keys(
            ab,
            ab_by,
            [True, False],
            [True, False],
            [4, 1, 7, 2, 6, 9, 3, 8, 0, 5],
            _ab_classes(),
            repeat,
        )
        _check_keys(
            ab,
            ab_by,
            [False, True],
            [False, True],
            [5, 0, 8, 3, 6, 9, 2, 7, 1, 4],
            _ab_classes(),
            repeat,
        )


def test_multi_key_sorts_place_nan_by_each_key_direction() raises:
    _multi_key_orders[DType.float64]()
    _multi_key_orders[DType.float32]()


def _top_and_bottom[D: DType]() raises:
    # Polars top_k / bottom_k with k covering every row; a smaller k takes
    # a prefix.
    for repeat in [1, _REPEAT]:
        var frame = _x_frame[D](repeat)
        var want_top = _repeated([1, 4, 6, 0, 3, 5, 2], _x_classes(), repeat)
        var want_bottom = _repeated([5, 3, 0, 6, 1, 4, 2], _x_classes(), repeat)
        for k in [1, 2, 3, 5, 7]:
            assert_equal(_ids(frame.top_k(k, "x")), _prefix(want_top, k))
            assert_equal(_ids(frame.bottom_k(k, "x")), _prefix(want_bottom, k))
    # Two keys, both with NaN: Polars top_k(by=["a", "b"]) and bottom_k.
    var by: List[String] = ["a", "b"]
    for repeat in [1, _REPEAT]:
        var ab = _ab_frame[D](repeat)
        var want_top = _repeated(
            [7, 1, 4, 6, 9, 2, 0, 8, 3, 5], _ab_classes(), repeat
        )
        var want_bottom = _repeated(
            [8, 0, 3, 2, 6, 9, 4, 1, 7, 5], _ab_classes(), repeat
        )
        for k in [1, 4, 10]:
            assert_equal(_ids(ab.top_k(k, by)), _prefix(want_top, k))
            assert_equal(_ids(ab.bottom_k(k, by)), _prefix(want_bottom, k))


def test_top_k_takes_nan_first_and_bottom_k_takes_it_last() raises:
    _top_and_bottom[DType.float64]()
    _top_and_bottom[DType.float32]()


def _check_rank(
    frame: DataFrame,
    method: String,
    descending: Bool,
    want: List[Float64],
) raises:
    """Row 2 is null and ranks null; `want` holds 0 there."""
    var ranks = frame.select(col("x").rank(method, descending)).column("x")
    for row in range(len(want)):
        var cell = ranks.get(row)
        assert_equal(cell.is_null(), row == 2)
        if row != 2:
            var got = cell.float64() if method == "average" else Float64(
                cell.int64()
            )
            assert_equal(got, want[row])


def _ranks[D: DType]() raises:
    var frame = _x_frame[D](1)
    # Polars rank(method) and rank(method, descending=True).
    _check_rank(frame, "average", False, [3, 5.5, 0, 2, 5.5, 1, 4])
    _check_rank(frame, "average", True, [4, 1.5, 0, 5, 1.5, 6, 3])
    _check_rank(frame, "min", False, [3, 5, 0, 2, 5, 1, 4])
    _check_rank(frame, "min", True, [4, 1, 0, 5, 1, 6, 3])
    _check_rank(frame, "max", False, [3, 6, 0, 2, 6, 1, 4])
    _check_rank(frame, "max", True, [4, 2, 0, 5, 2, 6, 3])
    _check_rank(frame, "dense", False, [3, 5, 0, 2, 5, 1, 4])
    _check_rank(frame, "dense", True, [3, 1, 0, 4, 1, 5, 2])
    _check_rank(frame, "ordinal", False, [3, 5, 0, 2, 6, 1, 4])
    _check_rank(frame, "ordinal", True, [4, 1, 0, 5, 2, 6, 3])


def test_rank_treats_nan_as_the_largest_value() raises:
    _ranks[DType.float64]()
    _ranks[DType.float32]()
