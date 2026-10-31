"""Integer membership and Bool equality retain packed physical-window rules."""
# contract-group: expr.membership
# contract-group: expr.chunk_windows
from std.sys.info import simd_width_of
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    BoolColumn,
    StringColumn,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
    when,
)
from dataframe.expr import EQ, null
from dataframe.kernels.binary import binary, integer_is_in
from tests.support.coverage_compare import assert_frame


def _cold(source: Series) raises:
    if source.is_chunked():
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))


def _pieces(source: Series) raises -> Series:
    var n = len(source)
    if n < 3:
        var parts = List[Series]()
        for row in range(n):
            parts.append(source.slice(row, 1))
        if n == 0:
            parts.append(source.copy())
        return Series._from_chunks(parts)
    return Series._from_chunks(
        [
            source.slice(0, 1),
            source.slice(1, 2),
            source.slice(3, 0),
            source.slice(3, n - 3),
        ]
    )


def _same(actual: Series, expected: Series) raises:
    assert_frame(
        DataFrame([actual.renamed("answer")]),
        DataFrame([expected.renamed("answer")]),
    )


def _wide[D: DType]() -> Scalar[D]:
    comptime if D == DType.uint64:
        return Scalar[D]((UInt64(1) << 63) + 5)
    elif D == DType.int64:
        return Scalar[D](Int64(9007199254740993))
    else:
        return Scalar[D](7)


def _integer_window[D: DType](offset: Int, n: Int) raises:
    var seed: List[Scalar[D]] = [
        0,
        1,
        2,
        7,
        Scalar[D].MIN,
        Scalar[D].MAX,
        1,
        Scalar[D].MAX,
        0,
        2,
        _wide[D](),
        Scalar[D].MIN,
    ]
    var flags: List[Bool] = [
        False,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        False,
    ]
    comptime if D != DType.int64 and D != DType.uint64:
        flags[3] = True
    var values = List[Scalar[D]]()
    var validity = List[Bool]()
    for row in range(offset + n + 1):
        values.append(seed[row % 12])
        validity.append(row % 12 != 6 and row % 12 != 7 and row % 12 != 8)
    var flat = Series("x", Column[Scalar[D]](values^, validity)).slice(
        offset, n
    )
    var source = _pieces(flat)
    var text = (
        String("1,")
        + String(Scalar[D].MAX.cast[DType.int64]())
        + ","
        + String(_wide[D]().cast[DType.int64]())
    )
    var actual = integer_is_in(source, text)
    var want = List[Bool]()
    var valid = List[Bool]()
    for row in range(n):
        var phase = (offset + row) % 12
        want.append(flags[phase])
        valid.append(phase != 6 and phase != 7 and phase != 8)
    assert_equal(actual.name(), "x")
    assert_true(actual.dtype() == DataType.BOOL)
    _same(actual, Series("x", BoolColumn(want, valid)))
    for row in range(n):
        assert_equal(actual.bool()._get(row), want[row])
    _same(actual, integer_is_in(flat, text))
    _cold(source)
    var members: List[Expr] = [
        lit(Scalar[D](1)),
        lit(Scalar[D].MAX),
        null(DataType.of(D)),
        lit(_wide[D]()),
        lit(Scalar[D](1)),
    ]
    var public_result = (
        DataFrame([source.copy()])
        .lazy()
        .select_exprs([col("x").is_in(members).alias("member")])
        .collect(engine="cpu", batch_size=3)
    )
    assert_equal(public_result.columns()[0], "member")
    _same(public_result.column("member"), Series("x", BoolColumn(want, valid)))
    _cold(source)


def test_integer_membership_all_widths_offsets_and_vector_tails() raises:
    comptime kinds = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    )
    comptime for k in range(len(kinds)):
        _integer_window[kinds[k]](3, 41)
    for offset in range(8):
        _integer_window[DType.int64](offset, 41)
        _integer_window[DType.uint64](offset, 41)
    for n in [0, 1, 7, 8, 9]:
        _integer_window[DType.int8](1, n)


def _owned_membership() raises -> Series:
    var flat = Series(
        "x",
        Column[UInt64](
            [
                UInt64.MAX,
                (UInt64(1) << 63) + 5,
                (UInt64(1) << 63) + 6,
                0,
                UInt64.MAX,
            ],
            [True, True, True, True, False],
        ),
    )
    var source = _pieces(flat)
    var result = integer_is_in(source, "-1,-9223372036854775803")
    _cold(source)
    return result^


def test_integer_membership_exact_uint64_ownership_and_published_cache() raises:
    var expected = Series(
        "x",
        BoolColumn(
            [True, True, False, False, True], [True, True, True, True, False]
        ),
    )
    var owned = _owned_membership()
    _same(owned, expected)
    assert_true(owned.bool()._get(4))
    var flat = Series(
        "x",
        Column[Int64](
            [9007199254740993, 9007199254740992, Int64.MAX, Int64.MIN]
        ),
    )
    var source = _pieces(flat)
    _same(
        integer_is_in(
            source, "9007199254740993,9223372036854775807,-9223372036854775808"
        ),
        Series("x", BoolColumn([True, False, True, True])),
    )
    _cold(source)
    var merged = source.rechunk()
    _ = merged^
    _same(
        integer_is_in(source, "9007199254740993"),
        Series("x", BoolColumn([True, False, False, False])),
    )
    assert_equal(source._chunked.value()[].cache[].state.load(), Int64(2))


def _parser_outcome(source: Series, text: String) raises:
    # The retained contiguous function is the frozen parser/wrapping adapter.
    # Empty/malformed atol behavior is compared, never assumed to be strict.
    var flat = source.rechunk()
    var expected = Series("x", BoolColumn([]))
    var failed = False
    var message = String()
    try:
        expected = integer_is_in(flat, text)
    except e:
        failed = True
        message = String(e)
    var fresh = Series._from_chunks(
        [flat.slice(0, 1), flat.slice(1, len(flat) - 1)]
    )
    if failed:
        var observed = False
        try:
            _ = integer_is_in(fresh, text)
        except e:
            observed = True
            assert_equal(String(e), message)
        assert_true(observed)
    else:
        var actual = integer_is_in(fresh, text)
        _same(actual, expected)
        for row in range(len(flat)):
            assert_equal(actual.bool()._get(row), expected.bool()._get(row))
    _cold(fresh)


def test_integer_membership_private_parser_and_temporal_fallback_are_unchanged() raises:
    var bytes = _pieces(
        Series("x", Column[UInt8]([255, 1, 0, 7], [True, True, False, True]))
    )
    _same(
        integer_is_in(bytes, "-1,257"),
        Series(
            "x",
            BoolColumn([True, True, False, False], [True, True, False, True]),
        ),
    )
    _cold(bytes)
    for text in ["", "not-an-integer", "1,,2", "18446744073709551615"]:
        _parser_outcome(bytes, text)
    var dates = _pieces(
        Series(
            "x", Column[Int64]([1, 2, 99, 1], [True, True, False, True])
        ).with_dtype(DataType.DATE)
    )
    _same(
        integer_is_in(dates, "1"),
        Series(
            "x",
            BoolColumn([True, False, False, True], [True, True, False, True]),
        ),
    )
    var result = DataFrame([dates.copy()]).select_exprs(
        [col("x").is_in([lit(Int64(1), DataType.DATE)]).alias("member")]
    )
    _same(
        result.column("member"),
        Series(
            "x",
            BoolColumn([True, False, False, True], [True, True, False, True]),
        ),
    )

    # with_dtype enforces physical compatibility; the private Storage
    # constructor can nevertheless carry a mismatched plain-integer tag.
    var physical = Series(
        "x", Column[Int64]([4294967297, 1, 2, 1], [True, True, False, True])
    )
    with assert_raises(contains="cannot tag"):
        _ = physical.with_dtype(DataType.INT32)
    var wrong_tag = Series("x", physical._data.copy(), DataType.INT32)
    var wrong_chunks = _pieces(wrong_tag)
    var wrong_result = integer_is_in(wrong_chunks, "4294967297")
    _same(wrong_result, integer_is_in(wrong_tag, "4294967297"))
    _same(
        wrong_result,
        Series(
            "x",
            BoolColumn([True, False, False, False], [True, True, False, True]),
        ),
    )


def _bool_source(right: Bool, n: Int) raises -> Series:
    var seed = List[Bool]()
    var valid = List[Bool]()
    if right:
        seed = [True, False, True, True, False, True, True, False, True]
        valid = [True, True, False, True, True, False, True, True, False]
    else:
        seed = [True, True, True, False, False, False, True, True, True]
        valid = [True, True, True, True, True, True, False, False, False]
    var values = List[Bool]()
    var present = List[Bool]()
    for row in range(n):
        values.append(seed[row % 9])
        present.append(valid[row % 9])
    return Series("y" if right else "x", BoolColumn(values, present))


def _bool_pair(offset: Int) raises:
    comptime W = simd_width_of[DType.uint8]()
    comptime n = 32 * W + 9
    var xf = _bool_source(False, n + offset + 1).slice(offset, n)
    var yf = _bool_source(True, n + offset + 1).slice(offset, n)
    var x = Series._from_chunks(
        [
            xf.slice(0, 3),
            xf.slice(3, 16 * W + 1),
            xf.slice(16 * W + 4, 0),
            xf.slice(16 * W + 4, 16 * W + 5),
        ]
    )
    var y = Series._from_chunks(
        [
            yf.slice(0, 2),
            yf.slice(2, 12 * W + 1),
            yf.slice(12 * W + 3, 0),
            yf.slice(12 * W + 3, 20 * W + 6),
        ]
    )
    var eq: List[Bool] = [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    var valid: List[Bool] = [
        True,
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    var expected_values = List[Bool]()
    var expected_valid = List[Bool]()
    var membership_values = List[Bool]()
    var membership_valid = List[Bool]()
    for row in range(n):
        var p = (row + offset) % 9
        expected_values.append(eq[p])
        expected_valid.append(valid[p])
        membership_values.append(eq[p])
        membership_valid.append(p < 6)
    var expected = Series("a", BoolColumn(expected_values, expected_valid))
    for mask in [
        List[Bool](),
        List[Bool](length=n, fill=False),
        List[Bool]([False, True]),
    ]:
        var result = binary[EQ](x, y, mask)
        _same(result, expected)
        for row in range(n):
            if not result.bool()._valid(row):
                assert_false(result.bool()._get(row))
    _same(binary[EQ](x, y), binary[EQ](xf, yf))
    var whole = DataFrame([x.copy(), y.copy()]).select_exprs(
        [col("x").is_in([col("y")]).alias("member")], batch_size=17
    )
    _same(
        whole.column("member"),
        Series("a", BoolColumn(membership_values, membership_valid)),
    )
    _cold(x)
    _cold(y)


def test_bool_equality_and_dynamic_membership_keep_distinct_null_rules() raises:
    for offset in range(8):
        _bool_pair(offset)


def _singleton_bools(right: Bool) raises -> Series:
    var values: List[Bool] = [False, False, False]
    var valid: List[Bool] = [True, True, True]
    for row in range(1000):
        var phase = row % 4
        values.append(phase != 2 if right else phase != 1)
        valid.append(phase != 1 if right else phase != 2)
    var flat = Series("y" if right else "x", BoolColumn(values, valid))
    var parts = List[Series]()
    for row in range(1000):
        parts.append(flat.slice(row + 3, 1))
    return Series._from_chunks(parts)


def _eq_raw(actual: Series, expected: Series) raises:
    assert_equal(actual.name(), "")
    _same(actual, expected)
    var values = actual.bool()
    var wanted = expected.bool()
    for row in range(len(actual)):
        assert_equal(values._get(row), wanted._get(row))


def _published_bool_eq() raises:
    var x = _singleton_bools(False)
    var y = _singleton_bools(True)
    assert_equal(x.n_chunks(), 1000)
    assert_equal(y.n_chunks(), 1000)
    var one = Series("a", BoolColumn([True]))
    var scalar_values = List[Bool]()
    var scalar_valid = List[Bool]()
    var pair_values = List[Bool]()
    var pair_valid = List[Bool]()
    for row in range(1000):
        var phase = row % 4
        scalar_values.append(phase == 0 or phase == 3)
        scalar_valid.append(phase != 2)
        pair_values.append(phase == 0 or phase == 3)
        pair_valid.append(phase == 0 or phase == 3)
    var scalar_expected = Series("a", BoolColumn(scalar_values, scalar_valid))
    var pair_expected = Series("a", BoolColumn(pair_values, pair_valid))
    _eq_raw(binary[EQ](x, one), scalar_expected)
    _eq_raw(binary[EQ](x, y), pair_expected)
    _cold(x)
    _cold(y)
    var cached_x = x.rechunk()
    _ = cached_x^
    var masks = List[List[Bool]]()
    masks.append([])
    masks.append(List[Bool](length=1000, fill=False))
    masks.append([False, True])
    for mask in masks:
        _eq_raw(binary[EQ](x, one, mask), scalar_expected)
        _eq_raw(binary[EQ](one, x, mask), scalar_expected)
        _eq_raw(binary[EQ](x, y, mask), pair_expected)
        _eq_raw(binary[EQ](y, x, mask), pair_expected)
        assert_equal(x._chunked.value()[].cache[].state.load(), Int64(2))
        _cold(y)
    var cached_y = y.rechunk()
    _ = cached_y^
    for mask in masks:
        _eq_raw(binary[EQ](x, y, mask), pair_expected)
        _eq_raw(binary[EQ](y, x, mask), pair_expected)
        assert_equal(x._chunked.value()[].cache[].state.load(), Int64(2))
        assert_equal(y._chunked.value()[].cache[].state.load(), Int64(2))


def test_bool_equality_whole_scalars_empty_lengths_and_masks() raises:
    _published_bool_eq()
    var xf = Series(
        "x", BoolColumn([True, False, True, False], [True, True, False, False])
    )
    var x = Series._from_chunks(
        [xf.slice(0, 1), xf.slice(1, 1), xf.slice(2, 1), xf.slice(3, 1)]
    )
    var one = Series("y", BoolColumn([True]))
    var no = Series("y", BoolColumn([False]))
    var missing = Series("y", BoolColumn([True], [False]))
    _same(
        binary[EQ](x, one),
        Series(
            "a",
            BoolColumn([True, False, False, False], [True, True, False, False]),
        ),
    )
    _same(
        binary[EQ](one, x),
        Series(
            "a",
            BoolColumn([True, False, False, False], [True, True, False, False]),
        ),
    )
    _same(
        binary[EQ](x, no),
        Series(
            "a",
            BoolColumn([False, True, False, False], [True, True, False, False]),
        ),
    )
    _same(
        binary[EQ](no, x),
        Series(
            "a",
            BoolColumn([False, True, False, False], [True, True, False, False]),
        ),
    )
    _same(
        binary[EQ](x, missing),
        Series(
            "a",
            BoolColumn(
                [False, False, False, False], [False, False, False, False]
            ),
        ),
    )
    _same(
        binary[EQ](missing, x),
        Series(
            "a",
            BoolColumn(
                [False, False, False, False], [False, False, False, False]
            ),
        ),
    )
    _cold(x)
    var empty = Series("y", BoolColumn([]))
    assert_equal(len(binary[EQ](empty, one)), 0)
    assert_equal(len(binary[EQ](one, empty)), 0)
    with assert_raises(contains="Incompatible expression lengths"):
        _ = binary[EQ](empty, x.slice(0, 2))
    with assert_raises(contains="Incompatible expression lengths"):
        _ = binary[EQ](x.slice(0, 2), empty)

    # A private Bool dtype tag need not describe BoolColumn storage. Such
    # inputs retain physical numeric equality and its observation-mask rule.
    var a = Series("a", Column[Int64]([1, 2, 0, 99], [True, True, True, False]))
    var b = Series("b", Column[Int64]([1, 1, 0, 99], [True, True, False, True]))
    var tagged_a = Series("a", a._data.copy(), DataType.BOOL)
    var tagged_b = Series("b", b._data.copy(), DataType.BOOL)
    var chunked_a = _pieces(tagged_a)
    var chunked_b = _pieces(tagged_b)
    var private_eq = binary[EQ](chunked_a, chunked_b)
    _same(private_eq, binary[EQ](tagged_a, tagged_b))
    _same(
        private_eq,
        Series(
            "a",
            BoolColumn([True, False, False, False], [True, True, False, False]),
        ),
    )
    var masked = binary[EQ](
        chunked_a, chunked_b, List[Bool](length=4, fill=False)
    )
    _same(
        masked, binary[EQ](tagged_a, tagged_b, List[Bool](length=4, fill=False))
    )
    _same(
        masked,
        Series(
            "a",
            BoolColumn(
                [False, False, False, False], [False, False, False, False]
            ),
        ),
    )


def test_membership_public_empty_null_candidates_and_dtype_errors() raises:
    var source = _pieces(
        Series("x", Column[Int64]([1, 2, 1, 0], [True, True, False, True]))
    )
    var frame = DataFrame([source.copy()])
    for candidates in [List[Expr](), List[Expr]([null(DataType.INT64)])]:
        _same(
            frame.select_exprs([col("x").is_in(candidates).alias("a")]).column(
                "a"
            ),
            Series(
                "a",
                BoolColumn(
                    [False, False, False, False], [True, True, False, True]
                ),
            ),
        )
    for selected in [frame.copy(), frame.clear()]:
        with assert_raises(contains="matching dtypes"):
            _ = selected.select_exprs([col("x").is_in([lit(Int32(1))])])
        with assert_raises(contains="matching dtypes"):
            _ = selected.select_exprs(
                [
                    when(lit(False))
                    .then(col("x").is_in([lit(Int32(1))]))
                    .otherwise(lit(False))
                ]
            )
    _cold(source)

    var bools = _pieces(
        Series(
            "x",
            BoolColumn([True, False, True, False], [True, True, False, True]),
        )
    )
    var boolean_frame = DataFrame([bools.copy()])
    _same(
        boolean_frame.select_exprs(
            [col("x").is_in([lit(True)]).alias("a")]
        ).column("a"),
        Series(
            "a",
            BoolColumn([True, False, False, False], [True, True, False, True]),
        ),
    )
    _same(
        boolean_frame.select_exprs(
            [
                col("x")
                .is_in([lit(True), null(DataType.BOOL), lit(False)])
                .alias("a")
            ]
        ).column("a"),
        Series(
            "a",
            BoolColumn([True, True, False, True], [True, True, False, True]),
        ),
    )
    _same(
        boolean_frame.select_exprs(
            [col("x").is_in(List[Expr]()).alias("a")]
        ).column("a"),
        Series(
            "a",
            BoolColumn([False, False, False, False], [True, True, False, True]),
        ),
    )
    for selected in [boolean_frame.copy(), boolean_frame.clear()]:
        with assert_raises(contains="matching dtypes"):
            _ = selected.select_exprs([col("x").is_in([lit(Int64(1))])])
    _cold(bools)


def test_float_general_membership_retains_nan_zero_and_dynamic_null_policy() raises:
    # Existing Float numeric EQ/windows are a regression control, not a new
    # membership specialization or a Float performance claim.
    var nan = Float64(0) / Float64(0)
    var xflat = Series(
        "x",
        Column[Float64](
            [88, 88, 88, nan, 0.0, -0.0, 1, 99, 2, nan, 88],
            [True, True, True, True, True, True, True, False, True, True, True],
        ),
    ).slice(3, 7)
    var yflat = Series(
        "y",
        Column[Float64](
            [77, 77, 77, nan, -0.0, 0.0, 88, 4, 2, 0, 77],
            [True, True, True, True, True, True, False, True, True, True, True],
        ),
    ).slice(3, 7)
    var x = _pieces(xflat)
    var y = Series._from_chunks(
        [yflat.slice(0, 2), yflat.slice(2, 1), yflat.slice(3, 4)]
    )
    var frame = DataFrame([x.copy(), y.copy()])
    var actual = frame.select_exprs(
        [
            col("x").is_in([lit(nan)]).alias("nan"),
            col("x").is_in([lit(Float64(-0.0))]).alias("zero"),
            col("x").is_in(List[Expr]()).alias("empty"),
            col("x").is_in([null(DataType.FLOAT64)]).alias("null"),
            col("x").is_in([col("y")]).alias("dynamic"),
        ],
        batch_size=3,
    )
    var valid: List[Bool] = [True, True, True, True, False, True, True]
    var expected = DataFrame(
        [
            Series(
                "nan",
                BoolColumn(
                    [False, False, False, False, False, False, False], valid
                ),
            ),
            Series(
                "zero",
                BoolColumn(
                    [False, True, True, False, False, False, False], valid
                ),
            ),
            Series(
                "empty",
                BoolColumn(
                    [False, False, False, False, False, False, False], valid
                ),
            ),
            Series(
                "null",
                BoolColumn(
                    [False, False, False, False, False, False, False], valid
                ),
            ),
            Series(
                "dynamic",
                BoolColumn(
                    [False, True, True, False, False, True, False], valid
                ),
            ),
        ]
    )
    assert_frame(actual, expected)
    _cold(x)
    _cold(y)
