"""Exact key membership and stable row order across encoding specializations."""
from std.testing import assert_equal, assert_true, assert_raises
from std.memory import bitcast
from dataframe import BoolColumn, Column, DataType, Series
from dataframe.dtype import CategoricalDictionary
from dataframe.string_column import StringColumn
from dataframe.hashing import (
    RowKeys,
    encode_rows,
    column_codes,
    _encode_dense,
    _encode_dense_chunks,
    _encode_string_rows,
    _encode_categorical,
    _U64Codes,
)
from dataframe.row_sort import row_arg_sort
from dataframe.packed_sort import packed_arg_sort, packed_top_rows
from dataframe.categorical import decode, sort_ranks, unify
from tests.support.key_kernel_reference import (
    KeyCell,
    cell_equal,
    cell_less,
    expected_key_ids,
    expected_order,
)
from tests.support.row_kernel_reference import integer_bounds, row_valid

comptime KEY_INTEGER_DTYPES = (
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
)


def assert_key_encoding(
    actual: RowKeys, rows: List[List[KeyCell]], nulls_equal: Bool
) raises:
    var expected = expected_key_ids(rows, nulls_equal)
    assert_equal(actual.ids, expected[0])
    assert_equal(actual.representatives, expected[1])
    assert_equal(actual.count(), len(expected[1]))


def key_layout(series: Series, layout: Int) raises -> Series:
    if layout != 2 or len(series) == 0:
        return series.copy()
    var cut = min(3, len(series))
    return Series._from_chunks(
        [series.slice(0, cut), series.slice(cut, len(series) - cut)]
    )


def test_key_reference_has_literal_null_nan_zero_and_stable_order_anchors() raises:
    var nan = Float64(0) / Float64(0)
    assert_true(cell_equal(KeyCell(1, floating=-0.0), KeyCell(1, floating=0.0)))
    assert_true(cell_equal(KeyCell(1, floating=nan), KeyCell(1, floating=nan)))
    assert_true(
        cell_less(KeyCell(1, floating=1), KeyCell(1, floating=nan), True, True)
    )
    var rows: List[List[KeyCell]] = [
        [KeyCell(0, integer=3)],
        [KeyCell(0, False, integer=90)],
        [KeyCell(0, integer=1)],
        [KeyCell(0, integer=3)],
        [KeyCell(0, False, integer=91)],
    ]
    var keys = expected_key_ids(rows, True)
    assert_equal(keys[0], [0, 1, 2, 0, 1])
    assert_equal(keys[1], [0, 1, 2])
    var excluded = expected_key_ids(rows, False)
    assert_equal(excluded[0], [0, -1, 1, 0, -1])
    assert_equal(expected_order(rows, [False], [True]), [2, 0, 3, 1, 4])
    assert_equal(expected_order(rows, [True], [False]), [1, 4, 0, 3, 2])


def integer_key_encoding[D: DType]() raises:
    var limits = integer_bounds[D]()
    for n in [0, 1, 7, 8, 9, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                for sparse in [False, True]:
                    var data = List[Scalar[D]]()
                    var valid = List[Bool]()
                    var rows = List[List[KeyCell]]()
                    for physical in range(n + offset):
                        var present = row_valid(physical, mode)
                        var value = (
                            limits[0]
                            + Int128(physical % 3) if not sparse else limits[
                                0
                            ] if physical
                            % 3
                            == 0 else limits[1] if physical % 3
                            == 1 else Int128(0)
                        )
                        data.append(
                            value.cast[D]() if present else Scalar[D].MAX
                        )
                        valid.append(present)
                        if physical >= offset:
                            rows.append([KeyCell(0, present, integer=value)])
                    var source = Series(
                        "k", Column[Scalar[D]](data^, valid^)
                    ).slice(offset, n)
                    for layout in [0, 2]:
                        var actual_source = key_layout(source, layout)
                        for nulls_equal in [False, True]:
                            assert_key_encoding(
                                encode_rows(
                                    [actual_source.copy()], nulls_equal
                                ),
                                rows,
                                nulls_equal,
                            )
                            var dense = _encode_dense_chunks(
                                actual_source, nulls_equal
                            ) if actual_source.is_chunked() else _encode_dense(
                                actual_source, nulls_equal
                            )
                            var span = limits[1] - limits[0]
                            var expected_hit = (
                                not sparse or mode == 2 or n < 2 or span < 4096
                            )
                            # For narrow integer domains the encoder allows
                            # the complete 8-bit range even in tiny samples.
                            if expected_hit:
                                assert_true(
                                    Bool(dense),
                                    "bounded integer domain must use dense encoding",
                                )
                                assert_key_encoding(
                                    dense.take(), rows, nulls_equal
                                )
                            elif n >= 9 and mode != 2:
                                assert_true(
                                    not dense,
                                    "wide sparse integer domain must fall back",
                                )
                            var codes = List[Int](length=n, fill=0)
                            var nulls = List[Bool](length=n, fill=False)
                            _ = column_codes(
                                actual_source, codes, nulls, dense=False
                            )
                            var expected = expected_key_ids(rows, False)
                            assert_equal(codes, expected[0])
                            for row in range(n):
                                assert_equal(nulls[row], not rows[row][0].valid)


def test_dense_and_sparse_integer_key_routes_match_scalar_membership_at_all_widths() raises:
    comptime for k in range(len(KEY_INTEGER_DTYPES)):
        integer_key_encoding[KEY_INTEGER_DTYPES[k]]()


def integer_sort_keys[D: DType]() raises:
    var limits = integer_bounds[D]()
    var pattern: List[Int128] = [
        limits[1],
        0,
        limits[0],
        1,
        limits[1],
        0,
        limits[0],
    ]
    for n in [0, 1, 7, 8, 9, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var values = List[Scalar[D]]()
                var valid = List[Bool]()
                var rows = List[List[KeyCell]]()
                for physical in range(n + offset):
                    var present = row_valid(physical, mode)
                    values.append(
                        pattern[physical % 7]
                        .cast[D]() if present else Scalar[D]
                        .MAX
                    )
                    valid.append(present)
                    if physical >= offset:
                        rows.append(
                            [KeyCell(0, present, integer=pattern[physical % 7])]
                        )
                var source = Series(
                    "k", Column[Scalar[D]](values^, valid^)
                ).slice(offset, n)
                for layout in [0, 2]:
                    var keys: List[Series] = [key_layout(source, layout)]
                    for descending in [False, True]:
                        for nulls_last in [False, True]:
                            var expected = expected_order(
                                rows, [descending], [nulls_last]
                            )
                            assert_equal(
                                row_arg_sort(keys, [descending], [nulls_last]),
                                expected,
                            )
                            var packed = packed_arg_sort(
                                keys, [descending], [nulls_last]
                            )
                            assert_true(
                                Bool(packed),
                                "one integer key must fit packed sort",
                            )
                            assert_equal(packed.take(), expected)


def test_integer_packed_and_general_sort_preserve_exact_width_and_stable_ties() raises:
    comptime for k in range(len(KEY_INTEGER_DTYPES)):
        integer_sort_keys[KEY_INTEGER_DTYPES[k]]()


def float_key_routes[D: DType]() raises:
    var inf = Float64(1) / Float64(0)
    var nan1: Scalar[D]
    var nan2: Scalar[D]
    comptime if D == DType.float32:
        nan1 = Scalar[D](bitcast[DType.float32](UInt32(0x7FC00001)))
        nan2 = Scalar[D](bitcast[DType.float32](UInt32(0xFFC00317)))
    else:
        nan1 = Scalar[D](bitcast[DType.float64](UInt64(0x7FF8000000000001)))
        nan2 = Scalar[D](bitcast[DType.float64](UInt64(0xFFF8000000031700)))
    var pattern: List[Scalar[D]] = [
        Scalar[D](-inf),
        Scalar[D](-3),
        Scalar[D](-0.0),
        Scalar[D](0.0),
        Scalar[D](3),
        Scalar[D](inf),
        nan1,
        nan2,
    ]
    for n in [0, 1, 7, 8, 9, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var values = List[Scalar[D]]()
                var valid = List[Bool]()
                var rows = List[List[KeyCell]]()
                for physical in range(n + offset):
                    var present = row_valid(physical, mode)
                    values.append(pattern[physical % 8])
                    valid.append(present)
                    if physical >= offset:
                        rows.append(
                            [
                                KeyCell(
                                    1,
                                    present,
                                    floating=Float64(pattern[physical % 8]),
                                )
                            ]
                        )
                var source = Series(
                    "f", Column[Scalar[D]](values^, valid^)
                ).slice(offset, n)
                for layout in [0, 2]:
                    var keys: List[Series] = [key_layout(source, layout)]
                    for nulls_equal in [False, True]:
                        assert_key_encoding(
                            encode_rows(keys, nulls_equal), rows, nulls_equal
                        )
                    assert_true(not _encode_dense(keys[0].rechunk(), True))
                    for descending in [False, True]:
                        for nulls_last in [False, True]:
                            var expected = expected_order(
                                rows, [descending], [nulls_last]
                            )
                            assert_equal(
                                row_arg_sort(keys, [descending], [nulls_last]),
                                expected,
                            )
                            var packed = packed_arg_sort(
                                keys, [descending], [nulls_last]
                            )
                            assert_true(
                                Bool(packed),
                                "one float key must fit packed sort",
                            )
                            assert_equal(packed.take(), expected)


def test_float_key_ids_and_order_canonicalize_nan_payloads_and_signed_zero() raises:
    float_key_routes[DType.float32]()
    float_key_routes[DType.float64]()


def test_hash_collision_clusters_and_growth_never_merge_distinct_words() raises:
    var table = _U64Codes()
    # This multiplicative inverse makes every hash equal a small ordinal,
    # forcing the same initial bucket through multiple table growths.
    var inverse = UInt64(0xF1DE83E19937733D)
    var multiplier = UInt64(0x9E3779B97F4A7C15)
    assert_equal(inverse * multiplier, UInt64(1))
    for row in range(1200):
        var key = UInt64(row) * inverse
        assert_equal((key * multiplier) >> table.shift, UInt64(0))
        assert_equal(table.code(key), row)
    assert_equal(table.count, 1200)
    assert_true(len(table.slots) > 1024)
    for row in range(1200):
        assert_equal(Int(table.slots[row]), row)
    for row in range(1200):
        var original = 1199 - row
        assert_equal(table.code(UInt64(original) * inverse), original)
    assert_equal(table.count, 1200)


def test_composite_key_columns_preserve_boundaries_and_reject_packing_overflow() raises:
    var left: List[String] = ["ab", "a", "", "a\x1fb", "a", "ab", "", "hidden"]
    var right: List[String] = [
        "c",
        "bc",
        "abc",
        "c",
        "b\x1fc",
        "c",
        "abc",
        "hidden",
    ]
    var valid: List[Bool] = [True, True, True, True, True, True, True, False]
    var rows = List[List[KeyCell]]()
    for row in range(8):
        rows.append(
            [
                KeyCell(2, valid[row], text=left[row].copy()),
                KeyCell(2, row != 6, text=right[row].copy()),
            ]
        )
    var a = Series("a", Column[String](left.copy(), valid.copy()))
    var b = Series(
        "b",
        Column[String](
            right.copy(), [True, True, True, True, True, True, False, True]
        ),
    )
    for layout in [0, 2]:
        var keys: List[Series] = [key_layout(a, layout), key_layout(b, layout)]
        for nulls_equal in [False, True]:
            assert_key_encoding(
                encode_rows(keys, nulls_equal), rows, nulls_equal
            )
        for desc_a in [False, True]:
            for desc_b in [False, True]:
                var expected = expected_order(
                    rows, [desc_a, desc_b], [False, True]
                )
                assert_equal(
                    row_arg_sort(keys, [desc_a, desc_b], [False, True]),
                    expected,
                )
                var packed = packed_arg_sort(
                    keys, [desc_a, desc_b], [False, True]
                )
                assert_true(Bool(packed))
                assert_equal(packed.take(), expected)
    var u = UInt64.MAX
    var wide = List[List[KeyCell]]()
    var first: List[UInt64] = [0, u, 0, u, 0]
    var second: List[UInt64] = [u, 0, 0, u, u]
    for row in range(5):
        wide.append(
            [
                KeyCell(0, integer=first[row].cast[DType.int128]()),
                KeyCell(0, integer=second[row].cast[DType.int128]()),
            ]
        )
    var wide_keys: List[Series] = [
        Series("a", Column[UInt64](first^)),
        Series("b", Column[UInt64](second^)),
    ]
    assert_true(
        not packed_arg_sort(wide_keys, [False, False], [True, True]),
        "two full UInt64 ranges exceed packed key capacity",
    )
    assert_equal(
        row_arg_sort(wide_keys, [False, False], [True, True]),
        expected_order(wide, [False, False], [True, True]),
    )


def test_inline_external_and_common_prefix_strings_have_exact_key_and_sort_semantics() raises:
    var pattern: List[String] = [
        "",
        "a",
        "a\x00",
        "a\x00b",
        "α",
        "abcdefghijkl",
        "abcdefghijklm",
        "prefix0123456789abcdefghijkl-A",
        "prefix0123456789abcdefghijkl-B",
        "prefix0123456789abcdefghijkl-A\x00",
        "z",
        "a",
        "abcdefghijkl",
    ]
    for n in [0, 1, 7, 8, 9, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var values = List[String]()
                var valid = List[Bool]()
                var rows = List[List[KeyCell]]()
                for physical in range(n + offset):
                    var present = row_valid(physical, mode)
                    values.append(
                        pattern[
                            physical % len(pattern)
                        ].copy() if present else "poison text"
                    )
                    valid.append(present)
                    if physical >= offset:
                        rows.append(
                            [
                                KeyCell(
                                    2,
                                    present,
                                    text=pattern[
                                        physical % len(pattern)
                                    ].copy(),
                                )
                            ]
                        )
                var source = Series("s", Column[String](values^, valid^)).slice(
                    offset, n
                )
                for layout in [0, 2]:
                    var actual_source = key_layout(source, layout)
                    for nulls_equal in [False, True]:
                        assert_key_encoding(
                            _encode_string_rows(actual_source, nulls_equal),
                            rows,
                            nulls_equal,
                        )
                        assert_key_encoding(
                            encode_rows([actual_source.copy()], nulls_equal),
                            rows,
                            nulls_equal,
                        )
                    for descending in [False, True]:
                        for nulls_last in [False, True]:
                            var expected = expected_order(
                                rows, [descending], [nulls_last]
                            )
                            assert_equal(
                                row_arg_sort(
                                    [actual_source.copy()],
                                    [descending],
                                    [nulls_last],
                                ),
                                expected,
                            )
                            var packed = packed_arg_sort(
                                [actual_source.copy()],
                                [descending],
                                [nulls_last],
                            )
                            assert_true(
                                Bool(packed),
                                "a final long string key is packable with full tie comparison",
                            )
                            assert_equal(packed.take(), expected)
    # A long string before another key cannot use the packed prefix; the
    # full string must be compared before the later key is considered.
    var texts: List[String] = [
        "x0123456789abcdefghijklmnop-A",
        "x0123456789abcdefghijklmnop-B",
        "a",
        "x0123456789abcdefghijklmnop-A",
        "x0123456789abcdefghijklmnop-B",
    ]
    var number: List[Int64] = [9, 1, 7, 2, 8]
    var rows = List[List[KeyCell]]()
    for row in range(5):
        rows.append(
            [
                KeyCell(2, text=texts[row].copy()),
                KeyCell(0, integer=Int128(number[row])),
            ]
        )
    var keys: List[Series] = [
        Series("s", Column[String](texts^)),
        Series("i", Column[Int64](number^)),
    ]
    assert_true(
        not packed_arg_sort(keys, [False, False], [True, True]),
        "a non-final long prefix must fall back to complete row comparison",
    )
    for first in [False, True]:
        for second in [False, True]:
            assert_equal(
                row_arg_sort(keys, [first, second], [True, True]),
                expected_order(rows, [first, second], [True, True]),
            )


def categorical_fixture(n: Int, offset: Int, mode: Int) raises -> Series:
    var dictionary = CategoricalDictionary()
    var labels: List[String] = [
        "zebra",
        "",
        "abcdefghijkl",
        "long-external-abcdefghijklmnop",
        "alpha",
    ]
    for label in labels:
        dictionary.append(label)
    var codes = List[UInt32]()
    var valid = List[Bool]()
    for row in range(n + offset):
        var present = row_valid(row, mode)
        codes.append(UInt32((row * 3 + 1) % 5) if present else UInt32.MAX)
        valid.append(present)
    return (
        Series("cat", Column[UInt32](codes^, valid^))
        .with_dtype(DataType.categorical(dictionary^))
        .slice(offset, n)
    )


def test_dictionary_key_specialization_and_decoded_string_views_match_values() raises:
    var labels: List[String] = [
        "zebra",
        "",
        "abcdefghijkl",
        "long-external-abcdefghijklmnop",
        "alpha",
    ]
    for n in [0, 1, 7, 8, 9, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var categorical = categorical_fixture(n, offset, mode)
                var rows = List[List[KeyCell]]()
                for row in range(n):
                    var original = row + offset
                    rows.append(
                        [
                            KeyCell(
                                2,
                                row_valid(original, mode),
                                text=labels[(original * 3 + 1) % 5].copy(),
                            )
                        ]
                    )
                for layout in [0, 2]:
                    var source = key_layout(categorical, layout)
                    assert_true(source.dtype().is_categorical())
                    assert_true(
                        len(source.dtype().dictionary()[]) <= max(n, 4096)
                    )
                    for nulls_equal in [False, True]:
                        assert_key_encoding(
                            encode_rows([source.copy()], nulls_equal),
                            rows,
                            nulls_equal,
                        )
                        var contiguous = source.rechunk()
                        assert_key_encoding(
                            _encode_categorical[checked=True](
                                contiguous._data[Column[UInt32]], 5, nulls_equal
                            ),
                            rows,
                            nulls_equal,
                        )
                    var decoded = decode(source)
                    assert_true(decoded._data[StringColumn]._is_view_storage())
                    if n == 65 and mode == 0:
                        var storage = decoded._data[
                            StringColumn
                        ]._view_storage_unchecked()
                        var inline = False
                        var external = False
                        for row in range(n):
                            var view = storage._view_unchecked(row)
                            inline = inline or view.is_inline()
                            external = external or not view.is_inline()
                        assert_true(
                            inline and external,
                            "dictionary values must exercise inline and external string view keys",
                        )
                    for nulls_equal in [False, True]:
                        assert_key_encoding(
                            _encode_string_rows(decoded, nulls_equal),
                            rows,
                            nulls_equal,
                        )
                    var ranks = sort_ranks(source)
                    for descending in [False, True]:
                        for nulls_last in [False, True]:
                            var expected = expected_order(
                                rows, [descending], [nulls_last]
                            )
                            assert_equal(
                                row_arg_sort(
                                    [ranks.copy()], [descending], [nulls_last]
                                ),
                                expected,
                            )
                            var packed = packed_arg_sort(
                                [ranks.copy()], [descending], [nulls_last]
                            )
                            assert_true(Bool(packed))
                            assert_equal(packed.take(), expected)
                            assert_equal(
                                row_arg_sort(
                                    [decoded.copy()], [descending], [nulls_last]
                                ),
                                expected,
                            )
    var malformed = Column[UInt32]([0, 5])
    with assert_raises():
        _ = _encode_categorical[checked=True](malformed, 5, True)
    var hidden = Column[UInt32]([0, UInt32.MAX], [True, False])
    var expected: List[List[KeyCell]] = [
        [KeyCell(0, integer=0)],
        [KeyCell(0, False, integer=0)],
    ]
    assert_key_encoding(
        _encode_categorical[checked=True](hidden, 5, True), expected, True
    )


def test_boolean_and_decimal_key_fallbacks_preserve_typed_membership_and_order() raises:
    for n in [0, 1, 7, 8, 9, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var flags = List[Bool]()
                var decimals = List[Int128]()
                var valid = List[Bool]()
                var bool_rows = List[List[KeyCell]]()
                var decimal_rows = List[List[KeyCell]]()
                var big = Int128(10_000_000_000_000_000) * Int128(
                    10_000_000_000_000_000
                )
                var pattern: List[Int128] = [-big, 0, big, 1, -big, big]
                for physical in range(n + offset):
                    var present = row_valid(physical, mode)
                    flags.append(physical % 3 == 0)
                    decimals.append(
                        pattern[physical % 6] if present else Int128.MAX
                    )
                    valid.append(present)
                    if physical >= offset:
                        bool_rows.append(
                            [
                                KeyCell(
                                    0,
                                    present,
                                    integer=Int128(physical % 3 == 0),
                                )
                            ]
                        )
                        decimal_rows.append(
                            [KeyCell(0, present, integer=pattern[physical % 6])]
                        )
                var boolean = Series(
                    "b", BoolColumn(flags^, valid.copy())
                ).slice(offset, n)
                var decimal = (
                    Series._wrap("d", Column[Int128](decimals^, valid^))
                    .with_dtype(DataType.decimal(38, 2))
                    .slice(offset, n)
                )
                for layout in [0, 2]:
                    var b = key_layout(boolean, layout)
                    var d = key_layout(decimal, layout)
                    for nulls_equal in [False, True]:
                        assert_key_encoding(
                            encode_rows([b.copy()], nulls_equal),
                            bool_rows,
                            nulls_equal,
                        )
                        assert_key_encoding(
                            encode_rows([d.copy()], nulls_equal),
                            decimal_rows,
                            nulls_equal,
                        )
                    assert_true(not _encode_dense(d.rechunk(), True))
                    for descending in [False, True]:
                        for nulls_last in [False, True]:
                            var expected_bool = expected_order(
                                bool_rows, [descending], [nulls_last]
                            )
                            var expected_decimal = expected_order(
                                decimal_rows, [descending], [nulls_last]
                            )
                            assert_equal(
                                row_arg_sort(
                                    [b.copy()], [descending], [nulls_last]
                                ),
                                expected_bool,
                            )
                            var packed = packed_arg_sort(
                                [b.copy()], [descending], [nulls_last]
                            )
                            assert_true(Bool(packed))
                            assert_equal(packed.take(), expected_bool)
                            assert_true(
                                not packed_arg_sort(
                                    [d.copy()], [descending], [nulls_last]
                                ),
                                "decimal128 keys must take the general row sort",
                            )
                            assert_equal(
                                row_arg_sort(
                                    [d.copy()], [descending], [nulls_last]
                                ),
                                expected_decimal,
                            )


def test_key_entry_points_reject_empty_key_sets_and_mismatched_columns() raises:
    with assert_raises():
        _ = encode_rows(List[Series](), True)
    with assert_raises():
        _ = encode_rows(
            [
                Series("a", Column[Int64]([1, 2])),
                Series("b", Column[Int64]([1])),
            ],
            True,
        )
    with assert_raises():
        _ = row_arg_sort(List[Series](), List[Bool](), List[Bool]())
    var keys: List[Series] = [Series("k", Column[Int64]([1, 2]))]
    with assert_raises():
        _ = row_arg_sort(keys, List[Bool](), [True])
    with assert_raises():
        _ = row_arg_sort(keys, [False], List[Bool]())


def test_dictionary_unification_preserves_values_and_remaps_only_codes() raises:
    var first_dictionary = CategoricalDictionary()
    var second_dictionary = CategoricalDictionary()
    var first_labels: List[String] = ["b", "a", "shared"]
    var second_labels: List[String] = ["shared", "c", "a"]
    for label in first_labels:
        first_dictionary.append(label)
    for label in second_labels:
        second_dictionary.append(label)
    var first = Series(
        "k", Column[UInt32]([0, 2, 1, UInt32.MAX], [True, True, True, False])
    ).with_dtype(DataType.categorical(first_dictionary^))
    var second = Series(
        "k", Column[UInt32]([2, 0, 1, UInt32.MAX], [True, True, True, False])
    ).with_dtype(DataType.categorical(second_dictionary^))
    unify(first, second)
    assert_true(first.dtype() == second.dtype())
    var dictionary = first.dtype().dictionary()
    assert_equal(len(dictionary[]), 4)
    var union_labels: List[String] = ["b", "a", "shared", "c"]
    for code in range(4):
        assert_equal(String(dictionary[].get(code)), union_labels[code])
    var first_values: List[String] = ["b", "shared", "a"]
    var second_values: List[String] = ["a", "shared", "c"]
    var first_codes: List[UInt32] = [0, 2, 1]
    var second_codes: List[UInt32] = [1, 2, 3]
    for row in range(3):
        assert_equal(first._data[Column[UInt32]].value(row), first_codes[row])
        assert_equal(second._data[Column[UInt32]].value(row), second_codes[row])
        assert_equal(first.get(row).string(), first_values[row])
        assert_equal(second.get(row).string(), second_values[row])
    assert_true(first.get(3).is_null())
    assert_true(second.get(3).is_null())
    var combined = Series._from_chunks([first.copy(), second.copy()])
    var rows = List[List[KeyCell]]()
    for value in first_values:
        rows.append([KeyCell(2, text=value.copy())])
    rows.append([KeyCell(2, False)])
    for value in second_values:
        rows.append([KeyCell(2, text=value.copy())])
    rows.append([KeyCell(2, False)])
    for nulls_equal in [False, True]:
        assert_key_encoding(
            encode_rows([combined.copy()], nulls_equal), rows, nulls_equal
        )
    var ranks = sort_ranks(combined)
    assert_equal(
        row_arg_sort([ranks.copy()], [False], [True]),
        expected_order(rows, [False], [True]),
    )


def test_packed_top_k_matches_stable_full_order_prefix_at_tie_boundaries() raises:
    var strings: List[String] = [
        "common-prefix-0123456789abcdefghijkl-A",
        "common-prefix-0123456789abcdefghijkl-B",
        "",
    ]
    for n in [0, 1, 9, 65]:
        for offset in [0, 1, 7]:
            var integers = List[Int64]()
            var text = List[String]()
            var iv = List[Bool]()
            var sv = List[Bool]()
            var rows = List[List[KeyCell]]()
            for physical in range(n + offset):
                integers.append(Int64(physical % 3))
                text.append(strings[(physical // 3) % 3].copy())
                iv.append(physical % 7 != 2)
                sv.append(physical % 5 != 1)
                if physical >= offset:
                    rows.append(
                        [
                            KeyCell(
                                0,
                                physical % 7 != 2,
                                integer=Int128(physical % 3),
                            ),
                            KeyCell(
                                2,
                                physical % 5 != 1,
                                text=strings[(physical // 3) % 3].copy(),
                            ),
                        ]
                    )
            var a = Series("a", Column[Int64](integers^, iv^)).slice(offset, n)
            var b = Series("b", Column[String](text^, sv^)).slice(offset, n)
            for layout in [0, 2]:
                var keys: List[Series] = [
                    key_layout(a, layout),
                    key_layout(b, layout),
                ]
                for descending in [False, True]:
                    for nulls_last in [False, True]:
                        var expected = expected_order(
                            rows,
                            [descending, not descending],
                            [nulls_last, not nulls_last],
                        )
                        for k in [0, 1, 3, 7, 8, 9, 64, 65, 66]:
                            var actual = packed_top_rows(
                                keys,
                                [descending, not descending],
                                [nulls_last, not nulls_last],
                                k,
                                threads=1,
                            )
                            assert_true(
                                Bool(actual),
                                "supported composite keys must use packed top-k",
                            )
                            var prefix = List[Int]()
                            for row in range(min(k, n)):
                                prefix.append(expected[row])
                            assert_equal(actual.take(), prefix)
    with assert_raises():
        _ = packed_top_rows(
            [Series("k", Column[Int64]([1]))], [False], [True], -1, threads=1
        )
