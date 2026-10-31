"""Accepted identity casts retain array windows and their original owners."""
# contract-group: types.cast
# contract-group: expr.chunk_windows
from std.memory import ArcPointer, bitcast
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    BoolColumn,
    StringColumn,
    ListColumn,
    StructColumn,
    DataFrame,
    DataType,
    Series,
    col,
)
from dataframe.dtype import NUMERIC_DTYPES, CategoricalDictionary
from dataframe.geometry import from_wkb
from dataframe.kernels.cast import cast_series
from dataframe.string_view import StringViewBuilder
from tests.support.coverage_compare import assert_frame
from tests.support.spatial_reference import point


def _metadata(var source: Series) -> Series:
    # One ordinary Arrow application metadata entry: k -> v.
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    source._field_metadata = ArcPointer(metadata^)
    return source^


def _pieces(source: Series) raises -> Series:
    return Series._from_chunks(
        [
            source.slice(0, 2),
            source.slice(2, 0),
            source.slice(2, 1),
            source.slice(3, 3),
        ]
    )


def _same(actual: Series, expected: Series) raises:
    assert_equal(actual.name(), expected.name())
    assert_frame(DataFrame([actual.copy()]), DataFrame([expected.copy()]))


def _cold(source: Series) raises:
    if source.is_chunked():
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))


def _field_owner(actual: Series, source: Series) raises:
    assert_true(Bool(actual._field_metadata))
    assert_equal(
        actual._field_metadata.value()[], source._field_metadata.value()[]
    )
    assert_true(
        actual._field_metadata.value().ptr()
        == source._field_metadata.value().ptr()
    )


def _array_owners(actual: Series, source: Series) raises:
    var ac = actual.chunks()
    var sc = source.chunks()
    assert_equal(len(ac), len(sc))
    for k in range(len(sc)):
        assert_equal(len(ac[k]), len(sc[k]))
        comptime widths = (
            DType.int8,
            DType.int16,
            DType.int32,
            DType.int64,
            DType.int128,
            DType.uint8,
            DType.uint16,
            DType.uint32,
            DType.uint64,
            DType.float32,
            DType.float64,
        )
        comptime for j in range(len(widths)):
            comptime D = widths[j]
            if sc[k]._data.isa[Column[Scalar[D]]]():
                ref a = ac[k]._data[Column[Scalar[D]]]
                ref s = sc[k]._data[Column[Scalar[D]]]
                assert_true(a._data._state.ptr() == s._data._state.ptr())
                assert_true(a._bits._state.ptr() == s._bits._state.ptr())
                assert_equal(a._offset, s._offset)
        if sc[k]._data.isa[BoolColumn]():
            ref a = ac[k]._data[BoolColumn]
            ref s = sc[k]._data[BoolColumn]
            assert_true(a._data._state.ptr() == s._data._state.ptr())
            assert_true(a._bits._state.ptr() == s._bits._state.ptr())
            assert_equal(a._offset, s._offset)
        if sc[k]._data.isa[StringColumn]():
            ref a = ac[k]._data[StringColumn]
            ref s = sc[k]._data[StringColumn]
            assert_true(a._bytes._state.ptr() == s._bytes._state.ptr())
            assert_true(a._offsets._state.ptr() == s._offsets._state.ptr())
            assert_true(a._bits._state.ptr() == s._bits._state.ptr())
            assert_equal(a._offset, s._offset)
            assert_equal(a._is_view_storage(), s._is_view_storage())
            if s._is_view_storage():
                var av = a._view_storage_unchecked()
                var sv = s._view_storage_unchecked()
                assert_true(av._views._state.ptr() == sv._views._state.ptr())
                assert_true(av._buffers.ptr() == sv._buffers.ptr())


def _identity(master: Series, expected: Series) raises:
    var flat = _metadata(master.slice(1, 6))
    var source = _pieces(flat)
    var masks = List[List[Bool]]()
    masks.append([])
    masks.append(List[Bool](length=6, fill=False))
    masks.append([False, True, False, True, False, True])
    masks.append([False, True])
    for strict in [False, True]:
        for mask in masks:
            var result = cast_series(source, source.dtype(), strict, 71, mask)
            _same(result, expected)
            _field_owner(result, source)
            _array_owners(result, source)
            _same(result, cast_series(flat, flat.dtype(), strict, 71, mask))
            _cold(source)
    var scalar = source.slice(2, 1)
    var scalar_result = cast_series(scalar, scalar.dtype(), False, 900, [False])
    _same(scalar_result, expected.slice(2, 1))
    _array_owners(scalar_result, scalar)
    var empty = flat.slice(0, 0)
    var empty_result = cast_series(empty, empty.dtype(), True, 3, [False])
    _same(empty_result, expected.slice(0, 0))
    _field_owner(empty_result, empty)
    var frame = DataFrame([source.copy()])
    var public_result = frame.select_exprs(
        [col(source.name()).cast(source.dtype()).alias(source.name())],
        batch_size=2,
    )
    assert_frame(public_result, DataFrame([expected.copy()]))
    _field_owner(public_result.column(source.name()), source)
    _cold(source)


def _numeric_identity[D: DType]() raises:
    var master = Series(
        "value",
        Column[Scalar[D]](
            [88, 2, 99, 0, 7, 9, 5, 77],
            [True, True, False, True, True, False, True, True],
        ),
    )
    var expected = Series(
        "value",
        Column[Scalar[D]](
            [2, 99, 0, 7, 9, 5], [True, False, True, True, False, True]
        ),
    )
    _identity(master, expected)


def test_chunked_identity_casts_all_numeric_widths_and_null_storage() raises:
    comptime for k in range(len(NUMERIC_DTYPES)):
        _numeric_identity[NUMERIC_DTYPES[k]]()
    var master = Series(
        "value",
        Column[Int8](
            List[Int8](length=8, fill=93), List[Bool](length=8, fill=False)
        ),
    ).with_dtype(DataType.NULL)
    var expected = Series(
        "value",
        Column[Int8](
            List[Int8](length=6, fill=93), List[Bool](length=6, fill=False)
        ),
    ).with_dtype(DataType.NULL)
    _identity(master, expected)


def test_chunked_identity_casts_boolean_text_and_arbitrary_binary() raises:
    _identity(
        Series(
            "value",
            BoolColumn(
                [False, True, True, False, True, True, False, True],
                [True, True, False, True, True, False, True, True],
            ),
        ),
        Series(
            "value",
            BoolColumn(
                [True, True, False, True, True, False],
                [True, False, True, True, False, True],
            ),
        ),
    )
    _identity(
        Series(
            "value",
            StringColumn(
                [
                    "pad",
                    "é",
                    "hidden",
                    "",
                    "long payload beyond inline",
                    "hidden",
                    "a\x00b",
                    "tail",
                ],
                [True, True, False, True, True, False, True, True],
            ),
        ),
        Series(
            "value",
            StringColumn(
                [
                    "é",
                    "hidden",
                    "",
                    "long payload beyond inline",
                    "hidden",
                    "a\x00b",
                ],
                [True, False, True, True, False, True],
            ),
        ),
    )
    _identity(
        Series.binary(
            "value",
            [
                [UInt8(0)],
                [UInt8(255)],
                [UInt8(254)],
                [],
                [UInt8(0), 255],
                [UInt8(253)],
                [UInt8(128)],
                [UInt8(1)],
            ],
            [True, True, False, True, True, False, True, True],
        ),
        Series.binary(
            "value",
            [
                [UInt8(255)],
                [UInt8(254)],
                [],
                [UInt8(0), 255],
                [UInt8(253)],
                [UInt8(128)],
            ],
            [True, False, True, True, False, True],
        ),
    )


def test_chunked_identity_casts_preserve_temporal_and_decimal_logical_dtypes() raises:
    var master = Series(
        "value",
        Column[Int64](
            [0, -1, Int64.MIN, 17, 29, 99, 31, 0],
            [True, True, False, True, True, False, True, True],
        ),
    )
    var expected = Series(
        "value",
        Column[Int64](
            [-1, Int64.MIN, 17, 29, 99, 31],
            [True, False, True, True, False, True],
        ),
    )
    var tags: List[DataType] = [
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("ns", "UTC"),
        DataType.duration("us"),
    ]
    for tag in tags:
        _identity(master.with_dtype(tag), expected.with_dtype(tag))
    _identity(
        Series(
            "value",
            Column[Int32](
                [0, -123, 777, 0, 456, 888, 321, 0],
                [True, True, False, True, True, False, True, True],
            ),
        ).with_dtype(DataType.decimal(9, 2, 32)),
        Series(
            "value",
            Column[Int32](
                [-123, 777, 0, 456, 888, 321],
                [True, False, True, True, False, True],
            ),
        ).with_dtype(DataType.decimal(9, 2, 32)),
    )
    _identity(
        master.with_dtype(DataType.decimal(18, 3, 64)),
        expected.with_dtype(DataType.decimal(18, 3, 64)),
    )
    var large = (Int128(1) << 80) + 7
    _identity(
        Series(
            "value",
            Column[Int128](
                [0, -large, 123, 0, large, 456, 31, 0],
                [True, True, False, True, True, False, True, True],
            ),
        ).with_dtype(DataType.decimal(38, 2, 128)),
        Series(
            "value",
            Column[Int128](
                [-large, 123, 0, large, 456, 31],
                [True, False, True, True, False, True],
            ),
        ).with_dtype(DataType.decimal(38, 2, 128)),
    )


def _dictionary(labels: List[String]) -> DataType:
    var values = CategoricalDictionary()
    for label in labels:
        values.append(label)
    return DataType.categorical(values^)


def _category() raises -> Series:
    return _metadata(
        Series(
            "value",
            Column[UInt32](
                [0, 1, 99, 0, 1, 98, 0, 1],
                [True, True, False, True, True, False, True, True],
            ),
        ).with_dtype(_dictionary(["red", "blue", "unused"]))
    )


def test_chunked_categorical_identity_keeps_source_dictionary_and_metadata_owners() raises:
    var master = _category()
    var flat = master.slice(1, 6)
    var source = _pieces(flat)
    var targets: List[DataType] = [
        source.dtype(),
        DataType.CATEGORICAL,
        _dictionary(["red", "blue", "unused"]),
    ]
    var expected = Series(
        "value",
        Column[UInt32](
            [1, 99, 0, 1, 98, 0], [True, False, True, True, False, True]
        ),
    ).with_dtype(source.dtype())
    for target in targets:
        for strict in [False, True]:
            var result = cast_series(
                source, target, strict, 91, List[Bool](length=6, fill=False)
            )
            _same(result, expected)
            _field_owner(result, source)
            assert_true(
                result.dtype().dictionary().ptr()
                == source.dtype().dictionary().ptr()
            )
            _array_owners(result, source)
            _same(
                result,
                cast_series(
                    flat, target, strict, 91, List[Bool](length=6, fill=False)
                ),
            )
            _cold(source)
        var scalar = source.slice(0, 1)
        var scalar_result = cast_series(scalar, target, False, 333, [False])
        _same(scalar_result, expected.slice(0, 1))
        _field_owner(scalar_result, scalar)
        _array_owners(scalar_result, scalar)
        var empty = flat.slice(0, 0)
        var empty_result = cast_series(empty, target, True, 333, [])
        _same(empty_result, expected.slice(0, 0))
        _field_owner(empty_result, empty)
        assert_true(
            empty_result.dtype().dictionary().ptr()
            == source.dtype().dictionary().ptr()
        )
    var reordered = _dictionary(["unused", "blue", "red"])
    var recoded = cast_series(source, reordered, True, 0, [])
    var wanted = Series(
        "value",
        Column[UInt32](
            [1, 0, 2, 1, 0, 2], [True, False, True, True, False, True]
        ),
    ).with_dtype(reordered)
    _same(recoded, wanted)
    assert_true(
        recoded.dtype().dictionary().ptr() == reordered.dictionary().ptr()
    )
    var missing_unused = _dictionary(["red", "blue"])
    for strict in [False, True]:
        with assert_raises():
            _ = cast_series(
                source,
                missing_unused,
                strict,
                0,
                List[Bool](length=6, fill=False),
            )
        with assert_raises():
            _ = cast_series(flat.slice(0, 0), missing_unused, strict, 0, [])


def _text_master(views: Bool) raises -> Series:
    var words: List[String] = [
        "pad",
        "é",
        "ignored",
        "",
        "long external string payload",
        "ignored",
        "a\x00b",
        "tail",
    ]
    var valid: List[Bool] = [True, True, False, True, True, False, True, True]
    if views:
        var builder = StringViewBuilder()
        for row in range(len(words)):
            if valid[row]:
                builder.append(StringSlice(words[row]))
            else:
                builder.append_null()
        return _metadata(Series("value", StringColumn(builder^.finish())))
    return _metadata(Series("value", StringColumn(words, valid)))


def _binary_expected() raises -> Series:
    # Literal UTF8 bytes, including an embedded NUL and a long view payload.
    return Series.binary(
        "value",
        [
            [UInt8(195), 169],
            [],
            [],
            [
                UInt8(108),
                111,
                110,
                103,
                32,
                101,
                120,
                116,
                101,
                114,
                110,
                97,
                108,
                32,
                115,
                116,
                114,
                105,
                110,
                103,
                32,
                112,
                97,
                121,
                108,
                111,
                97,
                100,
            ],
            [],
            [UInt8(97), 0, 98],
        ],
        [True, False, True, True, False, True],
    )


def test_chunked_string_to_binary_shares_offset_and_view_owners_but_drops_field_metadata() raises:
    for views in [False, True]:
        var master = _text_master(views)
        _identity(
            master,
            Series(
                "value",
                StringColumn(
                    [
                        "é",
                        "ignored",
                        "",
                        "long external string payload",
                        "ignored",
                        "a\x00b",
                    ],
                    [True, False, True, True, False, True],
                ),
            ),
        )
        var flat = master.slice(1, 6)
        var source = _pieces(flat)
        var masks = List[List[Bool]]()
        masks.append([])
        masks.append(List[Bool](length=6, fill=False))
        masks.append([False, True])
        for strict in [False, True]:
            for mask in masks:
                var result = cast_series(
                    source, DataType.BINARY, strict, 317, mask
                )
                _same(result, _binary_expected())
                assert_false(Bool(result._field_metadata))
                _array_owners(result, source)
                _same(
                    result,
                    cast_series(flat, DataType.BINARY, strict, 317, mask),
                )
                _cold(source)
        var scalar = source.slice(5, 1)
        var scalar_result = cast_series(
            scalar, DataType.BINARY, True, 400, [False]
        )
        _same(scalar_result, _binary_expected().slice(5, 1))
        _array_owners(scalar_result, scalar)
        var empty = cast_series(
            source.slice(0, 0), DataType.BINARY, True, 400, []
        )
        _same(empty, _binary_expected().slice(0, 0))
        assert_false(Bool(empty._field_metadata))
        var public_result = DataFrame([source.copy()]).select_exprs(
            [col("value").cast(DataType.BINARY)], batch_size=2
        )
        assert_frame(public_result, DataFrame([_binary_expected()]))
        assert_false(Bool(public_result.column("value")._field_metadata))
        _cold(source)


def test_chunked_geometry_identity_requires_complete_normalized_metadata() raises:
    var values = List[List[UInt8]]()
    for row in range(8):
        values.append(point([1, 2]))
    var master = _metadata(
        from_wkb(
            Series.binary(
                "value",
                values,
                [True, True, False, True, True, False, True, True],
            ),
            '{"crs":"EPSG:4326","edges":"planar"}',
        )
    )
    var expected_values = List[List[UInt8]]()
    for row in range(6):
        expected_values.append(point([1, 2]))
    var expected = from_wkb(
        Series.binary(
            "value", expected_values, [True, False, True, True, False, True]
        ),
        '{"crs":"EPSG:4326"}',
    )
    _identity(master, expected)
    var source = _pieces(master.slice(1, 6))
    var canonical = DataType.geometry(
        '{ "edges": "planar", "crs": "EPSG:4326" }'
    )
    var same = cast_series(
        source, canonical, False, 99, List[Bool](length=6, fill=False)
    )
    _same(same, expected)
    _array_owners(same, source)
    _field_owner(same, source)
    _cold(source)
    var rejected: List[DataType] = [
        DataType.geometry('{"crs":"EPSG:3857"}'),
        DataType.geometry('{"crs":"EPSG:4326","edges":"spherical"}'),
        DataType.BINARY,
        DataType.STRING,
    ]
    for target in rejected:
        with assert_raises(contains="Geometry casts require"):
            _ = cast_series(
                source, target, False, 88, List[Bool](length=6, fill=False)
            )
        with assert_raises(contains="Geometry casts require"):
            _ = cast_series(source.slice(0, 0), target, True, 88, [])


def test_nested_identity_casts_keep_direct_and_public_empty_plan_boundaries() raises:
    var child = Series("item", Column[Int64]([7, 8]))
    var nested: List[Series] = [
        Series("value", ListColumn([Int64(0), 1, 2], child.copy())),
        Series("value", StructColumn([child.copy()])),
    ]
    for source in nested:
        for size in [0, 2]:
            var selected = source.slice(0, size)
            with assert_raises(
                contains="list and struct casts are not supported"
            ):
                _ = cast_series(
                    selected,
                    selected.dtype(),
                    False,
                    88,
                    List[Bool](length=size, fill=False),
                )
        with assert_raises(contains="list and struct casts are not supported"):
            _ = DataFrame([source.copy()]).select_exprs(
                [col("value").cast(source.dtype())]
            )
        # Existing public zero-row output seeding bypasses kernel evaluation.
        var empty_frame = DataFrame([source.slice(0, 0)])
        var public_empty = empty_frame.select_exprs(
            [col("value").cast(source.dtype())]
        )
        assert_frame(public_empty, empty_frame)
    with assert_raises(contains="list and struct casts are not supported"):
        _ = cast_series(
            child.slice(0, 0), DataType.list(DataType.INT64), True, 0, []
        )


def _owned_binary(views: Bool) raises -> Series:
    var source = _pieces(_text_master(views).slice(1, 6))
    var result = cast_series(
        source, DataType.BINARY, False, 99, List[Bool](length=6, fill=False)
    )
    _array_owners(result, source)
    _cold(source)
    return result^


def _owned_integer() raises -> Series:
    var source = _pieces(
        _metadata(
            Series(
                "value",
                Column[Int64](
                    [88, 2, 99, 0, 7, 9, 5, 77],
                    [True, True, False, True, True, False, True, True],
                ),
            )
        ).slice(1, 6)
    )
    var result = cast_series(source, DataType.INT64, True, 7, [])
    _field_owner(result, source)
    _array_owners(result, source)
    _cold(source)
    return result^


def test_cast_metadata_results_outlive_sources_and_explicit_contiguous_access_still_merges() raises:
    _same(_owned_binary(False), _binary_expected())
    _same(_owned_binary(True), _binary_expected())
    var result = _owned_integer()
    _same(
        result,
        Series(
            "value",
            Column[Int64](
                [2, 99, 0, 7, 9, 5], [True, False, True, True, False, True]
            ),
        ),
    )
    # The public typed accessor retains its explicit contiguous contract.
    assert_true(result.is_chunked())
    var contiguous = result.numeric[DType.int64]()
    assert_equal(contiguous._get(0), Int64(2))
    assert_equal(contiguous._get(1), Int64(99))
    assert_equal(result._chunked.value()[].cache[].state.load(), Int64(2))
    var merged = result.rechunk()
    assert_false(merged.is_chunked())
    _same(merged, result)
    _field_owner(merged, result)
    var cached_identity = cast_series(
        result, DataType.INT64, False, 18, [False]
    )
    _same(cached_identity, result)
    _array_owners(cached_identity, result)
    _field_owner(cached_identity, result)
    assert_equal(result._chunked.value()[].cache[].state.load(), Int64(2))
