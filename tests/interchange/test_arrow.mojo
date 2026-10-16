"""Arrow contracts against independently authored C buffers and byte consumers."""
# contract-group: io.arrow
from std.memory import Pointer
from std.testing import assert_equal, assert_true
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    Series,
    export_arrow,
    export_arrow_series,
    import_arrow,
    import_arrow_series,
)
from tests.support.lifecycle_stream import at, array_release, schema_release
from tests.support.arrow_pyarrow_fixtures import pyarrow_producer
from tests.support.arrow_foreign import (
    ForeignArrow,
    integers,
    text_bytes,
    read_signed,
    read_unsigned,
    c_string,
    bitmap_valid,
    packed_metadata,
)


def buffer(array: ArrowArray, index: Int) -> Int:
    return at[Int](array.buffers)[unsafe_offset=index]


def fixed_foreign(
    format: String,
    width: Int,
    values: List[Int128],
    length: Int,
    offset: Int = 0,
    bits: List[UInt8] = [],
    nulls: Int = 0,
) -> ForeignArrow:
    var producer = ForeignArrow(1)
    producer.nodes[0].describe(format, "value", length, offset, nulls)
    producer.nodes[0].buffer(bits.copy())
    producer.nodes[0].buffer(integers(values, width, 1), 1)
    producer.seal()
    return producer^


def assert_bytes(address: Int, expected: List[UInt8]) raises:
    assert_true(address != 0)
    for index in range(len(expected)):
        assert_equal(at[UInt8](address)[unsafe_offset=index], expected[index])


def test_arrow_import_sliced_unaligned_integer_and_boolean_bitmaps() raises:
    var producer = fixed_foreign(
        "l",
        8,
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 11, 777, -33, 44],
        4,
        9,
        [255, 27],
        1,
    )
    assert_equal(buffer(producer.nodes[0].array, 1) % 8, 1)
    var result = import_arrow_series(
        producer.nodes[0].array, producer.nodes[0].schema
    )
    producer.assert_released()
    producer.poison()
    assert_equal(result.dtype(), DataType.INT64)
    assert_equal(len(result), 4)
    assert_equal(result.get(0).int64(), Int64(11))
    assert_true(result.get(1).is_null())
    assert_equal(result.get(2).int64(), Int64(-33))
    assert_equal(result.get(3).int64(), Int64(44))

    var booleans = ForeignArrow(1)
    booleans.nodes[0].describe("b", "flag", 4, 9, -1)
    booleans.nodes[0].buffer([255, 27])
    booleans.nodes[0].buffer([199, 18])
    booleans.seal()
    var flags = import_arrow_series(
        booleans.nodes[0].array, booleans.nodes[0].schema
    )
    booleans.assert_released()
    booleans.poison()
    assert_equal(flags.dtype(), DataType.BOOL)
    assert_equal(flags.get(0).bool(), True)
    assert_true(flags.get(1).is_null())
    assert_equal(flags.get(2).bool(), False)
    assert_equal(flags.get(3).bool(), True)


def test_arrow_import_small_large_text_and_binary_offsets_are_independent_owned_bytes() raises:
    for format in ["u", "U", "z", "Z"]:
        var producer = ForeignArrow(1)
        producer.nodes[0].describe(format, "bytes", 4, 2, 1)
        producer.nodes[0].buffer([47])
        producer.nodes[0].buffer(
            integers(
                [0, 3, 6, 6, 9, 15, 16],
                8 if format == "U" or format == "Z" else 4,
                1,
            ),
            1,
        )
        var data = text_bytes("oldprehépoisonz")
        if format == "z" or format == "Z":
            data[6] = 255
            data[7] = 0
            data[8] = 1
        producer.nodes[0].buffer(data^)
        producer.seal()
        var result = import_arrow_series(
            producer.nodes[0].array, producer.nodes[0].schema
        )
        producer.assert_released()
        producer.poison()
        assert_equal(len(result), 4)
        assert_true(result.get(2).is_null())
        if format == "z" or format == "Z":
            assert_equal(result.dtype(), DataType.BINARY)
            assert_equal(result.get(0).bytes(), List[UInt8]())
            assert_equal(
                result.get(1).bytes(),
                List[UInt8]([UInt8(255), UInt8(0), UInt8(1)]),
            )
            assert_equal(result.get(3).bytes(), List[UInt8]([UInt8(122)]))
        else:
            assert_equal(result.dtype(), DataType.STRING)
            assert_equal(result.get(0).string(), "")
            assert_equal(result.get(1).string(), "hé")
            assert_equal(result.get(3).string(), "z")


def dictionary_foreign(
    format: String,
    width: Int,
    duplicate: Bool = False,
    null_dictionary: Bool = False,
) -> ForeignArrow:
    var producer = ForeignArrow(2)
    producer.nodes[0].describe(format, "category", 4, 1, 1)
    producer.nodes[0].buffer([27])
    producer.nodes[0].buffer(integers([99, 1, 0, 2, 1], width, 1), 1)
    producer.nodes[0].dictionary = 1
    producer.nodes[1].describe("u", "", 3, 0, 1 if null_dictionary else 0)
    producer.nodes[1].buffer([3] if null_dictionary else List[UInt8]())
    if duplicate:
        producer.nodes[1].buffer(integers([0, 3, 6, 11], 4))
        producer.nodes[1].buffer(text_bytes("redredgreen"))
    else:
        producer.nodes[1].buffer(integers([0, 3, 7, 12], 4))
        producer.nodes[1].buffer(text_bytes("redbluegreen"))
    producer.seal()
    return producer^


def test_arrow_dictionary_import_accepts_each_integer_index_width_and_duplicate_values() raises:
    var formats: List[String] = ["c", "s", "i", "l", "C", "S", "I", "L"]
    var widths: List[Int] = [1, 2, 4, 8, 1, 2, 4, 8]
    for duplicate in [False, True]:
        for index in range(len(formats)):
            var producer = dictionary_foreign(
                formats[index], widths[index], duplicate
            )
            var result = import_arrow_series(
                producer.nodes[0].array, producer.nodes[0].schema
            )
            producer.assert_released()
            producer.poison()
            assert_true(result.dtype().is_categorical())
            assert_equal(result.get(0).string(), "red" if duplicate else "blue")
            assert_true(result.get(1).is_null())
            assert_equal(result.get(2).string(), "green")
            assert_equal(result.get(3).string(), "red" if duplicate else "blue")


# expected-failure: Valid Arrow dictionary indices that refer to null
# expected-failure: dictionary values raise instead of yielding null cells.
# expected-diagnostic: Cannot read a null value
def test_arrow_dictionary_null_values_decode_as_null_cells() raises:
    var producer = pyarrow_producer(4)
    var result = import_arrow_series(
        producer.nodes[0].array, producer.nodes[0].schema
    )
    producer.assert_released()
    producer.poison()
    assert_equal(result.get(0).string(), "blue")
    assert_true(result.get(1).is_null())
    assert_true(
        result.get(2).is_null(),
        "a valid dictionary index referring to a null value yields a null cell",
    )
    assert_equal(result.get(3).string(), "blue")


def nested_foreign(large: Bool) -> ForeignArrow:
    var producer = ForeignArrow(4)
    producer.nodes[0].describe("+L" if large else "+l", "records", 3, 1, 1)
    producer.nodes[0].buffer([7])
    producer.nodes[0].buffer(integers([0, 1, 3, 3, 4], 8 if large else 4))
    producer.nodes[0].children = [1]
    producer.nodes[0].metadata = packed_metadata(
        [[0, 255, 120]], [[0, 254, 13]]
    )
    producer.nodes[1].describe("+s", "item", 4, 0, 1)
    producer.nodes[1].buffer([11])
    producer.nodes[1].children = [2, 3]
    producer.nodes[1].metadata = packed_metadata(
        [text_bytes("application")], [text_bytes("nested")]
    )
    producer.nodes[2].describe("l", "id", 4)
    producer.nodes[2].buffer([])
    producer.nodes[2].buffer(integers([7, 8, 9, 10], 8))
    producer.nodes[2].metadata = packed_metadata(
        [
            text_bytes("ARROW:extension:name"),
            text_bytes("ARROW:extension:metadata"),
        ],
        [text_bytes("unregistered.test"), [0, 255, 123]],
    )
    producer.nodes[3].describe("u", "name", 4, 0, 1)
    producer.nodes[3].buffer([11])
    producer.nodes[3].buffer(integers([0, 1, 2, 2, 3], 4))
    producer.nodes[3].buffer(text_bytes("abd"))
    producer.seal()
    return producer^


def test_arrow_nested_lists_structs_and_arbitrary_child_metadata_are_owned() raises:
    for large in [False, True]:
        var producer = nested_foreign(large)
        var expected_root = producer.nodes[0].metadata.copy()
        var expected_struct = producer.nodes[1].metadata.copy()
        var expected_child = producer.nodes[2].metadata.copy()
        var result = import_arrow_series(
            producer.nodes[0].array, producer.nodes[0].schema
        )
        producer.assert_released()
        producer.poison()
        assert_true(result.dtype().is_list())
        assert_equal(len(result), 3)
        var items = result.get(0).list()
        assert_equal(len(items), 2)
        assert_equal(items.get(0).struct_field("id").int64(), Int64(8))
        assert_equal(items.get(0).struct_field("name").string(), "b")
        assert_true(items.get(1).is_null())
        assert_equal(len(result.get(1).list()), 0)
        assert_true(result.get(2).is_null())
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow_series(result, array, schema)
        assert_equal(c_string(schema.format), "+L")
        assert_bytes(schema.metadata, expected_root)
        var struct_schema = at[ArrowSchema](at[Int](schema.children)[])
        assert_bytes(struct_schema[].metadata, expected_struct)
        var numeric_schema = at[ArrowSchema](
            at[Int](struct_schema[].children)[]
        )
        assert_bytes(numeric_schema[].metadata, expected_child)
        assert_equal(c_string(numeric_schema[].format), "l")
        array_release(array)
        schema_release(schema)


def test_arrow_decimal_width_precision_and_scale_match_external_integer_buffers() raises:
    var formats: List[String] = ["d:9,2,32", "d:18,3,64", "d:38,4"]
    var widths: List[Int] = [4, 8, 16]
    var scales: List[Int] = [2, 3, 4]
    var precision: List[Int] = [9, 18, 38]
    var expected_negative: List[String] = ["-12.34", "-1.234", "-0.1234"]
    var expected_positive: List[String] = ["56.78", "5.678", "0.5678"]
    for index in range(3):
        var producer = fixed_foreign(
            formats[index],
            widths[index],
            [999, -1234, 777, 5678],
            3,
            1,
            [11],
            1,
        )
        producer.nodes[0].metadata = packed_metadata([[0, 255]], [[7, 0, 254]])
        # Metadata was added after seal, so publish its stable address explicitly.
        producer.nodes[0].schema.metadata = Int(
            producer.nodes[0].metadata.unsafe_ptr()
        )
        var expected_metadata = producer.nodes[0].metadata.copy()
        var result = import_arrow_series(
            producer.nodes[0].array, producer.nodes[0].schema
        )
        producer.assert_released()
        producer.poison()
        assert_equal(
            result.dtype(),
            DataType.decimal(
                precision[index], scales[index], widths[index] * 8
            ),
        )
        assert_equal(String(result.get(0)), expected_negative[index])
        assert_true(result.get(1).is_null())
        assert_equal(String(result.get(2)), expected_positive[index])
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow_series(result, array, schema)
        assert_equal(c_string(schema.format), formats[index])
        assert_equal(
            read_signed(buffer(array, 1), Int(array.offset), widths[index]),
            Int128(-1234),
        )
        assert_equal(
            read_signed(buffer(array, 1), Int(array.offset) + 2, widths[index]),
            Int128(5678),
        )
        assert_bytes(schema.metadata, expected_metadata)
        array_release(array)
        schema_release(schema)


def test_arrow_temporal_formats_scale_units_and_retain_timezone_metadata() raises:
    var formats: List[String] = [
        "tdD",
        "tdm",
        "tts",
        "ttm",
        "ttu",
        "ttn",
        "tDs",
        "tDm",
        "tDu",
        "tDn",
        "tss:UTC",
        "tsm:",
        "tsu:+05:30",
        "tsn:America/New_York",
    ]
    var widths: List[Int] = [4, 8, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    var factors: List[Int64] = [
        1,
        1,
        1_000_000_000,
        1_000_000,
        1000,
        1,
        1000,
        1,
        1,
        1,
        1000,
        1,
        1,
        1,
    ]
    var export_formats: List[String] = [
        "tdD",
        "tdD",
        "ttn",
        "ttn",
        "ttn",
        "ttn",
        "tDm",
        "tDm",
        "tDu",
        "tDn",
        "tsm:UTC",
        "tsm:",
        "tsu:+05:30",
        "tsn:America/New_York",
    ]
    for index in range(len(formats)):
        var producer = fixed_foreign(
            formats[index], widths[index], [999, -1, 777, 42], 3, 1, [11], 1
        )
        var result = import_arrow_series(
            producer.nodes[0].array, producer.nodes[0].schema
        )
        producer.assert_released()
        producer.poison()
        var negative = -factors[index]
        var positive = 42 * factors[index]
        if formats[index] == "tdm":
            negative = -1  # date64 milliseconds floor to civil days
            positive = 0
        assert_equal(result.get(0).to_physical(), negative)
        assert_true(result.get(1).is_null())
        assert_equal(result.get(2).to_physical(), positive)
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow_series(result, array, schema)
        assert_equal(c_string(schema.format), export_formats[index])
        var exported_width = (
            4 if formats[index] == "tdD" or formats[index] == "tdm" else 8
        )
        assert_equal(
            read_signed(buffer(array, 1), Int(array.offset), exported_width),
            Int128(negative),
        )
        assert_equal(
            read_signed(
                buffer(array, 1), Int(array.offset) + 2, exported_width
            ),
            Int128(positive),
        )
        array_release(array)
        schema_release(schema)


# expected-failure: Null second-unit temporal payloads are rescaled and
# expected-failure: overflow instead of remaining null.
# expected-diagnostic: Arrow seconds value overflows milliseconds
def test_arrow_null_seconds_payload_does_not_overflow_unit_conversion() raises:
    for format in ["tss:UTC", "tDs"]:
        var producer = fixed_foreign(
            format, 8, [1, Int128(Int64.MAX), 2], 3, 0, [5], 1
        )
        var result = import_arrow_series(
            producer.nodes[0].array, producer.nodes[0].schema
        )
        producer.assert_released()
        assert_equal(result.get(0).to_physical(), Int64(1000))
        assert_true(result.get(1).is_null())
        assert_equal(result.get(2).to_physical(), Int64(2000))


def expect_rejection(
    mut producer: ForeignArrow, diagnostic: String, frame: Bool = False
) raises:
    var rejected = False
    try:
        if frame:
            _ = import_arrow(producer.nodes[0].array, producer.nodes[0].schema)
        else:
            _ = import_arrow_series(
                producer.nodes[0].array, producer.nodes[0].schema
            )
    except error:
        if String(error).find(diagnostic) < 0:
            raise error^
        rejected = True
    assert_true(rejected, "Arrow must reject: " + diagnostic)
    producer.assert_released()


def test_arrow_malformed_formats_children_dictionaries_and_overflow_release_every_node() raises:
    var unknown = fixed_foreign("?", 8, [1], 1)
    expect_rejection(unknown, "Unsupported Arrow format '?'")
    var decimal = fixed_foreign("d:2", 8, [1], 1)
    expect_rejection(decimal, "Invalid Arrow decimal format")
    var unit = fixed_foreign("tsx:", 8, [1], 1)
    expect_rejection(unit, "Unsupported Arrow time unit 'x'")
    var seconds = fixed_foreign("tss:UTC", 8, [Int128(Int64.MAX)], 1)
    expect_rejection(seconds, "Arrow seconds value overflows milliseconds")
    var list = nested_foreign(False)
    list.nodes[0].array.n_children = 0
    expect_rejection(list, "Arrow list array must have one child")
    var structure = nested_foreign(False)
    structure.nodes[1].schema.n_children = 1
    expect_rejection(structure, "Arrow struct children counts differ")
    var dictionary = dictionary_foreign("i", 4)
    dictionary.nodes[0].schema.dictionary = 0
    expect_rejection(
        dictionary, "Arrow dictionary array without its dictionary"
    )
    var indices = dictionary_foreign("i", 4)
    # Replace index 1 (the first logical row) with an out-of-range code.
    indices.nodes[0].bytes[1][5] = 100
    expect_rejection(indices, "Arrow dictionary index out of range")
    var frame = ForeignArrow(2)
    frame.nodes[0].describe("+s", "", 1, 0, 1)
    frame.nodes[0].buffer([0])
    frame.nodes[0].children = [1]
    frame.nodes[1].describe("l", "x", 1)
    frame.nodes[1].buffer([])
    frame.nodes[1].buffer(integers([1], 8))
    frame.seal()
    expect_rejection(
        frame, "Arrow struct arrays with null rows are not supported", True
    )


# expected-failure: Malformed empty Int64 C layout with one buffer slot is
# expected-failure: accepted instead of rejected.
# expected-diagnostic: Arrow numeric layout must reject missing required
# expected-diagnostic: buffer slot
def test_arrow_empty_numeric_array_rejects_missing_required_buffer_slot() raises:
    var producer = ForeignArrow(1)
    producer.nodes[0].describe("l", "empty", 0)
    producer.nodes[0].buffer(
        []
    )  # validity only; Arrow Int64 requires two slots
    producer.seal()
    var rejected = False
    try:
        _ = import_arrow_series(
            producer.nodes[0].array, producer.nodes[0].schema
        )
    except error:
        if String(error).find("buffer") < 0:
            raise error^
        rejected = True
    producer.assert_released()
    assert_true(
        rejected,
        "Arrow numeric layout must reject missing required buffer slot",
    )


def test_arrow_public_export_is_read_by_an_independent_struct_and_bitmap_consumer() raises:
    var integers = Series(
        "integer",
        Column[Int64]([999, 11, 777, -33, 44], [True, True, False, True, True]),
    ).slice(1, 4)
    var texts = Series(
        "text",
        Column[String](
            ["prefix", "", "hé", "poison", "z"], [True, True, True, False, True]
        ),
    ).slice(1, 4)
    var flags = Series(
        "flag",
        Column[Bool](
            [False, True, True, False, True], [True, True, False, True, True]
        ),
    ).slice(1, 4)
    var source = DataFrame([integers^, texts^, flags^])
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow(source, array, schema)
    assert_equal(c_string(schema.format), "+s")
    assert_equal(array.length, Int64(4))
    assert_equal(array.n_children, Int64(3))
    var expected_formats: List[String] = ["l", "U", "b"]
    var expected_names: List[String] = ["integer", "text", "flag"]
    for column in range(3):
        ref child = at[ArrowArray](
            at[Int](array.children)[unsafe_offset=column]
        )[]
        ref field = at[ArrowSchema](
            at[Int](schema.children)[unsafe_offset=column]
        )[]
        assert_equal(c_string(field.format), expected_formats[column])
        assert_equal(c_string(field.name), expected_names[column])
        assert_equal(child.length, Int64(4))
        assert_equal(child.offset, Int64(1))
        for row in range(4):
            var valid = row != (2 if column == 1 else 1)
            assert_equal(
                bitmap_valid(buffer(child, 0), Int(child.offset) + row), valid
            )
            if not valid:
                continue
            if column == 0:
                var expected: List[Int128] = [11, 0, -33, 44]
                assert_equal(
                    read_signed(buffer(child, 1), Int(child.offset) + row, 8),
                    expected[row],
                )
            elif column == 1:
                var expected: List[String] = ["", "hé", "", "z"]
                var first = Int(
                    read_signed(buffer(child, 1), Int(child.offset) + row, 8)
                )
                var last = Int(
                    read_signed(
                        buffer(child, 1), Int(child.offset) + row + 1, 8
                    )
                )
                assert_equal(last - first, expected[row].byte_length())
                for byte in range(last - first):
                    assert_equal(
                        at[UInt8](buffer(child, 2))[unsafe_offset=first + byte],
                        expected[row].as_bytes()[byte],
                    )
            else:
                assert_equal(
                    bitmap_valid(buffer(child, 1), Int(child.offset) + row),
                    row == 0 or row == 3,
                )
    array_release(array)
    schema_release(schema)
    assert_equal(array.release, 0)
    assert_equal(schema.release, 0)


def test_arrow_imports_actual_pyarrow_c_exports_with_literal_typed_expectations() raises:
    for fixture_id in [0, 1, 2, 3, 5, 6, 7, 8, 9]:
        var producer = pyarrow_producer(fixture_id)
        var expected_metadata = producer.nodes[0].metadata.copy()
        var result = import_arrow_series(
            producer.nodes[0].array, producer.nodes[0].schema
        )
        producer.assert_released()
        producer.poison()
        if fixture_id == 0:
            assert_equal(result.dtype(), DataType.INT64)
            assert_equal(result.get(0).int64(), Int64(11))
            assert_true(result.get(1).is_null())
            assert_equal(result.get(2).int64(), Int64(-33))
            assert_equal(result.get(3).int64(), Int64(44))
        elif fixture_id == 1:
            assert_equal(result.dtype(), DataType.BOOL)
            assert_equal(result.get(0).bool(), True)
            assert_true(result.get(1).is_null())
            assert_equal(result.get(2).bool(), False)
            assert_equal(result.get(3).bool(), True)
        elif fixture_id == 2:
            assert_equal(result.dtype(), DataType.STRING)
            assert_equal(result.get(0).string(), "")
            assert_equal(result.get(1).string(), "hé")
            assert_true(result.get(2).is_null())
            assert_equal(result.get(3).string(), "z")
        elif fixture_id == 3:
            var bytes: List[UInt8] = [255, 0, 1]
            var z: List[UInt8] = [122]
            assert_equal(result.dtype(), DataType.BINARY)
            assert_equal(result.get(0).bytes(), List[UInt8]())
            assert_equal(result.get(1).bytes(), bytes)
            assert_true(result.get(2).is_null())
            assert_equal(result.get(3).bytes(), z)
        elif fixture_id == 5:
            assert_equal(len(result), 3)
            var items = result.get(0).list()
            assert_equal(len(items), 2)
            assert_equal(items.get(0).struct_field("id").int64(), Int64(8))
            assert_equal(items.get(0).struct_field("name").string(), "b")
            assert_true(items.get(1).is_null())
            assert_equal(len(result.get(1).list()), 0)
            assert_true(result.get(2).is_null())
        elif fixture_id < 9:
            var negative: List[String] = ["-12.34", "-1.234", "-0.1234"]
            var positive: List[String] = ["56.78", "5.678", "0.5678"]
            assert_true(result.dtype().is_decimal())
            assert_equal(String(result.get(0)), negative[fixture_id - 6])
            assert_true(result.get(1).is_null())
            assert_equal(String(result.get(2)), positive[fixture_id - 6])
        else:
            assert_equal(
                result.dtype(), DataType.datetime("ns", "America/New_York")
            )
            assert_equal(result.get(0).to_physical(), Int64(-1))
            assert_true(result.get(1).is_null())
            assert_equal(result.get(2).to_physical(), Int64(42))
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow_series(result, array, schema)
        assert_bytes(schema.metadata, expected_metadata)
        array_release(array)
        schema_release(schema)
    var batch = pyarrow_producer(10)
    var frame = import_arrow(batch.nodes[0].array, batch.nodes[0].schema)
    batch.assert_released()
    batch.poison()
    assert_equal(frame.columns(), List[String](["integer", "text"]))
    assert_equal(frame.height(), 3)
    assert_equal(frame.item(0, "integer").int64(), Int64(11))
    assert_true(frame.item(1, "integer").is_null())
    assert_equal(frame.item(2, "integer").int64(), Int64(-33))
    assert_equal(frame.item(0, "text").string(), "")
    assert_equal(frame.item(1, "text").string(), "hé")
    assert_true(frame.item(2, "text").is_null())


def test_arrow_empty_arrays_and_zero_batch_eof_release_each_live_producer() raises:
    from dataframe.io.parquet import _ParquetBatches
    from tests.support.lifecycle_stream import StreamState, counted_stream

    var empty = fixed_foreign("l", 8, [], 0)
    var result = import_arrow_series(
        empty.nodes[0].array, empty.nodes[0].schema
    )
    empty.assert_released()
    assert_equal(result.dtype(), DataType.INT64)
    assert_equal(len(result), 0)
    var state = StreamState(0, 0)
    var cursor = _ParquetBatches(counted_stream(state), 1)
    assert_true(not cursor.next())
    assert_true(not cursor.next())
    assert_equal(state.next_calls, 1)
    assert_equal(state.schema_calls, 0)
    assert_equal(state.root_array_releases, 0)
    assert_equal(state.root_schema_releases, 0)
    assert_equal(state.stream_releases, 1)
    _ = cursor^
    assert_equal(state.stream_releases, 1)


comptime ARROW_NUMERIC_DTYPES = (
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


def numeric_layout[D: DType](format: String, width: Int) raises:
    var first = Scalar[D](1.25) if D.is_floating_point() else Scalar[D].MAX
    var last = Scalar[D](-2.5) if D.is_floating_point() else Scalar[D].MIN
    var raw_first: Int128
    var raw_last: Int128
    comptime if D == DType.float32:
        raw_first = 0x3FA00000
        raw_last = 0xC0200000
    elif D == DType.float64:
        raw_first = 0x3FF4000000000000
        raw_last = 0xC004000000000000
    else:
        raw_first = first.cast[DType.int128]()
        raw_last = last.cast[DType.int128]()
    var producer = fixed_foreign(
        format, width, [999, raw_first, 777, raw_last], 3, 1, [11], 1
    )
    var imported = import_arrow_series(
        producer.nodes[0].array, producer.nodes[0].schema
    )
    producer.assert_released()
    producer.poison()
    assert_equal(imported.dtype(), DataType.of(D))
    assert_equal(imported.get(0).numeric[D](), first)
    assert_true(imported.get(1).is_null())
    assert_equal(imported.get(2).numeric[D](), last)
    var values: List[Scalar[D]] = [Scalar[D](0), first, Scalar[D](9), last]
    var native = Series(
        "number", Column[Scalar[D]](values^, [True, True, False, True])
    ).slice(1, 3)
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(native, array, schema)
    assert_equal(c_string(schema.format), format)
    assert_equal(array.offset, Int64(1))
    assert_true(bitmap_valid(buffer(array, 0), 1))
    assert_true(not bitmap_valid(buffer(array, 0), 2))
    assert_true(bitmap_valid(buffer(array, 0), 3))
    comptime if D.is_floating_point():
        assert_equal(
            read_unsigned(buffer(array, 1), 1, width),
            raw_first.cast[DType.uint128](),
        )
        assert_equal(
            read_unsigned(buffer(array, 1), 3, width),
            raw_last.cast[DType.uint128](),
        )
    elif D.is_signed():
        assert_equal(read_signed(buffer(array, 1), 1, width), raw_first)
        assert_equal(read_signed(buffer(array, 1), 3, width), raw_last)
    else:
        assert_equal(
            read_unsigned(buffer(array, 1), 1, width),
            raw_first.cast[DType.uint128](),
        )
        assert_equal(
            read_unsigned(buffer(array, 1), 3, width),
            raw_last.cast[DType.uint128](),
        )
    array_release(array)
    schema_release(schema)


def test_arrow_all_numeric_widths_use_standard_formats_and_exact_independent_buffers() raises:
    var formats: List[String] = [
        "c",
        "s",
        "i",
        "l",
        "C",
        "S",
        "I",
        "L",
        "f",
        "g",
    ]
    var widths: List[Int] = [1, 2, 4, 8, 1, 2, 4, 8, 4, 8]
    comptime for index in range(len(ARROW_NUMERIC_DTYPES)):
        numeric_layout[ARROW_NUMERIC_DTYPES[index]](
            formats[index], widths[index]
        )


def test_arrow_categorical_export_is_decoded_by_an_independent_dictionary_consumer() raises:
    var source = (
        Series(
            "category",
            Column[String](
                ["ignored", "blue", "red", "poison", "blue"],
                [True, True, True, False, True],
            ),
        )
        .cast(DataType.CATEGORICAL)
        .slice(1, 4)
    )
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(source, array, schema)
    assert_equal(c_string(schema.format), "I")
    assert_true(array.dictionary != 0 and schema.dictionary != 0)
    ref dictionary = at[ArrowArray](array.dictionary)[]
    ref dictionary_schema = at[ArrowSchema](schema.dictionary)[]
    assert_equal(c_string(dictionary_schema.format), "U")
    var expected: List[String] = ["blue", "red", "", "blue"]
    for row in range(4):
        assert_equal(
            bitmap_valid(buffer(array, 0), Int(array.offset) + row), row != 2
        )
        if row == 2:
            continue
        var index = Int(
            read_unsigned(buffer(array, 1), Int(array.offset) + row, 4)
        ) + Int(dictionary.offset)
        var first = Int(read_signed(buffer(dictionary, 1), index, 8))
        var last = Int(read_signed(buffer(dictionary, 1), index + 1, 8))
        assert_equal(last - first, expected[row].byte_length())
        for byte in range(last - first):
            assert_equal(
                at[UInt8](buffer(dictionary, 2))[unsafe_offset=first + byte],
                expected[row].as_bytes()[byte],
            )
    array_release(array)
    schema_release(schema)
