"""Decoder scratch and reusable builders cannot mutate completed owned output."""
# contract-group: storage.isolation
# contract-group: io.csv_builders
from std.testing import assert_equal, assert_true
from dataframe import CsvField, DataType, Series
from dataframe.io.csv_buffers import _CsvColumnBuffer
from tests.support.numeric_model import INTEGER_DTYPES, FLOAT_DTYPES


def _append(
    mut builder: _CsvColumnBuffer, text: String, escaped: Bool = False
) raises:
    builder.add(
        Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=text.as_bytes()
            .unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            length=text.byte_length(),
        ),
        escaped,
        False,
    )


def _numeric_outputs[D: DType]() raises -> List[Series]:
    var outputs = List[Series]()
    var builder = _CsvColumnBuffer(CsvField("value", DataType.of(D)), 1)
    for round in range(5):
        for row in range(17):
            if (row + round) % 3 == 1:
                builder.add_null()
            else:
                _append(builder, String(row + round * 17))
        outputs.append(builder.finish())
    return outputs^


def _numeric[D: DType]() raises:
    var outputs = _numeric_outputs[D]()
    for round in range(5):
        assert_true(outputs[round].dtype() == DataType.of(D))
        assert_equal(len(outputs[round]), 17)
        for row in range(17):
            var cell = outputs[round].get(row)
            assert_equal(cell.is_null(), (row + round) % 3 == 1)
            if not cell.is_null():
                assert_equal(cell.numeric[D](), Scalar[D](row + round * 17))


def test_numeric_csv_builder_finish_transfers_every_width_and_validity_without_aliasing_later_outputs() raises:
    comptime for i in range(len(INTEGER_DTYPES)):
        _numeric[INTEGER_DTYPES[i]]()
    comptime for i in range(len(FLOAT_DTYPES)):
        _numeric[FLOAT_DTYPES[i]]()


def _string_outputs() raises -> Tuple[List[Series], List[List[String]]]:
    var outputs = List[Series]()
    var models = List[List[String]]()
    var builder = _CsvColumnBuffer(CsvField.string("value"), 1)
    for round in range(5):
        var expected: List[String] = [
            "",
            "short",
            "twelve-bytes",
            "é世界-" + String(round),
            'long "quoted" field with\nline and owner ' + String(round),
            String("x") * (70000 + round),
        ]
        for value in expected:
            _append(builder, '"' + value.replace('"', '""') + '"', True)
        builder.add_null()
        outputs.append(builder.finish())
        models.append(expected^)
    return (outputs^, models^)


def test_csv_escaped_scratch_and_finished_string_owners_survive_reuse_and_destruction() raises:
    var pair = _string_outputs()
    var outputs = pair[0].copy()
    var models = pair[1].copy()
    for round in range(5):
        assert_true(outputs[round].dtype() == DataType.STRING)
        assert_equal(len(outputs[round]), 7)
        assert_true(outputs[round].get(6).is_null())
        for row in range(6):
            assert_equal(outputs[round].get(row).string(), models[round][row])
