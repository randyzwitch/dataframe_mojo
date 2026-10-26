"""Physical values and errors from independently executed typed JSON decoding."""
# contract-group: kernel.json
from std.testing import assert_equal, assert_true
from std.memory import bitcast
from dataframe import DataType, Series, Column
from dataframe.column import SCALAR_DTYPES
from dataframe.json_decode import decode_json
from tests.kernel_equivalence.test_jsonpath_corpus import _unhex


def _dtype(name: String) raises -> DataType:
    if name == "boolean":
        return DataType.BOOL
    if name == "string":
        return DataType.STRING
    if name == "categorical":
        return DataType.CATEGORICAL
    if name == "date":
        return DataType.DATE
    if name == "time":
        return DataType.TIME
    if name == "datetime_utc":
        return DataType.datetime("us", "UTC")
    if name == "datetime_fixed":
        return DataType.datetime("us", "+05:30")
    if name == "datetime_iana":
        return DataType.datetime("us", "America/New_York")
    if name.startswith("datetime_"):
        return DataType.datetime(String(name[byte=9:]))
    if name.startswith("duration_"):
        return DataType.duration(String(name[byte=9:]))
    if name == "decimal_8_2":
        return DataType.decimal(8, 2)
    if name == "decimal_20_5":
        return DataType.decimal(20, 5)
    if name == "decimal_38_8":
        return DataType.decimal(38, 8)
    comptime for D in SCALAR_DTYPES:
        if name == String(D):
            return DataType.of(D)
    raise Error("Unknown oracle dtype " + name)


def _cell(result: Series) raises -> String:
    var dtype = result.dtype()
    if result.null_count() != 0:
        return "NULL"
    if dtype.is_decimal():
        return "I" + String(result._decimal_raw(0))
    if dtype.is_temporal():
        return "I" + String(result._data[Column[Int64]]._get(0))
    if dtype == DataType.STRING or dtype.is_categorical():
        var text = result.get(0).string()
        var hex = StringSlice("0123456789abcdef").as_bytes()
        var out = List[UInt8]()
        out.append(83)
        for byte in text.as_bytes():
            out.append(hex[Int(byte) // 16])
            out.append(hex[Int(byte) % 16])
        return String(unsafe_from_utf8=out)
    if dtype == DataType.BOOL:
        return "B1" if result.get(0).bool() else "B0"
    comptime for D in SCALAR_DTYPES:
        if result._data.isa[Column[Scalar[D]]]():
            var value = result._data[Column[Scalar[D]]]._get(0)
            comptime if D.is_floating_point():
                return "F" + String(bitcast[DType.uint64](Float64(value)))
            else:
                return "I" + String(value)
    raise Error("Unknown oracle output dtype")


def test_json_decode_matches_independent_polars_scalar_corpus() raises:
    var corpus = open("tests/fixtures/json_strings/decode.tsv", "r").read()
    var failures = 0
    var cases = 0
    var strict_unicode = 0
    for line in corpus.split("\n"):
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var fields = line.split("\t")
        var document = _unhex(fields[1])
        var actual = String("ERROR")
        try:
            var source = Series("", Column[String]([document.copy()]))
            actual = _cell(decode_json(source, _dtype(String(fields[0]))))
        except:
            pass
        # The shared strict reader rejects unpaired Unicode surrogates.
        # Polars 2.0's decoder instead manufactures U+0000 for this input.
        # Keep the independent fixture unchanged and assert our documented
        # rejection explicitly, rather than accepting corrupted Unicode.
        var matches = actual == fields[2]
        if document == '"\\uD800"':
            matches = actual == "ERROR"
            strict_unicode += 1
        if not matches:
            failures += 1
            if failures <= 100:
                print(
                    "MISMATCH",
                    fields[0],
                    document,
                    "got",
                    actual,
                    "expected",
                    fields[2],
                )
        cases += 1
    print("JSON_DECODE_CORPUS", cases, "cases", failures, "mismatches")
    print("STRICT_UNICODE_REJECTIONS", strict_unicode)
    assert_true(cases > 2000)
    assert_equal(failures, 0)
