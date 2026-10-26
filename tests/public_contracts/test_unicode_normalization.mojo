"""Unicode UAX15 conformance and public normalization across storage layouts."""
# contract-group: expr.unicode_normalization
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, StringColumn, DataType, col
from dataframe.unicode_normalization import normalize_series
from tests.public_contracts.test_text_functions import _source


def _decode_hex(text: StringSlice) -> String:
    var out = String()
    for word in text.split(" "):
        var cp = 0
        for byte in word.as_bytes():
            cp = cp * 16 + Int(byte - 48 if byte <= 57 else byte - 55)
        out += chr(cp)
    return out^


def test_unicode17_official_normalization_conformance() raises:
    var vectors = List[List[String]]()
    for _ in range(5):
        vectors.append(List[String]())
    var corpus = open(
        "tests/fixtures/unicode/normalization-17.0.0.txt", "r"
    ).read()
    for line in corpus.split("\n"):
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var fields = line.split(";")
        for k in range(5):
            vectors[k].append(_decode_hex(fields[k]))
    assert_true(len(vectors[0]) > 19000)
    for form in [String("NFC"), String("NFD"), String("NFKC"), String("NFKD")]:
        for input in range(5):
            var expected = 1 if form == "NFC" else (
                2 if form == "NFD" else (3 if form == "NFKC" else 4)
            )
            if input >= 3 and form == "NFC":
                expected = 3
            if input >= 3 and form == "NFD":
                expected = 4
            var result = normalize_series(
                Series("s", Column[String](vectors[input].copy())), form
            )
            ref strings = result._data[StringColumn]
            for row in range(len(vectors[input])):
                assert_equal(
                    String(strings._get(row)),
                    vectors[expected][row],
                    form + " input " + String(input) + " vector " + String(row),
                )


def test_normalization_forms_nulls_layouts_eager_lazy_and_schema() raises:
    var values: List[String] = [
        String("e\u0301"),
        String("é"),
        String("ＫＡＤＯＫＡＷＡ ²"),
        String("\u1100\u1161\u11A8"),
        String("a\u0315\u0300"),
        String("ascii text long enough for external Arrow view storage"),
        String(""),
    ]
    var valid: List[Bool] = [True, True, True, True, True, True, False]
    var expected: List[String] = [
        String("é"),
        String("é"),
        String("KADOKAWA 2"),
        String("각"),
        String("à\u0315"),
        values[5],
        String(""),
    ]
    for layout in range(4):
        var frame = _source(values, valid, layout)
        var expression = col("s").str().normalize("NFKC").alias("normalized")
        for mode in range(2):
            var result = frame.select(
                expression
            ) if mode == 0 else frame.lazy().select(expression).collect(
                engine="cpu"
            )
            var strings = result.column("normalized").rechunk()
            for row in range(len(values)):
                assert_equal(
                    strings._data[StringColumn]._valid(row), valid[row]
                )
                if valid[row]:
                    assert_equal(
                        String(strings._data[StringColumn]._get(row)),
                        expected[row],
                    )
        var schema = frame.lazy().select(expression).collect_schema()
        assert_equal(schema, [String("normalized: ") + DataType.STRING.name()])
        with assert_raises(contains="NFC"):
            _ = (
                frame.lazy()
                .select(col("s").str().normalize("invalid"))
                .collect_schema()
            )


def test_normalization_keeps_normalized_buffers_and_orders_long_combining_runs() raises:
    var source = Series(
        "s",
        Column[String](
            [String("already normalized"), String("漢字"), String("é")]
        ),
    )
    var result = normalize_series(source, "NFC")
    assert_equal(
        Int(source._data[StringColumn].unsafe_bytes()),
        Int(result._data[StringColumn].unsafe_bytes()),
    )
    var text = String("a")
    for _ in range(1000):
        text += "\u0315\u0300"
    var ordered = String("a")
    for _ in range(1000):
        ordered += "\u0300"
    for _ in range(1000):
        ordered += "\u0315"
    var marks = normalize_series(Series("s", Column[String]([text])), "NFD")
    assert_equal(String(marks._data[StringColumn]._get(0)), ordered)
