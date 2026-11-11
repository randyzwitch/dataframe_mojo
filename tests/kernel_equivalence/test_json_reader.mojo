"""Shared strict JSON scanning, tape offsets, and legacy metadata behavior."""
# contract-group: kernel.json
from std.testing import assert_equal, assert_true, assert_raises
from dataframe.json.reader import (
    JsonReader,
    json_bytes,
)
from dataframe.json.value import (
    JsonValue,
    json_quote,
)


def test_json_tape_preserves_source_offsets_and_subtree_boundaries() raises:
    var text = String('{"a":[1,{"b":"x\\uD83D\\uDE00"}],"z":true}')
    var reader = JsonReader(json_bytes(text))
    reader.parse(json_bytes(text))
    assert_equal(len(reader.nodes), 6)
    assert_equal(reader.nodes[0].after, 6)
    assert_equal(reader.nodes[1].after, 5)
    assert_equal(reader.nodes[3].after, 5)
    assert_equal(String(reader.raw(2)), "1")
    assert_equal(reader.string_at(reader.nodes[4].start), "x😀")
    assert_equal(
        String(
            reader.slice(reader.nodes[5].key_start, reader.nodes[5].key_end)
        ),
        '"z"',
    )
    var again = String("[null,false,-12.5e+2]")
    reader.parse(json_bytes(again))
    assert_equal(len(reader.nodes), 4)
    assert_equal(String(reader.raw(3)), "-12.5e+2")
    _ = text^
    _ = again^


def test_json_source_spans_keep_wide_numbers_and_validate_every_token() raises:
    var text = String(
        ' {"wide":170141183460469231731687303715884105727,'
        '"unsigned":340282366920938463463374607431768211455,'
        '"fraction":-0.0012300e+12,"escaped":"\\u0000\\uD83D\\uDE00"} '
    )
    var reader = JsonReader(json_bytes(text))
    reader.parse(json_bytes(text))
    assert_equal(
        String(reader.raw(1)), "170141183460469231731687303715884105727"
    )
    assert_equal(
        String(reader.raw(2)), "340282366920938463463374607431768211455"
    )
    assert_equal(String(reader.raw(3)), "-0.0012300e+12")
    assert_equal(reader.string_at(reader.nodes[4].start), "\x00😀")
    for invalid in [
        String('{"selected":1,"ignored":[12x]}'),
        String('{"selected":1,"ignored":{"x":truefalse}}'),
        String('{"selected":1,"ignored":"\\uD800"}'),
        String('{"a":"x"true}'),
        String('{"unterminated:1}'),
    ]:
        with assert_raises():
            reader.parse(json_bytes(invalid))
        reader = JsonReader(json_bytes(invalid))
        with assert_raises():
            _ = reader.scan()
    _ = text^


def test_shared_json_reader_keeps_metadata_duplicate_policy_and_canonicalization() raises:
    var value = JsonValue(
        ' {"b":[true,null,{"escaped":"a\\n\\u00e9"}],"a":1.20e2} '
    )
    assert_equal(
        value.canonical(), '{"a":1.20e2,"b":[true,null,{"escaped":"a\\né"}]}'
    )
    assert_equal(value.get("missing").text, "null")
    assert_equal(value.get("b").items()[2].get("escaped").string(), "a\né")
    assert_equal(json_quote('"\\\t\n'), '"\\"\\\\\\t\\n"')
    with assert_raises(contains="Duplicate"):
        _ = JsonValue('{"a":1,"\\u0061":2}')
    var duplicates = String('{"a":1,"a":2}')
    var reader = JsonReader(json_bytes(duplicates))
    reader.parse(json_bytes(duplicates))
    assert_equal(len(reader.nodes), 3)
    _ = duplicates^


def test_shared_json_reader_rejects_invalid_tokens_and_unicode() raises:
    var invalid: List[String] = [
        "",
        " ",
        "+1",
        "01",
        "-01",
        "1.",
        "1e",
        "1e+",
        "true false",
        "\vnull",
        "null\f",
        "nul",
        "[1,]",
        '{"a":1,}',
        '{"a" 1}',
        "[}",
        '"\\x41"',
        '"\\uD800"',
        '"\\uDC00"',
        '"\\uD800\\u0041"',
        '"unterminated',
        '"a\nb"',
    ]
    for text in invalid:
        var raised = False
        try:
            _ = JsonValue(text)
        except:
            raised = True
        assert_true(raised, "accepted invalid JSON " + text)
    var deep = String()
    for _ in range(66):
        deep += "["
    deep += "0"
    for _ in range(66):
        deep += "]"
    with assert_raises(contains="nesting"):
        _ = JsonValue(deep)
    var malformed_bytes: List[UInt8] = [34, 192, 128, 34]
    var malformed = String(unsafe_from_utf8=malformed_bytes)
    with assert_raises(contains="UTF-8"):
        _ = JsonValue(malformed)
    var reader = JsonReader(json_bytes(malformed))
    with assert_raises(contains="UTF-8"):
        _ = reader.scan()


def test_json_index_overflow_is_rejected_before_accessing_input() raises:
    var empty = String("")
    var reader = JsonReader(json_bytes(empty))
    # This view intentionally describes inaccessible bytes. The adapter must
    # reject the unsupported size before UTF-8 validation or indexing reads it.
    var oversized = Span(
        unsafe_ptr=json_bytes(empty).unsafe_ptr(), length=Int(UInt32.MAX)
    )
    with assert_raises(contains="32-bit index"):
        reader.parse(oversized)
    _ = empty^


def _detached_json_child() raises -> JsonValue:
    var parent = JsonValue(
        ' \t{"ignored":0,"value":{"z":[{"\\u0061":"é"},2],"a":null}}\n'
    )
    return parent.get("value")


def test_indexed_json_children_keep_offsets_and_outlive_parent() raises:
    var child = _detached_json_child()
    var copy = child.copy()
    _ = child^
    assert_equal(copy.keys(), List[String](["z", "a"]))
    assert_true(copy.has("a"))
    assert_true(not copy.has("missing"))
    assert_equal(copy.get("a").kind(), UInt8(110))
    assert_equal(copy.get("missing").text, "null")
    assert_equal(copy.get("z").items()[0].get("a").string(), "é")
    assert_equal(copy.canonical(), '{"a":null,"z":[{"a":"é"},2]}')
    with assert_raises(contains="object"):
        _ = copy.get("z").has("a")
    with assert_raises(contains="string"):
        _ = copy.string()


def test_indexed_json_wide_reverse_keys_and_nested_values() raises:
    var source = String("{")
    var expected = String("{")
    var count = 2048
    for i in range(count):
        if i:
            source += ","
            expected += ","
        # Fixed-width keys make lexical and numerical order identical.
        var key = String(10000 + count - i - 1)
        source += json_quote(key) + ':[{"value":' + String(i) + "}]"
        expected += (
            json_quote(String(10000 + i))
            + ':[{"value":'
            + String(count - i - 1)
            + "}]"
        )
    source += "}"
    expected += "}"
    var value = JsonValue(source)
    assert_equal(len(value.keys()), count)
    for i in range(count):
        var key = String(10000 + i)
        assert_true(value.has(key))
        assert_equal(
            value.get(key).items()[0].get("value").text,
            String(count - i - 1),
        )
    assert_equal(value.canonical(), expected)
