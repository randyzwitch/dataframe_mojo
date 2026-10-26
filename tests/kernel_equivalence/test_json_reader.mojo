"""Shared strict JSON scanning, tape offsets, and legacy metadata behavior."""
# contract-group: kernel.json
from std.testing import assert_equal, assert_true, assert_raises
from dataframe.json_reader import JsonReader, json_bytes
from dataframe.json_value import JsonValue, json_quote


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
    var malformed = String(unsafe_from_utf8=malformed_bytes^)
    with assert_raises(contains="UTF-8"):
        _ = JsonValue(malformed)
