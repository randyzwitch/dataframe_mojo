"""End-to-end coverage for the clean explicit-schema CSV orchestration."""
from std.collections import Dict
from dataframe.dtype import DataType
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from dataframe.csv import CsvField, CsvOptions, CsvSchema
from dataframe.csv_reader import _decode_unmapped, _prelude, read_csv_explicit, read_csv_inferred
from dataframe.frame import DataFrame


comptime PATH = "/tmp/dataframe_mojo_explicit_reader_test.csv"


def schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("id"),
            CsvField.string("name"),
            CsvField.float64("score"),
        ]
    )


def options(
    comment: String = "", skip_rows: Int = 0, n_rows: Int = -1
) -> CsvOptions:
    return CsvOptions(
        ",", '"', comment, skip_rows, n_rows, [], False, False, "utf8"
    )


def prelude(text: String, options: CsvOptions) -> Tuple[Int, Int]:
    var owned = List[UInt8]()
    owned.extend(text.as_bytes())
    var bytes = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=owned.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(owned),
    )
    var result = _prelude(bytes, options, True)
    _ = owned^
    return result


def cells(frame: DataFrame) raises -> String:
    var out = String()
    for row in range(frame.height()):
        for column in range(frame.width()):
            out += String(frame._columns[column].get(row))
            if column + 1 < frame.width():
                out += "|"
        out += "\n"
    return out^


def test_prelude_bom_empty_comments_skip_and_header() raises:
    var text = String(
        "\ufeff\n# ignored before skip\n"
        'skip,"quoted\nrecord",0\n'
        "# ignored after skip\n"
        "id,name,score\n1,a,1\n"
    )
    var prelude = prelude(text, options("#", 1))
    # BOM, leading empty record, one valid skipped CSV record, comments, and
    # then the header are all consumed before decode begins at id=1.
    assert_equal(
        String(unsafe_from_utf8=text.as_bytes()[prelude[0] :]), "1,a,1\n"
    )


def test_mapped_explicit_reader_applies_same_prelude() raises:
    var text = String(
        "\ufeff\n# ignored before skip\n"
        'skip,"quoted\nrecord",0\n'
        "# ignored after skip\n"
        'id,name,score\n1,a,1\n2,"b,b",2\n'
    )
    with open(PATH, "w") as file:
        file.write(text)
    var frame = read_csv_explicit(
        PATH, schema(), comment_prefix="#", skip_rows=1
    )
    assert_equal(cells(frame), "1|a|1.0\n2|b,b|2.0\n")


def test_unmapped_fallback_uses_prelude_and_global_limit() raises:
    var input = List[UInt8]()
    input.extend(
        String("\ufeff# note\nid,name,score\n1,a,1\n2,b,2\n3,c,3\n").as_bytes()
    )
    var frame = _decode_unmapped(
        input^, schema(), options("#", 0, 2), [True, True, True], True
    )
    assert_equal(cells(frame), "1|a|1.0\n2|b|2.0\n")


def test_multichunk_projection_and_global_limit() raises:
    var text = String("id,name,score\n")
    for i in range(2200):
        text += String(i) + ',"value,' + String(i) + '",' + String(i) + ".5\n"
    with open(PATH, "w") as file:
        file.write(text)
    var frame = read_csv_explicit(PATH, schema(), columns=["name"], n_rows=1033)
    assert_equal(frame.width(), 1)
    assert_equal(frame.height(), 1033)
    assert_equal(frame.column("name").string().value(0), "value,0")
    assert_equal(frame.column("name").string().value(1032), "value,1032")


def test_late_worker_error_drains_before_mapping_is_released() raises:
    var text = String("id,name,score\n")
    for i in range(900):
        text += String(i) + ",ok,1\n"
    # This record lands after the first CountLines window, so an asynchronously
    # published job can fail after earlier range jobs already borrowed mmap.
    text += "901,late,not-a-float\n"
    with open(PATH, "w") as file:
        file.write(text)
    with assert_raises():
        _ = read_csv_explicit(PATH, schema())


def test_final_unterminated_record_and_comment_match_read_impl_count() raises:
    # CountLines has no LF to count here, so read_impl explicitly supplies one
    # final non-comment record for decode_chunk.
    with open(PATH, "w") as file:
        file.write("id,name,score\n1,last,1")
    var record = read_csv_explicit(PATH, schema())
    assert_equal(cells(record), "1|last|1.0\n")

    # Its paired EOF branch supplies zero only when the *post-prelude body*
    # begins with the comment prefix. This must not become a malformed-row
    # failure after the decoder correctly emits no records.
    with open(PATH, "w") as file:
        file.write("id,name,score\n# final comment")
    var comment = read_csv_explicit(PATH, schema(), comment_prefix="#")
    assert_equal(comment.height(), 0)


def test_inferred_reader_shares_mapping_and_projected_decode() raises:
    with open(PATH, "w") as file:
        file.write('id,score,active,name\n1,1.5,true,"a,b"\n2,,false,z\n')
    var result = read_csv_inferred(PATH)
    assert_equal(result.height(), 2)
    assert_equal(result.column("id").dtype(), DataType.INT64)
    assert_equal(result.column("score").dtype(), DataType.FLOAT64)
    assert_equal(result.column("active").dtype(), DataType.BOOL)
    assert_equal(result.column("name").string().value(0), "a,b")
    assert_equal(result.column("score").null_count(), 1)
    var override = Dict[String, String]()
    override["id"] = "string"
    var projected = read_csv_inferred(PATH, columns=["id"], schema_overrides=override, n_rows=1)
    assert_equal(projected.height(), 1)
    assert_equal(projected.column("id").dtype(), DataType.STRING)
    assert_equal(projected.column("id").string().value(0), "1")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
