"""End-to-end CSV data coverage formerly exercised by borrowed-field fast paths.

The implementation-specific ``_feed_borrowed`` contract is gone. These
fixtures retain its valuable boundary coverage through the public reader and
require equal typed frames with one and 32 pipeline workers.
"""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from dataframe import CsvField, CsvSchema, DataFrame, read_csv

comptime PATH = "/tmp/dataframe_mojo_csv_pipeline_fields.csv"


def c_string(text: String) -> List[UInt8]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    bytes.append(0)
    return bytes^


def set_threads(n: Int):
    var name = c_string("DATAFRAME_THREADS")
    var value = c_string(String(n))
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


def write(text: String) raises:
    with open(PATH, "w") as file:
        file.write(text)


def write_bytes(bytes: List[UInt8]) raises:
    with open(PATH, "w") as file:
        file.write_bytes(bytes)


def typed_schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("id", False),
            CsvField.string("name"),
            CsvField.float64("score"),
            CsvField.bool("active"),
        ]
    )


def read_at_workers(workers: Int, n_rows: Int = -1) raises -> DataFrame:
    set_threads(workers)
    return read_csv(PATH, typed_schema(), n_rows=n_rows)


def assert_workers_match(n_rows: Int = -1) raises -> DataFrame:
    var one = read_at_workers(1, n_rows)
    var many = read_at_workers(32, n_rows)
    assert_true(one.equals(many), "one and 32 workers differ")
    return many^


def test_fields_at_every_structural_offset_and_two_simd_blocks() raises:
    var text = String("id,name,score,active\n")
    for i in range(130):
        text += String(i) + ",plain" + String(i) + "," + String(i) + ".5,"
        text += "TRUE\n" if i % 2 == 0 else "false\n"
    write(text)
    var frame = assert_workers_match()
    assert_equal(frame.height(), 130)
    assert_equal(frame.column("id").int64().value(129), Int64(129))
    assert_true(frame.column("active").bool().value(128))


def test_quoted_multiline_escaped_and_crlf_fields() raises:
    write(
        "id,name,score,active\r\n"
        + '1,"quoted, value",1,True\r\n'
        + '2,"say ""hello""",2,FALSE\r\n'
        + '3,"line one\r\nline two",3,true\r\n'
        + '4,"",4,false\r\n'
    )
    var frame = assert_workers_match()
    assert_equal(frame.column("name").string().value(0), "quoted, value")
    assert_equal(frame.column("name").string().value(1), 'say "hello"')
    assert_equal(frame.column("name").string().value(2), "line one\r\nline two")
    assert_equal(frame.column("name").string().value(3), "")


def test_null_tokens_apply_to_quoted_and_unquoted_fields() raises:
    # Polars applies user null markers after quote handling. CsvField.nullable
    # is metadata, so the non-nullable id field still carries a null here.
    write('id,name,score,active\nNA,"NA",NA,TRUE\n1,ok,2,false\n')
    set_threads(1)
    var one = read_csv(PATH, typed_schema(), null_values=["NA"])
    set_threads(32)
    var many = read_csv(PATH, typed_schema(), null_values=["NA"])
    assert_true(one.equals(many))
    assert_true(many.column("id").int64().is_null(0))
    assert_true(many.column("name").string().is_null(0))
    assert_true(many.column("score").float64().is_null(0))


def test_projection_and_global_limit_are_worker_invariant() raises:
    var text = String("id,name,score,active\n")
    for i in range(1_100):
        text += (
            String(i) + ',"value,' + String(i) + '",' + String(i) + ".5,true\n"
        )
    write(text)
    set_threads(1)
    var one = read_csv(PATH, typed_schema(), columns=["name"], n_rows=1_033)
    set_threads(32)
    var many = read_csv(PATH, typed_schema(), columns=["name"], n_rows=1_033)
    assert_true(one.equals(many))
    assert_equal(many.height(), 1_033)
    assert_equal(many.column("name").string().value(1032), "value,1032")


def test_invalid_utf8_is_rejected_before_projection() raises:
    var bytes: List[UInt8] = []
    bytes.extend("id,name,score,active\n1,".as_bytes())
    bytes.append(255)
    bytes.extend(",2,true\n".as_bytes())
    write_bytes(bytes)
    for workers in [1, 32]:
        set_threads(workers)
        with assert_raises(contains="not valid UTF-8"):
            _ = read_csv(PATH, typed_schema(), columns=["id"])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
