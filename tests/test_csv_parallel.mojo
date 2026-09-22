"""Worker-count invariance for the single Polars-derived CSV pipeline.

The retired reader selected scalar, mmap, and range-tokenizer implementations
from file shape and options. The public replacement has one CountLines/decode
pipeline, so every fixture below compares its one-worker and 32-worker output.
"""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from dataframe import CsvField, CsvSchema, DataFrame, read_csv


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


def schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("id"),
            CsvField.string("text"),
            CsvField.float64("value"),
            CsvField.bool("flag"),
        ]
    )


def write(path: String, body: String) raises:
    with open(path, "w") as handle:
        handle.write(body)


def read_at_workers(path: String, workers: Int) raises -> DataFrame:
    set_threads(workers)
    return read_csv(path, schema())


def assert_workers_match(path: String) raises:
    var one = read_at_workers(path, 1)
    var many = read_at_workers(path, 32)
    assert_equal(one.height(), many.height())
    assert_true(one.equals(many), "one and 32 workers disagree: " + path)


def body(rows: Int, quoted_every: Int, crlf: Bool) raises -> String:
    var newline = String("\r\n") if crlf else String("\n")
    var out = String("id,text,value,flag") + newline
    for i in range(rows):
        out += String(i) + ","
        if quoted_every > 0 and i % quoted_every == 0:
            out += '"a,b' + newline + 'c""d"'
        else:
            out += "plain" + String(i % 97)
        out += "," + String(i) + ".5,"
        out += "true" if i % 2 == 0 else "FALSE"
        out += newline
    return out^


def test_plain_quoted_crlf_and_eof_records_match() raises:
    var complete = body(3_000, 9, False)
    var without_trailing_lf = String(
        complete[byte = 0 : complete.byte_length() - 1]
    )
    for fixture in [
        body(20_000, 0, False),
        body(8_000, 7, False),
        body(8_000, 11, True),
        without_trailing_lf,
    ]:
        var path = String("/tmp/dataframe_mojo_csv_parallel.csv")
        write(path, fixture)
        assert_workers_match(path)


def test_all_quoted_fields_and_many_simd_boundary_offsets() raises:
    var out = String("id,text,value,flag\n")
    for i in range(5_000):
        out += '"' + String(i) + '","q' + String(i % 7) + '","1.5","TrUe"\n'
    for extra in range(0, 66):
        out += String(extra) + ',"x\ny",2.5,false\n'
    var path = String("/tmp/dataframe_mojo_csv_parallel_quotes.csv")
    write(path, out)
    assert_workers_match(path)


def test_global_limit_projection_and_empty_shapes_match() raises:
    var path = String("/tmp/dataframe_mojo_csv_parallel_limits.csv")
    write(path, body(3_000, 5, False))
    set_threads(1)
    var one = read_csv(path, schema(), columns=["text"], n_rows=1_033)
    set_threads(32)
    var many = read_csv(path, schema(), columns=["text"], n_rows=1_033)
    assert_true(one.equals(many))
    assert_equal(many.height(), 1_033)
    assert_equal(many.width(), 1)

    write(path, "")
    assert_workers_match(path)
    write(path, "id,text,value,flag\n")
    assert_workers_match(path)


def test_parse_failures_surface_at_both_worker_counts() raises:
    var path = String("/tmp/dataframe_mojo_csv_parallel_bad.csv")
    write(path, body(8_000, 7, True) + "not-an-int,plain,1.0,true\n")
    for workers in [1, 32]:
        set_threads(workers)
        with assert_raises():
            _ = read_csv(path, schema())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
