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


# Tests from test_csv_consumers.mojo.
# Public CSV chunked output remains usable by dataframe consumers.
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import CsvField, CsvSchema, read_csv, col, lit, to_csv_string


def test_public_csv_chunks_match_contiguous_consumers() raises:
    var path = String("/tmp/dataframe_mojo_csv_consumer_replacement.csv")
    with open(path, "w") as output:
        output.write("id,label,value\n")
        for i in range(5000):
            output.write(
                String(i)
                + ",group"
                + String(i % 13)
                + ","
                + String(i % 97)
                + ".5\n"
            )
    var frame = read_csv(
        path,
        CsvSchema(
            [
                CsvField.int64("id"),
                CsvField.string("label"),
                CsvField.float64("value"),
            ]
        ),
    )
    assert_equal(frame.height(), 5000)
    assert_true(frame.column("id").n_chunks() > 1)
    var flat = frame.rechunk()
    assert_true(frame.equals(flat))
    assert_true(
        frame.select(col("value").sum()).equals(flat.select(col("value").sum()))
    )
    assert_true(
        frame.filter(col("id") > lit(Int64(4980))).equals(
            flat.filter(col("id") > lit(Int64(4980)))
        )
    )
    assert_true(frame.sort(["label", "id"]).equals(flat.sort(["label", "id"])))
    assert_true(
        frame.group_by(["label"], maintain_order=True)
        .agg([col("value").sum()])
        .equals(
            flat.group_by(["label"], maintain_order=True).agg(
                [col("value").sum()]
            )
        )
    )
    var right = flat.head(3)
    assert_true(
        frame.join(right, on=["id"], how="inner").equals(
            flat.join(right, on=["id"], how="inner")
        )
    )
    assert_equal(to_csv_string(frame), to_csv_string(flat))


# Tests from test_csv_integer_reader.mojo.
# Public read_csv coverage for the CSV-only atoi_simd integer parser.
from std.testing import TestSuite, assert_equal, assert_raises
from dataframe import CsvField, CsvSchema, DataType, read_csv


comptime CSV_PATH = "/tmp/dataframe_mojo_csv_integer_legacy.csv"


def _write(text: String) raises:
    with open(CSV_PATH, "w") as file:
        file.write(text)


def _schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField("i8", DataType.INT8, False),
            CsvField("u8", DataType.UINT8, False),
            CsvField("i16", DataType.INT16, False),
            CsvField("u16", DataType.UINT16, False),
            CsvField("i32", DataType.INT32, False),
            CsvField("u32", DataType.UINT32, False),
            CsvField.int64("i64", False),
            CsvField("u64", DataType.UINT64, False),
        ]
    )


def test_public_csv_integer_widths_and_source_boundaries() raises:
    _write(
        "i8,u8,i16,u16,i32,u32,i64,u64\n"
        "-128,255,-32768,65535,-2147483648,4294967295,"
        "-9223372036854775808,18446744073709551615\n"
        "+000000000000000000000001,0000000000000000000000002,"
        "000000000000000000000000003,0000000000000000000000000004,"
        "000000000000000000000000000005,0000000000000000000000000000006,"
        "+000000000000000000000000000000000000007,"
        "0000000000000000000000000000000000000008"
    )
    var frame = read_csv(CSV_PATH, _schema(), buffer_size=1)
    assert_equal(frame.height(), 2)
    assert_equal(frame.column("i8").int8().value(0), Int8(-128))
    assert_equal(frame.column("u8").uint8().value(0), UInt8(255))
    assert_equal(frame.column("i16").int16().value(0), Int16(-32768))
    assert_equal(frame.column("u16").uint16().value(0), UInt16(65535))
    assert_equal(frame.column("i32").int32().value(0), Int32(-2147483648))
    assert_equal(frame.column("u32").uint32().value(0), UInt32(4294967295))
    assert_equal(frame.column("i64").int64().value(0), Int64.MIN)
    assert_equal(frame.column("u64").uint64().value(0), UInt64.MAX)
    assert_equal(frame.column("i8").int8().value(1), Int8(1))
    assert_equal(frame.column("u64").uint64().value(1), UInt64(8))


def test_public_csv_integer_overflow_and_invalid_bytes() raises:
    var schema = CsvSchema([CsvField("u8", DataType.UINT8, False)])
    # atoi_simd's unsigned route rejects every negative spelling, including
    # -0. dataframe.parse keeps its historical generic-cast behavior instead.
    for text in ["u8\n256", "u8\n-1", "u8\n-0", "u8\n12x", "u8\n+"]:
        _write(text)
        with assert_raises():
            _ = read_csv(CSV_PATH, schema, buffer_size=1)
    # Polars has nullable CSV columns even when legacy schema metadata says
    # nullable=False; a bare empty integer field is a null, not a parse error.
    _write("u8\n\n")
    var empty = read_csv(CSV_PATH, schema, buffer_size=1)
    assert_equal(empty.column("u8").null_count(), 1)


def test_public_csv_temporal_fields_keep_their_existing_parser() raises:
    var schema = CsvSchema([CsvField.date("day", nullable=False)])
    _write("day\n1970-01-02")
    var frame = read_csv(CSV_PATH, schema, buffer_size=1)
    assert_equal(frame.column("day").int64().value(0), Int64(1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
