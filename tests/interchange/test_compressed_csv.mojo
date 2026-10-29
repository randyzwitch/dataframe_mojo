"""Compressed transport preserves the existing CSV parser and scan contracts."""
# contract-group: io.csv.compressed
from std.collections import Dict
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    CsvBuffer,
    CsvField,
    CsvSchema,
    DataFrame,
    Series,
    StringColumn,
    read_csv,
    scan_csv,
)
from tests.support.coverage_compare import assert_frame


def fixture(name: String) -> String:
    return "tests/fixtures/compressed_csv/" + name


def buffer(name: String) raises -> CsvBuffer:
    with open(fixture(name), "r") as file:
        return CsvBuffer(file.read_bytes())


def core() raises -> DataFrame:
    return DataFrame(
        [
            Series("id", Column[Int64]([1, 2, 3])),
            Series("text", StringColumn(["α\nβ", 'a "quote"', "last"])),
        ]
    )


def test_compressed_files_use_content_before_extension_and_join_frames() raises:
    var expected = core()
    for name in [
        "core.csv.gz",
        "core.csv.zst",
        "magic-gzip.bin",
        "magic-zstd.bin",
        "conflict-gzip.csv.zst",
        "conflict-zstd.csv.gz",
        "concat-gzip.csv.gz",
        "concat-zstd.csv.zst",
    ]:
        assert_frame(read_csv(fixture(name)), expected)
        var schema = CsvSchema([CsvField.int64("id"), CsvField.string("text")])
        assert_frame(read_csv(fixture(name), schema), expected)
        for streaming in [True, False]:
            assert_frame(
                scan_csv(fixture(name)).collect(
                    streaming=streaming, batch_size=1
                ),
                expected,
            )


def test_compressed_buffers_are_reusable_and_outputs_retain_values() raises:
    for name in [
        "core.csv.gz",
        "core.csv.zst",
        "concat-gzip.csv.gz",
        "concat-zstd.csv.zst",
    ]:
        var source = buffer(name)
        assert_frame(read_csv(source), core())
        assert_frame(read_csv(source, infer_schema_length=-1), core())
        assert_frame(
            read_csv(source, columns=["text"], n_rows=2),
            core().select(["text"]).head(2),
        )
    var result = read_csv(buffer("core.csv.zst"))
    assert_frame(result, core())


def test_compressed_options_projection_and_schema_only_match_literal_values() raises:
    var expected = DataFrame(
        [
            Series("id", Column[Int64]([1, 2, 3])),
            Series(
                "text", StringColumn(["α,\nβ", "", ""], [True, False, True])
            ),
        ]
    )
    for suffix in ["gz", "zst"]:
        var name = "options.csv." + suffix
        assert_frame(
            read_csv(
                buffer(name),
                separator=";",
                quote_char="'",
                comment_prefix="#",
                skip_rows=1,
                null_values=["NA"],
            ),
            expected,
        )
        var plan = scan_csv(
            fixture(name),
            separator=";",
            quote_char="'",
            comment_prefix="#",
            skip_rows=1,
            null_values=["NA"],
        )
        assert_equal(plan.collect_schema(), ["id: int64", "text: string"])
        for optimize in [True, False]:
            assert_frame(
                plan.select(["text"])
                .head(2)
                .collect(streaming=True, optimize=optimize, batch_size=1),
                expected.select(["text"]).head(2),
            )
        var overrides = Dict[String, String]()
        overrides["id"] = "int32"
        assert_frame(
            read_csv(fixture("core.csv." + suffix), schema_overrides=overrides),
            DataFrame(
                [
                    Series("id", Column[Int32]([1, 2, 3])),
                    Series("text", StringColumn(["α\nβ", 'a "quote"', "last"])),
                ]
            ),
        )


def test_compressed_full_inference_and_header_only_preserve_schema() raises:
    for suffix in ["gz", "zst"]:
        var expected = DataFrame(
            [Series("x", StringColumn(["1", "2", "late"]))]
        )
        assert_frame(
            read_csv(fixture("infer.csv." + suffix), infer_schema_length=-1),
            expected,
        )
        assert_frame(
            scan_csv(
                fixture("infer.csv." + suffix), infer_schema_length=-1
            ).collect(streaming=True, batch_size=1),
            expected,
        )
        var schema = CsvSchema([CsvField.int64("id"), CsvField.string("text")])
        var empty = DataFrame(
            [
                Series("id", Column[Int64](List[Int64]())),
                Series("text", StringColumn(List[String]())),
            ]
        )
        assert_frame(
            read_csv(fixture("header-only.csv." + suffix), schema), empty
        )
        assert_frame(read_csv(fixture("empty.csv." + suffix), schema), empty)
        assert_frame(
            read_csv(fixture("core.csv." + suffix), schema, n_rows=0), empty
        )
        assert_frame(
            read_csv(
                fixture("disabled-quote.csv." + suffix),
                separator="|",
                quote_char="",
            ),
            DataFrame(
                [
                    Series("x", Column[Int64]([1, 2])),
                    Series("s", StringColumn(['a"b', "c"])),
                ]
            ),
        )


def test_compressed_records_cross_read_boundaries_with_utf8_and_escaped_quotes() raises:
    var bytes = List[UInt8](length=65523, fill=120)
    bytes.extend('"α\n'.as_bytes())
    for _ in range(65536):
        bytes.append(121)
    var text = String(unsafe_from_utf8=bytes)
    var expected = DataFrame(
        [
            Series("id", Column[Int64]([1, 2])),
            Series("text", StringColumn([text, "tail"])),
        ]
    )
    for suffix in ["gz", "zst"]:
        var name = "boundary.csv." + suffix
        assert_frame(read_csv(buffer(name)), expected)
        assert_frame(
            scan_csv(fixture(name)).collect(streaming=True, batch_size=1),
            expected,
        )


def test_compressed_full_reads_validate_integrity_even_with_ignore_errors() raises:
    for suffix in ["gz", "zst"]:
        for stem in ["truncated", "junk", "late-truncated", "invalid"]:
            var name = stem + ".csv." + suffix
            var failed = False
            try:
                _ = read_csv(fixture(name), ignore_errors=True)
            except:
                failed = True
            assert_true(failed)
            failed = False
            try:
                _ = scan_csv(fixture(name)).collect(
                    streaming=True, batch_size=65536
                )
            except:
                failed = True
            assert_true(failed)
        var failed = False
        try:
            _ = read_csv(buffer("truncated.csv." + suffix))
        except:
            failed = True
        assert_true(failed)
    var failed = False
    try:
        _ = read_csv(buffer("checksum.csv.gz"), ignore_errors=True)
    except:
        failed = True
    assert_true(failed)


def test_compressed_row_limits_do_not_consume_a_large_unread_tail() raises:
    # The malformed trailer is beyond Arrow's 1 MiB decompression read-ahead.
    # Finite schema inference and head(1) need only the valid first records.
    var expected = DataFrame([Series("x", Column[Int64]([1]))])
    for suffix in ["gz", "zst"]:
        var name = "late-truncated.csv." + suffix
        assert_frame(read_csv(fixture(name), n_rows=1), expected)
        assert_frame(read_csv(buffer(name), n_rows=1), expected)
        assert_frame(
            scan_csv(fixture(name))
            .head(1)
            .collect(streaming=True, batch_size=1),
            expected,
        )


def test_compressed_no_header_skip_projection_and_zero_sample_inference() raises:
    var schema = CsvSchema(
        [
            CsvField.int64("id"),
            CsvField.float64("amount"),
            CsvField.string("s"),
        ]
    )
    var expected = DataFrame(
        [
            Series("id", Column[Int64]([1, 2])),
            Series("amount", Column[Float64]([1.5, 0], [True, False])),
        ]
    )
    for suffix in ["gz", "zst"]:
        var name = "no-header.csv." + suffix
        assert_frame(
            read_csv(
                buffer(name),
                schema,
                has_header=False,
                skip_rows=1,
                separator="|",
                columns=["id", "amount"],
                null_values=["NA"],
                n_rows=2,
            ),
            expected,
        )
        var inferred = read_csv(
            fixture(name),
            has_header=False,
            skip_rows=1,
            separator="|",
            infer_schema_length=0,
        )
        assert_frame(
            inferred,
            DataFrame(
                [
                    Series("column_1", StringColumn(["1", "2", "3"])),
                    Series("column_2", StringColumn(["1.5", "NA", "3.5"])),
                    Series("column_3", StringColumn(["keep", "other", "last"])),
                ]
            ),
        )
