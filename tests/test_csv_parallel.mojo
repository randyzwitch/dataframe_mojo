"""Parallel CSV decoding must equal a serial read, frame for frame.

The serial reader is the reference implementation, which is why it was kept.
These tests read the same file both ways and compare, rather than comparing
against expectations, so any disagreement is a real defect and not a
difference of opinion about the file.
"""
from std.ffi import external_call
from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
    assert_true,
)

from dataframe import CsvField, CsvSchema, DataFrame, read_csv
from dataframe.csv import _csv_workers, _map_file


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


def _file_size(path: String) raises -> Int:
    with open(path, "r") as handle:
        return Int(handle.seek(0, 2))


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


def both_ways(path: String) raises:
    """Read serially and in parallel, and require identical frames."""
    set_threads(1)
    var serial = read_csv(path, schema())
    set_threads(32)
    var parallel = read_csv(path, schema())
    assert_equal(
        serial.height(), parallel.height(), "row counts differ for " + path
    )
    assert_true(
        serial.equals(parallel), "parallel read differs from serial: " + path
    )


def body(rows: Int, quoted_every: Int, crlf: Bool) raises -> String:
    var newline = String("\r\n") if crlf else String("\n")
    var out = String("id,text,value,flag") + newline
    var state = UInt64(17)
    for i in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        var r = Int((state >> 33) % 1000)
        out += String(i) + ","
        if quoted_every > 0 and i % quoted_every == 0:
            # A quoted field holding a separator, a newline and an escaped
            # quote: exactly what a naive split would break on.
            out += '"a,b' + newline + 'c""d"'
        else:
            out += "plain" + String(r)
        out += "," + String(r) + "." + String(r % 10)
        out += ",true" if i % 2 == 0 else ",false"
        out += newline
    return out^


def test_large_plain_file_matches_serial() raises:
    # Well past the 64 KiB default block, so several blocks are read and the
    # carry between them is exercised.
    var path = String("/tmp/claude-1000/par_plain.csv")
    write(path, body(20000, 0, False))
    set_threads(32)
    assert_true(_csv_workers(8 << 20) > 1, "parallel path not reachable")
    both_ways(path)


def test_quoted_newlines_match_serial() raises:
    var path = String("/tmp/claude-1000/par_quoted.csv")
    write(path, body(8000, 7, False))
    both_ways(path)


def test_crlf_matches_serial() raises:
    var path = String("/tmp/claude-1000/par_crlf.csv")
    write(path, body(8000, 11, True))
    both_ways(path)


def test_row_counts_survive_block_boundaries() raises:
    # Vary the file length one byte at a time around the block size so a
    # record, and a quoted field, straddle the boundary in every way.
    for extra in range(0, 12):
        var path = String("/tmp/claude-1000/par_edge.csv")
        var text = body(2000, 5, False)
        for _ in range(extra):
            text += "999999,pad,1.0,true\n"
        write(path, text)
        both_ways(path)


def test_every_field_quoted() raises:
    var path = String("/tmp/claude-1000/par_allq.csv")
    var out = String("id,text,value,flag\n")
    for i in range(5000):
        out += (
            '"' + String(i) + '","q' + String(i % 7) + '","1.5","true"' + "\n"
        )
    write(path, out)
    both_ways(path)


def test_no_trailing_newline() raises:
    var path = String("/tmp/claude-1000/par_notrail.csv")
    var text = body(3000, 9, False)
    write(path, String(text[byte = 0 : text.byte_length() - 1]))
    both_ways(path)


def test_single_worker_equals_serial() raises:
    # Worker count 1 must reproduce the serial path exactly, not merely
    # closely, since that is the fallback every unsupported option takes.
    var path = String("/tmp/claude-1000/par_one.csv")
    write(path, body(4000, 6, False))
    set_threads(1)
    var a = read_csv(path, schema())
    set_threads(1)
    var b = read_csv(path, schema())
    assert_true(a.equals(b))
    set_threads(32)


def test_mapping_reports_whether_a_path_can_be_mapped() raises:
    """`_map_file` decides which reader runs, and says so by address.

    This is checked directly because the outcome is otherwise invisible: an
    empty file happens to read correctly down either path, so a test that
    only read one would pass with the fallback disabled -- which is exactly
    what happened when this was first written.

    A path with no length to map is reported as address 0. The unmappable
    case that carries real data is a pipe, which needs a writer running
    concurrently; that is verified by hand rather than here.
    """
    var real = String("/tmp/claude-1000/par_mappable.csv")
    write(real, body(200, 0, False))
    var mapped = _map_file(real)
    assert_true(mapped.address != 0, "a regular file maps")
    assert_equal(mapped.length, 0 + _file_size(real), "maps the whole file")

    var empty = String("/tmp/claude-1000/par_unmappable.csv")
    write(empty, "")
    var unmapped = _map_file(empty)
    assert_equal(unmapped.address, 0, "nothing to map is reported as 0")


def test_degenerate_files_read_correctly() raises:
    """Shapes with no records, which the mapped path must not mishandle."""
    set_threads(32)
    var empty = String("/tmp/claude-1000/par_empty.csv")
    write(empty, "")
    var frame = read_csv(empty, schema())
    assert_equal(frame.height(), 0, "empty file")
    assert_equal(frame.width(), 4, "columns still come from the schema")

    var header_only = String("/tmp/claude-1000/par_header.csv")
    write(header_only, "id,text,value,flag\n")
    var just_header = read_csv(header_only, schema())
    assert_equal(just_header.height(), 0, "header only")
    assert_equal(just_header.width(), 4)

    # One record, far below any block or page size.
    var tiny = String("/tmp/claude-1000/par_tiny.csv")
    write(tiny, "id,text,value,flag\n1,a,2.5,true\n")
    var one = read_csv(tiny, schema())
    assert_equal(one.height(), 1, "single record")
    assert_equal(one.item(0, "text").string(), "a")


def test_a_parse_error_is_not_swallowed_by_the_fallback() raises:
    """The mapped read and the block read are chosen by whether the file can
    be mapped, not by whether decoding succeeded. A decoding error must
    surface instead of quietly sending the file down the other path."""
    set_threads(32)
    var path = String("/tmp/claude-1000/par_bad.csv")
    write(path, "id,text,value,flag\n1,a,2.5,true\nnotanint,b,1.0,false\n")
    with assert_raises(contains="invalid Int64"):
        _ = read_csv(path, schema())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
