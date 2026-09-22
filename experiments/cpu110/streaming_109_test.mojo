from std.testing import TestSuite, assert_almost_equal, assert_raises

from dataframe import (
    CsvField,
    CsvSchema,
    stream_csv_filter_select_sum,
    stream_csv_parallel_filter_select_sum,
)

comptime PATH = "/tmp/dataframe_mojo_streaming109.csv"


def schema() raises -> CsvSchema:
    return CsvSchema([CsvField.float64("x"), CsvField.string("y")])


def write(text: String) raises:
    with open(PATH, "w") as file:
        file.write(text)


def test_multiline_quoted_records_drain_between_every_record() raises:
    write('x,y\n1,"alpha\nbeta"\n-2,drop\n3,"say ""yes"""\n')
    var result = stream_csv_filter_select_sum(PATH, schema(), batch_rows=1)
    assert_almost_equal(result.item().float64(), 8.0)


def test_parse_error_preserves_global_record_location() raises:
    write("x,y\n1,ok\nnot-a-number,bad\n")
    with assert_raises(contains="CSV record 3"):
        _ = stream_csv_filter_select_sum(PATH, schema(), batch_rows=1)


def test_parallel_window_pipeline_matches_multiline_and_errors() raises:
    write('x,y\n1,"alpha\nbeta"\n-2,drop\n3,"say ""yes"""\n')
    var result = stream_csv_parallel_filter_select_sum(
        PATH, schema(), window_bytes=4096
    )
    assert_almost_equal(result.item().float64(), 8.0)
    write("x,y\n1,ok\nnot-a-number,bad\n")
    with assert_raises(contains="CSV record 3"):
        _ = stream_csv_parallel_filter_select_sum(
            PATH, schema(), window_bytes=4096
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
