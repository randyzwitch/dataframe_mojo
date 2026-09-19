"""CSV schema inference: candidate order, sampling limits, and overrides."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import CsvField, CsvSchema, read_csv

comptime PATH = "/tmp/dataframe_mojo_infer_test.csv"


def write(text: String) raises:
    with open(PATH, "w") as file:
        file.write(text)


def test_candidate_order_and_identifiers() raises:
    write(
        "b,i,f,mixed,zip,big,s,empty,quoted_empty\n"
        + 'true,1,1.5,1,007,9223372036854775808,x,,""\n'
        + "false,-2,2,2.5,10,1,y,,a\n"
        + ",,,,,,,,\n"
    )
    var frame = read_csv(PATH)
    assert_equal(
        frame.dtypes(),
        [
            String("bool"),
            "int64",
            "float64",
            "float64",
            "string",
            "string",
            "string",
            "string",
            "string",
        ],
    )
    assert_equal(frame.item(0, "zip").string(), "007")
    assert_equal(frame.item(0, "big").string(), "9223372036854775808")
    assert_true(frame.item(2, "i").is_null())
    assert_equal(frame.item(1, "mixed").float64(), 2.5)
    assert_equal(frame.item(0, "quoted_empty").string(), "")


def test_inferred_equals_explicit_schema() raises:
    write("id,name,score\n1,a,1.5\n2,b,\n3,,4\n")
    var explicit = read_csv(
        PATH,
        CsvSchema(
            [
                CsvField.int64("id"),
                CsvField.string("name"),
                CsvField.float64("score"),
            ]
        ),
    )
    assert_true(read_csv(PATH).equals(explicit))
    assert_true(read_csv(PATH, buffer_size=3).equals(explicit))


def test_sample_limits_and_conflicts() raises:
    write("v\n1\n2\n3\nx\n")
    with assert_raises(
        contains="CSV record 5, field 'v': invalid Int64 value 'x'"
    ):
        _ = read_csv(PATH, infer_schema_length=2)
    with assert_raises(contains="pass schema_overrides"):
        _ = read_csv(PATH, infer_schema_length=3)
    assert_equal(read_csv(PATH, infer_schema_length=-1).dtypes()[0], "string")
    assert_equal(read_csv(PATH, infer_schema_length=4).dtypes()[0], "string")
    # A sample longer than the file is fine.
    assert_equal(read_csv(PATH, infer_schema_length=1000).height(), 4)
    # Overrides win over inference, and the sample can be empty.
    var overridden = read_csv(
        PATH, infer_schema_length=0, schema_overrides={"v": "string"}
    )
    assert_equal(overridden.item(3, "v").string(), "x")
    assert_equal(read_csv(PATH, infer_schema_length=0).dtypes()[0], "string")
    with assert_raises(contains="schema_overrides names unknown column: w"):
        _ = read_csv(PATH, schema_overrides={"w": "int64"})
    with assert_raises(contains="Unknown dtype in schema_overrides"):
        _ = read_csv(PATH, schema_overrides={"v": "date"})
    with assert_raises(contains="infer_schema_length"):
        _ = read_csv(PATH, infer_schema_length=-2)


def test_names_headers_and_empty_files() raises:
    write("a,a,,a\n1,2,3,4\n")
    assert_equal(read_csv(PATH).columns(), [String("a"), "a_1", "", "a_2"])
    write("1,x\n2,y\n")
    var headerless = read_csv(PATH, has_header=False)
    assert_equal(headerless.columns(), [String("column_1"), "column_2"])
    assert_equal(headerless.dtypes(), [String("int64"), "string"])
    assert_equal(headerless.height(), 2)
    write("a,b\n")
    var header_only = read_csv(PATH)
    assert_equal(header_only.columns(), [String("a"), "b"])
    assert_equal(header_only.height(), 0)
    assert_equal(header_only.dtypes()[0], "string")
    write("")
    assert_equal(read_csv(PATH).width(), 0)
    write("a,b\n1,2\n3\n")
    with assert_raises(contains="different field counts"):
        _ = read_csv(PATH)


def test_inference_uses_reader_options() raises:
    write("# note\nid;flag\n1;NA\n2;true\n")
    var frame = read_csv(
        PATH, separator=";", comment_prefix="#", null_values=["NA"]
    )
    assert_equal(frame.dtypes(), [String("int64"), "bool"])
    assert_true(frame.item(0, "flag").is_null())
    assert_equal(
        read_csv(PATH, separator=";", comment_prefix="#", n_rows=1).height(), 1
    )
    assert_equal(
        read_csv(
            PATH, separator=";", comment_prefix="#", columns=["flag"]
        ).columns(),
        [String("flag")],
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
