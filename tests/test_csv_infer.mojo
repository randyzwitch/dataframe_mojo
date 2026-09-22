"""Public CSV schema inference under the Polars-derived reader contract."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import DataType, CsvField, CsvSchema, read_csv


comptime PATH = "/tmp/dataframe_mojo_infer_test.csv"


def write(text: String) raises:
    with open(PATH, "w") as file:
        file.write(text)


def test_candidate_order_identifiers_and_overflow() raises:
    write(
        "b,i,f,mixed,zip,s,empty,quoted_empty\n"
        + 'true,1,1.5,1,007,x,,""\n'
        + "false,-2,2,2.5,10,y,,a\n"
        + ",,,,,,,\n"
    )
    var frame = read_csv(PATH)
    assert_equal(
        frame.dtypes(),
        [
            DataType.BOOL,
            DataType.INT64,
            DataType.FLOAT64,
            DataType.FLOAT64,
            DataType.INT64,
            DataType.STRING,
            DataType.STRING,
            DataType.STRING,
        ],
    )
    # INTEGER_RE recognizes leading zeroes, as in Polars.
    assert_equal(frame.item(0, "zip").int64(), Int64(7))
    assert_true(frame.item(2, "i").is_null())
    assert_equal(frame.item(1, "mixed").float64(), 2.5)
    assert_equal(frame.item(0, "quoted_empty").string(), "")

    # With dtype-i128 disabled Polars preserves INTEGER_RE as Int64. The
    # builder, not inference, rejects values outside the signed destination.
    for text in [
        "wide\n9223372036854775808\n",
        "wide\n18446744073709551615\n",
    ]:
        write(text)
        with assert_raises(contains="overflow"):
            _ = read_csv(PATH)


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


def test_sample_limits_default_100_and_overrides() raises:
    write("v\n1\n2\n3\nx\n")
    # A short sample infers Int64; later text is rejected by the parser. The
    # old reader appended a custom inference hint, which is not Polars behavior.
    for sample in [2, 3]:
        with assert_raises():
            _ = read_csv(PATH, infer_schema_length=sample)
    assert_equal(
        read_csv(PATH, infer_schema_length=-1).dtypes()[0], DataType.STRING
    )
    assert_equal(
        read_csv(PATH, infer_schema_length=4).dtypes()[0], DataType.STRING
    )
    assert_equal(read_csv(PATH, infer_schema_length=1000).height(), 4)
    assert_equal(
        read_csv(PATH, infer_schema_length=0).dtypes()[0], DataType.STRING
    )

    # The pinned Polars default is 100 sampled records rather than the old
    # public reader's 10,000.
    var sampled = String("v\n")
    for i in range(100):
        sampled += String(i) + "\n"
    sampled += "late-text\n"
    write(sampled)
    with assert_raises():
        _ = read_csv(PATH)
    assert_equal(
        read_csv(PATH, infer_schema_length=-1).dtypes()[0], DataType.STRING
    )

    var overridden = read_csv(
        PATH, infer_schema_length=0, schema_overrides={"v": "string"}
    )
    assert_equal(overridden.item(100, "v").string(), "late-text")
    # Polars name-keyed schema overrides ignore a name absent from the file.
    write("v\n1\n")
    assert_equal(
        read_csv(PATH, schema_overrides={"missing": "int64"}).dtypes()[0],
        DataType.INT64,
    )
    with assert_raises(contains="Unknown dtype in schema_overrides"):
        _ = read_csv(PATH, schema_overrides={"v": "decimal"})


def test_names_headers_and_empty_files() raises:
    write("a,a,,a\n1,2,3,4\n")
    assert_equal(
        read_csv(PATH).columns(),
        [String("a"), "a_duplicated_0", "", "a_duplicated_1"],
    )
    write("1,x\n2,y\n")
    var headerless = read_csv(PATH, has_header=False)
    assert_equal(headerless.columns(), [String("column_1"), "column_2"])
    assert_equal(headerless.dtypes(), [DataType.INT64, DataType.STRING])
    assert_equal(headerless.height(), 2)
    write("a,b\n")
    var header_only = read_csv(PATH)
    assert_equal(header_only.columns(), [String("a"), "b"])
    assert_equal(header_only.height(), 0)
    assert_equal(header_only.dtypes()[0], DataType.STRING)
    # The replacement uses Polars' default raise_if_empty=true.
    write("")
    with assert_raises(contains="CSV inference found no columns"):
        _ = read_csv(PATH)
    # Polars treats a missing tail field as null rather than a shape error.
    write("a,b\n1,2\n3\n")
    var ragged = read_csv(PATH)
    assert_equal(ragged.height(), 2)
    assert_true(ragged.item(1, "b").is_null())


def test_temporal_inference_is_opt_in() raises:
    write(
        "day,instant,clock\n"
        + "2024-02-29,2024-02-29T12:34:56,12:34:56\n"
        + "2024-03-01,2024-03-01 01:02:03,01:02\n"
    )
    assert_equal(
        read_csv(PATH).dtypes(),
        [DataType.STRING, DataType.STRING, DataType.STRING],
    )
    assert_equal(
        read_csv(PATH, try_parse_dates=True).dtypes(),
        [DataType.DATE, DataType.datetime("us"), DataType.TIME],
    )


def test_inference_uses_reader_options() raises:
    write("# note\nid;flag\n1;NA\n2;true\n")
    var frame = read_csv(
        PATH, separator=";", comment_prefix="#", null_values=["NA"]
    )
    assert_equal(frame.dtypes(), [DataType.INT64, DataType.BOOL])
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
