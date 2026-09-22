"""Pinned-Polars schema inference checks for the clean CSV inference port."""
from std.testing import TestSuite, assert_equal, assert_raises
from std.collections import Dict
from dataframe.csv import CsvOptions
from dataframe.csv_infer import infer_csv_schema
from dataframe.dtype import DataType


def options(
    separator: String = ",",
    comment: String = "",
    skip_rows: Int = 0,
    nulls: List[String] = List[String](),
) -> CsvOptions:
    return CsvOptions(
        separator,
        '"',
        comment,
        skip_rows,
        -1,
        nulls.copy(),
        False,
        False,
        "utf8",
    )


def infer(
    text: String,
    *,
    has_header: Bool = True,
    sample: Int = 100,
    input_options: CsvOptions = options(),
    overrides: Dict[String, String] = Dict[String, String](),
) raises -> String:
    var owned = List[UInt8]()
    owned.extend(text.as_bytes())
    var bytes = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=owned.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(owned),
    )
    var result = infer_csv_schema(
        bytes,
        input_options,
        has_header=has_header,
        infer_schema_length=sample,
        schema_overrides=overrides,
    )
    var out = String("offset=") + String(result.data_offset) + ";"
    for field in result.schema._fields:
        out += field.name + ":" + String(field.dtype) + ";"
    _ = owned^
    return out^


def infer_bytes(var owned: List[UInt8]) raises -> String:
    var bytes = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=owned.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(owned),
    )
    var result = infer_csv_schema(bytes, options())
    var out = String("offset=") + String(result.data_offset) + ";"
    for field in result.schema._fields:
        out += field.name + ":" + String(field.dtype) + ";"
    _ = owned^
    return out^


def test_polars_regex_candidate_order_and_header_dedup() raises:
    # FLOAT_RE recognizes signed inf/NaN, decimal/exponent forms, and \d+.
    # INTEGER_RE accepts leading zeroes, while +1 deliberately remains String.
    var actual = infer(
        'a,a,"b,b",a,i,f,sign\n'
        + "true,1,1.5,text,007,-7e-05,+1\n"
        + "FALSE,2,2.,other,-3,NaN,2\n"
    )
    assert_equal(
        actual,
        "offset=21;a:bool;a_duplicated_0:int64;b,b:float64;"
        + "a_duplicated_1:string;i:int64;f:float64;sign:string;",
    )


def test_prelude_matches_skipempty_comments_skipheader_and_bom() raises:
    var text = String(
        "\ufeff\n# before\n"
        + 'skip;"quoted\nrecord";x\n'
        + "# after\n"
        + "id;flag;value\n1;NA;1\n2;true;2.5\n"
    )
    var actual = infer(
        text,
        sample=-1,
        input_options=options(";", "#", 1, ["NA"]),
    )
    # Header and comments are excluded from the sample. `NA` is a configured
    # null candidate, so flag's remaining true supplies Boolean.
    assert_equal(
        actual,
        "offset=58;id:int64;flag:bool;value:float64;",
    )


def test_limits_headerless_quoted_and_null_only_columns() raises:
    assert_equal(
        infer("v,w\n1,\n2,\nlate,x\n", sample=2),
        "offset=4;v:int64;w:string;",
    )
    assert_equal(
        infer("v,w\n1,\n2,\nlate,x\n", sample=-1),
        "offset=4;v:string;w:string;",
    )
    # `Some(0)` in streaming.rs still observes the first content record, but
    # infer_all_as_str forces every encountered field to String.
    assert_equal(
        infer("v,w\n1,true\n", sample=0),
        "offset=4;v:string;w:string;",
    )
    assert_equal(
        infer('"1",x\n2,y\n', has_header=False),
        "offset=0;column_1:int64;column_2:string;",
    )


def test_default_sample_length_is_pinned_polars_100() raises:
    var source = String("v\n")
    for i in range(100):
        source += String(i) + "\n"
    source += "late-text\n"
    assert_equal(infer(source), "offset=2;v:int64;")
    var bytes = List[UInt8]()
    bytes.extend(source.as_bytes())
    # Omit the parameter entirely to guard the module's public default.
    assert_equal(infer_bytes(bytes^), "offset=2;v:int64;")
    assert_equal(infer(source, sample=-1), "offset=2;v:string;")


def test_crlf_and_exact_float_regex_spellings() raises:
    # `inf` and `NaN` are FLOAT_RE spellings, while uppercase INF/lowercase
    # nan and an incomplete decimal exponent are source-regex String values.
    assert_equal(
        infer("a,b,c,d\r\ninf,NaN,INF,1.e1\r\n"),
        "offset=9;a:float64;b:float64;c:string;d:string;",
    )


def test_oracle_regex_whitespace_sign_and_special_matrix() raises:
    # Expected values were obtained from the pinned Python Polars 1.44.2
    # fixture oracle. These are regex decisions before any numeric decoding.
    assert_equal(
        infer(
            "zero,negative,plus,leading,trailing,nan,upper,inf,NaN,decimal,exp,dot,bad\n"
            + "007,-3,+1, 1,1 ,nan,INF,inf,NaN,.5,1e2,2.,1.e1\n"
        ),
        "offset=74;zero:int64;negative:int64;plus:string;leading:string;"
        + "trailing:string;nan:string;upper:string;inf:float64;NaN:float64;"
        + "decimal:float64;exp:float64;dot:float64;bad:string;",
    )


def test_header_is_lossy_independent_of_body_encoding() raises:
    # infer_headers uses String::from_utf8_lossy even when the later decoder
    # will use strict utf8 checking for String body columns.
    var source = List[UInt8]()
    source.append(255)
    source.extend(String(",b\n1,2\n").as_bytes())
    assert_equal(infer_bytes(source^), "offset=4;�:int64;b:int64;")


def test_overrides_and_duplicate_collision_are_explicit() raises:
    var overrides = Dict[String, String]()
    overrides["v"] = "string"
    assert_equal(infer("v\n1\n", overrides=overrides), "offset=2;v:string;")
    assert_equal(
        infer("v\n1\n", overrides={"missing": "int64"}), "offset=2;v:int64;"
    )
    assert_equal(
        infer("v\n1\n", overrides={"v": "float32"}), "offset=2;v:float32;"
    )
    with assert_raises(contains="Unknown dtype in schema_overrides"):
        _ = infer("v\n1\n", overrides={"v": "decimal"})
    # This is the same collision rejected by infer_headers: a generated name
    # from the second a already appears as a physical header name.
    with assert_raises(contains="de-duplication"):
        _ = infer("a,a_duplicated_0,a\n1,2,3\n")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
