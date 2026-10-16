"""CPU literal serialization witness; engine labels do not claim GPU execution.

Set DATAFRAME_BENCH_ANSWER_EXPORT_PREFIX to a fresh owned output prefix.
"""
from std.os import getenv
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, StringColumn, read_parquet
from dataframe.json_value import JsonValue
from metal_suite_answer_exports import export_answer, schema_receipt


def source() raises -> DataFrame:
    var texts: List[String] = [
        "https://example/" + String("λ") * 300,
        "",
        'quote"\nline',
    ]
    var valid: List[Bool] = [True, False, True]
    var integers: List[Int64] = [Int64.MAX, -1, 0]
    var integer_valid: List[Bool] = [True, True, False]
    var floats: List[Float32] = [1.25, 0, -0.5]
    var float_valid: List[Bool] = [True, False, True]
    var columns: List[Series] = [
        Series("URL\tname", StringColumn(texts^, valid^)),
        Series("key", Column[Int64](integers^, integer_valid^)),
        Series("f32", Column[Float32](floats^, float_valid^)),
    ]
    return DataFrame(columns^)


def main() raises:
    var prefix = getenv("DATAFRAME_BENCH_ANSWER_EXPORT_PREFIX")
    assert_true(Bool(prefix), "export probe requires an owned prefix")
    var frame = source()
    assert_equal(frame.height(), 3)
    assert_true(frame.item(1, "URL\tname").is_null())
    assert_equal(
        frame.item(0, "URL\tname").string(),
        "https://example/" + String("λ") * 300,
    )
    for engine in ["cpu", "metal", "cpu_repeat"]:
        export_answer(frame, engine, 0)
        var restored = read_parquet(prefix + "-" + engine + "-0.parquet")
        assert_true(restored.equals(frame))
        assert_true(restored.dtypes() == frame.dtypes())
        with open(prefix + "-" + engine + "-0.schema.json", "r") as file:
            var original_schema = file.read()
            assert_equal(original_schema, schema_receipt(frame, engine, 0))
            var parsed = JsonValue(original_schema)
            assert_equal(
                parsed.get("format").string(), "dataframe-answer-schema-v1"
            )
    print("PASS complete literal strings/nulls/Int64/Float32 export roundtrip")
