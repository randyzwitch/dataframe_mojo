"""DataType values: equality, names, predicates, and use across the API."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import (
    AnyValue,
    Column,
    CsvSchema,
    DataFrame,
    DataType,
    Series,
    col,
    lit,
)


def test_equality_names_and_parse() raises:
    var all: List[DataType] = [
        DataType.INT64,
        DataType.FLOAT64,
        DataType.BOOL,
        DataType.STRING,
    ]
    var names: List[String] = ["int64", "float64", "bool", "string"]
    var short: List[String] = ["i64", "f64", "bool", "str"]
    for i in range(len(all)):
        assert_equal(String(all[i]), names[i])
        assert_equal(all[i].name(), names[i])
        assert_equal(all[i].short_name(), short[i])
        assert_true(DataType.parse(names[i]) == all[i])
        assert_true(DataType.is_known(names[i]))
        for j in range(len(all)):
            assert_equal(all[i] == all[j], i == j)
            assert_equal(all[i] != all[j], i != j)
    assert_false(DataType.is_known("int32"))
    with assert_raises(contains="Unknown dtype: date"):
        _ = DataType.parse("date")


def test_predicates() raises:
    assert_true(DataType.INT64.is_numeric())
    assert_true(DataType.INT64.is_integer())
    assert_false(DataType.INT64.is_float())
    assert_true(DataType.INT64.is_signed())
    assert_true(DataType.FLOAT64.is_float())
    assert_true(DataType.FLOAT64.is_numeric())
    assert_false(DataType.BOOL.is_numeric())
    assert_false(DataType.STRING.is_signed())
    assert_equal(DataType.INT64.bit_width(), 64)
    assert_equal(DataType.FLOAT64.bit_width(), 64)
    assert_equal(DataType.BOOL.bit_width(), 1)
    assert_equal(DataType.STRING.bit_width(), 0)


def test_used_across_the_api() raises:
    var frame = DataFrame(
        [
            Series("i", Column[Int64]([1])),
            Series("s", Column[String](["a"])),
        ]
    )
    assert_equal(frame.dtypes(), [DataType.INT64, DataType.STRING])
    assert_true(frame.schema()[1].dtype == DataType.STRING)
    assert_true(frame.column("i").dtype() == DataType.INT64)
    assert_true(frame.item(0, "s").dtype() == DataType.STRING)
    assert_true(AnyValue.null(DataType.BOOL).dtype() == DataType.BOOL)
    assert_true(AnyValue.null("float64") == AnyValue.null(DataType.FLOAT64))
    assert_true(
        frame.column("i").cast(DataType.FLOAT64).dtype() == DataType.FLOAT64
    )
    assert_true(
        Series.full_null("n", DataType.BOOL, 2).dtype() == DataType.BOOL
    )
    assert_true(CsvSchema.of(frame).field(0).dtype == DataType.INT64)
    assert_true(
        frame.select(col("i") / lit(Int64(2))).dtypes()[0] == DataType.FLOAT64
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
