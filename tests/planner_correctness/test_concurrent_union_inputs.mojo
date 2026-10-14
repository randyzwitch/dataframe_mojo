"""A union's inputs run at once, each from its own copy of the plan: the
stacked result is each input's own result in input order, nested unions
included, and an error in either input reaches the caller."""
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, DataType, LazyFrame, Series, col, lit
from dataframe.frame import concat
from tests.support.coverage_compare import assert_frame

comptime ROWS = 200_000


def source() raises -> DataFrame:
    var keys = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    var text = List[String]()
    for row in range(ROWS):
        keys.append(Int64((row * 7919) % 997))
        values.append(Int64(row % 1_000) - 500)
        valid.append(row % 23 != 5)
        text.append("r" + String(row % 50))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^, valid^)),
            Series("s", Column[String](text^)),
        ]
    )


def part(frame: DataFrame, shift: Int) raises -> LazyFrame:
    """One grouped input, distinct per `shift`."""
    return (
        frame.lazy()
        .filter((col("k") % lit(Int64(3))) == lit(Int64(shift)))
        .group_by("k")
        .agg([col("v").sum().alias("total"), col("v").count().alias("n")])
        .sort("k")
    )


def test_union_inputs_stack_in_order() raises:
    var frame = source()
    var inputs = List[LazyFrame]()
    for shift in range(3):
        inputs.append(part(frame, shift))
    var union = inputs[0].concat(inputs[1]).concat(inputs[2])
    var nested = inputs[0].concat(inputs[1].concat(inputs[2]))
    var expected = concat(
        [inputs[0].collect(), inputs[1].collect(), inputs[2].collect()]
    )
    assert_frame(union.collect(), expected)
    assert_frame(nested.collect(), expected)
    assert_equal(union.collect().height(), 997)


def test_an_error_in_either_input_reaches_the_caller() raises:
    var frame = source()
    var good = part(frame, 0)
    # Strings that are not integers fail a strict cast when it runs.
    var bad = frame.lazy().select_exprs(
        [col("s").cast(DataType.INT64).alias("k"), col("v").alias("total")]
    )
    with assert_raises():
        _ = good.select_exprs([col("k"), col("total")]).concat(bad).collect()
    with assert_raises():
        _ = bad.concat(good.select_exprs([col("k"), col("total")])).collect()
