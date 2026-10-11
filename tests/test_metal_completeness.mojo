"""Metal operator coverage beyond the original resident numeric subset."""
from std.testing import TestSuite, assert_true, assert_equal, assert_raises
from dataframe import (
    DataFrame,
    Series,
    StringColumn,
    Column,
    Expr,
    col,
    lit,
    when,
)
from dataframe.metal import MetalRuntime, metal_installed
from dataframe.dtype import DataType
from dataframe.accelerator.dictionaries import lower_dictionary_operations


def text_frame() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "k",
                StringColumn(
                    List[String](
                        [
                            " Abé ",
                            "google",
                            "",
                            "é🙂é",
                            "null",
                            "google",
                            "-12",
                            "a_b%",
                        ]
                    ),
                    List[Bool](
                        [True, True, True, True, False, True, True, True]
                    ),
                ),
            ),
            Series("v", Column[Int16]([1, 2, 3, 4, 5, 6, 7, 8])),
        ]
    )


def test_dictionary_unary_operators_and_chains() raises:
    if not metal_installed():
        return
    var frame = text_frame()
    var expressions = List[Expr](
        [
            col("k").str().contains("google"),
            col("k").str().starts_with("é"),
            col("k").str().ends_with("é"),
            col("k").str().like("%_é%"),
            col("k").is_in([lit("google"), lit(""), lit("missing")]),
            col("k").str().len_chars(),
            col("k").str().len_bytes(),
            col("k").str().to_uppercase(),
            col("k").str().to_lowercase(),
            col("k").str().strip_chars(),
            col("k").str().strip_chars_start(" A"),
            col("k").str().strip_chars_end(" é"),
            col("k").str().replace("é", "🙂"),
            col("k").str().replace_all("é", "🙂"),
            col("k").str().slice(-3, 2),
            col("k").str().reverse(),
            col("k").str().pad_start(7, "é"),
            col("k").str().pad_end(7, "🙂"),
            col("k").str().zfill(7),
            col("k")
            .str()
            .strip_chars()
            .str()
            .to_uppercase()
            .str()
            .contains("GOOGLE"),
            when(col("v") > lit(Int16(3)))
            .then(col("k").str().reverse())
            .otherwise(col("k")),
        ]
    )
    var runtime = MetalRuntime()
    for expression in expressions:
        var query = frame.lazy().select(expression.alias("result"))
        assert_true(
            query.collect(accelerator=runtime).equals(
                query.collect(engine="cpu")
            )
        )
        var empty = frame.slice(0, 0).lazy().select(expression.alias("result"))
        assert_true(
            empty.collect(accelerator=runtime).equals(
                empty.collect(engine="cpu")
            )
        )


def test_dictionary_operations_filter_group_join_and_internal_names() raises:
    if not metal_installed():
        return
    var frame = text_frame()
    frame = frame.with_column(
        Series(
            "__dataframe_metal_dictionary_key_0",
            Column[Int16]([1, 2, 3, 4, 5, 6, 7, 8]),
        )
    )
    var runtime = MetalRuntime()
    var filtered = frame.lazy().filter(col("k").str().contains("google"))
    assert_true(
        filtered.collect(accelerator=runtime).equals(
            filtered.collect(engine="cpu")
        )
    )
    var transformed = frame.lazy().with_columns(
        [col("k").str().to_uppercase().alias("upper")]
    )
    assert_true(
        transformed.collect(accelerator=runtime).equals(
            transformed.collect(engine="cpu")
        )
    )
    var grouped = transformed.group_by("upper", maintain_order=True).agg(
        [col("v").sum().alias("s")]
    )
    assert_true(
        grouped.collect(accelerator=runtime).equals(
            grouped.collect(engine="cpu")
        )
    )
    var joined = transformed.join(
        frame.lazy(), left_on=["upper"], right_on=["k"], how="full"
    )
    assert_true(
        joined.collect(accelerator=runtime).equals(joined.collect(engine="cpu"))
    )
    var chained = transformed.filter(
        col("upper").str().starts_with("G")
    ).select(col("upper").str().len_chars().alias("n"))
    assert_true(
        chained.collect(accelerator=runtime).equals(
            chained.collect(engine="cpu")
        )
    )


def test_dictionary_operation_budget_profile_and_numeric_casts() raises:
    var frame = text_frame()
    var query = frame.lazy().filter(col("k").str().contains("google"))
    with assert_raises(contains="staging exceeds payload budget"):
        _ = lower_dictionary_operations(query, 1)
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var measured = query.profile(accelerator=runtime)
    assert_true(measured[0].equals(query.collect(engine="cpu")))
    assert_true(measured[1].item(0, "dictionary_operation_tables").int64() > 0)
    assert_true(
        measured[1].item(0, "dictionary_operation_staging_bytes").int64() > 0
    )
    var casted = frame.lazy().select_exprs(
        [col("k"), col("v").cast(DataType.INT32).alias("v")]
    )
    assert_true(
        casted.collect(accelerator=runtime).equals(casted.collect(engine="cpu"))
    )
    var numeric = DataFrame([Series("v", Column[Int16]([1, 2, 3]))])
    var literal = numeric.lazy().select_exprs(
        [col("v"), lit("é").alias("constant")]
    )
    assert_true(
        literal.collect(accelerator=runtime).equals(
            literal.collect(engine="cpu")
        )
    )
    # Recovery after a staging rejection uses the same public runtime normally.
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )


def test_native_integer_membership_all_widths_exact_words() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    comptime TYPES: Array[DType, 8] = [
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    ]
    comptime for d in range(len(TYPES)):
        comptime D = TYPES[d]
        var values: List[Scalar[D]] = [
            0,
            1,
            2,
            Scalar[D].MAX,
            Scalar[D].MIN,
            7,
            2,
        ]
        var frame = DataFrame(
            [
                Series(
                    "x",
                    Column[Scalar[D]](
                        values.copy(),
                        [True, False, True, True, True, True, True],
                    ),
                )
            ]
        )
        var targets = List[Expr](
            [lit(values[0]), lit(values[3]), lit(values[3])]
        )
        var query = frame.lazy().select(col("x").is_in(targets).alias("hit"))
        assert_true(
            query.collect(accelerator=runtime).equals(
                query.collect(engine="cpu")
            )
        )
        var filtered = frame.lazy().filter(col("x").is_in(targets)).sort("x")
        assert_true(
            filtered.collect(accelerator=runtime).equals(
                filtered.collect(engine="cpu")
            )
        )
        var empty = (
            frame.slice(0, 0)
            .lazy()
            .select(col("x").is_in(targets).alias("hit"))
        )
        assert_true(
            empty.collect(accelerator=runtime).equals(
                empty.collect(engine="cpu")
            )
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
