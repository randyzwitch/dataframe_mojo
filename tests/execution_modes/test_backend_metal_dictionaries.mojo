"""Native ordered dictionaries must preserve logical text and source codes."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    DataType,
    Series,
    Column,
    StringColumn,
    Expr,
    col,
    lit,
    when,
)
from dataframe.categorical import encode
from dataframe_metal.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import require_metal
from tests.support.metal_domain_checks import check_domain
from tests.support.coverage_compare import assert_frame


def text(
    values: List[String], bits: List[Bool], name: String = "s"
) raises -> Series:
    return Series(name, StringColumn(values.copy(), bits.copy()))


def dictionary_operation_cases(runtime: MetalRuntime, native: Bool) raises:
    var bits: List[Bool] = [True, True, True, False, True, True]
    var raw: List[String] = [" Abé ", "é🙂é", "", "poison", "google", "-12"]
    var source = DataFrame(
        [
            text(raw, bits),
            Series("id", Column[Int16]([0, 1, 2, 3, 4, 5])),
            Series(
                "__dataframe_metal_dictionary_key_0",
                Column[Int16]([10, 11, 12, 13, 14, 15]),
            ),
        ]
    )
    var expressions: List[Expr] = [
        col("s").str().to_uppercase().alias("upper"),
        col("s").str().to_lowercase().alias("lower"),
        col("s").str().strip_chars().alias("strip"),
        col("s").str().strip_chars_start(" A").alias("left"),
        col("s").str().strip_chars_end(" é").alias("right"),
        col("s").str().replace("é", "X").alias("replace"),
        col("s").str().replace_all("é", "X").alias("replace_all"),
        col("s").str().slice(-3, 2).alias("slice"),
        col("s").str().reverse().alias("reverse"),
        col("s").str().pad_start(6, "é").alias("pad_left"),
        col("s").str().pad_end(6, "🙂").alias("pad_right"),
        col("s").str().zfill(6).alias("zfill"),
        col("s").str().len_chars().alias("chars"),
        col("s").str().len_bytes().alias("bytes"),
        col("s").str().contains("é").alias("contains"),
        col("s").str().starts_with("é").alias("starts"),
        col("s").str().ends_with("é").alias("ends"),
        col("s").str().like("%_é%").alias("like"),
        col("s").is_in([lit("google"), lit(""), lit("google")]).alias("member"),
        col("s")
        .str()
        .strip_chars()
        .str()
        .to_uppercase()
        .str()
        .contains("GOOGLE")
        .alias("chain"),
    ]
    var expected = DataFrame(
        [
            text([" ABÉ ", "É🙂É", "", "", "GOOGLE", "-12"], bits, "upper"),
            text([" abé ", "é🙂é", "", "", "google", "-12"], bits, "lower"),
            text(["Abé", "é🙂é", "", "", "google", "-12"], bits, "strip"),
            text(["bé ", "é🙂é", "", "", "google", "-12"], bits, "left"),
            text([" Ab", "é🙂", "", "", "google", "-12"], bits, "right"),
            text([" AbX ", "X🙂é", "", "", "google", "-12"], bits, "replace"),
            text(
                [" AbX ", "X🙂X", "", "", "google", "-12"], bits, "replace_all"
            ),
            text(["bé", "é🙂", "", "", "gl", "-1"], bits, "slice"),
            text([" ébA ", "é🙂é", "", "", "elgoog", "21-"], bits, "reverse"),
            text(
                ["é Abé ", "éééé🙂é", "éééééé", "", "google", "ééé-12"],
                bits,
                "pad_left",
            ),
            text(
                [" Abé 🙂", "é🙂é🙂🙂🙂", "🙂🙂🙂🙂🙂🙂", "", "google", "-12🙂🙂🙂"],
                bits,
                "pad_right",
            ),
            text(
                ["0 Abé ", "000é🙂é", "000000", "", "google", "-00012"],
                bits,
                "zfill",
            ),
            Series("chars", Column[Int64]([5, 3, 0, 0, 6, 3], bits)),
            Series("bytes", Column[Int64]([6, 8, 0, 0, 6, 3], bits)),
            Series(
                "contains",
                Column[Bool]([True, True, False, False, False, False], bits),
            ),
            Series(
                "starts",
                Column[Bool]([False, True, False, False, False, False], bits),
            ),
            Series(
                "ends",
                Column[Bool]([False, True, False, False, False, False], bits),
            ),
            Series(
                "like",
                Column[Bool]([True, True, False, False, False, False], bits),
            ),
            Series(
                "member",
                Column[Bool]([False, False, True, False, True, False], bits),
            ),
            Series(
                "chain",
                Column[Bool]([False, False, False, False, True, False], bits),
            ),
        ]
    )
    check_domain(
        source.lazy().select_exprs(expressions),
        expected,
        runtime,
        native,
        dictionary=True,
    )
    check_domain(
        source.slice(0, 0).lazy().select_exprs(expressions),
        expected.slice(0, 0),
        runtime,
        native,
        submitted=False,
    )
    var chained = (
        source.lazy()
        .with_columns(col("s").str().to_uppercase().alias("upper"))
        .filter(col("upper").str().contains("É"))
        .select_exprs([col("id"), col("upper").str().len_bytes().alias("n")])
    )
    check_domain(
        chained,
        DataFrame(
            [
                Series("id", Column[Int16]([0, 1])),
                Series("n", Column[Int64]([6, 8])),
            ]
        ),
        runtime,
        native,
        resident=True,
        dictionary=True,
    )
    if native:
        var measured = chained.profile(accelerator=runtime)
        var tables = Int64(0)
        var bytes = Int64(0)
        for row in range(measured[1].height()):
            tables += (
                measured[1].item(row, "dictionary_operation_tables").int64()
            )
            bytes += (
                measured[1]
                .item(row, "dictionary_operation_staging_bytes")
                .int64()
            )
        assert_true(tables > 0)
        assert_true(bytes > 0)
        with assert_raises(contains="staging exceeds payload budget"):
            _ = chained.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
        check_domain(
            chained,
            DataFrame(
                [
                    Series("id", Column[Int16]([0, 1])),
                    Series("n", Column[Int64]([6, 8])),
                ]
            ),
            runtime,
            native,
            dictionary=True,
        )


# requires: metal
def test_native_metal_dictionary_unary_lookup_tables_preserve_utf8_nulls_and_report_host_staging() raises:
    dictionary_operation_cases(require_metal(), True)


def categorical_cases(runtime: MetalRuntime, native: Bool) raises:
    var left_key = encode(
        text(
            ["z", "a", "unused", "z", "left", "poison"],
            [True, True, True, True, True, False],
            "k",
        )
    )
    var right_key = encode(
        text(["a", "z", "right", "poison"], [True, True, True, False], "k")
    )
    assert_true(
        left_key.dtype() != right_key.dtype(),
        "same values must have independently assigned codes",
    )
    var left = DataFrame(
        [left_key.copy(), Series("a", Column[Int32]([0, 1, 2, 3, 4, 5]))]
    ).slice(0, 6)
    var right = DataFrame(
        [right_key.copy(), Series("b", Column[Int32]([10, 11, 12, 13]))]
    )
    var styles: List[String] = [
        "inner",
        "left",
        "right",
        "full",
        "semi",
        "anti",
        "cross",
    ]
    var left_rows: List[List[Int]] = [
        [0, 1, 3],
        [0, 1, 2, 3, 4, 5],
        [1, 0, 3, -1, -1],
        [0, 1, 2, 3, 4, 5, -1, -1],
        [0, 1, 3],
        [2, 4, 5],
        [],
    ]
    var right_rows: List[List[Int]] = [
        [1, 0, 1],
        [1, 0, -1, 1, -1, -1],
        [0, 1, 1, 2, 3],
        [1, 0, -1, 1, -1, -1, 2, 3],
        [],
        [],
        [],
    ]
    for style in range(len(styles)):
        var how = styles[style]
        var li = left_rows[style].copy()
        var ri = right_rows[style].copy()
        if how == "cross":
            for l in range(6):
                for r in range(4):
                    li.append(l)
                    ri.append(r)
        var left_values = List[Int32]()
        var left_valid = List[Bool]()
        var right_values = List[Int32]()
        var right_valid = List[Bool]()
        for row in range(len(li)):
            left_values.append(Int32(max(0, li[row])))
            left_valid.append(li[row] >= 0)
            if how != "semi" and how != "anti":
                right_values.append(Int32(10 + max(0, ri[row])))
                right_valid.append(ri[row] >= 0)
        var names: List[Expr] = [col("a")]
        var columns: List[Series] = [
            Series("a", Column[Int32](left_values^, left_valid^))
        ]
        if how != "semi" and how != "anti":
            names.append(col("b"))
            columns.append(
                Series("b", Column[Int32](right_values^, right_valid^))
            )
        var query = left.lazy().join(
            right.lazy(), "k", how=how
        ) if how != "cross" else left.lazy().join(right.lazy(), how="cross")
        check_domain(
            query.select_exprs(names),
            DataFrame(columns^),
            runtime,
            native,
            resident=True,
            dictionary=True,
        )
        if native:
            var result = query.collect(accelerator=runtime)
            var cpu = query.collect(engine="cpu")
            # Full typed values plus the output's original dictionary identity.
            assert_equal(result.column("k").dtype(), cpu.column("k").dtype())
            assert_true(result.equals(cpu))
            var literal_left: List[String] = [
                "z",
                "a",
                "unused",
                "z",
                "left",
                "",
            ]
            var literal_right: List[String] = ["a", "z", "right", ""]
            for row in range(len(li)):
                var from_right = how == "right" or li[row] < 0
                var index = ri[row] if from_right else li[row]
                var absent = index == (3 if from_right else 5)
                assert_equal(result.item(row, "k").is_null(), absent)
                if not absent:
                    assert_equal(
                        result.item(row, "k").string(),
                        literal_right[index] if from_right else literal_left[
                            index
                        ],
                    )
                if how == "cross":
                    assert_equal(
                        result.item(row, "k_right").is_null(), ri[row] == 3
                    )
                    if ri[row] != 3:
                        assert_equal(
                            result.item(row, "k_right").string(),
                            literal_right[ri[row]],
                        )
    # Several dictionary members become unused by this slice; projection must
    # retain the source dictionary identity rather than re-encoding two rows.
    var projected = left.slice(0, 2).lazy().sort("k")
    if native:
        var out = projected.collect(accelerator=runtime)
        assert_equal(out.column("k").dtype(), left_key.dtype())
        assert_equal(out.item(0, "a").int32(), Int32(1))
        assert_equal(out.item(1, "a").int32(), Int32(0))
        assert_equal(out.item(0, "k").string(), "a")
        assert_equal(out.item(1, "k").string(), "z")
        assert_true(out.equals(projected.collect(engine="cpu")))


# requires: metal
def test_native_metal_categorical_join_styles_reconcile_independent_codes_and_restore_dictionary_identity() raises:
    categorical_cases(require_metal(), True)


def dictionary_distinct_cases(runtime: MetalRuntime, native: Bool) raises:
    var source = DataFrame(
        [
            text(
                ["é", "a", "é", "a", "poison", ""],
                [True, True, True, True, False, True],
            ),
            Series("g", Column[Int32]([1, 1, 1, 2, 2, 2])),
        ]
    )
    check_domain(
        source.lazy().select(col("s").n_unique().alias("n")),
        DataFrame([Series("n", Column[Int64]([4]))]),
        runtime,
        native,
        dictionary=True,
    )
    check_domain(
        source.lazy()
        .group_by("g", maintain_order=True)
        .agg(col("s").n_unique().alias("n")),
        DataFrame(
            [
                Series("g", Column[Int32]([1, 2])),
                Series("n", Column[Int64]([2, 3])),
            ]
        ),
        runtime,
        native,
        dictionary=True,
    )
    check_domain(
        source.slice(0, 0).lazy().select(col("s").n_unique().alias("n")),
        DataFrame([Series("n", Column[Int64]([0]))]),
        runtime,
        native,
        submitted=False,
    )
    check_domain(
        source.lazy()
        .filter(col("g") == lit(Int32(1)))
        .select((col("s").str().to_uppercase().n_unique() + 1).alias("n")),
        DataFrame([Series("n", Column[Int64]([3]))]),
        runtime,
        native,
        resident=True,
        dictionary=True,
    )


# requires: metal
def test_native_metal_dictionary_distinct_counts_include_nulls_and_feed_aggregate_compositions() raises:
    dictionary_distinct_cases(require_metal(), True)


def verify_metal_dictionary_cpu_models() raises:
    var runtime = MetalRuntime()
    dictionary_operation_cases(runtime, False)
    categorical_cases(runtime, False)
    dictionary_distinct_cases(runtime, False)
