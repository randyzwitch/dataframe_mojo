"""Opaque Float64 follows row identities and preserves every valid IEEE word."""
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import DataFrame, DataType, Series, Column, Expr, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe_metal.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import (
    require_metal,
    assert_native_report,
)
from tests.support.metal_domain_checks import assert_domain
from tests.support.lifecycle_jobs import WorkerSetting


def word(row: Int) -> UInt64:
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        0x7FF8000000000042,
        0x7FF0000000000042,
        0x8000000000000001,
        0x7FEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0xDEADBEEF,
    ]
    return words[row]


def key(row: Int) -> Int32:
    var keys: List[Int32] = [2, 1, 2, 1, 3, 3, 2, 4, 4]
    return keys[row]


def payload(name: String, positions: List[Int]) raises -> Series:
    var values = List[Float64]()
    var bits = List[Bool]()
    for row in positions:
        values.append(
            bitcast[DType.float64](word(row) if row >= 0 else UInt64(0))
        )
        bits.append(row >= 0 and row != 8)
    return Series(name, Column[Float64](values^, bits^))


def frame(positions: List[Int]) raises -> DataFrame:
    var keys = List[Int32]()
    var ids = List[Int32]()
    for row in positions:
        keys.append(key(row))
        ids.append(Int32(row))
    return DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("id", Column[Int32](ids^)),
            payload("f", positions),
        ]
    )


def assert_words(actual: DataFrame, expected: DataFrame) raises:
    assert_domain(actual, expected)
    for name in expected.columns():
        if expected.column(name).dtype() != DataType.FLOAT64:
            continue
        for row in range(expected.height()):
            if expected.item(row, name).is_null():
                continue
            assert_equal(
                bitcast[DType.uint64](actual.item(row, name).float64()),
                bitcast[DType.uint64](expected.item(row, name).float64()),
            )


def check(
    plan: LazyFrame, expected: DataFrame, runtime: MetalRuntime, native: Bool
) raises:
    if native:
        var measured = plan.profile(accelerator=runtime)
        assert_words(measured[0], expected)
        assert_native_report(measured[1], expected.height() > 0)
    var setting = WorkerSetting(1)
    for optimize in [False, True]:
        for streaming in [False, True]:
            for batch in [1, 7]:
                assert_words(
                    plan.collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=batch,
                    ),
                    expected,
                )
    _ = setting^


def opaque_cases(runtime: MetalRuntime, native: Bool) raises:
    var positions: List[Int] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    var source = frame(positions)
    var fill_word = UInt64(0x8000000000000001)
    var fill = bitcast[DType.float64](fill_word)
    var filled = List[Float64]()
    var choices = List[Float64]()
    var choice_bits = List[Bool]()
    for row in positions:
        filled.append(
            bitcast[DType.float64](word(row) if row != 8 else fill_word)
        )
        choices.append(
            bitcast[DType.float64](word(row) if row % 2 == 0 else fill_word)
        )
        choice_bits.append(row != 8)
    check(
        source.lazy().select_exprs(
            [
                col("f"),
                col("f").is_null().alias("null"),
                col("f").is_not_null().alias("valid"),
                col("f").fill_null(lit(fill)).alias("filled"),
                when(col("id") % lit(Int32(2)) == 0)
                .then(col("f"))
                .otherwise(lit(fill))
                .alias("choice"),
            ]
        ),
        DataFrame(
            [
                payload("f", positions),
                Series(
                    "null",
                    Column[Bool](
                        [
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            True,
                        ]
                    ),
                ),
                Series(
                    "valid",
                    Column[Bool](
                        [True, True, True, True, True, True, True, True, False]
                    ),
                ),
                Series("filled", Column[Float64](filled^)),
                Series("choice", Column[Float64](choices^, choice_bits^)),
            ]
        ),
        runtime,
        native,
    )
    check(
        source.slice(0, 0).lazy().select(col("f")),
        DataFrame([payload("f", List[Int]())]),
        runtime,
        native,
    )
    var combined = source.lazy().concat(frame([3, 0, 8]).lazy()).slice(-5)
    check(combined, frame([7, 8, 3, 0, 8]), runtime, native)
    for keep in ["first", "last", "none"]:
        var chosen: List[Int] = (
            [0, 1, 4, 7] if keep
            == "first" else [3, 5, 6, 8] if keep
            == "last" else List[Int]()
        )
        check(
            source.lazy().unique(["k"], keep=keep, maintain_order=True),
            frame(chosen),
            runtime,
            native,
        )
    check(
        source.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                col("f").first().alias("first"),
                col("f").last().alias("last"),
                col("f").count().alias("n"),
            ]
        ),
        DataFrame(
            [
                Series("k", Column[Int32]([2, 1, 3, 4])),
                payload("first", [0, 1, 4, 7]),
                payload("last", [6, 3, 5, 8]),
                Series("n", Column[Int64]([3, 2, 2, 1])),
            ]
        ),
        runtime,
        native,
    )
    check(
        source.lazy().select_exprs(
            [
                col("f").first().alias("first"),
                col("f").last().alias("last"),
                col("f").count().alias("n"),
                col("f").null_count().alias("nulls"),
            ]
        ),
        DataFrame(
            [
                payload("first", [0]),
                payload("last", [8]),
                Series("n", Column[Int64]([8])),
                Series("nulls", Column[Int64]([1])),
            ]
        ),
        runtime,
        native,
    )
    var right = frame([1, 4, 8])
    for how in ["inner", "left", "right", "full", "semi", "anti", "cross"]:
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        if how == "right":
            left_rows = [1, 3, 4, 5, 7, 8]
            right_rows = [1, 1, 4, 4, 8, 8]
        else:
            for row in positions:
                var matched_row = -1
                for candidate in [1, 4, 8]:
                    if key(row) == key(candidate):
                        matched_row = candidate
                if how == "cross":
                    for candidate in [1, 4, 8]:
                        left_rows.append(row)
                        right_rows.append(candidate)
                elif (
                    (how == "semi" and matched_row >= 0)
                    or (how == "anti" and matched_row < 0)
                    or how == "left"
                    or how == "full"
                    or (how == "inner" and matched_row >= 0)
                ):
                    left_rows.append(row)
                    right_rows.append(matched_row)
        var expressions: List[Expr] = [col("f")]
        var expected: List[Series] = [payload("f", left_rows)]
        if how != "semi" and how != "anti":
            expressions.append(col("f_right"))
            expected.append(payload("f_right", right_rows))
        var query = source.lazy().join(
            right.lazy(), how="cross"
        ) if how == "cross" else source.lazy().join(right.lazy(), "k", how=how)
        check(
            query.select_exprs(expressions),
            DataFrame(expected^),
            runtime,
            native,
        )


# requires: metal
def test_native_metal_opaque_float64_row_graphs_copy_literal_null_padding_and_positional_ieee_words() raises:
    opaque_cases(require_metal(), True)


def verify_metal_opaque_cpu_models() raises:
    opaque_cases(MetalRuntime(), False)
    opaque_preflight()


def opaque_preflight() raises:
    var source = frame([0, 1, 2])
    var invalid = MetalRuntime(device_id=999999)
    for expression in [
        col("f").pow(lit(Float64(2))),
        col("f").var(),
        col("f").std(),
    ]:
        assert_true(
            "unsupported"
            not in invalid.describe(source.lazy().select(expression))
        )


# requires: metal
def test_native_metal_opaque_float64_computation_rejects_before_device_setup() raises:
    _ = require_metal()
    opaque_preflight()
