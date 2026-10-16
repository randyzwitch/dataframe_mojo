"""Binding is schema-dependent; execution errors retain their query boundary."""
# contract-group: planner.row_rewrites
# contract-group: planner.errors_schema
from std.os import getenv
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    LazyFrame,
    Series,
    col,
    lit,
    scan_csv,
    when,
)
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


def _error(
    plan: LazyFrame, schema_only: Bool, optimize: Bool, fragment: String
) raises:
    var failed = False
    try:
        if schema_only:
            _ = plan.collect_schema()
        else:
            _ = plan.collect(optimize=optimize, streaming=False, engine="cpu")
    except error:
        if String(error).find(fragment) < 0:
            raise error
        failed = True
    assert_true(failed, "query must reject the specified invalid contract")


def test_explicit_csv_schema_probes_never_decode_invalid_data_values() raises:
    var path = getenv("COVERAGE_TEST_TMP") + "/schema-with-invalid-data.csv"
    with open(path, "w") as file:
        file.write(
            "i,s,b\nnot-an-integer,not-an-integer,true\n9223372036854775808,17,false\n"
        )
    var schema = CsvSchema(
        [CsvField.int64("i"), CsvField.string("s"), CsvField.bool("b")]
    )
    var source = scan_csv(path, schema)
    var plan = source.select_exprs(
        [
            (col("i") + 1).alias("increment"),
            col("s").cast(DataType.INT64).alias("parsed"),
            when(col("b")).then(col("i")).otherwise(0).alias("chosen"),
        ]
    )
    assert_equal(
        plan.collect_schema(),
        ["increment: int64", "parsed: int64", "chosen: int64"],
    )
    assert_equal(
        source.group_by("b", maintain_order=True)
        .agg(col("i").sum().alias("total"))
        .collect_schema(),
        ["b: bool", "total: int64"],
    )
    for optimize in [False, True]:
        _error(plan, False, optimize, "i")
    # A schema query still validates field dependencies, rather than silently
    # accepting an invalid plan because its scan produces no rows.
    _error(source.select(col("missing")), True, True, "missing")


def test_binding_errors_are_identical_for_empty_and_populated_plans() raises:
    var source = DataFrame(
        [
            Series("i", Column[Int64]([1, 2])),
            Series("s", Column[String](["1", "2"])),
            Series("b", Column[Bool]([True, False])),
        ]
    )
    for empty in [False, True]:
        var frame = source.clear() if empty else source.copy()
        var plans: List[LazyFrame] = [
            frame.lazy().select(col("missing")),
            frame.lazy().filter(col("missing") > 0),
            frame.lazy().group_by("missing").agg(col("i").sum()),
            frame.lazy().join(
                frame.lazy(), left_on=["missing"], right_on=["i"]
            ),
            frame.lazy().select(
                when(lit(False)).then(col("missing")).otherwise(col("i"))
            ),
        ]
        for plan in plans:
            _error(plan, True, True, "missing")
            for optimize in [False, True]:
                _error(plan, False, optimize, "missing")
        var typed: List[LazyFrame] = [
            frame.lazy().select(col("i") & lit(True)),
            frame.lazy().filter(col("i")),
            frame.lazy().select(when(col("i")).then(1).otherwise(2)),
            frame.lazy().select_exprs(
                [col("i").alias("same"), col("s").alias("same")]
            ),
            frame.lazy().join(frame.lazy(), left_on=["i"], right_on=["s"]),
        ]
        for plan in typed:
            with assert_raises():
                _ = plan.collect_schema()
            for optimize in [False, True]:
                with assert_raises():
                    _ = plan.collect(
                        optimize=optimize, streaming=False, engine="cpu"
                    )


def test_optimizer_preserves_conditional_selected_rows_and_null_poison() raises:
    var source = DataFrame(
        [
            Series(
                "i",
                Column[Int64](
                    [Int64.MAX, 1, -1, Int64.MAX], [True, True, True, False]
                ),
            )
        ]
    )
    var expected = DataFrame([Series("v", Column[Int64]([Int64.MAX, 2, 0]))])
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        var safe = (
            frame.lazy()
            .select(
                when(col("i") == lit(Int64.MAX))
                .then(col("i"))
                .otherwise(col("i") + 1)
                .alias("v")
            )
            .filter(col("v") >= 0)
        )
        for optimize in [False, True]:
            assert_frame(
                safe.collect(optimize=optimize, streaming=False, engine="cpu"),
                expected,
            )
            var bad = frame.lazy().select(
                when(col("i") == lit(Int64.MAX))
                .then(col("i") + 1)
                .otherwise(col("i"))
                .alias("v")
            )
            assert_equal(bad.collect_schema(), ["v: int64"])
            _error(bad, False, optimize, "overflow")


def test_late_cast_errors_are_preserved_across_materialization_boundaries() raises:
    var source = DataFrame(
        [
            Series("i", Column[Int64]([3, 1, 2, 4])),
            Series("s", Column[String](["3", "1", "2", "invalid-last-row"])),
        ]
    )
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        var parsed = frame.lazy().with_columns(
            col("s").cast(DataType.INT64).alias("parsed")
        )
        var plans: List[LazyFrame] = [
            parsed.sort("i").select(["parsed"]),
            parsed.unique(["parsed"]).select(["parsed"]),
            parsed.group_by("i").agg(col("parsed").sum()),
        ]
        for plan in plans:
            _ = plan.collect_schema()
            for optimize in [False, True]:
                _error(plan, False, optimize, "invalid-last-row")


# expected-failure: Predicate pushdown suppresses a checked arithmetic
# expected-failure: overflow that occurs before the later filter in the
# expected-failure: unoptimized plan.
# expected-diagnostic: optimizer must preserve an overflow before a later
# expected-diagnostic: filter
def test_filter_pushdown_preserves_preceding_arithmetic_errors() raises:
    var source = DataFrame([Series("i", Column[Int64]([1, Int64.MAX]))])
    var plan = (
        source.lazy()
        .with_columns((col("i") + 1).alias("increment"))
        .filter(col("i") < lit(Int64.MAX))
    )
    _error(plan, False, False, "overflow")
    var rejected = False
    try:
        _ = plan.collect(optimize=True, streaming=False, engine="cpu")
    except error:
        if String(error).find("overflow") < 0:
            raise error
        rejected = True
    assert_true(
        rejected, "optimizer must preserve an overflow before a later filter"
    )


def test_relational_schema_preserves_key_coalescing_suffixes_and_reduction_types() raises:
    var left = DataFrame(
        [
            Series("left_key", Column[Int64]([1, 2])),
            Series("value", Column[Int32]([3, 4])),
        ]
    )
    var right = DataFrame(
        [
            Series("right_key", Column[Int64]([1, 3])),
            Series("value", Column[String](["x", "y"])),
        ]
    )
    for how in ["inner", "left", "right", "full"]:
        for coalesce in [False, True]:
            var plan = left.lazy().join(
                right.lazy(),
                left_on=["left_key"],
                right_on=["right_key"],
                how=how,
                coalesce=coalesce,
            )
            var expected: List[String] = ["left_key: int64", "value: int32"]
            if not coalesce:
                expected.append("right_key: int64")
            expected.append("value_right: string")
            assert_equal(plan.collect_schema(), expected)
            var actual = plan.collect(streaming=False, engine="cpu")
            assert_equal(actual.width(), len(expected))
            assert_equal(
                left.clear()
                .lazy()
                .join(
                    right.clear().lazy(),
                    left_on=["left_key"],
                    right_on=["right_key"],
                    how=how,
                    coalesce=coalesce,
                )
                .collect_schema(),
                expected,
            )
    var grouped = (
        left.lazy()
        .group_by("left_key", maintain_order=True)
        .agg(
            [
                col("value").sum().alias("sum"),
                col("value").mean().alias("mean"),
                col("value").count().alias("count"),
            ]
        )
    )
    assert_equal(
        grouped.collect_schema(),
        ["left_key: int64", "sum: int32", "mean: float64", "count: int64"],
    )
