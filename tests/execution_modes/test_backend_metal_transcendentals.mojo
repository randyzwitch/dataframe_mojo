"""GPU exponential/logarithm approximations retain domain and numeric contracts."""
from std.testing import assert_true, assert_equal, assert_raises
from std.math import isnan, isfinite
from std.memory import bitcast
from dataframe import (
    DataFrame,
    DataType,
    Series,
    Column,
    Expr,
    LazyFrame,
    col,
    lit,
    when,
)
from dataframe.metal import MetalRuntime, _capabilities, _Descriptors
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal import require_metal


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    var actual = query.collect(accelerator=runtime)
    var expected = query.collect(engine="cpu", streaming=False)
    assert_equal(actual.columns(), expected.columns())
    assert_equal(actual.height(), expected.height())
    for name in actual.columns():
        var dtype = expected.column(name).dtype()
        assert_equal(actual.column(name).dtype(), dtype)
        if not dtype.is_float():
            assert_true(actual.column(name).equals(expected.column(name)))
            continue
        var a = actual.column(name).cast(DataType.FLOAT64).float64()
        var b = expected.column(name).cast(DataType.FLOAT64).float64()
        for i in range(actual.height()):
            assert_equal(a.is_valid(i), b.is_valid(i))
            if not b.is_valid(i):
                continue
            var x = a._get(i)
            var y = b._get(i)
            if isnan(y):
                assert_true(isnan(x))
            elif not isfinite(y) or y == 0:
                if bitcast[DType.uint64](x) != bitcast[DType.uint64](y):
                    print(
                        "classification mismatch", dtype.name(), name, i, x, y
                    )
                    print(query.explain(optimize=False, streaming=False))
                    print(actual, expected)
                assert_equal(bitcast[DType.uint64](x), bitcast[DType.uint64](y))
            else:
                var relative = Float64(
                    2e-6
                ) if dtype == DataType.FLOAT32 else Float64(5e-12)
                var absolute = Float64(
                    1e-44
                ) if dtype == DataType.FLOAT32 else Float64(1e-323)
                if not isfinite(x) or abs(x - y) > relative * abs(y) + absolute:
                    print("transcendental mismatch", name, i, x, y)
                assert_true(
                    isfinite(x) and abs(x - y) <= relative * abs(y) + absolute
                )


def fixture() raises -> DataFrame:
    var values: List[Float64] = [
        -750,
        -745.1332191019411,
        -710,
        -2,
        -1,
        -0.0,
        0,
        1e-300,
        0.5,
        1,
        1.0000000000000002,
        2,
        10,
        709.782712893384,
        710,
        Float64("inf"),
        Float64("-inf"),
        Float64("nan"),
    ]
    values.append(bitcast[DType.float64](UInt64(1)))
    values.append(bitcast[DType.float64](UInt64(0x000FFFFFFFFFFFFF)))
    var valid = List[Bool](length=len(values), fill=True)
    valid[4] = False
    return DataFrame([Series("x", Column[Float64](values^, valid^))])


def test_transcendental_capability_metadata_types_and_no_device_preflight() raises:
    assert_true(not RowCapabilities().float_transcendentals)
    var frame = fixture()
    var caps = _capabilities()
    caps.float_transcendentals = False
    with assert_raises(contains="resident expression operation"):
        _ = lower_rows(frame.lazy().select(col("x").exp()), caps)
    for expression in [col("x").exp(), col("x").log()]:
        var plan = lower_rows(frame.lazy().select(expression), _capabilities())
        _ = _Descriptors(plan)
        assert_true(
            "unsupported"
            not in MetalRuntime(device_id=999999).describe(
                frame.lazy().select(expression)
            )
        )


def test_native_metal_transcendentals_float_widths_subnormals_extremes_nulls_and_empty_rows() raises:
    var runtime = require_metal()
    for dtype in List[DataType]([DataType.FLOAT32, DataType.FLOAT64]):
        var frame = DataFrame([fixture().column("x").cast(dtype)]).slice(1, 19)
        check(frame.lazy().select(col("x").exp()), runtime)
        check(frame.lazy().select(col("x").log()), runtime)
        check(frame.head(0).lazy().select(col("x").exp()), runtime)
        check(frame.head(0).lazy().select(col("x").log()), runtime)
    for dtype in List[DataType](
        [
            DataType.INT8,
            DataType.UINT8,
            DataType.INT16,
            DataType.UINT16,
            DataType.INT32,
            DataType.UINT32,
            DataType.INT64,
            DataType.UINT64,
        ]
    ):
        var frame = DataFrame(
            [
                Series(
                    "x", Column[Int64]([0, 1, 2, 10], [True, True, False, True])
                ).cast(dtype)
            ]
        )
        check(frame.lazy().select(col("x").exp()), runtime)
        check(frame.lazy().select(col("x").log()), runtime)
    var wide = DataFrame(
        [Series("x", Column[UInt64]([UInt64.MAX, 0, UInt64(1) << 63]))]
    )
    check(wide.lazy().select(col("x").log()), runtime)
    check(wide.lazy().select(col("x").exp()), runtime)
    var known = (
        DataFrame([Series("x", Column[Float64]([0, 1]))])
        .lazy()
        .select_exprs([col("x").exp().alias("e"), col("x").log().alias("l")])
        .collect(accelerator=runtime)
    )
    assert_equal(known.column("e").float64()._get(0), Float64(1))
    assert_equal(known.column("l").float64()._get(0), Float64("-inf"))
    assert_equal(known.column("l").float64()._get(1), Float64(0))


def test_native_metal_transcendentals_resident_consumers_windows_chunks_and_scalar_literals() raises:
    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [1, 2, 4, 8, 16, 32], [True, True, False, True, True, True]
                ),
            ),
            Series("k", Column[Int32]([2, 1, 2, 1, 2, 1])),
        ]
    )
    check(
        frame.lazy()
        .with_columns(col("x").log().alias("l"))
        .filter(col("l") > 0)
        .sort("l"),
        runtime,
    )
    check(frame.lazy().select(col("x").log().mean()), runtime)
    check(
        frame.lazy()
        .with_columns(col("x").log().alias("l"))
        .group_by("k")
        .agg(col("l").mean())
        .sort("k"),
        runtime,
    )
    check(
        frame.lazy().with_columns(col("x").log().rolling_mean(3).alias("l")),
        runtime,
    )
    check(
        frame.lazy().select(
            when(col("x") > 1).then(col("x").log()).otherwise(lit(Float64(0)))
        ),
        runtime,
    )
    var chunks = Series._from_chunks(
        [frame.column("x").slice(0, 2), frame.column("x").slice(2, 4)]
    )
    check(DataFrame([chunks^]).lazy().select(col("x").log().exp()), runtime)
    check(frame.head(0).lazy().select(lit(Float64(1)).log()), runtime)
    check(frame.lazy().select(lit(Float64(0)).exp()), runtime)


def test_native_metal_transcendentals_profiles_family_exact_budget_and_runtime_reuse() raises:
    var runtime = require_metal()
    var query = (
        fixture()
        .lazy()
        .select_exprs([col("x").exp().alias("e"), col("x").log().alias("l")])
    )
    var estimate = runtime.estimate_auto(query)
    assert_true(";transcendentals-v1:e1:l1" in estimate.family)
    assert_true(";metal-transcendentals-v1:e1:l1" in estimate.family)
    var narrow = (
        DataFrame([fixture().column("x").cast(DataType.FLOAT32)])
        .lazy()
        .select(col("x").log())
    )
    assert_true(
        ";metal-emulated-float64-v1" in runtime.estimate_auto(narrow).family
    )
    var exact = MetalRuntime(memory_limit_bytes=estimate.peak_bytes)
    var measured = query.profile(accelerator=exact)
    assert_equal(
        measured[1].item(0, "peak_requested_bytes").int64(),
        Int64(estimate.peak_bytes),
    )
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=estimate.peak_bytes - 1)
        )
    check(query, exact)
