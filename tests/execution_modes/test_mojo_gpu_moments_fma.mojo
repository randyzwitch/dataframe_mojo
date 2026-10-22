"""Public scalar moment queries retain CPU local FMA and NaN payload bits."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col, cov, corr
from dataframe.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout


def expressions() -> List[Expr]:
    return [
        col("x").var(0).alias("v"),
        col("x").std(0).alias("s"),
        cov(col("x"), col("y"), 0).alias("c"),
        corr(col("x"), col("y")).alias("r"),
    ]


def cancellation(nullable: Bool) raises -> DataFrame:
    var x: List[Float64] = [-1e16, 3, 2, 2, 1e16, -1e16]
    var y: List[Float64] = [-1e8, 1e8, 3, -1e8, -1e8, -1e8]
    var valid = List[Bool](length=6, fill=True)
    if nullable:
        x.append(Float64(0))
        y.append(Float64(0))
        valid.append(False)
    return DataFrame(
        [
            Series("x", Column[Float64](x^, valid.copy())),
            Series("y", Column[Float64](y^, valid)),
        ]
    )


def nan_payloads[D: DType](nullable: Bool) raises -> DataFrame:
    var first: Scalar[D]
    var second: Scalar[D]
    comptime if D == DType.float32:
        first = Scalar[D](bitcast[DType.float32](UInt32(0x7FC00123)))
        second = Scalar[D](bitcast[DType.float32](UInt32(0xFFC00456)))
    else:
        first = Scalar[D](bitcast[DType.float64](UInt64(0x7FF8000000000123)))
        second = Scalar[D](bitcast[DType.float64](UInt64(0xFFF8000000000456)))
    var x: List[Scalar[D]] = [first, second, 1, 2, 3, 4]
    var y: List[Scalar[D]] = [2, 4, 6, 8, 10, 12]
    var valid = List[Bool](length=6, fill=True)
    if nullable:
        x.append(Scalar[D](0))
        y.append(Scalar[D](0))
        valid.append(False)
    return DataFrame(
        [
            Series("x", Column[Scalar[D]](x^, valid.copy())),
            Series("y", Column[Scalar[D]](y^, valid)),
        ]
    )


def assert_words(frame: DataFrame, expected: List[UInt64]) raises:
    assert_equal(frame.height(), 1)
    var names: List[String] = ["v", "s", "c", "r"]
    for at in range(len(names)):
        assert_equal(frame.column(names[at]).dtype(), DataType.FLOAT64)
        assert_true(not frame.item(0, names[at]).is_null())
        assert_equal(
            bitcast[DType.uint64](frame.item(0, names[at]).float64()),
            expected[at],
        )


def cancellation_words() -> List[UInt64]:
    # Actual M1 CPU scalar materialization and independent local-FMA model.
    # Separate multiply/add changes variance to...a4 and standard to...850.
    # Covariance/correlation are serial one-worker states in this route.
    return [
        0x4682A039D5A192A3,
        0x433869E78FA4D84F,
        0x44B1A5825431A808,
        0x3FC452CCA7885B67,
    ]


def payload_words[D: DType]() -> List[UInt64]:
    # Actual M1 CPU23b scalar/eager/stream query contexts retain FIRST.
    # Standalone constant VarState tests have a different compiled context.
    var first = UInt64(0)
    comptime if D == DType.float32:
        first = UInt64(0x7FF8002460000000)
    else:
        first = UInt64(0x7FF8000000000123)
    return [first, first, first, first]


def cpu_nan_layouts[D: DType]() raises:
    for nullable in [False, True]:
        var source = nan_payloads[D](nullable)
        for frame in [
            source.copy(),
            offset_layout(source),
            chunk_layout(source),
        ]:
            var query = frame.lazy().select_exprs(expressions())
            assert_words(
                query.collect(engine="cpu", streaming=False, optimize=False),
                payload_words[D](),
            )


def test_cpu_public_moments_local_fma_and_nan_payload_literal_layouts() raises:
    for nullable in [False, True]:
        var source = cancellation(nullable)
        for frame in [
            source.copy(),
            offset_layout(source),
            chunk_layout(source),
        ]:
            assert_words(
                frame.lazy()
                .select_exprs(expressions())
                .collect(engine="cpu", streaming=False, optimize=False),
                cancellation_words(),
            )
    cpu_nan_layouts[DType.float32]()
    cpu_nan_layouts[DType.float64]()


# requires: mojo_gpu
def test_mojo_gpu_public_moments_local_fma_exact_layouts_and_nulls() raises:
    var runtime = MetalRuntime()
    for nullable in [False, True]:
        var source = cancellation(nullable)
        for frame in [
            source.copy(),
            offset_layout(source),
            chunk_layout(source),
        ]:
            var query = frame.lazy().select_exprs(expressions())
            assert_words(
                query.collect(engine="cpu", streaming=False, optimize=False),
                cancellation_words(),
            )
            assert_words(
                query.collect(
                    accelerator=runtime,
                    engine="accel",
                    streaming=False,
                    optimize=False,
                ),
                cancellation_words(),
            )


def gpu_nan_layouts[D: DType](runtime: MetalRuntime) raises:
    for nullable in [False, True]:
        var source = nan_payloads[D](nullable)
        for frame in [
            source.copy(),
            offset_layout(source),
            chunk_layout(source),
        ]:
            var query = frame.lazy().select_exprs(expressions())
            assert_words(
                query.collect(engine="cpu", streaming=False, optimize=False),
                payload_words[D](),
            )
            assert_words(
                query.collect(
                    accelerator=runtime,
                    engine="accel",
                    streaming=False,
                    optimize=False,
                ),
                payload_words[D](),
            )


# requires: mojo_gpu
def test_mojo_gpu_public_moments_nan_payload_first_input_layouts_and_nulls() raises:
    var runtime = MetalRuntime()
    gpu_nan_layouts[DType.float32](runtime)
    gpu_nan_layouts[DType.float64](runtime)
