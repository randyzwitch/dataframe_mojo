"""CUT and opaque byte normalization share labels, fault order and budgets."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_mojo_gpu_opaque_bytes import (
    _binary,
    _assert_bytes,
)
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_frame


def test_mojo_gpu_cut_and_opaque_transport_preserve_labels_errors_and_budgets() raises:
    var binary = _binary()
    var values = List[Float64]()
    for row in range(len(binary)):
        values.append(Float64(row - 7))
    var source = DataFrame(
        [binary.copy(), Series("x", Column[Float64](values^))]
    )
    for variant in range(3):
        var frame = (
            source.copy() if variant
            == 0 else offset_layout(source) if variant
            == 1 else chunk_layout(source)
        )
        for breaks in [False, True]:
            var query = frame.lazy().select_exprs(
                [
                    col("b").alias("bytes"),
                    col("x")
                    .cut([-3, 3], ["ff", "00", "é🙂"], include_breaks=breaks)
                    .alias("bin"),
                    (col("b") == lit("\x00")).alias("zero"),
                    lit("ff").alias("literal"),
                ]
            )
            var expected = query.collect(
                engine="cpu", optimize=False, streaming=False
            )
            var actual = query.collect(accelerator=MetalRuntime())
            _assert_bytes(actual.column("bytes"), binary)
            assert_frame(actual, expected)
            var measured = query.profile(accelerator=MetalRuntime())
            var peak = Int(
                measured[1].item(0, "query_peak_requested_bytes").int64()
            )
            assert_frame(
                query.collect(
                    accelerator=MetalRuntime(memory_limit_bytes=peak)
                ),
                expected,
            )
            with assert_raises(contains="budget"):
                _ = query.collect(
                    accelerator=MetalRuntime(memory_limit_bytes=peak - 1)
                )
    print(
        "PASS CUT/Binary layouts, label/byte collisions, Struct breaks and exact budgets"
    )
    for empty in [False, True]:
        var frame = source.clear() if empty else source.copy()
        var bad_cast = lit(Int64(128)).cast(DataType.INT8).cast(DataType.INT64)
        var bad_power = lit(Int64(2)).pow(lit(Int64(-1)))
        for cast_first in [False, True]:
            var scalar = (
                bad_cast + bad_power if cast_first else bad_power + bad_cast
            )
            var query = frame.lazy().select_exprs(
                [
                    scalar.alias("fault"),
                    col("b"),
                    col("x").cut([0], ["00", "ff"]).alias("bin"),
                ]
            )
            with assert_raises(
                contains="cast from int64 to int8 failed" if cast_first else "nonnegative exponent"
            ):
                _ = query.collect(engine="cpu", optimize=False, streaming=False)
            with assert_raises(
                contains="strict cast failed" if cast_first else "nonnegative exponent"
            ):
                _ = query.collect(accelerator=MetalRuntime())
    print(
        "PASS CUT ordered scalar errors with opaque sources, including empty inputs"
    )
