"""Source uploads omit derived slots, zero high words and clean error words."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, col, lit
from dataframe.accelerator.gpu.runtime import _preflight, _source_high_words
from tests.execution_modes.test_backend_metal import require_metal
from tests.support.coverage_compare import assert_frame


# requires: metal
def test_mojo_gpu_compact_source_upload_numeric_chunks_and_null_offsets() raises:
    var values = Series(
        "x",
        Column[Int64](
            [999, -1, Int64.MIN, Int64.MAX, 7, 0, 22, -3, 888],
            [True, True, True, True, False, True, True, True, True],
        ),
    ).slice(1, 7)
    var source = DataFrame(
        [
            Series._from_chunks(
                [values.slice(0, 2), values.slice(2, 0), values.slice(2, 5)]
            )
        ]
    )
    var query = source.lazy().select(col("x"))
    assert_true(not _source_high_words(_preflight(query)))
    var measured = query.profile(accelerator=require_metal())
    assert_frame(measured[0], source)
    assert_equal(measured[1].item(0, "upload_bytes").int64(), Int64(7 * 9))
    assert_true("fault_adjudication" not in measured[1].columns())


# requires: metal
def test_mojo_gpu_compact_source_upload_retains_decimal_high_words() raises:
    var source = DataFrame(
        [
            Series(
                "x",
                Column[Int128](
                    [(Int128(1) << 100) + 7, -(Int128(1) << 100), 3]
                ),
            ).with_dtype(DataType.decimal(38, 0)),
        ]
    )
    var query = source.lazy().select(col("x"))
    assert_true(_source_high_words(_preflight(query)))
    var measured = query.profile(accelerator=require_metal())
    assert_frame(measured[0], source)
    assert_equal(measured[1].item(0, "upload_bytes").int64(), Int64(3 * 17))


# requires: metal
def test_mojo_gpu_compact_source_upload_initializes_derived_slots() raises:
    var source = DataFrame(
        [
            Series(
                "x", Column[Float32]([1, -2, 0, 4], [True, True, False, True])
            ),
        ]
    )
    var query = (
        source.lazy()
        .with_columns(
            [
                (col("x") * lit(Float32(2))).alias("twice"),
                (col("x") + lit(Float32(1))).alias("plus_one"),
            ]
        )
        .select_exprs(
            [
                col("x"),
                col("twice"),
                col("plus_one"),
                (col("twice") + col("plus_one")).alias("combined"),
            ]
        )
    )
    var measured = query.profile(accelerator=require_metal())
    assert_frame(measured[0], query.collect(engine="cpu"))
    assert_true("fault_adjudication" not in measured[1].columns())
