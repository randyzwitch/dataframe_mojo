"""Uncalibrated transport adapters cannot acquire direct ordered provenance."""
# contract-group: public.apple_parity
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    ListColumn,
    StructColumn,
    col,
    lit,
)
from dataframe.lazy import LazyFrame
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.gpu.calibration import estimate
from dataframe_metal.accelerator.gpu.runtime import _preflight


def _query(transport: Int) raises -> LazyFrame:
    var columns: List[Series] = [
        Series("x", Column[Int64]([128])),
        Series("base", Column[Int64]([2])),
        Series("exp", Column[Int64]([-1])),
    ]
    if transport == 1:
        var bytes: List[List[UInt8]] = [[255, 0, 128]]
        columns.append(Series.binary("unused", bytes))
    elif transport == 2:
        var fields: List[Series] = [Series("n", Column[Int64]([3]))]
        columns.append(Series("unused", StructColumn(fields^)))
    elif transport == 3:
        var child = Series("item", Column[Int64]([3]))
        columns.append(Series("unused", ListColumn([Int64(0), 1], child^)))
    return (
        DataFrame(columns^)
        .lazy()
        .select(
            (
                col("x").cast(DataType.INT8).cast(DataType.INT64)
                + col("base").pow(col("exp"))
            ).alias("checked")
        )
    )


def test_ordered_rows_transport_adapters_refuse_uncalibrated_families_without_device() raises:
    # Invalid devices are intentional: estimate and describe are pure metadata
    # operations, including when execution needs a source transport adapter.
    var runtime = MetalRuntime(device_id=999999)
    var control = _query(0)
    assert_equal(_preflight(control, ordered_source=True).ordered_mode, 2)
    var numeric = estimate(control)
    assert_true("ordered-errors-v1=2" in numeric.family)
    assert_equal(runtime.estimate_auto(control).family, numeric.family)
    assert_true(
        "ordered projection/batch/AST faults" in runtime.describe(control)
    )
    for transport in [1, 2, 3]:
        var query = _query(transport)
        # This is the dangerous pruning case: the unused transport field would
        # otherwise disappear before a numeric preflight enables token mode.
        assert_equal(_preflight(query, ordered_source=True).ordered_mode, 2)
        var reason = (
            "Mojo opaque staging has no analytical calibration" if transport
            == 1 else (
                "Mojo source Struct staging has no analytical calibration" if transport
                == 2 else "Mojo source List staging has no analytical calibration"
            )
        )
        with assert_raises(contains=reason):
            _ = estimate(query)
        with assert_raises(contains=reason):
            _ = runtime.estimate_auto(query)
        var description = runtime.describe(query)
        assert_true(
            (
                "opaque byte transport" if transport
                == 1 else (
                    "nullable source struct fields" if transport
                    == 2 else "resident source List len/get"
                )
            )
            in description
        )
        assert_true("ordered projection/batch/AST faults" not in description)
        assert_true("999999" not in description)
    # Ordinary unranked numeric families retain their existing admission.
    var plain = (
        DataFrame([Series("x", Column[Float32]([1, 2]))])
        .lazy()
        .select((col("x") * lit(Float32(2))).alias("value"))
    )
    var ordinary = estimate(plain)
    assert_true(";mojo-metal-v1;ops=" in ordinary.family)
    assert_true("ordered-errors-v1=" not in ordinary.family)
    assert_equal(runtime.estimate_auto(plain).family, ordinary.family)
