"""Grouped List siblings use the CPU fused local moment update."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.lifecycle_jobs import WorkerSetting


# requires: metal
def test_mojo_gpu_grouped_implode_var_std_siblings_local_fma_literal() raises:
    var values: List[Float64] = [-1e16, 3, 2, 2, 1e16, -1e16]
    var source = DataFrame(
        [
            Series("k", Column[Int64]([0, 0, 0, 0, 0, 0])),
            Series("x", Column[Float64](values.copy())),
        ]
    )
    for frame in [source.copy(), offset_layout(source), chunk_layout(source)]:
        var query = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("x").implode().alias("items"),
                    col("x").var(0).alias("variance"),
                    col("x").std(0).alias("standard"),
                ]
            )
        )
        for gpu in [False, True]:
            var actual = query.collect(
                accelerator=MetalRuntime(), streaming=False, optimize=False
            ) if gpu else query.collect(
                engine="cpu", streaming=False, optimize=False
            )
            assert_equal(actual.height(), 1)
            assert_equal(actual.item(0, "k").int64(), 0)
            assert_equal(actual.column("variance").dtype(), DataType.FLOAT64)
            assert_equal(actual.column("standard").dtype(), DataType.FLOAT64)
            assert_true(not actual.item(0, "variance").is_null())
            assert_true(not actual.item(0, "standard").is_null())
            # PR731 actual-M1 CPU anchors. Separate m2 multiply/add returns
            # variance ...92a4 and standard ...d850 for these same values.
            assert_equal(
                bitcast[DType.uint64](actual.item(0, "variance").float64()),
                UInt64(0x4682A039D5A192A3),
            )
            assert_equal(
                bitcast[DType.uint64](actual.item(0, "standard").float64()),
                UInt64(0x433869E78FA4D84F),
            )
            var items = actual.column("items").list_column()
            assert_equal(items._offsets[], [Int64(0), Int64(6)])
            assert_true(
                items.child().equals(
                    Series("items", Column[Float64](values.copy()))
                )
            )


def _higher_moment_source(
    fixture_index: Int, nullable: Bool
) raises -> DataFrame:
    var values: List[Float64] = [-1e16, 3, 2, 2, 1e16, -1e16]
    if fixture_index != 0:
        var positive = bitcast[DType.float64](UInt64(0x7FF8000000000123))
        var negative = bitcast[DType.float64](UInt64(0xFFF8000000000456))
        values = [
            Float64(1),
            Float64(2),
            positive if fixture_index == 1 else negative,
            negative if fixture_index == 1 else positive,
            Float64(3),
            Float64(4),
        ]
    var valid = List[Bool](length=len(values), fill=True)
    if nullable:
        values.append(Float64(99))
        valid.append(False)
    var columns = List[Series]()
    columns.append(
        Series("k", Column[Int32](List[Int32](length=len(values), fill=0)))
    )
    columns.append(Series("x", Column[Float64](values^, valid^)))
    return DataFrame(columns^)


def _higher_moment_expressions() -> List[Expr]:
    return [
        col("x").implode().alias("items"),
        col("x").skew(True).alias("s_b"),
        col("x").skew(False).alias("s_u"),
        col("x").kurtosis(True, True).alias("k_f_b"),
        col("x").kurtosis(False, True).alias("k_p_b"),
        col("x").kurtosis(True, False).alias("k_f_u"),
        col("x").kurtosis(False, False).alias("k_p_u"),
    ]


def _higher_moment_words(fixture_index: Int) -> List[UInt64]:
    # Actual M1 CPU words for these grouped List siblings, including nullable,
    # offset and chunked layouts. Both engines are checked against literals.
    if fixture_index == 0:
        return [
            0x3FCD37E9ADF37787,
            0x3FD401256E8D130A,
            0xBFEC9145FAAF6420,
            0x4000DBAE815426F8,
            0xBFBA930B83FC7595,
            0x40072B67A3E01C53,
        ]
    return List[UInt64](
        length=6,
        fill=UInt64(0x7FF8000000000123) if fixture_index
        == 1 else UInt64(0xFFF8000000000456),
    )


# requires: metal
def test_mojo_gpu_grouped_implode_higher_moment_siblings_literal_layouts_and_nan_payloads() raises:
    var setting = WorkerSetting(1)
    var names: List[String] = ["s_b", "s_u", "k_f_b", "k_p_b", "k_f_u", "k_p_u"]
    for fixture_index in range(3):
        var expected = _higher_moment_words(fixture_index)
        for layout in range(4):
            var raw = _higher_moment_source(fixture_index, layout != 0)
            var frame = (
                raw.copy() if layout
                < 2 else offset_layout(raw) if layout
                == 2 else chunk_layout(raw)
            )
            var query = (
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg(_higher_moment_expressions())
            )
            for gpu in [False, True]:
                var actual = query.collect(
                    accelerator=MetalRuntime(),
                    engine="accel",
                    streaming=False,
                    optimize=False,
                ) if gpu else query.collect(
                    engine="cpu", streaming=False, optimize=False
                )
                assert_equal(actual.height(), 1)
                assert_equal(actual.width(), 8)
                assert_equal(actual.item(0, "k").int32(), Int32(0))
                var items = actual.column("items").list_column()
                assert_equal(
                    items._offsets[], [Int64(0), Int64(frame.height())]
                )
                for at in range(len(names)):
                    assert_equal(
                        actual.column(names[at]).dtype(), DataType.FLOAT64
                    )
                    assert_true(not actual.item(0, names[at]).is_null())
                    assert_equal(
                        bitcast[DType.uint64](
                            actual.item(0, names[at]).float64()
                        ),
                        expected[at],
                    )
    _ = setting^
