"""Public as-of queries keep CPU cursor rules on resident Mojo GPU tables."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import AnyValue, Column, DataFrame, DataType, Series, col
from dataframe.lazy import LazyFrame
from dataframe.dtype import CategoricalDictionary
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.gpu.runtime import capabilities
from dataframe_metal.accelerator.joins import lower_join_dag
from tests.support.asof_reference import assert_asof_rows
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


def _checked(
    query: LazyFrame,
    runtime: MetalRuntime,
    anchor: String,
    cpu_anchor: String = "",
) raises:
    # Preserve CPU error categories and ordering, not just GPU rejection.
    with assert_raises(contains=cpu_anchor if cpu_anchor else anchor):
        _ = query.collect(engine="cpu", streaming=False)
    with assert_raises(contains=anchor):
        _ = query.collect(accelerator=runtime, engine="accel")


def _directions[D: DType](runtime: MetalRuntime) raises:
    var l = List[Scalar[D]]()
    var r = List[Scalar[D]]()
    for value in [1, 2, 3, 99]:
        l.append(Scalar[D](value))
    for value in [0, 2, 2, 99, 4]:
        r.append(Scalar[D](value))
    var left = offset_layout(
        DataFrame(
            [
                Series("k", Column[Scalar[D]](l^, [True, True, True, False])),
                Series("p", Column[Int32]([10, 11, 12, 13])),
            ]
        )
    )
    var right = chunk_layout(
        DataFrame(
            [
                Series(
                    "k", Column[Scalar[D]](r^, [True, True, True, False, True])
                ),
                Series("p", Column[Int32]([20, 21, 22, 23, 24])),
            ]
        )
    )
    # Nearest deliberately preserves CPU's null-position duplicate run rule.
    var anchors: List[List[Int]] = [
        [0, 0, 2, -1],
        [0, 2, 2, -1],
        [1, 4, 4, -1],
        [1, 1, 4, -1],
        [2, 4, 4, -1],
        [2, 4, 4, -1],
    ]
    var at = 0
    for strategy in ["backward", "forward", "nearest"]:
        for exact in [False, True]:
            var query = left.lazy().join_asof(
                right.lazy(),
                on="k",
                strategy=strategy,
                allow_exact_matches=exact,
            )
            var actual = query.collect(accelerator=runtime, engine="accel")
            assert_asof_rows(
                actual, left, right, anchors[at], ["k", "p", "p_right"], ["p"]
            )
            assert_frame(actual, query.collect(engine="cpu", streaming=False))
            assert_frame(
                actual.head(2), query.head(2).collect(accelerator=runtime)
            )
            at += 1


# requires: metal
def test_mojo_gpu_asof_public_directions_nullable_chunks_duplicate_runs() raises:
    var runtime = MetalRuntime()
    comptime types = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float32,
        DType.float64,
    )
    comptime for index in range(len(types)):
        _directions[types[index]](runtime)


# requires: metal
def test_mojo_gpu_asof_public_grouped_dictionary_ieee_keys_and_wide_payloads() raises:
    var runtime = MetalRuntime()
    var lf = List[Float64]()
    var rf = List[Float64]()
    var wide = List[Int128]()
    for row in range(8):
        lf.append(
            bitcast[DType.float64](UInt64(0x7FF8000000000001)) if row % 2
            == 0 else bitcast[DType.float64](UInt64(0x8000000000000000))
        )
        rf.append(
            bitcast[DType.float64](UInt64(0xFFF8000000000002)) if row % 2
            == 0 else Float64(0)
        )
        wide.append((Int128(1) << 100) + Int128(row))
    var group = Series(
        "g",
        Column[String](
            ["b", "a", "b", "a", "b", "a", "b", "a"],
            [True, True, True, True, True, True, False, False],
        ),
    )
    for categorical in [False, True]:
        var lg = group.copy()
        var rg = group.copy()
        if categorical:
            var ld = CategoricalDictionary()
            ld.append("b")
            ld.append("a")
            var rd = CategoricalDictionary()
            rd.append("a")
            rd.append("b")
            lg = lg.cast(DataType.categorical(ld^))
            rg = rg.cast(DataType.categorical(rd^))
        var left = offset_layout(
            DataFrame(
                [
                    Series("k", Column[Int64]([1, 1, 2, 2, 3, 3, 4, 4])),
                    lg^,
                    Series("f", Column[Float64](lf.copy())),
                ]
            )
        )
        var right = chunk_layout(
            DataFrame(
                [
                    Series("k", Column[Int64]([0, 0, 2, 2, 2, 2, 4, 4])),
                    rg^,
                    Series("f", Column[Float64](rf.copy())),
                    Series("p", Column[Int128](wide.copy())).with_dtype(
                        DataType.decimal(38, 0)
                    ),
                ]
            )
        )
        var query = left.lazy().join_asof(
            right.lazy(), on="k", by=["g", "f"], strategy="nearest"
        )
        var actual = query.collect(accelerator=runtime)
        assert_asof_rows(
            actual,
            left,
            right,
            [4, 5, 4, 5, 4, 5, -1, -1],
            ["k", "g", "f", "p"],
            ["p"],
        )
        assert_frame(actual, query.collect(engine="cpu", streaming=False))


# requires: metal
def test_mojo_gpu_asof_public_complete_sortedness_producer_errors_and_recovery() raises:
    var runtime = MetalRuntime()
    var left = DataFrame(
        [
            Series("k", Column[Int64]([0, 2, 3])),
            Series("p", Column[Int64]([1, 2, 3])),
        ]
    )
    var unsorted = DataFrame(
        [
            Series("k", Column[Int64]([0, 3, 2])),
            Series("p", Column[Int64]([10, 11, 12])),
        ]
    )
    for strategy in ["backward", "forward", "nearest"]:
        _checked(
            left.head(0)
            .lazy()
            .join_asof(unsorted.lazy(), on="k", strategy=strategy),
            runtime,
            "right column 'k'",
        )
        _checked(
            left.lazy()
            .join_asof(unsorted.lazy(), on="k", strategy=strategy)
            .head(0),
            runtime,
            "right column 'k'",
        )
        _checked(
            unsorted.lazy().join_asof(
                unsorted.lazy(), on="k", strategy=strategy
            ),
            runtime,
            "left column 'k'",
        )
    # Null BY rows cannot match but still participate in complete validation.
    var groups = Series("g", Column[Int32]([0, 0, 0], [False, False, False]))
    _checked(
        left.head(0)
        .with_column(groups.head(0))
        .lazy()
        .join_asof(
            unsorted.with_column(groups.copy()).lazy(), on="k", by=["g"]
        ),
        runtime,
        "right column 'k'",
    )
    var bad_left = (
        DataFrame(
            [
                Series("k", Column[Int64]([0])),
                Series("p", Column[UInt64]([UInt64.MAX])),
            ]
        )
        .lazy()
        .with_columns(col("p").cast("int32").alias("p"))
    )
    var bad_right = (
        DataFrame(
            [
                Series("k", Column[Int64]([0])),
                Series("p", Column[Int64]([Int64.MAX])),
            ]
        )
        .lazy()
        .with_columns((col("p") + 1).alias("p"))
    )
    _checked(
        bad_left.join_asof(bad_right, on="k"),
        runtime,
        "strict cast failed",
        "cast from uint64 to int32 failed",
    )
    _checked(
        left.head(0).lazy().join_asof(bad_right, on="k"), runtime, "overflow"
    )
    var source_failure = unsorted.lazy()
    source_failure._nodes[0].source_failure = "asof right source sentinel"
    _checked(
        bad_left.join_asof(source_failure, on="k"),
        runtime,
        "strict cast failed",
        "cast from uint64 to int32 failed",
    )
    _checked(
        left.lazy().join_asof(source_failure, on="k"),
        runtime,
        "asof right source sentinel",
    )
    var good = left.lazy().join_asof(left.lazy(), on="k")
    assert_frame(good.collect(accelerator=runtime), good.collect(engine="cpu"))


# requires: metal
def test_mojo_gpu_asof_public_resident_consumers_profiles_and_exact_budget() raises:
    var left = DataFrame(
        [
            Series("k", Column[Int64]([0, 1, 2, 3, 4])),
            Series("x", Column[Float64]([10, 11, 12, 13, 14])),
        ]
    )
    var right = DataFrame(
        [
            Series("t", Column[Int64]([0, 2, 2, 4])),
            Series("y", Column[Float64]([20, 21, 22, 24])),
        ]
    )
    var query = (
        left.lazy()
        .with_columns((col("x") + 1).alias("x"))
        .join_asof(
            right.lazy().with_columns((col("y") + 1).alias("y")),
            left_on="k",
            right_on="t",
        )
        .with_columns((col("x") + col("y")).alias("z"))
        .filter(col("z") > 32)
        .select_exprs([col("k"), col("z")])
        .sort("z", descending=True)
        .slice(1, 2)
    )
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    assert_frame(measured[0], query.collect(engine="cpu", streaming=False))
    assert_equal(
        measured[0].column("k").int64().to_list(), [Int64(3), Int64(2)]
    )
    assert_equal(
        measured[0].column("z").float64().to_list(), [Float64(37), Float64(36)]
    )
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    var asof_seen = False
    var exports = 0
    for row in range(measured[1].height()):
        asof_seen |= (
            measured[1].item(row, "algorithm").string()
            == "mojo_gpu_asof_backward"
        )
        assert_equal(
            measured[1].item(row, "runtime_implementation").string(), "mojo"
        )
        var boundary = (
            measured[1].item(row, "materialization_boundary").string()
        )
        if boundary == "Mojo GPU -> CPU columns":
            exports += 1
        else:
            assert_equal(
                measured[1].item(row, "host_result_bytes").int64(), Int64(0)
            )
    assert_true(asof_seen)
    assert_equal(exports, 1)
    assert_frame(
        measured[0],
        query.collect(accelerator=MetalRuntime(memory_limit_bytes=peak)),
    )
    var small = MetalRuntime(memory_limit_bytes=peak - 1)
    with assert_raises(contains="memory budget"):
        _ = query.collect(accelerator=small)
    assert_equal(
        left.head(1)
        .lazy()
        .select(col("x"))
        .collect(accelerator=small)
        .item(0, "x")
        .float64(),
        Float64(10),
    )
    var reduced = query.select_exprs(
        [col("z").sum().alias("s"), col("z").len().alias("n")]
    )
    var output = reduced.collect(accelerator=MetalRuntime())
    assert_frame(output, reduced.collect(engine="cpu", streaming=False))
    assert_equal(output.item(0, "s").float64(), Float64(73))
    assert_equal(output.item(0, "n").int64(), Int64(2))


# requires: metal
def test_mojo_gpu_asof_public_tolerance_temporal_identity_and_optional_capability() raises:
    var runtime = MetalRuntime()
    var left = DataFrame([Series("k", Column[UInt64]([0, UInt64.MAX]))])
    var right = DataFrame(
        [
            Series("k", Column[UInt64]([UInt64.MAX])),
            Series("p", Column[Int64]([7])),
        ]
    )
    for bound in [UInt64.MAX, UInt64.MAX - 1]:
        var query = left.lazy().join_asof(
            right.lazy(), on="k", strategy="nearest", tolerance=AnyValue(bound)
        )
        var actual = query.collect(accelerator=runtime)
        assert_asof_rows(
            actual,
            left,
            right,
            List[Int]([0, 0]) if bound == UInt64.MAX else List[Int]([-1, 0]),
            ["k", "p"],
            ["p"],
        )
        assert_frame(actual, query.collect(engine="cpu"))
    # Public tolerance lowering must preserve IEEE infinities and the CPU
    # forward cursor rule that differs from an external reference at -inf.
    comptime float_types = (DType.float32, DType.float64)
    comptime for index in range(len(float_types)):
        comptime D = float_types[index]
        var inf = Float64(1) / Float64(0)
        var nan = Float64(0) / Float64(0)
        var lv = List[Scalar[D]]()
        var rv = List[Scalar[D]]()
        for value in [-inf, -1.0, -0.0, 0.0, 1.0, inf, nan]:
            lv.append(value.cast[D]())
        for value in [-inf, 0.0, inf, nan]:
            rv.append(value.cast[D]())
        var l = DataFrame([Series("k", Column[Scalar[D]](lv^))])
        var r = DataFrame(
            [
                Series("k", Column[Scalar[D]](rv^)),
                Series("p", Column[Int64]([0, 1, 2, 3])),
            ]
        )
        var ieee = l.lazy().join_asof(
            r.lazy(),
            on="k",
            strategy="forward",
            allow_exact_matches=False,
            tolerance=AnyValue(inf),
        )
        var actual = ieee.collect(accelerator=runtime)
        assert_asof_rows(
            actual, l, r, [1, 1, 2, 2, 2, -1, -1], ["k", "p"], ["p"]
        )
        assert_frame(actual, ieee.collect(engine="cpu"))
    for unit in ["ms", "us", "ns"]:
        var dtype = DataType.datetime(unit, "+05:45")
        var l = DataFrame(
            [Series("k", Column[Int64]([-1, 0, 1])).with_dtype(dtype)]
        )
        var r = DataFrame(
            [
                Series("t", Column[Int64]([0])).with_dtype(dtype),
                Series("p", Column[Int64]([9])),
            ]
        )
        var ns = 1000000 if unit == "ms" else 1000 if unit == "us" else 1
        for bound in [ns - 1, ns]:
            var query = l.lazy().join_asof(
                r.lazy(),
                left_on="k",
                right_on="t",
                strategy="nearest",
                tolerance=AnyValue.temporal(
                    DataType.duration("ns"), Int64(bound)
                ),
            )
            var actual = query.collect(accelerator=runtime)
            assert_asof_rows(
                actual,
                l,
                r,
                List[Int]([-1, 0, -1]) if bound < ns else List[Int]([0, 0, 0]),
                ["k", "t", "p"],
                ["t", "p"],
            )
            assert_frame(actual, query.collect(engine="cpu"))
    var query = left.lazy().join_asof(right.lazy(), on="k")
    var caps = capabilities()
    caps.asof_joins = False
    with assert_raises(contains="as-of joins require backend support"):
        _ = lower_join_dag(query, caps)
    with assert_raises(contains="nonnegative"):
        _ = (
            left.lazy()
            .join_asof(right.lazy(), on="k", tolerance=AnyValue(Int64(-1)))
            .collect(accelerator=MetalRuntime(device_id=-99))
        )
    # Dictionary normalization may encode BY, but cannot legalize text ORDERED keys.
    var text = DataFrame([Series("k", Column[String](["a"]))])
    with assert_raises(contains="requires integer"):
        _ = (
            text.lazy()
            .join_asof(text.lazy(), on="k")
            .collect(accelerator=runtime)
        )
