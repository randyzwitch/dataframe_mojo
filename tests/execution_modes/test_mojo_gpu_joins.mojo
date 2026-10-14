"""Direct Mojo joins preserve typed pairs, IEEE keys and bounded cardinality."""
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_join_domains import (
    join_frame,
    pairs,
    expected_join,
)
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.support.coverage_generators import offset_layout, chunk_layout


def _typed_join[D: DType](runtime: MetalRuntime, dtype: DataType) raises:
    var left_rows: List[Int] = [0, 1, 2, 3, 4, 5, 6]
    var right_rows: List[Int] = [0, 1, 2, 3]
    var a = offset_layout(join_frame[D](left_rows, False, dtype))
    var b = chunk_layout(join_frame[D](right_rows, True, dtype))
    for how in ["inner", "left", "right", "full", "semi", "anti", "cross"]:
        var expected_pairs = pairs[D](left_rows, right_rows, how)
        var query = a.lazy().join(
            b.lazy(), how="cross"
        ) if how == "cross" else a.lazy().join(
            b.lazy(), left_on=["k", "b"], right_on=["r", "b"], how=how
        )
        same(
            query.collect(accelerator=runtime, engine="accel"),
            expected_join[D](
                expected_pairs[0], expected_pairs[1], how, True, dtype
            ),
        )
    for empty_side in range(3):
        var left = List[Int]() if empty_side != 2 else left_rows.copy()
        var right = List[Int]() if empty_side != 1 else right_rows.copy()
        var empty_a = join_frame[D](left, False, dtype)
        var empty_b = join_frame[D](right, True, dtype)
        for how in ["left", "right", "full", "anti"]:
            var expected_pairs = pairs[D](left, right, how)
            var query = empty_a.lazy().join(
                empty_b.lazy(),
                left_on=["k", "b"],
                right_on=["r", "b"],
                how=how,
                coalesce=False,
            )
            same(
                query.collect(accelerator=runtime, engine="accel"),
                expected_join[D](
                    expected_pairs[0], expected_pairs[1], how, False, dtype
                ),
            )


def test_mojo_gpu_join_styles_typed_composite_keys_coalescing_and_empty_sides() raises:
    var runtime = MetalRuntime()
    comptime types = (
        DType.float32,
        DType.int32,
        DType.int64,
        DType.int8,
        DType.int16,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float64,
    )
    comptime for at in range(len(types)):
        _typed_join[types[at]](runtime, DataType.of(types[at]))
    _typed_join[DType.bool](runtime, DataType.BOOL)
    _typed_join[DType.int64](runtime, DataType.DATE)


def test_mojo_gpu_join_nan_zero_subnormal_keys_and_exact_payloads() raises:
    var encodings: List[UInt64] = [
        0,
        0x8000000000000000,
        0x7FF8000000000001,
        0xFFF8000000000002,
        1,
        0x8000000000000001,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x3FF0000000000000,
        0xBFF0000000000000,
    ]
    var values = List[Float64]()
    var other = List[Float64]()
    var valid = List[Bool]()
    var other_valid = List[Bool]()
    var ids = List[Int32]()
    for row in range(len(encodings)):
        values.append(bitcast[DType.float64](encodings[row]))
        other.append(
            bitcast[DType.float64](encodings[len(encodings) - row - 1])
        )
        valid.append(row != 8)
        other_valid.append(row != 8)
        ids.append(Int32(row))
    var left = offset_layout(
        DataFrame(
            [
                Series("k", Column[Float64](values.copy(), valid)),
                Series("id", Column[Int32](ids.copy())),
            ]
        )
    )
    var right = chunk_layout(
        DataFrame(
            [
                Series("k", Column[Float64](other.copy(), other_valid)),
                Series("id", Column[Int32](ids.copy())),
            ]
        )
    )
    for how in ["inner", "left", "right", "full", "semi", "anti"]:
        var a = List[Int]()
        var b = List[Int]()
        var matched_right = List[Bool](length=len(other), fill=False)
        if how == "right":
            for r in range(len(other)):
                var found = False
                for l in range(len(values)):
                    if (
                        valid[l]
                        and other_valid[r]
                        and (
                            values[l] == other[r]
                            or (isnan(values[l]) and isnan(other[r]))
                        )
                    ):
                        a.append(l)
                        b.append(r)
                        found = True
                if not found:
                    a.append(-1)
                    b.append(r)
        else:
            for l in range(len(values)):
                var found = False
                for r in range(len(other)):
                    if (
                        valid[l]
                        and other_valid[r]
                        and (
                            values[l] == other[r]
                            or (isnan(values[l]) and isnan(other[r]))
                        )
                    ):
                        found = True
                        matched_right[r] = True
                        if how != "semi" and how != "anti":
                            a.append(l)
                            b.append(r)
                if (
                    (how == "semi" and found)
                    or (how == "anti" and not found)
                    or (not found and (how == "left" or how == "full"))
                ):
                    a.append(l)
                    b.append(-1)
            if how == "full":
                for r in range(len(other)):
                    if not matched_right[r]:
                        a.append(-1)
                        b.append(r)
        var query = left.lazy().join(
            right.lazy(), on="k", how=how, coalesce=False
        )
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        assert_equal(actual.height(), len(a))
        for row in range(len(a)):
            assert_equal(actual.item(row, "id").is_null(), a[row] < 0)
            if a[row] >= 0:
                assert_equal(
                    actual.item(row, "id").numeric[DType.int32](), Int32(a[row])
                )
                assert_equal(actual.item(row, "k").is_null(), not valid[a[row]])
                if valid[a[row]]:
                    assert_equal(
                        bitcast[DType.uint64](actual.item(row, "k").float64()),
                        encodings[a[row]],
                    )
            else:
                assert_true(actual.item(row, "k").is_null())
            if how != "semi" and how != "anti":
                assert_equal(actual.item(row, "id_right").is_null(), b[row] < 0)
                if b[row] >= 0:
                    assert_equal(
                        actual.item(row, "id_right").numeric[DType.int32](),
                        Int32(b[row]),
                    )
                    assert_equal(
                        actual.item(row, "k_right").is_null(),
                        not other_valid[b[row]],
                    )
                    if other_valid[b[row]]:
                        assert_equal(
                            bitcast[DType.uint64](
                                actual.item(row, "k_right").float64()
                            ),
                            encodings[len(encodings) - b[row] - 1],
                        )
        same(actual, query.collect(engine="cpu", streaming=False))


def test_mojo_gpu_join_bounded_cross_fanout_and_exact_memory_budget() raises:
    var ids = List[Int32]()
    for row in range(50_000):
        ids.append(Int32(row))
    var source = DataFrame([Series("id", Column[Int32](ids^))])
    var query = source.lazy().join(source.lazy(), how="cross").head(513)
    var runtime = MetalRuntime(memory_limit_bytes=8 * 1024 * 1024)
    var profile = query.profile(accelerator=runtime, engine="accel")
    assert_equal(profile[0].height(), 513)
    for row in range(513):
        assert_equal(
            profile[0].item(row, "id").numeric[DType.int32](), Int32(0)
        )
        assert_equal(
            profile[0].item(row, "id_right").numeric[DType.int32](), Int32(row)
        )
    var peak = 0
    for row in range(profile[1].height()):
        peak = max(
            peak, Int(profile[1].item(row, "peak_requested_bytes").int64())
        )
    assert_true(peak <= 8 * 1024 * 1024)
    with assert_raises(contains="memory budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    same(
        profile[0],
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ),
    )
    assert_equal(
        query.head(0).collect(accelerator=runtime, engine="accel").height(), 0
    )


def test_mojo_gpu_join_regions_consumers_retained_budget_and_preflight() raises:
    var left = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 2, 3])),
            Series("x", Column[Float64]([10, 20, 21, 30])),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64]([2, 2, 4])),
            Series("y", Column[Float64]([200, 201, 400])),
        ]
    )
    var a = left.lazy().with_columns((col("x") + 1).alias("x"))
    var b = right.lazy().with_columns((col("y") + 1).alias("y"))
    var query = (
        a.join(b, on="k")
        .with_columns((col("x") + col("y")).alias("z"))
        .filter(col("z") > 222)
    )
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    same(measured[0], query.collect(engine="cpu", streaming=False))
    assert_equal(measured[0].height(), 3)
    assert_equal(measured[0].item(0, "z").float64(), Float64(223))
    assert_equal(measured[0].item(1, "z").float64(), Float64(223))
    assert_equal(measured[0].item(2, "z").float64(), Float64(224))
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    var joined = False
    var retained = False
    for row in range(measured[1].height()):
        var local = Int(measured[1].item(row, "peak_requested_bytes").int64())
        var live = Int(
            measured[1].item(row, "retained_intermediate_bytes").int64()
        )
        assert_true(peak >= local + live)
        joined |= (
            measured[1].item(row, "algorithm").string()
            == "mojo_gpu_sorted_join"
        )
        retained |= live > 0
        assert_equal(
            measured[1].item(row, "runtime_implementation").string(), "mojo"
        )
    assert_true(joined and retained)
    with assert_raises(contains="memory budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    same(
        measured[0],
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ),
    )
    var reduced = query.select_exprs(
        [
            col("z").sum().alias("s"),
            col("z").mean().alias("m"),
            col("z").len().alias("n"),
        ]
    )
    var result = reduced.collect(accelerator=MetalRuntime(), engine="accel")
    same(result, reduced.collect(engine="cpu", streaming=False))
    assert_equal(result.item(0, "s").float64(), Float64(670))
    assert_equal(result.item(0, "n").int64(), Int64(3))
    var third = DataFrame(
        [Series("k", Column[Int64]([2])), Series("q", Column[Int64]([-10]))]
    )
    var chained = query.join(third.lazy(), on="k").select_exprs(
        [col("z"), col("q")]
    )
    same(
        chained.collect(accelerator=MetalRuntime(), engine="accel"),
        chained.collect(engine="cpu", streaming=False),
    )
    var floats = DataFrame([Series("v", Column[Float64]([1, 2, 3]))])
    var scalar = (
        floats.lazy()
        .select(col("v").sum())
        .with_columns((col("v") + 1).alias("v"))
    )
    assert_equal(
        scalar.collect(accelerator=MetalRuntime(), engine="accel")
        .item()
        .float64(),
        Float64(7),
    )
    var bad = DataFrame(
        [
            Series("k", Column[Int64]([1, 2])),
            Series("x", Column[Int64]([Int64.MAX, 1])),
        ]
    )
    var overflowing = (
        bad.lazy()
        .with_columns((col("x") + 1).alias("x"))
        .join(right.lazy(), on="k", how="left")
        .head(0)
    )
    with assert_raises(contains="checked integer arithmetic"):
        _ = overflowing.collect(accelerator=MetalRuntime(), engine="accel")
    var unsupported = left.lazy().join(
        right.lazy().with_columns(col("y").cast(DataType.STRING).alias("y")),
        on="k",
    )
    with assert_raises(contains="unsupported"):
        _ = unsupported.collect(
            accelerator=MetalRuntime(device_id=999999), engine="accel"
        )
    assert_true(
        "Mojo GPU joins" in MetalRuntime(device_id=999999).describe(query)
    )
    print(
        "PASS Mojo GPU DAG flows: branches, post-join rows/reductions, chained joins, scalar consumer, retained budget, preflight and checked errors"
    )


def test_mojo_gpu_join_empty_projection_preserves_cardinality_and_zero_column_contract() raises:
    var left = DataFrame([Series("k", Column[Int64]([1, 2, 2, 3]))])
    var right = DataFrame([Series("k", Column[Int64]([2, 2, 4]))])
    var styles: List[String] = [
        "inner",
        "left",
        "right",
        "full",
        "semi",
        "anti",
        "cross",
    ]
    var heights: List[Int] = [4, 6, 5, 7, 2, 2, 12]
    for at in range(len(styles)):
        var query = left.lazy().join(right.lazy(), how="cross") if styles[
            at
        ] == "cross" else left.lazy().join(right.lazy(), on="k", how=styles[at])
        var empty = query.select(List[String]())
        var actual = empty.collect(accelerator=MetalRuntime(), engine="accel")
        assert_equal(actual.shape(), (heights[at], 0))
        assert_equal(
            empty.head(1)
            .collect(accelerator=MetalRuntime(), engine="accel")
            .shape(),
            (1, 0),
        )
        assert_equal(
            empty.head(0)
            .collect(accelerator=MetalRuntime(), engine="accel")
            .shape(),
            (0, 0),
        )
        assert_true(actual.equals(empty.collect(engine="cpu", streaming=False)))
    # The shared/native join contract requires an internal output column.
    # A keyed empty projection still retains its keys inside the join; a
    # zero-column cross join has no such internal field and rejects before
    # device initialization, independently of its cardinality limit.
    var zero_left = DataFrame(List[Series](), height=50_000)
    var zero_right = DataFrame(List[Series](), height=50_000)
    var zero_cross = (
        zero_left.lazy().join(zero_right.lazy(), how="cross").head(513)
    )
    with assert_raises(contains="join requires 1"):
        _ = zero_cross.collect(
            accelerator=MetalRuntime(device_id=999999), engine="accel"
        )
