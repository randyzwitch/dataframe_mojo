"""Numeric group state spans native allocations and preserves logical row order."""
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col
from dataframe.execution_observation import DeviceObservation
from dataframe_metal.chunked_numeric_reduce import (
    NUMERIC_DEFAULT,
    NUMERIC_SEGMENTED,
    NUMERIC_SINGLE_SPAN,
    native_numeric_layout,
    scan_chunked_numeric,
)
from dataframe_metal.group_finish import finish_grouped
from dataframe_metal.group_plan import plan_grouped
from dataframe_metal.group_reduce import execute_grouped
from dataframe_metal.handoff import region_key
from dataframe_metal.accelerator.gpu.query_scope import Scope, shared_service
from dataframe_metal.accelerator.gpu.reduction_inputs import HandoffTimings


def _split[
    D: DType
](
    name: String, values: List[Scalar[D]], valid: List[Bool], boundary: Int
) raises -> Series:
    # Identical logical order, independent allocations, non-byte-aligned bitmaps.
    var left = Series(name, Column[Scalar[D]](values.copy(), valid.copy()))
    var right = Series(name, Column[Scalar[D]](values.copy(), valid.copy()))
    return Series._from_chunks(
        [left.slice(0, boundary), right.slice(boundary, len(values) - boundary)]
    )


def _frame() raises -> DataFrame:
    var wide = Int64(1) << 40
    return DataFrame(
        [
            _split[DType.int32](
                "zone",
                [0, 1, -3, 0, -3, 1, 7, 7],
                [True, False, True, True, True, False, True, True],
                3,
            ),
            _split[DType.int64](
                "balance",
                [wide, -3, 0, -wide, 9, 0, 0, 3],
                [True, True, False, True, True, False, False, True],
                2,
            ),
            _split[DType.float32](
                "reading",
                [1.5, 3, -6, 2.5, 6.5, 1, -0.0, 0.0],
                [True, False, True, True, True, False, True, True],
                5,
            ),
            _split[DType.int32](
                "units",
                [10, -7, -12, 4, 100, 42, 0, 0],
                [True, True, True, True, True, True, False, False],
                6,
            ),
        ]
    )


def _query(frame: DataFrame) raises -> LazyFrame:
    return (
        frame.lazy()
        .group_by("zone", maintain_order=True)
        .agg(
            [
                col("reading").mean().alias("avg"),
                col("balance").sum(min_count=2).alias("net"),
                col("units").min().alias("lo"),
                col("units").max().alias("hi"),
                col("zone").count().alias("known_zone"),
                col("balance").len().alias("nrows"),
            ]
        )
    )


def test_chunked_numeric_matcher_selects_native_columns_and_single_int32_key() raises:
    var frame = _frame()
    var plan = _query(frame)
    var matched = plan_grouped(plan)
    assert_true(Bool(matched))
    assert_equal(
        native_numeric_layout(plan, matched.value(), frame), NUMERIC_SEGMENTED
    )
    assert_true(region_key(plan).endswith(":chunked_native_numeric:v1"))
    var contiguous = frame.rechunk()
    var contiguous_plan = _query(contiguous)
    assert_equal(
        native_numeric_layout(
            contiguous_plan, plan_grouped(contiguous_plan).value(), contiguous
        ),
        NUMERIC_SINGLE_SPAN,
    )
    assert_true(region_key(contiguous_plan).endswith(":single_span_numeric:v1"))
    var repeated = (
        contiguous.lazy()
        .group_by("zone")
        .agg(
            [
                col("reading").mean().alias("avg"),
                col("reading").count().alias("n"),
            ]
        )
    )
    assert_equal(
        native_numeric_layout(
            repeated, plan_grouped(repeated).value(), contiguous
        ),
        NUMERIC_DEFAULT,
    )
    var unrelated = DataFrame(
        [
            contiguous.column("zone"),
            contiguous.column("reading"),
            frame.column("units"),
        ]
    )
    var simple = unrelated.lazy().group_by("zone").agg(col("reading").mean())
    assert_equal(
        native_numeric_layout(simple, plan_grouped(simple).value(), unrelated),
        NUMERIC_DEFAULT,
    )
    var filtered = (
        frame.lazy()
        .filter(col("units") > 0)
        .group_by("zone")
        .agg([col("reading").mean()])
    )
    assert_equal(
        native_numeric_layout(filtered, plan_grouped(filtered).value(), frame),
        NUMERIC_DEFAULT,
    )
    var extrema = frame.lazy().group_by("zone").agg(col("balance").max())
    assert_equal(
        native_numeric_layout(extrema, plan_grouped(extrema).value(), frame),
        NUMERIC_DEFAULT,
    )
    var computed = frame.lazy().group_by("zone").agg((col("balance") + 1).sum())
    assert_equal(
        native_numeric_layout(computed, plan_grouped(computed).value(), frame),
        NUMERIC_DEFAULT,
    )


# requires: metal
def test_chunked_numeric_groups_reuse_nullable_keys_and_accumulate_four_fields() raises:
    from tests.support.metal_outcomes import require_metal

    var runtime = require_metal()
    var frame = _frame()
    var query = _query(frame)
    var matched = plan_grouped(query)
    var plan = matched.take()
    var expected = DataFrame(
        [
            Series(
                "zone", Column[Int32]([0, 0, -3, 7], [True, False, True, True])
            ),
            Series(
                "avg",
                Column[Float64]([2, 0, 0.25, 0], [True, False, True, True]),
            ),
            Series(
                "net", Column[Int64]([0, 0, 0, 0], [True, False, False, False])
            ),
            Series(
                "lo", Column[Int32]([4, -7, -12, 0], [True, True, True, False])
            ),
            Series(
                "hi", Column[Int32]([10, 42, 100, 0], [True, True, True, False])
            ),
            Series("known_zone", Column[Int64]([2, 0, 2, 2])),
            Series("nrows", Column[Int64]([2, 2, 2, 2])),
        ]
    )
    for wrap in [True, False]:
        var obs = ArcPointer(DeviceObservation())
        var scope = Scope(service=shared_service(), observation=obs)
        var timings = HandoffTimings()
        var partials = scan_chunked_numeric(plan, frame, scope, wrap, timings)
        var answer = finish_grouped(
            plan,
            frame,
            partials.blocks,
            partials.first,
            partials.rows,
            partials.states,
            partials.extrema,
        )
        assert_true(answer.equals(expected))
        # One initialization and five independent input intersections (one
        # intersection in copy mode). The key also participates in COUNT.
        assert_equal(obs[].completed_launches.load(), Int64(6 if wrap else 2))
        assert_equal(obs[].upload_bytes.load() == 0, wrap)
        assert_equal(scope.info()[0], 0)
        scope.mark_success()
    assert_true(query.collect(accelerator=runtime).equals(expected))


# requires: metal
def test_metal_single_span_numeric_groups_match_cpu_for_shared_slices() raises:
    from tests.support.metal_outcomes import require_metal

    var runtime = require_metal()
    var contiguous = _frame().rechunk()
    var pieces = List[Series]()
    for name in contiguous.columns():
        var source = contiguous.column(name)
        pieces.append(
            Series._from_chunks(
                [source.slice(0, 3), source.slice(3, len(source) - 3)]
            )
        )
    var shared = DataFrame(pieces^)
    assert_equal(shared.column("reading").n_chunks(), 2)
    var expected = _query(contiguous).collect(engine="cpu")
    for frame in [contiguous^, shared^]:
        var plan = _query(frame)
        assert_true(region_key(plan).endswith(":single_span_numeric:v1"))
        var observation = ArcPointer(DeviceObservation())
        plan._plan._device_observation = observation
        var answer = execute_grouped(plan, True)
        assert_true(answer[0].equals(expected))
        assert_equal(observation[].completed_launches.load(), Int64(2))
        assert_equal(observation[].upload_bytes.load(), Int64(0))
        assert_true(plan.collect(accelerator=runtime).equals(expected))


# requires: metal
def test_metal_numeric_shared_group_state_survives_capacity_retry() raises:
    from dataframe_metal.group_properties import sample_group_tuples
    from tests.support.metal_outcomes import require_metal

    _ = require_metal()
    var n = 65536
    for common in [1, 300]:
        var keys = List[Int32]()
        var wide = List[Int64]()
        var value = List[Float32]()
        var key_valid = List[Bool]()
        var value_valid = List[Bool]()
        # A long repeated prefix keeps the sampled table below the actual number
        # of groups. The two prefix sizes exercise both initial table sizes.
        for i in range(n):
            var key = Int32(
                i % common if i < n - 2500 else common + i - (n - 2500)
            )
            keys.append(key - 150)
            key_valid.append(i % 113 != 0)
            wide.append((Int64(1) << 40) + Int64(i % 7))
            value.append(Float32(i % 41 - 22) / 8)
            value_valid.append(i % 17 != 0 and key != 42)
        var frame = DataFrame(
            [
                Series("team", Column[Int32](keys^, key_valid)),
                Series("income", Column[Int64](wide^)),
                Series("reading", Column[Float32](value^, value_valid)),
            ]
        )
        var query = (
            frame.lazy()
            .group_by("team", maintain_order=True)
            .agg(
                [
                    col("reading").mean().alias("avg"),
                    col("reading").min().alias("lo"),
                    col("income").sum().alias("total"),
                    col("reading").len().alias("rows"),
                ]
            )
        )
        var expected = query.collect(engine="cpu")
        var sample = sample_group_tuples(frame, [String("team")])
        assert_equal(sample.global_state(), common == 300)
        assert_true(sample.capacity() // 2 < expected.height())
        var variant = (
            ":shared_group_output:v1" if common
            == 300 else ":shared_group_retry:v1"
        )
        assert_true(variant in region_key(query))
        for wrap in [True, False]:
            var observation = ArcPointer(DeviceObservation())
            query._plan._device_observation = observation
            var result = execute_grouped(query, wrap)
            assert_true(result[0].equals(expected))
            assert_equal(observation[].completed_launches.load(), Int64(4))
            var copied = observation[].download_bytes.load()
            if not wrap:
                assert_true(copied > 32 * 1024 * 1024)
            elif sample.global_state():
                assert_equal(copied, Int64(0))
            else:
                # Only the first 512-slot attempt copied output back. Replay wrote
                # its large table directly into CPU-visible storage.
                assert_equal(copied, Int64(4 * (2 * 512 + 512 * 2 * 32 + 4)))
