"""Numeric group state spans native allocations and preserves logical row order."""
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_false, assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col
from dataframe.execution_observation import DeviceObservation
from dataframe_metal.chunked_numeric_reduce import (
    scan_chunked_numeric,
    uses_chunked_numeric,
)
from dataframe_metal.group_finish import finish_grouped
from dataframe_metal.group_plan import plan_grouped
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
    assert_true(uses_chunked_numeric(plan, matched.value(), frame))
    var contiguous = frame.rechunk()
    var contiguous_plan = _query(contiguous)
    assert_false(
        uses_chunked_numeric(
            contiguous_plan, plan_grouped(contiguous_plan).value(), contiguous
        )
    )
    var unrelated = DataFrame(
        [
            contiguous.column("zone"),
            contiguous.column("reading"),
            frame.column("units"),
        ]
    )
    var simple = unrelated.lazy().group_by("zone").agg(col("reading").mean())
    assert_false(
        uses_chunked_numeric(simple, plan_grouped(simple).value(), unrelated)
    )
    var filtered = (
        frame.lazy()
        .filter(col("units") > 0)
        .group_by("zone")
        .agg([col("reading").mean()])
    )
    assert_false(
        uses_chunked_numeric(filtered, plan_grouped(filtered).value(), frame)
    )
    var extrema = frame.lazy().group_by("zone").agg(col("balance").max())
    assert_false(
        uses_chunked_numeric(extrema, plan_grouped(extrema).value(), frame)
    )
    var computed = frame.lazy().group_by("zone").agg((col("balance") + 1).sum())
    assert_false(
        uses_chunked_numeric(computed, plan_grouped(computed).value(), frame)
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
