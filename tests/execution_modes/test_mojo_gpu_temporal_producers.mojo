"""Observed temporal producers retain original phase and mask behavior."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe.metal import MetalRuntime
from tests.support.coverage_compare import assert_frame
from tests.tooling.iana_scalar_any_proofs import _frame, _expression


def test_mojo_gpu_temporal_producer_head_order_group_faults_and_recovery() raises:
    var runtime = MetalRuntime()
    var naive = DataType.datetime("us")
    var ny = DataType.datetime("us", "America/New_York")
    var source = DataFrame(
        [
            Series("t", Column[Int64]([0, 1])).with_dtype(naive),
            Series("k", Column[Int32]([0, 0])),
        ]
    )
    var produced = source.lazy().with_columns(
        col("t").dt().replace_time_zone("America/New_York").alias("z")
    )
    var selected = produced.sort("z").head(1)
    var expected = DataFrame(
        [
            Series("t", Column[Int64]([0])).with_dtype(naive),
            Series("k", Column[Int32]([0])),
            Series("z", Column[Int64]([18000000000])).with_dtype(ny),
        ]
    )
    assert_frame(selected.collect(engine="cpu"), expected)
    assert_frame(selected.collect(accelerator=runtime), expected)
    var grouped = produced.group_by("k", maintain_order=True).agg(
        col("z").max()
    )
    var grouped_expected = DataFrame(
        [
            Series("k", Column[Int32]([0])),
            Series("z", Column[Int64]([18000000001])).with_dtype(ny),
        ]
    )
    assert_frame(grouped.collect(engine="cpu"), grouped_expected)
    assert_frame(grouped.collect(accelerator=runtime), grouped_expected)
    var gap = DataFrame(
        [Series("t", Column[Int64]([0, 1710037800000000])).with_dtype(naive)]
    )
    var checked = gap.lazy().with_columns(
        col("t").dt().replace_time_zone("America/New_York").alias("z")
    )
    for count in List[Int]([0, 1]):
        var failing = checked.sort("z").head(count)
        with assert_raises(contains="non-existent"):
            _ = failing.collect(engine="cpu")
        with assert_raises(contains="non-existent"):
            _ = failing.collect(accelerator=runtime)
    # The verified optimizer bridge moves this leading head before the
    # checked row producer, so the gap in the second source row is unseen.
    var optimized = checked.head(1).select(col("z"))
    var optimized_expected = DataFrame(
        [Series("z", Column[Int64]([18000000000])).with_dtype(ny)]
    )
    assert_frame(optimized.collect(engine="cpu"), optimized_expected)
    assert_frame(optimized.collect(accelerator=runtime), optimized_expected)
    with assert_raises(contains="budget"):
        _ = selected.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    assert_frame(selected.collect(accelerator=runtime), expected)
    print(
        "PASS public observed temporal producer head/order/group, faults and recovery"
    )


def test_mojo_gpu_temporal_producer_integer_prefix_sibling_order_and_recovery() raises:
    var runtime = MetalRuntime()
    var source = DataFrame(
        [
            Series("t", Column[Int64]([0, 1710037800000000])).with_dtype(
                DataType.datetime("us")
            ),
            Series("v", Column[Int64]([Int64.MAX, 0])),
        ]
    )
    # The unused checked integer producer precedes localization. A later
    # ordering/head cannot hide it or replace its diagnostic with the gap.
    var integer_first = (
        source.lazy()
        .with_columns((col("v") + lit(Int64(1))).alias("unused"))
        .with_columns(
            col("t").dt().replace_time_zone("America/New_York").alias("z")
        )
        .sort("z")
        .head(0)
    )
    with assert_raises(contains="overflow"):
        _ = integer_first.collect(engine="cpu", streaming=False, optimize=False)
    with assert_raises(contains="overflow"):
        _ = integer_first.collect(accelerator=runtime, optimize=False)
    # In the reverse phase order, localization must finish before the later
    # integer producer even though that producer faults at an earlier row.
    var temporal_first = (
        source.lazy()
        .with_columns(
            col("t").dt().replace_time_zone("America/New_York").alias("z")
        )
        .with_columns((col("v") + lit(Int64(1))).alias("unused"))
        .sort("z")
        .head(0)
    )
    with assert_raises(contains="non-existent"):
        _ = temporal_first.collect(
            engine="cpu", streaming=False, optimize=False
        )
    with assert_raises(contains="non-existent"):
        _ = temporal_first.collect(accelerator=runtime, optimize=False)
    # Within one certified producer the original projection rank is observed
    # before row rank, in both sibling orders.
    for gap_first in [False, True]:
        var gap = col("t").dt().replace_time_zone("America/New_York").alias("z")
        var overflow = (col("v") + lit(Int64(1))).alias("unused")
        var expressions: List[Expr] = [
            gap.copy(),
            overflow.copy(),
        ] if gap_first else [overflow.copy(), gap.copy()]
        var siblings = source.lazy().with_columns(expressions).sort("t").head(0)
        with assert_raises(
            contains="non-existent" if gap_first else "overflow"
        ):
            _ = siblings.collect(engine="cpu", streaming=False, optimize=False)
        with assert_raises(
            contains="non-existent" if gap_first else "overflow"
        ):
            _ = siblings.collect(accelerator=runtime, optimize=False)
    var safe = (
        source.head(1)
        .lazy()
        .with_columns(
            col("t").dt().replace_time_zone("America/New_York").alias("z")
        )
        .sort("z")
    )
    var recovered = safe.collect(accelerator=runtime, optimize=False)
    assert_frame(
        recovered,
        safe.collect(engine="cpu", streaming=False, optimize=False),
    )
    assert_equal(
        recovered.select(col("z").cast(DataType.INT64)).item().int64(),
        Int64(18000000000),
    )
    assert_equal(
        recovered["z"].dtype(), DataType.datetime("us", "America/New_York")
    )
    print(
        "PASS temporal/integer prefix and sibling fault order, head0 and recovery"
    )


def test_mojo_gpu_temporal_producer_scalar_any_streaming_batch_geometry() raises:
    var runtime = MetalRuntime()
    var mismatches = 0
    for batch in List[Int]([4096, 8191, 8192, 8193, 16384]):
        # The pairs below/above 8192 share an eager Frame evaluation batch,
        # but straddle the requested streaming morsel when batch != 8192.
        var gap_row = (
            batch - 1 if batch < 8192 else 8192 if batch == 8193 else 0
        )
        var active_row = gap_row + 1
        for all_false in [False, True]:
            var source = _frame(16386, gap_row, -1 if all_false else active_row)
            var query = (
                source.lazy().with_columns(_expression(False)).sort("out")
            )
            var eager = String("")
            try:
                _ = query.collect(engine="cpu", streaming=False, optimize=False)
            except error:
                eager = String(error)
            if all_false:
                assert_equal(eager, "")
            else:
                assert_true("non-existent" in eager)
            var cpu = String("")
            var gpu = String("")
            try:
                _ = query.collect(
                    engine="cpu",
                    streaming=True,
                    batch_size=batch,
                    optimize=False,
                )
            except error:
                cpu = String(error)
            try:
                _ = query.collect(
                    accelerator=runtime,
                    streaming=True,
                    batch_size=batch,
                    optimize=False,
                )
            except error:
                gpu = String(error)
            print(
                "mode6 batch="
                + String(batch)
                + " gap="
                + String(gap_row)
                + " active="
                + String(-1 if all_false else active_row)
                + " eager=["
                + eager
                + "] CPU=["
                + cpu
                + "] GPU=["
                + gpu
                + "]",
                flush=True,
            )
            mismatches += Int(cpu != gpu)
    assert_equal(mismatches, 0)
