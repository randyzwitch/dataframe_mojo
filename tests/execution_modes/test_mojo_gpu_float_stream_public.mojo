"""Public original-route dispatch for ordered grouped floating states."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, DataType, Expr, col
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.grouped_stream_routes import (
    capture,
    terminal_family,
    BASIC,
    candidate,
)
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.float_streaming import (
    source,
    priority_source,
    FlushSetting,
)


# requires: metal
def test_public_float_stream_finite_order_types_and_profile() raises:
    var workers = WorkerSetting(1)
    comptime for i in range(2):
        comptime D = DType.float32 if i == 0 else DType.float64
        for threshold in [4096, 1]:
            var flush = FlushSetting(threshold)
            var frame = source[D](0)
            var query = (
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg(
                    [
                        col("v").sum().alias("s"),
                        col("v").mean().alias("m"),
                        col("v").min().alias("mn"),
                        col("v").max().alias("mx"),
                    ]
                )
            )
            var measured = query.profile(
                accelerator=MetalRuntime(), optimize=False, batch_size=64
            )
            var actual = measured[0].copy()
            var expected = query.collect(
                engine="cpu", optimize=False, streaming=True, batch_size=64
            )
            assert_true(actual.equals(expected))
            assert_equal(actual.column("s").dtype(), DataType.of(D))
            assert_equal(actual.column("m").dtype(), DataType.FLOAT64)
            var literal = Float64(0 if threshold == 1 else 1)
            for group in range(8):
                assert_equal(
                    Float64(
                        actual.column("s")._data[Column[Scalar[D]]]._get(group)
                    ),
                    literal,
                )
                assert_equal(
                    actual.column("m")._data[Column[Float64]]._get(group),
                    literal / 32,
                )
            var report = measured[1].copy()
            assert_equal(
                report.item(0, "algorithm").string(),
                "mojo_gpu_single_worker_float_epochs",
            )
            var peak = Int(report.item(0, "query_peak_requested_bytes").int64())
            var exact = query.collect(
                accelerator=MetalRuntime(memory_limit_bytes=peak),
                optimize=False,
                batch_size=64,
            )
            assert_true(exact.equals(actual))
            with assert_raises(contains="memory budget"):
                _ = query.collect(
                    accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
                    optimize=False,
                    batch_size=64,
                )
            _ = flush^
    _ = workers^


def _priority_case[
    D: DType
](first: UInt64, second: UInt64, threshold: Int, invalid_inf: Bool) raises:
    var flush = FlushSetting(threshold)
    var frame = priority_source[D](first, second)
    var query = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg([col("v").sum().alias("s"), col("v").mean().alias("m")])
    )
    var actual = query.collect(
        accelerator=MetalRuntime(), optimize=False, batch_size=64
    )
    var cpu = query.collect(
        engine="cpu", streaming=True, optimize=False, batch_size=64
    )
    var expected_sum = first if D == DType.float32 or threshold == 1 else second
    expected_sum |= UInt64(0x00400000) if D == DType.float32 else UInt64(
        0x0008000000000000
    )
    if invalid_inf:
        expected_sum = UInt64(0x7FC00000) if D == DType.float32 else UInt64(
            0x7FF8000000000000
        )
    var expected_mean = expected_sum
    comptime if D == DType.float32:
        expected_mean = (
            ((expected_sum & 0x80000000) << 32)
            | UInt64(0x7FF0000000000000)
            | ((expected_sum & 0x007FFFFF) << 29)
        )
    for result in [actual.copy(), cpu.copy()]:
        assert_equal(result.column("s").dtype(), DataType.of(D))
        assert_equal(result.column("m").dtype(), DataType.FLOAT64)
        for group in range(8):
            var value = result.column("s")._data[Column[Scalar[D]]]._get(group)
            comptime if D == DType.float32:
                assert_equal(UInt64(bitcast[DType.uint32](value)), expected_sum)
            else:
                assert_equal(bitcast[DType.uint64](value), expected_sum)
            assert_equal(
                bitcast[DType.uint64](
                    result.column("m")._data[Column[Float64]]._get(group)
                ),
                expected_mean,
            )
    _ = flush^


# requires: metal
def test_public_float_stream_nan_literal_payloads() raises:
    var workers = WorkerSetting(1)
    comptime for i in range(2):
        comptime D = DType.float32 if i == 0 else DType.float64
        for family in range(4):
            var left = UInt64(0x7FC00123) if i == 0 else UInt64(
                0x7FF8000000000123
            )
            var right = UInt64(0xFFC00456) if i == 0 else UInt64(
                0xFFF8000000000456
            )
            if family == 1 or family == 2:
                right &= ~(
                    UInt64(0x00400000) if i == 0 else UInt64(0x0008000000000000)
                )
            if family == 2:
                left &= ~(
                    UInt64(0x00400000) if i == 0 else UInt64(0x0008000000000000)
                )
            if family == 3:
                left = UInt64(0x7F800000) if i == 0 else UInt64(
                    0x7FF0000000000000
                )
                right = UInt64(0xFF800000) if i == 0 else UInt64(
                    0xFFF0000000000000
                )
            for reverse in [False, True]:
                for threshold in [4096, 1]:
                    _priority_case[D](
                        right if reverse else left,
                        left if reverse else right,
                        threshold,
                        family == 3,
                    )
    _ = workers^


# requires: metal
def test_public_float_stream_empty_without_device_and_route_guards() raises:
    var workers = WorkerSetting(1)
    var frame = source[DType.float64](0)
    var empty = (
        frame.head(0)
        .lazy()
        .group_by("k", maintain_order=True)
        .agg(col("v").sum())
    )
    var result = empty.collect(
        accelerator=MetalRuntime(device_id=999999, memory_limit_bytes=0)
    )
    assert_equal(result.height(), 0)
    assert_equal(result.column("v").dtype(), DataType.FLOAT64)
    var query = (
        frame.lazy().group_by("k", maintain_order=True).agg(col("v").sum())
    )
    assert_true(
        "captured CPU single-job"
        in query.explain(
            accelerator=MetalRuntime(device_id=999999),
            optimize=False,
            batch_size=64,
        )
    )
    assert_true(
        not candidate(
            query, 1, 64, optimize=False, streaming=False, original_source=True
        )
    )
    assert_true(
        not candidate(
            query, 1, 64, optimize=False, streaming=True, original_source=False
        )
    )
    _ = workers^
    var two = WorkerSetting(2)
    assert_true(
        not candidate(
            query, 2, 64, optimize=False, streaming=True, original_source=True
        )
    )
    _ = two^


# requires: metal
def test_public_float_stream_cpu_optimized_source_morsels_and_single_actual_job() raises:
    var workers = WorkerSetting(2)
    var frame = source[DType.float64](0, n=64)
    var query = (
        frame.lazy().group_by("k", maintain_order=True).agg(col("v").sum())
    )
    assert_equal(terminal_family(query), BASIC)
    for optimize in [False, True]:
        var route = capture(
            query,
            2,
            65536,
            optimize=optimize,
            streaming=True,
            original_source=True,
        )
        assert_equal(route.configured_workers, 2)
        assert_equal(route.jobs, 1)
        var expected_ranges: List[UInt32] = [0, 64]
        assert_equal(route.ranges, expected_ranges)
        var actual = query.collect(
            accelerator=MetalRuntime(), optimize=optimize, batch_size=65536
        )
        var expected = query.collect(
            engine="cpu", optimize=optimize, streaming=True, batch_size=65536
        )
        assert_true(actual.equals(expected))
    _ = workers^

    # An unused column changes alignment only when CPU optimization keeps
    # it. Capture must precede the narrower device value projection.
    var serial = WorkerSetting(1)
    var flush = FlushSetting(1)
    comptime for i in range(2):
        comptime D = DType.float32 if i == 0 else DType.float64
        var frame = source[D](0, n=128)
        var columns = List[Series]()
        for name in frame.columns():
            var series = frame.column(name)
            columns.append(
                Series._from_chunks([series.slice(0, 64), series.slice(64, 64)])
            )
        columns.append(
            Series("unused", Column[Float64](List[Float64](length=128, fill=0)))
        )
        frame = DataFrame(columns^)
        var query = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(col("v").sum().alias("s"))
        )
        for optimize in [False, True]:
            var route = capture(
                query,
                1,
                128,
                optimize=optimize,
                streaming=True,
                original_source=True,
            )
            var expected_ranges: List[UInt32] = [0, 128]
            if optimize:
                expected_ranges = [0, 64, 64, 64]
            assert_equal(route.ranges, expected_ranges)
            var actual = query.collect(
                accelerator=MetalRuntime(), optimize=optimize, batch_size=128
            )
            var cpu = query.collect(
                engine="cpu", streaming=True, optimize=optimize, batch_size=128
            )
            assert_true(actual.equals(cpu))
            var literal = Float64(0 if optimize else 1)
            for result in [actual.copy(), cpu.copy()]:
                for group in range(8):
                    assert_equal(
                        Float64(
                            result.column("s")
                            ._data[Column[Scalar[D]]]
                            ._get(group)
                        ),
                        literal,
                    )
    _ = flush^
    _ = serial^


# requires: metal
def test_public_float_stream_all_null_and_large_min_count() raises:
    var workers = WorkerSetting(1)
    comptime for i in range(2):
        comptime D = DType.float32 if i == 0 else DType.float64
        var frame = source[D](0).with_column(
            Series(
                "v",
                Column[Scalar[D]](
                    List[Scalar[D]](length=256, fill=Scalar[D](1)),
                    List[Bool](length=256, fill=False),
                ),
            )
        )
        for minimum in [0, 1, Int(UInt32.MAX) + 1, Int(Int64.MAX)]:
            var query = (
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg([col("v").sum(minimum), col("v").mean().alias("mean")])
            )
            var actual = query.collect(
                accelerator=MetalRuntime(), optimize=False, batch_size=64
            )
            var expected = query.collect(
                engine="cpu", optimize=False, streaming=True, batch_size=64
            )
            assert_true(actual.equals(expected))
            assert_equal(actual.column("mean").dtype(), DataType.FLOAT64)
            assert_equal(actual.column("mean").null_count(), 8)
            assert_equal(
                actual.column("v").null_count(), 0 if minimum == 0 else 8
            )
            if minimum == 0:
                for row in range(8):
                    var value = (
                        actual.column("v")._data[Column[Scalar[D]]]._get(row)
                    )
                    assert_equal(Float64(value), Float64(0))
    _ = workers^


def test_public_float_capture_geometry_and_construction_budget() raises:
    from dataframe.pipeline import _morsel_ranges
    from dataframe_metal.accelerator.grouped_stream_routes import (
        _capture_ranges,
    )
    from tests.support.float_streaming import capture_source

    for chunked in [False, True]:
        var frame = capture_source(chunked)
        for workers in [1, 2, 4]:
            for batch in [1, 3, 64, 65537]:
                var cpu = _morsel_ranges(frame, batch, workers)
                var captured = _capture_ranges(frame, batch, workers, -1)
                assert_equal(len(captured.values), len(cpu))
                for at in range(len(cpu)):
                    assert_equal(Int(captured.values[at]), cpu[at])
                assert_equal(captured.values.capacity(), len(cpu))
                var exact = _capture_ranges(
                    frame, batch, workers, captured.peak
                )
                assert_equal(len(exact.values), len(cpu))
                with assert_raises(contains="memory budget"):
                    _ = _capture_ranges(
                        frame, batch, workers, captured.peak - 1
                    )
    var frame = source[DType.float64](0)
    with assert_raises(contains="certified arithmetic bounds"):
        _ = _capture_ranges(frame, Int.MAX, 1, -1)
    with assert_raises(contains="certified arithmetic bounds"):
        _ = _capture_ranges(frame, 64, Int.MAX, -1)
    for workers in [1, 2, 4]:
        for batch in [1, 3, 64, 65537]:
            var cpu = _morsel_ranges(frame, batch, workers)
            var captured = _capture_ranges(frame, batch, workers, -1)
            assert_equal(len(captured.values), len(cpu))
            for at in range(len(cpu)):
                assert_equal(Int(captured.values[at]), cpu[at])


# requires: metal
def test_public_float_stream_extrema_exact_zero_nan_inf_and_null_bits() raises:
    var workers = WorkerSetting(1)
    comptime for i in range(2):
        comptime D = DType.float32 if i == 0 else DType.float64
        var sign = UInt64(0x80000000) if i == 0 else UInt64(0x8000000000000000)
        var nan = UInt64(0x7FC00000) if i == 0 else UInt64(0x7FF8000000000000)
        var inf = UInt64(0x7F800000) if i == 0 else UInt64(0x7FF0000000000000)
        for threshold in [4096, 1]:
            var flush = FlushSetting(threshold)
            for reverse in [False, True]:
                for mode in [1, 2]:
                    var frame = source[D](mode, reverse)
                    var query = (
                        frame.lazy()
                        .group_by("k", maintain_order=True)
                        .agg(
                            [
                                col("v").min().alias("mn"),
                                col("v").max().alias("mx"),
                            ]
                        )
                    )
                    var actual = query.collect(
                        accelerator=MetalRuntime(),
                        optimize=False,
                        batch_size=64,
                    )
                    var cpu = query.collect(
                        engine="cpu",
                        streaming=True,
                        optimize=False,
                        batch_size=64,
                    )
                    for name in ["mn", "mx"]:
                        ref a = actual.column(name)._data[Column[Scalar[D]]]
                        ref b = cpu.column(name)._data[Column[Scalar[D]]]
                        for group in range(8):
                            assert_equal(a._valid(group), b._valid(group))
                            if not a._valid(group):
                                continue
                            var bits = UInt64(0)
                            comptime if D == DType.float32:
                                bits = UInt64(
                                    bitcast[DType.uint32](a._get(group))
                                )
                                assert_equal(
                                    bits,
                                    UInt64(
                                        bitcast[DType.uint32](b._get(group))
                                    ),
                                )
                            else:
                                bits = bitcast[DType.uint64](a._get(group))
                                assert_equal(
                                    bits, bitcast[DType.uint64](b._get(group))
                                )
                            if mode == 1:
                                var negative = (
                                    not reverse if threshold
                                    == 4096 else reverse
                                )
                                assert_equal(
                                    bits, sign if negative else UInt64(0)
                                )
                            elif group == 1 or (group == 2 and name == "mx"):
                                assert_equal(bits, nan)
                            elif group == 4:
                                assert_equal(
                                    bits,
                                    inf | (sign if name == "mn" else UInt64(0)),
                                )
                        if mode == 2:
                            assert_true(not a._valid(3))
            _ = flush^
    _ = workers^


# requires: metal
def test_public_float_stream_eligibility_preserves_generic_main_routes() raises:
    var workers = WorkerSetting(1)
    var frame = source[DType.float64](0)
    var query = (
        frame.lazy().group_by("k", maintain_order=True).agg(col("v").sum())
    )
    assert_true(
        not candidate(
            query, 1, 64, optimize=False, streaming=False, original_source=True
        )
    )
    assert_true(
        not candidate(
            query, 1, 64, optimize=False, streaming=True, original_source=False
        )
    )
    assert_true(
        not candidate(
            query, 2, 64, optimize=False, streaming=True, original_source=True
        )
    )
    var computed = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg((col("v") * 2).sum())
    )
    assert_equal(terminal_family(computed), 0)
    var moments = (
        frame.lazy().group_by("k", maintain_order=True).agg(col("v").var())
    )
    assert_equal(terminal_family(moments), 0)
    var mixed = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg([col("v").sum(), col("v").var().alias("var")])
    )
    assert_equal(terminal_family(mixed), 0)
    var unordered = frame.lazy().group_by("k").agg(col("v").sum())
    assert_equal(terminal_family(unordered), 0)
    # Exact powers of two make the established eager route independent of
    # association, so this tests route preservation without new eager policy.
    var finite = DataFrame(
        [
            Series("k", Column[Int32](List[Int32](length=256, fill=7))),
            Series("v", Column[Float64](List[Float64](length=256, fill=2))),
        ]
    )
    var eager = (
        finite.lazy().group_by("k", maintain_order=True).agg(col("v").sum())
    )
    var actual = eager.profile(
        accelerator=MetalRuntime(), streaming=False, optimize=False
    )
    var expected = eager.collect(engine="cpu", streaming=False, optimize=False)
    assert_true(actual[0].equals(expected))
    assert_equal(actual[0].item(0, "v").float64(), Float64(512))
    var report = actual[1].copy()
    if "algorithm" in report.columns():
        for row in range(report.height()):
            assert_true(
                "single_worker_float_epochs"
                not in report.item(row, "algorithm").string()
            )
    _ = workers^


# requires: metal
def test_public_float_stream_null_negative_keys_independent_partition_flush_and_layout() raises:
    from dataframe.frame import _word_hash, _equality_words

    var workers = WorkerSetting(1)
    var domain = List[Int32](length=32, fill=0)
    var found = List[Int](length=16, fill=0)
    var chosen = 0
    for value in range(-1024, 1024):
        var key = Series("k", Column[Int32]([Int32(value)]))
        var words = _equality_words([key.copy()])
        var part = Int(_word_hash(words, 0) >> 60)
        if found[part] < 2:
            domain[2 * part + found[part]] = Int32(value)
            found[part] += 1
            chosen += 1
        if chosen == 32:
            break
    assert_equal(chosen, 32)
    for count in found:
        assert_equal(count, 2)
    comptime for i in range(2):
        comptime D = DType.float32 if i == 0 else DType.float64
        var keys = List[Int32](length=264, fill=0)
        var key_valid = List[Bool](length=264, fill=True)
        var values = List[Scalar[D]](length=264, fill=0)
        var valid = List[Bool](length=264, fill=False)
        for row in range(264):
            var group = row % 33
            var round = row // 33
            if group == 32:
                keys[row] = Int32.MIN if round % 2 else Int32.MAX
                key_valid[row] = False
            else:
                keys[row] = domain[group]
            if round % 2 == 0:
                valid[row] = True
                values[row] = (
                    Scalar[D](1152921504606846976.0) if round
                    == 0 else Scalar[D](-1152921504606846976.0) if round
                    == 4 else Scalar[D](1)
                )
            else:
                comptime if D == DType.float32:
                    values[row] = Scalar[D](
                        bitcast[DType.float32](UInt32(0x7F800123))
                    )
                else:
                    values[row] = Scalar[D](
                        bitcast[DType.float64](UInt64(0x7FF0000000000123))
                    )
        var frame = DataFrame(
            [
                Series("k", Column[Int32](keys^, key_valid^)),
                Series("v", Column[Scalar[D]](values^, valid^)),
            ]
        )
        var columns = List[Series]()
        for name in frame.columns():
            var series = frame.column(name)
            columns.append(
                Series._from_chunks(
                    [
                        series.slice(0, 66),
                        series.slice(66, 66),
                        series.slice(132, 66),
                        series.slice(198, 66),
                    ]
                )
            )
        frame = DataFrame(columns^)
        for threshold in [4096, 1]:
            var flush = FlushSetting(threshold)
            var query = (
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg(
                    [
                        col("v").sum().alias("s"),
                        col("v").mean().alias("m"),
                        col("v").min().alias("mn"),
                        col("v").max().alias("mx"),
                    ]
                )
            )
            var actual = query.collect(
                accelerator=MetalRuntime(), optimize=False, batch_size=66
            )
            var cpu = query.collect(
                engine="cpu", streaming=True, optimize=False, batch_size=66
            )
            assert_true(actual.equals(cpu))
            assert_equal(actual.height(), 33)
            assert_equal(actual.column("k").null_count(), 1)
            var total = Float64(0 if threshold == 1 else 1)
            for row in range(33):
                if row < 32:
                    assert_equal(
                        actual.column("k")._data[Column[Int32]]._get(row),
                        domain[row],
                    )
                else:
                    assert_true(
                        not actual.column("k")._data[Column[Int32]]._valid(row)
                    )
                for result in [actual.copy(), cpu.copy()]:
                    assert_equal(
                        Float64(
                            result.column("s")
                            ._data[Column[Scalar[D]]]
                            ._get(row)
                        ),
                        total,
                    )
                    assert_equal(
                        result.column("m")._data[Column[Float64]]._get(row),
                        total / 4,
                    )
                    assert_equal(
                        Float64(
                            result.column("mn")
                            ._data[Column[Scalar[D]]]
                            ._get(row)
                        ),
                        Float64(-1152921504606846976.0),
                    )
                    assert_equal(
                        Float64(
                            result.column("mx")
                            ._data[Column[Scalar[D]]]
                            ._get(row)
                        ),
                        Float64(1152921504606846976.0),
                    )
            _ = flush^
    _ = workers^


# requires: metal
def test_public_float_stream_large_source_original_pipeline_boundary() raises:
    var workers = WorkerSetting(1)
    var frame = source[DType.float64](0, n=32769)
    var query = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                col("v").sum().alias("s"),
                col("v").mean().alias("m"),
                col("v").min().alias("mn"),
                col("v").max().alias("mx"),
            ]
        )
    )
    for threshold in [4096, 1]:
        var flush = FlushSetting(threshold)
        var actual = query.collect(
            accelerator=MetalRuntime(), optimize=False, batch_size=64
        )
        var cpu = query.collect(
            engine="cpu", streaming=True, optimize=False, batch_size=64
        )
        assert_true(actual.equals(cpu))
        _ = flush^
    _ = workers^


# requires: metal
def test_public_float_stream_collected_density_preserves_generic_gpu_behavior_and_budget() raises:
    var workers = WorkerSetting(1)
    comptime for i in range(2):
        comptime D = DType.float32 if i == 0 else DType.float64
        var keys = List[Int32](length=128, fill=0)
        for row in range(128):
            keys[row] = Int32(row % 64)
        var frame = DataFrame(
            [
                Series("k", Column[Int32](keys^)),
                Series(
                    "v",
                    Column[Scalar[D]](
                        List[Scalar[D]](length=128, fill=Scalar[D](2))
                    ),
                ),
            ]
        )
        for chunked in [False, True]:
            var input = frame.copy()
            if chunked:
                var columns = List[Series]()
                for name in frame.columns():
                    var series = frame.column(name)
                    var parts = List[Series]()
                    for row in range(128):
                        parts.append(series.slice(row, 1))
                    columns.append(Series._from_chunks(parts))
                input = DataFrame(columns^)
            for only_sum in [False, True]:
                var expressions: List[Expr] = [col("v").sum().alias("s")]
                if not only_sum:
                    expressions.append(col("v").mean().alias("m"))
                    expressions.append(col("v").min().alias("mn"))
                    expressions.append(col("v").max().alias("mx"))
                var query = (
                    input.lazy()
                    .group_by("k", maintain_order=True)
                    .agg(expressions)
                )
                var baseline = query.profile(
                    accelerator=MetalRuntime(),
                    streaming=False,
                    optimize=False,
                    batch_size=64,
                )
                var baseline_report = baseline[1].copy()
                var peak_name = (
                    "query_peak_requested_bytes" if "query_peak_requested_bytes"
                    in baseline_report.columns() else "peak_requested_bytes"
                )
                var generic_peak = Int(
                    baseline_report.item(0, peak_name).int64()
                )
                for limit in [-1, generic_peak]:
                    var measured = query.profile(
                        accelerator=MetalRuntime(memory_limit_bytes=limit),
                        streaming=True,
                        optimize=False,
                        batch_size=64,
                    )
                    var actual = measured[0].copy()
                    var report = measured[1].copy()
                    assert_true(actual.equals(baseline[0]))
                    assert_equal(actual.height(), 64)
                    assert_equal(
                        report.item(0, "algorithm").string(),
                        "mojo_gpu_collected_float_groups",
                    )
                    assert_true(
                        report.item(0, "query_peak_requested_bytes").int64()
                        <= Int64(generic_peak)
                    )
                    assert_true(
                        report.item(0, "query_upload_bytes").int64() > 0
                    )
                    assert_true(
                        report.item(0, "query_download_bytes").int64() >= 4
                    )
                    assert_equal(
                        report.item(0, "kernel_launches").int64(),
                        report.item(0, "query_kernel_launches").int64(),
                    )
                    assert_equal(
                        report.item(0, "synchronizations").int64(),
                        report.item(0, "query_synchronizations").int64(),
                    )
                    for row in range(64):
                        assert_equal(
                            actual.column("k")._data[Column[Int32]]._get(row),
                            Int32(row),
                        )
                        assert_equal(
                            Float64(
                                actual.column("s")
                                ._data[Column[Scalar[D]]]
                                ._get(row)
                            ),
                            Float64(4),
                        )
                        if not only_sum:
                            assert_equal(
                                actual.column("m")
                                ._data[Column[Float64]]
                                ._get(row),
                                Float64(2),
                            )
                            assert_equal(
                                Float64(
                                    actual.column("mn")
                                    ._data[Column[Scalar[D]]]
                                    ._get(row)
                                ),
                                Float64(2),
                            )
                            assert_equal(
                                Float64(
                                    actual.column("mx")
                                    ._data[Column[Scalar[D]]]
                                    ._get(row)
                                ),
                                Float64(2),
                            )
    _ = workers^
