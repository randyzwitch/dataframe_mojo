"""Independent integer SUM expectations over bounded device-resident state."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout


def _sums(frame: DataFrame, expression: Expr, outputs: Int = 16, min_count: Int = 0) -> LazyFrame:
    var expressions = List[Expr]()
    for index in range(outputs):
        expressions.append(expression.sum(min_count=min_count).alias("s" + String(index)))
    return frame.lazy().select_exprs(expressions)


def _check(query: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
    var runtime = MetalRuntime()
    var description = runtime.describe(query)
    if "native wide integer SUM" not in description:
        print("Unexpected wide SUM description:", description)
    assert_true("native wide integer SUM" in description)
    var expected = query.collect(engine="cpu", optimize=False, streaming=False)
    var measured = query.profile(engine="accel", accelerator=runtime)
    assert_true(measured[0].equals(expected))
    assert_equal(measured[1].item(0, "algorithm").string(), "mojo_gpu_native_wide_integer_sum")
    assert_true(query.collect(engine="accel", accelerator=runtime).equals(expected))
    return measured^


# requires: mojo_gpu
def test_wide_integer_sum_exact_cancellation_across_partials_and_outputs() raises:
    for boundary in [256, 4096, 65536]:
        var values = List[Int64](length=2 * boundary + 1, fill=Int64.MAX)
        for row in range(boundary, 2 * boundary):
            values[row] = -Int64.MAX
        values[2 * boundary] = 7
        var frame = DataFrame([Series("value", Column[Int64](values^))])
        for source in [frame.copy(), offset_layout(frame), chunk_layout(frame)]:
            var measured = _check(_sums(source, col("value")))
            assert_equal(measured[0].shape(), (1, 16))
            for index in range(16):
                var name = "s" + String(index)
                assert_equal(measured[0].column(name).dtype(), DataType.INT64)
                assert_equal(measured[0].item(0, name).int64(), Int64(7))
    var endpoints = DataFrame([Series("value", Column[Int64]([Int64.MAX, Int64.MIN]))])
    var result = _check(_sums(endpoints, col("value"), 1))
    assert_equal(result[0].item(0, "s0").int64(), Int64(-1))


# requires: mojo_gpu
def test_wide_integer_sum_unsigned_endpoints_nulls_empty_and_min_count() raises:
    var values = List[UInt64](length=4097, fill=0)
    var validity = List[Bool](length=4097, fill=False)
    values[0] = UInt64.MAX
    validity[0] = True
    var frame = DataFrame([Series("value", Column[UInt64](values^, validity^))])
    for source in [frame.copy(), offset_layout(frame), chunk_layout(frame)]:
        var measured = _check(_sums(source, col("value")))
        for index in range(16):
            var name = "s" + String(index)
            assert_equal(measured[0].column(name).dtype(), DataType.UINT64)
            assert_equal(measured[0].item(0, name).uint64(), UInt64.MAX)
        var absent = _check(_sums(source, col("value"), 2, 2))
        assert_true(absent[0].item(0, "s0").is_null())
    for dtype in [DataType.INT64, DataType.UINT64]:
        var nulls = DataFrame([Series.full_null("value", dtype, 4097)])
        for source in [nulls.copy(), nulls.clear()]:
            var zero = _check(_sums(source, col("value"), 2))
            assert_true(not zero[0].item(0, "s0").is_null())
            if dtype == DataType.INT64:
                assert_equal(zero[0].item(0, "s0").int64(), Int64(0))
            else:
                assert_equal(zero[0].item(0, "s0").uint64(), UInt64(0))
            var absent = _check(_sums(source, col("value"), 2, 1))
            assert_true(absent[0].item(0, "s0").is_null())
    # A null result suppresses final narrowing, while row arithmetic remains
    # independently checked. The exact wide total here exceeds UInt64.
    var overflow = DataFrame([Series("value", Column[UInt64]([UInt64.MAX, 1]))])
    var suppressed = _check(_sums(overflow, col("value"), 1, 3))
    assert_true(suppressed[0].item(0, "s0").is_null())


def _overflow(query: LazyFrame) raises:
    var cpu_error = String()
    try:
        _ = query.collect(engine="cpu", optimize=False, streaming=False)
    except error:
        cpu_error = String(error)
    assert_true("overflow" in cpu_error)
    var gpu_error = String()
    try:
        _ = query.collect(engine="accel", accelerator=MetalRuntime(), optimize=False, streaming=False)
    except error:
        gpu_error = String(error)
    assert_true(cpu_error in gpu_error)


# requires: mojo_gpu
def test_wide_integer_sum_final_overflow_and_late_checked_row_faults() raises:
    for values in [List[Int64]([Int64.MAX, 1]), List[Int64]([Int64.MIN, -1])]:
        var frame = DataFrame([Series("value", Column[Int64](values.copy()))])
        _overflow(_sums(frame, col("value")))
    var unsigned = DataFrame([Series("value", Column[UInt64]([UInt64.MAX, 1]))])
    _overflow(_sums(unsigned, col("value")))
    _overflow(_sums(unsigned, col("value") + lit(UInt64(1))))
    _overflow(_sums(unsigned, col("value") * lit(UInt64(2))))
    var minimum = DataFrame([Series("value", Column[Int64]([Int64.MIN]))])
    _overflow(_sums(minimum, col("value") * lit(Int64(-1))))
    var values = List[Int64](length=65537, fill=0)
    values[65536] = Int64.MAX
    var late = DataFrame([Series("value", Column[Int64](values^))])
    _overflow(_sums(late, col("value") + lit(Int64(1))))
    _overflow(_sums(late, col("value") * lit(Int64(2))))
    # Distinct output constants exercise the same bounded partial set without
    # retaining one whole-row intermediate per scalar output.
    var frame = DataFrame([Series("value", Column[Int64]([-2, 0, 5]))])
    var query = frame.lazy().select_exprs([
        col("value").sum().alias("a"),
        (col("value") + lit(Int64(2))).sum().alias("b"),
        (col("value") * lit(Int64(-3))).sum().alias("c"),
        ((col("value") + lit(Int64(1))) * lit(Int64(2))).sum().alias("d"),
    ])
    var measured = _check(query)
    for pair in List[Tuple[String, Int64]]([("a", 3), ("b", 9), ("c", -9), ("d", 12)]):
        assert_equal(measured[0].item(0, pair[0]).int64(), pair[1])


# requires: mojo_gpu
def test_wide_integer_sum_bounded_state_budget_and_reuse() raises:
    var rows = 1048577
    var values = List[Int64](length=rows, fill=1)
    var frame = DataFrame([Series("value", Column[Int64](values^))])
    var query = _sums(frame, col("value"), 2)
    var runtime = MetalRuntime()
    var first = query.profile(engine="accel", accelerator=runtime)
    assert_equal(first[0].item(0, "s0").int64(), Int64(rows))
    assert_equal(first[1].item(0, "reduction_partials").int64(), Int64(4096))
    assert_equal(first[1].item(0, "download_bytes").int64(), Int64(48))
    assert_equal(first[1].item(0, "kernel_launches").int64(), Int64(4))
    var peak = Int(first[1].item(0, "peak_requested_bytes").int64())
    assert_equal(peak, rows * 8 + 4096 * 24 + 2 * 57)
    var bounded = runtime.copy()
    bounded._budget = peak - 1
    with assert_raises(contains="memory budget"):
        _ = query.collect(engine="accel", accelerator=bounded)
    bounded._budget = peak
    var second = query.profile(engine="accel", accelerator=bounded)
    assert_true(second[0].equals(first[0]))
    assert_equal(second[1].item(0, "service_scratch_hits").int64(), Int64(5))
