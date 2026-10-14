"""Direct terminal ROWS faults follow projection, batch, AST and row order."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe.expr import ADD, COL, WHEN, _node, _append_shifted
from dataframe.metal import MetalRuntime
from dataframe.parallel import worker_count
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.coverage_generators import offset_layout, chunk_layout
from dataframe.accelerator.gpu.runtime import _preflight, estimate_resources
from dataframe.accelerator.gpu.calibration import estimate


def _source(ticks: List[Int64], integers: List[Int64]) raises -> DataFrame:
    return DataFrame(
        [
            Series("t", Column[Int64](ticks.copy())).with_dtype(
                DataType.duration("ms")
            ),
            Series("x", Column[Int64](integers.copy())),
        ]
    )


def _time() raises -> Expr:
    return col("t").cast(DataType.duration("ns")).cast(DataType.INT64)


def _cast() -> Expr:
    return col("x").cast(DataType.INT8).cast(DataType.INT64)


def _fails(query: LazyFrame, cpu: String, gpu: String) raises:
    var plan = _preflight(query, ordered_source=True)
    assert_equal(plan.ordered_mode, 2)
    with assert_raises(contains=cpu):
        _ = query.collect(engine="cpu", optimize=False, streaming=False)
    with assert_raises(contains=gpu):
        _ = query.collect(
            accelerator=MetalRuntime(), engine="accel", optimize=False
        )


def test_mojo_gpu_ordered_rows_cross_row_batch_projection_and_workers() raises:
    var raw = _source([0, Int64.MAX], [128, 0])
    for variant in range(3):
        var frame = (
            raw.copy() if variant
            == 0 else offset_layout(raw) if variant
            == 1 else chunk_layout(raw)
        )
        _fails(
            frame.lazy().select(_time() + _cast()),
            "temporal value overflows Int64",
            "temporal value overflows Int64",
        )
        _fails(
            frame.lazy().select(_cast() + _time()),
            "cast from int64 to int8 failed",
            "strict cast failed",
        )
    # Both first and later batches contain faults, with opposite AST priority.
    var ticks = List[Int64](length=8193, fill=0)
    var integers = List[Int64](length=8193, fill=0)
    ticks[8192] = Int64.MAX
    integers[8191] = 128
    var frame = _source(ticks, integers)
    _fails(
        frame.lazy().select(_time() + _cast()),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )
    _fails(
        frame.lazy().select_exprs(
            [_time().alias("first"), _cast().alias("second")]
        ),
        "temporal value overflows Int64",
        "temporal value overflows Int64",
    )
    _fails(
        frame.lazy().select_exprs(
            [_cast().alias("first"), _time().alias("second")]
        ),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )
    # Actual CPU _RowsJob path: contiguous batch-aligned jobs, with a failure
    # in the first job and a different category in the last (caller) job.
    var n = 262145
    ticks = List[Int64](length=n, fill=0)
    integers = List[Int64](length=n, fill=0)
    ticks[65535] = Int64.MAX
    integers[n - 1] = 128
    frame = _source(ticks, integers)
    for threads in [1, 2, 4, 8]:
        var setting = WorkerSetting(threads)
        assert_equal(worker_count(n), min(threads, 4))
        _fails(
            frame.lazy().select(_cast() + _time()),
            "temporal value overflows Int64",
            "temporal value overflows Int64",
        )
        _ = setting^


def test_mojo_gpu_ordered_rows_masks_nulls_empty_and_mixed_numeric_faults() raises:
    var frame = _source([0, Int64.MAX], [128, 0]).with_column(
        Series("keep", Column[Bool]([False, True]))
    )
    # CPU evaluates then over its active rows before else, even when the
    # earlier logical row is selected by else.
    _fails(
        frame.lazy().select(when(col("keep")).then(_time()).otherwise(_cast())),
        "temporal value overflows Int64",
        "temporal value overflows Int64",
    )
    var predicate = col("x").cast(DataType.INT8) > lit(Int8(0))
    _fails(
        frame.lazy().select(when(predicate).then(_time()).otherwise(col("x"))),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )
    frame = frame.with_column(Series("keep", Column[Bool]([False, False])))
    var masked = frame.lazy().select(
        when(col("keep")).then(_time() + _cast()).otherwise(col("x")).alias("v")
    )
    var actual = masked.collect(accelerator=MetalRuntime())
    assert_equal(actual.item(0, "v").int64(), Int64(128))
    assert_equal(actual.item(1, "v").int64(), Int64(0))
    assert_true(
        actual.equals(
            masked.collect(engine="cpu", streaming=False, optimize=False)
        )
    )
    var empty = frame.clear().lazy().select((_time() + _cast()).alias("v"))
    assert_equal(_preflight(empty, ordered_source=True).ordered_mode, 2)
    assert_equal(empty.collect(accelerator=MetalRuntime()).height(), 0)
    assert_equal(empty.collect(engine="cpu", streaming=False).height(), 0)
    var numbers = DataFrame(
        [
            Series("x", Column[Int64]([128])),
            Series("base", Column[Int64]([2])),
            Series("exp", Column[Int64]([-1])),
        ]
    )
    var power = col("base").pow(col("exp"))
    _fails(
        numbers.lazy().select(_cast() + power),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )
    _fails(
        numbers.lazy().select(power + _cast()),
        "nonnegative exponent",
        "nonnegative exponent",
    )
    # Reusing one checked child under differently masked occurrences must not
    # give a hidden earlier visit's rank to the observed later visit.
    var shared = _time()
    var nodes = shared._nodes.copy()
    var time_root = len(nodes) - 1
    nodes.append(_node(COL, text="keep"))
    var predicate_root = len(nodes) - 1
    nodes.append(_node(COL, text="x"))
    var fallback_root = len(nodes) - 1
    nodes.append(_node(WHEN, predicate_root, time_root, extra=fallback_root))
    var hidden_root = len(nodes) - 1
    var strict = _cast()
    _append_shifted(nodes, strict._nodes, len(nodes))
    var cast_root = len(nodes) - 1
    nodes.append(_node(ADD, hidden_root, cast_root))
    var first_root = len(nodes) - 1
    nodes.append(_node(ADD, first_root, time_root))
    var reused = Expr(nodes^, "v")
    _fails(
        frame.lazy().select(reused),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )
    # A single POW already observes rows in order on the legacy route;
    # separate cells are not OR-reduced before category decoding.
    for negative_first in [False, True]:
        var exponents: List[Int64] = [-1, 63] if negative_first else [63, -1]
        var single = (
            DataFrame(
                [
                    Series("base", Column[Int64]([2, 2])),
                    Series("exp", Column[Int64](exponents^)),
                ]
            )
            .lazy()
            .select(col("base").pow(col("exp")))
        )
        assert_equal(_preflight(single, ordered_source=True).ordered_mode, 0)
        var anchor = "nonnegative exponent" if negative_first else "overflow"
        with assert_raises(contains=anchor):
            _ = single.collect(engine="cpu", optimize=False, streaming=False)
        with assert_raises(contains=anchor):
            _ = single.collect(accelerator=MetalRuntime(), optimize=False)
    var division = (
        DataFrame(
            [
                Series("n", Column[Int64]([1, Int64.MIN])),
                Series("d", Column[Int64]([0, -1])),
            ]
        )
        .lazy()
        .select(col("n") // col("d"))
    )
    assert_equal(_preflight(division, ordered_source=True).ordered_mode, 0)
    with assert_raises(contains="overflow"):
        _ = division.collect(engine="cpu", streaming=False, optimize=False)
    with assert_raises(contains="overflow"):
        _ = division.collect(accelerator=MetalRuntime(), optimize=False)
    # Paired Decimal128 faults retain their own category and AST priority.
    var huge = Int128(9)
    for _ in range(37):
        huge *= Int128(10)
    var decimal_type = DataType.decimal(38, 0, 128)
    var decimals = DataFrame(
        [
            Series("d", Column[Int128]([huge])).with_dtype(decimal_type),
            Series("two", Column[Int128]([Int128(2)])).with_dtype(decimal_type),
            Series("x", Column[Int64]([128])),
        ]
    )
    var product = (col("d") * col("two")).cast(DataType.INT64)
    _fails(
        decimals.lazy().select(_cast() + product),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )
    _fails(
        decimals.lazy().select(product + _cast()),
        "decimal multiplication overflow",
        "decimal multiplication overflow",
    )


def test_mojo_gpu_ordered_rows_profile_budget_estimate_and_provenance() raises:
    var frame = _source([-1, 0, 1], [-2, 0, 2])
    var query = frame.lazy().select((_time() + _cast()).alias("v"))
    var plan = _preflight(query, ordered_source=True)
    var facts = estimate_resources(plan)
    var analytical = estimate(query)
    assert_true(analytical.family.find("ordered-errors-v1=2") >= 0)
    assert_equal(analytical.peak_bytes, facts.peak_bytes)
    var described = MetalRuntime().describe(query)
    assert_true(described.find("ordered projection/batch/AST faults") >= 0)
    var result = query.collect(
        accelerator=MetalRuntime(memory_limit_bytes=facts.peak_bytes)
    )
    var measured = query.profile(accelerator=MetalRuntime())
    assert_equal(
        measured[1].item(0, "peak_requested_bytes").int64(),
        Int64(facts.peak_bytes),
    )
    assert_true(
        measured[1].item(0, "kernel_launches").int64()
        <= Int64(facts.kernel_launches)
    )
    assert_true(
        measured[1].item(0, "synchronizations").int64()
        <= Int64(facts.synchronizations)
    )
    assert_true(
        measured[1].item(0, "upload_bytes").int64() <= Int64(facts.upload_bytes)
    )
    assert_true(
        measured[1].item(0, "download_bytes").int64()
        <= Int64(facts.download_bytes)
    )
    var expected: List[Int64] = [-1000002, 0, 1000002]
    for row in range(3):
        assert_equal(result.item(row, "v").int64(), expected[row])
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=facts.peak_bytes - 1)
        )
    # No foreign error scope may reach the new mode through synthetic scans.
    with assert_raises(contains="direct scan"):
        _ = _preflight(
            frame.lazy()
            .with_columns(col("t").alias("alias"))
            .select(
                col("alias").cast(DataType.duration("ns")).cast(DataType.INT64)
                + _cast()
            ),
            ordered_source=True,
        )
    with assert_raises(contains="direct scan"):
        _ = _preflight(query, ordered_source=True, source_rows=3)
