"""Paired Float64 covariance, Pearson and Spearman through resident Metal."""
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    Series,
    StringColumn,
    Column,
    Expr,
    LazyFrame,
    DataType,
    col,
    lit,
    when,
    cov,
    corr,
)
from dataframe.metal import MetalRuntime, _capabilities, _Descriptors
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.paired_moments import lower_spearman
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_moments import near
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.public_contracts.test_pair_reductions import (
    _numeric_columns,
    _expressions,
    _assert_pairs,
)
from tests.support.pair_reduction_reference import (
    PairReference,
    centered_comoment,
)
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.execution_modes.test_backend_metal_decimal_casts import (
    fixture as decimal_fixture,
)


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    near(
        query.collect(accelerator=runtime),
        query.collect(engine="cpu", streaming=False),
    )


def test_pair_moment_capability_right_slots_ddof_and_spearman_staging() raises:
    assert_true(not RowCapabilities().paired_moments)
    var setting = WorkerSetting(8)
    var frame = DataFrame(
        [
            Series("x", Column[UInt64]([0, UInt64.MAX, UInt64.MAX - 1024])),
            Series("y", Column[Float32]([1, 2, 3])),
            Series("k", Column[Int32]([0, 1, 0])),
        ]
    )
    for ddof in List[Int]([0, 1, Int(Int64.MAX)]):
        var query = frame.lazy().select_exprs(
            [
                cov(col("x"), col("y"), ddof).alias("c"),
                corr(col("y"), col("x")).alias("r"),
            ]
        )
        var caps = _capabilities()
        caps.paired_moments = False
        with assert_raises(
            contains="covariance and correlation require backend support"
        ):
            _ = lower_rows(query, caps)
        var plan = lower_rows(query, _capabilities())
        for output in plan.outputs:
            assert_equal(plan.slot_dtypes[output.slot], DataType.FLOAT64)
            var right = (output.min_count >> 31) & 65535
            assert_equal(plan.slot_dtypes[right], DataType.FLOAT64)
            assert_equal(right, output.slot + 1)
            assert_equal(output.min_count >> 47, 8)
            assert_equal(
                output.min_count & 0x7FFFFFFF,
                min(ddof, Int(Int32.MAX)) if output.reduction == 161 else 0,
            )
        _ = _Descriptors(plan)
        assert_true(
            "unsupported" not in MetalRuntime(device_id=999999).describe(query)
        )
    with assert_raises(contains="explicit Float64 cast"):
        _ = lower_rows(
            decimal_fixture(3, 2).lazy().select(cov(col("small"), col("k"))),
            _capabilities(),
        )
    with assert_raises(contains="explicit Float64 cast"):
        _ = lower_rows(
            decimal_fixture(3, 2).lazy().select(corr(col("k"), col("wide"))),
            _capabilities(),
        )
    with assert_raises(contains="ddof"):
        _ = lower_rows(
            frame.lazy().select(cov(col("x"), col("y"), -1)), _capabilities()
        )
    var spearman = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(corr(col("x"), col("y"), "spearman"))
    )
    near(
        lower_spearman(spearman).collect(engine="cpu"),
        spearman.collect(engine="cpu"),
    )
    assert_true(
        "unsupported" not in MetalRuntime(device_id=999999).describe(spearman)
    )
    _ = setting^


def test_native_metal_pair_moments_mixed_widths_independent_pairs_and_layouts() raises:
    _ = require_metal()
    var setting = WorkerSetting(1)
    var runtime = MetalRuntime()
    var xs: List[Float64] = [3, 1, 3, 2, 1, 9]
    var ys: List[Float64] = [4, 1, 4, 3, 2, 8]
    for mode in range(4):
        var xv = List[Bool](length=6, fill=mode < 2)
        var yv = List[Bool](length=6, fill=mode < 3)
        if mode == 1:
            xv = [True, False, True, True, False, True]
            yv = [False, True, True, True, True, False]
        if mode == 3:
            xv[0] = True
            yv[0] = True
        var reference = PairReference(xs, ys, xv, yv, [0, 1, 2, 3, 4, 5])
        for x in _numeric_columns("x", xs, xv):
            # Both directions exercise every width with a different storage.
            for reverse in range(2):
                var columns = List[Series]()
                if reverse:
                    columns.append(
                        Series("x", Column[Float32]([4, 1, 4, 3, 2, 8], yv))
                    )
                    var other = x.copy()
                    other = other.renamed("y")
                    columns.append(other^)
                else:
                    columns.append(x.copy())
                    columns.append(Series("y", Column[Float64](ys.copy(), yv)))
                columns.append(Series("g", Column[Int64]([0, 1, 0, 1, 0, 1])))
                var raw = DataFrame(columns^)
                for layout in range(3):
                    var frame = (
                        raw.copy() if layout
                        == 0 else offset_layout(raw) if layout
                        == 1 else chunk_layout(raw).rechunk()
                    )
                    var actual = (
                        frame.lazy()
                        .select_exprs(_expressions())
                        .collect(accelerator=runtime)
                    )
                    _assert_pairs(actual, 0, reference)
                    check(
                        frame.lazy()
                        .group_by("g", maintain_order=True)
                        .agg(_expressions()),
                        runtime,
                    )
                    check(
                        frame.lazy().select_exprs(
                            [
                                cov(col("x"), col("y")).over("g").alias("c"),
                                corr(col("x"), col("y")).over("g").alias("p"),
                                corr(col("x"), col("y"), "spearman")
                                .over("g")
                                .alias("s"),
                            ]
                        ),
                        runtime,
                    )
    var anchors = DataFrame(
        [
            Series("x", Column[Int32]([1, 2, 3])),
            Series("y", Column[Int64]([2, 4, 6])),
        ]
    )
    same(
        anchors.lazy()
        .select_exprs(_expressions())
        .collect(accelerator=runtime),
        DataFrame(
            [
                Series("pearson", Column[Float64]([1])),
                Series("spearman", Column[Float64]([1])),
                Series("cov0", Column[Float64]([Float64(4) / 3])),
                Series("cov1", Column[Float64]([2])),
                Series("cov6", Column[Float64]([0])),
            ]
        ),
    )
    _ = setting^


def test_native_metal_pair_moments_ieee_null_pairs_unsigned_and_wide_physical_slots() raises:
    _ = require_metal()
    var setting = WorkerSetting(1)
    var runtime = MetalRuntime()
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        1,
        0x8000000000000001,
        0x0010000000000000,
        0x3FF0000000000001,
        0x7FEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000042,
    ]
    for word in words:
        for paired in List[Bool]([False, True]):
            var frame = DataFrame(
                [
                    Series(
                        "x",
                        Column[Float64](
                            [bitcast[DType.float64](word), 1, 1, 3, 4, 4],
                            [True, True, True, False, True, True],
                        ),
                    ),
                    Series(
                        "y",
                        Column[Float64](
                            [2, 5, 4, 3, 2, 1],
                            [paired, True, False, True, True, True],
                        ),
                    ),
                    Series("g", Column[Int32]([0, 1, 0, 1, 0, 1])),
                ]
            )
            check(frame.lazy().select_exprs(_expressions()), runtime)
            check(
                frame.lazy()
                .group_by("g", maintain_order=True)
                .agg(_expressions()),
                runtime,
            )
            for rows in List[Int]([0, 1]):
                check(
                    frame.head(rows).lazy().select_exprs(_expressions()),
                    runtime,
                )
    var unsigned = DataFrame(
        [
            Series(
                "x",
                Column[UInt64](
                    [
                        UInt64.MAX,
                        UInt64.MAX - 1024,
                        UInt64.MAX - 4096,
                        UInt64.MAX - 8192,
                    ]
                ),
            ),
            Series(
                "y",
                Column[Int64](
                    [
                        Int64.MAX,
                        Int64.MAX - 1024,
                        Int64.MAX - 4096,
                        Int64.MAX - 8192,
                    ]
                ),
            ),
        ]
    )
    check(unsigned.lazy().select_exprs(_expressions()), runtime)
    var columns = List[Series]()
    for i in range(128):
        columns.append(
            Series(
                "wide" + String(i),
                Column[Int128](
                    [Int128(1) << 100, -Int128(1) << 100, Int128(0), Int128(1)]
                ),
            ).with_dtype(DataType.decimal(38, 0))
        )
    columns.append(Series("x", Column[Int64]([1, 2, 3, 4])))
    columns.append(Series("y", Column[Float64]([4, 1, 3, 2])))
    columns.append(Series("g", Column[Int32]([0, 1, 0, 1])))
    var wide = DataFrame(columns^)
    # Logical right slots are below 256; physical slots exceed 255 because
    # every unused Decimal128 input occupies two GPU words.
    check(wide.lazy().select_exprs(_expressions()), runtime)
    check(
        wide.lazy()
        .filter(col("g") == 0)
        .group_by("g", maintain_order=True)
        .agg(_expressions()),
        runtime,
    )
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000042))
    var partitions = DataFrame(
        [
            Series("x", Column[Float64]([1, 2, 2, 4, 5, 6, 7, 8])),
            Series(
                "y",
                Column[Int64](
                    [8, 7, 6, 5, 4, 3, 2, 1],
                    [True, False, True, True, True, True, True, True],
                ),
            ),
            Series(
                "text",
                StringColumn(
                    ["a", "b", "a", "b", "a", "b", "a", "b"],
                    [True, True, False, True, True, True, True, True],
                ),
            ),
            Series(
                "floating",
                Column[Float64](
                    [nan, 1, nan, 1, nan, 1, nan, 1],
                    [True, True, True, True, True, False, True, True],
                ),
            ),
        ]
    )
    check(
        partitions.lazy()
        .group_by(["text", "floating"], maintain_order=True)
        .agg(_expressions()),
        runtime,
    )
    check(
        partitions.lazy().select_exprs(
            [
                cov(col("x"), col("y")).over(["text", "floating"]).alias("c"),
                corr(col("x"), col("y"), "spearman")
                .over(["text", "floating"])
                .alias("s"),
            ]
        ),
        runtime,
    )
    for scale in List[Int]([0, 2, 9]):
        var decimals = decimal_fixture(13, scale)
        check(
            decimals.lazy().select_exprs(
                [
                    cov(
                        col("wide").cast(DataType.FLOAT64),
                        col("small").cast(DataType.FLOAT64),
                    ).alias("c"),
                    corr(
                        col("large").cast(DataType.FLOAT64),
                        col("wide").cast(DataType.FLOAT64),
                        "spearman",
                    ).alias("s"),
                ]
            ),
            runtime,
        )
    _ = setting^


def test_native_metal_pair_moments_large_workers_centered_accuracy_and_spearman_ties() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    for workers in List[Int]([1, 2, 8]):
        var setting = WorkerSetting(workers)
        for rows in List[Int]([65537, 131073]):
            var xs = List[Float64]()
            var ys = List[Float64]()
            var residual_x = List[Float64]()
            var residual_y = List[Float64]()
            var valid = List[Bool]()
            var keys = List[Int32]()
            for i in range(rows):
                var x = Float64(i % 17) / 8
                var y = Float64(i % 17) / 4 + Float64(i % 3) / 16
                xs.append(Float64(1048576) + x)
                ys.append(Float64(2097152) + y)
                valid.append(i % 7 != 0)
                keys.append(Int32(i % 3))
                if i % 7 != 0:
                    residual_x.append(x)
                    residual_y.append(y)
            var frame = DataFrame(
                [
                    Series("x", Column[Float64](xs^, valid)),
                    Series("y", Column[Float64](ys^, valid^)),
                    Series("g", Column[Int32](keys^)),
                ]
            )
            var query = frame.lazy().select_exprs(
                [
                    cov(col("x"), col("y")).alias("c"),
                    corr(col("x"), col("y")).alias("r"),
                ]
            )
            var plan = lower_rows(query, _capabilities())
            for output in plan.outputs:
                assert_equal(output.min_count >> 47, workers)
            var actual = query.collect(accelerator=runtime)
            var centered = centered_comoment(residual_x, residual_y)
            var expected_cov = centered[2] / Float64(len(residual_x) - 1)
            assert_true(
                abs(actual.column("c").float64()._get(0) - expected_cov)
                < Float64(1e-9)
            )
            var cpu = query.collect(engine="cpu")
            for name in List[String](["c", "r"]):
                assert_true(
                    abs(
                        actual.column(name).float64()._get(0)
                        - cpu.column(name).float64()._get(0)
                    )
                    < Float64(1e-9)
                )
            var window = frame.lazy().select_exprs(
                [
                    cov(col("x"), col("y")).over("g").alias("c"),
                    corr(col("x"), col("y")).over("g").alias("r"),
                ]
            )
            var a = window.collect(accelerator=runtime)
            var b = window.collect(engine="cpu")
            for i in range(rows):
                for name in List[String](["c", "r"]):
                    assert_true(
                        abs(
                            a.column(name).float64()._get(i)
                            - b.column(name).float64()._get(i)
                        )
                        < Float64(1e-9)
                    )
            if workers == 8 and rows == 65537:
                check(
                    frame.lazy().select(corr(col("x"), col("y"), "spearman")),
                    runtime,
                )
                check(
                    frame.lazy().select(
                        corr(col("x"), col("y"), "spearman").over("g")
                    ),
                    runtime,
                )
        _ = setting^


def test_native_metal_pair_moments_nested_consumers_checked_inputs_budget_cost_and_reuse() raises:
    _ = require_metal()
    var setting = WorkerSetting(1)
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Int32](
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [True, False, True, True, True, True, False, True],
                ),
            ),
            Series(
                "y",
                Column[Float64](
                    [3, 2, 1, 4, 8, 6, 7, 5],
                    [True, True, True, False, True, True, True, True],
                ),
            ),
            Series("g", Column[Int32]([0, 1, 0, 1, 0, 1, 0, 1])),
            Series("j", Column[Int32]([1, 1, 0, 0, 1, 1, 0, 0])),
        ]
    )
    check(
        frame.lazy().select_exprs(
            [
                cov(col("x"), col("y"))
                .over("g")
                .std(0)
                .over("j")
                .alias("nested"),
                corr(col("x"), col("y"), "spearman")
                .over("g")
                .rolling_mean(3, 1)
                .alias("rolling"),
                when(col("g") == 0)
                .then(corr(col("x"), col("y")).over("j"))
                .otherwise(cov(col("x"), col("y"), 0).over("g"))
                .alias("conditional"),
            ]
        ),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(corr(col("x"), col("y"), "spearman").alias("r"))
        .filter(col("r").is_finite())
        .sort("j")
        .head(5),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(cov(col("x"), col("y")).over("g").alias("c"))
        .group_by("j", maintain_order=True)
        .agg(corr(col("x"), col("c"))),
        runtime,
    )
    check(
        frame.lazy().select_exprs(
            [
                col("x").var().alias("v"),
                col("y").std().alias("s"),
                cov(col("x"), col("y")).alias("c"),
                corr(col("x"), col("y")).alias("r"),
                col("y").mean().alias("mean"),
            ]
        ),
        runtime,
    )
    var right = DataFrame(
        [
            Series("g", Column[Int32]([0, 1])),
            Series("extra", Column[Float64]([3, 2])),
        ]
    )
    check(
        frame.lazy()
        .join(right.lazy(), "g")
        .select(corr(col("x"), col("y") + col("extra"), "spearman")),
        runtime,
    )
    check(
        frame.lazy().select(
            cov(col("x").cast(DataType.INT64) + lit(Int64(1)), col("y"))
        ),
        runtime,
    )
    var bad = DataFrame(
        [
            Series("x", Column[Int32]([Int32.MAX, 1])),
            Series("y", Column[Float64]([2, 3])),
        ]
    )
    for side in range(2):
        var overflow = cov(
            col("x") + lit(Int32(1)), col("y")
        ) if side == 0 else corr(col("y"), col("x") + lit(Int32(1)))
        with assert_raises(contains="overflow"):
            _ = bad.lazy().select(overflow).collect(accelerator=runtime)
        with assert_raises(contains="overflow"):
            _ = bad.lazy().select(overflow).collect(engine="cpu")
    var unproven = (
        frame.lazy().select(cov(col("x") + lit(Int32(1)), col("y"))).head(1)
    )
    assert_true(
        "unsupported" in MetalRuntime(device_id=999999).describe(unproven)
    )
    var unproven_right = (
        frame.lazy().select(corr(col("y"), col("x") + lit(Int32(1)))).head(1)
    )
    assert_true(
        "unsupported" in MetalRuntime(device_id=999999).describe(unproven_right)
    )
    var query = frame.lazy().select_exprs(
        [
            cov(col("x"), col("y")).alias("c"),
            corr(col("x"), col("y")).alias("r"),
        ]
    )
    var estimate = runtime.estimate_auto(query)
    assert_true(";paired-moments-v1" in estimate.family)
    assert_true(";metal-paired-moments-v1" in estimate.family)
    var profiled = (
        frame.lazy()
        .select(corr(col("x"), col("y"), "spearman").over("g"))
        .profile(accelerator=runtime)
    )
    for i in range(profiled[1].height() - 1):
        assert_equal(
            profiled[1].column("host_result_bytes").int64()._get(i), Int64(0)
        )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    for ddof in List[Int]([0, 1, 7, Int(Int64.MAX), 2, 1]):
        check(frame.lazy().select(cov(col("x"), col("y"), ddof)), runtime)
    _ = setting^
