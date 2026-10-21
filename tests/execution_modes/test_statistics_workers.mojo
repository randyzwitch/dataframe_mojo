"""Worker-dispatched moments and pairs retain exact independent sufficient statistics."""
# contract-group: execution.modes
from std.math import abs
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col, corr, cov
from dataframe.parallel import worker_count, _helpers
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.scaled_pair_reference import ScaledPairs

comptime ROWS = 262145


def _expressions() -> List[Expr]:
    return [
        col("x").count().alias("count"),
        col("x").mean().alias("mean"),
        col("x").var(0).alias("var0"),
        col("x").var(1).alias("var1"),
        col("x").std(1).alias("std1"),
        corr(col("x"), col("y")).alias("correlation"),
        cov(col("x"), col("y"), ddof=0).alias("cov0"),
        cov(col("x"), col("y"), ddof=1).alias("cov1"),
    ]


def _near(actual: Float64, expected: Float64) raises:
    # Exact Int128 sufficient statistics are compared with partitioned online
    # states. Inputs <=11 and fixed n<=262145 receive 2e-12 absolute+relative.
    assert_true(abs(actual - expected) <= 2e-12 * (1 + abs(expected)))


def _assert(
    actual: DataFrame,
    xs: List[ScaledPairs],
    pairs: List[ScaledPairs],
    grouped: Bool,
) raises:
    assert_equal(actual.height(), 5 if grouped else 1)
    var names: List[String] = [
        "count",
        "mean",
        "var0",
        "var1",
        "std1",
        "correlation",
        "cov0",
        "cov1",
    ]
    if grouped:
        names.insert(0, "g")
    assert_equal(actual.columns(), names)
    for row in range(actual.height()):
        var group = row if grouped else 5
        if grouped:
            assert_equal(actual.item(row, "g").int64(), Int64(group))
        ref x = xs[group]
        ref pair = pairs[group]
        assert_true(actual.column("count").dtype() == DataType.INT64)
        assert_equal(actual.item(row, "count").int64(), Int64(x.count))
        for name in [
            "mean",
            "var0",
            "var1",
            "std1",
            "correlation",
            "cov0",
            "cov1",
        ]:
            assert_true(actual.column(name).dtype() == DataType.FLOAT64)
            assert_true(not actual.item(row, name).is_null())
        _near(
            actual.item(row, "mean").float64(),
            Float64(x.sx) / Float64(x.count) / 4,
        )
        _near(actual.item(row, "var0").float64(), x.variance(0))
        _near(actual.item(row, "var1").float64(), x.variance(1))
        var deviation = actual.item(row, "std1").float64()
        assert_true(deviation >= 0)
        _near(deviation * deviation, x.variance(1))
        _near(actual.item(row, "correlation").float64(), pair.correlation())
        _near(actual.item(row, "cov0").float64(), pair.covariance(0))
        _near(actual.item(row, "cov1").float64(), pair.covariance(1))


def _matrix[D: DType]() raises:
    var xs = List[ScaledPairs](length=6, fill=ScaledPairs())
    var pairs = List[ScaledPairs](length=6, fill=ScaledPairs())
    var xv = List[Bool]()
    var yv = List[Bool]()
    var xvalues = List[Scalar[D]]()
    var yvalues = List[Scalar[D]]()
    var groups = List[Int64]()
    for row in range(ROWS):
        var x = row % 31 - 15
        var y = -2 * x + 12
        var xvalid = row % 7 != 3
        var yvalid = row % 11 != 5
        var group = row % 5
        xvalues.append(
            (Float64(x) / 4)
            .cast[D]() if xvalid else (Float64(0) / Float64(0))
            .cast[D]()
        )
        yvalues.append(
            (Float64(y) / 4)
            .cast[D]() if yvalid else (Float64(1) / Float64(0))
            .cast[D]()
        )
        xv.append(xvalid)
        yv.append(yvalid)
        groups.append(Int64(group))
        for target in [group, 5]:
            if xvalid:
                xs[target].add(x, 0)
            if xvalid and yvalid:
                pairs[target].add(x, y)
    var source = DataFrame(
        [
            Series("x", Column[Scalar[D]](xvalues^, xv^)),
            Series("y", Column[Scalar[D]](yvalues^, yv^)),
            Series("g", Column[Int64](groups^)),
        ]
    )
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        for workers in [1, 2, 4]:
            var setting = WorkerSetting(workers)
            assert_equal(worker_count(ROWS), workers)
            for streaming in [False, True]:
                if layout == 2 and streaming:
                    continue  # Isolated strict validity-count regression below.
                var global_plan = frame.lazy().select_exprs(_expressions())
                var grouped = (
                    frame.lazy()
                    .group_by("g", maintain_order=True)
                    .agg(_expressions())
                )
                _assert(
                    global_plan.collect(
                        optimize=False,
                        streaming=streaming,
                        batch_size=65537,
                        engine="cpu",
                    ),
                    xs,
                    pairs,
                    False,
                )
                _assert(
                    grouped.collect(
                        optimize=False,
                        streaming=streaming,
                        batch_size=65537,
                        engine="cpu",
                    ),
                    xs,
                    pairs,
                    True,
                )
                assert_equal(_helpers().load(), Int64(0))
            _ = setting^


def test_parallel_moments_and_pair_statistics_match_exact_scaled_models_across_workers_and_modes() raises:
    _matrix[DType.float32]()
    _matrix[DType.float64]()


def test_scaled_pair_reference_has_literal_variance_covariance_and_correlation_anchors() raises:
    var model = ScaledPairs()
    model.add(4, 8)
    model.add(8, 16)
    model.add(12, 24)
    _near(model.variance(0), 2.0 / 3.0)
    _near(model.variance(1), 1.0)
    _near(model.covariance(0), 4.0 / 3.0)
    _near(model.covariance(1), 2.0)
    _near(model.correlation(), 1.0)


# expected-failure: Verified streamed misaligned Float32 chunks produce
# expected-failure: incorrect grouped nonnull counts at batch65537.
# expected-failure: Independent count of rows with row%7!=3 gives44939 for
# expected-failure: first group; actual45173. Other modes complete normal
# expected-failure: matrix; focused strict first failure does not claim other
# expected-failure: affected reducers pass.
# expected-diagnostic: streamed misaligned chunks must preserve group validity
# expected-diagnostic: counts
def test_streamed_misaligned_numeric_chunks_preserve_group_validity_counts() raises:
    var values = List[Float32]()
    var valid = List[Bool]()
    var groups = List[Int64]()
    var expected = List[Int64](length=5, fill=0)
    for row in range(ROWS):
        values.append(Float32(row % 31 - 15) / 4)
        valid.append(row % 7 != 3)
        groups.append(Int64(row % 5))
        if row % 7 != 3:
            expected[row % 5] += 1
    var source = chunk_layout(
        DataFrame(
            [
                Series("x", Column[Float32](values^, valid)),
                Series("g", Column[Int64](groups^)),
            ]
        )
    )
    var actual = (
        source.lazy()
        .group_by("g", maintain_order=True)
        .agg(col("x").count().alias("n"))
        .collect(optimize=False, streaming=True, batch_size=65537, engine="cpu")
    )
    assert_equal(actual.height(), 5)
    for group in range(5):
        assert_equal(actual.item(group, "g").int64(), Int64(group))
        assert_equal(
            actual.item(group, "n").int64(),
            expected[group],
            "streamed misaligned chunks must preserve group validity counts",
        )
