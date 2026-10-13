"""Schema selector expansion precedes every Metal expression normalization."""
from std.testing import assert_true, assert_equal, assert_raises
from dataframe import (
    DataFrame,
    Series,
    Column,
    StringColumn,
    col,
    lit,
    all,
    exclude,
    by_dtype,
    nth,
    scan_csv,
)
from dataframe.metal import MetalRuntime
from dataframe.accelerator.selectors import has_selectors, lower_selectors
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_scalar_broadcasts import (
    check,
    same,
)
from tests.execution_modes.test_backend_metal_moments import near


def fixture() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [1, 2, 3, 4, 5, 6], [True, False, True, True, True, True]
                ),
            ),
            Series("y", Column[Float64]([6, 5, 4, 3, 2, 1])),
            Series("k", Column[Int64]([0, 1, 0, 1, 0, 1])),
            Series("s", StringColumn(["a", "β", "a", "β", "c", "c"])),
        ]
    )


def test_metal_selectors_metadata_names_positions_dtypes_and_errors() raises:
    var frame = fixture()
    var runtime = MetalRuntime(device_id=999999)
    for selector in [
        all(),
        col(["s", "x"]),
        exclude(["k", "s"]),
        by_dtype(["float64"]),
        nth(-1),
    ]:
        var query = frame.lazy().select(
            selector.name_prefix("p_").name_suffix("_s")
        )
        assert_true(has_selectors(query))
        var lowered = lower_selectors(query)
        assert_true(not has_selectors(lowered))
        same(lowered.collect(engine="cpu"), query.collect(engine="cpu"))
        assert_true("unsupported" not in runtime.describe(query))
        assert_true(has_selectors(query))
    for selector in [col(["missing"]), nth(9), by_dtype(["invalid dtype"])]:
        with assert_raises():
            _ = lower_selectors(frame.lazy().select(selector))
    with assert_raises(contains="exactly one column"):
        _ = lower_selectors(
            frame.lazy().filter(by_dtype(["float64"]) > lit(Float64(2)))
        )
    var empty = frame.lazy().select(by_dtype(["uint8"]))
    same(
        lower_selectors(empty).collect(engine="cpu"),
        empty.collect(engine="cpu"),
    )
    with assert_raises(contains="at most one selector"):
        _ = lower_selectors(frame.lazy().select(nth(0) + nth(1)))
    var missing = scan_csv("/nonexistent/metal-selector-must-not-read.csv")
    with assert_raises(contains="in-memory scans"):
        _ = missing.select(all()).collect(accelerator=runtime)


def test_native_metal_selectors_projection_replacement_filters_strings_and_estimates() raises:
    _ = require_metal()
    var frame = fixture()
    check(frame.lazy().select(all()))
    check(frame.lazy().select(col(["s", "y", "x"])))
    check(frame.lazy().with_columns(by_dtype(["float64"]) + lit(Float64(2))))
    check(
        frame.lazy()
        .select(exclude(["k", "s"]).name_prefix("before_"))
        .select(all().name_suffix("_after"))
    )
    check(frame.lazy().filter(nth(0) > lit(Float64(2))).select(all()))
    check(frame.lazy().select(nth(-1).str().to_uppercase().alias("upper")))
    check(frame.lazy().with_columns(by_dtype(["uint8"]) + lit(UInt8(1))))
    var query = frame.lazy().select(by_dtype(["float64"]) + lit(Float64(1)))
    var runtime = MetalRuntime()
    var expanded = lower_selectors(query)
    assert_equal(
        runtime.estimate_auto(query).family,
        runtime.estimate_auto(expanded).family,
    )
    var profiled = query.profile(accelerator=runtime)
    same(profiled[0], query.collect(engine="cpu"))
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    check(query)


def test_native_metal_selectors_aggregates_partition_windows_join_and_consumer_scopes() raises:
    _ = require_metal()
    var frame = fixture()
    check(frame.lazy().select(by_dtype(["float64"]).sum().name_suffix("_sum")))
    check(frame.lazy().group_by("k").agg(by_dtype(["float64"]).mean()))
    check(frame.lazy().group_by("k").agg(by_dtype(["uint8"]).sum()))
    check(
        frame.lazy().select(
            (by_dtype(["float64"]).mean() + lit(Float64(2))).name_suffix(
                "_mean"
            )
        )
    )
    var query = frame.lazy().select(
        by_dtype(["float64"]).rolling_var(3, 1, 0).over("k")
    )
    near(
        query.collect(accelerator=MetalRuntime()),
        query.collect(engine="cpu", streaming=False),
    )
    var ranked = frame.lazy().select(by_dtype(["float64"]).rank().over("k"))
    near(
        ranked.collect(accelerator=MetalRuntime()),
        ranked.collect(engine="cpu", streaming=False),
    )
    var joined = frame.lazy().join(frame.lazy(), ["k"], how="inner")
    check(joined.select(by_dtype(["float64"]).name_suffix("_j")))
    check(frame.slice(0, 0).lazy().select(all()))
