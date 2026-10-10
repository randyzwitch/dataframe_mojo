"""Backend policy stays above CPU scheduling and rejects unavailable engines."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit, scan_csv
from dataframe.lazy import AcceleratorBackend, LazyFrame


def test_cpu_and_auto_preserve_query_results() raises:
    var frame = DataFrame(
        [Series("x", Column[Float32]([-2, 1, 9, 3], [True, True, False, True]))]
    )
    var plan = (
        frame.lazy()
        .filter(col("x") > lit(Float32(0)))
        .select((col("x") * lit(Float32(1.25))).sum().alias("total"))
    )
    var expected = plan.collect()
    assert_equal(expected.item(0, "total").float32(), Float32(5.0))
    for engine in ["cpu", "auto"]:
        for streaming in [False, True]:
            for optimize in [False, True]:
                assert_true(
                    plan.collect(
                        engine=engine, streaming=streaming, optimize=optimize
                    ).equals(expected)
                )
        assert_true(plan.profile(engine=engine)[0].equals(expected))
        assert_true(plan.fetch(1, engine=engine).equals(expected))
    assert_equal(plan.explain(), plan.explain(engine="auto"))
    assert_true("ENGINE cpu: auto uses CPU" in plan.explain(engine="auto"))
    # Automatic diagnostics prepend a decision without changing the CPU plan.
    assert_true(plan.explain().endswith(plan.explain(engine="cpu")))
    assert_true(not plan.explain(engine="cpu").startswith("ENGINE "))


def test_unavailable_backend_rejected_before_reading_source() raises:
    var plan = scan_csv("/nonexistent/dataframe_backend_selection/input.csv")
    with assert_raises(contains="Accelerator provider is not installed"):
        _ = plan.collect(engine="accel")
    with assert_raises(contains="Accelerator provider is not installed"):
        _ = plan.profile(engine="accel")
    with assert_raises(contains="Accelerator provider is not installed"):
        _ = plan.fetch(engine="accel")
    var description = plan.explain(engine="accel")
    assert_true("ENGINE accel:" in description)
    assert_true("Accelerator provider is not installed" in description)


def test_invalid_engine_is_not_silently_ignored() raises:
    var plan = DataFrame([Series("x", Column[Int64]([1]))]).lazy()
    for engine in ["", "gpu", "nvidia", "CPU", "apple", "typo"]:
        with assert_raises(contains="Unknown engine"):
            _ = plan.collect(engine=engine)
        with assert_raises(contains="Unknown engine"):
            _ = plan.profile(engine=engine)
        with assert_raises(contains="Unknown engine"):
            _ = plan.fetch(engine=engine)
        with assert_raises(contains="Unknown engine"):
            _ = plan.explain(engine=engine)


@fieldwise_init
struct ProbeBackend(AcceleratorBackend):
    def execute(self, plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        raise Error("provider invoked")

    def describe(self, plan: LazyFrame) -> String:
        return "probe provider"


def test_explicit_provider_dispatch_without_gpu_dependencies() raises:
    var provider = ProbeBackend()
    var plan = (
        DataFrame([Series("x", Column[Int64]([1, 2]))])
        .lazy()
        .select(col("x").sum())
    )
    for engine in ["cpu", "auto"]:
        assert_true(
            plan.collect(engine=engine, accelerator=provider).equals(
                plan.collect()
            )
        )
        assert_true(
            plan.profile(engine=engine, accelerator=provider)[0].equals(
                plan.collect()
            )
        )
    with assert_raises(contains="provider invoked"):
        _ = plan.collect(engine="accel", accelerator=provider)
    with assert_raises(contains="Unknown engine"):
        _ = plan.collect(engine="typo", accelerator=provider)
    assert_equal(plan.explain(accelerator=provider), "probe provider")


@fieldwise_init
struct AutomaticProbe(AcceleratorBackend):
    var choice: String

    def select_auto(
        self,
        plan: LazyFrame,
        optimize: Bool,
        streaming: Bool,
        batch_size: Int,
    ) -> Tuple[String, String]:
        return (self.choice, "test placement reason")

    def execute(self, plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        raise Error("execution fault must propagate")

    def describe(self, plan: LazyFrame) -> String:
        return "automatic probe"


def test_auto_provider_dispatch_and_errors() raises:
    var plan = DataFrame([Series("x", Column[Int64]([1, 2]))]).lazy()
    for choice in ["accel", "mixed"]:
        var provider = AutomaticProbe(choice)
        assert_true(
            ("ENGINE " + choice + ": test placement reason")
            in plan.explain(engine="auto", accelerator=provider)
        )
        with assert_raises(contains="execution fault must propagate"):
            _ = plan.collect(engine="auto", accelerator=provider)
        with assert_raises(contains="execution fault must propagate"):
            _ = plan.profile(engine="auto", accelerator=provider)
        assert_equal(
            plan.collect(engine="cpu", accelerator=provider).height(), 2
        )
        with assert_raises(contains="batch_size must be positive"):
            _ = plan.collect(engine="auto", accelerator=provider, batch_size=0)
    var bad = AutomaticProbe("typo")
    with assert_raises(contains="Invalid automatic provider decision"):
        _ = plan.collect(engine="auto", accelerator=bad)
    with assert_raises(contains="Invalid automatic provider decision"):
        _ = plan.profile(engine="auto", accelerator=bad)
    with assert_raises(contains="Invalid automatic provider decision"):
        _ = plan.explain(engine="auto", accelerator=bad)


def test_auto_cpu_reason_in_profile() raises:
    var plan = DataFrame([Series("x", Column[Int64]([1, 2]))]).lazy()
    var provider = AutomaticProbe("cpu")
    var result = plan.profile(engine="auto", accelerator=provider)
    assert_equal(result[0].height(), 2)
    assert_equal(
        result[1].item(0, "selection_reason").string(), "test placement reason"
    )


@fieldwise_init
struct OptionsProbe(AcceleratorBackend):
    def select_auto(
        self,
        plan: LazyFrame,
        optimize: Bool,
        streaming: Bool,
        batch_size: Int,
    ) -> Tuple[String, String]:
        return (
            "cpu" if not optimize
            and not streaming
            and batch_size == 123 else "accel",
            "matched CPU execution settings",
        )

    def execute(self, plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        raise Error("CPU options were not forwarded")

    def describe(self, plan: LazyFrame) -> String:
        return "options probe"


def test_auto_policy_receives_execution_settings() raises:
    var plan = DataFrame([Series("x", Column[Int64]([1, 2]))]).lazy()
    var provider = OptionsProbe()
    assert_equal(
        plan.collect(
            engine="auto",
            accelerator=provider,
            optimize=False,
            streaming=False,
            batch_size=123,
        ).height(),
        2,
    )
    var result = plan.profile(
        engine="auto",
        accelerator=provider,
        optimize=False,
        streaming=False,
        batch_size=123,
    )
    assert_equal(
        result[1].item(0, "selection_reason").string(),
        "matched CPU execution settings",
    )
    assert_true(
        "ENGINE cpu: matched CPU execution settings"
        in plan.explain(
            engine="auto",
            accelerator=provider,
            optimize=False,
            streaming=False,
            batch_size=123,
        )
    )
    with assert_raises(contains="CPU options were not forwarded"):
        _ = plan.collect(engine="auto", accelerator=provider)


@fieldwise_init
struct SuccessfulAutoProbe(AcceleratorBackend):
    var choice: String

    def select_auto(
        self,
        plan: LazyFrame,
        optimize: Bool,
        streaming: Bool,
        batch_size: Int,
    ) -> Tuple[String, String]:
        return (self.choice, "successful automatic selection")

    def execute(self, plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        return plan.profile(engine="cpu")

    def describe(self, plan: LazyFrame) -> String:
        return "successful probe"


def test_selected_provider_profile_includes_selection_reason() raises:
    var plan = DataFrame([Series("x", Column[Int64]([1, 2]))]).lazy()
    for choice in ["accel", "mixed"]:
        var provider = SuccessfulAutoProbe(choice)
        var result = plan.profile(engine="auto", accelerator=provider)
        assert_equal(result[0].height(), 2)
        assert_equal(
            result[1].item(0, "selection_reason").string(),
            "successful automatic selection",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
