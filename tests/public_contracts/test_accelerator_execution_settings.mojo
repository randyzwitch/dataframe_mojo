"""Settings reach existing providers without changing their interfaces."""
# contract-group: public.apple_parity
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.execution_settings import ExecutionSettings
from dataframe.lazy import AcceleratorBackend, LazyFrame
from dataframe.metal import MetalRuntime


@fieldwise_init
struct _LegacySettingsBackend(AcceleratorBackend):
    var observed: ArcPointer[List[ExecutionSettings]]

    def select_auto(
        self, plan: LazyFrame, optimize: Bool, streaming: Bool, batch_size: Int
    ) -> Tuple[String, String]:
        return ("accel", "settings test provider")

    def execute(self, plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        assert_true(Bool(plan._accelerator_settings))
        self.observed[].append(plan._accelerator_settings.value().copy())
        return (plan._frames[0].copy(), DataFrame(List[Series]()))

    def describe(self, plan: LazyFrame) -> String:
        self.observed[].append(plan._accelerator_settings.value().copy())
        return "ENGINE accel: legacy settings provider\n"


def test_accelerator_settings_preserve_legacy_provider_dispatch_and_plan_copy() raises:
    var query = (
        DataFrame([Series("x", Column[Int64]([7]))]).lazy().select(col("x"))
    )
    var observed = ArcPointer(List[ExecutionSettings]())
    var backend = _LegacySettingsBackend(observed.copy())
    _ = query.collect(
        accelerator=backend, optimize=False, streaming=False, batch_size=1
    )
    _ = query.profile(
        accelerator=backend, optimize=True, streaming=False, batch_size=7
    )
    _ = query.explain(
        accelerator=backend, optimize=False, streaming=True, batch_size=3
    )
    _ = query.collect(
        accelerator=backend,
        engine="auto",
        optimize=False,
        streaming=True,
        batch_size=11,
    )
    _ = query.profile(
        accelerator=backend,
        engine="auto",
        optimize=True,
        streaming=False,
        batch_size=13,
    )
    assert_equal(len(observed[]), 5)
    var batches: List[Int] = [1, 7, 3, 11, 13]
    var optimize: List[Bool] = [False, True, False, False, True]
    var streaming: List[Bool] = [False, False, True, True, False]
    for at in range(5):
        assert_equal(observed[][at].batch_size, batches[at])
        assert_equal(observed[][at].optimize, optimize[at])
        assert_equal(observed[][at].streaming, streaming[at])
    assert_false(Bool(query._accelerator_settings))
    var copied = query._with_accelerator_settings(False, False, 2).copy()
    assert_equal(copied._accelerator_settings.value().batch_size, 2)
    var original_runtime = MetalRuntime(device_id=999999)
    var configured_runtime = original_runtime._configured(copied)
    assert_false(configured_runtime._execution_settings.optimize)
    assert_false(configured_runtime._execution_settings.streaming)
    assert_equal(configured_runtime._execution_settings.batch_size, 2)
    assert_true(original_runtime._execution_settings.optimize)
    assert_equal(original_runtime._execution_settings.batch_size, 65536)
    # New fields do not require legacy providers to implement new methods.
    assert_true(
        "legacy settings provider" in query.explain(accelerator=backend)
    )


def test_lazy_requested_stream_batch_does_not_override_eager_expression_batches() raises:
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([128, 0])),
            Series("t", Column[Int64]([0, Int64.MAX])).with_dtype(
                DataType.duration("ms")
            ),
        ]
    )
    var expression = col("t").cast(DataType.duration("ns")).cast(
        DataType.INT64
    ) + col("x").cast(DataType.INT8).cast(DataType.INT64)
    # DataFrame's explicit expression batch changes which AST visit fails;
    # LazyFrame's batch argument configures stream morsels instead.
    for implode in [False, True]:
        var output = expression.implode() if implode else expression.copy()
        with assert_raises(contains="cast from int64 to int8 failed"):
            _ = frame.select(output, batch_size=1)
        with assert_raises(contains="temporal value overflows Int64"):
            _ = frame.select(output, batch_size=2)
        for optimize in [False, True]:
            for streaming in [False, True]:
                for batch_size in [1, 2, 65536]:
                    with assert_raises(
                        contains="temporal value overflows Int64"
                    ):
                        _ = (
                            frame.lazy()
                            .select(output)
                            .collect(
                                engine="cpu",
                                optimize=optimize,
                                streaming=streaming,
                                batch_size=batch_size,
                            )
                        )
