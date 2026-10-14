"""Apple Metal execution through Mojo kernels inside the dataframe library.

Construction, import and explanation do not initialize a device. Shared Mojo
planners bind the complete query before MAX creates a Metal context. Values
stay in Mojo-owned device tables between regions and export only at the final
CPU boundary. Automatic placement requires separately measured Mojo curves.
"""
from .accelerator.sets import has_set_operations, lower_set_operations
from .accelerator.selectors import has_selectors, lower_selectors
from .accelerator.scalars import (
    has_scalar_projections,
    lower_scalar_projections,
)
from .accelerator.constants import (
    has_literal_text_parsing,
    fold_literal_text_parsing,
)
from .accelerator.distinct_counts import (
    has_distinct_counts,
    lower_distinct_counts,
)
from .accelerator.paired_moments import has_spearman, lower_spearman
from .accelerator.windows import (
    has_partition_aggregates,
    lower_partition_aggregates,
)
from .accelerator.aggregates import (
    has_aggregate_compositions,
    lower_aggregate_compositions,
)
from .accelerator.dictionaries import (
    DictionaryCodes,
    has_dictionary_columns,
    encode_query,
    restore_frame,
    lower_dictionary_operations,
)
from .accelerator.memory import checked_add
from .accelerator.metadata import schema_frame
from .accelerator.cost import PlanEstimate, CalibrationKey
from .accelerator.capabilities import RowCapabilities
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.joins import has_join, has_scalar_consumers
from .lazy import AcceleratorBackend, LazyFrame
from .column import Column
from .frame import DataFrame
from .series import Series
from std.time import perf_counter_ns


def metal_library_candidates() -> List[String]:
    """Legacy discovery API; the Mojo provider needs no dataframe dylib."""
    return List[String]()


def metal_installed() -> Bool:
    """Whether this target supports Mojo Metal; no device is initialized."""
    from .accelerator.gpu.identity import available

    return available()


def _capabilities() -> RowCapabilities:
    from .accelerator.gpu.runtime import capabilities

    return capabilities()


def _describe_mojo_gpu(query: LazyFrame) -> String:
    from .accelerator.gpu.runtime import describe, capabilities
    from .accelerator.gpu.dag_runtime import describe as describe_dag

    if has_join(query) or has_scalar_consumers(query):
        return describe_dag(query, capabilities())
    return describe(query)


struct MetalRuntime(AcceleratorBackend):
    """Apple Metal provider with Mojo-owned kernels, buffers and query lifetime."""

    var _device: Int
    var _budget: Int

    def __init__(
        out self, device_id: Int = 0, *, memory_limit_bytes: Int = -1
    ) raises:
        if device_id < 0 or memory_limit_bytes < -1:
            raise Error(
                "Metal requires nonnegative device_id and memory_limit_bytes"
                " >= -1"
            )
        self._device = device_id
        self._budget = memory_limit_bytes

    @staticmethod
    def device_count() raises -> Int:
        from .accelerator.gpu.identity import device_count

        return device_count()

    def estimate_auto(self, query: LazyFrame) raises -> PlanEstimate:
        """Bound supported numeric work without reading values or device setup."""
        from .accelerator.gpu.calibration import estimate

        return estimate(query)

    def calibration_key(
        self, optimize: Bool, streaming: Bool, batch_size: Int
    ) raises -> CalibrationKey:
        from .accelerator.gpu.identity import key

        return key(self._device, optimize, streaming, batch_size)

    def select_auto(
        self, query: LazyFrame, optimize: Bool, streaming: Bool, batch_size: Int
    ) -> Tuple[String, String]:
        from .accelerator.gpu.calibration import select_auto

        return select_auto(
            query, self._device, self._budget, optimize, streaming, batch_size
        )

    def execute(self, query: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        comptime if METAL_TARGET:
            return self._run(query, False)
        else:
            raise Error(
                "Mojo Metal execution requires an Apple Metal compilation target; "
                + self.describe(query)
            )

    def execute_profiled(
        self, query: LazyFrame
    ) raises -> Tuple[DataFrame, DataFrame]:
        comptime if METAL_TARGET:
            return self._run(query, True)
        else:
            raise Error(
                "Mojo Metal execution requires an Apple Metal compilation target; "
                + self.describe(query)
            )

    def _run(
        self, query: LazyFrame, profiling: Bool
    ) raises -> Tuple[DataFrame, DataFrame]:
        from .accelerator.gpu.runtime import execute, capabilities
        from .accelerator.gpu.dag_runtime import execute as execute_dag

        var start = Int(perf_counter_ns())
        if has_selectors(query):
            var executed = self._run(lower_selectors(query), profiling)
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_scalar_projections(query):
            var executed = self._run(lower_scalar_projections(query), profiling)
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_literal_text_parsing(query):
            var executed = self._run(
                fold_literal_text_parsing(query), profiling
            )
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_partition_aggregates(query):
            var executed = self._run(
                lower_partition_aggregates(query), profiling
            )
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_spearman(query):
            var executed = self._run(lower_spearman(query), profiling)
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_aggregate_compositions(query):
            var executed = self._run(
                lower_aggregate_compositions(query, exact_float32=True),
                profiling,
            )
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_distinct_counts(query):
            var executed = self._run(lower_distinct_counts(query), profiling)
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_set_operations(query):
            var executed = self._run(lower_set_operations(query), profiling)
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_dictionary_columns(query):
            # Validate original string semantics before replacing values with
            # ordered codes. Schema execution reads no source rows.
            var schema = schema_frame(query, len(query._nodes) - 1)
            var lowered = lower_dictionary_operations(query, self._budget)
            var staged_query = lowered[0].copy()
            var dictionary = DictionaryCodes(staged_query)
            var staging_bytes = checked_add(
                dictionary.staging_bytes(staged_query), lowered[1]
            )
            var budget = self._budget
            if budget >= 0:
                if staging_bytes > budget:
                    raise Error(
                        "Metal dictionary staging exceeds payload budget"
                    )
                budget -= staging_bytes
            var encoded = encode_query(staged_query, dictionary)
            var runtime = MetalRuntime(self._device, memory_limit_bytes=budget)
            var staged_ns = Int(perf_counter_ns()) - start
            var executed = runtime._run(encoded, profiling)
            var restoration_bytes = dictionary.restoration_bytes(
                executed[0], schema
            )
            if self._budget >= 0:
                if restoration_bytes > self._budget - staging_bytes:
                    raise Error(
                        "Metal dictionary restoration exceeds payload budget"
                    )
            var decode_start = Int(perf_counter_ns())
            var result = restore_frame(executed[0], schema, dictionary)
            var decoded_ns = Int(perf_counter_ns()) - decode_start
            var report = executed[1].copy()
            # Dictionary owners survive both GPU execution and restoration.
            # Preserve region peaks while reporting the original query scope.
            var inner_peak = 0
            var peak_field = (
                "query_peak_requested_bytes" if "query_peak_requested_bytes"
                in report.columns() else "peak_requested_bytes"
            )
            for row in range(report.height()):
                inner_peak = max(
                    inner_peak, Int(report.item(row, peak_field).int64())
                )
            var query_peak = checked_add(
                staging_bytes, max(inner_peak, restoration_bytes)
            )
            report = report.with_column(
                Series(
                    "query_peak_requested_bytes",
                    Column[Int64](
                        List[Int64](
                            length=report.height(), fill=Int64(query_peak)
                        )
                    ),
                )
            )
            var elapsed = Float64(Int(perf_counter_ns()) - start) / 1e6
            report = report.with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](length=report.height(), fill=elapsed)
                    ),
                )
            )
            report = report.with_column(
                Series(
                    "dictionary_staging_bytes",
                    Column[Int64](
                        List[Int64](
                            length=report.height(), fill=Int64(staging_bytes)
                        )
                    ),
                )
            )
            report = report.with_column(
                Series(
                    "dictionary_operation_tables",
                    Column[Int64](
                        List[Int64](
                            length=report.height(), fill=Int64(lowered[2])
                        )
                    ),
                )
            )
            report = report.with_column(
                Series(
                    "dictionary_operation_staging_bytes",
                    Column[Int64](
                        List[Int64](
                            length=report.height(), fill=Int64(lowered[1])
                        )
                    ),
                )
            )
            report = report.with_column(
                Series(
                    "dictionary_encode_ms",
                    Column[Float64](
                        List[Float64](
                            length=report.height(),
                            fill=Float64(staged_ns) / 1e6,
                        )
                    ),
                )
            )
            report = report.with_column(
                Series(
                    "dictionary_decode_ms",
                    Column[Float64](
                        List[Float64](
                            length=report.height(),
                            fill=Float64(decoded_ns) / 1e6,
                        )
                    ),
                )
            )
            return (result^, report^)
        if has_join(query) or has_scalar_consumers(query):
            return execute_dag(
                query, self._device, self._budget, profiling, capabilities()
            )
        return execute(query, self._device, self._budget, profiling)

    def describe(self, query: LazyFrame) -> String:
        try:
            if has_selectors(query):
                return "Metal schema selector expansion; " + self.describe(
                    lower_selectors(query)
                )
            if has_scalar_projections(query):
                return "Metal resident scalar projection; " + self.describe(
                    lower_scalar_projections(query)
                )
            if has_literal_text_parsing(query):
                return "Metal literal text parsing; " + self.describe(
                    fold_literal_text_parsing(query)
                )
            if has_partition_aggregates(query):
                return "Metal resident partition aggregates; " + self.describe(
                    lower_partition_aggregates(query)
                )
            if has_spearman(query):
                return "Metal resident Spearman ranks; " + self.describe(
                    lower_spearman(query)
                )
            if has_aggregate_compositions(query):
                return "Metal resident aggregate expressions; " + self.describe(
                    lower_aggregate_compositions(query, exact_float32=True)
                )
            if has_distinct_counts(query):
                return "Metal resident distinct counts; " + self.describe(
                    lower_distinct_counts(query)
                )
            if has_set_operations(query):
                return "Metal resident set operations; " + self.describe(
                    lower_set_operations(query)
                )
            if has_dictionary_columns(query):
                _ = schema_frame(query, len(query._nodes) - 1)
                var metadata = query.copy()
                for i in range(len(metadata._frames)):
                    metadata._frames[i] = metadata._frames[i].slice(0, 0)
                var lowered = lower_dictionary_operations(metadata)
                var staged_query = lowered[0].copy()
                var dictionary = DictionaryCodes(staged_query)
                var encoded = encode_query(staged_query, dictionary)
                return "Metal ordered dictionary staging; " + self.describe(
                    encoded
                )
            return _describe_mojo_gpu(query)
        except error:
            return "ENGINE accel: " + String(error) + "\n"
