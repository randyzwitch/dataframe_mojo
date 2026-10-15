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
from .accelerator.opaque import (
    has_opaque_columns,
    prepare as prepare_opaque,
    restore as restore_opaque,
    restoration_bytes as opaque_restoration_bytes,
)
from .accelerator.dictionaries import (
    DictionaryCodes,
    has_dictionary_columns,
    encode_query,
    restore_frame,
    lower_dictionary_operations,
)
from .accelerator.memory import checked_add
from .accelerator.code_domains import TextQuery
from .accelerator.qcuts import has_default_qcuts
from std.memory import ArcPointer
from .accelerator.sources import (
    file_source_nodes,
    stage_file_sources,
    prepare_file_sources,
)
from .accelerator.cuts import (
    has_cut_operations,
    lower_cuts,
    has_cut_materializations,
    staging_bytes as cut_staging_bytes,
)
from .accelerator.distributions import has_scalar_modes, lower_scalar_modes
from .accelerator.source_lists import (
    has_source_lists,
    lower_source_lists,
)
from .accelerator.grouped_lists import has_grouped_implode
from .accelerator.grouped_feeds import has_checked_grouped_feeds
from .accelerator.grouped_key_queries import (
    prepare_keys as prepare_grouped_keys,
)
from .accelerator.lists import (
    has_implode,
    has_implode_materializations,
    lower_terminal_implode,
    assemble as assemble_implode,
    observe_trailing_failures as observe_implode_trailing,
)
from .accelerator.source_structs import (
    has_source_structs,
    lower_source_structs,
    attach_metadata as attach_source_fields,
)
from .accelerator.structs import (
    has_terminal_structs,
    lower_terminal_structs,
    assemble as assemble_structs,
)
from .accelerator.metadata import schema_frame
from .accelerator.cost import PlanEstimate, CalibrationKey
from .accelerator.capabilities import RowCapabilities
from .accelerator.gpu.target import METAL_TARGET
from .accelerator.joins import (
    has_join,
    has_scalar_consumers,
    has_decimal_consumers,
    has_checked_integer_consumers,
)
from .accelerator.explodes import has_explode
from .lazy import AcceleratorBackend, LazyFrame
from .column import Column
from .frame import DataFrame
from .series import Series
from .execution_settings import ExecutionSettings
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


def _describe_mojo_gpu(
    query: LazyFrame, ordered_source: Bool = False
) raises -> String:
    from .accelerator.gpu.runtime import describe, capabilities
    from .accelerator.gpu.dag_runtime import describe as describe_dag

    if (
        has_join(query)
        or has_decimal_consumers(query)
        or has_checked_integer_consumers(query)
        or has_explode(query)
        or has_scalar_consumers(query)
        or has_cut_materializations(query)
        or has_implode_materializations(query)
    ):
        return describe_dag(query, capabilities())
    return describe(query, ordered_source=ordered_source)


struct MetalRuntime(AcceleratorBackend):
    """Apple Metal provider with Mojo-owned kernels, buffers and query lifetime."""

    var _device: Int
    var _budget: Int
    var _text_context: Optional[ArcPointer[TextQuery]]
    var _observation_only: Bool
    var _execution_settings: ExecutionSettings

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
        self._observation_only = False
        self._execution_settings = ExecutionSettings(True, True, 65536)
        self._text_context = None

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
            return self._configured(query)._run(query, False)
        else:
            raise Error(
                "Mojo Metal execution requires an Apple Metal compilation target; "
                + self.describe(query)
            )

    def execute_profiled(
        self, query: LazyFrame
    ) raises -> Tuple[DataFrame, DataFrame]:
        comptime if METAL_TARGET:
            return self._configured(query)._run(query, True)
        else:
            raise Error(
                "Mojo Metal execution requires an Apple Metal compilation target; "
                + self.describe(query)
            )

    def _configured(self, query: LazyFrame) -> Self:
        var configured = self.copy()
        if query._accelerator_settings:
            configured._execution_settings = (
                query._accelerator_settings.value().copy()
            )
        return configured^

    def _run(
        self, query: LazyFrame, profiling: Bool, *, original_entry: Bool = True
    ) raises -> Tuple[DataFrame, DataFrame]:
        from .accelerator.gpu.runtime import execute, capabilities
        from .accelerator.gpu.dag_runtime import execute as execute_dag
        from .accelerator.gpu.grouped_lists_runtime import (
            execute as execute_grouped_lists,
        )

        from .accelerator.gpu.source_lists_runtime import (
            execute as execute_source_lists,
        )

        if self._execution_settings.batch_size <= 0:
            raise Error("batch_size must be positive")
        from .accelerator.grouped_stream_routes import candidate
        from .accelerator.gpu.grouped_stream_dispatch import (
            execute as execute_float_stream,
        )
        from .parallel import configured_workers

        var float_started = Int(perf_counter_ns())
        var float_route = candidate(
            query,
            configured_workers(),
            self._execution_settings.batch_size,
            optimize=self._execution_settings.optimize,
            streaming=self._execution_settings.streaming,
            original_source=original_entry,
            budget=self._budget,
        )
        if float_route:
            var selected_float_route = float_route.take()
            return execute_float_stream(
                selected_float_route,
                self._device,
                self._budget,
                self._execution_settings.batch_size,
                profiling=profiling,
                observation_only=self._observation_only,
                started=float_started,
            )
        var start = Int(perf_counter_ns())
        if len(file_source_nodes(query)):
            var staged = prepare_file_sources(
                query,
                memory_limit_bytes=self._budget,
                optimize=self._execution_settings.optimize,
            )
            var staging_bytes = staged.staging_bytes
            var budget = self._budget
            if budget >= 0:
                budget -= staging_bytes
            var runtime = MetalRuntime(self._device, memory_limit_bytes=budget)
            runtime._observation_only = self._observation_only
            runtime._execution_settings = self._execution_settings.copy()
            runtime._text_context = self._text_context.copy()
            if staged.failure:
                runtime._observation_only = True
                for prefix in staged.prefixes:
                    _ = runtime._run(prefix, False, original_entry=False)
                raise Error(staged.failure)
            var decode_ms = Float64(Int(perf_counter_ns()) - start) / 1e6
            var executed = runtime._run(
                staged.query, profiling, original_entry=False
            )
            var report = executed[1].copy()
            var peak_field = (
                "query_peak_requested_bytes" if "query_peak_requested_bytes"
                in report.columns() else "peak_requested_bytes"
            )
            var inner_peak = 0
            for row in range(report.height()):
                inner_peak = max(
                    inner_peak, Int(report.item(row, peak_field).int64())
                )
            for field in List[Tuple[String, Int64]](
                [
                    ("file_scan_count", Int64(staged.scan_count)),
                    ("source_staging_bytes", Int64(staging_bytes)),
                    (
                        "query_peak_requested_bytes",
                        Int64(checked_add(staging_bytes, inner_peak)),
                    ),
                ]
            ):
                report = report.with_column(
                    Series(
                        field[0],
                        Column[Int64](
                            List[Int64](length=report.height(), fill=field[1])
                        ),
                    )
                )
            for field in List[Tuple[String, Float64]](
                [
                    ("source_decode_ms", decode_ms),
                    (
                        "query_wall_ms",
                        Float64(Int(perf_counter_ns()) - start) / 1e6,
                    ),
                ]
            ):
                report = report.with_column(
                    Series(
                        field[0],
                        Column[Float64](
                            List[Float64](length=report.height(), fill=field[1])
                        ),
                    )
                )
            return (executed[0].copy(), report^)
        if has_selectors(query):
            var executed = self._run(
                lower_selectors(query), profiling, original_entry=False
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
        if has_explode(query):
            return execute_dag(
                query,
                self._device,
                self._budget,
                profiling,
                capabilities(),
                observation_only=self._observation_only,
                text_context=self._text_context,
                ordered_source=original_entry,
                eager_feed=not self._execution_settings.streaming,
            )
        if has_grouped_implode(query) or has_checked_grouped_feeds(
            query, eager_feed=not self._execution_settings.streaming
        ):
            return execute_grouped_lists(
                query,
                self._device,
                self._budget,
                profiling,
                observation_only=self._observation_only,
                ordered_source=original_entry,
                eager_feed=not self._execution_settings.streaming,
            )
        if has_implode(query):
            var plan = lower_terminal_implode(
                query, ordered_source=original_entry
            )
            var assembly_bytes = (
                0 if self._observation_only else plan.assembly_bytes
            )
            if self._budget >= 0 and assembly_bytes > self._budget:
                raise Error("Metal implode assembly exceeds memory budget")
            var budget = self._budget
            if budget >= 0:
                budget -= assembly_bytes
            var runtime = MetalRuntime(self._device, memory_limit_bytes=budget)
            runtime._observation_only = self._observation_only
            runtime._execution_settings = self._execution_settings.copy()
            runtime._text_context = self._text_context.copy()
            var executed = runtime._run(
                plan.query, profiling, original_entry=False
            )
            observe_implode_trailing(plan)
            var result = plan.schema.copy() if self._observation_only else assemble_implode(
                executed[0], plan
            )
            var report = executed[1].copy()
            var peak_field = (
                "query_peak_requested_bytes" if "query_peak_requested_bytes"
                in report.columns() else "peak_requested_bytes"
            )
            var inner_peak = 0
            for row in range(report.height()):
                inner_peak = max(
                    inner_peak, Int(report.item(row, peak_field).int64())
                )
            var child_export_bytes = 0
            var region_exports = List[Int64](length=report.height(), fill=0)
            if "host_result_bytes" in report.columns():
                for row in range(report.height()):
                    var bytes = Int(
                        report.item(row, "host_result_bytes").int64()
                    )
                    region_exports[row] = Int64(bytes)
                    child_export_bytes = checked_add(child_export_bytes, bytes)
            # DAG reports place the final root/export last. Preserve every
            # region's typed export count and charge new List offsets there.
            if report.height():
                var last = report.height() - 1
                region_exports[last] = Int64(
                    checked_add(Int(region_exports[last]), assembly_bytes)
                )
            report = report.with_column(
                Series("host_result_bytes", Column[Int64](region_exports^))
            )
            for field in List[Tuple[String, Int64]](
                [
                    (
                        "query_implode_child_export_bytes",
                        Int64(child_export_bytes),
                    ),
                    (
                        "query_host_result_bytes",
                        Int64(checked_add(child_export_bytes, assembly_bytes)),
                    ),
                    ("implode_assembly_bytes", Int64(assembly_bytes)),
                    ("implode_output_rows", Int64(result.height())),
                    (
                        "query_peak_requested_bytes",
                        Int64(checked_add(assembly_bytes, inner_peak)),
                    ),
                ]
            ):
                report = report.with_column(
                    Series(
                        field[0],
                        Column[Int64](
                            List[Int64](length=report.height(), fill=field[1])
                        ),
                    )
                )
            report = report.with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=report.height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (result^, report^)

        if has_source_lists(query):
            return execute_source_lists(
                query,
                self._device,
                self._budget,
                profiling,
                observation_only=self._observation_only,
            )

        if has_source_structs(query):
            var plan = lower_source_structs(
                query, memory_limit_bytes=self._budget
            )
            var budget = self._budget
            if budget >= 0:
                budget -= plan.staging_bytes
            var runtime = MetalRuntime(self._device, memory_limit_bytes=budget)
            runtime._observation_only = self._observation_only
            runtime._execution_settings = self._execution_settings.copy()
            runtime._text_context = self._text_context.copy()
            var executed = runtime._run(
                plan.query, profiling, original_entry=False
            )
            var result = executed[
                0
            ].copy() if self._observation_only else attach_source_fields(
                executed[0], plan
            )
            var report = executed[1].copy()
            var peak_field = (
                "query_peak_requested_bytes" if "query_peak_requested_bytes"
                in report.columns() else "peak_requested_bytes"
            )
            var inner_peak = 0
            for row in range(report.height()):
                inner_peak = max(
                    inner_peak, Int(report.item(row, peak_field).int64())
                )
            for field in List[Tuple[String, Int64]](
                [
                    ("source_struct_staging_bytes", Int64(plan.staging_bytes)),
                    (
                        "query_peak_requested_bytes",
                        Int64(checked_add(plan.staging_bytes, inner_peak)),
                    ),
                ]
            ):
                report = report.with_column(
                    Series(
                        field[0],
                        Column[Int64](
                            List[Int64](length=report.height(), fill=field[1])
                        ),
                    )
                )
            report = report.with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=report.height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (result^, report^)
        if has_scalar_modes(query):
            var executed = self._run(
                lower_scalar_modes(query), profiling, original_entry=False
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
        if has_cut_operations(query):
            var lowered = lower_cuts(query)
            var staging_bytes = cut_staging_bytes(lowered)
            var budget = self._budget
            if budget >= 0:
                if staging_bytes > budget:
                    raise Error(
                        "Metal CUT scalar staging exceeds payload budget"
                    )
                budget -= staging_bytes
            var runtime = MetalRuntime(self._device, memory_limit_bytes=budget)
            runtime._observation_only = self._observation_only
            runtime._execution_settings = self._execution_settings.copy()
            runtime._text_context = self._text_context.copy()
            var executed = runtime._run(
                lowered, profiling, original_entry=False
            )
            var report = executed[1].copy()
            var peak_field = (
                "query_peak_requested_bytes" if "query_peak_requested_bytes"
                in report.columns() else "peak_requested_bytes"
            )
            var inner_peak = 0
            for row in range(report.height()):
                inner_peak = max(
                    inner_peak, Int(report.item(row, peak_field).int64())
                )
            report = report.with_column(
                Series(
                    "cut_staging_bytes",
                    Column[Int64](
                        List[Int64](
                            length=report.height(), fill=Int64(staging_bytes)
                        )
                    ),
                )
            )
            report = report.with_column(
                Series(
                    "query_peak_requested_bytes",
                    Column[Int64](
                        List[Int64](
                            length=report.height(),
                            fill=Int64(checked_add(staging_bytes, inner_peak)),
                        )
                    ),
                )
            )
            report = report.with_column(
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

        if has_terminal_structs(query):
            var plan = lower_terminal_structs(query)
            var executed = self._run(
                plan.query, profiling, original_entry=False
            )
            if self._observation_only:
                return executed^
            var result = assemble_structs(executed[0], plan)
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
            return (result^, report^)
        if has_scalar_projections(query):
            var executed = self._run(
                lower_scalar_projections(query), profiling, original_entry=False
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
        if has_literal_text_parsing(query):
            var executed = self._run(
                fold_literal_text_parsing(query),
                profiling,
                original_entry=False,
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
                lower_partition_aggregates(query),
                profiling,
                original_entry=False,
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
            var executed = self._run(
                lower_spearman(query), profiling, original_entry=False
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
        if has_aggregate_compositions(query):
            var executed = self._run(
                lower_aggregate_compositions(query, exact_float32=True),
                profiling,
                original_entry=False,
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
            var executed = self._run(
                lower_distinct_counts(query), profiling, original_entry=False
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
        if has_set_operations(query):
            var executed = self._run(
                lower_set_operations(query), profiling, original_entry=False
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
        if has_opaque_columns(query):
            var plan = prepare_opaque(query, memory_limit_bytes=self._budget)
            var opaque_staged_ns = Int(perf_counter_ns()) - start
            var budget = self._budget
            if budget >= 0:
                budget -= plan.staging_bytes
            var runtime = MetalRuntime(self._device, memory_limit_bytes=budget)
            runtime._observation_only = self._observation_only
            runtime._execution_settings = self._execution_settings.copy()
            runtime._text_context = self._text_context.copy()
            var executed = runtime._run(
                plan.query, profiling, original_entry=False
            )
            var result = executed[0].copy()
            var restoration_bytes = 0
            var opaque_restore_ns = 0
            if not self._observation_only:
                var opaque_restore_start = Int(perf_counter_ns())
                restoration_bytes = opaque_restoration_bytes(executed[0], plan)
                if (
                    self._budget >= 0
                    and restoration_bytes > self._budget - plan.staging_bytes
                ):
                    raise Error(
                        "Metal opaque restoration exceeds memory budget"
                    )
                result = restore_opaque(executed[0], plan)
                opaque_restore_ns = (
                    Int(perf_counter_ns()) - opaque_restore_start
                )
            var report = executed[1].copy()
            var peak_field = (
                "query_peak_requested_bytes" if "query_peak_requested_bytes"
                in report.columns() else "peak_requested_bytes"
            )
            var inner_peak = 0
            for row in range(report.height()):
                inner_peak = max(
                    inner_peak, Int(report.item(row, peak_field).int64())
                )
            for field in List[Tuple[String, Int64]](
                [
                    ("opaque_staging_bytes", Int64(plan.staging_bytes)),
                    ("opaque_restoration_bytes", Int64(restoration_bytes)),
                    (
                        "query_peak_requested_bytes",
                        Int64(
                            checked_add(
                                plan.staging_bytes,
                                max(inner_peak, restoration_bytes),
                            )
                        ),
                    ),
                ]
            ):
                report = report.with_column(
                    Series(
                        field[0],
                        Column[Int64](
                            List[Int64](length=report.height(), fill=field[1])
                        ),
                    )
                )
            for field in List[Tuple[String, Float64]](
                [
                    ("opaque_encode_ms", Float64(opaque_staged_ns) / 1e6),
                    ("opaque_decode_ms", Float64(opaque_restore_ns) / 1e6),
                    (
                        "query_wall_ms",
                        Float64(Int(perf_counter_ns()) - start) / 1e6,
                    ),
                ]
            ):
                report = report.with_column(
                    Series(
                        field[0],
                        Column[Float64](
                            List[Float64](length=report.height(), fill=field[1])
                        ),
                    )
                )
            return (result^, report^)
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
            var dynamic_text = has_default_qcuts(staged_query)
            var encoded = encode_query(
                staged_query, dictionary, dynamic_text=dynamic_text
            )
            var runtime = MetalRuntime(self._device, memory_limit_bytes=budget)
            runtime._observation_only = self._observation_only
            runtime._execution_settings = self._execution_settings.copy()
            runtime._text_context = self._text_context.copy()
            if dynamic_text:
                runtime._text_context = ArcPointer(TextQuery(dictionary.dtype))
            var staged_ns = Int(perf_counter_ns()) - start
            var executed = runtime._run(
                encoded, profiling, original_entry=False
            )
            if self._observation_only:
                return executed^
            var restoration_bytes = dictionary.restoration_bytes(
                executed[0], schema, runtime._text_context
            )
            if self._budget >= 0:
                if restoration_bytes > self._budget - staging_bytes:
                    raise Error(
                        "Metal dictionary restoration exceeds payload budget"
                    )
            var decode_start = Int(perf_counter_ns())
            var result = restore_frame(
                executed[0], schema, dictionary, runtime._text_context
            )
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
        if (
            has_join(query)
            or has_decimal_consumers(query)
            or has_checked_integer_consumers(query)
            or has_scalar_consumers(query)
            or has_cut_materializations(query)
            or has_implode_materializations(query)
        ):
            return execute_dag(
                query,
                self._device,
                self._budget,
                profiling,
                capabilities(),
                observation_only=self._observation_only,
                text_context=self._text_context,
            )
        return execute(
            query,
            self._device,
            self._budget,
            profiling,
            observation_only=self._observation_only,
            ordered_source=original_entry,
        )

    def describe(self, query: LazyFrame) -> String:
        return self._configured(query)._describe(query, True)

    def _describe(self, query: LazyFrame, original_entry: Bool) -> String:
        from .accelerator.gpu.dag_runtime import describe as describe_nested_dag
        from .accelerator.gpu.runtime import _preflight
        from .accelerator.grouped_stream_routes import candidate
        from .parallel import configured_workers

        try:
            var float_route = candidate(
                query,
                configured_workers(),
                self._execution_settings.batch_size,
                optimize=self._execution_settings.optimize,
                streaming=self._execution_settings.streaming,
                original_source=original_entry,
            )
            if float_route:
                return "Metal ordered grouped floating states; captured CPU single-job morsels; device sample selects inserted epochs or generic collected GPU reductions"
            if len(file_source_nodes(query)):
                var staged = stage_file_sources(
                    query,
                    empty=True,
                    optimize=self._execution_settings.optimize,
                )
                return "Metal file source decoding; " + self._describe(
                    staged[0], False
                )
            if has_selectors(query):
                return "Metal schema selector expansion; " + self._describe(
                    lower_selectors(query), False
                )
            if has_explode(query):
                return describe_nested_dag(
                    query,
                    _capabilities(),
                    ordered_source=original_entry,
                    eager_feed=not self._execution_settings.streaming,
                )
            if has_grouped_implode(query) or has_checked_grouped_feeds(
                query, eager_feed=not self._execution_settings.streaming
            ):
                _ = prepare_grouped_keys(
                    query,
                    metadata_only=True,
                    ordered_source=original_entry,
                    eager_feed=not self._execution_settings.streaming,
                )
                return "Metal resident grouped primitive Lists and typed key domains; CPU bucket producer order"
            if has_implode(query):
                var plan = lower_terminal_implode(
                    query, ordered_source=original_entry
                )
                return "Metal terminal primitive implode; " + self._describe(
                    plan.query, False
                )

            if has_source_lists(query):
                var plan = lower_source_lists(query)
                _ = _preflight(
                    plan.query,
                    source_rows=plan.source.height(),
                    terminal_boundary=True,
                )
                return "Metal resident source List len/get; " + self._describe(
                    plan.query, False
                )
            if has_source_structs(query):
                var plan = lower_source_structs(query, metadata_only=True)
                return "Metal nullable source struct fields; " + self._describe(
                    plan.query, False
                )
            if has_scalar_modes(query):
                return "Metal resident scalar mode; " + self._describe(
                    lower_scalar_modes(query), False
                )
            if has_cut_operations(query):
                return (
                    "Metal resident explicit CUT intervals; "
                    + self._describe(lower_cuts(query), False)
                )

            if has_terminal_structs(query):
                var plan = lower_terminal_structs(query)
                return "Metal terminal struct fields; " + self._describe(
                    plan.query, False
                )
            if has_scalar_projections(query):
                return "Metal resident scalar projection; " + self._describe(
                    lower_scalar_projections(query), False
                )
            if has_literal_text_parsing(query):
                return "Metal literal text parsing; " + self._describe(
                    fold_literal_text_parsing(query), False
                )
            if has_partition_aggregates(query):
                return "Metal resident partition aggregates; " + self._describe(
                    lower_partition_aggregates(query), False
                )
            if has_spearman(query):
                return "Metal resident Spearman ranks; " + self._describe(
                    lower_spearman(query), False
                )
            if has_aggregate_compositions(query):
                return (
                    "Metal resident aggregate expressions; "
                    + self._describe(
                        lower_aggregate_compositions(query, exact_float32=True),
                        False,
                    )
                )
            if has_distinct_counts(query):
                return "Metal resident distinct counts; " + self._describe(
                    lower_distinct_counts(query), False
                )
            if has_set_operations(query):
                return "Metal resident set operations; " + self._describe(
                    lower_set_operations(query), False
                )
            if has_opaque_columns(query):
                var plan = prepare_opaque(query, metadata_only=True)
                return "Metal opaque byte transport; " + self._describe(
                    plan.query, False
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
                return "Metal ordered dictionary staging; " + self._describe(
                    encoded, False
                )
            return _describe_mojo_gpu(query, original_entry)
        except error:
            return "ENGINE accel: " + String(error) + "\n"
