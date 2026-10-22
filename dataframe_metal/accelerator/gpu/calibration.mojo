"""Measured direct Mojo calibration and context-free analytical placement.

The existing accelerator calibration runner/fitter consumes these metadata
facts and complete caller-wall measurements. Imported native cost tables never
match the Mojo build, hardware and operation-family identity. Selection reads
no source values and never constructs a GPU context.
"""
from std.ffi import external_call
from std.math import isfinite
from std.os import getenv
from std.time import perf_counter_ns
from dataframe._build_identity import CPU_BUILD_ID
from dataframe.lazy import LazyFrame, AcceleratorBackend
from dataframe.json_value import JsonValue
from dataframe_metal.accelerator.cost import (
    PlanEstimate,
    CalibrationKey,
    select_auto as select_cost,
)
from dataframe_metal.accelerator.cpu_work import (
    cpu_features,
    features_from_plan,
)
from dataframe_metal.accelerator.selectors import has_selectors, lower_selectors
from dataframe_metal.accelerator.scalars import (
    has_scalar_projections,
    lower_scalar_projections,
)
from dataframe_metal.accelerator.constants import (
    has_literal_text_parsing,
    fold_literal_text_parsing,
)
from dataframe_metal.accelerator.windows import (
    has_partition_aggregates,
    lower_partition_aggregates,
)
from dataframe_metal.accelerator.paired_moments import (
    has_spearman,
    lower_spearman,
)
from dataframe_metal.accelerator.aggregates import (
    has_aggregate_compositions,
    lower_aggregate_compositions,
)
from dataframe_metal.accelerator.distinct_counts import (
    has_distinct_counts,
    lower_distinct_counts,
)
from dataframe_metal.accelerator.sets import (
    has_set_operations,
    lower_set_operations,
)
from dataframe_metal.accelerator.dictionaries import has_dictionary_columns
from dataframe_metal.accelerator.opaque import has_opaque_columns
from dataframe_metal.accelerator.source_lists import has_source_lists
from dataframe_metal.accelerator.source_structs import has_source_structs
from dataframe_metal.accelerator.joins import has_join, has_scalar_consumers
from dataframe_metal.accelerator.explodes import has_explode
from dataframe_metal.accelerator.grouped_lists import has_grouped_implode
from dataframe_metal.accelerator.grouped_feeds import has_checked_grouped_feeds
from dataframe_metal.accelerator.lists import (
    has_implode,
    has_implode_materializations,
)
from .identity import hardware, key
from .runtime import _preflight, _kind, estimate_resources


def _normalized(query: LazyFrame) raises -> LazyFrame:
    # Match MetalRuntime._run's shared, metadata-only normalization sequence.
    if has_explode(query):
        raise Error("Mojo scoped List expansion has no analytical calibration")
    if has_grouped_implode(query) or has_checked_grouped_feeds(
        query, eager_feed=True
    ):
        raise Error(
            "Mojo grouped List/key/feed staging has no analytical calibration"
        )
    if has_implode(query) or has_implode_materializations(query):
        raise Error(
            "Mojo private IMPLODE regions have no analytical calibration"
        )
    if has_selectors(query):
        return _normalized(lower_selectors(query))
    if has_source_lists(query):
        raise Error("Mojo source List staging has no analytical calibration")
    if has_source_structs(query):
        # Transport owns staging/restoration costs and a recursive execution
        # boundary, even when row pruning later drops every Struct field.
        raise Error("Mojo source Struct staging has no analytical calibration")
    if has_scalar_projections(query):
        return _normalized(lower_scalar_projections(query))
    if has_literal_text_parsing(query):
        return _normalized(fold_literal_text_parsing(query))
    if has_partition_aggregates(query):
        return _normalized(lower_partition_aggregates(query))
    if has_spearman(query):
        return _normalized(lower_spearman(query))
    if has_aggregate_compositions(query):
        return _normalized(
            lower_aggregate_compositions(query, exact_float32=True)
        )
    if has_distinct_counts(query):
        return _normalized(lower_distinct_counts(query))
    if has_set_operations(query):
        return _normalized(lower_set_operations(query))
    if has_opaque_columns(query):
        # Raw-byte domain construction is not numeric row work. In particular,
        # pruning an unused byte column cannot grant direct ordered provenance.
        raise Error("Mojo opaque staging has no analytical calibration")
    if has_dictionary_columns(query):
        # Ordered dictionary construction/restoration reads source values.
        # Its separately measured staging costs are not numeric row features.
        raise Error("Mojo dictionary staging has no analytical calibration")
    if has_join(query) or has_scalar_consumers(query):
        raise Error("Mojo resident DAG regions have no analytical calibration")
    return query.copy()


def _signature(values: List[Int]) -> String:
    var unique = List[Int]()
    for value in values:
        if value not in unique:
            unique.append(value)
    # Sorted sets keep expression/node order out of the calibration family.
    for i in range(1, len(unique)):
        var value = unique[i]
        var j = i
        while j > 0 and unique[j - 1] > value:
            unique[j] = unique[j - 1]
            j -= 1
        unique[j] = value
    var result = String()
    for value in unique:
        result += ("," if result else "") + String(value)
    return result^


def estimate(query: LazyFrame) raises -> PlanEstimate:
    """Bound numeric row execution without reading values or opening a device."""
    var normalized = _normalized(query)
    # Only an original query with no recursive structural normalization can
    # use the direct terminal ROWS token mode; mirror MetalRuntime._run.
    var original_entry = not (
        has_selectors(query)
        or has_scalar_projections(query)
        or has_literal_text_parsing(query)
        or has_partition_aggregates(query)
        or has_spearman(query)
        or has_aggregate_compositions(query)
        or has_distinct_counts(query)
        or has_set_operations(query)
    )
    var plan = _preflight(normalized, ordered_source=original_entry)
    var cpu = features_from_plan(
        plan
    ) if plan.ordered_mode == 2 else cpu_features(normalized)
    var resources = estimate_resources(plan)
    var device = cpu[0].copy()
    device[8] = Float64(resources.kernel_launches)
    device[9] = Float64(resources.synchronizations)
    device[10] = Float64(resources.upload_bytes)
    device[11] = Float64(resources.download_bytes)
    device[12] = Float64(resources.peak_bytes)
    # Retain a conservative first-use feature. Runtime-owned context reuse
    # depends on the instance's history; a measured reuse-aware cost model
    # needs its own family rather than assuming every invocation is warm.
    device[13] = 1
    device[14] = 0
    var operations = List[Int]()
    var reductions = List[Int]()
    var storage = List[Int]()
    for node in range(len(plan.code) // 4):
        operations.append(Int(plan.code[4 * node]))
    for output in plan.outputs:
        if plan.reductions:
            reductions.append(output.reduction)
    for dtype in plan.slot_dtypes:
        storage.append(Int(_kind(dtype)))
    var normalizers = List[Int](
        [
            Int(has_selectors(query)),
            Int(has_scalar_projections(query)),
            Int(has_literal_text_parsing(query)),
            Int(has_partition_aggregates(query)),
            Int(has_spearman(query)),
            Int(has_aggregate_compositions(query)),
            Int(has_distinct_counts(query)),
            Int(has_set_operations(query)),
        ]
    )
    var normalization = String()
    for flag in normalizers:
        normalization += String(flag)
    var family = (
        cpu[1]
        + ";mojo-metal-v1;ops="
        + _signature(operations)
        + ";reductions="
        + _signature(reductions)
        + ";storage="
        + _signature(storage)
        + ";normalizers="
        + normalization
    )
    if plan.ordered_mode:
        # Existing unordered samples cannot match a ranked program's added
        # sidecar/control costs. New curves require this distinct family.
        family += ";ordered-errors-v1=" + String(plan.ordered_mode)
    return PlanEstimate(
        cpu[0].copy(), device^, resources.peak_bytes, family^, "accel"
    )


def select_auto(
    query: LazyFrame,
    device: Int,
    budget: Int,
    optimize: Bool,
    streaming: Bool,
    batch_size: Int,
) -> Tuple[String, String]:
    """Apply the shared validated cost policy before any GPU initialization."""
    try:
        var path = getenv("DATAFRAME_ACCEL_COST_TABLE")
        if not path:
            return (
                "cpu",
                "auto uses CPU; no analytical calibration configured",
            )
        if CPU_BUILD_ID == "development-source-no-calibration":
            return (
                "cpu",
                "automatic calibration requires an identified CPU/provider package",
            )
        var facts = estimate(query)
        if budget >= 0 and facts.peak_bytes > budget:
            return (
                "cpu",
                "estimated Mojo Metal payload exceeds configured memory limit",
            )
        var table: JsonValue
        with open(path, "r") as file:
            table = JsonValue(file.read())
        var fraction = Float64(table.get("memory_fraction").text)
        if not isfinite(fraction) or fraction <= 0 or fraction > 1:
            return ("cpu", "invalid automatic memory headroom fraction")
        var host = hardware(device)
        var available = Int(Float64(host.available_bytes) * fraction)
        if budget >= 0:
            available = min(available, budget)
        var identity = key(device, optimize, streaming, batch_size)
        var decision = select_cost(
            facts,
            table,
            identity,
            available,
            external_call["time", Int64](Int(0)),
        )
        return (
            decision.engine.copy(),
            decision.reason
            + "; peak_bytes="
            + String(facts.peak_bytes)
            + "; budget_bytes="
            + String(available),
        )
    except error:
        return (
            "cpu",
            "automatic analytical preflight rejected: " + String(error),
        )


@fieldwise_init
struct Measurements(Copyable):
    var identity: CalibrationKey
    var facts: PlanEstimate
    var cold_ns: Int
    var cpu_ns: List[Int]
    var accelerator_ns: List[Int]


def measure[
    B: AcceleratorBackend
](
    query: LazyFrame,
    accelerator: B,
    identity: CalibrationKey,
    repetitions: Int = 7,
) raises -> Measurements:
    """Time complete CPU/Mojo materialization with parity outside each timing.

    These are raw samples, not fitted or validated curves. The existing
    scripts/bench_accelerator_cost.py fits nonnegative models and requires
    independent held-out queries/distributions before automatic use.
    """
    if identity.provider != "metal" or not identity.device.startswith(
        "mojo-metal;"
    ):
        raise Error(
            "Calibration requires a direct Mojo Metal provider identity"
        )
    if repetitions < 1:
        raise Error("Calibration repetitions must be positive")
    var facts = estimate(query)
    var expected = query.collect(
        engine="cpu",
        optimize=identity.optimize,
        streaming=identity.streaming,
        batch_size=identity.batch_size,
    )
    var started = perf_counter_ns()
    var first = query.collect(
        engine="accel",
        accelerator=accelerator,
        optimize=identity.optimize,
        streaming=identity.streaming,
        batch_size=identity.batch_size,
    )
    var cold = Int(perf_counter_ns() - started)
    if cold <= 0 or not first.equals(expected):
        raise Error("Cold Mojo calibration timing/parity failed")
    for _ in range(2):
        _ = query.collect(
            engine="cpu",
            optimize=identity.optimize,
            streaming=identity.streaming,
            batch_size=identity.batch_size,
        )
        _ = query.collect(
            engine="accel",
            accelerator=accelerator,
            optimize=identity.optimize,
            streaming=identity.streaming,
            batch_size=identity.batch_size,
        )
    var cpu = List[Int](capacity=repetitions)
    var gpu = List[Int](capacity=repetitions)
    for i in range(repetitions):
        var cpu_time = 0
        var gpu_time = 0
        for j in range(2):
            var on_gpu = (i + j) % 2 == 0
            started = perf_counter_ns()
            var result = query.collect(
                engine="accel",
                accelerator=accelerator,
                optimize=identity.optimize,
                streaming=identity.streaming,
                batch_size=identity.batch_size,
            ) if on_gpu else query.collect(
                engine="cpu",
                optimize=identity.optimize,
                streaming=identity.streaming,
                batch_size=identity.batch_size,
            )
            var elapsed = Int(perf_counter_ns() - started)
            if elapsed <= 0 or not result.equals(expected):
                raise Error("Warm Mojo calibration timing/parity failed")
            if on_gpu:
                gpu_time = elapsed
            else:
                cpu_time = elapsed
        cpu.append(cpu_time)
        gpu.append(gpu_time)
    return Measurements(identity.copy(), facts^, cold, cpu^, gpu^)
