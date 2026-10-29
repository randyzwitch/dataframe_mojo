"""Public lazy query construction and backend dispatch."""
from dataframe.io.file_sources import has_pattern, resolve_paths
from dataframe.planning.node_builders import (
    _aggregate_node,
    _concat_node,
    _drop_node,
    _dynamic_window_node,
    _explode_node,
    _fill_null_strategy_node,
    _filter_node,
    _frame_scan_node,
    _gather_every_node,
    _gather_node,
    _grouping_sets_node,
    _hconcat_node,
    _insert_column_node,
    _ipc_scan_node,
    _join_node,
    _ndjson_scan_node,
    _parquet_scan_node,
    _pivot_node,
    _reverse_node,
    _rolling_window_node,
    _row_index_node,
    _select_node,
    _shift_node,
    _slice_node,
    _sort_node,
    _tail_node,
    _unique_node,
    _unnest_node,
    _unpivot_node,
    _with_columns_node,
)
from dataframe._accel_provider import (
    describe as _describe_accel,
    execute as _execute_accel,
    execute_auto as _execute_auto_accel,
    execute_profiled as _profile_accel,
    handoff_estimate as _handoff_estimate,
    handoff_execute as _handoff_execute,
    handoff_supports as _handoff_supports,
    installed as _accel_installed,
    select_auto as _select_auto_accel,
)
from dataframe.storage.column import (
    Column,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.execution_backend import (
    _BackendDecision,
    _select_backend,
)
from dataframe.execution_report import (
    ExecutionReport,
)
from dataframe.execution_settings import (
    ExecutionSettings,
)
from dataframe.expr import (
    Expr,
    changes_length,
    col,
    exclude,
    has_grouping,
    is_grouping,
    lit,
    null,
)
from dataframe.frame import (
    DataFrame,
    _describe_percentiles,
    concat,
)
from dataframe.aggregation.group_by import (
    _reduce_column,
)
from dataframe.io.csv import (
    CsvSchema,
)
from dataframe.io.csv_types import (
    _CsvScanOptions,
)
from dataframe.io.json import (
    _options as _json_options,
)
from dataframe.join_order import (
    JoinEdge,
)
from dataframe.join_type import (
    JOIN_CROSS,
    join_code,
)
from dataframe.optimizer.expressions import (
    _fold_casts,
)
from dataframe.optimizer.passes import (
    plan_align_union_inputs,
    plan_drop_constant_filters,
    plan_eliminate_common_subexpressions,
    plan_filter_shared_subplans,
    plan_fold_casts,
    plan_fuse_top_k,
    plan_join_input_columns,
    plan_merge_filters,
    plan_narrow_join_inputs,
    plan_optimized,
    plan_passes,
    plan_prune_outputs,
    plan_push_predicates,
    plan_push_predicates_now,
    plan_push_projections,
    plan_push_slices,
    plan_slice_over_ipc,
    plan_split_filters,
    plan_swap_down,
    plan_swap_in_place,
    plan_transfer_key_filter,
    plan_unique_as_aggregate,
    plan_unstick,
)
from dataframe.parallel import (
    Crew,
)
from dataframe.placement import (
    calibration_path,
)
from dataframe.devices import DeviceInfo, PipelineFeatures, TransferModel
from dataframe.plan_types import (
    PlanEstimate,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.explain import (
    plan_describe,
    plan_label,
)
from dataframe.planning.graph import (
    plan_aggregated,
    plan_available,
    plan_canonical_slot,
    plan_collect_graph,
    plan_columns_of,
    plan_compact_scan_slots,
    plan_cover,
    plan_ends_in_breaker,
    plan_estimated_rows,
    plan_filtered_table,
    plan_filters_rows,
    plan_forget_schema,
    plan_forget_schemas_above,
    plan_graph_step,
    plan_grouped_on,
    plan_has_join,
    plan_has_kind,
    plan_height_bound,
    plan_indexed_relation,
    plan_input_name,
    plan_keeps_column,
    plan_key_source,
    plan_known_height,
    plan_largest_scan,
    plan_live_parents,
    plan_measurable,
    plan_node_hash,
    plan_order_free_above,
    plan_parents,
    plan_progression_key,
    plan_projection_visits,
    plan_provider_scan,
    plan_relation_estimate,
    plan_remember_schemas,
    plan_reorder,
    plan_repeats_a_scan,
    plan_same_node,
    plan_share_copy_schemas,
    plan_spine_frame,
    plan_spine_rows,
    plan_subplan_classes,
    plan_topological,
)
from dataframe.planning.model import (
    AGG,
    ASOF,
    CONCAT,
    DROP,
    EXPLODE,
    FILL_NULL_STRATEGY,
    FILTER,
    GATHER,
    GATHER_EVERY,
    GROUPING_SETS,
    GROUP_BY_DYNAMIC,
    HANDOFF_AUTO,
    HANDOFF_FORCED,
    HCONCAT,
    INSERT_COLUMN,
    JOIN,
    PIVOT,
    PlanNode,
    REVERSE,
    ROLLING,
    ROW_INDEX,
    SCAN_FRAME,
    SCAN_IPC,
    SCAN_NDJSON,
    SCAN_PARQUET,
    SELECT,
    SHIFT,
    SLICE,
    SORT,
    TAIL,
    UNIQUE,
    UNNEST,
    UNPIVOT,
    WITH_COLUMNS,
)
from dataframe.planning.physical import (
    plan_any_selective,
    plan_cost_order_joins,
    plan_direct_key,
    plan_joins_below,
    plan_key_set_side,
    plan_order_chain,
    plan_order_joins,
    plan_probe_largest_input,
    plan_probe_largest_inputs,
    plan_push_join_key_sets,
    plan_restore_measured_chain,
    plan_selective_on,
    plan_selectivity,
    plan_side_comparisons,
    plan_swap_join_sides,
    plan_swap_left_join_sides,
)
from dataframe.planning.cost.aggregation import (
    plan_batch_states_win,
    plan_row_number_shape,
    plan_sampled_groups,
)
from dataframe.planning.properties import (
    _bare_column,
    _is_scan,
    _same_names,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from dataframe.planning.scans import (
    _csv_scan_node,
)
from dataframe.query_execution.executor import (
    plan_both_inputs,
    plan_concat,
    plan_execute,
    plan_execute_node,
)
from dataframe.query_execution.materialize import (
    plan_prefetch_breakers,
    plan_read_whole,
    plan_share_subplans,
)
from dataframe.query_execution.operators import (
    plan_join_frames,
)
from dataframe.query_execution.pipeline import (
    plan_reduce_by_probe_keys,
    plan_stream_execute,
)
from dataframe.query_execution.placement import (
    plan_handoff_input,
    plan_handoff_nodes,
    plan_handoff_plan,
    plan_handoff_region,
    plan_placement_lines,
)
from dataframe.query_execution.reporting import (
    plan_record,
)
from dataframe.series import (
    Series,
)
from dataframe.time_windows import (
    negate_period,
)
from dataframe.value import (
    AnyValue,
)
from std.collections import (
    Dict,
    Optional,
)
from std.io import (
    RawFileDescriptor,
)
from std.memory import (
    ArcPointer,
)
from std.os import (
    getenv,
)


struct LazyFrame(Copyable, Deinitable, Movable):
    """A deferred query; build it with DataFrame.lazy() or scan_csv()."""

    var _plan: QueryContext

    def __init__(out self, frame: DataFrame):
        self._plan = _new_context(
            [_frame_scan_node(slot=0)],
            [frame.copy()],
            [Optional[CsvSchema]()],
        )

    def __init__(
        out self,
        var nodes: List[PlanNode],
        var frames: List[DataFrame],
        var schemas: List[Optional[CsvSchema]],
    ):
        self._plan = _new_context(nodes^, frames^, schemas^)

    def _with_accelerator_settings(
        self, optimize: Bool, streaming: Bool, batch_size: Int
    ) raises -> Self:
        var configured = self.copy()
        configured._plan._accelerator_settings = ExecutionSettings(
            optimize, streaming, batch_size
        )
        return configured^

    def _with_handoff(self, mode: Int = HANDOFF_FORCED) -> Self:
        """This plan, run by the CPU executor with each node offered to the
        registered accelerator provider (`_handoff_region`). A provider whose
        device shares memory with the CPU runs its `execute` this way, with
        HANDOFF_FORCED; `engine="auto"` uses HANDOFF_AUTO."""
        var plan = self.copy()
        plan._plan._handoff = mode
        return plan^

    def _handoff_input(self, index: Int) -> Tuple[Int, List[Int]]:
        return plan_handoff_input(self._plan, index=index)

    def _handoff_plan(
        self, index: Int, filters: List[Int], columns: DataFrame
    ) -> Self:
        return _from_context(
            plan_handoff_plan(
                self._plan, index=index, filters=filters, columns=columns
            )
        )

    def _handoff_region(
        self, index: Int, streaming: Bool, batch_size: Int
    ) raises -> Optional[DataFrame]:
        return plan_handoff_region(
            self._plan, index=index, streaming=streaming, batch_size=batch_size
        )

    def _placement_lines(self) raises -> String:
        return plan_placement_lines(self._plan)

    def _handoff_nodes(self) raises -> List[String]:
        return plan_handoff_nodes(self._plan)

    def _record(
        self,
        node: Int,
        executor: String,
        input_rows: Int,
        output_rows: Int,
        *,
        executions: Int = 1,
        build_rows: Int = 0,
        builds: Int = 0,
        algorithm: String = "",
        build_side: String = "",
        busy_ns: Int = 0,
        wall_ns: Int = 0,
    ):
        plan_record(
            self._plan,
            node=node,
            executor=executor,
            input_rows=input_rows,
            output_rows=output_rows,
            executions=executions,
            build_rows=build_rows,
            builds=builds,
            algorithm=algorithm,
            build_side=build_side,
            busy_ns=busy_ns,
            wall_ns=wall_ns,
        )

    def _push(self, var node: PlanNode) -> Self:
        var result = self.copy()
        node.left = len(result._plan._nodes) - 1
        result._plan._nodes.append(node^)
        return result^

    def filter(self, predicate: Expr) -> Self:
        return self._push(_filter_node(exprs=[predicate.copy()]))

    def select(self, expr: Expr) -> Self:
        return self.select_exprs([expr.copy()])

    def select(self, names: List[String]) -> Self:
        var exprs = List[Expr]()
        for name in names:
            exprs.append(col(name))
        return self.select_exprs(exprs)

    def select_exprs(self, exprs: List[Expr]) -> Self:
        return self._push(_select_node(exprs=exprs))

    def with_columns(self, expr: Expr) -> Self:
        return self.with_columns([expr.copy()])

    def with_columns(self, exprs: List[Expr]) -> Self:
        return self._push(_with_columns_node(exprs=exprs))

    def group_by(
        self, keys: List[String], *, maintain_order: Bool = False
    ) -> LazyGroupBy:
        return LazyGroupBy(
            self.copy(), keys.copy(), maintain_order, List[Expr]()
        )

    def group_by(
        self, key: String, *, maintain_order: Bool = False
    ) -> LazyGroupBy:
        return LazyGroupBy(self.copy(), [key], maintain_order, List[Expr]())

    def group_by(
        self, keys: List[Expr], *, maintain_order: Bool = False
    ) -> LazyGroupBy:
        """Group by computed keys, as DataFrame.group_by(List[Expr]): each
        key is evaluated once and named by its output name, and the
        aggregations still read the input's columns. Keys that are columns
        under their own names group as names do."""
        var names = List[String](capacity=len(keys))
        var computed = False
        for key in keys:
            names.append(key._name)
            computed = computed or not _bare_column(key)
        return LazyGroupBy(
            self.copy(),
            names^,
            maintain_order,
            keys.copy() if computed else List[Expr](),
        )

    def group_by_rollup(self, keys: List[String]) -> LazyGroupingSets:
        """Expand to the rollup sets (k1, k2, ...), (k1, ..., k(n-1)),
        ..., (k1), and () in one plan. Finish with `agg`; see
        LazyGroupingSets for the output."""
        var sets = List[List[String]]()
        for n in range(len(keys), -1, -1):
            sets.append(List[String](keys[:n]))
        return LazyGroupingSets(self.copy(), sets^)

    def group_by_cube(self, keys: List[String]) -> LazyGroupingSets:
        """Every subset of the keys as a grouping set, for 2^n sets. A
        repeated key in `keys` yields repeated sets. Finish with `agg`."""
        var sets = List[List[String]]()
        var n = len(keys)
        for mask in range((1 << n) - 1, -1, -1):
            var members = List[String]()
            for j in range(n):
                if (mask >> (n - 1 - j)) & 1 == 1:
                    members.append(keys[j])
            sets.append(members^)
        return LazyGroupingSets(self.copy(), sets^)

    def group_by_grouping_sets(
        self, sets: List[List[String]]
    ) -> LazyGroupingSets:
        """One grouping per listed set; an empty set produces the grand
        total. A set listed twice yields its rows twice. Finish with `agg`."""
        return LazyGroupingSets(self.copy(), sets.copy())

    def sort(
        self,
        by: List[String],
        descending: Bool = False,
        nulls_last: Bool = True,
        *,
        maintain_order: Bool = False,
    ) -> Self:
        """Sort by `by`. Rows with equal keys have unspecified order unless
        `maintain_order` preserves their input order."""
        var n = len(by)
        return self._push(
            _sort_node(
                keys=by,
                maintain_order=maintain_order,
                descending=List[Bool](length=n, fill=descending),
                nulls_last=List[Bool](length=n, fill=nulls_last),
            )
        )

    def sort(
        self,
        by: List[String],
        *,
        descending: List[Bool],
        nulls_last: List[Bool],
        maintain_order: Bool = False,
    ) raises -> Self:
        """Sort with a direction and null placement per key, as the eager
        DataFrame.sort takes them; equal keys in no particular order unless
        `maintain_order`."""
        if len(descending) != len(by) or len(nulls_last) != len(by):
            raise Error("sort needs one descending and nulls_last flag per key")
        return self._push(
            _sort_node(
                keys=by,
                maintain_order=maintain_order,
                descending=descending,
                nulls_last=nulls_last,
            )
        )

    def sort(
        self,
        by: String,
        descending: Bool = False,
        *,
        maintain_order: Bool = False,
    ) -> Self:
        return self.sort([by], descending, maintain_order=maintain_order)

    def sort(
        self,
        by: Expr,
        descending: Bool = False,
        nulls_last: Bool = True,
        *,
        maintain_order: Bool = False,
    ) raises -> Self:
        """Sort by an expression, evaluated once before sorting."""
        return self.sort(
            [by.copy()], descending, nulls_last, maintain_order=maintain_order
        )

    def sort(
        self,
        by: List[Expr],
        descending: Bool = False,
        nulls_last: Bool = True,
        *,
        maintain_order: Bool = False,
    ) raises -> Self:
        """Sort by expression keys with common direction and null placement."""
        var keys = self._input_schema()._sort_key_exprs(by)
        return self._sort_by_keys(
            keys,
            List[Bool](length=len(keys), fill=descending),
            List[Bool](length=len(keys), fill=nulls_last),
            maintain_order,
        )

    def sort(
        self,
        by: List[Expr],
        *,
        descending: List[Bool],
        nulls_last: List[Bool],
        maintain_order: Bool = False,
    ) raises -> Self:
        """Sort by expression keys with a direction and null placement each."""
        return self._sort_by_keys(
            self._input_schema()._sort_key_exprs(by),
            descending,
            nulls_last,
            maintain_order,
        )

    def _sort_by_keys(
        self,
        keys: List[Expr],
        descending: List[Bool],
        nulls_last: List[Bool],
        maintain_order: Bool,
    ) raises -> Self:
        if len(descending) != len(keys) or len(nulls_last) != len(keys):
            raise Error("sort needs one descending and nulls_last flag per key")
        if len(keys) == 0:
            return self.copy()
        var names = List[String](capacity=len(keys))
        for key in keys:
            names.append(key._name)
        return (
            self.with_columns(keys)
            .sort(
                names,
                descending=descending,
                nulls_last=nulls_last,
                maintain_order=maintain_order,
            )
            .drop(names)
        )

    def gather_every(self, n: Int, offset: Int = 0) raises -> Self:
        """Take every nth row from a nonnegative offset; n must be positive."""
        if n <= 0 or offset < 0:
            raise Error("gather_every needs positive n and nonnegative offset")
        return self._push(_gather_every_node(row_offset=offset, stride=n))

    def insert_column(self, index: Int, column: Series) raises -> Self:
        """Insert a column by position; validate its height when collecting."""
        var schema = self._input_schema()
        var position = index + schema.width() if index < 0 else index
        if position < 0 or position > schema.width():
            raise Error("Column insertion index out of bounds")
        if column.name() in schema.columns():
            raise Error("Duplicate column name: " + column.name())
        return self._binary(
            DataFrame([column.copy()]).lazy(),
            _insert_column_node(position=position),
        )

    def slice(self, offset: Int, length: Int = -1) -> Self:
        return self._push(_slice_node(row_offset=offset, row_count=length))

    def head(self, n: Int = 5) -> Self:
        return self.slice(0, n)

    def limit(self, n: Int = 5) -> Self:
        return self.head(n)

    def pivot(
        self,
        on: String,
        on_columns: List[String],
        *,
        index: List[String],
        values: String,
        aggregate_function: String = "",
    ) raises -> Self:
        """Long to wide, as DataFrame.pivot, with the new columns named up
        front so the plan's schema
        is known before it runs: one column per entry of `on_columns`, in
        that order, holding the cells whose `on` value reads as that name
        ("null" for a null). A name no row has gives a column of nulls;
        rows whose `on` value is not listed still give their index key a
        row. Filters above it on index columns alone move below it."""
        var seen = Dict[String, Bool]()
        for name in index:
            seen[name] = True
        for name in on_columns:
            if name in seen:
                raise Error("pivot column name collides: " + name)
            seen[name] = True
        return self._push(
            _pivot_node(
                index=index,
                on=on,
                values=values,
                output_columns=on_columns,
                aggregate_function=aggregate_function,
            )
        )

    def pivot(
        self,
        on: String,
        on_columns: Series,
        *,
        index: List[String],
        values: String,
        aggregate_function: String = "",
    ) raises -> Self:
        """`pivot` with the new columns given as `on` values, named as
        DataFrame.pivot names them."""
        var names = List[String](capacity=len(on_columns))
        for k in range(len(on_columns)):
            names.append(String(on_columns.get(k)))
        return self.pivot(
            on,
            names,
            index=index,
            values=values,
            aggregate_function=aggregate_function,
        )

    def unique(
        self,
        subset: List[String] = List[String](),
        *,
        keep: String = "any",
        maintain_order: Bool = False,
    ) -> Self:
        return self._push(
            _unique_node(
                subset=subset, keep=keep, maintain_order=maintain_order
            )
        )

    def drop(self, names: List[String]) -> Self:
        return self._push(_drop_node(columns=names))

    def rename(
        self, mapping: Dict[String, String], *, strict: Bool = True
    ) raises -> Self:
        """Rename columns by old name, with the result and errors of
        DataFrame.rename. This is a select of every
        input column, aliased where renamed, so filters and column
        selections above it move below it in the input's names. The
        input's names are resolved here, as collect_schema resolves them,
        without reading rows. With strict=False, names the input lacks are
        ignored."""
        var names = self._input_schema().columns()
        if strict:
            for old in mapping.keys():
                if old not in names:
                    raise Error("Unknown column: " + old)
        var exprs = List[Expr](capacity=len(names))
        var seen = Dict[String, Bool]()
        var renamed = False
        for name in names:
            var output = mapping.get(name, name)
            if output in seen:
                raise Error("Rename produces duplicate column name: " + output)
            seen[output] = True
            if output != name:
                exprs.append(col(name).alias(output))
                renamed = True
            else:
                exprs.append(col(name))
        if not renamed:
            return self.copy()
        return self.select_exprs(exprs)

    def rename(
        self, function: def(String) raises thin -> String
    ) raises -> Self:
        """Rename every column using a Mojo function, resolving only schema."""
        var mapping = Dict[String, String]()
        for name in self._input_schema().columns():
            mapping[name] = function(name)
        return self.rename(mapping)

    def rename(self, function: def(String) thin -> String) raises -> Self:
        """Rename every column using a Mojo function, resolving only schema."""
        var mapping = Dict[String, String]()
        for name in self._input_schema().columns():
            mapping[name] = function(name)
        return self.rename(mapping)

    def cast(
        self, dtypes: Dict[String, String], *, strict: Bool = True
    ) -> Self:
        """Cast named columns in place, as DataFrame.cast: a with_columns of
        `col(name).cast(dtype, strict)` per entry, so
        filters on other columns move below it and a cast column nothing
        above reads is not computed. An unknown name raises when the plan
        runs."""
        var casts = List[Expr](capacity=len(dtypes))
        for item in dtypes.items():
            casts.append(col(item.key).cast(item.value, strict))
        if len(casts) == 0:
            return self.copy()
        return self.with_columns(casts)

    def concat(self, other: Self) -> Self:
        """This plan's rows followed by `other`'s (SQL UNION ALL), as one
        plan: both inputs are optimized with the rest of it, and a column
        selection above reaches both scans. The inputs must have the same
        column names and types; `other`'s columns are taken in this plan's
        order. SQL UNION is `concat` followed by `unique`."""
        return self._binary(other, _concat_node())

    def _binary(self, other: Self, var node: PlanNode) -> Self:
        """`node` over this plan (left) and `other`'s (right), as one plan."""
        var result = self.copy()
        var left_root = len(result._plan._nodes) - 1
        var shift = len(result._plan._nodes)
        var frame_shift = len(result._plan._frames)
        for frame in other._plan._frames:
            result._plan._frames.append(frame.copy())
        for schema in other._plan._schemas:
            result._plan._schemas.append(schema.copy())
        for node in other._plan._nodes:
            var copied = node.copy()
            if copied.left >= 0:
                copied.left += shift
            if copied.right >= 0:
                copied.right += shift
            if _is_scan(copied.kind):
                copied.set_source_slot(copied.source_slot() + frame_shift)
            result._plan._nodes.append(copied^)
        node.left = left_root
        node.right = len(result._plan._nodes) - 1
        result._plan._nodes.append(node^)
        return result^

    def sum(self) raises -> Self:
        """Sum each numeric column; unsupported columns are null."""
        return self._reduce_frame("sum")

    def mean(self) raises -> Self:
        """Mean each numeric column; unsupported columns are null."""
        return self._reduce_frame("mean")

    def min(self) raises -> Self:
        """Minimum of each ordered column; nested columns are null."""
        return self._reduce_frame("min")

    def max(self) raises -> Self:
        """Maximum of each ordered column; nested columns are null."""
        return self._reduce_frame("max")

    def median(self) raises -> Self:
        """Median of each numeric column; unsupported columns are null."""
        return self._reduce_frame("median")

    def count(self) raises -> Self:
        """Count non-null values in every column, as UInt32."""
        return self._reduce_frame("count")

    def product(self) raises -> Self:
        """Product of each numeric column; unsupported columns are null."""
        return self._reduce_frame("product")

    def std(self, ddof: Int = 1) raises -> Self:
        """Column-wise standard deviation, with nonnegative ddof."""
        return self._reduce_frame("std", ddof=ddof)

    def var(self, ddof: Int = 1) raises -> Self:
        """Column-wise variance, with nonnegative ddof."""
        return self._reduce_frame("var", ddof=ddof)

    def quantile(
        self, quantile: Float64, interpolation: String = "nearest"
    ) raises -> Self:
        """Numeric column quantiles; interpolation defaults to nearest."""
        return self._reduce_frame(
            "quantile", quantile=quantile, interpolation=interpolation
        )

    def _reduce_frame(
        self,
        reduction: String,
        *,
        ddof: Int = 1,
        quantile: Float64 = 0.5,
        interpolation: String = "nearest",
    ) raises -> Self:
        var expressions = self._input_schema()._frame_reduction_exprs(
            reduction, ddof, quantile, interpolation
        )
        if len(expressions) == 0:
            return DataFrame(List[Series](), height=1).lazy()
        return self.select_exprs(expressions)

    def describe(
        self,
        percentiles: List[Float64] = [0.25, 0.5, 0.75],
        interpolation: String = "nearest",
    ) raises -> DataFrame:
        """DataFrame.describe's summary of this plan's result: one select of
        the summary reductions over the plan, collected without
        materializing all rows."""
        var schema = self._input_schema()
        var qs = _describe_percentiles(percentiles)
        var exprs = schema._describe_exprs(qs, interpolation)
        var stats = self.select_exprs(exprs).collect() if len(
            exprs
        ) > 0 else DataFrame(List[Series](), height=1)
        return schema._described(qs, stats)

    def clear(self, n: Int = 0) raises -> Self:
        """This plan's schema with n rows of nulls, as DataFrame.clear. With
        n = 0 it is head(0) of this plan; otherwise a new
        plan over a frame of nulls built from the schema."""
        if n < 0:
            raise Error("clear: n must be nonnegative")
        if n == 0:
            return self.head(0)
        return Self(self._input_schema().clear(n))

    def gather(
        self, indices: List[Int], *, null_on_oob: Bool = False
    ) raises -> Self:
        """The rows at `indices`, in that order; see DataFrame.gather."""
        var values = List[Int64](capacity=len(indices))
        for i in indices:
            values.append(Int64(i))
        return self.gather(
            Self(DataFrame([Series("index", Column[Int64](values^))])),
            null_on_oob=null_on_oob,
        )

    def gather(self, indices: Self, *, null_on_oob: Bool = False) -> Self:
        """The rows at the positions `indices` (a plan with one integer
        column) lists, in its order: a negative index
        counts from the end, a null index gives a row of nulls, and an
        index out of range raises when the plan runs, or with
        null_on_oob=True gives a row of nulls. Filters and slices above it
        stay above it."""
        return self._binary(indices, _gather_node(null_on_oob=null_on_oob))

    def top_k(
        self, k: Int, by: List[String], *, reverse: Bool = False
    ) raises -> Self:
        """The k rows with greatest keys in `by`, using descending order
        (ascending with reverse=True), nulls last, and a top-k selection."""
        if k < 0:
            raise Error("k must be nonnegative")
        var n = len(by)
        return self.sort(
            by,
            descending=List[Bool](length=n, fill=not reverse),
            nulls_last=List[Bool](length=n, fill=True),
        ).head(k)

    def top_k(
        self, k: Int, by: String, *, reverse: Bool = False
    ) raises -> Self:
        return self.top_k(k, [by], reverse=reverse)

    def bottom_k(
        self, k: Int, by: List[String], *, reverse: Bool = False
    ) raises -> Self:
        """The k rows DataFrame.bottom_k gives: a sort by `by` ascending
        (descending with reverse=True), nulls last, then head(k)."""
        if k < 0:
            raise Error("k must be nonnegative")
        var n = len(by)
        return self.sort(
            by,
            descending=List[Bool](length=n, fill=reverse),
            nulls_last=List[Bool](length=n, fill=True),
        ).head(k)

    def bottom_k(
        self, k: Int, by: String, *, reverse: Bool = False
    ) raises -> Self:
        return self.bottom_k(k, [by], reverse=reverse)

    def tail(self, n: Int = 5) -> Self:
        """The last n rows; a negative n drops the first -n rows. See
        DataFrame.tail. Filters and slices written above it stay above
        it."""
        return self._push(_tail_node(count=n))

    def reverse(self) -> Self:
        """The rows in reverse order; see DataFrame.reverse."""
        return self._push(_reverse_node())

    def shift(self, n: Int = 1, *, fill_value: Optional[Expr] = None) -> Self:
        """Every column moved n rows later (earlier when negative), the
        vacated rows null or `fill_value`; see DataFrame.shift."""
        var exprs = List[Expr]()
        if fill_value:
            exprs.append(fill_value.value().copy())
        return self._push(_shift_node(fill=exprs, count=n))

    def with_row_index(self, name: String = "index", offset: Int64 = 0) -> Self:
        """Prepend an Int64 row number starting at offset; see
        DataFrame.with_row_index. The numbers count the rows this plan
        produces here, so filters and slices above it stay above it."""
        return self._push(_row_index_node(name=name, start=Int(offset)))

    def explode(self, columns: List[String]) -> Self:
        """One row per list element; see DataFrame.explode."""
        return self._push(_explode_node(columns=columns))

    def explode(self, column: String) -> Self:
        return self.explode([column])

    def unnest(self, column: String) -> Self:
        """Replace a struct column with its fields; see DataFrame.unnest."""
        return self._push(_unnest_node(column=column))

    def unpivot(
        self,
        on: List[String] = List[String](),
        index: List[String] = List[String](),
        *,
        variable_name: String = "variable",
        value_name: String = "value",
    ) -> Self:
        """Wide to long: one row per (input row, `on` column), with the
        same output and errors as DataFrame.unpivot. `on` defaults to every
        non-index column. Filters and slices above it stay above it."""
        return self._push(
            _unpivot_node(
                columns=on,
                index=index,
                variable_name=variable_name,
                value_name=value_name,
            )
        )

    def _input_schema(self) raises -> DataFrame:
        """This plan's output names and types with no rows, using
        `collect_schema` (every scan yields no rows), for
        the methods below that choose columns by type or take every
        column, by expanding selectors against the input schema."""
        return self._execute(len(self._plan._nodes) - 1, True).clear()

    def fill_null(
        self, value: String, subset: List[String] = List[String]()
    ) raises -> Self:
        """Fill nulls in string columns with a string; see the Expr
        overload."""
        return self.fill_null(lit(value), subset)

    def fill_null(
        self, value: Expr, subset: List[String] = List[String]()
    ) raises -> Self:
        """Fill nulls with a scalar value, choosing columns from this plan's
        schema: a with_columns of `col(c).fill_null(value)` for each
        selected column.
        Unknown subset names raise here, while the plan is built."""
        var fills = self._input_schema()._fill_null_exprs(value, subset)
        if len(fills) == 0:
            return self.copy()
        return self.with_columns(fills)

    def fill_null(
        self,
        *,
        strategy: String,
        limit: Optional[Int] = None,
        subset: List[String] = List[String](),
    ) raises -> Self:
        """Fill nulls by strategy over the whole input; see DataFrame.fill_null."""
        var run_limit = self._input_schema()._fill_strategy_limit(
            strategy, limit, subset
        )
        return self._push(
            _fill_null_strategy_node(
                strategy=strategy, limit=run_limit, subset=subset
            )
        )

    def fill_nan(self, value: Float64) raises -> Self:
        """Replace floating NaNs with a numeric scalar, preserving column dtypes."""
        return self.fill_nan(lit(value))

    def fill_nan(self, value: Expr) raises -> Self:
        """Replace NaNs in floating columns; see DataFrame.fill_nan."""
        return self.with_columns(self._input_schema()._fill_nan_exprs(value))

    def drop_nans(self, subset: List[String] = List[String]()) raises -> Self:
        """Remove NaN rows, retaining nulls; see DataFrame.drop_nans."""
        var predicate = self._input_schema()._drop_nan_predicate(subset)
        return self.filter(predicate.value()) if predicate else self.copy()

    def drop_nulls(self, subset: List[String] = List[String]()) raises -> Self:
        """Keep rows with no null in subset (default: every column), using
        each column's `is_not_null` predicate."""
        var names = subset.copy()
        if len(names) == 0:
            names = self._input_schema().columns()
        if len(names) == 0:
            return self.copy()
        var keep = col(names[0]).is_not_null()
        for k in range(1, len(names)):
            keep = keep & col(names[k]).is_not_null()
        return self.filter(keep)

    def null_count(self) raises -> Self:
        """One row holding each column's null count as Int64, as
        DataFrame.null_count: a select of `col(c).null_count()` per
        column."""
        var counts = List[Expr]()
        for name in self._input_schema().columns():
            counts.append(col(name).null_count())
        return self.select_exprs(counts)

    def n_unique(self, subset: List[String] = List[String]()) raises -> Self:
        """One row, one Int64 column "n_unique": the number of distinct
        rows on subset (default: every column), the value
        DataFrame.n_unique returns. Built as the subset's unique rows
        counted."""
        var keys = subset.copy()
        if len(keys) == 0:
            keys = self._input_schema().columns()
        if len(keys) == 0:
            raise Error("n_unique requires at least one column")
        return (
            self.select(keys)
            .unique()
            .select(col(keys[0]).len().alias("n_unique"))
        )

    def is_duplicated(
        self, subset: List[String] = List[String]()
    ) raises -> Self:
        """One Boolean column "is_duplicated", true for every row whose key
        on subset (default: every column) occurs more than once: the
        Series DataFrame.is_duplicated returns, as a frame."""
        return self._key_occurs(subset, "is_duplicated")

    def is_unique(self, subset: List[String] = List[String]()) raises -> Self:
        """One Boolean column "is_unique", true for every row whose key on
        subset (default: every column) occurs exactly once: the Series
        DataFrame.is_unique returns, as a frame."""
        return self._key_occurs(subset, "is_unique")

    def _key_occurs(self, subset: List[String], name: String) raises -> Self:
        # Each row's key count is a row count over the key's partition
        # (nulls are one key, as in DataFrame.is_duplicated).
        var keys = subset.copy()
        if len(keys) == 0:
            keys = self._input_schema().columns()
        if len(keys) == 0:
            raise Error(name + " requires at least one column")
        var occurs = col(keys[0]).len().over(keys)
        var flag = occurs == lit(Int64(1))
        if name == "is_duplicated":
            flag = occurs > lit(Int64(1))
        return self.select_exprs([flag.alias(name)])

    def group_by_dynamic(
        self,
        index_column: String,
        every: String,
        period: Optional[String] = None,
        offset: Optional[String] = None,
        closed: String = "left",
        label: String = "left",
        group_by: List[String] = List[String](),
    ) -> LazyTimeGroupBy:
        """Group rows into calendar-aligned, possibly overlapping windows of
        the index column; see DataFrame.group_by_dynamic.

        The index must be sorted (within each `group_by` group) when the
        plan runs. Filters and slices above the aggregation stay above it,
        and its input is read in full before any window is formed.
        """
        return LazyTimeGroupBy(
            self.copy(),
            _dynamic_window_node(
                index_column=index_column,
                groups=group_by,
                every=every,
                period=period.value() if period else every,
                offset=offset.value() if offset else "0d",
                closed=closed,
                label=label,
            ),
        )

    def group_by_dynamic(
        self,
        index_column: String,
        every: String,
        period: Optional[String] = None,
        offset: Optional[String] = None,
        closed: String = "left",
        label: String = "left",
        *,
        group_by: String,
    ) -> LazyTimeGroupBy:
        return self.group_by_dynamic(
            index_column,
            every,
            period,
            offset,
            closed,
            label,
            List[String]([group_by]),
        )

    def group_by_dynamic(
        self,
        index_column: String,
        every: String,
        period: Optional[String] = None,
        offset: Optional[String] = None,
        closed: String = "left",
        label: String = "left",
        *,
        group_by: NoneType,
    ) -> LazyTimeGroupBy:
        return self.group_by_dynamic(
            index_column, every, period, offset, closed, label, List[String]()
        )

    def rolling(
        self,
        index_column: String,
        period: String,
        offset: Optional[String] = None,
        closed: String = "right",
        group_by: List[String] = List[String](),
    ) -> LazyTimeGroupBy:
        """Aggregate a duration window ending at each row's index value;
        see DataFrame.rolling.

        The index must be sorted (within each `group_by` group) when the
        plan runs. Filters and slices above the aggregation stay above it,
        and its input is read in full before any window is formed.
        """
        return LazyTimeGroupBy(
            self.copy(),
            _rolling_window_node(
                index_column=index_column,
                groups=group_by,
                period=period,
                offset=offset.value() if offset else negate_period(period),
                closed=closed,
            ),
        )

    def rolling(
        self,
        index_column: String,
        period: String,
        offset: Optional[String] = None,
        closed: String = "right",
        *,
        group_by: String,
    ) -> LazyTimeGroupBy:
        return self.rolling(
            index_column, period, offset, closed, List[String]([group_by])
        )

    def rolling(
        self,
        index_column: String,
        period: String,
        offset: Optional[String] = None,
        closed: String = "right",
        *,
        group_by: NoneType,
    ) -> LazyTimeGroupBy:
        return self.rolling(
            index_column, period, offset, closed, List[String]()
        )

    def join_asof(
        self,
        other: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: List[String] = List[String](),
        strategy: String = "backward",
        tolerance: Optional[AnyValue] = None,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        """As-of join with sortedness checked at execution; see DataFrame.join_asof.

        Inputs are materialized for a complete ordered scan. Optimizations
        do not push filters or slices across this order-sensitive join.
        """
        var lhs = left_on
        var rhs = right_on
        if on != "":
            if lhs != "" or rhs != "":
                raise Error("join_asof: use on or left_on/right_on, not both")
            lhs = on
            rhs = on
        if lhs == "" or rhs == "":
            raise Error("join_asof requires on or both left_on and right_on")
        var result = self.join(
            other, left_on=[lhs], right_on=[rhs], suffix=suffix
        )
        ref node = result._plan._nodes[len(result._plan._nodes) - 1]
        node.set_asof_options(strategy, allow_exact_matches, tolerance, by)
        return result^

    def join_asof(
        self,
        other: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: List[String] = List[String](),
        strategy: String = "backward",
        tolerance: Int,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            by=by,
            strategy=strategy,
            tolerance=Optional(AnyValue(Int64(tolerance))),
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        other: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: List[String] = List[String](),
        strategy: String = "backward",
        tolerance: Float64,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            by=by,
            strategy=strategy,
            tolerance=Optional(AnyValue(tolerance)),
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        other: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: String,
        strategy: String = "backward",
        tolerance: Optional[AnyValue] = None,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            by=[by],
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        other: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: String,
        strategy: String = "backward",
        tolerance: Int,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            by=[by],
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        other: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: String,
        strategy: String = "backward",
        tolerance: Float64,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            by=[by],
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        other: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: NoneType,
        strategy: String = "backward",
        tolerance: Optional[AnyValue] = None,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            by=List[String](),
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        other: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: NoneType,
        strategy: String = "backward",
        tolerance: Int,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            by=List[String](),
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        other: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: NoneType,
        strategy: String = "backward",
        tolerance: Float64,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            by=List[String](),
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join(
        self,
        other: Self,
        on: List[String],
        how: String = "inner",
        suffix: String = "_right",
        coalesce: Bool = True,
    ) -> Self:
        """Join with another lazy plan on keys named alike on both sides;
        see DataFrame.join."""
        return self.join(
            other,
            left_on=on,
            right_on=on,
            how=how,
            suffix=suffix,
            coalesce=coalesce,
        )

    def join(
        self,
        other: Self,
        *,
        left_on: List[String],
        right_on: List[String],
        how: String = "inner",
        suffix: String = "_right",
        coalesce: Bool = True,
    ) -> Self:
        """Join with another lazy plan on paired keys: left_on[i] matches
        right_on[i], as in DataFrame.join. Projection and predicate
        pushdown see both key lists, so each input reads only its own keys
        and the columns used above the join."""
        var result = self.copy()
        var left_root = len(result._plan._nodes) - 1
        var shift = len(result._plan._nodes)
        var frame_shift = len(result._plan._frames)
        for frame in other._plan._frames:
            result._plan._frames.append(frame.copy())
        for schema in other._plan._schemas:
            result._plan._schemas.append(schema.copy())
        for node in other._plan._nodes:
            var copied = node.copy()
            if copied.left >= 0:
                copied.left += shift
            if copied.right >= 0:
                copied.right += shift
            if _is_scan(copied.kind):
                copied.set_source_slot(copied.source_slot() + frame_shift)
            result._plan._nodes.append(copied^)
        var join = _join_node(
            left=left_root,
            right=len(result._plan._nodes) - 1,
            left_keys=left_on,
            join_type=how,
            type_code=join_code(how),
            right_keys=right_on,
            coalesce=coalesce,
            suffix=suffix,
        )
        result._plan._nodes.append(join^)
        return result^

    def join(
        self,
        other: Self,
        on: String,
        how: String = "inner",
        suffix: String = "_right",
        coalesce: Bool = True,
    ) -> Self:
        return self.join(other, [on], how, suffix, coalesce)

    def join(
        self, other: Self, *, how: String, suffix: String = "_right"
    ) raises -> Self:
        """Cross join: every row of this plan paired with every row of
        `other`, left-major; see DataFrame.join(right, how="cross")."""
        if join_code(how) != JOIN_CROSS:
            raise Error("Join how='" + how + "' requires key columns")
        return self.join(
            other,
            left_on=List[String](),
            right_on=List[String](),
            how=how,
            suffix=suffix,
        )

    def _backend(
        self,
        engine: String,
        optimize: Bool,
        streaming: Bool,
        batch_size: Int,
    ) raises -> _BackendDecision:
        var decision = _select_backend(engine)
        if engine == "auto":
            var selected = _select_auto_accel(
                self, optimize, streaming, batch_size
            )
            decision = _automatic_decision(selected)
        return decision^

    def collect(
        self,
        *,
        optimize: Bool = True,
        streaming: Bool = True,
        batch_size: Int = 65536,
        engine: String = "auto",
    ) raises -> DataFrame:
        """Optimize (unless disabled) and execute the plan.

        Streaming batches default to 65,536 rows. Set streaming=False to use
        the materializing executor. Stateful/global operations retain their
        documented boundaries; collecting still retains the final output.

        Parallel work inside runs on one crew of threads started for the
        query (`parallel.Crew`) and joined before this returns.

        engine='auto' lets the optional provider select before execution.
        Without a provider, auto uses CPU. 'cpu' forces CPU;
        'accel' uses an optional registered provider or an explicit runtime.
        """
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        var backend = self._backend(engine, optimize, streaming, batch_size)
        backend.require_available()
        if engine == "auto" and backend.engine != "cpu":
            return _execute_auto_accel(
                self._with_accelerator_settings(
                    optimize, streaming, batch_size
                ),
                optimize,
                streaming,
                batch_size,
                False,
            )[0].copy()
        if engine == "accel":
            return _execute_accel(
                self._with_accelerator_settings(optimize, streaming, batch_size)
            )[0].copy()
        var crew = Crew.start()
        try:
            var result: DataFrame
            if engine == "auto" and _automatic_handoff():
                result = self._with_handoff(HANDOFF_AUTO)._collect(
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=batch_size,
                )
            else:
                result = self._collect(
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=batch_size,
                )
            crew.release()
            return result^
        except e:
            crew.release()
            raise e^

    def collect[
        B: AcceleratorBackend
    ](
        self,
        *,
        accelerator: B,
        engine: String = "accel",
        optimize: Bool = True,
        streaming: Bool = True,
        batch_size: Int = 65536,
    ) raises -> DataFrame:
        """Execute with an explicit optional runtime and its placement policy.

        Requested execution settings travel with a dispatch copy of the plan.
        A provider runs the plan its own way: a discrete device runs whole
        plans, and a device sharing memory with the CPU runs the plan on the
        CPU executor with supported nodes handed over (`_with_handoff`).
        """
        _ = _select_backend(engine)
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        if engine == "auto":
            var selected = accelerator.select_auto(
                self, optimize, streaming, batch_size
            )
            _ = _automatic_decision(selected)
            if selected[0] != "cpu":
                return accelerator.execute_auto(
                    self._with_accelerator_settings(
                        optimize, streaming, batch_size
                    ),
                    optimize,
                    streaming,
                    batch_size,
                    False,
                )[0].copy()
            return self.collect(
                engine="cpu",
                optimize=optimize,
                streaming=streaming,
                batch_size=batch_size,
            )
        if engine != "accel":
            return self.collect(
                engine=engine,
                optimize=optimize,
                streaming=streaming,
                batch_size=batch_size,
            )
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        return accelerator.execute(
            self._with_accelerator_settings(optimize, streaming, batch_size)
        )[0].copy()

    def profile[
        B: AcceleratorBackend
    ](
        self,
        *,
        accelerator: B,
        engine: String = "accel",
        optimize: Bool = True,
        streaming: Bool = True,
        batch_size: Int = 65536,
    ) raises -> Tuple[DataFrame, DataFrame]:
        """Execute with a provider and report its observed execution regions."""
        _ = _select_backend(engine)
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        if engine == "auto":
            var selected = accelerator.select_auto(
                self, optimize, streaming, batch_size
            )
            _ = _automatic_decision(selected)
            if selected[0] != "cpu":
                var result = accelerator.execute_auto(
                    self._with_accelerator_settings(
                        optimize, streaming, batch_size
                    ),
                    optimize,
                    streaming,
                    batch_size,
                    True,
                )
                return (
                    result[0].copy(),
                    _selection_report(result[1], selected[1]),
                )
            var result = self.profile(
                engine="cpu",
                optimize=optimize,
                streaming=streaming,
                batch_size=batch_size,
            )
            return (result[0].copy(), _selection_report(result[1], selected[1]))
        if engine != "accel":
            return self.profile(
                engine=engine,
                optimize=optimize,
                streaming=streaming,
                batch_size=batch_size,
            )
        return accelerator.execute_profiled(
            self._with_accelerator_settings(optimize, streaming, batch_size)
        )

    def profile(
        self,
        *,
        optimize: Bool = True,
        streaming: Bool = True,
        batch_size: Int = 65536,
        engine: String = "auto",
    ) raises -> Tuple[DataFrame, DataFrame]:
        """Collect, and report what each plan node did (#439).

        The result, then one row per executed node in node order: `node`,
        `operator` (its `explain` label), `executor` ("pipeline" for
        steps run as thread-owned pipelines over morsels, "streaming" or
        "eager"), `algorithm` and `build_side` for joins, `input_rows`
        (read from the left or probe input), `build_rows` (the right input
        a join indexed), `output_rows`, `builds` (indexes built),
        `executions` (times the node ran), `busy_ms` (a streamed operation's
        time summed over its batches, across workers) and `wall_ms` (an
        eager node's wall time including its inputs; on a stream's last
        node, the whole stream's). Counts are observed, never estimated,
        and the result is the one `collect` returns.
        Engine selection follows `collect`. Auto mode adds selection_reason;
        accelerator providers may append device and timing diagnostics.
        """
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        var backend = self._backend(engine, optimize, streaming, batch_size)
        backend.require_available()
        if engine == "auto" and backend.engine != "cpu":
            var result = _execute_auto_accel(
                self._with_accelerator_settings(
                    optimize, streaming, batch_size
                ),
                optimize,
                streaming,
                batch_size,
                True,
            )
            return (
                result[0].copy(),
                _selection_report(result[1], backend.reason),
            )
        if engine == "accel":
            return _profile_accel(
                self._with_accelerator_settings(optimize, streaming, batch_size)
            )
        var plan = self._optimized() if optimize else self.copy()
        if optimize:
            plan._push_join_key_sets(streaming, batch_size)
            plan._share_subplans(streaming, batch_size)
            plan._order_joins(streaming, batch_size)
        plan._plan._report = Optional(ArcPointer(ExecutionReport()))
        if engine == "auto" and _automatic_handoff():
            plan._plan._handoff = HANDOFF_AUTO
        var result = plan._execute(
            len(plan._plan._nodes) - 1, False, streaming, batch_size
        )
        var report = plan._plan._report.value()[].frame()
        if engine == "auto":
            report = _selection_report(report, backend.reason)
        return (result^, report^)

    def explain[
        B: AcceleratorBackend
    ](
        self,
        *,
        accelerator: B,
        engine: String = "accel",
        optimize: Bool = True,
        streaming: Bool = True,
        batch_size: Int = 65536,
    ) raises -> String:
        """Describe provider capability without submitting device work."""
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        _ = _select_backend(engine)
        if engine == "auto":
            var selected = accelerator.select_auto(
                self, optimize, streaming, batch_size
            )
            _ = _automatic_decision(selected)
            return (
                "ENGINE "
                + selected[0]
                + ": "
                + selected[1]
                + "\n"
                + (
                    self.explain(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=batch_size,
                    ) if selected[0]
                    == "cpu" else ""
                )
            )
        if engine != "accel":
            return self.explain(
                engine=engine,
                optimize=optimize,
                streaming=streaming,
                batch_size=batch_size,
            )
        return accelerator.describe(
            self._with_accelerator_settings(optimize, streaming, batch_size)
        )

    def explain(
        self,
        *,
        optimize: Bool = True,
        streaming: Bool = True,
        batch_size: Int = 65536,
        engine: String = "cpu",
    ) raises -> String:
        """The (optimized) plan, one operator per line, root first. With
        `engine="auto"` (what `collect` uses), an `ENGINE` line naming the
        selected engine and why comes first.

        Streaming annotations show batch-capable operators, aggregate state
        and materialization boundaries. streaming=False omits annotations.
        Non-default engines prepend the backend decision. An unavailable
        backend can be explained without executing or probing a device;
        the operator lines still describe the logical/CPU plan.
        """
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        var backend = self._backend(engine, optimize, streaming, batch_size)
        if engine == "accel" and backend.available:
            return _describe_accel(
                self._with_accelerator_settings(optimize, streaming, batch_size)
            )
        if engine == "auto" and backend.engine != "cpu":
            return backend.describe()
        var plan = self._optimized() if optimize else self.copy()
        var out = String()
        if engine != "cpu":
            out = backend.describe()
        if engine == "auto" and _automatic_handoff():
            out += plan._placement_lines()
        plan._describe(len(plan._plan._nodes) - 1, 0, out, streaming)
        return out^

    # --- execution -----------------------------------------------------

    def fetch[
        B: AcceleratorBackend
    ](
        self, n: Int = 5, *, accelerator: B, engine: String = "accel"
    ) raises -> DataFrame:
        """Collect the first n rows using an explicit provider."""
        return self.head(n).collect(accelerator=accelerator, engine=engine)

    def fetch(self, n: Int = 5, *, engine: String = "auto") raises -> DataFrame:
        """Collect only the first n rows of the result."""
        return self.head(n).collect(engine=engine)

    def _collect(
        self,
        *,
        optimize: Bool,
        streaming: Bool,
        batch_size: Int,
    ) raises -> DataFrame:
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        if getenv("DATAFRAME_EXECUTION_REPORT"):
            # One `dataframe-operator:` line per executed node on stderr,
            # the columns of `profile`'s report, for benchmark traces.
            var profiled = self.profile(
                optimize=optimize,
                streaming=streaming,
                batch_size=batch_size,
                engine="cpu",
            )
            var report = profiled[1].copy()
            for r in range(report.height()):
                var parts = List[String]()
                for name in report.columns():
                    var cell = report.item(r, name)
                    if cell.dtype() == DataType.INT64:
                        parts.append(String(cell.int64()))
                    elif cell.dtype() == DataType.FLOAT64:
                        parts.append(String(round(cell.float64(), 3)))
                    else:
                        parts.append(cell.string())
                print(
                    "dataframe-operator:\t" + String("\t").join(parts),
                    file=RawFileDescriptor(2),
                )
            return profiled[0].copy()
        var plan = self._optimized() if optimize else self.copy()
        if optimize:
            plan._push_join_key_sets(streaming, batch_size)
            plan._share_subplans(streaming, batch_size)
            plan._order_joins(streaming, batch_size)
        plan._plan._prefetched = Optional(ArcPointer(Dict[Int, DataFrame]()))
        return plan._execute(
            len(plan._plan._nodes) - 1, False, streaming, batch_size
        )

    def collect_schema(self) raises -> List[String]:
        """Output names and dtypes as "name: dtype", computed without reading
        rows: every scan yields zero rows, then the plan runs as usual, so
        binding validates each expression exactly as collect would."""
        var plan = self._optimized()
        var frame = plan._execute(len(plan._plan._nodes) - 1, True)
        var out = List[String]()
        for field in frame.schema():
            out.append(field.name + ": " + field.dtype.name())
        return out^

    def _sampled_groups(self, keys: List[String], frame: Int) raises -> Int:
        return plan_sampled_groups(self._plan, keys=keys, frame=frame)

    def _batch_states_win(
        self, keys: List[String], frame: Int, batch_size: Int
    ) raises -> Bool:
        return plan_batch_states_win(
            self._plan, keys=keys, frame=frame, batch_size=batch_size
        )

    def _joins_below(self, index: Int) -> Bool:
        return plan_joins_below(self._plan, index=index)

    def _direct_key(self, index: Int, names: List[String]) raises -> Bool:
        return plan_direct_key(self._plan, index=index, names=names)

    def _swap_join_sides(
        self, at: Int, built: DataFrame, batch_size: Int
    ) raises -> Optional[LazyFrame]:
        var swapped = plan_swap_join_sides(
            self._plan, at=at, built=built, batch_size=batch_size
        )
        if swapped:
            return Optional(_from_context(swapped.take()))
        return None

    def _row_number_shape(
        self,
        source: Int,
        keys: List[String],
        carried: List[String],
        holder: Int,
        joins: List[DataFrame],
        operations: List[PlanNode],
    ) raises -> Tuple[Float64, Float64]:
        return plan_row_number_shape(
            self._plan,
            source=source,
            keys=keys,
            carried=carried,
            holder=holder,
            joins=joins,
            operations=operations,
        )

    def _swap_left_join_sides(
        self, at: Int, built: DataFrame, batch_size: Int
    ) raises -> Optional[LazyFrame]:
        var swapped = plan_swap_left_join_sides(
            self._plan, at=at, built=built, batch_size=batch_size
        )
        if swapped:
            return Optional(_from_context(swapped.take()))
        return None

    def _height_bound(self, index: Int) -> Int:
        return plan_height_bound(self._plan, index=index)

    def _keeps_column(self, index: Int, name: String) -> Bool:
        return plan_keeps_column(self._plan, index=index, name=name)

    def _input_name(self, index: Int, name: String) -> Optional[String]:
        return plan_input_name(self._plan, index=index, name=name)

    def _grouped_on(self, index: Int, key: String) -> Tuple[Int, String]:
        return plan_grouped_on(self._plan, index=index, key=key)

    def _key_source(self, index: Int, key: String) raises -> Optional[Series]:
        return plan_key_source(self._plan, index=index, key=key)

    def _reduce_by_probe_keys(
        self, top: Int, index: Int, batch_size: Int
    ) raises -> Optional[DataFrame]:
        return plan_reduce_by_probe_keys(
            self._plan, top=top, index=index, batch_size=batch_size
        )

    def _ends_in_breaker(self, index: Int) -> Bool:
        return plan_ends_in_breaker(self._plan, index=index)

    def _prefetch_breakers(self, top: Int, batch_size: Int) raises:
        plan_prefetch_breakers(self._plan, top=top, batch_size=batch_size)

    def _known_height(self, index: Int) -> Int:
        return plan_known_height(self._plan, index=index)

    def _stream_execute(
        self, index: Int, batch_size: Int
    ) raises -> Optional[DataFrame]:
        return plan_stream_execute(
            self._plan, index=index, batch_size=batch_size
        )

    def _execute(
        self,
        index: Int,
        empty: Bool,
        streaming: Bool = False,
        batch_size: Int = 65536,
    ) raises -> DataFrame:
        return plan_execute(
            self._plan,
            index=index,
            empty=empty,
            streaming=streaming,
            batch_size=batch_size,
        )

    def _join_frames(
        self,
        index: Int,
        var input: DataFrame,
        var right: DataFrame,
        empty: Bool,
    ) raises -> DataFrame:
        return plan_join_frames(self._plan, index, input^, right^, empty)

    def _both_inputs(
        self, index: Int, empty: Bool, streaming: Bool, batch_size: Int
    ) raises -> Tuple[DataFrame, DataFrame]:
        return plan_both_inputs(
            self._plan,
            index=index,
            empty=empty,
            streaming=streaming,
            batch_size=batch_size,
        )

    def _concat(
        self, index: Int, empty: Bool, streaming: Bool, batch_size: Int
    ) raises -> DataFrame:
        return plan_concat(
            self._plan,
            index=index,
            empty=empty,
            streaming=streaming,
            batch_size=batch_size,
        )

    def _execute_node(
        self,
        index: Int,
        empty: Bool,
        streaming: Bool,
        batch_size: Int,
    ) raises -> DataFrame:
        return plan_execute_node(
            self._plan,
            index=index,
            empty=empty,
            streaming=streaming,
            batch_size=batch_size,
        )

    def _optimized(self) raises -> Self:
        return _from_context(plan_optimized(self._plan))

    def _eliminate_common_subexpressions(mut self) raises:
        plan_eliminate_common_subexpressions(self._plan)

    def _fold_casts(mut self) raises:
        plan_fold_casts(self._plan)

    def _share_subplans(mut self, streaming: Bool, batch_size: Int) raises:
        plan_share_subplans(
            self._plan, streaming=streaming, batch_size=batch_size
        )

    def _read_whole(
        self,
        index: Int,
        classes: List[Int],
        parents: List[Int],
        joining: List[Bool],
        covered: List[Bool],
    ) -> Bool:
        return plan_read_whole(
            self._plan,
            index=index,
            classes=classes,
            parents=parents,
            joining=joining,
            covered=covered,
        )

    def _cover(self, index: Int, mut covered: List[Bool]):
        plan_cover(self._plan, index, covered)

    def _node_hash(
        self, index: Int, classes: List[Int], canonical: List[Int]
    ) -> UInt64:
        return plan_node_hash(
            self._plan, index=index, classes=classes, canonical=canonical
        )

    def _canonical_slot(self, index: Int, canonical: List[Int]) -> Int:
        return plan_canonical_slot(self._plan, index=index, canonical=canonical)

    def _same_node(
        self, a: Int, b: Int, classes: List[Int], canonical: List[Int]
    ) -> Bool:
        return plan_same_node(
            self._plan, a=a, b=b, classes=classes, canonical=canonical
        )

    def _parents(self) -> List[Int]:
        return plan_parents(self._plan)

    def _columns_of(self, index: Int) raises -> List[String]:
        return plan_columns_of(self._plan, index=index)

    def _unique_as_aggregate(mut self) raises:
        plan_unique_as_aggregate(self._plan)

    def _repeats_a_scan(self) -> Bool:
        return plan_repeats_a_scan(self._plan)

    def _projection_visits(self) -> Tuple[List[Int], List[Int]]:
        return plan_projection_visits(self._plan)

    def _subplan_classes(
        self,
    ) -> Tuple[List[Int], List[Int], List[Bool]]:
        return plan_subplan_classes(self._plan)

    def _filter_shared_subplans(mut self):
        plan_filter_shared_subplans(self._plan)

    def _split_filters(mut self, first: Int = 0):
        plan_split_filters(self._plan, first=first)

    def _merge_filters(mut self):
        plan_merge_filters(self._plan)

    def _order_joins(mut self, streaming: Bool, batch_size: Int) raises:
        plan_order_joins(self._plan, streaming=streaming, batch_size=batch_size)

    def _push_join_key_sets(mut self, streaming: Bool, batch_size: Int) raises:
        plan_push_join_key_sets(
            self._plan, streaming=streaming, batch_size=batch_size
        )

    def _key_set_side(
        mut self, j: Int, small_left: Bool, streaming: Bool, batch_size: Int
    ) raises -> Bool:
        return plan_key_set_side(
            self._plan,
            j=j,
            small_left=small_left,
            streaming=streaming,
            batch_size=batch_size,
        )

    def _side_comparisons(
        self, j: Int, small_left: Bool
    ) raises -> List[Tuple[String, Int, String]]:
        return plan_side_comparisons(self._plan, j=j, small_left=small_left)

    def _filtered_table(self, index: Int) -> Bool:
        return plan_filtered_table(self._plan, index=index)

    def _provider_scan(
        self, index: Int, name: String
    ) raises -> Tuple[Int, String]:
        return plan_provider_scan(self._plan, index=index, name=name)

    def _selective_on(mut self, scan: Int, keys: Expr) raises -> Bool:
        return plan_selective_on(self._plan, scan=scan, keys=keys)

    def _any_selective(
        mut self,
        result: DataFrame,
        keys: List[String],
        providers: List[Tuple[Int, String]],
    ) raises -> Bool:
        return plan_any_selective(
            self._plan, result=result, keys=keys, providers=providers
        )

    def _has_kind(self, index: Int, kind: Int) -> Bool:
        return plan_has_kind(self._plan, index=index, kind=kind)

    def _filters_rows(self, index: Int) -> Bool:
        return plan_filters_rows(self._plan, index=index)

    def _has_join(self, index: Int) -> Bool:
        return plan_has_join(self._plan, index=index)

    def _largest_scan(self, index: Int) -> Int:
        return plan_largest_scan(self._plan, index=index)

    def _live_parents(self, root: Int) -> List[Int]:
        return plan_live_parents(self._plan, root=root)

    def _order_free_above(
        self, top: Int, parents: List[Int], through_joins: Bool = False
    ) -> Bool:
        return plan_order_free_above(
            self._plan, top=top, parents=parents, through_joins=through_joins
        )

    def _estimated_rows(self, index: Int) raises -> Int:
        return plan_estimated_rows(self._plan, index=index)

    def _progression_key(self, index: Int, keys: List[String]) raises -> Bool:
        return plan_progression_key(self._plan, index=index, keys=keys)

    def _graph_step(self, index: Int) -> Bool:
        return plan_graph_step(self._plan, index=index)

    def _collect_graph(
        self,
        index: Int,
        mut relations: List[Int],
        mut pairs: List[Tuple[String, String]],
        mut filters: List[Int],
    ):
        plan_collect_graph(self._plan, index, relations, pairs, filters)

    def _indexed_relation(
        self,
        entry: Int,
        other: Int,
        joined: List[Bool],
        node_of: List[Int],
        edges: List[JoinEdge],
        masks: List[Int],
    ) raises -> Bool:
        return plan_indexed_relation(
            self._plan,
            entry=entry,
            other=other,
            joined=joined,
            node_of=node_of,
            edges=edges,
            masks=masks,
        )

    def _cost_order_joins(
        mut self, root: Int, streaming: Bool, batch_size: Int
    ) raises -> Bool:
        return plan_cost_order_joins(
            self._plan, root=root, streaming=streaming, batch_size=batch_size
        )

    def _probe_largest_inputs(
        mut self, root: Int, streaming: Bool, batch_size: Int
    ) raises -> Bool:
        return plan_probe_largest_inputs(
            self._plan, root=root, streaming=streaming, batch_size=batch_size
        )

    def _probe_largest_input(
        mut self, top: Int, streaming: Bool, batch_size: Int
    ) raises -> Bool:
        return plan_probe_largest_input(
            self._plan, top=top, streaming=streaming, batch_size=batch_size
        )

    def _relation_estimate(self, index: Int) raises -> Int:
        return plan_relation_estimate(self._plan, index=index)

    def _spine_frame(self, index: Int) -> Optional[DataFrame]:
        return plan_spine_frame(self._plan, index=index)

    def _aggregated(self, index: Int) -> Bool:
        return plan_aggregated(self._plan, index=index)

    def _spine_rows(self, index: Int) -> Int:
        return plan_spine_rows(self._plan, index=index)

    def _measurable(self, index: Int) -> Bool:
        return plan_measurable(self._plan, index=index)

    def _selectivity(
        mut self, index: Int, streaming: Bool, batch_size: Int
    ) raises -> Float64:
        return plan_selectivity(
            self._plan, index=index, streaming=streaming, batch_size=batch_size
        )

    def _order_chain(
        mut self, top: Int, streaming: Bool, batch_size: Int
    ) raises -> Bool:
        return plan_order_chain(
            self._plan, top=top, streaming=streaming, batch_size=batch_size
        )

    def _restore_measured_chain(mut self, var saved: List[PlanNode]):
        plan_restore_measured_chain(self._plan, saved^)

    def _compact_scan_slots(mut self):
        plan_compact_scan_slots(self._plan)

    def _available(
        self,
        names: List[String],
        provider: Dict[String, Int],
        owner: List[Int],
        placed: List[Bool],
    ) raises -> Bool:
        return plan_available(
            self._plan,
            names=names,
            provider=provider,
            owner=owner,
            placed=placed,
        )

    def _remember_schemas(mut self) -> Bool:
        return plan_remember_schemas(self._plan)

    def _forget_schema(mut self, index: Int):
        plan_forget_schema(self._plan, index=index)

    def _push_predicates(mut self) raises:
        plan_push_predicates(self._plan)

    def _drop_constant_filters(mut self) raises:
        plan_drop_constant_filters(self._plan)

    def _push_predicates_now(mut self) raises:
        plan_push_predicates_now(self._plan)

    def _unstick(self, filter: Int, child: Int, mut stuck: List[Bool]):
        plan_unstick(self._plan, filter, child, stuck)

    def _passes(self, filter: Int, child: Int) raises -> Int:
        return plan_passes(self._plan, filter=filter, child=child)

    def _transfer_key_filter(
        mut self, filter: Int, join: Int, right_side: Bool
    ):
        plan_transfer_key_filter(
            self._plan, filter=filter, join=join, right_side=right_side
        )

    def _swap_down(mut self, filter: Int, child: Int, right_side: Bool):
        plan_swap_down(
            self._plan, filter=filter, child=child, right_side=right_side
        )

    def _swap_in_place(mut self, filter: Int, child: Int, right_side: Bool):
        plan_swap_in_place(
            self._plan, filter=filter, child=child, right_side=right_side
        )

    def _reorder(mut self, root: Int = -1):
        plan_reorder(self._plan, root=root)

    def _topological(self, root: Int = -1) -> List[Int]:
        return plan_topological(self._plan, root=root)

    def _push_slices(mut self) raises:
        plan_push_slices(self._plan)

    def _slice_over_ipc(self, index: Int) -> Bool:
        return plan_slice_over_ipc(self._plan, index=index)

    def _fuse_top_k(mut self):
        plan_fuse_top_k(self._plan)

    def _push_projections(mut self) raises:
        plan_push_projections(self._plan)

    def _align_union_inputs(
        mut self, unions: List[Int], columns: List[List[String]]
    ) raises -> Bool:
        return plan_align_union_inputs(
            self._plan, unions=unions, columns=columns
        )

    def _forget_schemas_above(mut self, index: Int, parents: List[Int]):
        plan_forget_schemas_above(self._plan, index=index, parents=parents)

    def _share_copy_schemas(mut self, copy_of: List[Int], head: Int):
        plan_share_copy_schemas(self._plan, copy_of=copy_of, head=head)

    def _prune_outputs(mut self, index: Int, wanted: List[String]):
        plan_prune_outputs(self._plan, index=index, wanted=wanted)

    def _join_input_columns(
        self, index: Int, wanted: List[String]
    ) raises -> Optional[List[List[String]]]:
        return plan_join_input_columns(self._plan, index=index, wanted=wanted)

    def _narrow_join_inputs(
        mut self,
        joins: List[Int],
        right_sides: List[Bool],
        columns: List[List[String]],
    ) raises:
        plan_narrow_join_inputs(
            self._plan, joins=joins, right_sides=right_sides, columns=columns
        )

    def _label(self, index: Int) -> String:
        return plan_label(self._plan, index=index)

    def _describe(
        self, index: Int, depth: Int, mut out: String, streaming: Bool = True
    ):
        plan_describe(self._plan, index, depth, out, streaming)


def _automatic_handoff() -> Bool:
    """Whether `engine="auto"` places handed-off nodes by measured cost:
    a provider is installed and a calibration is configured."""
    return _accel_installed() and Bool(calibration_path())


def _automatic_decision(
    selected: Tuple[String, String]
) raises -> _BackendDecision:
    if (
        selected[0] != "cpu"
        and selected[0] != "accel"
        and selected[0] != "mixed"
    ):
        raise Error("Invalid automatic provider decision: " + selected[0])
    return _BackendDecision(selected[0], True, selected[1])


def _selection_report(report: DataFrame, reason: String) raises -> DataFrame:
    return report.with_column(
        Series(
            "selection_reason",
            Column[String](List[String](length=report.height(), fill=reason)),
        )
    )


trait AcceleratorBackend(Copyable):
    """Optional execution provider; importing dataframe needs no GPU SDK."""

    def estimate_auto(self, plan: LazyFrame) raises -> PlanEstimate:
        """Optional plan-time facts; no device submission or source-value reads."""
        raise Error("provider has no analytical plan estimate")

    def select_auto(
        self,
        plan: LazyFrame,
        optimize: Bool,
        streaming: Bool,
        batch_size: Int,
    ) -> Tuple[String, String]:
        """Default keeps existing third-party providers on CPU in auto mode."""
        return ("cpu", "provider has no automatic placement policy")

    def execute_auto(
        self,
        plan: LazyFrame,
        optimize: Bool,
        streaming: Bool,
        batch_size: Int,
        profiling: Bool,
    ) raises -> Tuple[DataFrame, DataFrame]:
        return self.execute_profiled(plan) if profiling else self.execute(plan)

    def execute(self, plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        ...

    def describe(self, plan: LazyFrame) -> String:
        ...

    def execute_profiled(
        self, plan: LazyFrame
    ) raises -> Tuple[DataFrame, DataFrame]:
        """Optional measured execution; providers may retain the base report."""
        return self.execute(plan)

    def devices(self) -> List[DeviceInfo]:
        """Devices this runtime runs pipelines on; none by default."""
        return List[DeviceInfo]()

    def transfer_model(self, device: Int) -> TransferModel:
        return TransferModel.unmeasured()

    def pipeline_supports(self, pipeline: LazyFrame, device: Int) -> Bool:
        return False

    def pipeline_estimate(
        self, pipeline: LazyFrame, device: Int, features: PipelineFeatures
    ) -> Float64:
        return Float64(-1)

    def run_pipeline(
        self, pipeline: LazyFrame, device: Int
    ) raises -> DataFrame:
        raise Error("provider does not run pipelines")


@fieldwise_init
struct LazyGroupBy(Copyable):
    """A pending lazy grouping; finish it with agg, len or a reduction of
    every other column such as sum."""

    var _frame: LazyFrame
    var _keys: List[String]
    var _maintain_order: Bool
    # A group_by on expressions: its keys, parallel to `_keys` (their
    # output names); empty when every key is a column under its own name.
    var _key_exprs: List[Expr]

    def _grouped(
        self, reads: List[String], every_column: Bool
    ) -> Tuple[LazyFrame, List[String], Bool]:
        """The input with computed keys added as columns, the names to
        group by, and whether some key has a placeholder name.

        Computed keys become a with_columns below the group_by. A key
        named like a column the
        aggregations read (`reads`, or any column when `every_column`)
        would hide that column from them, so it is computed under a
        placeholder name and renamed after."""
        if len(self._key_exprs) == 0:
            return (self._frame.copy(), self._keys.copy(), False)
        var computed = List[Expr]()
        var grouped = List[String](capacity=len(self._keys))
        var hidden = False
        for k in range(len(self._key_exprs)):
            ref key = self._key_exprs[k]
            var name = self._keys[k]
            if _bare_column(key):
                grouped.append(name)
            elif every_column or name in reads:
                var placeholder = "__group_key_" + String(k)
                computed.append(key.alias(placeholder))
                grouped.append(placeholder)
                hidden = True
            else:
                computed.append(key.copy())
                grouped.append(name)
        return (self._frame.with_columns(computed), grouped^, hidden)

    def agg(self, exprs: List[Expr]) -> LazyFrame:
        """One row per group: the keys, then the aggregations; see
        GroupBy.agg."""
        var reads = List[String]()
        var every_column = False
        for e in exprs:
            var r = _references(e)
            if r:
                reads += r.value().copy()
            else:
                every_column = True
        var grouped = self._grouped(reads, every_column)
        var prepared = List[Expr](capacity=len(exprs))
        for expression in exprs:
            prepared.append(
                expression.implode() if changes_length(
                    expression
                ) else expression.copy()
            )
        var plan = grouped[0]._push(
            _aggregate_node(
                exprs=prepared,
                keys=grouped[1],
                maintain_order=self._maintain_order,
            )
        )
        if not grouped[2]:
            return plan^
        var outputs = List[Expr](capacity=len(self._keys) + 1)
        for k in range(len(self._keys)):
            outputs.append(col(grouped[1][k]).alias(self._keys[k]))
        outputs.append(exclude(grouped[1]))
        return plan.select_exprs(outputs)

    def agg(self, expr: Expr) -> LazyFrame:
        return self.agg([expr.copy()])

    def len(self, name: String = "len") -> LazyFrame:
        """Rows per group as Int64, rows with nulls included; see
        GroupBy.len."""
        var grouped = self._grouped(List[String](), False)
        return grouped[0]._push(
            _aggregate_node(
                exprs=[col(grouped[1][0]).len().alias(name)],
                keys=grouped[1],
                maintain_order=self._maintain_order,
            )
        )

    def _each(
        self,
        reduction: String,
        quantile: Float64 = 0.5,
        interpolation: String = "nearest",
    ) raises -> LazyFrame:
        """Apply `reduction` to every non-key column. Resolve the columns
        from the input schema."""
        var columns = self._frame._input_schema().columns()
        var exprs = List[Expr]()
        for name in columns:
            if name not in self._keys:
                exprs.append(
                    _reduce_column(name, reduction, quantile, interpolation)
                )
        return self.agg(exprs)

    def sum(self) raises -> LazyFrame:
        """Each other column's sum per group; see GroupBy.sum."""
        return self._each("sum")

    def mean(self) raises -> LazyFrame:
        """Each other column's mean per group; see GroupBy.mean."""
        return self._each("mean")

    def min(self) raises -> LazyFrame:
        """Each other column's minimum per group; see GroupBy.min."""
        return self._each("min")

    def max(self) raises -> LazyFrame:
        """Each other column's maximum per group; see GroupBy.max."""
        return self._each("max")

    def median(self) raises -> LazyFrame:
        """Each other column's median per group; see GroupBy.median."""
        return self._each("median")

    def first(self) raises -> LazyFrame:
        """Each other column's first value per group; see GroupBy.first."""
        return self._each("first")

    def last(self) raises -> LazyFrame:
        """Each other column's last value per group; see GroupBy.last."""
        return self._each("last")

    def n_unique(self) raises -> LazyFrame:
        """Each other column's distinct values per group; see
        GroupBy.n_unique."""
        return self._each("n_unique")

    def quantile(
        self, quantile: Float64, interpolation: String = "nearest"
    ) raises -> LazyFrame:
        """Each other column's quantile per group; see GroupBy.quantile."""
        return self._each("quantile", quantile, interpolation)


@fieldwise_init
struct LazyGroupingSets(Copyable):
    """A pending `group_by_rollup`, `group_by_cube` or
    `group_by_grouping_sets`; finish it with agg."""

    var _frame: LazyFrame
    var _sets: List[List[String]]

    def agg(self, exprs: List[Expr]) raises -> LazyFrame:
        """The rows of every grouping set, in set order: the keys (every
        key any set names, in order of first appearance), then the
        aggregations. A key a set does not group by is null in that set's
        rows, and `grouping` / `grouping_id` in `exprs` tell those subtotal
        nulls from null key values. Within a set the aggregations act as in
        `group_by(set).agg(exprs)`; the empty set gives one row, over the
        whole input, even when it has no rows. Row order is unspecified
        unless the result is sorted.

        The plan has one node that reads its input once: when the
        aggregations keep their states inline, every morsel of the input is
        sunk into one aggregate table per set (DuckDB's single sink over all
        grouping sets); otherwise each set groups the collected input.
        """
        if len(self._sets) == 0:
            raise Error("grouping sets need at least one set")
        var keys = List[String]()
        var membership = List[Bool]()
        for one in self._sets:
            for name in one:
                if name not in keys:
                    keys.append(name)
        if len(keys) == 0:
            raise Error("grouping sets need at least one key")
        for one in self._sets:
            for name in keys:
                membership.append(name in one)
        var schema = self._frame._input_schema()
        for name in keys:
            _ = schema.column(name)
        var aggregates = 0
        for e in exprs:
            if is_grouping(e):
                continue
            if has_grouping(e):
                raise Error(
                    "grouping and grouping_id must be whole aggregation "
                    "expressions, not parts of one"
                )
            aggregates += 1
        if aggregates == 0:
            raise Error("agg needs an aggregation besides grouping")
        return self._frame._push(
            _grouping_sets_node(
                exprs=exprs.copy(), keys=keys, membership=membership
            )
        )

    def agg(self, expr: Expr) raises -> LazyFrame:
        return self.agg([expr.copy()])


@fieldwise_init
struct LazyTimeGroupBy(Copyable):
    """A pending lazy `group_by_dynamic` or `rolling`; finish it with agg."""

    var _frame: LazyFrame
    var _node: PlanNode

    def agg(self, exprs: List[Expr], *, batch_size: Int = 1024) -> LazyFrame:
        """One row per window: the `group_by` keys, the index column (the
        window's label) and the aggregations; see TimeGroupBy.agg."""
        var node = self._node.copy()
        node.set_window_aggregations(exprs.copy(), batch_size)
        return self._frame._push(node^)

    def agg(self, expr: Expr, *, batch_size: Int = 1024) -> LazyFrame:
        return self.agg([expr.copy()], batch_size=batch_size)


def scan_csv(
    path: String,
    *,
    has_header: Bool = True,
    separator: String = ",",
    quote_char: String = '"',
    comment_prefix: String = "",
    skip_rows: Int = 0,
    null_values: List[String] = List[String](),
    ignore_errors: Bool = False,
    truncate_ragged_lines: Bool = False,
    encoding: String = "utf8",
    infer_schema_length: Int = 100,
    schema_overrides: Dict[String, String] = Dict[String, String](),
    try_parse_dates: Bool = False,
    glob: Bool = True,
) raises -> LazyFrame:
    """Lazily read a CSV file with an inferred schema. Nothing is read until
    collect; projection and head() are pushed into the reader.

    The options are read_csv's, with the same meaning, and reach the reader
    however the plan runs (eagerly, streaming, or for its schema); invalid
    ones raise here, as read_csv would raise them.
    """
    if glob and has_pattern(path):
        return scan_csv(
            [path],
            has_header=has_header,
            separator=separator,
            quote_char=quote_char,
            comment_prefix=comment_prefix,
            skip_rows=skip_rows,
            null_values=null_values,
            ignore_errors=ignore_errors,
            truncate_ragged_lines=truncate_ragged_lines,
            encoding=encoding,
            infer_schema_length=infer_schema_length,
            schema_overrides=schema_overrides,
            try_parse_dates=try_parse_dates,
            glob=True,
        )
    var options = _CsvScanOptions(
        has_header,
        separator,
        quote_char,
        comment_prefix,
        skip_rows,
        null_values.copy(),
        ignore_errors,
        truncate_ragged_lines,
        encoding,
        infer_schema_length,
        schema_overrides.copy(),
        try_parse_dates,
    )
    # Scan slots are parallel: frames[k] and schemas[k] belong to one scan.
    return LazyFrame(
        [_csv_scan_node(path, options)],
        [DataFrame([])],
        [Optional[CsvSchema]()],
    )


def scan_csv(
    path: String,
    schema: CsvSchema,
    *,
    has_header: Bool = True,
    separator: String = ",",
    quote_char: String = '"',
    comment_prefix: String = "",
    skip_rows: Int = 0,
    null_values: List[String] = List[String](),
    ignore_errors: Bool = False,
    truncate_ragged_lines: Bool = False,
    encoding: String = "utf8",
    glob: Bool = True,
) raises -> LazyFrame:
    """Lazily read a CSV file with an explicit schema, which labels the
    fields by position; the options are read_csv's (see the overload
    above)."""
    if glob and has_pattern(path):
        return scan_csv(
            [path],
            schema,
            has_header=has_header,
            separator=separator,
            quote_char=quote_char,
            comment_prefix=comment_prefix,
            skip_rows=skip_rows,
            null_values=null_values,
            ignore_errors=ignore_errors,
            truncate_ragged_lines=truncate_ragged_lines,
            encoding=encoding,
            glob=True,
        )
    var options = _CsvScanOptions.defaults()
    options.has_header = has_header
    options.separator = separator
    options.quote_char = quote_char
    options.comment_prefix = comment_prefix
    options.skip_rows = skip_rows
    options.null_values = null_values.copy()
    options.ignore_errors = ignore_errors
    options.truncate_ragged_lines = truncate_ragged_lines
    options.encoding = encoding
    return LazyFrame(
        [_csv_scan_node(path, options)],
        [DataFrame([])],
        [Optional[CsvSchema](schema.copy())],
    )


def scan_ndjson(
    path: String,
    schema: Optional[DataType] = None,
    *,
    infer_schema_length: Int = 100,
    ignore_errors: Bool = False,
) raises -> LazyFrame:
    """Defer NDJSON reading until collection or sampled schema discovery.

    Projection and head push into the reader. Streaming parses bounded
    row batches on the existing worker pool and preserves source order.
    Schema inference and conversion rules match read_ndjson.
    """
    _json_options(schema, infer_schema_length, 65536)
    return LazyFrame(
        [
            _ndjson_scan_node(
                path=path,
                schema=schema,
                infer_schema_length=infer_schema_length,
                ignore_errors=ignore_errors,
                slot=0,
            )
        ],
        [DataFrame([])],
        [Optional[CsvSchema]()],
    )


def scan_ipc(path: String, *, memory_map: Bool = True) raises -> LazyFrame:
    """Lazily scan an IPC/Feather v2 file in record batches.

    Projection and nonnegative slices push into the reader. Skipped batches
    need only metadata; schema queries read no record-batch bodies. Mapped
    input still copies into owned Mojo buffers. Files are opened on collect
    or schema discovery, so a plan can be collected repeatedly.
    """
    return LazyFrame(
        [_ipc_scan_node(path=path, memory_map=memory_map, slot=0)],
        [DataFrame([])],
        [Optional[CsvSchema]()],
    )


def scan_parquet(
    path: String,
    *,
    glob: Bool = True,
) raises -> LazyFrame:
    """Lazily read a Parquet file. Nothing is read until collect; projection
    is pushed into the reader, and a filter directly above the scan decodes
    only the row groups whose footer statistics can hold a match."""
    if glob and has_pattern(path):
        return scan_parquet(
            [path],
            glob=True,
        )
    return LazyFrame(
        [_parquet_scan_node(path=path, slot=0)],
        [DataFrame([])],
        [Optional[CsvSchema]()],
    )


def _concat_plans(frames: List[LazyFrame], how: String) raises -> LazyFrame:
    """The plan `concat(frames, how)` builds (see frame.concat)."""
    if len(frames) == 0:
        raise Error("concat requires at least one dataframe")
    if how != "vertical" and how != "diagonal" and how != "horizontal":
        raise Error(
            "concat how must be 'vertical', 'diagonal', or 'horizontal'"
        )
    var inputs = frames.copy()
    if how == "diagonal":
        # The union of the columns in first-seen order; a name shared by
        # inputs must have one type in all of them.
        var names = List[String]()
        var dtypes = Dict[String, DataType]()
        var schemas = List[DataFrame](capacity=len(frames))
        for f in range(len(frames)):
            schemas.append(frames[f]._input_schema())
            for field in schemas[f].schema():
                if field.name not in dtypes:
                    dtypes[field.name] = field.dtype
                    names.append(field.name)
                elif dtypes[field.name] != field.dtype:
                    raise Error(
                        "concat diagonal: column "
                        + field.name
                        + " is "
                        + field.dtype.name()
                        + " in frame "
                        + String(f)
                        + ", expected "
                        + dtypes[field.name].name()
                    )
        for f in range(len(frames)):
            var present = schemas[f].columns()
            if _same_names(present, names):
                continue
            var exprs = List[Expr](capacity=len(names))
            for name in names:
                if name in present:
                    exprs.append(col(name))
                else:
                    exprs.append(null(dtypes[name]).alias(name))
            inputs[f] = frames[f].select_exprs(exprs)
    var result = inputs[0].copy()
    for f in range(1, len(inputs)):
        if how == "horizontal":
            result = result._binary(inputs[f], _hconcat_node())
        else:
            result = result.concat(inputs[f])
    return result^


def _new_context(
    var nodes: List[PlanNode],
    var frames: List[DataFrame],
    var schemas: List[Optional[CsvSchema]],
) -> QueryContext:
    return QueryContext(
        nodes^,
        frames^,
        schemas^,
        plan_execute,
        _collect_context,
        _supports_context,
        _estimate_context,
        _handoff_context,
    )


def _collect_context(
    plan: QueryContext,
    optimize: Bool,
    streaming: Bool,
    batch_size: Int,
) raises -> DataFrame:
    return _from_context(plan.copy())._collect(
        optimize=optimize,
        streaming=streaming,
        batch_size=batch_size,
    )


def _supports_context(plan: QueryContext) -> Bool:
    return _handoff_supports(_from_context(plan.copy()))


def _estimate_context(plan: QueryContext, rows: Int) -> Tuple[String, Float64]:
    return _handoff_estimate(_from_context(plan.copy()), rows)


def _handoff_context(plan: QueryContext) raises -> DataFrame:
    return _handoff_execute(_from_context(plan.copy()))


def _from_context(var context: QueryContext) -> LazyFrame:
    """Move internal plan state into the public facade without copying buffers."""
    # Initialize through the existing constructor, then transfer the context
    # whole. Empty lists need no source storage or intermediate plan copies.
    var result = LazyFrame(
        List[PlanNode](), List[DataFrame](), List[Optional[CsvSchema]]()
    )
    result._plan = context^
    return result^


def scan_csv(
    paths: List[String],
    schema: CsvSchema,
    *,
    has_header: Bool = True,
    separator: String = ",",
    quote_char: String = '"',
    comment_prefix: String = "",
    skip_rows: Int = 0,
    null_values: List[String] = List[String](),
    ignore_errors: Bool = False,
    truncate_ragged_lines: Bool = False,
    encoding: String = "utf8",
    glob: Bool = True,
) raises -> LazyFrame:
    """Scan sorted local file sources; input rows are read during collection.

    Strict concatenation requires matching output names, order and dtypes.
    Repeated sources retain repeated rows. Use glob=False for literal paths.
    """
    var files = resolve_paths(paths, glob)
    var frames = List[LazyFrame](capacity=len(files))
    for path in files:
        var frame = scan_csv(
            path,
            schema,
            has_header=has_header,
            separator=separator,
            quote_char=quote_char,
            comment_prefix=comment_prefix,
            skip_rows=skip_rows,
            null_values=null_values,
            ignore_errors=ignore_errors,
            truncate_ragged_lines=truncate_ragged_lines,
            encoding=encoding,
            glob=False,
        )
        frames.append(frame^)
    return _concat_plans(frames, "vertical")


def scan_csv(
    paths: List[String],
    *,
    has_header: Bool = True,
    separator: String = ",",
    quote_char: String = '"',
    comment_prefix: String = "",
    skip_rows: Int = 0,
    null_values: List[String] = List[String](),
    ignore_errors: Bool = False,
    truncate_ragged_lines: Bool = False,
    encoding: String = "utf8",
    infer_schema_length: Int = 100,
    schema_overrides: Dict[String, String] = Dict[String, String](),
    try_parse_dates: Bool = False,
    glob: Bool = True,
) raises -> LazyFrame:
    """Scan sorted local file sources; input rows are read during collection.

    Strict concatenation requires matching output names, order and dtypes.
    Repeated sources retain repeated rows. Use glob=False for literal paths.
    """
    var files = resolve_paths(paths, glob)
    var frames = List[LazyFrame](capacity=len(files))
    for path in files:
        var frame = scan_csv(
            path,
            has_header=has_header,
            separator=separator,
            quote_char=quote_char,
            comment_prefix=comment_prefix,
            skip_rows=skip_rows,
            null_values=null_values,
            ignore_errors=ignore_errors,
            truncate_ragged_lines=truncate_ragged_lines,
            encoding=encoding,
            infer_schema_length=infer_schema_length,
            schema_overrides=schema_overrides,
            try_parse_dates=try_parse_dates,
            glob=False,
        )
        frames.append(frame^)
    return _concat_plans(frames, "vertical")


def scan_parquet(
    paths: List[String],
    *,
    glob: Bool = True,
) raises -> LazyFrame:
    """Scan sorted local file sources; input rows are read during collection.

    Strict concatenation requires matching output names, order and dtypes.
    Repeated sources retain repeated rows. Use glob=False for literal paths.
    """
    var files = resolve_paths(paths, glob)
    var frames = List[LazyFrame](capacity=len(files))
    for path in files:
        var frame = scan_parquet(
            path,
            glob=False,
        )
        frames.append(frame^)
    return _concat_plans(frames, "vertical")
