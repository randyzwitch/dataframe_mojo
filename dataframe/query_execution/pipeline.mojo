"""Streaming pipeline lowering and execution."""
from dataframe.io.dataset_scan import _DatasetBatches
from dataframe.io.sinks import FileSink
from dataframe.io.csv_reader import _open_csv_batches
from dataframe.io.csv_types import _CsvScanOptions

from dataframe.planning.node_builders import (
    _frame_scan_node,
    _join_node,
)
from dataframe.planning.cost.aggregation import (
    select_aggregation_strategy,
    refine_string_sample,
)
from dataframe.planning.cost.eligibility import (
    pipeline_eligibility,
    consider_inline_table,
)
from dataframe.planning.cost.model import trace_decision

from dataframe.categorical import (
    decode,
)
from dataframe.storage.column import (
    Column,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.execution import direct_reductions
from dataframe.expr import (
    COL,
    Expr,
    MAX,
    conjuncts,
    is_grouping,
    lit,
)
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.aggregation.expressions import (
    _row_local,
    _stream_reductions,
)
from dataframe.aggregation.state import (
    table_supported,
)
from dataframe.kernels.hashing import (
    encode_rows,
)
from dataframe.io.csv import (
    CsvSchema,
)
from dataframe.io.csv_reader import (
    _CsvBatches,
    _DecodeJob,
)
from dataframe.io.csv_types import (
    _map_file,
)
from dataframe.io.ipc import (
    _IpcBatches,
)
from dataframe.io.json_batches import (
    _NdjsonBatches,
)
from dataframe.io.json_parallel import (
    _DecodeJob as _JsonDecodeJob,
)
from dataframe.io.parquet import (
    _ParquetBatches,
    _pruned_row_groups,
    parquet_row_group_statistics,
)
from dataframe.kernels.join_hash import (
    PreparedHashIndex,
    prefer_left_build,
    prepare_hash_index,
    prepare_progression_index,
)
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_SEMI,
)
from dataframe.parallel import (
    Pool,
    configured_workers,
)
from dataframe.pipeline import (
    _TopBound,
    _ensure_zones,
    _range_key,
    _window_bounds,
    run_pipeline,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.graph import (
    plan_columns_of,
    plan_estimated_rows,
    plan_filters_rows,
    plan_grouped_on,
    plan_height_bound,
    plan_keeps_column,
    plan_key_source,
    plan_known_height,
    plan_progression_key,
)
from dataframe.planning.model import (
    AGG,
    DROP,
    FILTER,
    GROUPING_SETS,
    JOIN,
    PlanNode,
    SCAN_CSV,
    SCAN_FRAME,
    SCAN_IPC,
    SCAN_NDJSON,
    SCAN_PARQUET,
    SELECT,
    SLICE,
    SORT,
    WITH_COLUMNS,
)
from dataframe.planning.cost.aggregation import (
    plan_batch_states_win,
    plan_row_number_shape,
    plan_sampled_groups,
)
from dataframe.planning.physical import (
    plan_direct_key,
    plan_joins_below,
    plan_swap_join_sides,
    plan_swap_left_join_sides,
)
from dataframe.planning.properties import (
    _distinct_inputs,
    _is_scan,
    _same_names,
    _set_members,
    _stream_rows,
)
from dataframe.planning.cost.eligibility import (
    _counts_distinct,
    _filters,
    _late_key_build,
)
from dataframe.planning.cost.sampling import (
    _key_group_bound,
    _keys_repeat,
    _unique_key_column,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from dataframe.planning.scans import (
    _scan_options,
)
from dataframe.planning.cost.sampling import (
    _all_reduced,
    _frame_bytes,
    _judge_held,
    _stream_collect_bytes,
    _stream_split_groups,
)
from dataframe.query_execution.materialize import (
    plan_prefetch_breakers,
)
from dataframe.query_execution.operators import (
    _set_rows,
    plan_join_frames,
)
from dataframe.query_execution.reporting import (
    plan_record,
)
from dataframe.query_execution.stream_state import (
    _StateAccumulator,
    _StreamJob,
    _apply_operation,
    _coalesced,
    _steps_of,
)
from dataframe.planning.cost.eligibility import (
    _pipeline_steps,
)
from dataframe.series import (
    Series,
)
from dataframe.sorted_range import sorted_window
from dataframe.aggregation.state import (
    _StreamReduction,
    _finish_parts,
)
from dataframe.trace import (
    trace_path,
)
from std.collections import (
    Dict,
    Optional,
)
from std.memory import (
    ArcPointer,
    Pointer,
)
from std.time import (
    perf_counter_ns,
)


def plan_stream_execute(
    self: QueryContext, index: Int, batch_size: Int, sink: Int = 0
) raises -> Optional[DataFrame]:
    var cursor = index
    var expressions = List[Expr]()
    var keys = List[String]()
    var ordered = False
    var skip = 0
    var limit = -1
    ref terminal = self._nodes[index]
    var grouping = terminal.kind == GROUPING_SETS
    if grouping:
        # The aggregations, without the grouping columns (those are
        # constants of each set).
        for e in terminal.exprs:
            if not is_grouping(e):
                expressions.append(e.copy())
        if not _stream_reductions(expressions, True):
            return None
        keys = terminal.names.copy()
        cursor = terminal.left
    elif terminal.kind == AGG or terminal.kind == SELECT:
        # A grouped median or quantile streams only into the aggregate
        # table (checked below): its other routes keep no such state.
        if _stream_reductions(terminal.exprs, terminal.kind == AGG):
            expressions = terminal.exprs.copy()
            if terminal.kind == AGG:
                keys = terminal.names.copy()
                ordered = terminal.maintain_order
                if len(keys) == 0:
                    return None
            cursor = terminal.left
    if terminal.kind == SLICE:
        if terminal.slice_offset() < 0:
            return None
        skip = terminal.slice_offset()
        limit = terminal.slice_length()
        cursor = terminal.left
    # A sort limited to its first rows (#332) streams: each batch keeps
    # its own first rows, which contain the overall first rows, since a
    # row among those is among its own batch's. Batches arrive in input
    # order, so ties keep the stable sort's order.
    var top = -1
    var top_node = -1
    var top_names = List[String]()
    var top_descending = List[Bool]()
    var top_nulls_last = List[Bool]()
    if (
        terminal.kind == SLICE
        and cursor >= 0
        and self._nodes[cursor].kind == SORT
        and self._nodes[cursor].sort_limit() >= 0
    ):
        top = self._nodes[cursor].sort_limit()
        top_node = cursor
        top_names = self._nodes[cursor].names.copy()
        var n = len(top_names)
        for k in range(n):
            top_descending.append(self._nodes[cursor].sort_descending(k))
            top_nulls_last.append(self._nodes[cursor].sort_nulls_last(k))
        cursor = self._nodes[cursor].left
    var candidates = List[DataFrame]()
    var candidate_rows = 0
    var operations = List[PlanNode]()
    # The plan node of each operation, for the execution report.
    var operation_nodes = List[Int]()
    var joins = List[DataFrame]()
    var indexes = List[Optional[PreparedHashIndex]]()
    var reduced = plan_reduce_by_probe_keys(self, cursor, index, batch_size)
    if reduced:
        return reduced^
    if self._prefetched and not self._report:
        plan_prefetch_breakers(self, cursor, batch_size)
    # A computed source (a union, an aggregate, a sort) is run in full
    # whatever the stream does with it; as an in-memory table its rows
    # run through the morsel pipelines and the aggregate table on every
    # worker, as DuckDB's union children sink into their parent's
    # aggregate. Batched, a grouped reduction over it ran on one thread
    # (TPC-DS q76's three channels). It is found and run before any
    # join below builds, so no build is made twice.
    var first = cursor
    while first >= 0:
        ref step = self._nodes[first]
        if _stream_rows(step):
            first = step.left
        elif step.kind == JOIN and (
            step.how
            in [
                JOIN_INNER,
                JOIN_LEFT,
                JOIN_SEMI,
                JOIN_ANTI,
                JOIN_CROSS,
            ]
            or (step.how == JOIN_RIGHT and not step.coalesce)
        ):
            first = step.left
        else:
            break
    if first >= 0 and first != index and not _is_scan(self._nodes[first].kind):
        var computed = self._execute(first, False, True, batch_size)
        var plan = self.copy()
        plan._frames.append(computed^)
        plan._schemas.append(Optional[CsvSchema]())
        plan._nodes[first] = _frame_scan_node(slot=len(plan._frames) - 1)
        return _stream_after_materialization(plan, index, batch_size, sink)
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if _stream_rows(node):
            operations.append(node.copy())
            operation_nodes.append(cursor)
        elif node.kind == JOIN and (
            node.how
            in [
                JOIN_INNER,
                JOIN_LEFT,
                JOIN_SEMI,
                JOIN_ANTI,
                JOIN_CROSS,
            ]
            or (node.how == JOIN_RIGHT and not node.coalesce)
        ):
            var prepared = Optional[PreparedHashIndex]()
            # A left input that can only be small next to a large known
            # right may be the side to hash: decline streaming, so the
            # plan runs eagerly and `DataFrame.join` builds on the smaller
            # side, as Polars' `det_hash_prone_order` and DuckDB's
            # build/probe optimizer do (#377). Streaming would build on
            # the right: PDS-H q17 hashed 6M lineitem rows to match the
            # few hundred parts its filter keeps.
            var bound = plan_height_bound(self, node.left)
            var right_rows = plan_known_height(self, node.right)
            # A left input made by another join has no bound; how many
            # rows it holds is known only once it has run.
            var unbounded = bound < 0 and plan_joins_below(self, node.left)
            if (
                (node.how == JOIN_INNER or node.how == JOIN_LEFT)
                and plan_known_height(self, node.left) < 0
                and right_rows >= 524288
                and (unbounded or (bound >= 0 and 4 * bound <= right_rows))
            ):
                # A join promises no row order: build on the
                # small left and stream the large right through it in
                # the pipeline, as DuckDB's build side optimizer does
                # whatever side a join was written on, rather than run
                # the join outside the pipeline (TPC-DS q43 probed its
                # 366 dates with 2.9M sales rows eagerly).
                # A right side whose one key is a progression is no hash
                # table to build: the left streams through its rows by
                # position, which no swap beats.
                if (
                    node.how == JOIN_INNER
                    and not node.sides_fixed
                    and plan_direct_key(self, node.left, node.names)
                    and not plan_progression_key(
                        self, node.right, node.right_keys
                    )
                ):
                    var large = self._execute(
                        node.right, False, True, batch_size
                    )
                    var swapped = plan_swap_join_sides(
                        self, cursor, large, batch_size
                    )
                    if swapped:
                        var plan = swapped.take()
                        for k in range(len(operations)):
                            if operations[k].kind != JOIN:
                                continue
                            plan._frames.append(
                                joins[operations[k].join_build_slot()].copy()
                            )
                            plan._schemas.append(Optional[CsvSchema]())
                            plan._nodes.append(
                                _frame_scan_node(slot=len(plan._frames) - 1)
                            )
                            plan._nodes[operation_nodes[k]].right = (
                                len(plan._nodes) - 1
                            )
                        return _stream_after_materialization(
                            plan, index, batch_size, sink
                        )
                # The bound only says the left side may be small; the
                # eager join then decides with the real heights (a
                # filter often keeps far fewer rows than it scans).
                if len(joins) == 0:
                    return None
                # Joins above this one stream faster than they run
                # eagerly (PDS-H q8 and q9 were 1.5 times slower eager),
                # so only this join leaves the stream: it becomes the
                # stream's source, executed as a plan of its own, where
                # the rule above runs it eagerly. PDS-H q8 hashed 6M
                # lineitem rows to match the 1,300 parts its filter
                # keeps. Leaving the stream costs the joins above their
                # prepared probe of this one's output, so it takes a
                # wider margin: PDS-H q16 (200K parts at most, 800K
                # partsupp rows) was 10% slower leaving it. A left input
                # made by other joins has no bound and leaves too: PDS-H
                # q5 hashed 6M lineitem rows for about 45K orders, and
                # q18, q11, q7 and q2 gained 15-44%. Where that input is
                # large, streaming was faster: q9's partsupp and orders
                # joins (about 320K rows on the left) are 14% slower.
                if unbounded or 8 * bound <= right_rows:
                    break
            if (
                plan_known_height(self, node.left) <= batch_size
                and (node.how == JOIN_INNER or node.how == JOIN_LEFT)
                and prefer_left_build(
                    plan_known_height(self, node.left),
                    plan_known_height(self, node.right),
                )
            ):
                # This left input already fits one stream batch, so
                # physical reversal does not increase the materialized
                # join output over the existing batch bound. Keep larger
                # left inputs on the bounded streaming executor.
                # A directly available progression already has a cheaper
                # compact lookup. Validate once and retain its prepared
                # state instead of falling back to eager progression scans.
                ref build_node = self._nodes[node.right]
                if build_node.kind == SCAN_FRAME and len(node.right_keys) == 1:
                    var sources: List[Series] = [
                        self._frames[build_node.source_slot()][
                            node.right_keys[0]
                        ].copy()
                    ]
                    prepared = prepare_progression_index(sources)
                if not prepared:
                    break
            var operation = node.copy()
            operation.set_join_build_slot(len(joins))
            var algorithm = String(
                "progression" if prepared else (
                    "cross" if node.how == JOIN_CROSS else "eager_hash"
                )
            )
            var built = self._execute(node.right, False, True, batch_size)
            if (
                node.how == JOIN_INNER
                and not node.sides_fixed
                and bound >= 0
                and built.height() > bound
                and not plan_progression_key(self, node.right, node.right_keys)
            ):
                # The built right side holds more rows than the left
                # can, and is no progression (above): hash the left
                # instead and stream the right
                # through it, as DuckDB's build/probe side optimizer
                # and Polars' join both build on the smaller side.
                # A join promises no row order, and the stream
                # keeps every worker busy over the larger side, where
                # probing from a small left ran its few morsels on one
                # or two workers.
                var swapped = plan_swap_join_sides(
                    self, cursor, built, batch_size
                )
                if swapped:
                    # Joins above keep the right sides they built.
                    var plan = swapped.take()
                    for k in range(len(operations)):
                        if operations[k].kind != JOIN:
                            continue
                        plan._frames.append(
                            joins[operations[k].join_build_slot()].copy()
                        )
                        plan._schemas.append(Optional[CsvSchema]())
                        plan._nodes.append(
                            _frame_scan_node(slot=len(plan._frames) - 1)
                        )
                        plan._nodes[operation_nodes[k]].right = (
                            len(plan._nodes) - 1
                        )
                    return _stream_after_materialization(
                        plan, index, batch_size, sink
                    )
            if (
                (node.how == JOIN_INNER or node.how == JOIN_LEFT)
                and node.coalesce
                and not node.sides_fixed
            ):
                # DuckDB's build side (`BuildProbeSideOptimizer`): the
                # side of fewer rows times columns builds. The right
                # side has run; when the left one, estimated, is the
                # smaller, the join runs here on both, building on the
                # left and restoring its row order (`DataFrame.join`),
                # and the stream goes on from its result, rather than
                # hash every built row to probe a few: PDS-H q2 hashed
                # and rechunked 159K wide supplier rows for its 747
                # parts. The left's bound is read first, so a left that
                # cannot be smaller is never sampled.
                var right_cost = built.height() * max(built.width(), 1)
                var columns = len(plan_columns_of(self, node.left))
                var most = plan_height_bound(self, node.left)
                if most < 0 or most * columns < right_cost:
                    var rows = plan_estimated_rows(self, node.left)
                    if rows >= 0 and rows * columns < right_cost:
                        # The join builds on the smaller left inside the
                        # pipeline (a join promises no row order): the
                        # right streams through the left's index, a left
                        # join's unmatched left rows coming out last, as
                        # DuckDB flips a join's sides. The join's output
                        # is then never materialized outside the
                        # pipeline. The eager join below is left for a
                        # swap that declines (a shared name).
                        var swapped = plan_swap_left_join_sides(
                            self, cursor, built, batch_size
                        ) if node.how == JOIN_LEFT else plan_swap_join_sides(
                            self, cursor, built, batch_size
                        )
                        if swapped:
                            var plan = swapped.take()
                            for k in range(len(operations)):
                                if operations[k].kind != JOIN:
                                    continue
                                plan._frames.append(
                                    joins[
                                        operations[k].join_build_slot()
                                    ].copy()
                                )
                                plan._schemas.append(Optional[CsvSchema]())
                                plan._nodes.append(
                                    _frame_scan_node(slot=len(plan._frames) - 1)
                                )
                                plan._nodes[operation_nodes[k]].right = (
                                    len(plan._nodes) - 1
                                )
                            return _stream_after_materialization(
                                plan, index, batch_size, sink
                            )
                        trace_path("stream.smaller_left_build")
                        var left_frame = self._execute(
                            node.left, False, True, batch_size
                        )
                        var joined = plan_join_frames(
                            self, cursor, left_frame^, built^, False
                        )
                        var plan = self.copy()
                        plan._frames.append(joined^)
                        plan._schemas.append(Optional[CsvSchema]())
                        plan._nodes[cursor] = _frame_scan_node(
                            slot=len(plan._frames) - 1
                        )
                        return _stream_after_materialization(
                            plan, index, batch_size, sink
                        )
            if (
                (node.how == JOIN_SEMI or node.how == JOIN_ANTI)
                and built.height() >= 524288
                and plan_known_height(self, node.left) < 0
            ):
                # A semi or anti join whose built right side turned out
                # large: run the left too, and decide from both real
                # sizes, as DuckDB does. A left at most a quarter of the
                # right is joined eagerly, hashing the left (PDS-H q4:
                # about 57K orders against 3.8M late lineitem rows);
                # otherwise the stream continues from the left's rows.
                # Either way both results replace their nodes in a copy
                # of the plan, so neither side runs twice.
                var left_frame = self._execute(
                    node.left, False, True, batch_size
                )
                var plan = self.copy()
                if 4 * left_frame.height() <= built.height():
                    var joined = left_frame.join(
                        built,
                        left_on=node.names,
                        right_on=node.right_keys,
                        how="semi" if node.how == JOIN_SEMI else "anti",
                    )
                    plan._frames.append(joined^)
                    plan._schemas.append(Optional[CsvSchema]())
                    plan._nodes[cursor] = _frame_scan_node(
                        slot=len(plan._frames) - 1
                    )
                else:
                    plan._frames.append(left_frame^)
                    plan._schemas.append(Optional[CsvSchema]())
                    plan._nodes.append(
                        _frame_scan_node(slot=len(plan._frames) - 1)
                    )
                    plan._nodes[cursor].left = len(plan._nodes) - 1
                    plan._frames.append(built^)
                    plan._schemas.append(Optional[CsvSchema]())
                    plan._nodes.append(
                        _frame_scan_node(slot=len(plan._frames) - 1)
                    )
                    plan._nodes[cursor].right = len(plan._nodes) - 1
                return _stream_after_materialization(
                    plan, index, batch_size, sink
                )
            # The build frame in one chunk per column, once: a probe
            # gathers the build's payload through `take_parallel`,
            # which rechunks a chunked column every time it is
            # called, once per morsel or batch (PDS-H q2's 159K-row
            # build with five string columns: 3.8 ms a probe).
            for column in built._columns:
                if column.is_chunked():
                    built = built.rechunk()
                    break
            joins.append(built^)
            if not prepared and node.how != JOIN_CROSS and len(node.names):
                ref build = joins[len(joins) - 1]
                var sources = List[Series]()
                var supported = build.height() <= Int(Int32.MAX)
                for name in node.right_keys:
                    var key = build[name]
                    supported = supported and not key.dtype().is_nested()
                    sources.append(key.copy())
                if supported:
                    prepared = prepare_hash_index(sources, probe_rows=bound)
                    if prepared:
                        algorithm = "hash_index"
            plan_record(
                self,
                cursor,
                "streaming",
                0,
                0,
                executions=0,
                build_rows=joins[len(joins) - 1].height(),
                builds=1,
                algorithm=algorithm,
                build_side="right",
            )
            indexes.append(prepared^)
            operations.append(operation^)
            operation_nodes.append(cursor)
        else:
            break
        cursor = node.left
    if cursor < 0 or cursor == index and not _is_scan(self._nodes[cursor].kind):
        return None
    if cursor == index and self._nodes[cursor].kind == SCAN_FRAME:
        # The frame itself, with nothing to apply: batches would only
        # copy it into chunks that the eager step above rechunks
        # again (ClickBench q16 over 10M in-memory rows, 275 -> 181
        # ms).
        return None
    if (
        len(expressions) == 0
        and top < 0
        and limit < 0
        and skip == 0
        and len(joins) == 0
        and self._nodes[cursor].kind == SCAN_FRAME
        and not _filters(operations)
    ):
        # Row-local steps that keep every row of an in-memory frame
        # (with_columns, select, drop) under an eager step: a stream
        # applies them to batches and copies the results back
        # together, while the eager steps share the columns they do
        # not touch (ClickBench q18 adds a minute column to 10M rows
        # before a 4.9M-group aggregation).
        return None
    operations.reverse()
    operation_nodes.reverse()
    # Late materialization of group keys a join gathers from its build
    # side (a dimension's names): each such key's build column becomes
    # its values' codes, encoded once over the build frame, so the
    # probe gathers an integer instead of a string for every joined
    # row and the sink groups by integers; the result's codes become
    # the values again for its few groups. TPC-DS q99 spent a third of
    # its time gathering warehouse, ship mode and call center names
    # into 287K joined rows that formed 90 groups.
    var late_names = List[String]()
    var late_builds = List[Int]()
    # Group keys carried as one build's row numbers (see below).
    var row_original = Optional[DataFrame]()
    var original_keys = keys.copy()
    if (
        len(keys) > 0
        and len(expressions) > 0
        and len(joins) > 0
        and not ordered
        and top < 0
        and not grouping
        and self._nodes[cursor].kind == SCAN_FRAME
    ):
        var source_columns = self._frames[
            self._nodes[cursor].source_slot()
        ].columns()
        for name in keys:
            var b = _late_key_build(
                name, source_columns, joins, operations, expressions
            )
            if b < 0:
                continue
            late_names.append(name)
            late_builds.append(b)
    var distinct = _counts_distinct(expressions)
    var eligibility = pipeline_eligibility(
        self._nodes[cursor].kind,
        top,
        limit,
        skip,
        operations,
        indexes,
        distinct,
        len(keys),
        len(joins),
        ordered,
        expressions,
    )
    var inline_table_eligible = False
    if consider_inline_table(
        eligibility, len(keys), len(expressions), ordered, distinct
    ):
        var empty = self._frames[self._nodes[cursor].source_slot()].clear()
        var empty_joins = ArcPointer(joins.copy())
        var empty_indexes = ArcPointer(indexes.copy())
        for operation in operations:
            _apply_operation(operation, empty, empty_joins, empty_indexes)
        inline_table_eligible = table_supported(empty, keys, expressions)
    var decision = select_aggregation_strategy(
        self,
        cursor,
        batch_size,
        keys,
        expressions,
        operations,
        joins,
        distinct,
        ordered,
        top,
        grouping,
        late_names,
        late_builds,
        eligibility,
        inline_table_eligible,
    )
    if decision.is_eager():
        trace_decision(decision)
        return None
    var row_names = decision.row_names.copy()
    var row_holder = decision.row_holder
    var row_unique = decision.row_unique
    var row_merge = decision.row_merge.copy()
    if not decision.code_late_keys:
        late_names = List[String]()
    if len(row_names) > 0:
        trace_path("lazy.row_number_keys")
        row_original = Optional(joins[row_holder].copy())
        var first = row_names[0]
        var ids = List[Int64](capacity=joins[row_holder].height())
        for r in range(joins[row_holder].height()):
            ids.append(Int64(r))
        var surplus = List[String]()
        for k in range(1, len(row_names)):
            surplus.append(row_names[k])
        var carried = joins[row_holder].drop(surplus)
        joins[row_holder] = carried.with_column(
            Series(first, Column[Int64](ids^))
        )
        var kept_keys = List[String]()
        for name in keys:
            if name == first or name not in row_names:
                kept_keys.append(name)
        keys = kept_keys^
        # The select steps that passed the dropped names on.
        for k in range(len(operations)):
            if operations[k].kind != SELECT:
                continue
            var passed = List[Expr]()
            for e in operations[k].exprs:
                var dropped = False
                if (
                    len(e._nodes) == 1
                    and e._nodes[0].op == COL
                    and e._name == e._nodes[0].text
                    and e._name in row_names
                    and e._name != first
                ):
                    dropped = True
                if not dropped:
                    passed.append(e.copy())
            operations[k].exprs = passed^
    var late_values = List[Series]()
    var late_rows = List[List[Int]]()
    for k in range(len(late_names)):
        ref name = late_names[k]
        var b = late_builds[k]
        var values = joins[b].column(name).copy()
        var columns: List[Series] = [values.copy()]
        var groups = encode_rows(columns, nulls_equal=True)
        # A null value gets a null code: a left join's missing build
        # row is null too, and both are one null group.
        var codes = List[Int64](capacity=len(groups.ids))
        var valid = List[Bool](capacity=len(groups.ids))
        var nulls = values.null_count() > 0
        for r in range(len(groups.ids)):
            codes.append(Int64(groups.ids[r]))
            valid.append(not nulls or not values.get(r).is_null())
        joins[b] = joins[b].with_column(
            Series(name, Column[Int64](codes^, valid))
        )
        late_values.append(values^)
        late_rows.append(groups.representatives.copy())
    if top >= 0:
        operations.append(self._nodes[top_node].copy())
        operation_nodes.append(top_node)
    var shared_joins = ArcPointer(joins^)
    var shared_indexes = ArcPointer(indexes^)
    ref source = self._nodes[cursor]
    var csv = List[_CsvBatches]()
    var dataset = List[_DatasetBatches]()
    var parquet = List[_ParquetBatches]()
    var ipc = List[_IpcBatches]()
    var ndjson = List[_NdjsonBatches]()
    var input = DataFrame(List[Series](), height=0)
    if source.has_dataset():
        var columns = Optional[List[String]]()
        if source.dataset_projected:
            columns = source.scan_projection().copy()
        var scan = _CsvScanOptions.defaults()
        if source.kind == SCAN_CSV:
            scan = _scan_options(source)
        var predicate = Optional[Expr]()
        # A source limit precedes these operations. Pruning a file here
        # would change which rows an earlier head selects.
        if (
            source.scan_row_limit() < 0
            and len(operations)
            and operations[0].kind == FILTER
        ):
            predicate = operations[0].exprs[0].copy()
        source._record_execution()
        dataset.append(
            _DatasetBatches(
                source.dataset_spec(),
                source.kind == SCAN_PARQUET,
                schema=self._schemas[source.source_slot()],
                scan=scan,
                columns=columns,
                limit=source.scan_row_limit(),
                batch_size=batch_size,
                predicate=predicate,
            )
        )
    elif source.kind == SCAN_CSV:
        var opened = _open_csv_batches(
            source.text,
            self._schemas[source.source_slot()],
            source.names,
            source.scan_row_limit(),
            batch_size,
            _scan_options(source),
        )
        if not opened:
            return None
        csv.append(opened.take())
    elif source.kind == SCAN_PARQUET:
        var groups = Optional[List[Int]]()
        if len(operations) and operations[0].kind == FILTER:
            groups = _pruned_row_groups(
                parquet_row_group_statistics(source.text),
                operations[0].exprs[0],
            )
        parquet.append(
            _ParquetBatches(source.text, source.names, groups, batch_size)
        )
    elif source.kind == SCAN_NDJSON:
        source._record_execution()
        ndjson.append(
            _NdjsonBatches(
                source.text,
                source.json_schema,
                Int(source.ndjson_infer_schema_length()),
                source.ndjson_ignore_errors(),
                source.names,
                source.scan_row_limit(),
                batch_size,
            )
        )
    elif source.kind == SCAN_IPC:
        source._record_execution()
        ipc.append(
            _IpcBatches(
                source.text,
                source.names,
                source.ipc_has_projection(),
                source.ipc_memory_map(),
                Int(source.ipc_row_offset()),
                source.scan_row_limit(),
                batch_size,
            )
        )
    elif source.kind == SCAN_FRAME:
        source._record_execution()
        input = self._frames[source.source_slot()].copy()
        if len(source.names):
            input = input.select(source.names)
        # A first filter whose parts a column stored in ascending order
        # answers narrows the frame once to its run of rows, so only
        # those rows are batched (ClickBench q41: 613K of 10M rows, in
        # 10 batches instead of 153 that each searched again).
        if (
            len(operations) > 0
            and operations[0].kind == FILTER
            and _row_local(operations[0].exprs)
            and input.height() > 0
        ):
            var parts = conjuncts(operations[0].exprs[0])
            var window = sorted_window(input._columns, input.height(), parts)
            var any = False
            for answered in window.used:
                any = any or answered
            if any:
                trace_path("filter.sorted_range")
                input = input.slice(window.low, window.high - window.low)
                var rest = List[Expr]()
                for k in range(len(parts)):
                    if not window.used[k]:
                        rest.append(parts[k].copy())
                if len(rest) == 0:
                    operations[0].exprs = [lit(True)]
                else:
                    var remaining = rest[0].copy()
                    for k in range(1, len(rest)):
                        remaining = remaining & rest[k]
                    operations[0].exprs = [remaining^]
    else:
        input = self._execute(cursor, False, True, batch_size)
    decision = refine_string_sample(decision^, input, keys, operations)
    trace_decision(decision)
    if decision.is_eager():
        return None
    var workers = configured_workers()
    var pool = Pool(1)
    var pool_ready = False
    var offset = 0
    var emitted = False
    var ended = False
    var outputs = List[DataFrame]()
    # Execution report: rows fed from the source, and rows into and out
    # of each operation, summed over batches.
    var counting = Bool(self._report)
    var fed_rows = 0
    var operation_rows = List[Int](
        length=2 * len(operations) if counting else 0, fill=0
    )
    # Per operation, then the batches' reductions.
    var operation_times = List[Int](
        length=len(operations) + 1 if counting else 0, fill=0
    )
    var stream_began = Int(perf_counter_ns())
    # Aggregate state (#326). Batch states wait in `pending` and are
    # merged together once their groups reach the accumulated count (or
    # 64 batches), so each merge covers at least as much new work as old
    # and total merge work stays linear, with pending state about the
    # size of the accumulated state. When the first batch shows many
    # groups, every batch state is split into hash parts that are merged
    # on separate workers and interleaved by first occurrence at the end.
    var split_groups = _stream_split_groups()
    var acc = _StateAccumulator(workers, split_groups)
    # A grouped reduction whose first batch shows many groups collects
    # the batches (the columns it reads) and runs one eager group-by
    # at the end, where the hash-partitioned reduce encodes every key
    # once and never merges: a state per batch merged through a key
    # index cost TPC-DS q39 (1M groups) 761 ms against DuckDB's 32.
    var collecting = False
    var collected = List[DataFrame]()
    var collected_bytes = 0
    var collected_rows = 0
    var collect_budget = _stream_collect_bytes()
    # The decision waits for a batch with enough rows to judge: an
    # input sorted by date yields empty batches first where a filter
    # keeps a later year. Until then the batches' frames are kept
    # beside their states, to be collected or dropped.
    var undecided = len(expressions) > 0 and len(keys) > 0
    # The columns distinct counts read, judged with the keys: a batch's
    # state holds each group's set of values, so its size is the
    # distinct (key, value) pairs, not the groups. One that reads an
    # expression collects (its values are assumed many).
    var distinct_inputs = _distinct_inputs(expressions)
    if undecided and not distinct_inputs:
        collecting = True
        undecided = False
    var held = List[DataFrame]()
    # Each held batch's state, when its job built one (a batch under
    # 8 * split_groups rows: a few ms at most, in parallel, and used
    # if the stream keeps batch states); a longer batch whose first
    # rows show many groups keeps its frame only, since its state
    # would be the expensive one to drop (TPC-DS q39: 10 ms for 95K
    # rows), and is reduced in a replay round if needed after all.
    var held_states = List[List[_StreamReduction]]()
    var held_rows = 0
    var projection = List[String]()
    if len(expressions) and len(keys):
        var wanted = List[String]()
        var any = True
        for name in keys:
            wanted.append(name)
        for expression in expressions:
            var reads = _references(expression)
            if not reads:
                any = False
                break
            for name in reads.value():
                if name not in wanted:
                    wanted.append(name)
        if any:
            projection = wanted^
    # Batches of a large in-memory input that streams through joins grow
    # up to four times the default (not into the aggregate table, which
    # holds no state per batch: there a morsel's size is what spreads
    # rows a selective join keeps from a few stretches of sorted input
    # over the workers, as TPC-DS q39's date join keeps 180K of 11.7M
    # inventory rows from two of 45 grown batches): each batch probes every join, and
    # fewer, larger batches cost less per row (PDS-H q21's two joins of
    # 3.8M rows against 1.5M-row builds: 186 -> 148 ms). They grow only
    # while every worker still gets a batch: a 150K-row customer table
    # in one batch ran on one worker (q10, q13 and q22 were 7-23% slower).
    # A top-k bounds its first key once it holds k rows, when that key
    # is read from the input as it is: no join, and no operation that
    # writes a column of that name.
    var bound = _TopBound()
    var bounded = top > 0 and len(shared_joins[]) == 0
    if bounded:
        for operation in operations:
            if operation.kind == WITH_COLUMNS or operation.kind == SELECT:
                for e in operation.exprs:
                    if e._name != top_names[0]:
                        continue
                    var r = _references(e)
                    if not (
                        r
                        and len(r.value()) == 1
                        and r.value()[0] == e._name
                        and len(e._nodes) == 1
                    ):
                        bounded = False
    var rows_per_batch = batch_size
    if (
        len(shared_joins[]) > 0
        and input.height() > 0
        and not decision.uses_aggregate_table()
    ):
        # Still four batches a worker: input stored in key order (sales
        # by date) puts every row a selective join keeps into a few
        # batches, and one batch per worker left the others idle while
        # one worked through them (TPC-DS q99: 175K of 1.4M rows in one
        # 180K-row batch, 74 ms on one worker).
        rows_per_batch = max(
            batch_size, min(4 * batch_size, input.height() // (4 * workers))
        )
    elif (
        len(operations) == 0
        and len(expressions) > 0
        and len(keys) == 0
        and input.height() > 0
        and direct_reductions(expressions, input._columns)
    ):
        # Ungrouped reductions straight off an in-memory input that
        # each run as one scan of their column: four batches a worker
        # and no more, since each batch binds and sets up every
        # expression (ClickBench q29: ninety sums, 153 batches of 64K
        # rows cost 15 ms of that; 32 batches cost 4). A reduction
        # that materializes per-batch values keeps `batch_size`, which
        # fits them in cache (TPC-DS q9, 34 ms against 25 at 90K rows).
        rows_per_batch = max(batch_size, input.height() // (4 * workers))
    # Row-local steps over an in-memory frame into a materialize or an
    # ungrouped reduce sink run as a thread-owned pipeline (#538): each
    # worker takes morsels from a shared cursor and runs every step on
    # them; nothing is held, judged or merged on the main thread.
    if (
        sink == 0
        and len(dataset) == 0
        and decision.uses_pipeline()
        and len(csv) == 0
        and len(parquet) == 0
        and len(ipc) == 0
        and len(ndjson) == 0
    ):
        var began = Int(perf_counter_ns())
        var piped_counts = List[Int]()
        # A top-k's TOP_K node was appended to the operations for the
        # stream's per-batch selection; the pipeline's sink keeps the
        # k best itself, so the steps stop before it.
        var step_count = len(operations) - 1 if top > 0 else len(operations)
        var step_nodes = List[PlanNode](capacity=step_count)
        for k in range(step_count):
            step_nodes.append(operations[k].copy())
        var set_frames = List[DataFrame]()
        var set_indices = List[List[Int]]()
        if grouping:
            for members in _set_members(terminal):
                var indices = List[Int]()
                for name in members:
                    indices.append(keys.index(name))
                set_indices.append(indices^)
        var piped_result = run_pipeline(
            input,
            _steps_of(step_nodes),
            shared_joins,
            shared_indexes,
            expressions,
            keys,
            ordered,
            decision.inserts_groups(),
            workers,
            rows_per_batch,
            batch_size,
            piped_counts,
            set_frames,
            sets=set_indices,
            top=top,
            top_names=top_names,
            top_descending=top_descending,
            top_nulls_last=top_nulls_last,
            bounded=bounded,
            table=decision.uses_aggregate_table(),
        )
        if grouping:
            if len(set_frames) != len(set_indices):
                return None
            # The empty frame through the steps types the keys a set
            # leaves null.
            var dtypes = List[DataType]()
            for name in keys:
                dtypes.append(piped_result.column(name).dtype())
            var parts = List[DataFrame](capacity=len(set_frames))
            var members = _set_members(terminal)
            for k in range(len(set_frames)):
                parts.append(
                    _set_rows(
                        terminal, set_frames[k].copy(), members[k], dtypes
                    )
                )
            piped_result = concat(parts)
        if top > 0:
            piped_result = piped_result.slice(skip, limit)
        if len(row_names) > 0:
            # The groups' build row numbers back to the key values,
            # merging the groups whose rows hold the same values.
            var numbers = piped_result.column(row_names[0]).rechunk()
            ref ids = numbers._data[Column[Int64]]
            var rows = List[Int](capacity=len(numbers))
            for r in range(len(numbers)):
                rows.append(Int(ids._get(r)) if ids._valid(r) else -1)
            var columns = List[Series]()
            for name in original_keys:
                if name in row_names:
                    columns.append(
                        row_original.value()
                        .column(name)
                        .take_or_null(rows)
                        .renamed(name)
                    )
                else:
                    columns.append(piped_result.column(name))
            for e in expressions:
                columns.append(piped_result.column(e._name))
            if row_unique:
                piped_result = DataFrame(columns^)
            else:
                piped_result = (
                    DataFrame(columns^).group_by(original_keys).agg(row_merge)
                )
        for i in range(len(late_names)):
            # The groups' codes back to their values (a code is a
            # representative build row; a left join's missing row is a
            # null code and stays null).
            var coded = piped_result.column(late_names[i]).rechunk()
            ref codes = coded._data[Column[Int64]]
            var representatives = late_rows[i].unsafe_ptr()
            var rows = List[Int](capacity=len(coded))
            for r in range(len(coded)):
                if not codes._valid(r):
                    rows.append(-1)
                else:
                    rows.append(
                        representatives[unsafe_offset=Int(codes._get(r))]
                    )
            piped_result = piped_result.with_column(
                late_values[i].take_or_null(rows).renamed(late_names[i])
            )
        if counting:
            plan_record(self, cursor, "pipeline", 0, piped_counts[0])
            var into_terminal = piped_counts[0]
            for k in range(step_count):
                plan_record(
                    self,
                    operation_nodes[k],
                    "pipeline",
                    piped_counts[2 * k + 1],
                    piped_counts[2 * k + 2],
                    busy_ns=piped_counts[2 * step_count + 1 + k],
                )
                into_terminal = piped_counts[2 * k + 2]
            if top > 0:
                plan_record(
                    self,
                    top_node,
                    "pipeline",
                    into_terminal,
                    piped_result.height(),
                )
            var wall = Int(perf_counter_ns()) - began
            if index != cursor and (index not in operation_nodes):
                plan_record(
                    self,
                    index,
                    "pipeline",
                    into_terminal,
                    piped_result.height(),
                    wall_ns=wall,
                )
            else:
                plan_record(
                    self, index, "pipeline", 0, 0, executions=0, wall_ns=wall
                )
        return piped_result^
    # An in-memory input hands each round four batches a worker, which
    # the pool claims as workers free up, so uneven batches balance.
    # Batches decoded from a file are held until their round ends, so
    # those keep one a worker, as does a top-k (ClickBench q26 lost its
    # bound for a first round four times as long).
    var per_round = workers
    if (
        len(csv) == 0
        and len(dataset) == 0
        and len(parquet) == 0
        and len(ipc) == 0
        and len(ndjson) == 0
        and top < 0
    ):
        # Not for a top-k, whose bound tightens between rounds.
        per_round = 4 * workers
    # Batches held while undecided, to reduce in the next round once
    # the stream decides not to collect.
    var replay = List[DataFrame]()
    while not ended or len(replay) > 0:
        var jobs = List[_StreamJob]()
        for frame in replay:
            var job = _StreamJob(
                frame.copy(),
                List[PlanNode](),
                shared_joins,
                shared_indexes,
                expressions,
                keys,
            )
            job.bits = acc.bits
            job.counting = counting
            job.replay = True
            jobs.append(job^)
        replay = List[DataFrame]()
        for _ in range(per_round if not ended else 0):
            var frame = DataFrame(List[Series](), height=0)
            var decode = List[_DecodeJob]()
            var json_decode = List[_JsonDecodeJob]()
            if len(dataset):
                var next = dataset[0].next()
                if not next:
                    ended = True
                    break
                frame = next.take()
            elif len(csv):
                var next = csv[0].next()
                if not next:
                    ended = True
                    break
                decode.append(next.take())
            elif len(parquet):
                var next = parquet[0].next()
                if not next:
                    ended = True
                    break
                frame = next.take()
            elif len(ndjson):
                var next = ndjson[0].next()
                if not next:
                    ended = True
                    break
                json_decode.append(next.take())
            elif len(ipc):
                var next = ipc[0].next()
                if not next:
                    ended = True
                    break
                frame = next.take()
            else:
                if emitted and offset >= input.height():
                    ended = True
                    break
                frame = input.slice(offset, rows_per_batch)
                offset += frame.height()
                emitted = True
            var job = _StreamJob(
                frame^,
                operations,
                shared_joins,
                shared_indexes,
                expressions,
                keys,
            )
            job.decode = decode^
            job.json_decode = json_decode^
            job.bits = acc.bits
            job.split_groups = split_groups
            job.counting = counting
            job.collect_only = collecting
            job.keep_frame = undecided
            job.projection = projection.copy()
            job.bound = bound.copy()
            jobs.append(job^)
        if len(jobs) == 0:
            break
        if not pool_ready:
            pool = Pool(min(len(jobs), workers))
            pool_ready = True
        pool.run(jobs, claim=True)
        if counting:
            for i in range(len(jobs)):
                ref counts = jobs[i].counts
                if jobs[i].replay:
                    continue
                if len(counts) > 0:
                    fed_rows += counts[0]
                elif len(expressions):
                    fed_rows += jobs[i].rows
                else:
                    fed_rows += jobs[i].frame.height()
                for k in range(len(counts)):
                    operation_rows[k] += counts[k]
                for k in range(len(jobs[i].times)):
                    if k < len(operation_times):
                        operation_times[k] += jobs[i].times[k]
        # Pool returns jobs in submission order, independently of worker
        # completion order. Merge states and assemble rows in that order.
        for i in range(len(jobs)):
            if collecting:
                collected_bytes += _frame_bytes(jobs[i].frame)
                collected_rows += jobs[i].rows
                collected.append(jobs[i].frame.copy())
            elif len(expressions) and undecided:
                held.append(jobs[i].frame.copy())
                held_states.append(jobs[i].take_reduced())
                held_rows += jobs[i].rows
                # Judged on a sample of the rows held so far, once
                # there are twice `split_groups` of them: a selective
                # join before the group-by leaves a few hundred rows a
                # batch (TPC-DS q39: 500 of 32K), and one batch alone
                # never decides. The sample stays small: encoding it
                # is on the main thread, 1 ms for 8K rows.
                if held_rows >= 2 * split_groups:
                    undecided = False
                    # Many groups, at least one row in eight
                    # starting one: then most rows would go
                    # through the index on merge. TPC-DS q39
                    # (three rows a group) and q65 gain 35-40%,
                    # PDS-H q16 (one in four) 35%; PDS-H q13 (one
                    # in ten) pays 7% either way.
                    if _judge_held(
                        held,
                        held_states,
                        keys,
                        held_rows,
                        split_groups,
                        distinct_inputs.value(),
                    ):
                        # Many groups: collect every batch held so
                        # far and from here on; the states built
                        # for them are dropped.
                        collecting = True
                        for frame in held:
                            var kept = frame.select(projection) if len(
                                projection
                            ) else frame.copy()
                            collected_bytes += _frame_bytes(kept)
                            collected_rows += kept.height()
                            collected.append(kept^)
                        held = List[DataFrame]()
                        held_states = List[List[_StreamReduction]]()
                        acc.rows_seen = collected_rows
                        for j in range(i + 1, len(jobs)):
                            var kept = (
                                jobs[j]
                                .frame.select(projection) if len(
                                    projection
                                ) else jobs[j]
                                .frame.copy()
                            )
                            collected_bytes += _frame_bytes(kept)
                            collected_rows += jobs[j].rows
                            collected.append(kept^)
                        break
                    # Few groups, or keys too long to collect: the
                    # states, from the held batches and those of
                    # this round still to come, all in input order.
                    for j in range(i + 1, len(jobs)):
                        held.append(jobs[j].frame.copy())
                        held_states.append(jobs[j].take_reduced())
                    if _all_reduced(held_states):
                        for h in range(len(held)):
                            acc.absorb(held_states.pop(0), held[h].height())
                    else:
                        replay = _coalesced(held, 2 * split_groups)
                    held = List[DataFrame]()
                    held_states = List[List[_StreamReduction]]()
                    break
            elif len(expressions):
                acc.absorb(jobs[i].take_reduced(), jobs[i].rows)
            elif top >= 0:
                # The job has already cut its batch to its first rows.
                candidate_rows += jobs[i].frame.height()
                candidates.append(jobs[i].frame.copy())
            else:
                var part = jobs[i].frame.copy()
                var dropped = min(skip, part.height())
                skip -= dropped
                part = part.slice(dropped, limit)
                if limit >= 0:
                    limit -= part.height()
                if sink != 0:
                    Pointer[FileSink, MutAnyOrigin](
                        unsafe_from_address=sink
                    )[].append(part)
                else:
                    outputs.append(part^)
        if collecting and collected_bytes >= collect_budget:
            # The budget is reached: reduce what is collected into one
            # state (every key encoded once) and keep collecting. The
            # states merge at the end as batch states do.
            var part = _StreamReduction(concat(collected), expressions, keys)
            part.shift_firsts(acc.rows_seen)
            acc.rows_seen += collected_rows
            collected = List[DataFrame]()
            collected_bytes = 0
            collected_rows = 0
            if len(acc.states) == 0:
                acc.pending.append(List[_StreamReduction]())
                acc.pending_groups.append(0)
                acc.states.append(part^)
            else:
                acc.pending_groups[0] += part.group_count()
                acc.pending[0].append(part^)
        acc.merge(False)
        if top >= 0 and len(candidates) > 1 and candidate_rows > 4 * top:
            var merged = concat(candidates)
            merged = merged.take(
                merged._arg_sort_head(
                    top_names, top_descending, top_nulls_last, top
                )
            )
            candidate_rows = merged.height()
            if bounded:
                bound = _TopBound(merged, top_names[0], top_descending[0], top)
            candidates = [merged^]
        if len(csv):
            csv[0].discard()
        if limit == 0 and top < 0:
            ended = True
        if ended and undecided and len(held) > 0:
            # The input ran out first: the same judgement on what was
            # held. Many groups among few rows collect too (TPC-DS
            # q7: 5K groups in 7.7K rows), where splitting the states
            # into hash parts cost more than one eager group-by.
            undecided = False
            if _judge_held(
                held,
                held_states,
                keys,
                held_rows,
                split_groups,
                distinct_inputs.value(),
            ):
                collecting = True
                for frame in held:
                    var kept = frame.select(projection) if len(
                        projection
                    ) else frame.copy()
                    collected_bytes += _frame_bytes(kept)
                    collected_rows += kept.height()
                    collected.append(kept^)
                acc.rows_seen = collected_rows
            elif _all_reduced(held_states):
                for h in range(len(held)):
                    acc.absorb(held_states.pop(0), held[h].height())
            else:
                replay = _coalesced(held, 2 * split_groups)
            held = List[DataFrame]()
            held_states = List[List[_StreamReduction]]()
    pool.release()
    if len(csv):
        csv[0].finish()
    if len(dataset):
        dataset[0].finish()
    # NDJSON jobs borrow mapped addresses; retain the source through
    # the final worker join, even after an early head or an error.
    _ = ndjson^
    var result = Optional[DataFrame]()
    if collecting and len(acc.states) == 0:
        trace_path("lazy.collect_group_by")
        # Contiguous first: the bucket encode reads string keys in
        # view storage through a buffer table, slow across hundreds
        # of batches (TPC-DS q39: 10 ms of 40 on 358 batches).
        var whole = concat(collected).rechunk()
        # Groups come back in first-occurrence order only when asked,
        # as the eager group-by's do.
        result = whole.group_by(keys, maintain_order=ordered).agg(expressions)
    elif collecting:
        # Spilled at least once: the rest joins the states.
        trace_path("lazy.collect_group_by_bounded")
        if len(collected):
            var part = _StreamReduction(concat(collected), expressions, keys)
            part.shift_firsts(acc.rows_seen)
            acc.pending_groups[0] += part.group_count()
            acc.pending[0].append(part^)
        acc.merge(True)
        result = acc.states[0].finish()
    elif len(acc.states):
        acc.merge(True)
        if len(acc.states) == 1:
            result = acc.states[0].finish()
        else:
            var parts = List[_StreamReduction]()
            swap(parts, acc.states)
            result = _finish_parts(parts^)
    elif top >= 0:
        if len(candidates) > 0:
            var merged = concat(candidates)
            merged = merged.take(
                merged._arg_sort_head(
                    top_names, top_descending, top_nulls_last, top
                )
            )
            result = merged.slice(skip, limit)
    elif len(outputs):
        result = concat(outputs)
    if (
        sink != 0
        and not result
        and (
            len(expressions) == 0
            or (len(keys) > 0 and not grouping)
            or top >= 0
        )
    ):
        result = Pointer[FileSink, MutAnyOrigin](
            unsafe_from_address=sink
        )[].schema.copy()
    if counting:
        if _is_scan(source.kind):
            plan_record(self, cursor, "streaming", 0, fed_rows)
        var into_terminal = fed_rows
        for k in range(len(operations)):
            plan_record(
                self,
                operation_nodes[k],
                "streaming",
                operation_rows[2 * k],
                operation_rows[2 * k + 1],
                busy_ns=operation_times[k],
            )
            into_terminal = operation_rows[2 * k + 1]
        var stream_wall = Int(perf_counter_ns()) - stream_began
        if index != cursor and (index not in operation_nodes):
            plan_record(
                self,
                index,
                "streaming",
                into_terminal,
                result.value().height() if result else 0,
                busy_ns=operation_times[len(operations)],
                wall_ns=stream_wall,
            )
        else:
            # The stream's last node is one of its operations: add the
            # stream's wall time there, with nothing else.
            plan_record(
                self,
                index,
                "streaming",
                0,
                0,
                executions=0,
                wall_ns=stream_wall,
            )
    return result^


def plan_reduce_by_probe_keys(
    self: QueryContext, top: Int, index: Int, batch_size: Int
) raises -> Optional[DataFrame]:
    """Reduce aggregate input with a semi-join when a streaming join
    builds on aggregates grouped by the join keys. Run the probe-side
    filters first, then restrict each aggregate input to the probe
    side's distinct key tuples. Missing keys cannot contribute to the
    join, and filtering a group key removes whole groups without
    changing the result. For example, PDS-H q21 reduces 1.5M orders to
    the 76K lines retained by supplier and order filters. Apply this
    when the key count does not exceed the aggregate input row count.
    Return None when there is no such join or no probe-side filter."""
    var joins = List[Int]()
    var aggregates = List[Int]()
    # Each aggregate's names for the join's right keys.
    var group_keys = List[List[String]]()
    var key = List[String]()
    var probe = top
    var lowest = -1
    var steps = List[Int]()
    while probe >= 0:
        ref step = self._nodes[probe]
        if _stream_rows(step) or step.kind == DROP:
            steps.append(probe)
            probe = step.left
            continue
        if step.kind != JOIN or not (
            step.how
            in [JOIN_INNER, JOIN_LEFT, JOIN_SEMI, JOIN_ANTI, JOIN_CROSS]
        ):
            break
        if (
            (step.how == JOIN_INNER or step.how == JOIN_SEMI)
            and len(step.names) > 0
            and (len(key) == 0 or _same_names(step.names, key))
        ):
            # One aggregate grouped on every right key (PDS-H q20's
            # shipped quantities per part and supplier), each perhaps
            # renamed on the way up (TPC-DS q16's orders shipped from
            # two warehouses).
            var aggregate = plan_grouped_on(
                self, step.right, step.right_keys[0]
            )[0]
            var names = List[String]()
            for name in step.right_keys:
                var grouped = plan_grouped_on(self, step.right, name)
                if grouped[0] != aggregate:
                    aggregate = -1
                names.append(grouped[1])
            if aggregate >= 0:
                # The row-local steps between this join and the one
                # above keep the keys, so the keys reaching the higher
                # join are a subset of those reaching this one (joins
                # only drop or repeat left rows, keeping their columns).
                var kept = True
                for name in key:
                    for at in steps:
                        kept = kept and plan_keeps_column(self, at, name)
                if kept:
                    joins.append(probe)
                    aggregates.append(aggregate)
                    group_keys.append(names^)
                    key = step.names.copy()
                    lowest = probe
                    steps = List[Int]()
        probe = step.left
    if lowest < 0:
        return None
    var below = self._nodes[lowest].left
    # Its keys can be few only if something filters the probe side: a
    # filter, or an inner or semi join, which keeps only the rows its
    # build side matches (the filters on TPC-DS q16's dates, addresses
    # and call centers reach it as joins with their results, which the
    # join order ran to count them).
    if not plan_filters_rows(self, below):
        return None
    trace_path("stream.probe_key_reduction")
    var probed = self._execute(below, False, True, batch_size)
    var keys = probed.select(key).unique(key)
    var plan = self.copy()
    plan._frames.append(probed^)
    plan._schemas.append(Optional[CsvSchema]())
    plan._nodes.append(_frame_scan_node(slot=len(plan._frames) - 1))
    plan._nodes[lowest].left = len(plan._nodes) - 1
    var key_frame = len(plan._frames)
    # Each probe key's distinct values, and each source column's span,
    # found once: PDS-H q21's two aggregates read one lineitem column.
    var distinct = List[Int]()
    for name in key:
        distinct.append(
            keys.height() if len(key)
            == 1 else keys.select([name]).unique([name]).height()
        )
    var spans = Dict[String, Int]()
    plan._frames.append(keys^)
    plan._schemas.append(Optional[CsvSchema]())
    for k in range(len(joins)):
        var aggregate = aggregates[k]
        var input = plan._nodes[aggregate].left
        # The filter pays when it drops most of the groups: some key's
        # probe values are at most half of the values the aggregate's
        # key column spans (the half rule the pipeline's key ranges
        # keep). TPC-DS q35's 97K probing customers kept 94% of a
        # year's store sales, for 16 ms of probes.
        var pays = False
        ref grouped = group_keys[k]
        for i in range(len(grouped)):
            var source = plan_key_source(plan, input, grouped[i])
            if not source or not _range_key(source.value().dtype()):
                continue
            var named = grouped[i] + "/" + String(len(source.value()))
            if named not in spans:
                # From the column's zone maps, kept with its buffers as
                # DuckDB keeps a table's statistics: scanning the whole
                # column on one thread each run held PDS-H q20 at 19.6
                # ms (lineitem's 6M part and supplier keys), 12.3 now.
                _ensure_zones(source.value())
                var bounds = _window_bounds(source.value())
                spans[named] = (
                    Int(bounds.value()[1]) - Int(bounds.value()[0]) + 1
                ) if bounds else 0
            pays = pays or 2 * distinct[i] <= spans[named]
        if not pays:
            continue
        plan._nodes.append(_frame_scan_node(slot=key_frame))
        var semi = _join_node(
            left=input,
            right=len(plan._nodes) - 1,
            left_keys=grouped.copy(),
            join_type="semi",
            type_code=JOIN_SEMI,
            right_keys=key.copy(),
            suffix="_right",
        )
        plan._nodes.append(semi^)
        plan._nodes[aggregate].left = len(plan._nodes) - 1
    return plan._execute(index, False, True, batch_size)


def _stream_after_materialization(
    plan: QueryContext,
    index: Int,
    batch_size: Int,
    sink: Int,
) raises -> Optional[DataFrame]:
    if sink != 0:
        var streamed = plan_stream_execute(plan, index, batch_size, sink)
        if streamed:
            return streamed^
    return plan._execute(index, False, True, batch_size)
