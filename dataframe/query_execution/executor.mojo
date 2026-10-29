"""CPU plan interpretation, eager operators, and execution reporting."""
from dataframe.expr import (
    COL,
)
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.aggregation.expressions import (
    _row_local,
)
from dataframe.io.csv import (
    read_csv,
)
from dataframe.io.csv_reader import (
    _CsvBatches,
    read_csv_explicit,
    read_csv_inferred,
)
from dataframe.io.csv_types import (
    _map_file,
)
from dataframe.io.ipc import (
    _read_scan,
    _schema_frame as _ipc_schema_frame,
)
from dataframe.io.json import (
    read_ndjson,
)
from dataframe.io.parquet import (
    _pruned_row_groups,
    parquet_row_group_statistics,
    read_parquet,
)
from dataframe.kernels.join_hash import (
    prefer_left_build,
)
from dataframe.join_type import (
    JOIN_CROSS,
    JOIN_INNER,
    JOIN_LEFT,
)
from dataframe.parallel import (
    run_jobs,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.inputs import (
    _SubplanJob,
)
from dataframe.planning.model import (
    AGG,
    ASOF,
    CONCAT,
    EXPLODE,
    FILL_NULL_STRATEGY,
    FILTER,
    GATHER,
    GATHER_EVERY,
    GROUPING_SETS,
    HANDOFF_OFF,
    HCONCAT,
    INSERT_COLUMN,
    JOIN,
    PIVOT,
    REVERSE,
    ROW_INDEX,
    SCAN_CSV,
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
from dataframe.planning.properties import (
    _is_scan,
    _is_time_window,
    _schema_of,
)
from dataframe.planning.scans import (
    _scan_options,
)
from dataframe.query_execution.operators import (
    _gathered,
    _pivoted,
    _run_grouping_sets,
    _run_time_window,
    plan_join_frames,
)
from dataframe.query_execution.pipeline import (
    plan_stream_execute,
)
from dataframe.query_execution.placement import (
    plan_handoff_region,
)
from dataframe.query_execution.reporting import (
    plan_record,
)
from std.collections import (
    Optional,
)
from std.time import (
    perf_counter_ns,
)


def plan_execute(
    self: QueryContext,
    index: Int,
    empty: Bool,
    streaming: Bool = False,
    batch_size: Int = 65536,
) raises -> DataFrame:
    if self._prefetched and not empty:
        var ahead = self._prefetched.value()
        if index in ahead[]:
            return ahead[].pop(index)
    if self._handoff != HANDOFF_OFF and not empty:
        var handed = plan_handoff_region(self, index, streaming, batch_size)
        if handed:
            return handed.take()
    if streaming and not empty:
        var streamed = plan_stream_execute(self, index, batch_size)
        if streamed:
            return streamed.take()
    if empty and self._schema_memo:
        var memo = self._schema_memo.value()
        var known = memo[].get(index)
        if known:
            return known.take()
        var schema = plan_execute_node(self, index, True, streaming, batch_size)
        memo[][index] = schema.copy()
        return schema^
    if not self._report or empty:
        return plan_execute_node(self, index, empty, streaming, batch_size)
    # Record what the node did: its inputs record themselves as they
    # run below it, so the input rows are read back from the report.
    var began = Int(perf_counter_ns())
    var result = plan_execute_node(self, index, empty, streaming, batch_size)
    var took = Int(perf_counter_ns()) - began
    ref node = self._nodes[index]
    if _is_scan(node.kind):
        plan_record(self, index, "eager", 0, result.height(), wall_ns=took)
    elif node.kind == JOIN:
        var left = self._report.value()[].find(node.left)
        var right = self._report.value()[].find(node.right)
        var left_rows = left.value().output_rows if left else -1
        var right_rows = right.value().output_rows if right else -1
        var build_left = (
            node.how == JOIN_INNER or node.how == JOIN_LEFT
        ) and prefer_left_build(left_rows, right_rows)
        plan_record(
            self,
            index,
            "eager",
            left_rows,
            result.height(),
            build_rows=right_rows,
            builds=1,
            algorithm="cross" if node.how == JOIN_CROSS else "eager_hash",
            build_side="left" if build_left else "right",
            wall_ns=took,
        )
    else:
        var left = self._report.value()[].find(node.left)
        plan_record(
            self,
            index,
            "eager",
            left.value().output_rows if left else -1,
            result.height(),
            wall_ns=took,
        )
    return result^


def plan_execute_node(
    self: QueryContext,
    index: Int,
    empty: Bool,
    streaming: Bool,
    batch_size: Int,
) raises -> DataFrame:
    ref node = self._nodes[index]
    if node.kind == SCAN_FRAME:
        if not empty:
            if node.source_failure:
                raise Error(node.source_failure)
            node._record_execution()
        var frame = self._frames[node.source_slot()].copy()
        if len(node.scan_projection()) > 0:
            frame = frame.select(node.scan_projection())
        return frame.clear() if empty else frame^
    if node.kind == SCAN_CSV:
        if empty:
            # Metadata probes must not use eager n_rows=0: that API
            # intentionally decodes a source chunk before truncation.
            var mapping = _map_file(node.text)
            if mapping.address != 0:
                var batches = _CsvBatches(
                    mapping^,
                    self._schemas[node.source_slot()],
                    node.scan_projection(),
                    0,
                    batch_size,
                    _scan_options(node),
                )
                var next = batches.next()
                var job = next.take()
                job.run()
                return job^.into_frame()
        var rows = 0 if empty else node.scan_row_limit()
        var scan = _scan_options(node)
        if self._schemas[node.source_slot()]:
            return read_csv(
                node.text,
                self._schemas[node.source_slot()].value(),
                has_header=scan.has_header,
                separator=scan.separator,
                quote_char=scan.quote_char,
                comment_prefix=scan.comment_prefix,
                skip_rows=scan.skip_rows,
                n_rows=rows,
                columns=node.scan_projection(),
                null_values=scan.null_values,
                ignore_errors=scan.ignore_errors,
                truncate_ragged_lines=scan.truncate_ragged_lines,
                encoding=scan.encoding,
                glob=False,
            )
        return read_csv(
            node.text,
            infer_schema_length=scan.infer_schema_length,
            schema_overrides=scan.schema_overrides,
            try_parse_dates=scan.try_parse_dates,
            has_header=scan.has_header,
            separator=scan.separator,
            quote_char=scan.quote_char,
            comment_prefix=scan.comment_prefix,
            skip_rows=scan.skip_rows,
            n_rows=rows,
            columns=node.scan_projection(),
            null_values=scan.null_values,
            ignore_errors=scan.ignore_errors,
            truncate_ragged_lines=scan.truncate_ragged_lines,
            encoding=scan.encoding,
            glob=False,
        )
    if node.kind == SCAN_NDJSON:
        if not empty:
            node._record_execution()
        return read_ndjson(
            node.text,
            node.json_schema,
            infer_schema_length=Int(node.ndjson_infer_schema_length()),
            ignore_errors=node.ndjson_ignore_errors(),
            n_rows=0 if empty else node.scan_row_limit(),
            batch_size=batch_size,
            columns=node.scan_projection(),
        )
    if node.kind == SCAN_IPC:
        if empty:
            var schema = _ipc_schema_frame(node.text)
            return (
                schema.select(
                    node.scan_projection()
                ) if node.ipc_has_projection() else schema
                ^
            )
        node._record_execution()
        return _read_scan(
            node.text,
            node.scan_projection(),
            node.ipc_has_projection(),
            node.ipc_memory_map(),
            Int(node.ipc_row_offset()),
            node.scan_row_limit(),
        )
    if node.kind == SCAN_PARQUET:
        if empty:
            return read_parquet(
                node.text,
                columns=node.scan_projection(),
                row_groups=List[Int](),
                glob=False,
            )
        return read_parquet(
            node.text, columns=node.scan_projection(), glob=False
        )
    # Keep a sort's row permutation until after a plain projection, so
    # columns used only as sort keys are never gathered into output.
    if node.kind == SELECT and not empty:
        ref sorted = self._nodes[node.left]
        if sorted.kind == SORT:
            var plain = True
            for expression in node.exprs:
                plain = plain and len(expression._nodes) == 1
                if len(expression._nodes) == 1:
                    plain = plain and expression._nodes[0].op == COL
            if plain:
                var source = self._execute(
                    sorted.left, False, streaming, batch_size
                )
                var n = len(sorted.sort_keys())
                var descending = List[Bool]()
                var nulls_last = List[Bool]()
                for i in range(n):
                    descending.append(sorted.sort_descending(i))
                    nulls_last.append(sorted.sort_nulls_last(i))
                var order = source._arg_sort_head(
                    sorted.sort_keys(),
                    descending,
                    nulls_last,
                    sorted.sort_limit(),
                ) if sorted.sort_limit() >= 0 else source.arg_sort(
                    sorted.sort_keys(),
                    descending=descending,
                    nulls_last=nulls_last,
                )
                return source.select_exprs(node.exprs).take(order)
    # Filter each decoded CSV range before assembling the scan result.
    # A row limit and a whole-column predicate require the original
    # materialization order, so retain the eager path for those cases.
    if node.kind == FILTER and not empty and _row_local(node.exprs):
        ref source = self._nodes[node.left]
        # Decode only the row groups whose footer bounds admit a match;
        # the filter still runs on what was read.
        if source.kind == SCAN_PARQUET:
            var groups = _pruned_row_groups(
                parquet_row_group_statistics(source.text), node.exprs[0]
            )
            return read_parquet(
                source.text, columns=source.names, row_groups=groups, glob=False
            ).filter(node.exprs[0])
        if source.kind == SCAN_CSV and source.scan_row_limit() < 0:
            var scan = _scan_options(source)
            if self._schemas[source.source_slot()]:
                return read_csv_explicit(
                    source.text,
                    self._schemas[source.source_slot()].value(),
                    has_header=scan.has_header,
                    separator=scan.separator,
                    quote_char=scan.quote_char,
                    comment_prefix=scan.comment_prefix,
                    skip_rows=scan.skip_rows,
                    columns=source.names,
                    null_values=scan.null_values,
                    ignore_errors=scan.ignore_errors,
                    truncate_ragged_lines=scan.truncate_ragged_lines,
                    encoding=scan.encoding,
                    predicate=Optional(node.exprs[0].copy()),
                )
            return read_csv_inferred(
                source.text,
                infer_schema_length=scan.infer_schema_length,
                schema_overrides=scan.schema_overrides,
                has_header=scan.has_header,
                separator=scan.separator,
                quote_char=scan.quote_char,
                comment_prefix=scan.comment_prefix,
                skip_rows=scan.skip_rows,
                columns=source.names,
                null_values=scan.null_values,
                ignore_errors=scan.ignore_errors,
                truncate_ragged_lines=scan.truncate_ragged_lines,
                encoding=scan.encoding,
                try_parse_dates=scan.try_parse_dates,
                predicate=Optional(node.exprs[0].copy()),
            )
    if node.kind == CONCAT:
        return plan_concat(self, index, empty, streaming, batch_size)
    if (
        node.kind == HCONCAT
        or node.kind == GATHER
        or node.kind == INSERT_COLUMN
    ):
        var both = plan_both_inputs(self, index, empty, streaming, batch_size)
        var left = both[0].copy()
        var right = both[1].copy()
        if empty:
            # Only the types matter here; a scalar input still has a
            # row, which would not line up with the other's none.
            left = left.clear()
            right = right.clear()
        if node.kind == INSERT_COLUMN:
            return left.insert_column(node.insert_position(), right._columns[0])
        if node.kind == HCONCAT:
            return concat([left^, right^], "horizontal")
        return _gathered(left, right, node.gather_null_on_oob())
    if (
        node.kind == JOIN
        and not _is_scan(self._nodes[node.left].kind)
        and not _is_scan(self._nodes[node.right].kind)
    ):
        # Both inputs computed (not scans): at once, as a union's.
        var both = plan_both_inputs(self, index, empty, streaming, batch_size)
        return plan_join_frames(
            self, index, both[0].copy(), both[1].copy(), empty
        )
    var input = self._execute(node.left, empty, streaming, batch_size)
    # "Empty" inputs can still have rows: a scalar aggregate over no
    # rows gives one.
    if empty and input.height() == 0:
        var schema = _schema_of(node, input)
        if schema:
            return schema.take()
    if node.kind == FILTER:
        return input.filter(node.exprs[0])
    if node.kind == SELECT:
        return input.select_exprs(node.exprs)
    if node.kind == WITH_COLUMNS:
        return input.with_columns(node.exprs)
    if node.kind == AGG:
        return input.group_by(
            node.group_keys(), maintain_order=node.maintain_order
        ).agg(node.exprs)
    if node.kind == ASOF:
        var right = self._execute(node.right, empty, streaming, batch_size)
        var groups = List[String]()
        for i in range(node.asof_group_count()):
            groups.append(node.asof_group(i))
        return input.join_asof(
            right,
            left_on=node.join_left_keys()[0],
            right_on=node.join_right_keys()[0],
            by=groups,
            strategy=node.text,
            tolerance=node.asof_tolerance,
            suffix=node.join_suffix(),
            allow_exact_matches=node.asof_allow_exact_matches(),
        )
    if node.kind == JOIN:
        var right = self._execute(node.right, empty, streaming, batch_size)
        return plan_join_frames(self, index, input^, right^, empty)
    if node.kind == SORT:
        var n = len(node.sort_keys())
        var descending = List[Bool]()
        var nulls_last = List[Bool]()
        for i in range(n):
            descending.append(node.sort_descending(i))
            nulls_last.append(node.sort_nulls_last(i))
        if node.sort_limit() >= 0:
            return input.take(
                input._arg_sort_head(
                    node.sort_keys(), descending, nulls_last, node.sort_limit()
                )
            )
        return input.sort(
            node.sort_keys(), descending=descending, nulls_last=nulls_last
        )
    if node.kind == SLICE:
        return input.slice(node.slice_offset(), node.slice_length())
    if node.kind == PIVOT:
        return _pivoted(input, node)
    if node.kind == UNIQUE:
        return input.unique(
            node.names, keep=node.text, maintain_order=node.maintain_order
        )
    if node.kind == EXPLODE:
        return input.explode(node.names)
    if node.kind == UNNEST:
        return input.unnest(node.text)
    if node.kind == UNPIVOT:
        return input.unpivot(
            node.unpivot_columns(),
            node.unpivot_index(),
            variable_name=node.unpivot_output_names()[0],
            value_name=node.unpivot_output_names()[1],
        )
    if _is_time_window(node.kind):
        return _run_time_window(node, input)
    if node.kind == GROUPING_SETS:
        return _run_grouping_sets(node, input)
    if node.kind == TAIL:
        return input.tail(node.tail_count())
    if node.kind == REVERSE:
        return input.reverse()
    if node.kind == SHIFT:
        if len(node.exprs) > 0:
            return input.shift(
                node.shift_count(), fill_value=node.exprs[0].copy()
            )
        return input.shift(node.shift_count())
    if node.kind == GATHER_EVERY:
        return input.gather_every(node.gather_stride(), node.gather_offset())
    if node.kind == ROW_INDEX:
        return input.with_row_index(
            node.row_index_name(), Int64(node.row_index_offset())
        )
    if node.kind == FILL_NULL_STRATEGY:
        var limit: Optional[Int] = None
        if node.fill_limit() >= 0:
            limit = node.fill_limit()
        return input.fill_null(
            strategy=node.text, limit=limit, subset=node.names
        )
    return input.drop(node.names)


# --- optimization --------------------------------------------------


def plan_both_inputs(
    self: QueryContext,
    index: Int,
    empty: Bool,
    streaming: Bool,
    batch_size: Int,
) raises -> Tuple[DataFrame, DataFrame]:
    """A two-input node's inputs, executed at once as jobs on their own
    copies of the plan outside the empty-schema run and when no
    execution report is kept (both share the crew's thread budget),
    else one after the other."""
    ref node = self._nodes[index]
    if not empty and not self._report:
        var jobs = List[_SubplanJob](capacity=2)
        jobs.append(_SubplanJob(self.copy(), node.left, streaming, batch_size))
        jobs.append(_SubplanJob(self.copy(), node.right, streaming, batch_size))
        run_jobs(jobs)
        return (jobs[0].result.copy(), jobs[1].result.copy())
    var left = self._execute(node.left, empty, streaming, batch_size)
    var right = self._execute(node.right, empty, streaming, batch_size)
    return (left^, right^)


def plan_concat(
    self: QueryContext,
    index: Int,
    empty: Bool,
    streaming: Bool,
    batch_size: Int,
) raises -> DataFrame:
    """A union's two inputs, each executed, then stacked. The inputs are
    independent, so outside the empty-schema run, and when no
    execution report is kept, each runs as a job on its own copy of
    the plan: a worker runs one while the caller runs the other, and
    both share the crew's thread budget. A worker the budget cannot
    cover leaves both on
    the caller, one after the other."""
    var both = plan_both_inputs(self, index, empty, streaming, batch_size)
    var input = both[0].copy()
    var right = both[1].copy()
    var names = input.columns()
    var other = right.columns()
    if len(other) != len(names):
        raise Error(
            "concat: inputs have "
            + String(len(names))
            + " and "
            + String(len(other))
            + " columns"
        )
    for name in names:
        if name not in other:
            raise Error("concat: right input has no column " + name)
    # The right input's columns in the left's order; `concat`
    # checks the types.
    var parts = List[DataFrame]()
    parts.append(input^)
    parts.append(right.select(names))
    return concat(parts)
