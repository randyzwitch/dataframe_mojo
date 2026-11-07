"""Eager grouping requests, strategy dispatch, and worker jobs."""
from dataframe.aggregation.reducer import (
    Reducer,
)
from dataframe.aggregation.hash_table import (
    AggregateLayout,
    AggregateSink,
    _Cursor,
    aggregate_finish,
    aggregate_layout,
    ReducerSpec,
    key_supported,
    reduction_supported,
    states_supported,
)
from dataframe.aggregation.table_sink import (
    TableSinkInputs,
    TableSinkLayout,
    aggregate_table,
    sink_columns,
)
from dataframe.aggregation.expressions import (
    count_rows_by,
    _bind_all,
    _stream_reductions,
)
from dataframe.aggregation.keys import (
    _key_prefix,
    _pack_struct_keys,
)
from dataframe.aggregation.state import (
    _ReductionPlan,
    _StreamReduction,
    _merge_ascending_runs,
    _reduction_batch,
    table_frame,
    table_inputs,
    table_specs,
)
from dataframe.binding import (
    AGGREGATE,
    BoundExpr,
    bind,
)
from dataframe.kernels.canonical_labels import (
    canonical_labels,
)
from dataframe.categorical import (
    decode,
)
from dataframe.storage.column import (
    Column,
)
from dataframe.execution import (
    _ColumnStats,
    _new_reducer,
    evaluate,
    reduce_rows,
)
from dataframe.expr import (
    COL,
    Expr,
    N_UNIQUE,
    SELECTOR,
    group_sequence_rows,
    col,
    is_reduction,
)
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.kernels.gather import (
    can_filter_aligned_chunks,
    take_parallel,
    take_sorted_chunked,
)
from dataframe.kernels.hashing import (
    RowKeys,
    encode_rows,
    encode_rows_parallel,
    encode_string_rows_parallel,
)
from dataframe.aggregation.selected_rows import (
    indexed_reductions,
    reduce_indexed,
    reduce_indexed_batches,
)
from dataframe.lazy import (
    LazyGroupingSets,
)
from dataframe.parallel import (
    Job,
    configured_workers,
    partitions,
    run_jobs,
    worker_count,
)
from dataframe.kernels.partition import (
    Partitioner,
    encode_bucket,
    low_cardinality,
    small_key_product,
)
from dataframe.aggregation.dense import (
    PerfectLayout,
    perfect_layout,
    perfect_slots,
)
from dataframe.series import (
    Series,
    sort_indices,
)
from dataframe.storage.string import (
    StringColumn,
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


struct _EncodeJob(Job):
    """Encode one row range of a heavy bucket with a private dictionary."""

    var keys: List[Series]
    var ids: List[Int]
    var representatives: List[Int]

    def __init__(out self, var keys: List[Series]):
        self.keys = keys^
        self.ids = List[Int]()
        self.representatives = List[Int]()

    def run(mut self) raises:
        var local = encode_rows(self.keys, nulls_equal=True)
        self.ids = local.ids.copy()
        self.representatives = local.representatives.copy()


def _encode_parallel(keys: List[Series], workers: Int) raises -> RowKeys:
    """Dense ids in first-occurrence order, encoded per row range and then
    reconciled through the representative rows.

    Only worth it when distinct keys are few relative to rows, which is
    what makes a bucket heavy: the reconciliation re-encodes one row per
    (worker, distinct key), so its cost is workers x distinct keys.
    """
    var n = len(keys[0])
    var bounds = partitions(n, workers, 64)
    var jobs = List[_EncodeJob](capacity=workers)
    for w in range(workers):
        var slices = List[Series](capacity=len(keys))
        for key in keys:
            slices.append(key.slice(bounds[w], bounds[w + 1] - bounds[w]))
        jobs.append(_EncodeJob(slices^))
    run_jobs(jobs)

    # One row per (worker, local id), in worker order, so encoding those
    # rows yields each local id's global id at a known position.
    var rows = List[Int]()
    var offsets = List[Int](capacity=workers + 1)
    for w in range(workers):
        offsets.append(len(rows))
        for r in jobs[w].representatives:
            rows.append(bounds[w] + r)
    offsets.append(len(rows))
    var samples = List[Series](capacity=len(keys))
    for key in keys:
        samples.append(key.take(rows))
    var merged = encode_rows(samples, nulls_equal=True)

    var ids = List[Int](length=n, fill=0)
    for w in range(workers):
        var base = offsets[w]
        var start = bounds[w]
        for i in range(len(jobs[w].ids)):
            ids[start + i] = merged.ids[base + jobs[w].ids[i]]
    var representatives = List[Int](capacity=merged.count())
    for r in merged.representatives:
        representatives.append(rows[r])
    return RowKeys(ids^, representatives^)


# Rows a perfect hash aggregate gives slots at a time. Their slots (8
# bytes each) take 128 KB, within a core's L2, so every reduction reads
# them back from cache, as DuckDB updates each aggregate from one vector's
# group addresses before it takes the next vector.
comptime _PERFECT_WINDOW = 16384


struct _PerfectAggJob(Job):
    """Reduce rows [start, end) into the slots of a perfect hash table
    (`perfect_aggregate.mojo`): each window of rows gets its slots, then
    every reduction folds the window in one vector at a time. `firsts` keeps
    each slot's first row, -1 while it is empty."""

    var frame: DataFrame
    var keys: List[Series]
    var layout: PerfectLayout
    var plan: _ReductionPlan
    var start: Int
    var end: Int
    var states: List[Reducer]
    var firsts: List[Int]

    def __init__(
        out self,
        frame: DataFrame,
        keys: List[Series],
        layout: PerfectLayout,
        plan: _ReductionPlan,
        start: Int,
        end: Int,
    ):
        self.frame = frame.copy()
        self.keys = keys.copy()
        self.layout = layout.copy()
        self.plan = plan.copy()
        self.start = start
        self.end = end
        self.states = List[Reducer](capacity=len(plan.bounds))
        for r in range(len(plan.bounds)):
            ref bound = plan.bounds[r]
            self.states.append(
                _new_reducer(
                    bound, bound.expr._nodes[plan.nodes[r]], layout.slots()
                )
            )
        self.firsts = List[Int](length=layout.slots(), fill=-1)

    def run(mut self) raises:
        var slots = List[Int](length=_PERFECT_WINDOW, fill=0)
        # Only the columns the reductions read are cut to each window.
        var read = List[Int]()
        for bound in self.plan.bounds:
            for source in bound.sources:
                if source >= 0 and source not in read:
                    read.append(source)
        var columns = self.frame._columns.copy()
        var first = self.firsts.unsafe_ptr()
        var at = self.start
        while at < self.end:
            var length = min(_PERFECT_WINDOW, self.end - at)
            perfect_slots(self.keys, self.layout, at, length, slots)
            var slot = slots.unsafe_ptr()
            for i in range(length):
                ref seen = first[unsafe_offset=slot[unsafe_offset=i]]
                if seen < 0:
                    seen = at + i
            for c in read:
                columns[c] = self.frame._columns[c].slice(at, length)
            for r in range(len(self.states)):
                ref bound = self.plan.bounds[r]
                reduce_rows[8](
                    self.states[r],
                    bound,
                    columns,
                    List[Series](),
                    bound.expr._nodes[self.plan.nodes[r]],
                    0,
                    length,
                    _reduction_batch(True),
                    True,
                    slots,
                    ArcPointer(List[_ColumnStats]()),
                )
            at += length


def _plain_inputs(expressions: List[Expr], frame: DataFrame) -> Bool:
    """Whether every reduction reads a column of `frame` as it is (or no
    input), so nothing is evaluated before its states update."""
    for expression in expressions:
        ref nodes = expression._nodes
        for node in nodes:
            if not is_reduction(node.op):
                continue
            for child in [node.left, node.right]:
                if child < 0:
                    continue
                if nodes[child].op != COL or (
                    nodes[child].text not in frame.columns()
                ):
                    return False
    return True


struct _BatchSinkJob(Job):
    """One worker's morsels into its aggregate table sink, each morsel's
    reduction inputs evaluated `batch` rows at a time."""

    var frame: DataFrame
    var keys: List[Series]
    var expressions: List[Expr]
    var layout: AggregateLayout
    var shared: TableSinkLayout
    var ranges: List[Int]
    var cursor: _Cursor
    var workers: Int
    var batch: Int
    var sinks: List[AggregateSink]

    def __init__(
        out self,
        frame: DataFrame,
        keys: List[Series],
        expressions: List[Expr],
        layout: AggregateLayout,
        empty_keys: List[Series],
        empty: List[Series],
        ranges: List[Int],
        cursor: _Cursor,
        workers: Int,
        batch: Int,
    ):
        self.frame = frame.copy()
        self.keys = keys.copy()
        self.expressions = expressions.copy()
        self.layout = layout.copy()
        self.shared = TableSinkLayout(layout, empty_keys, empty)
        self.ranges = ranges.copy()
        self.cursor = cursor.copy()
        self.workers = workers
        self.batch = max(1, batch)
        self.sinks = List[AggregateSink]()

    def run(mut self) raises:
        var sink = AggregateSink(self.layout, self.workers)
        var reader = TableSinkInputs(self.shared)
        var morsels = len(self.ranges) // 2
        while True:
            var m = self.cursor.take()
            if m >= morsels:
                break
            var offset = self.ranges[2 * m]
            var length = self.ranges[2 * m + 1]
            for lo in range(0, length, self.batch):
                var start = offset + lo
                var n = min(self.batch, length - lo)
                var keys = List[Series](capacity=len(self.keys))
                for key in self.keys:
                    keys.append(key.slice(start, n))
                var inputs = table_inputs(
                    self.frame.slice(start, n), keys, self.expressions
                )
                sink_columns(sink, reader, keys, inputs, start)
        self.sinks.append(sink^)


@fieldwise_init
struct GroupingSets(Copyable):
    """A pending `group_by_rollup`, `group_by_cube` or
    `group_by_grouping_sets`; finish it with agg."""

    var _plan: LazyGroupingSets

    def agg(self, exprs: List[Expr]) raises -> DataFrame:
        """The rows of every grouping set, in set order: the keys (null in
        the sets that do not group by them), then the aggregations. Use
        `grouping` / `grouping_id` in `exprs` to tell subtotal rows from
        null key values. See LazyGroupingSets.agg."""
        return self._plan.agg(exprs).collect()

    def agg(self, expr: Expr) raises -> DataFrame:
        return self.agg([expr.copy()])


@fieldwise_init
struct GroupBy(Copyable):
    """An eager grouping request over a snapshot and evaluated key columns."""

    var _frame: DataFrame
    var _keys: List[Series]
    var _maintain_order: Bool

    def head(self, n: Int = 5) raises -> DataFrame:
        """First n rows of each group, retaining every original column."""
        return self._edge_rows(n, False)

    def tail(self, n: Int = 5) raises -> DataFrame:
        """Last n rows of each group, retaining every original column."""
        return self._edge_rows(n, True)

    def _edge_rows(self, n: Int, tail: Bool) raises -> DataFrame:
        if n < 0:
            raise Error("GroupBy.head/tail count must be nonnegative")
        var encoded = encode_rows(self._keys, nulls_equal=True)
        var rows = List[List[Int]](length=encoded.count(), fill=List[Int]())
        for row in range(self._frame.height()):
            rows[encoded.ids[row]].append(row)
        var selected = List[Int]()
        for group in rows:
            var start = max(0, len(group) - n) if tail else 0
            var end = len(group) if tail else min(len(group), n)
            for k in range(start, end):
                selected.append(group[k])
        return self._frame.take(selected)

    def _coded_keys(self) raises -> Optional[GroupBy]:
        """This grouping with each String key that carries dictionary codes
        replaced by those codes as a categorical; None when no key does."""
        var keys = List[Series](capacity=len(self._keys))
        var any = False
        for key in self._keys:
            var codes = key._dictionary_codes()
            if codes:
                keys.append(canonical_labels(codes.take()))
                any = True
            else:
                keys.append(key.copy())
        if not any:
            return None
        return GroupBy(self._frame.copy(), keys^, self._maintain_order)

    def _key_names(self) -> Dict[String, Bool]:
        var names = Dict[String, Bool]()
        for key in self._keys:
            var prefix = _key_prefix(key.name())
            names[prefix if prefix != "" else key.name()] = True
        return names^

    def agg(
        self, expression: Expr, *, batch_size: Int = 1024
    ) raises -> DataFrame:
        return self.agg([expression.copy()], batch_size=batch_size)

    def agg(
        self, expressions: List[Expr], *, batch_size: Int = 1024
    ) raises -> DataFrame:
        var prepared = List[Expr](capacity=len(expressions))
        var changed = False
        var counted = len(self._keys) > 0 and (
            self._keys[0].name() in self._frame.columns()
        )
        for expression in expressions:
            if group_sequence_rows(expression):
                prepared.append(expression.implode())
                changed = True
            elif counted:
                var rows = count_rows_by(expression, self._keys[0].name())
                changed = changed or len(rows._nodes) != len(expression._nodes)
                prepared.append(rows^)
            else:
                prepared.append(expression.copy())
        if changed:
            return self.agg(prepared, batch_size=batch_size)
        # String keys that carry their source's dictionary codes (a Parquet
        # scan) group by the codes, as categorical keys do, and come back as
        # strings: H2O q2's two string keys 102 -> 49 ms.
        var coded = self._coded_keys()
        if coded:
            var result = coded.value().agg(expressions, batch_size=batch_size)
            var columns = List[Series](capacity=result.width())
            for column in result._columns:
                if column.dtype().is_categorical() and column.name() in (
                    self._key_names()
                ):
                    columns.append(decode(column))
                else:
                    columns.append(column.copy())
            return DataFrame(columns^)
        var bound = _bind_all(expressions, self._frame._columns)
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        var key_names = self._key_names()
        for expression in bound:
            if expression.shape() != AGGREGATE:
                raise Error(
                    "Group aggregation requires scalar aggregate expressions"
                )
            if expression.expr._name in key_names:
                raise Error(
                    "Aggregate output name collides with grouping key: "
                    + expression.expr._name
                )
        # Hashing and encoding a key cost 50 to 80 ns a row, not the
        # nanosecond of the scans `worker_count`'s floor is sized for, so
        # the routes below divide at the partition floor: 75K rows of
        # ClickBench q40 (35K groups) encoded whole on one thread took
        # 3.5 ms while seven threads idled.
        var height = self._frame.height()
        var workers = group_by_workers(height)
        if _stream_reductions(expressions) and not self._counts_distinct(
            expressions
        ):
            var perfect = self._agg_perfect(expressions, key_names, workers)
            if perfect:
                return perfect.take()
        # Keys of a small dense domain went to the perfect hash above, as
        # DuckDB's do; any other grouping goes through the aggregate table
        # whatever its number of groups, whose per-thread tables adapt to
        # however many there are. Neither DuckDB nor Polars picks a route
        # by an estimate of the groups. An expression the table has no
        # state for is evaluated over the groups, as Polars evaluates an
        # expression over its group tuples.
        if self._table_supported(expressions):
            var table = self._agg_table(expressions, workers, batch_size)
            if table:
                return _pack_struct_keys(table.value())
        return _pack_struct_keys(self._agg_whole(bound, batch_size))

    def _table_supported(self, expressions: List[Expr]) raises -> Bool:
        """Whether the aggregate table (`aggregate_table.mojo`) groups these
        keys and reduces these expressions."""
        if len(self._keys) == 0 or not _stream_reductions(expressions, True):
            return False
        for expression in expressions:
            for node in expression._nodes:
                if is_reduction(node.op) and not reduction_supported(node.op):
                    return False
        for key in self._keys:
            if not key_supported(key):
                return False
        return True

    def _agg_table(
        self, expressions: List[Expr], workers: Int, batch_size: Int
    ) raises -> Optional[DataFrame]:
        """Grouped reductions through the aggregate table: each worker
        hashes its morsels' keys where they are and probes its own table,
        updating its states by group; partitions of the groups then merge
        across workers in parallel (`aggregate_table.mojo`). None when some reduction's
        state does not fit inline in a group row."""
        # A float key's words are canonical (-0.0 and 0.0, or two NaNs, are
        # one group), so its group shows its first row's value, as DuckDB's
        # rows keep the value they were made from: each float key's first
        # value comes from a FIRST state and replaces the key column.
        var float_keys = List[Int]()
        var all_expressions = expressions.copy()
        for k in range(len(self._keys)):
            var name = self._keys[k].name()
            if (
                self._keys[k].dtype().is_float()
                and name in self._frame.columns()
            ):
                float_keys.append(k)
                all_expressions.append(
                    col(name).first().alias("__key_first_" + String(k))
                )
        if len(float_keys) > 0:
            var grouped = self._agg_canonical_table(
                all_expressions, workers, batch_size
            )
            if not grouped:
                return None
            var frame = grouped.take()
            var hidden = List[String]()
            for k in float_keys:
                var name = "__key_first_" + String(k)
                frame = frame.with_column(
                    frame.column(name).renamed(self._keys[k].name())
                )
                hidden.append(name)
            return frame.drop(hidden)
        return self._agg_canonical_table(expressions, workers, batch_size)

    def _agg_canonical_table(
        self, expressions: List[Expr], workers: Int, batch_size: Int
    ) raises -> Optional[DataFrame]:
        """`_agg_table` with float keys as their canonical words."""
        var empty_keys = List[Series](capacity=len(self._keys))
        for key in self._keys:
            empty_keys.append(key.slice(0, 0))
        var template = _StreamReduction(
            self._frame.clear(), expressions, empty_keys, self._key_names()
        )
        var specs = table_specs(template)
        var empty = table_inputs(self._frame.clear(), empty_keys, expressions)
        if not states_supported(specs, empty):
            return None
        trace_path("group_by.aggregate_table")
        if not _plain_inputs(expressions, self._frame):
            # Computed reduction inputs are evaluated a batch of
            # `batch_size` rows at a time, each batch then updating the
            # states (DuckDB's sink evaluates a DataChunk's inputs, then
            # UpdateStates): an input error past the batch comes after the
            # batch's update errors, as the batch size says.
            return self._agg_batches(
                expressions, specs, empty, template, workers, batch_size
            )
        var inputs = table_inputs(self._frame, self._keys, expressions)
        # Morsels within chunks: the columns' common chunks, or one
        # contiguous copy of each chunked column when they differ.
        var keys = self._keys.copy()
        var all_columns = keys.copy()
        for item in inputs:
            all_columns.append(item.copy())
        var ends = List[Int]()
        if can_filter_aligned_chunks(all_columns):
            ends = all_columns[0]._chunked.value()[].ends.copy()
        else:
            for k in range(len(keys)):
                if keys[k].is_chunked():
                    keys[k] = keys[k].rechunk()
            for k in range(len(inputs)):
                if inputs[k].is_chunked():
                    inputs[k] = inputs[k].rechunk()
            ends.append(self._frame.height())
        var ranges = List[Int]()
        var start = 0
        var morsel = _MORSEL_ROWS
        for end in ends:
            var at = start
            while at < end:
                var length = min(morsel, end - at)
                ranges.append(at)
                ranges.append(length)
                at += length
            start = end
        var result = aggregate_table(
            keys, inputs, specs, ranges, workers, self._maintain_order
        )
        return table_frame(template, result^, self._maintain_order)

    def _agg_batches(
        self,
        expressions: List[Expr],
        specs: List[ReducerSpec],
        empty: List[Series],
        template: _StreamReduction,
        workers: Int,
        batch_size: Int,
    ) raises -> Optional[DataFrame]:
        """The aggregate table over morsels whose reduction inputs are
        evaluated `batch_size` rows at a time on the workers."""
        var frame = self._frame.rechunk()
        var keys = List[Series](capacity=len(self._keys))
        for key in self._keys:
            keys.append(key.rechunk() if key.is_chunked() else key.copy())
        var empty_keys = List[Series](capacity=len(keys))
        for key in keys:
            empty_keys.append(key.slice(0, 0))
        var layout = aggregate_layout(empty_keys, empty, specs)
        var ranges = List[Int]()
        var height = frame.height()
        var morsel = _MORSEL_ROWS
        var at = 0
        while at < height:
            ranges.append(at)
            ranges.append(min(morsel, height - at))
            at += morsel
        var cursor = _Cursor.new()
        var jobs = List[_BatchSinkJob](capacity=workers)
        for _ in range(max(1, workers)):
            jobs.append(
                _BatchSinkJob(
                    frame,
                    keys,
                    expressions,
                    layout,
                    empty_keys,
                    empty,
                    ranges,
                    cursor,
                    workers,
                    batch_size,
                )
            )
        try:
            run_jobs(jobs)
        except e:
            cursor.free()
            raise e^
        cursor.free()
        var sinks = List[AggregateSink](capacity=len(jobs))
        for j in range(len(jobs)):
            while len(jobs[j].sinks) > 0:
                sinks.append(jobs[j].sinks.pop())
        var result = aggregate_finish(
            sinks, layout, ordered=self._maintain_order
        )
        return table_frame(template, result^, self._maintain_order)

    def _counts_distinct(self, expressions: List[Expr]) -> Bool:
        """Whether some expression counts distinct values: the aggregate
        table counts them in its distinct tables (`AggregateSink.distinct`),
        which the per-worker and per-range states do not have."""
        for expression in expressions:
            for node in expression._nodes:
                if node.op == N_UNIQUE:
                    return True
        return False

    def _agg_perfect(
        self,
        expressions: List[Expr],
        key_names: Dict[String, Bool],
        workers: Int,
    ) raises -> Optional[DataFrame]:
        """Integer keys spanning few enough values (`perfect_layout`) give each
        row its group's slot directly, with no table to probe and no group
        ids to number or store; each worker reduces its range into its own
        slots, the workers' slots merge in row order, and the occupied
        slots come out in first-occurrence order. None when the keys do
        not qualify."""
        var height = self._frame.height()
        if height == 0:
            return None
        var layout = perfect_layout(self._keys)
        if not layout:
            return None
        trace_path("group_by.perfect_hash")
        var plan = _ReductionPlan(self._frame, expressions, key_names)
        var bounds = partitions(height, workers, 64)
        var jobs = List[_PerfectAggJob](capacity=workers)
        for w in range(workers):
            if bounds[w] < bounds[w + 1]:
                jobs.append(
                    _PerfectAggJob(
                        self._frame,
                        self._keys,
                        layout.value(),
                        plan,
                        bounds[w],
                        bounds[w + 1],
                    )
                )
        try:
            run_jobs(jobs)
        except:
            # Windows here span several feed batches, so the first error
            # raised need not be the one the batch order raises first.
            # A failing aggregation is left to the other routes, which
            # feed batch by batch and raise the error in that order.
            return None
        # DuckDB's `Combine`: every worker's slots fold into the first's,
        # in row order, so first and last keep a single pass's answer.
        var merged = jobs.pop(0)
        for w in range(len(jobs)):
            ref job = jobs[w]
            for r in range(len(merged.states)):
                merged.states[r].merge(job.states[r])
            for s in range(len(merged.firsts)):
                if merged.firsts[s] < 0:
                    merged.firsts[s] = job.firsts[s]
        # DuckDB's `Scan` takes the occupied slots in slot order; these keep
        # the first-occurrence order of the other routes.
        var occupied = List[Int]()
        var rows = List[Int]()
        for s in range(len(merged.firsts)):
            if merged.firsts[s] >= 0:
                occupied.append(s)
                rows.append(merged.firsts[s])
        var order = sort_indices([rows.copy()])
        var picked = List[Int](capacity=len(order))
        var firsts = List[Int](capacity=len(order))
        for o in order:
            picked.append(occupied[o])
            firsts.append(rows[o])
        var count = len(picked)
        var states = List[Reducer](capacity=len(merged.states))
        for r in range(len(merged.states)):
            ref source = merged.states[r]
            var state = Reducer(
                source.op,
                source.input,
                count,
                source.min_count,
                source.integer,
                source.floating,
                source.text,
                source.logical,
            )
            state.merge(source, sources=picked)
            states.append(state^)
        var keys = List[Series](capacity=len(self._keys))
        for key in self._keys:
            keys.append(key.take(firsts))
        var reduction = _StreamReduction(
            keys=DataFrame(keys^, height=count),
            states=states^,
            names=plan.names,
            dtypes=plan.dtypes,
            outputs=plan.outputs,
            grouped=True,
            firsts=firsts^,
            rows=height,
        )
        return reduction.finish()

    def _agg_whole(
        self, bound: List[BoundExpr], batch_size: Int
    ) raises -> DataFrame:
        """Serial key encoding, then the parallel per-group reduce."""
        trace_path("group_by.whole")
        var groups: RowKeys
        if len(self._keys) == 1 and self._keys[0]._data.isa[StringColumn]():
            groups = encode_string_rows_parallel(
                self._keys[0], True, worker_count(self._frame.height())
            )
        else:
            groups = encode_rows_parallel(
                self._keys, True, worker_count(self._frame.height())
            )
        # First-occurrence representatives are ordered by source row.
        var columns = take_sorted_chunked(
            self._keys, groups.representatives.copy(), 1
        )
        for expression in bound:
            columns.append(
                evaluate(
                    expression,
                    self._frame._columns,
                    self._frame.height(),
                    batch_size=batch_size,
                    grouped=True,
                    groups=groups.ids,
                    group_count=groups.count(),
                )
            )
        return DataFrame(columns^, height=groups.count())

    def len(self, name: String = "len") raises -> DataFrame:
        """Row count per group, including rows with null values."""
        if name in self._key_names():
            raise Error(
                "Aggregate output name collides with grouping key: " + name
            )
        var groups = encode_rows(self._keys, nulls_equal=True)
        var counts = List[Int64](length=groups.count(), fill=0)
        for id in groups.ids:
            counts[id] += 1
        var columns = take_sorted_chunked(
            self._keys, groups.representatives.copy(), 1
        )
        columns.append(Series(name, Column[Int64](counts^)))
        return _pack_struct_keys(DataFrame(columns^, height=groups.count()))

    def _each(
        self,
        reduction: String,
        quantile: Float64 = 0.5,
        interpolation: String = "nearest",
    ) raises -> DataFrame:
        """Apply `reduction` to every non-key column."""
        var keys = self._key_names()
        var exprs = List[Expr]()
        for column in self._frame._columns:
            if column.name() not in keys:
                exprs.append(
                    _reduce_column(
                        column.name(), reduction, quantile, interpolation
                    )
                )
        return self.agg(exprs)

    def sum(self) raises -> DataFrame:
        """Each other column's sum per group."""
        return self._each("sum")

    def mean(self) raises -> DataFrame:
        """Each other column's mean per group, as Float64."""
        return self._each("mean")

    def min(self) raises -> DataFrame:
        """Each other column's minimum per group."""
        return self._each("min")

    def max(self) raises -> DataFrame:
        """Each other column's maximum per group."""
        return self._each("max")

    def median(self) raises -> DataFrame:
        """Each other column's median per group, as Float64."""
        return self._each("median")

    def first(self) raises -> DataFrame:
        """Each other column's first value per group (null when that row's
        value is null)."""
        return self._each("first")

    def last(self) raises -> DataFrame:
        """Each other column's last value per group (null when that row's
        value is null)."""
        return self._each("last")

    def n_unique(self) raises -> DataFrame:
        """Each other column's distinct values per group, null counted once,
        as Int64."""
        return self._each("n_unique")

    def quantile(
        self, quantile: Float64, interpolation: String = "nearest"
    ) raises -> DataFrame:
        """Each other column's quantile per group, as Float64; interpolation
        defaults to "nearest"."""
        return self._each("quantile", quantile, interpolation)


def _reduce_column(
    name: String, reduction: String, quantile: Float64, interpolation: String
) raises -> Expr:
    """`col(name)` reduced by a GroupBy convenience method's reduction."""
    var c = col(name)
    if reduction == "sum":
        return c.sum()
    if reduction == "mean":
        return c.mean()
    if reduction == "min":
        return c.min()
    if reduction == "max":
        return c.max()
    if reduction == "median":
        return c.median()
    if reduction == "first":
        return c.first()
    if reduction == "last":
        return c.last()
    if reduction == "n_unique":
        return c.n_unique()
    if reduction == "quantile":
        return c.quantile(quantile, interpolation)
    raise Error("Unknown group reduction: " + reduction)


# Rows a worker takes at a time: DuckDB's DEFAULT_ROW_GROUP_SIZE, the unit
# its table scans hand to threads.
comptime _MORSEL_ROWS = 122_880


def group_by_workers(height: Int) -> Int:
    """Workers the eager group-by divides `height` rows among: one per
    morsel of `_MORSEL_ROWS`, at most the configured threads (DuckDB runs at
    most one thread per row group)."""
    return max(1, min(configured_workers(), -(-height // _MORSEL_ROWS)))


def range_states(keys: List[Series]) raises -> Bool:
    """Whether the keys have few values (`low_cardinality`) or form few
    groups (`small_key_product`), by their sampled tests; read by the
    planner's batch-states decision (`plan_batch_states_win`)."""
    return low_cardinality(keys) or small_key_product(keys)
