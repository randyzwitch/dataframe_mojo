"""Eager grouping requests, strategy dispatch, and worker jobs."""
from dataframe.aggregation.reducer import (
    Reducer,
)
from dataframe.aggregation.clustered import (
    ClusteredRuns,
    update_clustered,
    update_clustered_pair,
)
from dataframe.aggregation.function import (
    AggregateFunction,
    _BATCH,
    _K_COMOMENT,
    _Groups,
    _HeldBuffers,
    _ValueArena,
    _column_words,
    _update_states,
    combine,
)
from std.ffi import external_call
from dataframe.aggregation.hash_table import (
    _width,
    groups_result,
    AggregateLayout,
    AggregateSink,
    _Cursor,
    _one_chunk,
    aggregate_finish,
    aggregate_layout,
    aggregate_table,
    ReducerSpec,
    key_supported,
    reduction_supported,
    states_supported,
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
    bitcast,
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


struct _PerfectSinkJob(Job):
    """One worker's rows [start, end) into its own perfect hash table
    (DuckDB's `PerfectAggregateHashTable`): a group row for every slot the
    keys' bits can name, its states the aggregate table's
    (`AggregateFunction`s), updated a vector of rows at a time through the
    rows' slots, with no probe. A slot's first word is 1 once a row has
    reached it. Computed reduction inputs are evaluated `batch` rows at a
    time, before those rows update the states."""

    var frame: DataFrame
    var keys: List[Series]
    var expressions: List[Expr]
    var perfect: PerfectLayout
    var layout: AggregateLayout
    var start: Int
    var end: Int
    var batch: Int
    var groups: List[_Groups]
    # The arena of the job's quantile values, alive until the finish.
    var values: List[_ValueArena]

    def __init__(
        out self,
        frame: DataFrame,
        keys: List[Series],
        expressions: List[Expr],
        perfect: PerfectLayout,
        layout: AggregateLayout,
        start: Int,
        end: Int,
        batch: Int,
    ):
        self.frame = frame.copy()
        self.keys = keys.copy()
        self.expressions = expressions.copy()
        self.perfect = perfect.copy()
        self.layout = layout.copy()
        self.start = start
        self.end = end
        self.batch = max(1, batch)
        self.groups = List[_Groups]()
        self.values = List[_ValueArena]()

    def run(mut self) raises:
        var slots_n = self.perfect.slots()
        var groups = _Groups(
            len(self.layout.addressed),
            self.layout.addressed.copy(),
            self.layout.state_words,
            slots_n,
        )
        var stride = groups.stride
        var base = Int(groups.rows.unsafe_ptr())
        _ = external_call["memset", Int](base, 0, 8 * stride * slots_n)
        groups.n = slots_n
        self.values.append(_ValueArena())
        var arena = Pointer[_ValueArena, MutAnyOrigin](
            unsafe_from_address=Int(Pointer(to=self.values[0]))
        )
        var slots = List[Int](length=_BATCH, fill=0)
        var rows = List[Int](length=_BATCH, fill=0)
        var scratch = List[UInt64](length=3 * _BATCH, fill=0)
        var discard = List[UInt64](length=stride + 8, fill=0)
        var clustering = ClusteredRuns()
        var at = self.start
        while at < self.end:
            var length = min(self.batch, self.end - at)
            var key_part = List[Series](capacity=len(self.keys))
            for key in self.keys:
                key_part.append(_one_chunk(key.slice(at, length)))
            var inputs = table_inputs(
                self.frame.slice(at, length), key_part, self.expressions
            )
            for i in range(len(inputs)):
                inputs[i] = _one_chunk(inputs[i])
            for lo in range(0, length, _BATCH):
                var count = min(_BATCH, length - lo)
                perfect_slots(self.keys, self.perfect, at + lo, count, slots)
                var slot_at = slots.unsafe_ptr()
                var row_at = rows.unsafe_ptr()
                if clustering.try_build(slot_at, count):
                    self._update_clustered(
                        clustering,
                        inputs,
                        lo,
                        count,
                        at,
                        base,
                        stride,
                        groups.states_at,
                        slots,
                        rows,
                        arena,
                        Int(discard.unsafe_ptr()),
                        Int(scratch.unsafe_ptr()),
                    )
                    continue
                for k in range(count):
                    var address = base + 8 * stride * slot_at[unsafe_offset=k]
                    row_at[unsafe_offset=k] = address
                    var row = Pointer[UInt64, MutAnyOrigin](
                        unsafe_from_address=address
                    )
                    if row[] != 0:
                        continue
                    # The slot's first row; its keys are the slot's own bits,
                    # written when the slots are read out.
                    row[] = 1
                    row[unsafe_offset=2] = UInt64(at + lo + k)
                _update_states(
                    self.layout.slots,
                    inputs,
                    lo,
                    count,
                    rows.unsafe_ptr(),
                    groups.states_at,
                    at,
                    arena,
                    Int(discard.unsafe_ptr()),
                    Int(scratch.unsafe_ptr()),
                )
            at += length
        self.groups.append(groups^)

    def _update_clustered(
        self,
        runs: ClusteredRuns,
        inputs: List[Series],
        lo: Int,
        count: Int,
        at: Int,
        base: Int,
        stride: Int,
        states_at: Int,
        slots: List[Int],
        mut rows: List[Int],
        arena: Pointer[_ValueArena, MutAnyOrigin],
        discard: Int,
        scratch: Int,
    ) raises:
        """A clustered chunk's rows into their slots' states, a run at a
        time (DuckDB's `PerfectAggregateHashTable::AddChunkClustered`):
        each run marks its slot and keeps its first row, then each
        reduction updates the run's state once; a reduction with no
        clustered update takes the chunk's rows one by one."""
        var row_bytes = 8 * stride
        var positions = runs.arena.unsafe_ptr()
        for r in range(runs.n_runs):
            var row = Pointer[UInt64, MutAnyOrigin](
                unsafe_from_address=base + Int(runs.gids[r]) * row_bytes
            )
            # A run's positions rise, so its first is its earliest row.
            var first = UInt64(
                at + lo + Int(positions[unsafe_offset=runs.starts[r]])
            )
            if row[] == 0:
                row[] = 1
                row[unsafe_offset=2] = first
            elif first < row[unsafe_offset=2]:
                row[unsafe_offset=2] = first
        var addressed = False
        for s in range(len(self.layout.slots)):
            ref slot = self.layout.slots[s]
            var state = 8 * (states_at + slot.offset)
            if slot.kind == _K_COMOMENT:
                update_clustered_pair(
                    inputs[s],
                    inputs[slot.k],
                    lo,
                    count,
                    runs,
                    base,
                    row_bytes,
                    state,
                    scratch,
                )
                continue
            if update_clustered(
                slot, inputs[s], lo, count, runs, base, row_bytes, state
            ):
                continue
            if not addressed:
                for k in range(count):
                    rows[k] = base + row_bytes * slots[k]
                addressed = True
            var one = List[AggregateFunction]()
            one.append(slot.copy())
            var lone = List[Series]()
            lone.append(inputs[s].copy())
            _update_states(
                one,
                lone,
                lo,
                count,
                rows.unsafe_ptr(),
                states_at,
                at,
                arena,
                discard,
                scratch,
            )


def _perfect_keys(
    perfect: PerfectLayout,
    keys: Int,
    slot: Int,
    row: Pointer[UInt64, MutAnyOrigin],
):
    """A perfect hash slot's keys written into its group row as key words
    (DuckDB's perfect hash aggregate rebuilds the group values from the
    slot when it scans): each key's bits less one, plus its minimum, or
    null when its bits are zero."""
    var nulls = UInt64(0)
    var top = perfect.bits
    for k in range(keys):
        var shift = perfect.shifts[k]
        var width = top - shift
        top = shift
        var part = (slot >> shift) & ((1 << width) - 1)
        if part == 0:
            nulls |= UInt64(1) << UInt64(k)
            row[unsafe_offset=3 + k] = 0
            continue
        var value = perfect.minima[k] + Int64(part - 1)
        row[unsafe_offset=3 + k] = bitcast[DType.uint64](value)
    row[unsafe_offset=1] = nulls


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
        ranges: List[Int],
        cursor: _Cursor,
        workers: Int,
        batch: Int,
    ):
        self.frame = frame.copy()
        self.keys = keys.copy()
        self.expressions = expressions.copy()
        self.layout = layout.copy()
        self.ranges = ranges.copy()
        self.cursor = cursor.copy()
        self.workers = workers
        self.batch = max(1, batch)
        self.sinks = List[AggregateSink]()

    def run(mut self) raises:
        var sink = AggregateSink(self.layout, self.workers)
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
                    keys.append(_one_chunk(key.slice(start, n)))
                var inputs = table_inputs(
                    self.frame.slice(start, n), keys, self.expressions
                )
                for i in range(len(inputs)):
                    inputs[i] = _one_chunk(inputs[i])
                sink.sink(keys, inputs, start)
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
        var perfect = self._agg_perfect(expressions, workers, batch_size)
        if perfect:
            return _pack_struct_keys(perfect.take())
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
        var result = aggregate_table(keys, inputs, specs, ranges, workers)
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
        var result = aggregate_finish(sinks, layout)
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
        self, expressions: List[Expr], workers: Int, batch_size: Int
    ) raises -> Optional[DataFrame]:
        """DuckDB's perfect hash aggregate (`PhysicalPerfectHashAggregate`),
        chosen as DuckDB chooses it, from the keys' statistics: integer
        keys whose ranges (the segment statistics' minimum and maximum, or
        a categorical's dictionary size) need at most `PERFECT_HASH_BITS`
        bits together give each row its group's slot directly. Each worker
        updates its own slots' states, the aggregate table's
        `AggregateFunction`s, over its range of rows; the workers' slots
        combine in order, and the occupied slots are read out as the
        table's groups are. None when the keys do not qualify, or a
        reduction counts distinct values (DuckDB's perfect hash aggregate
        has no distinct tables)."""
        var height = self._frame.height()
        if height == 0 or self._counts_distinct(expressions):
            return None
        if not self._table_supported(expressions):
            return None
        var perfect = perfect_layout(self._keys)
        if not perfect:
            return None
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
        trace_path("group_by.perfect_hash")
        var frame = self._frame.rechunk()
        var keys = List[Series](capacity=len(self._keys))
        for key in self._keys:
            keys.append(key.rechunk() if key.is_chunked() else key.copy())
        # Plain columns are read where they are, and their statistics can
        # choose the states (an Int64 sum that cannot overflow); computed
        # inputs are evaluated a batch at a time, as `batch_size` says
        # (see `_agg_batches`).
        var plain = _plain_inputs(expressions, frame)
        var layout = aggregate_layout(
            empty_keys,
            table_inputs(frame, keys, expressions) if plain else empty.copy(),
            specs,
        )
        var batch = height if plain else batch_size
        var bounds = partitions(height, workers, 64)
        var jobs = List[_PerfectSinkJob](capacity=workers)
        for w in range(workers):
            if bounds[w] < bounds[w + 1]:
                jobs.append(
                    _PerfectSinkJob(
                        frame,
                        keys,
                        expressions,
                        perfect.value(),
                        layout,
                        bounds[w],
                        bounds[w + 1],
                        batch,
                    )
                )
        run_jobs(jobs)
        # DuckDB's `Combine`: every worker's slots fold into the first's,
        # in row order, so first and last keep their rows' order.
        ref merged = jobs[0].groups[0]
        var stride = merged.stride
        var states_bytes = 8 * merged.states_at
        for j in range(1, len(jobs)):
            ref other = jobs[j].groups[0]
            for g in range(merged.count()):
                var theirs = other.at(g)
                var row = Pointer[UInt64, MutAnyOrigin](
                    unsafe_from_address=theirs
                )
                if row[] == 0:
                    continue
                var mine = merged.at(g)
                var into = Pointer[UInt64, MutAnyOrigin](
                    unsafe_from_address=mine
                )
                if into[] == 0:
                    _ = external_call["memcpy", Int](mine, theirs, 8 * stride)
                    continue
                combine(
                    layout.slots, mine + states_bytes, theirs + states_bytes
                )
                if row[unsafe_offset=2] < into[unsafe_offset=2]:
                    into[unsafe_offset=2] = row[unsafe_offset=2]
        # The occupied slots, read out in the order of their first rows,
        # as every other eager route gives its groups (DuckDB's `Scan` takes
        # slot order; at most 2^12 slots, so the sort is small).
        var occupied = List[UInt128]()
        for g in range(merged.count()):
            var row = Pointer[UInt64, MutAnyOrigin](
                unsafe_from_address=merged.at(g)
            )
            if row[] != 0:
                occupied.append(
                    (UInt128(row[unsafe_offset=2]) << 64) | UInt128(g)
                )
        sort(occupied)
        var final = _Groups(
            len(layout.addressed),
            layout.addressed.copy(),
            layout.state_words,
            len(occupied),
        )
        for entry in occupied:
            var g = Int(UInt64(entry & 0xFFFFFFFFFFFFFFFF))
            var row = Pointer[UInt64, MutAnyOrigin](
                unsafe_from_address=final.at(final.copy_row(merged.at(g)))
            )
            _perfect_keys(perfect.value(), len(keys), g, row)
        var held = _HeldBuffers()
        var result = groups_result(final, layout, held)
        _ = jobs^
        return table_frame(template, result^, self._maintain_order)

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
