"""Eager grouping requests, strategy dispatch, and worker jobs."""
from dataframe.aggregate import (
    Reducer,
)
from dataframe.aggregate_table import (
    aggregate_table,
    key_supported,
    reduction_supported,
    states_supported,
)
from dataframe.aggregation.expressions import (
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
from dataframe.canonical_labels import (
    canonical_labels,
)
from dataframe.categorical import (
    decode,
)
from dataframe.column import (
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
    changes_length,
    col,
    is_reduction,
)
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.gather import (
    can_filter_aligned_chunks,
    take_parallel,
    take_sorted_chunked,
)
from dataframe.hash_agg import (
    hash_agg_eligible,
    hash_aggregate,
)
from dataframe.hashing import (
    RowKeys,
    encode_rows,
    encode_rows_parallel,
    encode_string_rows_parallel,
)
from dataframe.indexed_reduce import (
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
from dataframe.partition import (
    Partitioner,
    encode_bucket,
    low_cardinality,
    small_key_product,
)
from dataframe.perfect_aggregate import (
    PerfectLayout,
    perfect_layout,
    perfect_slots,
)
from dataframe.series import (
    Series,
    sort_indices,
)
from dataframe.string_column import (
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


struct _RangeAggJob(Job):
    """Reduce rows [start, end) into mergeable per-group state."""

    var frame: DataFrame
    var keys: List[Series]
    var expressions: List[Expr]
    var key_names: Dict[String, Bool]
    var start: Int
    var end: Int
    var budget: Int
    # None when the range held more groups than the budget.
    var state: Optional[_StreamReduction]

    def __init__(
        out self,
        frame: DataFrame,
        keys: List[Series],
        expressions: List[Expr],
        key_names: Dict[String, Bool],
        start: Int,
        end: Int,
        budget: Int,
    ):
        self.frame = frame.copy()
        self.keys = keys.copy()
        self.expressions = expressions.copy()
        self.key_names = key_names.copy()
        self.start = start
        self.end = end
        self.budget = budget
        self.state = None

    def run(mut self) raises:
        var length = self.end - self.start
        var frame = self.frame.slice(self.start, length)
        var keys = List[Series](capacity=len(self.keys))
        for key in self.keys:
            keys.append(key.slice(self.start, length))
        var state = _StreamReduction(
            frame, self.expressions, keys, self.key_names
        )
        if state.keys.height() <= self.budget:
            self.state = state^


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


struct _BucketJob(Job):
    """Group one hash bucket on its own: encode its keys, evaluate the
    aggregates, and remember each group's first row for reordering."""

    var keys: List[Series]
    var columns: List[Series]
    var expressions: List[Expr]
    var batch_size: Int
    var height: Int
    # Workers for the encode; 1 inside a batch, more when run on the caller.
    var encoders: Int
    var result: List[Series]
    var firsts: List[Int]

    def __init__(
        out self,
        var keys: List[Series],
        var columns: List[Series],
        expressions: List[Expr],
        batch_size: Int,
        height: Int,
        encoders: Int = 1,
    ):
        self.keys = keys^
        self.columns = columns^
        self.expressions = expressions.copy()
        self.batch_size = batch_size
        self.height = height
        self.encoders = encoders
        self.result = List[Series]()
        self.firsts = List[Int]()

    def run(mut self) raises:
        var groups = _encode_parallel(
            self.keys, self.encoders
        ) if self.encoders > 1 else encode_rows(self.keys, nulls_equal=True)
        for key in self.keys:
            self.result.append(key.take(groups.representatives))
        # Bind against this bucket's columns so source indices line up.
        var bound = _bind_all(self.expressions, self.columns)
        for expression in bound:
            self.result.append(
                evaluate(
                    expression,
                    self.columns,
                    self.height,
                    batch_size=self.batch_size,
                    grouped=True,
                    groups=groups.ids,
                    group_count=groups.count(),
                )
            )
        self.firsts = groups.representatives.copy()


struct _HashedBucketJob(Job):
    """Group one hash bucket from the partitioner's key hashes (#334): the
    keys stay in the source columns, compared in place on a hash match, and
    only the aggregated columns were gathered into bucket order."""

    var keys: List[Series]
    var hashes: Int
    var order: Int
    var lo: Int
    var hi: Int
    var columns: List[Series]
    var expressions: List[Expr]
    var batch_size: Int
    var sources: List[Series]
    var indexed: List[Bool]
    var result: List[Series]
    var firsts: List[Int]

    def __init__(
        out self,
        keys: List[Series],
        hashes: Int,
        order: Int,
        lo: Int,
        hi: Int,
        var columns: List[Series],
        expressions: List[Expr],
        batch_size: Int,
        sources: List[Series],
        indexed: List[Bool],
    ):
        self.keys = keys.copy()
        self.hashes = hashes
        self.order = order
        self.lo = lo
        self.hi = hi
        self.columns = columns^
        self.expressions = expressions.copy()
        self.batch_size = batch_size
        self.sources = sources.copy()
        self.indexed = indexed.copy()
        self.result = List[Series]()
        self.firsts = List[Int]()

    def run(mut self) raises:
        ref hashes = Pointer[List[UInt64], MutAnyOrigin](
            unsafe_from_address=self.hashes
        )[]
        ref order = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.order
        )[]
        var ids = List[Int]()
        var firsts = List[Int]()
        encode_bucket(
            self.keys,
            Span(hashes)[self.lo : self.hi],
            Span(order)[self.lo : self.hi],
            ids,
            firsts,
            exact_hashes=True,
        )
        for key in self.keys:
            self.result.append(key.take(firsts))
        var rows = order.unsafe_ptr().unsafe_offset(self.lo)
        for i in range(len(self.expressions)):
            if self.indexed[i]:
                var expression = bind(self.expressions[i], self.sources)
                var result: Series
                if indexed_reductions([expression.copy()], self.sources):
                    result = reduce_indexed(
                        expression, self.sources, rows, ids, len(firsts)
                    )
                else:
                    result = reduce_indexed_batches(
                        expression,
                        self.sources,
                        rows,
                        ids,
                        len(firsts),
                        self.batch_size,
                    )
                self.result.append(result.renamed(self.expressions[i]._name))
            else:
                var expression = bind(self.expressions[i], self.columns)
                self.result.append(
                    evaluate(
                        expression,
                        self.columns,
                        self.hi - self.lo,
                        batch_size=self.batch_size,
                        grouped=True,
                        groups=ids,
                        group_count=len(firsts),
                    )
                )
        self.firsts = firsts^


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
    """An eager grouping request. No per-group dataframe materialization.

    It owns a snapshot of the input and the evaluated key columns.
    """

    var _frame: DataFrame
    var _keys: List[Series]
    var _maintain_order: Bool

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
        for expression in expressions:
            if changes_length(expression):
                prepared.append(expression.implode())
                changed = True
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
        if _stream_reductions(expressions) and not self._many_distinct(
            expressions
        ):
            var perfect = self._agg_perfect(expressions, key_names, workers)
            if perfect:
                return perfect.take()
        var result: DataFrame
        # Keys are unchanged throughout this aggregation. Reuse the sampled
        # preference across hash, range and partitioned strategy selection.
        var whole = low_cardinality(self._keys) if workers > 1 else False
        # Keys of a small dense domain went to the perfect hash above, as
        # DuckDB's do; any other grouping goes through the aggregate table
        # whatever its number of groups, whose per-thread tables adapt to
        # however many there are. Neither DuckDB nor Polars picks a route
        # by an estimate of the groups. The per-range routes below keep the
        # reductions the table has no state for.
        if workers > 1 and self._table_supported(expressions):
            var table = self._agg_table(expressions, workers)
            if table:
                return _pack_struct_keys(table.value())
        if (
            workers > 1
            and len(self._keys) == 1
            and not whole
            and hash_agg_eligible(self._keys[0], bound, self._frame._columns)
        ):
            return _pack_struct_keys(self._agg_hash(bound, workers))
        var ranged = Optional[DataFrame]()
        if (
            workers > 1
            and _stream_reductions(expressions)
            and not self._many_distinct(expressions)
        ):
            ranged = self._agg_ranges(expressions, key_names, workers, whole)
        if ranged:
            result = ranged.take()
        elif workers > 1:
            result = self._agg_partitioned(
                expressions, bound, batch_size, workers, whole
            )
        else:
            result = self._agg_whole(bound, batch_size)
        return _pack_struct_keys(result)

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
        self, expressions: List[Expr], workers: Int
    ) raises -> Optional[DataFrame]:
        """Grouped reductions through the aggregate table: each worker
        hashes its morsels' keys where they are and probes its own table,
        updating its states by group; partitions of the groups then merge
        across workers in parallel (`aggregate_table.mojo`). None when some reduction's
        state does not fit inline in a group row."""
        var empty_keys = List[Series](capacity=len(self._keys))
        for key in self._keys:
            empty_keys.append(key.slice(0, 0))
        var template = _StreamReduction(
            self._frame.clear(), expressions, empty_keys, self._key_names()
        )
        var specs = table_specs(template)
        var inputs = table_inputs(self._frame, self._keys, expressions)
        if not states_supported(specs, inputs):
            return None
        trace_path("group_by.aggregate_table")
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
        var morsel = max(
            4096, min(1 << 16, self._frame.height() // (4 * workers) + 1)
        )
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

    def _agg_hash(
        self, bound: List[BoundExpr], workers: Int
    ) raises -> DataFrame:
        """Many groups of one Int64 key: states updated in place per worker
        range, then merged by hash part (`hash_agg.mojo`), with no column
        gathered into bucket order."""
        trace_path("group_by.hash_agg")
        var parts = hash_aggregate(
            self._keys[0], bound, self._frame._columns, workers
        )
        ref firsts = parts[0]
        var columns = List[Series](capacity=len(bound) + 1)
        columns.append(self._keys[0].take(firsts))
        for j in range(len(bound)):
            var pieces = List[Series](capacity=len(parts[1]))
            for p in range(len(parts[1])):
                pieces.append(parts[1][p][j].copy())
            var merged = Series._from_chunks(pieces).rechunk()
            columns.append(merged.renamed(bound[j].expr._name))
        var grouped = DataFrame(columns^, height=len(firsts))
        if not self._maintain_order:
            return grouped^
        return grouped.take(sort_indices([firsts.copy()]))

    def _many_distinct(self, expressions: List[Expr]) raises -> Bool:
        """Whether some `n_unique` reads many distinct values (#336).

        Per-range sets of (group, value) pairs merge on one thread, which
        is cheap only when values repeat: grouping IsRefresh by RegionID's
        3,586 values takes 57 ms that way and 75 ms through the whole-frame
        path's hash partitions, while UserID's 1.5M values take 314 ms and
        87 ms. A sample of evenly spaced rows tells them apart: at most a
        tenth of it distinct (RegionID shows 9%, SearchPhrase 14%, UserID
        all) keeps the ranges. Any input that is not a column is assumed
        to have many values.
        """
        var height = self._frame.height()
        var sample = min(height, 4096)
        if sample == 0:
            return False
        var rows = List[Int](capacity=sample)
        for k in range(sample):
            rows.append(k * height // sample)
        for expression in expressions:
            ref nodes = expression._nodes
            for node in nodes:
                if node.op != N_UNIQUE:
                    continue
                ref input = nodes[node.left]
                if input.op != COL:
                    return True
                var picked = self._frame.column(input.text).take(rows)
                var distinct = encode_rows([picked^], nulls_equal=True).count()
                if 10 * distinct > sample:
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

    def _agg_ranges(
        self,
        expressions: List[Expr],
        key_names: Dict[String, Bool],
        workers: Int,
        whole: Bool,
    ) raises -> Optional[DataFrame]:
        """Reduce each worker's row range to per-group state, then merge.

        Every row-local reduction the streaming executor supports keeps a
        mergeable state (see _StreamReduction), so this serves any list of
        them over any key types. Ranges are zero-copy slices; no column is
        gathered. Private states must stay small for the serial merge, so
        only low-cardinality keys qualify, and any range holding more groups
        than the budget falls back to hash partitioning. Groups keep
        first-occurrence order, as the whole-frame path does.
        """
        var height = self._frame.height()
        var budget = max(1, 1_000_000 // (workers * len(expressions)))
        # The same sampled estimate that picks whole-frame encoding over
        # hash partitioning: every range of a high-cardinality key would
        # hold most groups, and merging those states serially loses.
        if not (whole or small_key_product(self._keys)):
            return None
        var bounds = partitions(height, workers, 64)
        var jobs = List[_RangeAggJob](capacity=workers)
        for w in range(workers):
            jobs.append(
                _RangeAggJob(
                    self._frame,
                    self._keys,
                    expressions,
                    key_names,
                    bounds[w],
                    bounds[w + 1],
                    budget,
                )
            )
        run_jobs(jobs)
        for w in range(len(jobs)):
            if not jobs[w].state:
                return None
        var merged = jobs[0].state.take()
        for w in range(1, len(jobs)):
            merged.merge(jobs[w].state.value())
        trace_path("group_by.ranges")
        return merged.finish()

    def _referenced(self, bound: List[BoundExpr]) -> List[Series]:
        """The frame columns the aggregates read, in frame order; every
        column if a selector is still unexpanded."""
        var wanted = Dict[String, Bool]()
        var everything = False
        for expression in bound:
            for node in expression.expr._nodes:
                if node.op == COL:
                    wanted[node.text] = True
                elif node.op == SELECTOR:
                    everything = True
        var columns = List[Series]()
        for column in self._frame._columns:
            if everything or column.name() in wanted:
                columns.append(column.copy())
        return columns^

    def _partitioned_references(
        self, bound: List[BoundExpr]
    ) -> Tuple[List[Bool], List[Series], List[Series]]:
        """Per-expression capabilities and distinct indexed/gathered sources.

        A column shared by both routes intentionally appears in both lists;
        a supported-only source never enters the full-column gather.
        """
        var indexed = List[Bool]()
        var fallback = List[BoundExpr]()
        var direct = List[BoundExpr]()
        for expression in bound:
            var capable = _stream_reductions(
                [expression.expr.copy()]
            ) or indexed_reductions([expression.copy()], self._frame._columns)
            # Categorical evaluation decodes/re-encodes dictionary values;
            # nested expressions have their own preparation requirements.
            # Keep those on the ordinary evaluator, independently of other
            # aggregates in the same request.
            for dtype in expression.dtypes:
                if dtype.is_categorical() or dtype.is_nested():
                    capable = False
            for i in range(len(expression.expr._nodes)):
                if (
                    expression.expr._nodes[i].op == COL
                    and expression.sources[i] >= 0
                ):
                    var dtype = self._frame._columns[
                        expression.sources[i]
                    ].dtype()
                    if dtype.is_categorical() or dtype.is_nested():
                        capable = False
            indexed.append(capable)
            if capable:
                direct.append(expression.copy())
            else:
                fallback.append(expression.copy())
        return (indexed^, self._referenced(direct), self._referenced(fallback))

    def _agg_partitioned(
        self,
        expressions: List[Expr],
        bound: List[BoundExpr],
        batch_size: Int,
        workers: Int,
        whole: Bool,
    ) raises -> DataFrame:
        """Group by hash bucket in parallel; see dataframe/partition.mojo.

        Each bucket holds a disjoint set of keys, so buckets are encoded
        and reduced independently and their outputs concatenated. Output
        order is bucket order unless maintain_order, which sorts groups by
        their first input row afterwards (O(groups), not O(rows)).

        A bucket holding several times its share of rows (a skewed key)
        would serialize the batch, so such buckets run on the calling
        thread, where their reduce can use the pool, and only the light
        buckets are jobs.
        """
        var height = self._frame.height()
        # A sampled estimate decides whether scattering is worth its gather;
        # on a low-cardinality key the serial encode it replaces is cheap.
        if whole:
            return self._agg_whole(bound, batch_size)
        trace_path("group_by.partitioned")
        # Hashing, scattering and encoding a composite key cost 50 to 80 ns
        # a row of thread time, against under 1 ns for the scans the shared
        # 64K-row floor is sized for, so this route divides finer: 180K
        # rows of TPC-DS q39 (45K groups) took 15 ms on the floor's two
        # workers, the same at any thread count.
        var threads = max(
            workers,
            min(configured_workers(), height // _PARTITION_ROWS_PER_WORKER),
        )
        var partitioner = Partitioner(self._keys, threads)
        var parts = partitioner.scatter(threads, with_hashes=True)
        var buckets = parts.buckets()
        var referenced = self._referenced(bound)
        # Heavy: a bucket big enough to serialize the batch on its own. It
        # must be large relative to the frame, not only to its share, or
        # low cardinality (few occupied buckets) would count as heavy.
        var heavy_rows = max(height // 8, 4 * height // buckets)
        var any_heavy = False
        for b in range(buckets):
            if parts.bounds[b + 1] - parts.bounds[b] > heavy_rows:
                any_heavy = True
        if not any_heavy:
            # Encode each bucket from the key hashes. Plain reductions read
            # values at their source rows; others gather values into bucket
            # order first.
            var preparation = self._partitioned_references(bound)
            var indexed = preparation[0].copy()
            var sources = preparation[1].copy()
            var gathered_sources = preparation[2].copy()
            var any_indexed = False
            var any_fallback = False
            for capable in indexed:
                any_indexed = any_indexed or capable
                any_fallback = any_fallback or not capable
            for i in range(len(sources)):
                if sources[i].is_chunked():
                    sources[i] = sources[i].rechunk()
            if any_indexed:
                trace_path("group_by.partitioned.indexed")
            if any_indexed and any_fallback:
                trace_path("group_by.partitioned.mixed")
            # Only aggregates the indexed readers cannot serve need their
            # sources in bucket order. With none, there is nothing to
            # gather and no reason to copy the row order for it.
            var values = List[Series]()
            if len(gathered_sources) > 0:
                values = take_parallel(
                    gathered_sources, parts.order.copy(), threads
                )
            var jobs = List[_HashedBucketJob]()
            for b in range(buckets):
                var lo = parts.bounds[b]
                var hi = parts.bounds[b + 1]
                if hi == lo:
                    continue
                var bucket_columns = List[Series](capacity=len(values))
                for column in values:
                    bucket_columns.append(column.slice(lo, hi - lo))
                jobs.append(
                    _HashedBucketJob(
                        partitioner.keys,
                        Int(Pointer(to=parts.hashes)),
                        Int(Pointer(to=parts.order)),
                        lo,
                        hi,
                        bucket_columns^,
                        expressions,
                        batch_size,
                        sources,
                        indexed,
                    )
                )
            run_jobs(jobs)
            var frames = List[DataFrame](capacity=len(jobs))
            var starts = List[Int]()
            var runs = List[Int]()
            runs.append(0)
            for j in range(len(jobs)):
                frames.append(
                    DataFrame(jobs[j].result.copy(), height=len(jobs[j].firsts))
                )
                for row in jobs[j].firsts:
                    starts.append(row)
                runs.append(len(starts))
            _ = partitioner^
            _ = parts^
            var grouped = concat(frames)
            if not self._maintain_order:
                return grouped^
            # Each bucket's groups are already in first-row order (its
            # rows were scattered in input order), so the whole order is
            # a merge of the buckets' runs, not a sort: 45K groups of
            # TPC-DS q39 took 2.4 ms to sort and 0.2 ms to merge.
            return grouped.take(_merge_ascending_runs(starts, runs))
        var sources = List[Series](capacity=len(self._keys) + len(referenced))
        for key in self._keys:
            sources.append(key.copy())
        for column in referenced:
            sources.append(column.copy())
        var gathered = take_parallel(sources, parts.order.copy(), threads)
        var keys = List[Series](capacity=len(self._keys))
        var columns = List[Series](capacity=len(referenced))
        for i in range(len(gathered)):
            if i < len(self._keys):
                keys.append(gathered[i].copy())
            else:
                columns.append(gathered[i].copy())
        var light = List[_BucketJob]()
        var light_offsets = List[Int]()
        var done = List[_BucketJob]()
        var done_offsets = List[Int]()
        for b in range(buckets):
            var lo = parts.bounds[b]
            var hi = parts.bounds[b + 1]
            if hi == lo:
                continue
            var bucket_keys = List[Series](capacity=len(keys))
            for key in keys:
                bucket_keys.append(key.slice(lo, hi - lo))
            var bucket_columns = List[Series](capacity=len(columns))
            for column in columns:
                bucket_columns.append(column.slice(lo, hi - lo))
            var heavy = hi - lo > heavy_rows
            var job = _BucketJob(
                bucket_keys^,
                bucket_columns^,
                expressions,
                batch_size,
                hi - lo,
                worker_count(hi - lo) if heavy else 1,
            )
            if heavy:
                # On the caller, so its encode and reduce can use the pool.
                job.run()
                done.append(job^)
                done_offsets.append(lo)
            else:
                light.append(job^)
                light_offsets.append(lo)
        run_jobs(light)
        while len(light) > 0:
            done.append(light.pop(0))
            done_offsets.append(light_offsets.pop(0))

        var frames = List[DataFrame](capacity=len(done))
        var firsts = List[Int]()
        for j in range(len(done)):
            var groups = len(done[j].firsts)
            frames.append(DataFrame(done[j].result.copy(), height=groups))
            for r in done[j].firsts:
                firsts.append(parts.order[done_offsets[j] + r])
        var result = concat(frames)
        if not self._maintain_order:
            return result^
        return result.take(sort_indices([firsts^]))

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


# Rows a worker in the hash-partitioned group-by: about 1 ms of work each.
comptime _PARTITION_ROWS_PER_WORKER = 16384


def group_by_workers(height: Int) -> Int:
    """Workers the eager group-by (`GroupBy.agg`) divides `height` rows
    among: the shared floor, or one per `_PARTITION_ROWS_PER_WORKER` rows
    of hashing and encoding."""
    return max(
        worker_count(height),
        min(configured_workers(), height // _PARTITION_ROWS_PER_WORKER),
    )


def range_states(keys: List[Series]) raises -> Bool:
    """Whether the eager group-by keeps groups of `keys` in each worker's
    private state over its own rows (the whole-frame and range routes, by
    their sampled tests) rather than hash-partitioning every row first."""
    return low_cardinality(keys) or small_key_product(keys)
