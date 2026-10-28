"""Streaming jobs, reduction state, and bounded batch helpers."""
from dataframe.planning.cost.eligibility import _pipeline_steps

from dataframe.categorical import (
    decode,
)
from dataframe.storage.column import (
    Column,
)
from dataframe.expr import (
    Expr,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.aggregation.expressions import (
    _row_local,
)
from dataframe.io.csv_reader import (
    _DecodeJob,
)
from dataframe.io.json_parallel import (
    _DecodeJob as _JsonDecodeJob,
)
from dataframe.join_hash import (
    PreparedHashIndex,
)
from dataframe.join_type import (
    JOIN_CROSS,
)
from dataframe.parallel import (
    Job,
    run_jobs,
)
from dataframe.pipeline import (
    STEP_DROP,
    STEP_FILTER,
    STEP_JOIN,
    STEP_SELECT,
    STEP_WITH_COLUMNS,
    Step,
    _BOUND_NONE,
    _TopBound,
)
from dataframe.planning.model import (
    DROP,
    EXPLODE,
    FILTER,
    JOIN,
    PlanNode,
    SELECT,
    SORT,
    UNNEST,
    WITH_COLUMNS,
)
from dataframe.planning.properties import (
    _distinct_inputs,
    _joined,
    _references,
)
from dataframe.planning.strategies import (
    _batch_shows_many_groups,
)
from dataframe.series import (
    Series,
)
from dataframe.aggregation.state import (
    _StreamMergeJob,
    _StreamReduction,
)
from std.collections import (
    Optional,
)
from std.memory import (
    ArcPointer,
)
from std.time import (
    perf_counter_ns,
)


def _coalesced(held: List[DataFrame], rows: Int) raises -> List[DataFrame]:
    """The held batches, in order, with runs of batches shorter than
    `rows` joined into batches of at least that many (the last may be
    shorter): one reduction job and one state each, where a job per batch
    of a few hundred rows cost 90 us of fixed work and left as many states
    to merge (TPC-DS q46: 33 batches of 330 rows). A batch already that
    long is left alone."""
    var out = List[DataFrame]()
    var parts = List[DataFrame]()
    var count = 0
    for frame in held:
        if frame.height() >= rows:
            if len(parts) > 0:
                out.append(_joined(parts))
                parts = List[DataFrame]()
                count = 0
            out.append(frame.copy())
            continue
        parts.append(frame.copy())
        count += frame.height()
        if count >= rows:
            out.append(_joined(parts))
            parts = List[DataFrame]()
            count = 0
    if len(parts) > 0:
        out.append(_joined(parts))
    return out^


def _stream_split_bits(workers: Int) -> Int:
    """Hash parts for a split state: a power of two, at least the workers
    (so every worker has a part to merge) and at least 2."""
    var bits = 1
    while (1 << bits) < workers and bits < 8:
        bits += 1
    return bits


struct _StateAccumulator(Movable):
    """A streamed reduction's accumulated states: one, or 2^bits hash parts
    once a batch shows many groups, each with the batch states waiting to
    merge into it (`_merge_parts`)."""

    var states: List[_StreamReduction]
    var pending: List[List[_StreamReduction]]
    var pending_groups: List[Int]
    var bits: Int
    # Rows absorbed so far: each batch's first-occurrence rows shift by it.
    var rows_seen: Int
    var workers: Int
    var split_groups: Int

    def __init__(out self, workers: Int, split_groups: Int):
        self.states = List[_StreamReduction]()
        self.pending = List[List[_StreamReduction]]()
        self.pending_groups = List[Int]()
        self.bits = 0
        self.rows_seen = 0
        self.workers = workers
        self.split_groups = split_groups

    def absorb(mut self, var pieces: List[_StreamReduction], rows: Int) raises:
        """Take one batch's state (or its parts), reduced from `rows` rows
        that follow every row absorbed before."""
        for k in range(len(pieces)):
            pieces[k].shift_firsts(self.rows_seen)
        self.rows_seen += rows
        if (
            len(self.states) == 0
            and pieces[0].grouped
            and pieces[0].group_count() >= self.split_groups
        ):
            self.bits = _stream_split_bits(self.workers)
        if self.bits > 0 and len(pieces) == 1 and pieces[0].grouped:
            var whole = pieces.pop()
            pieces = whole.split(self.bits)
        if len(self.states) == 0:
            for _ in range(len(pieces)):
                self.pending.append(List[_StreamReduction]())
                self.pending_groups.append(0)
            self.states = pieces^
            return
        for p in range(len(self.pending)):
            var piece = pieces.pop(0)
            self.pending_groups[p] += piece.group_count()
            self.pending[p].append(piece^)

    def merge(mut self, force: Bool) raises:
        if len(self.states):
            _merge_parts(
                self.states,
                self.pending,
                self.pending_groups,
                self.workers,
                force,
            )


def _merge_parts(
    mut states: List[_StreamReduction],
    mut pending: List[List[_StreamReduction]],
    mut pending_groups: List[Int],
    workers: Int,
    force: Bool,
) raises:
    """Merge each part whose pending groups reached its accumulated count
    (or 64 batches), or every part when forced. One state merges in place;
    parts are spread over `workers` jobs, each merging its parts in turn."""
    var parts = len(states)
    var due = List[Bool](capacity=parts)
    var any_due = False
    for p in range(parts):
        var ready = len(pending[p]) > 0 and (
            force
            or states[p].indexing == 1
            or pending_groups[p] >= states[p].group_count()
            or len(pending[p]) >= 64
        )
        due.append(ready)
        any_due = any_due or ready
    if not any_due:
        return
    if parts == 1:
        states[0].merge_all(pending[0])
        pending[0] = List[_StreamReduction]()
        pending_groups[0] = 0
        return
    var job_count = max(1, min(workers, parts))
    var jobs = List[_StreamMergeJob](capacity=job_count)
    for _ in range(job_count):
        jobs.append(_StreamMergeJob())
    # Part p goes to job p % job_count; parts not due keep their pending
    # batches here and ride along with nothing to merge.
    var carried = List[List[_StreamReduction]](capacity=parts)
    for p in range(parts):
        var state = states.pop(0)
        var waiting = pending.pop(0)
        ref job = jobs[p % job_count]
        job.states.append(state^)
        if due[p]:
            job.pending.append(waiting^)
            carried.append(List[_StreamReduction]())
            pending_groups[p] = 0
        else:
            job.pending.append(List[_StreamReduction]())
            carried.append(waiting^)
    run_jobs(jobs)
    for p in range(parts):
        states.append(jobs[p % job_count].states.pop(0))
        pending.append(carried.pop(0))


def _bounded_batch(
    mut frame: DataFrame, rows: List[Int], first: PlanNode
) raises -> Bool:
    """Narrow `frame` to the bounded `rows` of a top-k; when the first
    operation is a row-local filter, apply it too, reading only its own
    columns at those rows, and return True (the filter is done). The full
    rows are copied once, for the rows that pass both: a batch keeps every
    column, and copying all of them for rows the filter then drops cost
    more than the bound saved (ClickBench q23 reads ~100 columns)."""
    if first.kind == FILTER and _row_local(first.exprs):
        var reads = _references(first.exprs[0])
        if reads:
            var names = reads.value().copy()
            var picked = frame.select(names).take(rows)
            var positions = List[Int64](capacity=len(rows))
            for r in rows:
                positions.append(Int64(r))
            var marker = String("\x00bounded_row")
            picked = picked.with_column(
                Series(marker, Column[Int64](positions^))
            )
            var kept = picked.filter(first.exprs[0]).column(marker)
            if kept.is_chunked():
                kept = kept.rechunk()
            ref column = kept._data[Column[Int64]]
            var final = List[Int](capacity=len(column))
            for i in range(len(column)):
                final.append(Int(column._get(i)))
            frame = frame.take(final)
            return True
    if 8 * len(rows) <= frame.height():
        frame = frame.take(rows)
    return False


struct _StreamJob(Job):
    var decode: List[_DecodeJob]
    var json_decode: List[_JsonDecodeJob]
    var frame: DataFrame
    var operations: List[PlanNode]
    var joins: ArcPointer[List[DataFrame]]
    var indexes: ArcPointer[List[Optional[PreparedHashIndex]]]
    var expressions: List[Expr]
    var keys: List[String]
    var reduced: List[_StreamReduction]
    # Split a grouped state into 2^bits hash parts (0: keep one state), and
    # the rows it was reduced from.
    var bits: Int
    var rows: Int
    # With `counting`, the rows each operation read and wrote in this
    # batch (two entries per operation), summed by the caller into the
    # plan's execution report.
    var counting: Bool
    var counts: List[Int]
    # With `counting`, nanoseconds each operation took on this batch.
    var times: List[Int]
    # A grouped reduction keeps its batch (projected to the columns the
    # reduction reads) and builds no state while the stream has not yet
    # decided whether to collect batches for one eager group-by: a state
    # built then would be dropped if it does, and is built in a `replay`
    # job (no operations, the kept frame as input) if it does not.
    # `collect_only` keeps the frame and skips the reduction for good.
    var keep_frame: Bool
    var collect_only: Bool
    var replay: Bool
    # With `keep_frame`, a batch of fewer than 8 * split_groups rows
    # builds its state regardless; a longer one judges its first
    # 2 * split_groups rows and builds it only when they show few groups
    # (see the stream's `held_states`).
    var split_groups: Int
    var projection: List[String]
    # A streamed top-k's running bound on its first sort key (DuckDB's
    # Top-N dynamic filter): rows that cannot beat it are dropped before
    # any operation runs. See `_TopBound`.
    var bound: _TopBound

    def __init__(
        out self,
        var frame: DataFrame,
        operations: List[PlanNode],
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[PreparedHashIndex]]],
        expressions: List[Expr],
        keys: List[String],
    ):
        self.decode = List[_DecodeJob]()
        self.json_decode = List[_JsonDecodeJob]()
        self.frame = frame^
        self.operations = operations.copy()
        self.joins = joins.copy()
        self.indexes = indexes.copy()
        self.expressions = expressions.copy()
        self.keys = keys.copy()
        self.reduced = List[_StreamReduction]()
        self.bits = 0
        self.rows = 0
        self.counting = False
        self.counts = List[Int]()
        self.times = List[Int]()
        self.keep_frame = False
        self.collect_only = False
        self.replay = False
        self.split_groups = 0
        self.projection = List[String]()
        self.bound = _TopBound()

    def _shows_many(self) raises -> Bool:
        """Whether this batch's own rows show a many-group stream, so it
        keeps only its frame. A distinct count's state is its (key,
        value) pairs, judged on them at any batch length (PDS-H q16's
        swapped join gave 9.7K-row batches, each building sets of nearly
        a pair a row that the stream then dropped); otherwise a batch of
        8 * split_groups rows or more judges its groups."""
        var distinct = _distinct_inputs(self.expressions)
        if distinct and len(distinct.value()) > 0:
            for name in distinct.value():
                var pair = self.keys.copy()
                if name not in pair:
                    pair.append(name)
                if _batch_shows_many_groups(
                    self.frame, pair, self.split_groups
                ):
                    return True
            return False
        return self.frame.height() >= 8 * self.split_groups and (
            _batch_shows_many_groups(self.frame, self.keys, self.split_groups)
        )

    def take_reduced(mut self) -> List[_StreamReduction]:
        var out = self.reduced^
        self.reduced = List[_StreamReduction]()
        return out^

    def run(mut self) raises:
        if len(self.decode):
            self.decode[0].run()
            self.frame = self.decode.pop().into_frame()
        if len(self.json_decode):
            self.json_decode[0].run()
            self.frame = self.json_decode.pop().into_frame()
        if self.counting:
            self.counts = List[Int](length=2 * len(self.operations), fill=0)
            self.times = List[Int](length=len(self.operations) + 1, fill=0)
        var start = 0
        if self.bound.kind != _BOUND_NONE and len(self.operations) > 0:
            var bounded = self.bound.rows(self.frame)
            if bounded:
                var height = self.frame.height()
                if _bounded_batch(
                    self.frame, bounded.value(), self.operations[0]
                ):
                    start = 1
                    if self.counting:
                        self.counts[0] = height
                        self.counts[1] = self.frame.height()
        var k = start
        while k < len(self.operations):
            var began = 0
            if self.counting:
                self.counts[2 * k] = self.frame.height()
                began = Int(perf_counter_ns())
            var keeps = _filter_keeps(
                self.operations,
                k,
                self.expressions.copy() if not self.keep_frame else List[
                    Expr
                ](),
                self.keys,
            )
            if keeps:
                self.frame = self.frame.filter(
                    self.operations[k].exprs[0], keep=keeps.value()
                )
            else:
                _apply_operation(
                    self.operations[k], self.frame, self.joins, self.indexes
                )
            if self.counting:
                self.counts[2 * k + 1] = self.frame.height()
                self.times[k] += Int(perf_counter_ns()) - began
            k += 1
        if self.keep_frame and len(self.projection):
            self.frame = self.frame.select(self.projection)
        if self.collect_only or (self.keep_frame and self._shows_many()):
            self.rows = self.frame.height()
            if self.collect_only and len(self.projection):
                self.frame = self.frame.select(self.projection)
            return
        if len(self.expressions):
            var began = Int(perf_counter_ns()) if self.counting else 0
            var reduction = _StreamReduction(
                self.frame, self.expressions, self.keys
            )
            if self.counting:
                self.times[len(self.operations)] += (
                    Int(perf_counter_ns()) - began
                )
            self.rows = reduction.rows
            if self.bits > 0 and reduction.grouped:
                self.reduced = reduction.split(self.bits)
            else:
                self.reduced.append(reduction^)
            if not self.keep_frame:
                self.frame = self.frame.clear()


def _filter_keeps(
    operations: List[PlanNode], k: Int, after: List[Expr], keys: List[String]
) -> Optional[List[String]]:
    """The columns read after the filter at `k`, when that is known: by the
    selection right after it, or, when it is the last operation, by the
    stream's reduction (`after` and its group `keys`; empty `after` means
    the batch itself is kept). The filter then copies only those columns,
    not the ones only its predicate reads (PDS-H q6 filters on dates and
    quantities and sums prices and discounts)."""
    if operations[k].kind != FILTER:
        return None
    var exprs: List[Expr]
    var names = List[String]()
    if k + 1 < len(operations):
        if operations[k + 1].kind != SELECT:
            return None
        exprs = operations[k + 1].exprs.copy()
    elif len(after) > 0:
        # The group keys are read at the end of the pipeline, which is here.
        exprs = after.copy()
        names = keys.copy()
    else:
        return None
    for e in exprs:
        var reads = _references(e)
        if not reads:
            return None
        for name in reads.value():
            if name not in names:
                names.append(name)
    if len(names) == 0:
        return None
    return Optional(names^)


def _steps_of(operations: List[PlanNode]) -> List[Step]:
    var steps = List[Step](capacity=len(operations))
    for node in operations:
        var kind = STEP_DROP
        if node.kind == FILTER:
            kind = STEP_FILTER
        elif node.kind == SELECT:
            kind = STEP_SELECT
        elif node.kind == WITH_COLUMNS:
            kind = STEP_WITH_COLUMNS
        elif node.kind == JOIN:
            kind = STEP_JOIN
        steps.append(
            Step(
                kind,
                node.exprs.copy(),
                node.names.copy(),
                node.join_build_slot(),
                node.right_keys.copy(),
                node.how,
                node.join_suffix() if node.kind == JOIN else String(""),
                node.coalesce,
            )
        )
    return steps^


def _apply_operation(
    node: PlanNode,
    mut frame: DataFrame,
    joins: ArcPointer[List[DataFrame]],
    indexes: ArcPointer[List[Optional[PreparedHashIndex]]],
) raises:
    """Apply one row-local operation or join to a batch."""
    if node.kind == JOIN and indexes[][node.join_build_slot()]:
        frame = frame._join_impl(
            joins[][node.join_build_slot()],
            left_on=node.names,
            right_on=node.right_keys,
            how=node.how,
            suffix=node.join_suffix(),
            coalesce=node.coalesce,
            prepared=indexes[][node.join_build_slot()],
        )
    elif node.kind == JOIN and node.how == JOIN_CROSS:
        frame = frame.join(
            joins[][node.join_build_slot()],
            how=node.text,
            suffix=node.join_suffix(),
        )
    elif node.kind == JOIN:
        frame = frame.join(
            joins[][node.join_build_slot()],
            left_on=node.names,
            right_on=node.right_keys,
            how=node.text,
            suffix=node.join_suffix(),
            coalesce=node.coalesce,
        )
    elif node.kind == FILTER:
        frame = frame.filter(node.exprs[0])
    elif node.kind == SELECT:
        frame = frame.select_exprs(node.exprs)
    elif node.kind == WITH_COLUMNS:
        frame = frame.with_columns(node.exprs)
    elif node.kind == DROP:
        frame = frame.drop(node.names)
    elif node.kind == EXPLODE:
        frame = frame.explode(node.names)
    elif node.kind == UNNEST:
        frame = frame.unnest(node.text)
    elif node.kind == SORT:
        # A sort limited to its first rows (#332): this batch's own
        # first rows, selected on this worker alone.
        var n = len(node.names)
        frame = frame.take(
            frame._arg_sort_head(
                node.names,
                List[Bool](node.sort_options()[:n]),
                List[Bool](node.sort_options()[n:]),
                node.sort_limit(),
                threads=1,
            )
        )
