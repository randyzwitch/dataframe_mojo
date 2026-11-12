"""Streaming jobs and reduction state."""
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
from dataframe.io.scan_source import (
    ScanBatch,
)
from dataframe.joins.hash_table import (
    JoinHashTable,
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
    _steps_over_frame,
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
    _references,
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


struct _StreamJob(Job):
    # A file's batch, decoded by the job.
    var scan: List[ScanBatch]
    var frame: DataFrame
    var operations: List[PlanNode]
    var joins: ArcPointer[List[DataFrame]]
    var indexes: ArcPointer[List[Optional[JoinHashTable]]]
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
    # A grouped reduction's batch is collected for the eager group-by:
    # the job keeps its frame (projected to the columns the reduction
    # reads) and builds no state.
    var collect_only: Bool
    var projection: List[String]

    def __init__(
        out self,
        var frame: DataFrame,
        operations: List[PlanNode],
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[JoinHashTable]]],
        expressions: List[Expr],
        keys: List[String],
    ):
        self.scan = List[ScanBatch]()
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
        self.collect_only = False
        self.projection = List[String]()

    def take_reduced(mut self) -> List[_StreamReduction]:
        var out = self.reduced^
        self.reduced = List[_StreamReduction]()
        return out^

    def run(mut self) raises:
        if len(self.scan):
            self.frame = self.scan.pop().into_frame()
        if self.counting:
            self.counts = List[Int](length=2 * len(self.operations), fill=0)
            self.times = List[Int](length=len(self.operations) + 1, fill=0)
        var k = 0
        while k < len(self.operations):
            var began = 0
            if self.counting:
                self.counts[2 * k] = self.frame.height()
                began = Int(perf_counter_ns())
            var keeps = _filter_keeps(
                self.operations, k, self.expressions, self.keys
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
        if self.collect_only:
            self.rows = self.frame.height()
            if len(self.projection):
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
    indexes: ArcPointer[List[Optional[JoinHashTable]]],
) raises:
    """Apply one row-local operation or join to a batch."""
    if node.kind == JOIN and indexes[][node.join_build_slot()]:
        # A batch probes the join's shared table as a pipeline's morsel does.
        var counts = List[Int]()
        frame = _steps_over_frame(
            frame,
            [
                Step(
                    STEP_JOIN,
                    List[Expr](),
                    node.names.copy(),
                    node.join_build_slot(),
                    node.right_keys.copy(),
                    node.how,
                    node.join_suffix(),
                    node.coalesce,
                )
            ],
            joins,
            indexes,
            counts,
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
                maintain_order=node.maintain_order,
            )
        )
