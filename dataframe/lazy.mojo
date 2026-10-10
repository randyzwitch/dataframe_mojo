"""Lazy query plans over the eager operators.

A LazyFrame records operations as a flat list of plan nodes (children always
precede parents). Nothing reads data until `collect`. Optimization rewrites
the plan before execution:

- filter splitting: a filter on `a & b` moves as two filters, and one on an
  OR of ANDs also gets the filters it implies on each input's columns
  (`(a1 & b1) | (a2 & b2)` implies `a1 | a2`); filters that end up
  together merge back into one;
- predicate pushdown: filters move below with_columns/select that do not
  produce the columns they read, below a sort when they are row-local, and
  into the side of an inner join (or the left side of a left/semi/anti join)
  that owns every column they read, and below a group_by when they read
  only its keys;
  row-local filters directly above unrestricted CSV scans run per decode range;
- projection pushdown: scans read only the columns the rest of the plan uses
  (CSV and Parquet scans decode only those fields); a join passes each input
  only its keys and the columns read above it from that side;
- slice pushdown: a head/slice directly over a CSV scan becomes `n_rows`;
- join order, when the plan runs: in a chain of inner joins, inputs that
  filters narrow run first, a selective join moves into the input holding
  its keys, and the most selective joins run first (`_order_joins`);
- row-group pruning: a row-local filter directly above a Parquet scan reads
  the footer statistics first and decodes only the row groups that can
  hold a match (see `parquet._pruned_row_groups`).

Filters never move past a slice, unique, right/full join, or a group_by
whose aggregates they read, because that would change which rows those
operators see.
"""
from std.os import getenv
from std.time import perf_counter_ns
from std.io import FileDescriptor
from std.collections import Dict, Optional
from std.memory import ArcPointer
from .csv_reader import _CsvBatches, _DecodeJob
from .csv_types import _map_file
from .parquet import _ParquetBatches
from .pipeline import (
    _BOUND_FLOAT,
    _BOUND_INT,
    _BOUND_NONE,
    _TopBound,
    STEP_DROP,
    STEP_FILTER,
    STEP_JOIN,
    STEP_SELECT,
    STEP_WITH_COLUMNS,
    Step,
    run_pipeline,
)
from .parallel import Crew, Job, Pool, configured_workers, run_jobs
from .execution_backend import _select_backend, _BackendDecision
from ._accel_provider import (
    execute as _execute_accel,
    execute_profiled as _profile_accel,
    describe as _describe_accel,
    select_auto as _select_auto_accel,
    execute_auto as _execute_auto_accel,
)
from .streaming import _StreamReduction, _StreamMergeJob, _finish_parts
from .expr import (
    ARG_MAX,
    ARG_MIN,
    SUM,
    COUNT,
    MIN,
    MAX,
    MEAN,
    FIRST,
    LAST,
    STD,
    VAR,
    LEN,
    ANY,
    ALL,
    NULL_COUNT,
    N_UNIQUE,
    MEDIAN,
    QUANTILE,
    CORR,
    COV,
)
from .frame import concat
from .sorted_range import sorted_window
from .value import AnyValue
from .csv import CsvSchema, read_csv
from .csv_reader import read_csv_explicit, read_csv_inferred
from .expr import (
    AND,
    COL,
    Node,
    OR,
    OVER,
    SELECTOR,
    Expr,
    col,
    conjuncts,
    lit,
    is_reduction,
    is_window,
    subtree,
)
from .frame import DataFrame, GroupBy, _row_local, _stream_reductions
from .parquet import (
    _pruned_row_groups,
    parquet_row_group_statistics,
    read_parquet,
)
from .column import Column
from .series import Series
from .dtype import DataType, NUMERIC_DTYPES
from .hashing import encode_rows
from .trace import trace_path
from .execution import direct_reductions
from .execution_report import ExecutionReport
from .join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_SEMI,
    join_code,
    join_type,
)
from .join_hash import (
    PreparedHashIndex,
    int64_progression,
    prepare_hash_index,
    prepare_progression_index,
    prefer_left_build,
)

comptime SCAN_FRAME = 0
comptime SCAN_CSV = 1
comptime FILTER = 2
comptime SELECT = 3
comptime WITH_COLUMNS = 4
comptime AGG = 5
comptime JOIN = 6
comptime SORT = 7
comptime SLICE = 8
comptime UNIQUE = 9
comptime DROP = 10
comptime SCAN_PARQUET = 11
comptime EXPLODE = 12
comptime UNNEST = 13
comptime ASOF = 14


@fieldwise_init
struct PlanNode(Copyable):
    var kind: Int
    var left: Int
    var right: Int
    var exprs: List[Expr]
    var names: List[String]
    var names2: List[String]
    var flags: List[Bool]
    var text: String
    var offset: Int
    var length: Int
    var maintain_order: Bool
    # A JOIN's type code (see join_type); -1 for an unknown name, which the
    # eager join reports from the original text when the plan executes.
    var how: Int
    # A JOIN's right-side key names, paired with `names` (the left keys),
    # and whether key columns coalesce into the left names.
    var right_keys: List[String]
    var coalesce: Bool
    # Optional internal source-execution counter, shared across plan copies.
    # No counter storage is allocated unless profiling/tests attach one.
    var _executions: Optional[ArcPointer[Int]]
    var asof_tolerance: Optional[AnyValue]

    def _record_execution(self):
        if self._executions:
            var counter = self._executions.value()
            counter[] += 1


def _plan_node(
    kind: Int,
    left: Int = -1,
    right: Int = -1,
    exprs: List[Expr] = List[Expr](),
    names: List[String] = List[String](),
    names2: List[String] = List[String](),
    flags: List[Bool] = List[Bool](),
    text: String = "",
    offset: Int = 0,
    length: Int = -1,
    maintain_order: Bool = False,
    how: Int = -1,
    right_keys: List[String] = List[String](),
    coalesce: Bool = True,
) -> PlanNode:
    return PlanNode(
        kind,
        left,
        right,
        exprs.copy(),
        names.copy(),
        names2.copy(),
        flags.copy(),
        text,
        offset,
        length,
        maintain_order,
        how,
        right_keys.copy(),
        coalesce,
        Optional[ArcPointer[Int]](),
        Optional[AnyValue](),
    )


def _references(expr: Expr) -> Optional[List[String]]:
    """Columns an expression reads, or None when a selector reads any."""
    var names = List[String]()
    for node in expr._nodes:
        if node.op == SELECTOR:
            return None
        if node.op == COL:
            names.append(node.text)
        if node.op == OVER and node.text2.byte_length() > 0:
            # over() partitions read their key columns. Other operators
            # keep other text there (a strptime dtype, cut labels).
            for part in node.text2.split("\x1f"):
                names.append(String(part))
    return names^


def _output_names(exprs: List[Expr]) -> List[String]:
    var names = List[String]()
    for e in exprs:
        names.append(e._name)
    return names^


def _boolean_parts(expr: Expr, op: Int) -> List[Expr]:
    """The operands of a chain of `op` (AND or OR) at the root of expr, left
    to right; expr alone when its root is another operator."""
    var parts = List[Expr]()
    _collect_parts(expr, len(expr._nodes) - 1, op, parts)
    return parts^


def _collect_parts(expr: Expr, index: Int, op: Int, mut parts: List[Expr]):
    ref node = expr._nodes[index]
    if node.op == op and node.left >= 0 and node.right >= 0:
        _collect_parts(expr, node.left, op, parts)
        _collect_parts(expr, node.right, op, parts)
    else:
        parts.append(subtree(expr, index))


def _implied_filters(predicate: Expr) -> List[Expr]:
    """Filters that an OR of ANDs implies on fewer columns.

    `(a1 & b1) | (a2 & b2)` keeps a row only if some branch holds, and so
    only if `a1 | a2` holds, where a1 and a2 read one set of columns: that
    filter can move into the join input owning those columns, as DuckDB
    derives it. A branch with no part on those columns implies nothing.
    Under SQL's three-valued logic a true branch has every part true, so
    the derived filter keeps every row the predicate keeps.
    """
    var out = List[Expr]()
    var branches = _boolean_parts(predicate, OR)
    if len(branches) < 2:
        return out^
    var atoms = List[List[Expr]]()
    var reads = List[List[List[String]]]()
    for branch in branches:
        var parts = _boolean_parts(branch, AND)
        var part_reads = List[List[String]]()
        for part in parts:
            var r = _references(part)
            if not r:
                return out^
            part_reads.append(r.value().copy())
        atoms.append(parts^)
        reads.append(part_reads^)
    var tried = List[List[String]]()
    for a in range(len(atoms[0])):
        ref columns = reads[0][a]
        if len(columns) == 0:
            continue
        var seen = False
        for t in tried:
            seen = seen or (_covers(t, columns) and _covers(columns, t))
        if seen:
            continue
        tried.append(columns.copy())
        var implied = Expr(List[Node](), "")
        var found = True
        var partial = False
        for b in range(len(atoms)):
            var kept = Expr(List[Node](), "")
            for j in range(len(atoms[b])):
                if len(reads[b][j]) > 0 and _covers(columns, reads[b][j]):
                    kept = (
                        atoms[b][j].copy() if len(kept._nodes)
                        == 0 else kept & atoms[b][j]
                    )
                else:
                    partial = True
            if len(kept._nodes) == 0:
                found = False
                break
            implied = kept^ if len(implied._nodes) == 0 else implied | kept
        # A derived filter equal to the predicate adds nothing.
        if found and partial:
            out.append(implied^)
    return out^


def _stream_rows(node: PlanNode) -> Bool:
    if node.kind == FILTER or node.kind == WITH_COLUMNS:
        return _row_local(node.exprs)
    if node.kind == SELECT:
        if not _row_local(node.exprs):
            return False
        # An all-scalar select produces one row for the whole input.
        for expression in node.exprs:
            for item in expression._nodes:
                if item.op == COL or item.op == SELECTOR:
                    return True
        return False
    return node.kind == DROP or node.kind == EXPLODE or node.kind == UNNEST


def _stream_split_groups() -> Int:
    """Groups in the first batch at which streaming aggregation splits its
    state into hash parts. DATAFRAME_STREAM_SPLIT_GROUPS overrides it, so
    tests can exercise the split path on small inputs."""
    var setting = getenv("DATAFRAME_STREAM_SPLIT_GROUPS")
    if setting.byte_length() > 0:
        try:
            return max(1, Int(setting))
        except:
            pass
    return 4096


def _stream_collect_bytes() -> Int:
    """Bytes of collected batches a many-group streaming aggregation holds
    before reducing them into a mergeable state and continuing, so its
    memory stays bounded whatever the input's length. Default 4 GiB;
    DATAFRAME_STREAM_COLLECT_BYTES overrides it, so tests can exercise the
    bound on small inputs."""
    var setting = getenv("DATAFRAME_STREAM_COLLECT_BYTES")
    if setting.byte_length() > 0:
        try:
            return max(1, Int(setting))
        except:
            pass
    return 4 << 30


def _frame_bytes(frame: DataFrame) -> Int:
    """Approximate bytes a frame's columns hold: fixed widths by dtype,
    strings at 16 bytes a row plus their descriptors (an estimate; the
    exact text length of a view-backed window costs a pass over it)."""
    var total = 0
    for column in frame._columns:
        var bits = column.dtype().bit_width()
        if bits > 0:
            total += (frame.height() * bits + 7) // 8
        else:
            total += 32 * frame.height()
    return total


def _held_groups(
    held: List[DataFrame], keys: List[String], limit: Int
) raises -> Tuple[Int, Int]:
    """Distinct key rows among the first `limit` rows of the held batches,
    and how many rows that was."""
    var parts = List[DataFrame](capacity=len(held))
    var rows = 0
    for frame in held:
        if rows >= limit:
            break
        var take = min(frame.height(), limit - rows)
        parts.append(frame.select(keys).slice(0, take))
        rows += take
    var whole = concat(parts).rechunk()
    return (encode_rows(whole._columns, nulls_equal=True).count(), rows)


def _many_groups(groups: Int, rows: Int, held: Int, split_groups: Int) -> Bool:
    """Whether `groups` distinct keys among `rows` sampled of `held` rows
    mean a many-group stream: `split_groups` groups at one row in eight
    starting one, which take eight times `split_groups` rows to show. A
    sample of at least half the rows is read as exact (the stream may
    be about this long: PDS-H q16 at a tenth scale has 2.8K groups in
    11.6K rows, few); a smaller one extrapolates to a quarter of eight
    times `split_groups` rows."""
    if 2 * rows >= held:
        return groups >= split_groups and 8 * groups >= rows
    return 4 * groups >= split_groups and 8 * groups >= rows


def _batch_shows_many_groups(
    frame: DataFrame, keys: List[String], split_groups: Int
) raises -> Bool:
    """A long batch's own judgement from its first 2 * split_groups rows,
    by the stream's rules (`_judge_held`): a batch that shows many groups
    keeps only its frame, since the stream will most likely collect and
    a state built now would be dropped (TPC-DS q39: 10 ms for 95K rows);
    one that shows few builds its state in this job, where a replay later
    would cost a round."""
    var sample = frame.slice(0, min(frame.height(), 2 * split_groups))
    if _key_text_bytes(sample.slice(0, 1024), keys) > 64 * 1024:
        return False
    var groups = encode_rows(
        sample.select(keys)._columns, nulls_equal=True
    ).count()
    return _many_groups(groups, sample.height(), frame.height(), split_groups)


def _judge_held(
    held: List[DataFrame],
    states: List[List[_StreamReduction]],
    keys: List[String],
    held_rows: Int,
    split_groups: Int,
) raises -> Bool:
    """Whether the held batches show many groups (`_many_groups`) with
    keys short enough to collect (`_key_text_bytes`). The held states'
    group counts bound the union's from above: when even that bound is
    few groups, no sample is encoded."""
    if _key_text_bytes(held[0].slice(0, 1024), keys) > 64 * 1024:
        return False
    # One batch's state showing many groups on its own settles it.
    for h in range(len(states)):
        if len(states[h]) == 1 and _many_groups(
            states[h][0].group_count(),
            held[h].height(),
            held[h].height(),
            split_groups,
        ):
            return True
    var groups = _held_group_bound(states)
    var rows = held_rows
    if not (_all_reduced(states) and 8 * groups < rows):
        var sampled = _held_groups(held, keys, 2 * split_groups)
        groups = sampled[0]
        rows = sampled[1]
    return _many_groups(groups, rows, held_rows, split_groups)


def _held_group_bound(states: List[List[_StreamReduction]]) -> Int:
    """The held states' group counts summed: at least the distinct groups
    among their rows."""
    var total = 0
    for h in range(len(states)):
        for k in range(len(states[h])):
            total += states[h][k].group_count()
    return total


def _all_reduced(states: List[List[_StreamReduction]]) -> Bool:
    for h in range(len(states)):
        if len(states[h]) == 0:
            return False
    return True


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


def _joined(parts: List[DataFrame]) raises -> DataFrame:
    if len(parts) == 1:
        return parts[0].copy()
    return concat(parts).rechunk()


def _key_text_bytes(frame: DataFrame, keys: List[String]) raises -> Int:
    """Bytes of string text among a batch's key columns. Composite keys
    with long strings keep the batch states: the eager bucket encode
    confirms such keys row by row through scattered reads (#486), and
    PDS-H q10 (seven keys, c_comment among them, about 140 text bytes a
    row) was 22% slower collected, where TPC-DS q39 (one short name
    beside three integers) gains 40%."""
    var total = 0
    for name in keys:
        var column = frame.column(name)
        if column.dtype().bit_width() == 0:
            total += column._text_bytes()
    return total


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

    def take_reduced(mut self) -> List[_StreamReduction]:
        var out = self.reduced^
        self.reduced = List[_StreamReduction]()
        return out^

    def run(mut self) raises:
        if len(self.decode):
            self.decode[0].run()
            self.frame = self.decode.pop().into_frame()
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
        if self.collect_only or (
            self.keep_frame
            and self.frame.height() >= 8 * self.split_groups
            and _batch_shows_many_groups(
                self.frame, self.keys, self.split_groups
            )
        ):
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
    var exprs = List[Expr]()
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


def _pipeline_steps(
    operations: List[PlanNode], indexes: List[Optional[PreparedHashIndex]]
) -> Bool:
    """Whether every operation is a step a pipeline runs: a row-local
    step, or an inner, left, semi or anti join whose build index was
    prepared (its morsels probe the shared index on their own worker)."""
    for node in operations:
        if (
            node.kind == FILTER
            or node.kind == SELECT
            or node.kind == WITH_COLUMNS
        ):
            if not _row_local(node.exprs):
                return False
        elif node.kind == JOIN:
            if node.how not in [JOIN_INNER, JOIN_LEFT, JOIN_SEMI, JOIN_ANTI]:
                return False
            if node.offset >= len(indexes) or not indexes[node.offset]:
                return False
        elif node.kind != DROP:
            return False
    return True


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
                node.offset,
                node.right_keys.copy(),
                node.how,
                node.names2[0] if len(node.names2) > 0 else String(""),
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
    if node.kind == JOIN and indexes[][node.offset]:
        frame = frame._join_impl(
            joins[][node.offset],
            left_on=node.names,
            right_on=node.right_keys,
            how=node.how,
            suffix=node.names2[0],
            coalesce=node.coalesce,
            prepared=indexes[][node.offset],
        )
    elif node.kind == JOIN and node.how == JOIN_CROSS:
        frame = frame.join(
            joins[][node.offset],
            how=node.text,
            suffix=node.names2[0],
        )
    elif node.kind == JOIN:
        frame = frame.join(
            joins[][node.offset],
            left_on=node.names,
            right_on=node.right_keys,
            how=node.text,
            suffix=node.names2[0],
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
                List[Bool](node.flags[:n]),
                List[Bool](node.flags[n:]),
                node.length,
                threads=1,
            )
        )


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


struct LazyFrame(Copyable):
    """A deferred query; build it with DataFrame.lazy() or scan_csv()."""

    var _nodes: List[PlanNode]
    var _frames: List[DataFrame]
    var _schemas: List[Optional[CsvSchema]]
    # Observed execution counters (#439), attached by `profile` and shared
    # by every copy of the plan made while it runs; None otherwise.
    var _report: Optional[ArcPointer[ExecutionReport]]

    def __init__(out self, frame: DataFrame):
        self._frames = [frame.copy()]
        self._schemas = [Optional[CsvSchema]()]
        self._nodes = [_plan_node(SCAN_FRAME, offset=0)]
        self._report = None

    def __init__(
        out self,
        var nodes: List[PlanNode],
        var frames: List[DataFrame],
        var schemas: List[Optional[CsvSchema]],
    ):
        self._nodes = nodes^
        self._frames = frames^
        self._schemas = schemas^
        self._report = None

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
        """Add to the attached report, if any."""
        if not self._report:
            return
        var report = self._report.value()
        report[].record(
            node,
            self._label(node),
            executor,
            input_rows,
            output_rows,
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
        node.left = len(result._nodes) - 1
        result._nodes.append(node^)
        return result^

    def filter(self, predicate: Expr) -> Self:
        return self._push(_plan_node(FILTER, exprs=[predicate.copy()]))

    def select(self, expr: Expr) -> Self:
        return self.select_exprs([expr.copy()])

    def select(self, names: List[String]) -> Self:
        var exprs = List[Expr]()
        for name in names:
            exprs.append(col(name))
        return self.select_exprs(exprs)

    def select_exprs(self, exprs: List[Expr]) -> Self:
        return self._push(_plan_node(SELECT, exprs=exprs))

    def with_columns(self, expr: Expr) -> Self:
        return self.with_columns([expr.copy()])

    def with_columns(self, exprs: List[Expr]) -> Self:
        return self._push(_plan_node(WITH_COLUMNS, exprs=exprs))

    def group_by(
        self, keys: List[String], *, maintain_order: Bool = False
    ) -> LazyGroupBy:
        return LazyGroupBy(self.copy(), keys.copy(), maintain_order)

    def group_by(
        self, key: String, *, maintain_order: Bool = False
    ) -> LazyGroupBy:
        return LazyGroupBy(self.copy(), [key], maintain_order)

    def sort(
        self,
        by: List[String],
        descending: Bool = False,
        nulls_last: Bool = True,
    ) -> Self:
        var n = len(by)
        return self._push(
            _plan_node(
                SORT,
                names=by,
                flags=List[Bool](length=n, fill=descending)
                + List[Bool](length=n, fill=nulls_last),
            )
        )

    def sort(
        self,
        by: List[String],
        *,
        descending: List[Bool],
        nulls_last: List[Bool],
    ) raises -> Self:
        """Sort with a direction and null placement per key, as the eager
        DataFrame.sort takes them."""
        if len(descending) != len(by) or len(nulls_last) != len(by):
            raise Error("sort needs one descending and nulls_last flag per key")
        return self._push(
            _plan_node(SORT, names=by, flags=descending + nulls_last.copy())
        )

    def sort(self, by: String, descending: Bool = False) -> Self:
        return self.sort([by], descending)

    def slice(self, offset: Int, length: Int = -1) -> Self:
        return self._push(_plan_node(SLICE, offset=offset, length=length))

    def head(self, n: Int = 5) -> Self:
        return self.slice(0, n)

    def limit(self, n: Int = 5) -> Self:
        return self.head(n)

    def unique(
        self,
        subset: List[String] = List[String](),
        *,
        keep: String = "any",
        maintain_order: Bool = False,
    ) -> Self:
        return self._push(
            _plan_node(
                UNIQUE, names=subset, text=keep, maintain_order=maintain_order
            )
        )

    def drop(self, names: List[String]) -> Self:
        return self._push(_plan_node(DROP, names=names))

    def explode(self, columns: List[String]) -> Self:
        """One row per list element; see DataFrame.explode."""
        return self._push(_plan_node(EXPLODE, names=columns))

    def explode(self, column: String) -> Self:
        return self.explode([column])

    def unnest(self, column: String) -> Self:
        """Replace a struct column with its fields; see DataFrame.unnest."""
        return self._push(_plan_node(UNNEST, text=column))

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
        ref node = result._nodes[len(result._nodes) - 1]
        node.kind = ASOF
        node.text = strategy
        node.flags = [allow_exact_matches]
        node.asof_tolerance = tolerance.copy()
        for name in by:
            node.names2.append(name)
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
        var left_root = len(result._nodes) - 1
        var shift = len(result._nodes)
        var frame_shift = len(result._frames)
        for frame in other._frames:
            result._frames.append(frame.copy())
        for schema in other._schemas:
            result._schemas.append(schema.copy())
        for node in other._nodes:
            var copied = node.copy()
            if copied.left >= 0:
                copied.left += shift
            if copied.right >= 0:
                copied.right += shift
            if _is_scan(copied.kind):
                copied.offset += frame_shift
            result._nodes.append(copied^)
        var join = _plan_node(
            JOIN,
            left_root,
            len(result._nodes) - 1,
            names=left_on,
            text=how,
            how=join_code(how),
            right_keys=right_on,
            coalesce=coalesce,
        )
        join.names2 = [suffix]
        result._nodes.append(join^)
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
                self, optimize, streaming, batch_size, False
            )[0].copy()
        if engine == "accel":
            return _execute_accel(self)[0].copy()
        var crew = Crew.start()
        try:
            var result = self._collect(
                optimize=optimize, streaming=streaming, batch_size=batch_size
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

        The provider lowers the logical plan independently of CPU streaming
        and optimization. Unsupported plans raise before device submission.
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
                    self, optimize, streaming, batch_size, False
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
        return accelerator.execute(self)[0].copy()

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
                    self, optimize, streaming, batch_size, True
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
        return accelerator.execute_profiled(self)

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
        return accelerator.describe(self)

    def fetch[
        B: AcceleratorBackend
    ](
        self, n: Int = 5, *, accelerator: B, engine: String = "accel"
    ) raises -> DataFrame:
        """Collect the first n rows using an explicit provider."""
        return self.head(n).collect(accelerator=accelerator, engine=engine)

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
                    file=FileDescriptor(2),
                )
            return profiled[0].copy()
        var plan = self._optimized() if optimize else self.copy()
        if optimize:
            plan._push_join_key_sets(streaming, batch_size)
            plan._share_subplans(streaming, batch_size)
            plan._order_joins(streaming, batch_size)
        return plan._execute(len(plan._nodes) - 1, False, streaming, batch_size)

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
                self, optimize, streaming, batch_size, True
            )
            return (
                result[0].copy(),
                _selection_report(result[1], backend.reason),
            )
        if engine == "accel":
            return _profile_accel(self)
        var plan = self._optimized() if optimize else self.copy()
        if optimize:
            plan._push_join_key_sets(streaming, batch_size)
            plan._share_subplans(streaming, batch_size)
            plan._order_joins(streaming, batch_size)
        plan._report = Optional(ArcPointer(ExecutionReport()))
        var result = plan._execute(
            len(plan._nodes) - 1, False, streaming, batch_size
        )
        var report = plan._report.value()[].frame()
        if engine == "auto":
            report = _selection_report(report, backend.reason)
        return (result^, report^)

    def fetch(self, n: Int = 5, *, engine: String = "auto") raises -> DataFrame:
        """Collect only the first n rows of the result."""
        return self.head(n).collect(engine=engine)

    def collect_schema(self) raises -> List[String]:
        """Output names and dtypes as "name: dtype", computed without reading
        rows: every scan yields zero rows, then the plan runs as usual, so
        binding validates each expression exactly as collect would."""
        var plan = self._optimized()
        var frame = plan._execute(len(plan._nodes) - 1, True)
        var out = List[String]()
        for field in frame.schema():
            out.append(field.name + ": " + field.dtype.name())
        return out^

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
            return _describe_accel(self)
        if engine == "auto" and backend.engine != "cpu":
            return backend.describe()
        var plan = self._optimized() if optimize else self.copy()
        var out = String()
        if engine != "cpu":
            out = backend.describe()
        plan._describe(len(plan._nodes) - 1, 0, out, streaming)
        return out^

    # --- execution -----------------------------------------------------

    def _sampled_groups(self, keys: List[String], frame: Int) raises -> Int:
        """Distinct grouping keys among 4,096 evenly spaced rows of an
        in-memory frame, or -1 when the frame does not hold every key (one
        made by an earlier step).
        """
        ref source = self._frames[frame]
        var height = source.height()
        var sample = min(height, 4096)
        if sample == 0:
            return -1
        var rows = List[Int](capacity=sample)
        for k in range(sample):
            rows.append(k * height // sample)
        var picked = List[Series](capacity=len(keys))
        for name in keys:
            if name not in source.columns():
                return -1
            picked.append(source.column(name).take(rows))
        return encode_rows(picked, nulls_equal=True).count()

    def _joins_below(self, index: Int) -> Bool:
        """Whether a join sits under `index` along its left inputs, through
        the row-local steps a stream applies."""
        var cursor = index
        while cursor >= 0:
            ref node = self._nodes[cursor]
            if node.kind == JOIN:
                return True
            if _is_scan(node.kind):
                return False
            cursor = node.left
        return False

    def _height_bound(self, index: Int) -> Int:
        """An upper bound on the rows a node yields without executing it:
        the scanned frame's height through steps that only drop or keep
        rows; -1 when unknown (joins, aggregations, file scans)."""
        ref node = self._nodes[index]
        if node.kind == SCAN_FRAME:
            return self._frames[node.offset].height()
        if (
            node.kind == FILTER
            or node.kind == DROP
            or (
                (node.kind == SELECT or node.kind == WITH_COLUMNS)
                and _stream_rows(node)
            )
        ):
            return self._height_bound(node.left)
        if node.kind == SLICE:
            var height = self._height_bound(node.left)
            if height >= 0 and node.length >= 0:
                return min(height, node.length)
            return height
        return -1

    def _known_height(self, index: Int) -> Int:
        """Exact cheap cardinalities only; -1 means execution is required."""
        ref node = self._nodes[index]
        if node.kind == SCAN_FRAME:
            return self._frames[node.offset].height()
        if (
            node.kind == DROP
            or node.kind == UNNEST
            or (
                (node.kind == SELECT or node.kind == WITH_COLUMNS)
                and _stream_rows(node)
            )
        ):
            return self._known_height(node.left)
        if node.kind == SLICE and node.offset >= 0:
            var height = self._known_height(node.left)
            if height >= 0:
                height = max(0, height - node.offset)
                return min(height, node.length) if node.length >= 0 else height
        return -1

    def _stream_execute(
        self, index: Int, batch_size: Int
    ) raises -> Optional[DataFrame]:
        var cursor = index
        var expressions = List[Expr]()
        var keys = List[String]()
        var ordered = False
        var skip = 0
        var limit = -1
        ref terminal = self._nodes[index]
        if terminal.kind == AGG or terminal.kind == SELECT:
            if _stream_reductions(terminal.exprs):
                expressions = terminal.exprs.copy()
                if terminal.kind == AGG:
                    keys = terminal.names.copy()
                    ordered = terminal.maintain_order
                    if len(keys) == 0:
                        return None
                cursor = terminal.left
        if terminal.kind == SLICE:
            if terminal.offset < 0:
                return None
            skip = terminal.offset
            limit = terminal.length
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
            and self._nodes[cursor].length >= 0
        ):
            top = self._nodes[cursor].length
            top_node = cursor
            top_names = self._nodes[cursor].names.copy()
            var n = len(top_names)
            for k in range(n):
                top_descending.append(self._nodes[cursor].flags[k])
                top_nulls_last.append(self._nodes[cursor].flags[n + k])
            cursor = self._nodes[cursor].left
        var candidates = List[DataFrame]()
        var candidate_rows = 0
        var operations = List[PlanNode]()
        # The plan node of each operation, for the execution report.
        var operation_nodes = List[Int]()
        var joins = List[DataFrame]()
        var indexes = List[Optional[PreparedHashIndex]]()
        while cursor >= 0:
            ref node = self._nodes[cursor]
            if _stream_rows(node):
                operations.append(node.copy())
                operation_nodes.append(cursor)
            elif node.kind == JOIN and node.how in [
                JOIN_INNER,
                JOIN_LEFT,
                JOIN_SEMI,
                JOIN_ANTI,
                JOIN_CROSS,
            ]:
                var prepared = Optional[PreparedHashIndex]()
                # A left input that can only be small next to a large known
                # right may be the side to hash: decline streaming, so the
                # plan runs eagerly and `DataFrame.join` builds on the smaller
                # side, as Polars' `det_hash_prone_order` and DuckDB's
                # build/probe optimizer do (#377). Streaming would build on
                # the right: PDS-H q17 hashed 6M lineitem rows to match the
                # few hundred parts its filter keeps.
                var bound = self._height_bound(node.left)
                var right_rows = self._known_height(node.right)
                # A left input made by another join has no bound; how many
                # rows it holds is known only once it has run.
                var unbounded = bound < 0 and self._joins_below(node.left)
                if (
                    (node.how == JOIN_INNER or node.how == JOIN_LEFT)
                    and self._known_height(node.left) < 0
                    and right_rows >= 524288
                    and (unbounded or (bound >= 0 and 4 * bound <= right_rows))
                ):
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
                    self._known_height(node.left) <= batch_size
                    and (node.how == JOIN_INNER or node.how == JOIN_LEFT)
                    and prefer_left_build(
                        self._known_height(node.left),
                        self._known_height(node.right),
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
                    if (
                        build_node.kind == SCAN_FRAME
                        and len(node.right_keys) == 1
                    ):
                        var sources: List[Series] = [
                            self._frames[build_node.offset][
                                node.right_keys[0]
                            ].copy()
                        ]
                        prepared = prepare_progression_index(sources)
                    if not prepared:
                        break
                var operation = node.copy()
                operation.offset = len(joins)
                var algorithm = String(
                    "progression" if prepared else (
                        "cross" if node.how == JOIN_CROSS else "eager_hash"
                    )
                )
                var built = self._execute(node.right, False, True, batch_size)
                if (
                    (node.how == JOIN_SEMI or node.how == JOIN_ANTI)
                    and built.height() >= 524288
                    and self._known_height(node.left) < 0
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
                        plan._nodes[cursor] = _plan_node(
                            SCAN_FRAME, offset=len(plan._frames) - 1
                        )
                    else:
                        plan._frames.append(left_frame^)
                        plan._schemas.append(Optional[CsvSchema]())
                        plan._nodes.append(
                            _plan_node(SCAN_FRAME, offset=len(plan._frames) - 1)
                        )
                        plan._nodes[cursor].left = len(plan._nodes) - 1
                        plan._frames.append(built^)
                        plan._schemas.append(Optional[CsvSchema]())
                        plan._nodes.append(
                            _plan_node(SCAN_FRAME, offset=len(plan._frames) - 1)
                        )
                        plan._nodes[cursor].right = len(plan._nodes) - 1
                    return plan._execute(index, False, True, batch_size)
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
                        prepared = prepare_hash_index(sources)
                        if prepared:
                            algorithm = "hash_index"
                self._record(
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
        if (
            cursor < 0
            or cursor == index
            and not _is_scan(self._nodes[cursor].kind)
        ):
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
        if (
            len(keys) > 0
            and len(expressions) > 0
            and len(joins) > 0
            and not ordered
            and top < 0
            and self._nodes[cursor].kind == SCAN_FRAME
        ):
            var source_columns = self._frames[
                self._nodes[cursor].offset
            ].columns()
            for name in keys:
                var b = _late_key_build(
                    name, source_columns, joins, operations, expressions
                )
                if b < 0:
                    continue
                late_names.append(name)
                late_builds.append(b)
        # Row-local steps and a reduction over an in-memory frame run as a
        # thread-owned pipeline (#538), grouped or not, unless the
        # reduction counts distinct values (per-group sets, which the
        # eager partitioned group-by handles on every worker, #336).
        # Joins run in the pipeline as probe steps; a grouped sink that
        # must keep first-occurrence order does not take them, since a
        # morsel's output rows carry no unique position in the join's
        # left-major order.
        var piped = (
            self._nodes[cursor].kind == SCAN_FRAME
            and (top < 0 and limit < 0 and skip == 0 or top > 0)
            and _pipeline_steps(operations, indexes)
            and not _counts_distinct(expressions)
            and (len(joins) == 0 or len(keys) == 0 or not ordered)
        )
        # A grouped sink either inserts rows into per-worker group tables
        # (few groups among 4,096 sampled key rows, fixed-width keys) or
        # keeps the morsels' rows for the eager partitioned group-by at
        # the finish. The finish gathers kept rows once over the whole
        # frame; strings gathered that way cost more than the eager
        # route's per-chunk filter (ClickBench q12: 85 ms against 48), so
        # with steps, a collecting plan is piped only when every column
        # it groups or reduces is fixed-width.
        var insert = False
        if piped and len(keys) > 0:
            var frame = self._nodes[cursor].offset
            ref source_frame = self._frames[frame]
            var fixed = True
            var used = keys.copy()
            for expression in expressions:
                var reads = _references(expression)
                if not reads:
                    fixed = False
                    break
                for name in reads.value():
                    if name not in used:
                        used.append(name)
            for name in used:
                if not fixed:
                    break
                if name in late_names:
                    # Grouped by codes if the plan inserts (below).
                    continue
                var dtype = DataType.INT64
                if name in source_frame.columns():
                    dtype = source_frame.column(name).dtype()
                else:
                    # A join's build column, or made by a step (then its
                    # dtype is not known here).
                    var found = False
                    for build in joins:
                        if name in build.columns():
                            dtype = build.column(name).dtype()
                            found = True
                            break
                    if not found:
                        continue
                if (
                    dtype.is_nested()
                    or dtype.is_decimal()
                    or dtype.physical() == DataType.STRING
                ):
                    fixed = False
            if not fixed and len(operations) > 0:
                piped = False
            elif fixed and not _order_sensitive(expressions):
                # Sampled only when the mode is still open: the sample
                # takes and encodes 4,096 key rows (a millisecond on
                # strings, an eighth of ClickBench q38).
                var sample = min(source_frame.height(), 4096)
                # A key from a join's build side is not in the source to
                # sample (and the source's key columns are not taken).
                var sampled = -1 if len(late_names) > 0 else (
                    self._sampled_groups(keys, frame)
                )
                if sampled >= 0:
                    insert = 4 * sampled <= sample
                else:
                    # The groups' bound from where each key is stored,
                    # held to the sampled rule's density: at most a
                    # quarter of the sample's rows.
                    var bound = _key_group_bound(
                        keys, source_frame, joins, sample // 4
                    )
                    insert = bound >= 0
        if len(late_names) > 0 and not (piped and insert):
            # Coded keys pay only in the insert tables: collecting them
            # for the finish's group-by ran slower than the eager route
            # their strings take (TPC-DS q45 5.9 -> 13.4 ms, q24 40 -> 81),
            # so those plans take the strings' route, as before.
            if len(operations) > 0:
                piped = False
            late_names = List[String]()
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
                Series(name, Column[Int64](codes^, valid^))
            )
            late_values.append(values^)
            late_rows.append(groups.representatives.copy())
        # Grouped aggregations over an in-memory frame, filtered or
        # projected at most: whether the eager group-by is faster.
        var eager = False
        if (
            not piped
            and len(expressions)
            and self._nodes[cursor].kind == SCAN_FRAME
            and _row_steps_only(operations)
        ):
            if _counts_distinct(expressions):
                eager = True
            elif len(keys) > 0:
                if _filters(operations):
                    eager = True
                else:
                    var frame = self._nodes[cursor].offset
                    var sampled = self._sampled_groups(keys, frame)
                    var sample = min(self._frames[frame].height(), 4096)
                    eager = (
                        sampled < 0 or sampled <= 192 or 2 * sampled > sample
                    )
        if eager:
            # Over a frame already in memory, filtered or projected at most,
            # batches save no memory, and most grouped aggregations run
            # faster eagerly, measured on 10M ClickBench rows (#381):
            # - n_unique keeps per-group sets in batch states that the eager
            #   aggregation replaces with hash partitions on every worker
            #   (#336).
            # - After a filter, the eager group-by only sees the kept rows,
            #   where every batch state holds and merges its groups again:
            #   non-empty SearchPhrase counts take 66 ms against 358
            #   streamed, and q39's five keys 99 against 371.
            # - Keys made by an earlier step, which the sample cannot read:
            #   q18's (UserID, minute, SearchPhrase) take 354 ms against
            #   1,645 streamed.
            # - Few groups (at most 192 in the sample) take 33 ms against
            #   64; ResolutionWidth (134 in the sample) 35 against 67.
            # - Many groups (more than half the sample) are encoded once by
            #   hash partition, where every batch state would hold most of
            #   them: on 1M rows URL takes 40 ms against 63 streamed.
            # Unfiltered keys in between keep streaming, which is faster
            # there: RegionID (364 in the sample) takes 75 ms against 99,
            # and SearchPhrase (574, one dominant value) 353 against 375,
            # or on 1M rows (212) 8 against 13.
            # Ungrouped reductions stream, as do plans with a join, whose
            # batched probes beat the eager join.
            return None
        if top >= 0:
            # Over a frame already in memory with nothing to apply first,
            # batches save no memory, and the eager selection already works
            # in cache-sized chunks on every worker.
            if len(operations) == 0 and self._nodes[cursor].kind == SCAN_FRAME:
                return None
            operations.append(self._nodes[top_node].copy())
            operation_nodes.append(top_node)
        var shared_joins = ArcPointer(joins^)
        var shared_indexes = ArcPointer(indexes^)
        ref source = self._nodes[cursor]
        var csv = List[_CsvBatches]()
        var parquet = List[_ParquetBatches]()
        var input = DataFrame(List[Series](), height=0)
        if source.kind == SCAN_CSV:
            var mapping = _map_file(source.text)
            if mapping.address == 0:
                return None
            csv.append(
                _CsvBatches(
                    mapping^,
                    self._schemas[source.offset],
                    source.names,
                    source.length,
                    batch_size,
                )
            )
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
        elif source.kind == SCAN_FRAME:
            source._record_execution()
            input = self._frames[source.offset].copy()
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
                var window = sorted_window(
                    input._columns, input.height(), parts
                )
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
        # up to four times the default: each batch probes every join, and
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
        if len(shared_joins[]) > 0 and input.height() > 0:
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
        if piped and len(csv) == 0 and len(parquet) == 0:
            var began = Int(perf_counter_ns())
            var piped_counts = List[Int]()
            # A top-k's TOP_K node was appended to the operations for the
            # stream's per-batch selection; the pipeline's sink keeps the
            # k best itself, so the steps stop before it.
            var step_count = len(operations) - 1 if top > 0 else len(operations)
            var step_nodes = List[PlanNode](capacity=step_count)
            for k in range(step_count):
                step_nodes.append(operations[k].copy())
            var piped_result = run_pipeline(
                input,
                _steps_of(step_nodes),
                shared_joins,
                shared_indexes,
                expressions,
                keys,
                ordered,
                insert,
                workers,
                rows_per_batch,
                batch_size,
                piped_counts,
                top=top,
                top_names=top_names,
                top_descending=top_descending,
                top_nulls_last=top_nulls_last,
                bounded=bounded,
            )
            if top > 0:
                piped_result = piped_result.slice(skip, limit)
            for i in range(len(late_names)):
                # The groups' codes back to their values (a code is a
                # representative build row; a left join's missing row is a
                # null code and stays null).
                var coded = piped_result.column(late_names[i]).copy()
                var rows = List[Int](capacity=len(coded))
                for r in range(len(coded)):
                    var cell = coded.get(r)
                    if cell.is_null():
                        rows.append(-1)
                    else:
                        rows.append(late_rows[i][Int(cell.int64())])
                piped_result = piped_result.with_column(
                    late_values[i].take_or_null(rows).renamed(late_names[i])
                )
            if counting:
                self._record(cursor, "pipeline", 0, piped_counts[0])
                var into_terminal = piped_counts[0]
                for k in range(step_count):
                    self._record(
                        operation_nodes[k],
                        "pipeline",
                        piped_counts[2 * k + 1],
                        piped_counts[2 * k + 2],
                        busy_ns=piped_counts[2 * step_count + 1 + k],
                    )
                    into_terminal = piped_counts[2 * k + 2]
                if top > 0:
                    self._record(
                        top_node,
                        "pipeline",
                        into_terminal,
                        piped_result.height(),
                    )
                var wall = Int(perf_counter_ns()) - began
                if index != cursor and (index not in operation_nodes):
                    self._record(
                        index,
                        "pipeline",
                        into_terminal,
                        piped_result.height(),
                        wall_ns=wall,
                    )
                else:
                    self._record(
                        index, "pipeline", 0, 0, executions=0, wall_ns=wall
                    )
            return piped_result^
        # An in-memory input hands each round four batches a worker, which
        # the pool claims as workers free up, so uneven batches balance.
        # Batches decoded from a file are held until their round ends, so
        # those keep one a worker, as does a top-k (ClickBench q26 lost its
        # bound for a first round four times as long).
        var per_round = workers
        if len(csv) == 0 and len(parquet) == 0 and top < 0:
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
                if len(csv):
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
                            held, held_states, keys, held_rows, split_groups
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
                    outputs.append(part^)
            if collecting and collected_bytes >= collect_budget:
                # The budget is reached: reduce what is collected into one
                # state (every key encoded once) and keep collecting. The
                # states merge at the end as batch states do.
                var part = _StreamReduction(
                    concat(collected), expressions, keys
                )
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
                    bound = _TopBound(
                        merged, top_names[0], top_descending[0], top
                    )
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
                    held, held_states, keys, held_rows, split_groups
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
        var result = Optional[DataFrame]()
        if collecting and len(acc.states) == 0:
            trace_path("lazy.collect_group_by")
            # Contiguous first: the bucket encode reads string keys in
            # view storage through a buffer table, slow across hundreds
            # of batches (TPC-DS q39: 10 ms of 40 on 358 batches).
            var whole = concat(collected).rechunk()
            # Groups come back in first-occurrence order only when asked,
            # as the eager group-by's do.
            result = whole.group_by(keys, maintain_order=ordered).agg(
                expressions
            )
        elif collecting:
            # Spilled at least once: the rest joins the states.
            trace_path("lazy.collect_group_by_bounded")
            if len(collected):
                var part = _StreamReduction(
                    concat(collected), expressions, keys
                )
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
        if counting:
            if _is_scan(source.kind):
                self._record(cursor, "streaming", 0, fed_rows)
            var into_terminal = fed_rows
            for k in range(len(operations)):
                self._record(
                    operation_nodes[k],
                    "streaming",
                    operation_rows[2 * k],
                    operation_rows[2 * k + 1],
                    busy_ns=operation_times[k],
                )
                into_terminal = operation_rows[2 * k + 1]
            var stream_wall = Int(perf_counter_ns()) - stream_began
            if index != cursor and (index not in operation_nodes):
                self._record(
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
                self._record(
                    index, "streaming", 0, 0, executions=0, wall_ns=stream_wall
                )
        return result^

    def _execute(
        self,
        index: Int,
        empty: Bool,
        streaming: Bool = False,
        batch_size: Int = 65536,
    ) raises -> DataFrame:
        if streaming and not empty:
            var streamed = self._stream_execute(index, batch_size)
            if streamed:
                return streamed.take()
        if not self._report or empty:
            return self._execute_node(index, empty, streaming, batch_size)
        # Record what the node did: its inputs record themselves as they
        # run below it, so the input rows are read back from the report.
        var began = Int(perf_counter_ns())
        var result = self._execute_node(index, empty, streaming, batch_size)
        var took = Int(perf_counter_ns()) - began
        ref node = self._nodes[index]
        if _is_scan(node.kind):
            self._record(index, "eager", 0, result.height(), wall_ns=took)
        elif node.kind == JOIN:
            var left = self._report.value()[].find(node.left)
            var right = self._report.value()[].find(node.right)
            var left_rows = left.value().output_rows if left else -1
            var right_rows = right.value().output_rows if right else -1
            var build_left = (
                node.how == JOIN_INNER or node.how == JOIN_LEFT
            ) and prefer_left_build(left_rows, right_rows)
            self._record(
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
            self._record(
                index,
                "eager",
                left.value().output_rows if left else -1,
                result.height(),
                wall_ns=took,
            )
        return result^

    def _execute_node(
        self,
        index: Int,
        empty: Bool,
        streaming: Bool,
        batch_size: Int,
    ) raises -> DataFrame:
        ref node = self._nodes[index]
        if node.kind == SCAN_FRAME:
            if not empty:
                node._record_execution()
            var frame = self._frames[node.offset].copy()
            if len(node.names) > 0:
                frame = frame.select(node.names)
            return frame.clear() if empty else frame^
        if node.kind == SCAN_CSV:
            if empty:
                # Metadata probes must not use eager n_rows=0: that API
                # intentionally decodes a source chunk before truncation.
                var mapping = _map_file(node.text)
                if mapping.address != 0:
                    var batches = _CsvBatches(
                        mapping^,
                        self._schemas[node.offset],
                        node.names,
                        0,
                        batch_size,
                    )
                    var next = batches.next()
                    var job = next.take()
                    job.run()
                    return job^.into_frame()
            var rows = 0 if empty else node.length
            if self._schemas[node.offset]:
                return read_csv(
                    node.text,
                    self._schemas[node.offset].value(),
                    n_rows=rows,
                    columns=node.names,
                )
            return read_csv(node.text, n_rows=rows, columns=node.names)
        if node.kind == SCAN_PARQUET:
            if empty:
                return read_parquet(
                    node.text, columns=node.names, row_groups=List[Int]()
                )
            return read_parquet(node.text, columns=node.names)
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
                    var n = len(sorted.names)
                    var descending = List[Bool]()
                    var nulls_last = List[Bool]()
                    for i in range(n):
                        descending.append(sorted.flags[i])
                        nulls_last.append(sorted.flags[n + i])
                    var order = source._arg_sort_head(
                        sorted.names, descending, nulls_last, sorted.length
                    ) if sorted.length >= 0 else source.arg_sort(
                        sorted.names,
                        descending=descending,
                        nulls_last=nulls_last,
                    )
                    return source.select_exprs(node.exprs).take(order^)
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
                    source.text, columns=source.names, row_groups=groups
                ).filter(node.exprs[0])
            if source.kind == SCAN_CSV and source.length < 0:
                if self._schemas[source.offset]:
                    return read_csv_explicit(
                        source.text,
                        self._schemas[source.offset].value(),
                        columns=source.names,
                        predicate=Optional(node.exprs[0].copy()),
                    )
                return read_csv_inferred(
                    source.text,
                    columns=source.names,
                    predicate=Optional(node.exprs[0].copy()),
                )
        var input = self._execute(node.left, empty, streaming, batch_size)
        if node.kind == FILTER:
            return input.filter(node.exprs[0])
        if node.kind == SELECT:
            return input.select_exprs(node.exprs)
        if node.kind == WITH_COLUMNS:
            return input.with_columns(node.exprs)
        if node.kind == AGG:
            return input.group_by(
                node.names, maintain_order=node.maintain_order
            ).agg(node.exprs)
        if node.kind == ASOF:
            var right = self._execute(node.right, empty, streaming, batch_size)
            var groups = List[String]()
            for i in range(1, len(node.names2)):
                groups.append(node.names2[i])
            return input.join_asof(
                right,
                left_on=node.names[0],
                right_on=node.right_keys[0],
                by=groups,
                strategy=node.text,
                tolerance=node.asof_tolerance,
                suffix=node.names2[0],
                allow_exact_matches=node.flags[0],
            )
        if node.kind == JOIN:
            var right = self._execute(node.right, empty, streaming, batch_size)
            if node.how == JOIN_CROSS:
                return input.join(right, how=node.text, suffix=node.names2[0])
            # Consumers that ignore row order let the join skip restoring
            # left-major order after probing.
            var free = self._order_free_above(
                index, self._live_parents(len(self._nodes) - 1), True
            )
            return input._join_impl(
                right,
                left_on=node.names,
                right_on=node.right_keys,
                # Parsed here, as `DataFrame.join` would: an unknown type
                # raises.
                how=join_type(node.text),
                suffix=node.names2[0],
                coalesce=node.coalesce,
                keep_order=not free,
            )
        if node.kind == SORT:
            var n = len(node.names)
            var descending = List[Bool]()
            var nulls_last = List[Bool]()
            for i in range(n):
                descending.append(node.flags[i])
                nulls_last.append(node.flags[n + i])
            if node.length >= 0:
                return input.take(
                    input._arg_sort_head(
                        node.names, descending, nulls_last, node.length
                    )
                )
            return input.sort(
                node.names, descending=descending, nulls_last=nulls_last
            )
        if node.kind == SLICE:
            return input.slice(node.offset, node.length)
        if node.kind == UNIQUE:
            return input.unique(
                node.names, keep=node.text, maintain_order=node.maintain_order
            )
        if node.kind == EXPLODE:
            return input.explode(node.names)
        if node.kind == UNNEST:
            return input.unnest(node.text)
        return input.drop(node.names)

    # --- optimization --------------------------------------------------

    def _optimized(self) raises -> Self:
        var plan = self.copy()
        plan._split_filters()
        plan._push_predicates()
        plan._merge_filters()
        plan._push_slices()
        plan._fuse_top_k()
        plan._push_projections()
        return plan^

    def _share_subplans(mut self, streaming: Bool, batch_size: Int) raises:
        """Execute a subplan used in two places once (common-subplan
        elimination, as Polars' `comm_subplan_elim` and DuckDB's shared
        CTEs do). Two nodes whose subtrees read the same (`_signature`)
        would compute the same frame twice: PDS-H q21 filters lineitem
        for late shipments both as a join input and under a grouped
        n_unique, TPC-DS q39 joins and groups inventory once per month
        compared. The largest such subtree is collected on its own, with
        every planner pass, and every node holding it becomes a scan of
        the result. Runs after join key sets pass sideways: copies that
        receive different key sets are left apart, since one shared
        result could not take either set (TPC-DS q59's weekly sales per
        store are joined with a different year's weeks on each side, and
        shared whole cost 40 to 56 ms).

        Only a subtree holding an aggregation or a unique is shared: its
        result is small next to its input, and holding it whole costs
        little. Joins and filters alone are not, since their output can
        be as large as their input (TPC-DS q39's two branches agree only
        on the 11.7M-row joins below their date filters, and holding
        those doubled the query) and a stream processes them in batches
        in whatever join order the planner picks."""
        # Two plans joined carry their own copies of a frame they both
        # scan, in separate slots: a scan's signature names the first slot
        # holding the same buffers.
        var canonical = List[Int](capacity=len(self._frames))
        for f in range(len(self._frames)):
            var first = f
            for g in range(f):
                if canonical[g] == g and self._frames[g]._shares_buffers_with(
                    self._frames[f]
                ):
                    first = g
                    break
            canonical.append(first)
        var signatures = List[String]()
        var sizes = List[Int]()
        var reducing = List[Bool]()
        for i in range(len(self._nodes)):
            signatures.append(self._signature(i, signatures, canonical))
            ref node = self._nodes[i]
            var size = 1
            var reduces = node.kind == AGG or node.kind == UNIQUE
            if node.left >= 0:
                size += sizes[node.left]
                reduces = reduces or reducing[node.left]
            if node.right >= 0:
                size += sizes[node.right]
                reduces = reduces or reducing[node.right]
            sizes.append(size)
            reducing.append(reduces)
        # Nodes under a shared subtree are replaced with it.
        var covered = List[Bool](length=len(self._nodes), fill=False)
        var shared = False
        while True:
            # The largest subtree that two uncovered nodes hold.
            var best = -1
            for i in range(len(self._nodes)):
                if covered[i] or not reducing[i]:
                    continue
                if best >= 0 and sizes[i] <= sizes[best]:
                    continue
                for j in range(len(self._nodes)):
                    if (
                        j != i
                        and not covered[j]
                        and signatures[j] == signatures[i]
                    ):
                        best = i
                        break
            if best < 0:
                break
            trace_path("lazy.shared_subplan")
            var sub = self.copy()
            sub._reorder(best)
            var frame = sub._collect(
                optimize=True, streaming=streaming, batch_size=batch_size
            )
            self._frames.append(frame^)
            self._schemas.append(Optional[CsvSchema]())
            var slot = len(self._frames) - 1
            for j in range(len(self._nodes)):
                if not covered[j] and signatures[j] == signatures[best]:
                    self._cover(j, covered)
                    self._nodes[j] = _plan_node(SCAN_FRAME, offset=slot)
            shared = True
        if shared:
            self._reorder()
            self._compact_scan_slots()

    def _cover(self, index: Int, mut covered: List[Bool]):
        var stack = List[Int]()
        stack.append(index)
        while len(stack) > 0:
            var i = stack.pop()
            if covered[i]:
                continue
            covered[i] = True
            for child in [self._nodes[i].left, self._nodes[i].right]:
                if child >= 0:
                    stack.append(child)

    def _signature(
        self, index: Int, below: List[String], canonical: List[Int]
    ) -> String:
        """What a node computes, as text: its kind and every setting, its
        expressions node by node, and its inputs' signatures (already in
        `below`, since children come before parents). A frame scan names
        its slot through `canonical`."""
        ref node = self._nodes[index]
        var text = String(node.kind)
        text += "|" + _joined(node.names) + "|" + _joined(node.names2)
        for flag in node.flags:
            text += "1" if flag else "0"
        var slot = node.offset
        if node.kind == SCAN_FRAME and slot >= 0 and slot < len(canonical):
            slot = canonical[slot]
        text += "|" + node.text + "|" + String(slot)
        text += "|" + String(node.length)
        text += "|" + ("o" if node.maintain_order else "u")
        text += "|" + String(node.how) + "|" + _joined(node.right_keys)
        text += "|" + ("c" if node.coalesce else "n")
        if node.asof_tolerance:
            text += "|" + String(node.asof_tolerance.value())
        for e in node.exprs:
            text += "|" + _expr_signature(e)
        # Children by a hash of their text, so a signature stays short
        # however deep the plan.
        text += (
            "|("
            + (String(_text_hash(below[node.left])) if node.left >= 0 else "")
            + ")"
        )
        text += (
            "("
            + (String(_text_hash(below[node.right])) if node.right >= 0 else "")
            + ")"
        )
        return text^

    def _parents(self) -> List[Int]:
        var parents = List[Int](length=len(self._nodes), fill=-1)
        for i in range(len(self._nodes)):
            if self._nodes[i].left >= 0:
                parents[self._nodes[i].left] = i
            if self._nodes[i].right >= 0:
                parents[self._nodes[i].right] = i
        return parents^

    def _columns_of(self, index: Int) raises -> List[String]:
        return self._execute(index, True).columns()

    def _split_filters(mut self):
        """A row-local filter on `a & b` becomes a filter on b over one on
        a, so predicate pushdown moves each part as far as it can go; a
        filter on an OR of ANDs also gets the single-input filters it
        implies (`_implied_filters`). `_merge_filters` rejoins the parts
        that end up together."""
        var count = len(self._nodes)
        var added = False
        for i in range(count):
            if self._nodes[i].kind != FILTER:
                continue
            if not _row_local(self._nodes[i].exprs):
                continue
            var parts = _boolean_parts(self._nodes[i].exprs[0], AND)
            var extra = List[Expr]()
            for part in parts:
                for implied in _implied_filters(part):
                    extra.append(implied.copy())
            if len(parts) + len(extra) < 2:
                continue
            var below = self._nodes[i].left
            for implied in extra:
                self._nodes.append(
                    _plan_node(FILTER, below, exprs=[implied.copy()])
                )
                below = len(self._nodes) - 1
            for k in range(len(parts) - 1, 0, -1):
                self._nodes.append(
                    _plan_node(FILTER, below, exprs=[parts[k].copy()])
                )
                below = len(self._nodes) - 1
            self._nodes[i].left = below
            self._nodes[i].exprs = [parts[0].copy()]
            added = True
        if added:
            self._reorder()

    def _merge_filters(mut self):
        """Join a row-local filter directly over another into one filter
        on both predicates, lower first, so a scan below sees them all (CSV
        range filters, Parquet row-group pruning)."""
        var changed = True
        while changed:
            changed = False
            for i in range(len(self._nodes)):
                ref node = self._nodes[i]
                if node.kind != FILTER or node.left < 0:
                    continue
                ref below = self._nodes[node.left]
                if below.kind != FILTER:
                    continue
                if not (_row_local(node.exprs) and _row_local(below.exprs)):
                    continue
                var merged = below.exprs[0] & node.exprs[0]
                var child = below.left
                self._nodes[i].exprs = [merged^]
                self._nodes[i].left = child
                self._reorder()
                changed = True
                break

    def _order_joins(mut self, streaming: Bool, batch_size: Int) raises:
        """Order each chain of inner joins by how much their inputs narrow.

        A chain is a run of inner joins, filters and plain column selections
        along left inputs, over one base input. Run when the plan executes,
        not in `explain`: a join's right input is executed first when it
        is a known table narrowed by filters or joins, and how many of the
        table's rows it kept is its selectivity, as DuckDB's join order
        optimizer estimates it from statistics. Then:

        - A join whose keys come from one earlier join's right input, that
          keeps under half its table and holds at most a quarter as many
          rows as that input's table, joins that input instead (a bushy
          plan): PDS-H q7 joins customer to the two nations its
          filter allows, and orders to those customers, before any
          lineitem row is probed.
        - The remaining joins run most selective first, each once its keys
          are available, with every filter as soon as its columns are.

        Inputs run here replace their subtrees with their results, so
        nothing runs twice. Projection pushdown runs again afterwards, as
        the chain's column selections are dropped and the chain ends with
        one selecting its original output.
        """
        var original_frames = len(self._frames)
        var root = len(self._nodes) - 1
        # First choose each chain's first input, then order what joins it.
        var changed = self._probe_largest_inputs(root, streaming, batch_size)
        # The chains' top nodes, found before any is reordered: reordering
        # one appends nodes and leaves others unreachable.
        var parents = self._live_parents(root)
        var tops = List[Int]()
        for top in range(len(parents)):
            if parents[top] == -2:
                continue
            if not _chain_member(self._nodes[top]):
                continue
            var parent = parents[top]
            if (
                parent >= 0
                and self._nodes[parent].left == top
                and _chain_member(self._nodes[parent])
            ):
                continue
            tops.append(top)
        for top in tops:
            if self._order_chain(top, streaming, batch_size):
                changed = True
        if changed or len(self._frames) != original_frames:
            self._reorder(root)
            self._compact_scan_slots()
        if changed:
            self._push_projections()

    def _push_join_key_sets(mut self, streaming: Bool, batch_size: Int) raises:
        """Sideways information passing (DuckDB's join filter pushdown):
        an inner or semi join where one input is a filtered in-memory
        table or a join chain over tables at most a quarter the size of
        the other side's largest runs that input first, and each key's
        values become a filter on the other side's key when a sample of the table
        supplying it says the filter keeps at most one row in eight;
        predicate pushdown then moves it toward that table. Rows whose key no left
        row has join nothing, so the result is unchanged. Up to 50
        distinct keys filter as a list (an equality per value per row);
        more as their min-max range, as DuckDB does
        (`dynamic_or_filter_threshold`). The left input is replaced by its
        result, so it does not run twice. Only for a right side with a
        join of its own, which the filter then reaches first. TPC-DS q37
        and q82 join four items to an 11.7M-row inventory that is joined
        to dates first; q72 joins 1999's sales to inventory by week, and
        the weeks' range narrows the dates the inventory joins; q59 joins
        a year's weeks (the small right side) to weekly sales, whose week
        filter passes the group_by down to the date join.
        """
        var root = len(self._nodes) - 1
        var changed = True
        while changed:
            changed = False
            var parents = self._live_parents(root)
            for j in range(len(self._nodes)):
                if parents[j] == -2:
                    continue
                ref node = self._nodes[j]
                if node.kind != JOIN or not (
                    node.how == JOIN_INNER or node.how == JOIN_SEMI
                ):
                    continue
                if len(node.names) != len(node.right_keys):
                    continue
                if self._key_set_side(j, True, streaming, batch_size):
                    changed = True
                elif self._key_set_side(j, False, streaming, batch_size):
                    changed = True
                if changed:
                    root = len(self._nodes) - 1
                    break

    def _key_set_side(
        mut self, j: Int, small_left: Bool, streaming: Bool, batch_size: Int
    ) raises -> Bool:
        """One direction of `_push_join_key_sets` for the join at `j`: the
        small side (left when `small_left`) runs first and its keys filter
        the other side. True when the plan changed."""
        var small = self._nodes[j].left if small_left else self._nodes[j].right
        var large = self._nodes[j].right if small_left else self._nodes[j].left
        # A filtered table, or a join chain: inputs that keep few rows of
        # what they read. A bare table keeps all of them.
        if not (self._filtered_table(small) or self._has_join(small)):
            return False
        # Only a large side that does more than scan and filter before this
        # join (a join of its own) gains: against a bare table the join
        # already builds on its smaller side, and a filter there only
        # copies rows (TPC-DS q43, 2x).
        if not self._has_join(large):
            return False
        # A small right side already filters the left one inside the
        # stream, where the join order puts the selective joins first;
        # its keys pay only when a group_by on the left would otherwise
        # aggregate rows the join then drops (TPC-DS q59's weekly sales).
        # Pushed into plain join chains they cost PDS-H q8 5x and q21 2x.
        if not small_left and not self._has_kind(large, AGG):
            return False
        var large_rows = self._largest_scan(large)
        # The tables under the small side, unfiltered, must be at most a
        # quarter of the other side's largest: then running it first and
        # keeping its result costs at most a quarter of reading that table
        # (TPC-DS q72's sales chain reads 1.9M demographics against 11.7M
        # inventory rows).
        var small_bound = self._height_bound(small)
        if small_bound < 0:
            small_bound = self._largest_scan(small)
        if large_rows <= 0 or small_bound <= 0:
            return False
        # Against a left side the join probes anyway, a right side is run
        # early only when it is tiny next to it (32x: TPC-DS q59's weeks
        # against store sales are 39x smaller); PDS-H q21's 1.5M orders
        # against 6M lineitem rows ran twice and filtered nothing.
        if (4 if small_left else 32) * small_bound > large_rows:
            return False
        var small_keys = (
            self._nodes[j]
            .names.copy() if small_left else self._nodes[j]
            .right_keys.copy()
        )
        var large_keys = (
            self._nodes[j]
            .right_keys.copy() if small_left else self._nodes[j]
            .names.copy()
        )
        # Each key's table on the large side and its name there (through
        # renames and group_by keys); computed keys get none.
        var providers = List[Tuple[Int, String]]()
        var any = False
        for k in range(len(large_keys)):
            var found = self._provider_scan(large, large_keys[k])
            any = any or found[0] >= 0
            providers.append(found)
        if not any:
            return False
        var result = self._execute(small, False, streaming, batch_size)
        if not small_left and not self._any_selective(
            result, small_keys, providers
        ):
            # Nothing to pass: keep a right input as written, so the join
            # order still ranks it by its filters (a table already run looks
            # unfiltered, and PDS-H q21 then ran its selective supplier join
            # last). A left input is the stream's base, which nothing ranks:
            # its result is kept below, so it does not run twice.
            return False
        self._frames.append(result.copy())
        self._schemas.append(Optional[CsvSchema]())
        self._nodes.append(_plan_node(SCAN_FRAME, offset=len(self._frames) - 1))
        if small_left:
            self._nodes[j].left = len(self._nodes) - 1
        else:
            self._nodes[j].right = len(self._nodes) - 1
        var marker = len(self._frames) - 1
        var added = False
        for k in range(len(providers)):
            var scan = providers[k][0]
            if scan < 0:
                continue
            var base = providers[k][1]
            var dtype = (
                self._frames[self._nodes[scan].offset].column(base).dtype()
            )
            var keys = result.column(small_keys[k])
            var sampled = _key_set_filter(keys, base, dtype)
            if not sampled or not self._selective_on(scan, sampled.value()):
                continue
            var pushed = _key_set_filter(keys, large_keys[k], dtype)
            var target = (
                self._nodes[j].right if small_left else self._nodes[j].left
            )
            self._nodes.append(
                _plan_node(FILTER, target, exprs=[pushed.take()])
            )
            if small_left:
                self._nodes[j].right = len(self._nodes) - 1
            else:
                self._nodes[j].left = len(self._nodes) - 1
            added = True
        self._reorder()
        if not added:
            return True
        # Not split again: splitting a merged filter and merging it back
        # reverses its parts, and the written order is the order they are
        # evaluated in (ClickBench q21 then ran its string search first).
        self._push_predicates()
        self._merge_filters()
        if small_left:
            # The right side is built next whatever happens: run it now,
            # filtered, so the join sees both real heights and builds on
            # the smaller side (TPC-DS q72 built its two-key index on 2.4M
            # stock rows for 8,689 sales).
            for at in range(len(self._nodes)):
                ref joined = self._nodes[at]
                if (
                    joined.kind == JOIN
                    and joined.left >= 0
                    and self._nodes[joined.left].kind == SCAN_FRAME
                    and self._nodes[joined.left].offset == marker
                ):
                    var right = self._execute(
                        joined.right, False, streaming, batch_size
                    )
                    self._frames.append(right^)
                    self._schemas.append(Optional[CsvSchema]())
                    self._nodes.append(
                        _plan_node(SCAN_FRAME, offset=len(self._frames) - 1)
                    )
                    self._nodes[at].right = len(self._nodes) - 1
                    break
            self._reorder()
        return True

    def _filtered_table(self, index: Int) -> Bool:
        """Whether node `index` is an in-memory table under row-local
        steps, at least one of them a filter."""
        var cursor = index
        var filtered = False
        while cursor >= 0:
            ref node = self._nodes[cursor]
            if node.kind == SCAN_FRAME:
                return filtered
            if node.kind == FILTER and _row_local(node.exprs):
                filtered = True
            elif not (
                node.kind == DROP
                or (
                    (node.kind == SELECT or node.kind == WITH_COLUMNS)
                    and _stream_rows(node)
                )
            ):
                return False
            cursor = node.left
        return False

    def _provider_scan(
        self, index: Int, name: String
    ) raises -> Tuple[Int, String]:
        """The in-memory table node under node `index` that supplies
        column `name`, and the column's name there, following renames
        (`col(a).alias(b)`); (-1, "") when none does or the column is
        computed."""
        var stack = List[Tuple[Int, String]]()
        stack.append((index, name))
        while len(stack) > 0:
            var top = stack.pop()
            var i = top[0]
            var wanted = top[1]
            ref node = self._nodes[i]
            if node.kind == SCAN_FRAME:
                if wanted in self._frames[node.offset].columns():
                    return (i, wanted)
                continue
            if node.kind == AGG and wanted not in node.names:
                # An aggregate, not a key: computed here.
                continue
            if node.kind == SELECT or node.kind == WITH_COLUMNS:
                var renamed = False
                var computed = False
                for e in node.exprs:
                    if e._name != wanted:
                        continue
                    var reads = _references(e)
                    if reads and len(reads.value()) == 1 and len(e._nodes) == 1:
                        wanted = reads.value()[0]
                        renamed = True
                    else:
                        computed = True
                if computed and not renamed:
                    continue
            if node.left >= 0:
                stack.append((node.left, wanted))
            if node.right >= 0:
                stack.append((node.right, wanted))
        return (-1, String())

    def _selective_on(mut self, scan: Int, keys: Expr) raises -> Bool:
        """Whether `keys` keeps at most one row in eight of the table at
        node `scan`, estimated on evenly spaced runs of it
        (`_estimated_rows`). A key set spanning most of the right side's
        keys filters nothing and only moves the join off its better
        build side."""
        var height = self._frames[self._nodes[scan].offset].height()
        self._nodes.append(_plan_node(FILTER, scan, exprs=[keys.copy()]))
        var estimated = self._estimated_rows(len(self._nodes) - 1)
        _ = self._nodes.pop()
        return estimated >= 0 and 8 * estimated <= height

    def _any_selective(
        mut self,
        result: DataFrame,
        keys: List[String],
        providers: List[Tuple[Int, String]],
    ) raises -> Bool:
        """Whether any key of `result` would filter its provider table to at
        most one row in eight (`_selective_on`)."""
        for k in range(len(providers)):
            var scan = providers[k][0]
            if scan < 0:
                continue
            var base = providers[k][1]
            var dtype = (
                self._frames[self._nodes[scan].offset].column(base).dtype()
            )
            var sampled = _key_set_filter(result.column(keys[k]), base, dtype)
            if sampled and self._selective_on(scan, sampled.value()):
                return True
        return False

    def _has_kind(self, index: Int, kind: Int) -> Bool:
        """Whether a node of `kind` sits anywhere under node `index`."""
        var stack: List[Int] = [index]
        while len(stack) > 0:
            var i = stack.pop()
            ref node = self._nodes[i]
            if node.kind == kind:
                return True
            if node.left >= 0:
                stack.append(node.left)
            if node.right >= 0:
                stack.append(node.right)
        return False

    def _has_join(self, index: Int) -> Bool:
        """Whether a join sits anywhere under node `index`."""
        var stack: List[Int] = [index]
        while len(stack) > 0:
            var i = stack.pop()
            ref node = self._nodes[i]
            if node.kind == JOIN:
                return True
            if node.left >= 0:
                stack.append(node.left)
            if node.right >= 0:
                stack.append(node.right)
        return False

    def _largest_scan(self, index: Int) -> Int:
        """The most rows any in-memory table under node `index` holds."""
        var largest = 0
        var stack: List[Int] = [index]
        while len(stack) > 0:
            var i = stack.pop()
            ref node = self._nodes[i]
            if node.kind == SCAN_FRAME:
                largest = max(largest, self._frames[node.offset].height())
            if node.left >= 0:
                stack.append(node.left)
            if node.right >= 0:
                stack.append(node.right)
        return largest

    def _live_parents(self, root: Int) -> List[Int]:
        """Each node's parent in the plan under `root`: -1 for the root and
        -2 for a node no longer reachable (an earlier rewrite's leftovers)."""
        var parents = List[Int](length=len(self._nodes), fill=-2)
        parents[root] = -1
        var stack: List[Int] = [root]
        while len(stack) > 0:
            var i = stack.pop()
            for child in [self._nodes[i].left, self._nodes[i].right]:
                if child >= 0 and parents[child] == -2:
                    parents[child] = i
                    stack.append(child)
        return parents^

    def _order_free_above(
        self, top: Int, parents: List[Int], through_joins: Bool = False
    ) -> Bool:
        """Whether the plan above `top` gives the same result whatever
        order `top`'s rows arrive in: row-local steps up to an aggregation
        that reads no row position. With `through_joins`, inner, left, semi
        and anti joins on the way pass any order on, as they keep their
        inputs' rows."""
        var cursor = parents[top]
        while cursor >= 0:
            ref node = self._nodes[cursor]
            if (
                through_joins
                and node.kind == JOIN
                and (
                    node.how == JOIN_INNER
                    or node.how == JOIN_LEFT
                    or node.how == JOIN_SEMI
                    or node.how == JOIN_ANTI
                )
            ):
                cursor = parents[cursor]
                continue
            if node.kind == AGG:
                return not node.maintain_order and _order_insensitive(
                    node.exprs
                )
            if node.kind == SELECT and not _row_local(node.exprs):
                return _stream_reductions(node.exprs) and _order_insensitive(
                    node.exprs
                )
            if (
                node.kind == FILTER
                or node.kind == SELECT
                or node.kind == WITH_COLUMNS
            ):
                if not _row_local(node.exprs):
                    return False
            elif node.kind != DROP:
                return False
            cursor = parents[cursor]
        return False

    def _estimated_rows(self, index: Int) raises -> Int:
        """Rows node `index` yields, without running it: exact for a table,
        estimated for one row-local filter over a table by applying the
        filter to 64 evenly spaced runs of 1,024 rows (every row of a table
        that small). -1 for anything else. Evenly spaced runs, not a
        prefix: tables are often stored in date or key order, which a
        prefix would misjudge."""
        ref node = self._nodes[index]
        if node.kind == SCAN_FRAME:
            return self._frames[node.offset].height()
        if node.kind == DROP or (
            (node.kind == SELECT or node.kind == WITH_COLUMNS)
            and _stream_rows(node)
        ):
            return self._estimated_rows(node.left)
        if node.kind != FILTER or not _row_local(node.exprs):
            return -1
        ref source = self._nodes[node.left]
        if source.kind != SCAN_FRAME:
            return -1
        var height = self._frames[source.offset].height()
        var runs = 64
        var run = 1024
        # Only the columns the filter reads.
        var reads = _references(node.exprs[0])
        if not reads:
            return -1
        var names = List[String]()
        for name in reads.value():
            if name not in names:
                names.append(name)
        var sample = self._frames[source.offset].select(names)
        if height > runs * run:
            var picked = List[Int](capacity=runs * run)
            for k in range(runs):
                var first = k * (height // runs)
                for i in range(run):
                    picked.append(first + i)
            # Gathered within each stored chunk, so no column is merged first.
            sample = sample._filter_rows(picked^)
        var kept = sample.filter(node.exprs[0]).height()
        if height <= runs * run:
            return kept
        return Int(Float64(kept) / Float64(runs * run) * Float64(height))

    def _progression_key(self, index: Int, keys: List[String]) raises -> Bool:
        """Whether a join building on node `index` with these keys needs no
        hash table: one integer key of a whole table whose values are an
        arithmetic progression, which is looked up by position."""
        if len(keys) != 1:
            return False
        var cursor = index
        while cursor >= 0:
            ref node = self._nodes[cursor]
            if node.kind == SCAN_FRAME:
                var key = self._frames[node.offset][keys[0]]
                if key.dtype().physical() != DataType.INT64:
                    return False
                return int64_progression(key)[0]
            if node.kind != DROP and not (
                (node.kind == SELECT or node.kind == WITH_COLUMNS)
                and _chain_member(node)
            ):
                return False
            cursor = node.left
        return False

    def _probe_largest_inputs(
        mut self, root: Int, streaming: Bool, batch_size: Int
    ) raises -> Bool:
        """Start each chain of inner joins from its largest input.

        A chain runs by streaming its first input through hash tables built
        on every other input, so the first input is the only one never
        hashed. Written as `small.join(large)`, a plan hashes the large
        table. Polars and DuckDB choose the build side of each join by
        size; here the chain is re-rooted at its largest input, and the
        other inputs join outward from it along the same key pairs.

        Only where the rows' order cannot show (`_order_free_above`): an
        inner join's rows follow its left input, so a new first input
        changes their order.
        """
        var parents = self._live_parents(root)
        var changed = False
        for top in range(len(parents)):
            if parents[top] == -2:
                continue
            ref node = self._nodes[top]
            if node.kind != JOIN or not _chain_member(node):
                continue
            var parent = parents[top]
            if (
                parent >= 0
                and self._nodes[parent].left == top
                and (
                    self._nodes[parent].kind == SELECT
                    or self._nodes[parent].kind == JOIN
                )
                and _chain_member(self._nodes[parent])
            ):
                continue
            if not self._order_free_above(top, parents):
                continue
            if self._probe_largest_input(top, streaming, batch_size):
                changed = True
        return changed

    def _probe_largest_input(
        mut self, top: Int, streaming: Bool, batch_size: Int
    ) raises -> Bool:
        """Re-root the join chain ending at `top`; see
        `_probe_largest_inputs`. False when the chain stays as it is."""
        var joins = List[Int]()
        var cursor = top
        while cursor >= 0:
            ref node = self._nodes[cursor]
            if node.kind == JOIN and _chain_member(node):
                joins.append(cursor)
            elif not (node.kind == SELECT and _chain_member(node)):
                break
            cursor = node.left
        if cursor < 0 or len(joins) == 0:
            return False
        # The first input is the lowest join's whole left input, with any
        # column selection on it, so it keeps supplying only those columns.
        var base = self._nodes[joins[len(joins) - 1]].left
        joins.reverse()
        var n = len(joins)
        # Table 0 is the chain's first input; table p + 1 is join p's right.
        var tables: List[Int] = [base]
        for p in range(n):
            tables.append(self._nodes[joins[p]].right)
        # Which table supplies each column. A name two tables share is
        # allowed only as a join's key pair of one name; anything else
        # would be renamed by a suffix in one order and not in another.
        var provider = Dict[String, Int]()
        var shared = List[Tuple[String, Int, Int]]()
        for t in range(n + 1):
            for c in self._columns_of(tables[t]):
                if c in provider:
                    shared.append((c, provider[c], t))
                else:
                    provider[c] = t
        # Join p links table p + 1 to the one table holding its left keys.
        var linked = List[Int]()
        for p in range(n):
            ref node = self._nodes[joins[p]]
            if len(node.names) != len(node.right_keys):
                return False
            var from_table = -1
            for name in node.names:
                if name not in provider:
                    return False
                if from_table == -1:
                    from_table = provider[name]
                elif from_table != provider[name]:
                    return False
            if from_table > p:
                return False
            linked.append(from_table)
        for item in shared:
            var allowed = False
            for p in range(n):
                ref node = self._nodes[joins[p]]
                if not (
                    (linked[p] == item[1] and p + 1 == item[2])
                    or (linked[p] == item[2] and p + 1 == item[1])
                ):
                    continue
                for i in range(len(node.names)):
                    if (
                        node.names[i] == item[0]
                        and node.right_keys[i] == item[0]
                    ):
                        allowed = True
            if not allowed:
                return False
        # The first input streams through the chain as slices, so the
        # columns it supplies cost nothing to carry; as a right input each
        # would be gathered for every joined row. Move it only when it
        # supplies nothing but its join keys.
        var top_columns = self._columns_of(top)
        for name in top_columns:
            if provider[name] != 0:
                continue
            var key = False
            for p in range(n):
                if linked[p] == 0 and name in self._nodes[joins[p]].names:
                    key = True
            if not key:
                return False
        # Rows per table, estimated only when needed (-2: not yet). A
        # table that cannot hold more rows than the first input is never
        # a candidate, so most chains estimate nothing beyond the first.
        var rows = List[Int](length=n + 1, fill=-2)
        rows[0] = self._estimated_rows(tables[0])
        if rows[0] < 0:
            return False
        # Candidates by size. For each, compare the rows hashed along the
        # path between it and the first input, the only joins whose build
        # side changes: now each table on the path but the first is built;
        # re-rooted, each but the candidate is.
        var largest = 0
        var saved = 0
        for candidate in range(1, n + 1):
            var bound = self._height_bound(tables[candidate])
            if bound < 0 or bound <= rows[0]:
                continue
            rows[candidate] = self._estimated_rows(tables[candidate])
            if rows[candidate] <= rows[0]:
                continue
            var path = List[Int]()
            var t = candidate
            var known = True
            while t != 0:
                if rows[t] == -2:
                    rows[t] = self._estimated_rows(tables[t])
                known = known and rows[t] >= 0
                path.append(t)
                t = linked[t - 1]
            if not known:
                continue
            var before = 0
            var after = 0
            for k in range(len(path)):
                var table = path[k]
                ref edge = self._nodes[joins[table - 1]]
                # Built now, keyed by the join's right keys.
                if not self._progression_key(tables[table], edge.right_keys):
                    before += rows[table]
                # Built after re-rooting: the table on the other end,
                # keyed by the join's left keys.
                var other = linked[table - 1]
                if not self._progression_key(tables[other], edge.names):
                    after += rows[other]
            if before - after > saved:
                saved = before - after
                largest = candidate
        if largest == 0:
            return False
        # Join outward from the largest table. A join met from its right
        # table's side swaps its key lists; the keys it then drops are the
        # other table's, whose values the surviving keys hold, so later
        # uses of those names read the surviving ones (`renamed`).
        var visited = List[Bool](length=n + 1, fill=False)
        visited[largest] = True
        var placed = List[Bool](length=n, fill=False)
        var renamed = Dict[String, String]()
        var added = List[PlanNode]()
        var current = tables[largest]
        # The columns the chain holds so far. Each join must find its left
        # keys there and add no name already present.
        var have = Dict[String, Bool]()
        for c in self._columns_of(tables[largest]):
            have[c] = True
        for _ in range(n):
            var pick = -1
            for p in range(n):
                if not placed[p] and visited[linked[p]] != visited[p + 1]:
                    pick = p
                    break
            if pick < 0:
                return False
            var node = self._nodes[joins[pick]].copy()
            node.left = current
            var forward = visited[linked[pick]]
            var left_keys = (
                node.names.copy() if forward else node.right_keys.copy()
            )
            for i in range(len(left_keys)):
                while left_keys[i] in renamed:
                    left_keys[i] = renamed[left_keys[i]]
            if forward:
                visited[pick + 1] = True
            else:
                var dropped = node.names.copy()
                node.right = tables[linked[pick]]
                node.right_keys = dropped.copy()
                for i in range(len(dropped)):
                    if dropped[i] != left_keys[i]:
                        renamed[dropped[i]] = left_keys[i]
                visited[linked[pick]] = True
            for key in left_keys:
                if key not in have:
                    return False
            for c in self._columns_of(node.right):
                if c in node.right_keys:
                    continue
                if c in have:
                    return False
                have[c] = True
            node.names = left_keys^
            placed[pick] = True
            current = len(self._nodes) + len(added)
            added.append(node^)
        for node in added:
            self._nodes.append(node.copy())
        var keep = List[Expr]()
        for name in top_columns:
            var source = name
            while source in renamed:
                source = renamed[source]
            keep.append(
                col(source) if source == name else col(source).alias(name)
            )
        self._nodes[top] = _plan_node(SELECT, current, exprs=keep)
        trace_path("lazy.probe_largest")
        return True

    def _spine_rows(self, index: Int) -> Int:
        """Rows of the table under `index` along steps that keep, drop or
        join rows (a filter, a projection, an inner join's left input);
        -1 through anything else (an aggregation) or a file scan."""
        var cursor = index
        while cursor >= 0:
            ref node = self._nodes[cursor]
            if node.kind == SCAN_FRAME:
                return self._frames[node.offset].height()
            if (
                node.kind == FILTER
                or node.kind == DROP
                or (node.kind == JOIN and node.how == JOIN_INNER)
                or (
                    (node.kind == SELECT or node.kind == WITH_COLUMNS)
                    and _stream_rows(node)
                )
            ):
                cursor = node.left
                continue
            return -1
        return -1

    def _measurable(self, index: Int) -> Bool:
        """Whether `_selectivity` would run node `index`: a known table,
        narrowed by steps above it."""
        return (
            self._nodes[index].kind != SCAN_FRAME
            and self._spine_rows(index) > 0
        )

    def _selectivity(
        mut self, index: Int, streaming: Bool, batch_size: Int
    ) raises -> Float64:
        """The share of its table's rows that node `index` keeps, executing
        it and replacing it with its result; 1 when it is a plain table or
        no table is known."""
        var table = self._spine_rows(index)
        if table <= 0 or self._nodes[index].kind == SCAN_FRAME:
            return 1.0
        var frame = self._execute(index, False, streaming, batch_size)
        var rows = frame.height()
        self._frames.append(frame^)
        self._schemas.append(Optional[CsvSchema]())
        self._nodes[index] = _plan_node(
            SCAN_FRAME, offset=len(self._frames) - 1
        )
        return Float64(rows) / Float64(table)

    def _order_chain(
        mut self, top: Int, streaming: Bool, batch_size: Int
    ) raises -> Bool:
        """Reorder the chain whose last node is `top`; see `_order_joins`.
        False when join topology stays unchanged. Measured inputs remain
        cached even when no step moves or a candidate is rejected."""
        var spine = List[Int]()
        var cursor = top
        while cursor >= 0 and _chain_member(self._nodes[cursor]):
            spine.append(cursor)
            cursor = self._nodes[cursor].left
        var base = cursor
        if base < 0:
            return False
        # Steps bottom to top, without the column selections.
        var steps = List[Int]()
        var join_count = 0
        for k in range(len(spine)):
            var i = spine[len(spine) - 1 - k]
            if self._nodes[i].kind == SELECT:
                continue
            steps.append(i)
            if self._nodes[i].kind == JOIN:
                join_count += 1
        if join_count < 2:
            return False
        var top_columns = self._columns_of(top)
        # Which step supplies each column (-1 the base). Every name must be
        # unique across the chain, so no join renames a column in any order.
        var provider = Dict[String, Int]()
        for c in self._columns_of(base):
            provider[c] = -1
        var outputs = List[List[String]]()
        for p in range(len(steps)):
            ref node = self._nodes[steps[p]]
            var produced = List[String]()
            if node.kind == JOIN:
                for c in self._columns_of(node.right):
                    if c in node.right_keys:
                        continue
                    if c in provider:
                        return False
                    provider[c] = p
                    produced.append(c)
            outputs.append(produced^)
        var needs = List[List[String]]()
        for p in range(len(steps)):
            ref node = self._nodes[steps[p]]
            if node.kind == JOIN:
                needs.append(node.names.copy())
            else:
                var reads = _references(node.exprs[0])
                if not reads:
                    return False
                needs.append(reads.value().copy())
            for name in needs[p]:
                if name not in provider:
                    return False
        # owner[p]: the step whose output now carries step p's columns.
        var owner = List[Int](capacity=len(steps))
        for p in range(len(steps)):
            owner.append(p)
        var selectivity = List[Float64](length=len(steps), fill=1.0)
        var measured = List[Bool](length=len(steps), fill=False)
        var pushed = False
        # Keep tentative topology changes separate from input results.
        # Restoring a chain must preserve scans produced by measurement.
        var saved = self._nodes.copy()
        for reverse in range(len(steps)):
            var p = len(steps) - 1 - reverse
            if self._nodes[steps[p]].kind != JOIN:
                continue
            var keys = self._nodes[steps[p]].names.copy()
            var host = -2
            for name in keys:
                var from_step = provider[name]
                var o = owner[from_step] if from_step >= 0 else -1
                if host == -2:
                    host = o
                elif host != o:
                    host = -1
            if host < 0:
                continue
            # The host's right input must not hold a name this join adds.
            var host_columns = self._columns_of(self._nodes[steps[host]].right)
            var clash = False
            for c in outputs[p]:
                clash = clash or c in host_columns
            if clash:
                continue
            # The pushed input must be small next to the host's table, as a
            # dimension of it: joining 1.5M lineitem rows into 57K orders
            # first made PDS-H q10 12% slower. Its table bounds its rows
            # before it runs, and its rows bound them after.
            var host_rows = self._spine_rows(self._nodes[steps[host]].right)
            var table = self._spine_rows(self._nodes[steps[p]].right)
            if host_rows < 0 or table < 0 or 4 * table > host_rows:
                continue
            var share = self._selectivity(
                self._nodes[steps[p]].right, streaming, batch_size
            )
            selectivity[p] = share
            measured[p] = True
            if share >= 0.5:
                continue
            var rows = self._known_height(self._nodes[steps[p]].right)
            if rows < 0 or 4 * rows > host_rows:
                continue
            var inner = self._nodes[steps[p]].copy()
            inner.left = self._nodes[steps[host]].right
            self._nodes.append(inner^)
            self._nodes[steps[host]].right = len(self._nodes) - 1
            for q in range(len(steps)):
                if owner[q] == p:
                    owner[q] = host
            pushed = True
        # Selectivity of the joins that stay in the chain, measured only
        # where it can change the order: a narrowed input whose join could
        # run before one that precedes it, and that one. A plain table keeps
        # all its rows and never moves ahead.
        var need = List[Bool](length=len(steps), fill=False)
        var done = List[Bool](length=len(steps), fill=False)
        for p in range(len(steps)):
            if owner[p] != p:
                done[p] = True
        for p in range(len(steps)):
            if owner[p] != p:
                continue
            if self._nodes[steps[p]].kind == JOIN:
                for q in range(p + 1, len(steps)):
                    if (
                        owner[q] != q
                        or self._nodes[steps[q]].kind != JOIN
                        or not self._available(needs[q], provider, owner, done)
                    ):
                        continue
                    if self._measurable(self._nodes[steps[q]].right):
                        need[q] = True
                        need[p] = need[p] or self._measurable(
                            self._nodes[steps[p]].right
                        )
            done[p] = True
        for p in range(len(steps)):
            if not need[p] or measured[p]:
                continue
            selectivity[p] = self._selectivity(
                self._nodes[steps[p]].right, streaming, batch_size
            )
        # Greedy order: filters as soon as their columns are available, then
        # the most selective available join (ties keep the original order).
        var placed = List[Bool](length=len(steps), fill=False)
        for p in range(len(steps)):
            if owner[p] != p:
                placed[p] = True
        var order = List[Int]()
        var remaining = 0
        for p in range(len(steps)):
            if not placed[p]:
                remaining += 1
        while len(order) < remaining:
            var progress = False
            for p in range(len(steps)):
                if placed[p] or self._nodes[steps[p]].kind != FILTER:
                    continue
                if self._available(needs[p], provider, owner, placed):
                    placed[p] = True
                    order.append(p)
                    progress = True
            var best = -1
            for p in range(len(steps)):
                if placed[p] or self._nodes[steps[p]].kind != JOIN:
                    continue
                if not self._available(needs[p], provider, owner, placed):
                    continue
                if best < 0 or selectivity[p] < selectivity[best]:
                    best = p
            if best >= 0:
                placed[best] = True
                order.append(best)
                progress = True
            if not progress:
                self._restore_measured_chain(saved^)
                return False
        var moved = pushed
        var expected = 0
        for p in range(len(steps)):
            if owner[p] != p:
                continue
            if order[expected] != p:
                moved = True
            expected += 1
        if not moved:
            self._restore_measured_chain(saved^)
            return False
        var current = base
        for p in order:
            var node = self._nodes[steps[p]].copy()
            node.left = current
            self._nodes.append(node^)
            current = len(self._nodes) - 1
        var keep = List[Expr]()
        for name in top_columns:
            keep.append(col(name))
        self._nodes[top] = _plan_node(SELECT, current, exprs=keep)
        trace_path("lazy.join_order")
        return True

    def _restore_measured_chain(mut self, var saved: List[PlanNode]):
        """Discard speculative topology, retaining executed original inputs.

        Only original nodes can be restored as cached scans. Results of
        newly appended speculative joins have no equivalent original node;
        scan-slot compaction drops those once unreachable.
        """
        for i in range(len(saved)):
            if (
                self._nodes[i].kind == SCAN_FRAME
                and saved[i].kind != SCAN_FRAME
            ):
                saved[i] = self._nodes[i].copy()
        self._nodes = saved^

    def _compact_scan_slots(mut self):
        """Release scan inputs/materializations no reachable node owns."""
        var slots = List[Int](length=len(self._frames), fill=-1)
        var frames = List[DataFrame]()
        var schemas = List[Optional[CsvSchema]]()
        for i in range(len(self._nodes)):
            if not _is_scan(self._nodes[i].kind):
                continue
            var old = self._nodes[i].offset
            if slots[old] < 0:
                slots[old] = len(frames)
                frames.append(self._frames[old].copy())
                schemas.append(self._schemas[old].copy())
            self._nodes[i].offset = slots[old]
        self._frames = frames^
        self._schemas = schemas^

    def _available(
        self,
        names: List[String],
        provider: Dict[String, Int],
        owner: List[Int],
        placed: List[Bool],
    ) raises -> Bool:
        """Whether every name comes from the base or a placed step."""
        for name in names:
            var p = provider[name]
            if p >= 0 and not placed[owner[p]]:
                return False
        return True

    def _push_predicates(mut self) raises:
        """Swap each filter below operators that cannot change its result."""
        var changed = True
        while changed:
            changed = False
            for i in range(len(self._nodes)):
                if self._nodes[i].kind != FILTER:
                    continue
                var child = self._nodes[i].left
                var side = self._passes(i, child)
                if side >= 0:
                    self._swap_down(i, child, side == 1)
                    changed = True
                    break
                # Over another row-local filter that is stuck where this one
                # could go further (a filter on an AND split into parts,
                # one on the keys of a group_by, one on its aggregates):
                # trade places, so this one moves on next time round.
                if not _row_local(self._nodes[i].exprs):
                    continue
                # The operator under the run of row-local filters below.
                var under = child
                while (
                    under >= 0
                    and self._nodes[under].kind == FILTER
                    and _row_local(self._nodes[under].exprs)
                ):
                    under = self._nodes[under].left
                # Not over a join: the join order and stream batches are
                # tuned to where filters sit, and moving an OR-implied range
                # below the joins cost TPC-DS q13 10% (an OR of decimal
                # ranges over 2.9M rows, and joins on batches a tenth the
                # size) and q69 14%. Over a group_by (TPC-DS q39) or a
                # with_columns (PDS-H q22's phone prefix, 8%) it pays.
                if (
                    under != child
                    and under >= 0
                    and self._nodes[under].kind != JOIN
                    and self._passes(child, under) < 0
                    and self._passes(i, under) >= 0
                ):
                    self._swap_down(i, child, False)
                    changed = True
                    break

    def _passes(self, filter: Int, child: Int) raises -> Int:
        """Whether the filter at `filter` can move below the node at
        `child` without changing its result: -1 when not, else the side of
        `child` it moves into (0 left or only input, 1 right)."""
        var reads = _references(self._nodes[filter].exprs[0])
        if not reads or child < 0:
            return -1
        ref below = self._nodes[child]
        if below.kind == WITH_COLUMNS or below.kind == SELECT:
            # Safe when the filter reads only unchanged input columns: a
            # select of plain columns changes none, so it checks only that
            # (below); with_columns may replace what it names.
            if below.kind == WITH_COLUMNS:
                var produced = _output_names(below.exprs)
                for name in reads.value():
                    for p in produced:
                        if p == name:
                            return -1
            if below.kind == SELECT:
                for e in below.exprs:
                    var r = _references(e)
                    if (
                        not r
                        or len(r.value()) != 1
                        or r.value()[0] != e._name
                        or len(e._nodes) != 1
                    ):
                        return -1
            return 0
        if (
            below.kind == AGG
            and len(below.names) > 0
            and _row_local(self._nodes[filter].exprs)
            and _covers(below.names, reads.value())
        ):
            # A predicate on the grouping keys alone is the same for every
            # row of a group, so filtering rows before the aggregation keeps
            # exactly the groups it would keep after, with the same values
            # and first-occurrence order (Polars and DuckDB push it down
            # too; TPC-DS q39 filters d_moy after grouping 2.3M rows).
            return 0
        if below.kind == SORT and _row_local(self._nodes[filter].exprs):
            # Stable sorting and a row-local filter commute: the surviving
            # rows retain the same relative sort order.
            return 0
        if below.kind == JOIN and (
            below.how == JOIN_INNER
            or below.how == JOIN_LEFT
            or below.how == JOIN_SEMI
            or below.how == JOIN_ANTI
        ):
            var left_cols = self._columns_of(below.left)
            var right_cols = List[String]()
            for c in self._columns_of(below.right):
                # Coalesced right keys are not in the join output, so no
                # filter above can mean them.
                if not (below.coalesce and c in below.right_keys):
                    right_cols.append(c)
            if _covers(left_cols, reads.value()):
                return 0
            if below.how == JOIN_INNER and _covers_exclusive(
                right_cols, left_cols, reads.value()
            ):
                return 1
        return -1

    def _swap_down(mut self, filter: Int, child: Int, right_side: Bool):
        """Move filter from above child to directly above child's input."""
        var parents = self._parents()
        var parent = parents[filter]
        var target = (
            self._nodes[child].right if right_side else self._nodes[child].left
        )
        self._nodes[filter].left = target
        if right_side:
            self._nodes[child].right = filter
        else:
            self._nodes[child].left = filter
        if parent >= 0:
            if self._nodes[parent].left == filter:
                self._nodes[parent].left = child
            else:
                self._nodes[parent].right = child
        self._reorder()

    def _reorder(mut self, root: Int = -1):
        """Restore children-before-parents order after rewiring."""
        var selected = root
        if selected < 0:
            var parents = self._parents()
            for i in range(len(self._nodes)):
                if parents[i] < 0:
                    selected = i
        var order = List[Int]()
        var stack = List[Int]()
        var visited = List[Bool](length=len(self._nodes), fill=False)
        stack.append(selected)
        while len(stack) > 0:
            var top = stack[len(stack) - 1]
            var pending = False
            for child in [self._nodes[top].left, self._nodes[top].right]:
                if child >= 0 and not visited[child]:
                    stack.append(child)
                    pending = True
            if not pending:
                _ = stack.pop()
                if not visited[top]:
                    visited[top] = True
                    order.append(top)
        var position = List[Int](length=len(self._nodes), fill=-1)
        for k in range(len(order)):
            position[order[k]] = k
        var nodes = List[PlanNode]()
        for k in range(len(order)):
            var node = self._nodes[order[k]].copy()
            if node.left >= 0:
                node.left = position[node.left]
            if node.right >= 0:
                node.right = position[node.right]
            nodes.append(node^)
        self._nodes = nodes^

    def _push_slices(mut self):
        """Move leading slices below row-local projections; a leading slice
        directly over a CSV scan also becomes the reader's n_rows."""
        var changed = True
        while changed:
            changed = False
            for i in range(len(self._nodes)):
                ref node = self._nodes[i]
                if node.kind != SLICE or node.offset != 0 or node.length < 0:
                    continue
                ref below = self._nodes[node.left]
                if (
                    below.kind == SELECT or below.kind == WITH_COLUMNS
                ) and _row_local(below.exprs):
                    self._swap_down(i, node.left, False)
                    changed = True
                    break
        for i in range(len(self._nodes)):
            ref node = self._nodes[i]
            if node.kind != SLICE or node.offset != 0 or node.length < 0:
                continue
            ref scan = self._nodes[node.left]
            if scan.kind == SCAN_CSV and (
                scan.length < 0 or scan.length > node.length
            ):
                self._nodes[node.left].length = node.length

    def _fuse_top_k(mut self):
        """A slice directly above a sort needs only the sort's first
        offset + length rows (SQL's ORDER BY ... LIMIT), so the sort records
        that limit and selects those rows instead of sorting them all
        (#332). The slice stays to apply its offset; a sort holds its limit
        in `length`, which it otherwise leaves at -1."""
        for i in range(len(self._nodes)):
            ref node = self._nodes[i]
            if node.kind != SLICE or node.offset < 0 or node.length < 0:
                continue
            ref below = self._nodes[node.left]
            if below.kind != SORT:
                continue
            var limit = node.offset + node.length
            if below.length < 0 or limit < below.length:
                self._nodes[node.left].length = limit

    def _push_projections(mut self) raises:
        """Scans read only columns that some operator above them uses.

        A join passes each input only its keys and the columns the plan
        above reads from that side, so unused payload columns are never
        read or gathered. A join input that is not a scan gets a plain
        column selection above it.
        """
        var needed = List[Optional[List[String]]](
            length=len(self._nodes), fill=Optional[List[String]]()
        )
        var all_needed = List[Bool](length=len(self._nodes), fill=False)
        all_needed[len(self._nodes) - 1] = True
        # (join index, right side?, columns) for join inputs to narrow.
        var narrow_joins = List[Int]()
        var narrow_sides = List[Bool]()
        var narrow_columns = List[List[String]]()
        for reverse in range(len(self._nodes)):
            var i = len(self._nodes) - 1 - reverse
            ref node = self._nodes[i]
            if node.kind == JOIN and not all_needed[i] and needed[i]:
                var sides = self._join_input_columns(i, needed[i].value())
                if sides:
                    for side in range(2):
                        var child = node.right if side == 1 else node.left
                        ref keep = sides.value()[side]
                        needed[child] = keep.copy()
                        narrow_joins.append(i)
                        narrow_sides.append(side == 1)
                        narrow_columns.append(keep.copy())
                    continue
            if _is_scan(node.kind):
                if not all_needed[i] and needed[i]:
                    var columns = self._columns_of(i)
                    var keep = List[String]()
                    for c in columns:
                        for want in needed[i].value():
                            if want == c:
                                keep.append(c)
                                break
                    if len(keep) > 0 and len(keep) < len(columns):
                        self._nodes[i].names = keep^
                continue
            # What this node reads from its input(s).
            var reads = List[String]()
            var reads_all = all_needed[i] and (
                node.kind == FILTER
                or node.kind == SORT
                or node.kind == SLICE
                or node.kind == UNIQUE
                or node.kind == WITH_COLUMNS
                or node.kind == DROP
                or node.kind == JOIN
                or node.kind == EXPLODE
                or node.kind == UNNEST
            )
            if node.kind == UNNEST:
                reads.append(node.text)
            if node.kind == UNIQUE and len(node.names) == 0:
                reads_all = True
            if not reads_all:
                if needed[i]:
                    for n in needed[i].value():
                        reads.append(n)
                for e in node.exprs:
                    var r = _references(e)
                    if not r:
                        reads_all = True
                    else:
                        for n in r.value():
                            reads.append(n)
                for n in node.names:
                    reads.append(n)
            for child in [node.left, node.right]:
                if child < 0:
                    continue
                if reads_all or node.kind == JOIN or node.kind == ASOF:
                    all_needed[child] = True
                elif needed[child]:
                    var merged = needed[child].value().copy()
                    for n in reads:
                        merged.append(n)
                    needed[child] = merged^
                else:
                    needed[child] = reads.copy()
        self._narrow_join_inputs(narrow_joins, narrow_sides, narrow_columns)

    def _join_input_columns(
        self, index: Int, wanted: List[String]
    ) raises -> Optional[List[List[String]]]:
        """Columns each join input must supply for the wanted output names.

        Keys are always kept. A right column whose output name carries the
        suffix keeps its colliding left column too, so no output is renamed.
        Returns None when the unpruned join would reject its output names,
        so execution still raises that error.
        """
        ref node = self._nodes[index]
        var left_cols = self._columns_of(node.left)
        var right_cols = self._columns_of(node.right)
        var suffix = node.names2[0]
        var membership = node.how == JOIN_SEMI or node.how == JOIN_ANTI
        var outputs = Dict[String, Bool]()
        for c in left_cols:
            outputs[c] = True
        var left_names = outputs.copy()
        var want = Dict[String, Bool]()
        for w in wanted:
            want[w] = True
        if not node.coalesce:
            # Right keys then appear in the output as well; do not prune.
            return None
        var keep_left = Dict[String, Bool]()
        var keep_right = Dict[String, Bool]()
        for k in node.names:
            keep_left[k] = True
        for k in node.right_keys:
            keep_right[k] = True
        for c in left_cols:
            if c in want:
                keep_left[c] = True
        if not membership:
            for c in right_cols:
                if c in keep_right:
                    continue
                var name = c + suffix if c in left_names else c
                if name in outputs:
                    return None
                outputs[name] = True
                if name in want:
                    keep_right[c] = True
                    if c in left_names:
                        keep_left[c] = True
        var left_keep = List[String]()
        for c in left_cols:
            if c in keep_left:
                left_keep.append(c)
        var right_keep = List[String]()
        for c in right_cols:
            if c in keep_right:
                right_keep.append(c)
        var sides = List[List[String]]()
        sides.append(left_keep^)
        sides.append(right_keep^)
        return sides^

    def _narrow_join_inputs(
        mut self,
        joins: List[Int],
        right_sides: List[Bool],
        columns: List[List[String]],
    ) raises:
        """Select only the kept columns above join inputs that are not scans.

        Scans already read only those columns. Other inputs (a filter, a
        computed column) would otherwise hand the join every column they
        produce. The selection is only added when it drops a column.
        """
        var added = False
        for k in range(len(joins)):
            var join = joins[k]
            var child = self._nodes[join].left
            if right_sides[k]:
                child = self._nodes[join].right
            if _is_scan(self._nodes[child].kind):
                continue
            var produced = self._columns_of(child)
            if len(columns[k]) >= len(produced):
                continue
            var exprs = List[Expr]()
            for name in columns[k]:
                exprs.append(col(name))
            self._nodes.append(_plan_node(SELECT, child, exprs=exprs))
            if right_sides[k]:
                self._nodes[join].right = len(self._nodes) - 1
            else:
                self._nodes[join].left = len(self._nodes) - 1
            added = True
        if added:
            self._reorder()

    def _label(self, index: Int) -> String:
        """The operator name `explain` prints for a node."""
        ref node = self._nodes[index]
        var label: String
        if node.kind == SCAN_FRAME:
            label = "SCAN frame"
        elif node.kind == SCAN_CSV:
            label = "SCAN CSV " + node.text
        elif node.kind == SCAN_PARQUET:
            label = "SCAN PARQUET " + node.text
        elif node.kind == FILTER:
            label = "FILTER"
        elif node.kind == SELECT:
            label = "SELECT " + _joined(_output_names(node.exprs))
        elif node.kind == WITH_COLUMNS:
            label = "WITH_COLUMNS " + _joined(_output_names(node.exprs))
        elif node.kind == AGG:
            label = (
                "GROUP_BY "
                + _joined(node.names)
                + " AGG "
                + _joined(_output_names(node.exprs))
            )
        elif node.kind == ASOF:
            label = "ASOF " + node.text + " " + _joined(node.names)
        elif node.kind == JOIN:
            label = "JOIN " + node.text + " on " + _key_pairs(node)
        elif node.kind == SORT and node.length >= 0:
            label = (
                "TOP_K " + String(node.length) + " by " + _joined(node.names)
            )
        elif node.kind == SORT:
            label = "SORT " + _joined(node.names)
        elif node.kind == SLICE:
            label = "SLICE " + String(node.offset) + " " + String(node.length)
        elif node.kind == UNIQUE:
            label = "UNIQUE " + _joined(node.names)
        elif node.kind == EXPLODE:
            label = "EXPLODE " + _joined(node.names)
        elif node.kind == UNNEST:
            label = "UNNEST " + node.text
        else:
            label = "DROP " + _joined(node.names)
        return label^

    def _describe(
        self, index: Int, depth: Int, mut out: String, streaming: Bool = True
    ):
        ref node = self._nodes[index]
        var pad = String("  ") * depth
        var label = self._label(index)
        if _is_scan(node.kind):
            if len(node.names) > 0:
                label += " [project " + _joined(node.names) + "]"
            if node.kind == SCAN_CSV and node.length >= 0:
                label += " [n_rows " + String(node.length) + "]"
        if streaming:
            if _is_scan(node.kind) or _stream_rows(node):
                label += " [stream]"
            elif (
                node.kind == AGG or node.kind == SELECT
            ) and _stream_reductions(node.exprs):
                label += " [stream aggregate state]"
            elif node.kind == JOIN and node.how in [
                JOIN_INNER,
                JOIN_LEFT,
                JOIN_SEMI,
                JOIN_ANTI,
                JOIN_CROSS,
            ]:
                if (
                    self._known_height(node.left) <= 65536
                    and (node.how == JOIN_INNER or node.how == JOIN_LEFT)
                    and prefer_left_build(
                        self._known_height(node.left),
                        self._known_height(node.right),
                    )
                ):
                    label += " [small left input; materialize unless compact progression]"
                else:
                    label += " [stream probe; materialize build]"
            elif node.kind == SORT and node.length >= 0:
                label += " [stream top-k state]"
            elif node.kind == SLICE and node.offset >= 0:
                label += " [ordered slice]"
            else:
                label += " [materialize]"
        out += pad + label + "\n"
        if node.left >= 0:
            self._describe(node.left, depth + 1, out, streaming)
        if node.right >= 0:
            self._describe(node.right, depth + 1, out, streaming)


def _row_steps_only(operations: List[PlanNode]) -> Bool:
    for operation in operations:
        if operation.kind not in [FILTER, SELECT, WITH_COLUMNS, DROP]:
            return False
    return True


def _filters(operations: List[PlanNode]) -> Bool:
    for operation in operations:
        if operation.kind == FILTER:
            return True
    return False


def _order_sensitive(expressions: List[Expr]) -> Bool:
    """Whether a reduction's answer depends on the order rows arrive
    (first, last, the position of a minimum or maximum): its states
    merge by row order, which workers taking morsels from a shared
    cursor do not keep."""
    for expression in expressions:
        for node in expression._nodes:
            if node.op in [FIRST, LAST, ARG_MIN, ARG_MAX]:
                return True
    return False


def _key_group_bound(
    keys: List[String], source: DataFrame, builds: List[DataFrame], limit: Int
) raises -> Int:
    """An upper bound on the groups of `keys`, or -1 once it passes `limit`
    or when a key is in no frame (made by a step). Keys held by one frame
    count together, as that frame's distinct key rows; the frames' counts
    multiply. A frame's height bounds its count for free, so frames are
    taken smallest first and sampled (evenly spaced rows, a sample with
    every row distinct standing for the whole height) only where the
    height alone would pass the limit."""
    # Each key's holder: -2 the source, else a build's index.
    var holders = List[Int]()
    for name in keys:
        var holder = -1
        if name in source.columns():
            holder = -2
        else:
            for b in range(len(builds)):
                if name in builds[b].columns():
                    holder = b
                    break
        if holder == -1:
            return -1
        holders.append(holder)
    var frames = List[Int]()
    for h in holders:
        if h not in frames:
            frames.append(h)
    # Smallest frame first.
    var heights = List[Int]()
    for h in frames:
        heights.append(source.height() if h == -2 else builds[h].height())
    for i in range(1, len(frames)):
        var j = i
        while j > 0 and heights[j - 1] > heights[j]:
            heights.swap_elements(j - 1, j)
            frames.swap_elements(j - 1, j)
            j -= 1
    var bound = 1
    for f in range(len(frames)):
        var height = heights[f]
        var count = max(height, 1)
        if bound * count > limit:
            # Only whether this frame's count passes what is left of the
            # limit matters, so the sample is twice that remainder: a
            # sample with every row distinct ends the bound (q46's three
            # source keys over 2.9M rows: six rows, not 4,096, whose
            # strided reads took 1.3 ms).
            var left = limit // bound
            var sample = min(height, 4096, 2 * (left + 1))
            var rows = List[Int](capacity=sample)
            for k in range(sample):
                rows.append(k * height // sample)
            var picked = List[Series]()
            for i in range(len(keys)):
                if holders[i] != frames[f]:
                    continue
                if frames[f] == -2:
                    picked.append(source.column(keys[i]).take(rows))
                else:
                    picked.append(builds[frames[f]].column(keys[i]).take(rows))
            var distinct = encode_rows(picked, nulls_equal=True).count()
            count = height if distinct == sample else max(distinct, 1)
        bound *= count
        if bound > limit:
            return -1
    return bound


def _late_key_build(
    name: String,
    source_columns: List[String],
    builds: List[DataFrame],
    operations: List[PlanNode],
    expressions: List[Expr],
) raises -> Int:
    """The build frame a group key `name` comes from when the key can be
    grouped by its codes instead of its values: a String column of exactly
    one inner or left join's build side, not in the source, not a join key,
    read by no step or reduction (a plain pass-through select aside) and
    made by none. -1 otherwise."""
    if name in source_columns:
        return -1
    var holder = -1
    for b in range(len(builds)):
        if name in builds[b].columns():
            if holder >= 0:
                return -1
            holder = b
    if holder < 0:
        return -1
    if builds[holder].column(name).dtype() != DataType.STRING:
        return -1
    for expression in expressions:
        var reads = _references(expression)
        if not reads or name in reads.value():
            return -1
    for operation in operations:
        if operation.kind == JOIN:
            if name in operation.names:
                return -1
            if operation.offset == holder:
                if operation.how != JOIN_INNER and operation.how != JOIN_LEFT:
                    return -1
                if name in operation.right_keys:
                    return -1
            continue
        if operation.kind == DROP:
            if name in operation.names:
                return -1
            continue
        for e in operation.exprs:
            # A select passing the column through unchanged is the one
            # step that may read or name it.
            var passes = (
                operation.kind == SELECT
                and len(e._nodes) == 1
                and e._nodes[0].op == COL
                and e._nodes[0].text == name
                and e._name == name
            )
            if passes:
                continue
            var reads = _references(e)
            if not reads or name in reads.value() or e._name == name:
                return -1
    return holder


def _counts_distinct(expressions: List[Expr]) -> Bool:
    for expression in expressions:
        for node in expression._nodes:
            if node.op == N_UNIQUE:
                return True
    return False


def _order_insensitive(exprs: List[Expr]) -> Bool:
    """Whether aggregates give the same values whatever order their input
    rows arrive in: no window, and only reductions that read no position
    (so not first, last, arg_min, arg_max or a list of the rows)."""
    for e in exprs:
        for node in e._nodes:
            if is_window(node.op) or node.op == OVER:
                return False
            if is_reduction(node.op) and node.op not in [
                SUM,
                COUNT,
                MIN,
                MAX,
                MEAN,
                STD,
                VAR,
                LEN,
                ANY,
                ALL,
                NULL_COUNT,
                N_UNIQUE,
                MEDIAN,
                QUANTILE,
                CORR,
                COV,
            ]:
                return False
    return True


def _chain_member(node: PlanNode) -> Bool:
    """A step `_order_chain` can move: an inner join with coalesced keys, a
    row-local filter, or a selection of plain columns."""
    if node.kind == JOIN:
        return node.how == JOIN_INNER and node.coalesce and len(node.names) > 0
    if node.kind == FILTER:
        return _row_local(node.exprs)
    if node.kind == SELECT:
        for e in node.exprs:
            if len(e._nodes) != 1 or e._nodes[0].op != COL:
                return False
            if e._nodes[0].text != e._name:
                return False
        return True
    return False


def _is_scan(kind: Int) -> Bool:
    return kind == SCAN_FRAME or kind == SCAN_CSV or kind == SCAN_PARQUET


def _text_hash(text: String) -> UInt64:
    """FNV-1a over the text's bytes."""
    var h = UInt64(0xCBF29CE484222325)
    for b in text.as_bytes():
        h = (h ^ UInt64(b)) * 0x100000001B3
    return h


def _expr_signature(expr: Expr) -> String:
    """An expression as text, node by node, for `_signature`."""
    var text = expr._name
    for n in expr._nodes:
        text += (
            ";"
            + String(n.op)
            + ","
            + String(n.left)
            + ","
            + String(n.right)
            + ","
            + n.text
            + ","
            + String(n.integer)
            + ","
            + String(n.floating)
            + ","
            + String(n.min_count)
            + ","
            + String(n.extra)
            + ","
            + n.text2
        )
        for d in n.dtypes:
            text += "," + (d.value().name() if d else "_")
    return text^


def _joined(names: List[String]) -> String:
    var out = String()
    for i in range(len(names)):
        if i > 0:
            out += ", "
        out += names[i]
    return out^


def _key_set_filter(
    keys: Series, name: String, dtype: DataType
) raises -> Optional[Expr]:
    """A filter keeping the rows whose `name` is among `keys`' non-null
    values: the values themselves when there are at most 50, else their
    min-max range. None when the dtypes differ or the key is neither an
    integer (other than UInt64) nor a string."""
    if keys.dtype() != dtype:
        return None
    var distinct = keys.unique()
    if distinct.is_chunked():
        distinct = distinct.rechunk()
    if dtype == DataType.STRING:
        var values = List[String]()
        for i in range(len(distinct)):
            var cell = distinct.get(i)
            if not cell.is_null():
                values.append(cell.string())
        if len(values) > 50:
            return None
        if len(values) == 0:
            return Optional(lit(False))
        return Optional(col(name).is_in(values))
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        comptime if D.is_integral() and D != DType.uint64:
            if distinct._data.isa[Column[Scalar[D]]]():
                ref column = distinct._data[Column[Scalar[D]]]
                # Bounds first: a literal per value only for a short list
                # (PDS-H q21 built 729K literals for a range).
                var count = 0
                var low = Scalar[D].MAX
                var high = Scalar[D].MIN
                for i in range(len(column)):
                    if not column._valid(i):
                        continue
                    var value = column._get(i)
                    low = min(low, value)
                    high = max(high, value)
                    count += 1
                if count == 0:
                    return Optional(lit(False))
                if count > 50:
                    return Optional(
                        (col(name) >= lit(low)) & (col(name) <= lit(high))
                    )
                var literals = List[Expr](capacity=count)
                for i in range(len(column)):
                    if column._valid(i):
                        literals.append(lit(column._get(i)))
                return Optional(col(name).is_in(literals))
    return None


def _covers(columns: List[String], reads: List[String]) -> Bool:
    for name in reads:
        var found = False
        for c in columns:
            found = found or c == name
        if not found:
            return False
    return True


def _key_pairs(node: PlanNode) -> String:
    """`a` for keys named alike, `a = b` for differently named pairs."""
    var out = String()
    for i in range(len(node.names)):
        if i > 0:
            out += ", "
        out += node.names[i]
        if i < len(node.right_keys) and node.right_keys[i] != node.names[i]:
            out += " = " + node.right_keys[i]
    return out^


def _covers_exclusive(
    columns: List[String], other: List[String], reads: List[String]
) -> Bool:
    """Right-side names that the join output does not rename or shadow."""
    for name in reads:
        for c in other:
            if c == name:
                return False
    return _covers(columns, reads)


@fieldwise_init
struct LazyGroupBy(Copyable):
    """A pending lazy grouping; finish it with agg."""

    var _frame: LazyFrame
    var _keys: List[String]
    var _maintain_order: Bool

    def agg(self, exprs: List[Expr]) -> LazyFrame:
        return self._frame._push(
            _plan_node(
                AGG,
                exprs=exprs,
                names=self._keys,
                maintain_order=self._maintain_order,
            )
        )

    def agg(self, expr: Expr) -> LazyFrame:
        return self.agg([expr.copy()])


def scan_csv(path: String) raises -> LazyFrame:
    """Lazily read a CSV file with an inferred schema. Nothing is read until
    collect; projection and head() are pushed into the reader."""
    # Scan slots are parallel: frames[k] and schemas[k] belong to one scan.
    return LazyFrame(
        [_plan_node(SCAN_CSV, text=path, offset=0)],
        [DataFrame([])],
        [Optional[CsvSchema]()],
    )


def scan_csv(path: String, schema: CsvSchema) raises -> LazyFrame:
    return LazyFrame(
        [_plan_node(SCAN_CSV, text=path, offset=0)],
        [DataFrame([])],
        [Optional[CsvSchema](schema.copy())],
    )


def scan_parquet(path: String) raises -> LazyFrame:
    """Lazily read a Parquet file. Nothing is read until collect; projection
    is pushed into the reader, and a filter directly above the scan decodes
    only the row groups whose footer statistics can hold a match."""
    return LazyFrame(
        [_plan_node(SCAN_PARQUET, text=path, offset=0)],
        [DataFrame([])],
        [Optional[CsvSchema]()],
    )
