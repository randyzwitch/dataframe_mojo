"""Thread-owned pipelines over morsels (docs/executor-design.md, #538).

A pipeline is a source, a list of row-local steps and a sink. Every worker
thread runs the whole pipeline on the morsels it takes from the source's
cursor and pushes into its own sink state; the states are combined once
when the source is exhausted. The main thread does no per-morsel work.
DuckDB's `PipelineExecutor` runs one per thread the same way, and Polars'
streaming engine one task per lane per node.

A morsel is a frame and a selection. A filter narrows the selection and
copies nothing; a column is gathered through the selection when a step
reads it, once, and the gathered column replaces the shared one. DuckDB
slices a chunk into dictionary vectors the same way.
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import ArcPointer, Pointer
from std.sys import size_of

from .bool_column import BoolColumn, both_true, true_count
from .expr import COL, SELECTOR, Expr
from .frame import DataFrame, _StreamReduction, concat
from .gather import can_filter_aligned_chunks, true_rows
from .mask_filter import filter_columns
from .parallel import Job, run_jobs
from .series import Series
from .trace import trace_path

# Step kinds, the row-local plan nodes a pipeline runs between its source
# and its sink. Numbered here so this module does not import the planner.
comptime STEP_FILTER = 0
comptime STEP_SELECT = 1
comptime STEP_WITH_COLUMNS = 2
comptime STEP_DROP = 3


@fieldwise_init
struct Step(Copyable, Movable):
    var kind: Int
    var exprs: List[Expr]
    var names: List[String]


def _reads(exprs: List[Expr]) -> Optional[List[String]]:
    """The columns the expressions read, or None when a selector may read
    any column."""
    var names = List[String]()
    for expr in exprs:
        for node in expr._nodes:
            if node.op == SELECTOR:
                return None
            if node.op == COL and node.text not in names:
                names.append(node.text)
    return Optional(names^)


struct Morsel(Movable):
    """A frame and a selection over its rows.

    Columns not yet `compact` are the source's, shared, and read through
    `mask`; a compact column already holds only the selected rows.
    """

    var frame: DataFrame
    var mask: BoolColumn
    var selected: Bool
    var count: Int
    var compact: List[Bool]
    var sequence: Int

    def __init__(out self, var frame: DataFrame, sequence: Int):
        self.count = frame.height()
        self.compact = List[Bool](length=frame.width(), fill=True)
        self.frame = frame^
        self.mask = BoolColumn(List[Bool]())
        self.selected = False
        self.sequence = sequence

    def height(self) -> Int:
        return self.count

    def _index(self, name: String) raises -> Int:
        for i in range(self.frame.width()):
            if self.frame._columns[i].name() == name:
                return i
        raise Error("Column not found: " + name)

    def gather(mut self, names: List[String]) raises:
        """Make the named columns compact: gathered through the selection
        in one pass over the mask for all of them."""
        if not self.selected:
            return
        var indices = List[Int]()
        var pending = List[Series]()
        for name in names:
            var i = self._index(name)
            if not self.compact[i]:
                indices.append(i)
                pending.append(self.frame._columns[i].copy())
        if len(pending) == 0:
            return
        var filtered = filter_columns(pending, self.mask)
        ref others = filtered[1]
        if len(others) > 0:
            # String and nested columns: the row-list route.
            var rows = true_rows(self.mask)
            var next_fixed = 0
            var next_other = 0
            for k in range(len(indices)):
                if next_other < len(others) and others[next_other] == k:
                    self.frame._columns[indices[k]] = pending[k].take(rows)
                    next_other += 1
                else:
                    self.frame._columns[indices[k]] = filtered[0][
                        next_fixed
                    ].copy()
                    next_fixed += 1
        else:
            for k in range(len(indices)):
                self.frame._columns[indices[k]] = filtered[0][k].copy()
        for i in indices:
            self.compact[i] = True

    def gather_all(mut self) raises:
        """Every column compact; the selection is then dropped."""
        if not self.selected:
            return
        var names = List[String]()
        for i in range(self.frame.width()):
            if not self.compact[i]:
                names.append(self.frame._columns[i].name())
        self.gather(names)
        self.frame._height = self.count
        self.selected = False

    def _compact_frame(mut self, names: List[String]) raises -> DataFrame:
        """The named columns, compact, as a frame of `count` rows."""
        self.gather(names)
        var columns = List[Series](capacity=len(names))
        for name in names:
            columns.append(self.frame._columns[self._index(name)].copy())
        return DataFrame(columns^, height=self.count)

    def filter(mut self, predicate: Expr, batch_size: Int) raises:
        """Narrow the selection to the rows `predicate` keeps."""
        var reads = _reads([predicate.copy()])
        var names: List[String]
        if reads:
            names = reads.take()
        else:
            names = self.frame.columns()
        var narrowed: BoolColumn
        if not self.selected:
            var input = self._compact_frame(names)
            narrowed = input._selection(predicate, batch_size)
            var kept = true_count(narrowed)
            if kept == self.count:
                return
            self.mask = narrowed^
            self.selected = True
            self.count = kept
            for i in range(len(self.compact)):
                self.compact[i] = False
            return
        var input = self._compact_frame(names)
        narrowed = input._selection(predicate, batch_size)
        var kept = true_count(narrowed)
        if kept == self.count:
            return
        # Compact columns are at the old selection's rows: filter them by
        # the new mask directly. Shared columns are read through the
        # composed mask: the new mask's bits spread over the old mask's
        # set bits.
        var compact_columns = List[Series]()
        var compact_at = List[Int]()
        for i in range(self.frame.width()):
            if self.compact[i]:
                compact_at.append(i)
                compact_columns.append(self.frame._columns[i].copy())
        if len(compact_columns) > 0:
            var sliced = DataFrame(compact_columns^, height=self.count).filter(
                narrowed
            )
            for k in range(len(compact_at)):
                self.frame._columns[compact_at[k]] = sliced._columns[k].copy()
        self.mask = _spread(self.mask, narrowed)
        self.count = kept

    def select(mut self, exprs: List[Expr], batch_size: Int) raises:
        var reads = _reads(exprs)
        var names: List[String]
        if reads:
            names = reads.take()
        else:
            names = self.frame.columns()
        var input = self._compact_frame(names)
        var result = input.select_exprs(exprs, batch_size=batch_size)
        self.frame = result^
        self.count = self.frame.height()
        self.compact = List[Bool](length=self.frame.width(), fill=True)
        self.selected = False

    def with_columns(mut self, exprs: List[Expr], batch_size: Int) raises:
        var reads = _reads(exprs)
        var names: List[String]
        if reads:
            names = reads.take()
        else:
            names = self.frame.columns()
        var input = self._compact_frame(names)
        var result = input.with_columns(exprs, batch_size=batch_size)
        # Columns the expressions produced or replaced join the frame as
        # compact; the rest stay as they are.
        for c in range(result.width()):
            ref column = result._columns[c]
            var found = -1
            for i in range(self.frame.width()):
                if self.frame._columns[i].name() == column.name():
                    found = i
            if found >= 0:
                self.frame._columns[found] = column.copy()
                self.compact[found] = True
            else:
                self.frame._columns.append(column.copy())
                self.compact.append(True)

    def drop(mut self, names: List[String]) raises:
        var kept = List[Series]()
        var kept_compact = List[Bool]()
        for i in range(self.frame.width()):
            if self.frame._columns[i].name() not in names:
                kept.append(self.frame._columns[i].copy())
                kept_compact.append(self.compact[i])
        var height = self.frame._height
        self.frame = DataFrame(kept^, height=height)
        self.compact = kept_compact^

    def apply(mut self, step: Step, batch_size: Int) raises:
        if step.kind == STEP_FILTER:
            self.filter(step.exprs[0], batch_size)
        elif step.kind == STEP_SELECT:
            self.select(step.exprs, batch_size)
        elif step.kind == STEP_WITH_COLUMNS:
            self.with_columns(step.exprs, batch_size)
        else:
            self.drop(step.names)

    def materialized(mut self) raises -> DataFrame:
        self.gather_all()
        return self.frame.copy()


def _spread(outer: BoolColumn, inner: BoolColumn) raises -> BoolColumn:
    """`outer` with its set bits replaced, in order, by `inner`'s bits:
    the selection of a filter applied to an already selected morsel."""
    var values = List[Bool](capacity=len(outer))
    var next = 0
    for r in range(len(outer)):
        if outer._get(r) and outer._valid(r):
            values.append(inner._get(next) and inner._valid(next))
            next += 1
        else:
            values.append(False)
    return BoolColumn(values^)


@fieldwise_init
struct _Cursor(Copyable, Movable):
    """The source's next morsel offset, shared by every worker: an atomic
    in C-allocated memory, as the thread budget of `parallel` keeps its
    counters."""

    var address: Int

    @staticmethod
    def new() -> _Cursor:
        return _Cursor(
            external_call["calloc", Int](1, size_of[Atomic[Int64]]())
        )

    def take(self, rows: Int) -> Int:
        return Int(
            Pointer[Atomic[Int64], MutAnyOrigin](
                unsafe_from_address=self.address
            )[].fetch_add(Int64(rows))
        )

    def free(self):
        _ = external_call["free", NoneType](self.address)


struct _PipelineJob(Job):
    """One worker's run of the pipeline: morsels from the shared cursor
    through every step into this worker's own sink state."""

    var frame: ArcPointer[DataFrame]
    var cursor: Int
    # Morsel ranges as (offset, length) pairs, none crossing a chunk
    # boundary of the frame's columns (a slice within one Parquet row
    # group is a view; one across two is copied when a kernel reads it).
    var ranges: ArcPointer[List[Int]]
    var steps: List[Step]
    var expressions: List[Expr]
    var batch_size: Int
    # Materialize sink: the morsels' frames and their sequence numbers.
    var parts: List[DataFrame]
    var sequences: List[Int]
    # Reduce sink: this worker's merged state.
    var reduction: List[_StreamReduction]
    var rows: Int
    # Rows fed, then each step's input and output rows, for the report.
    var counts: List[Int]

    def __init__(
        out self,
        frame: ArcPointer[DataFrame],
        cursor: Int,
        ranges: ArcPointer[List[Int]],
        steps: List[Step],
        expressions: List[Expr],
        batch_size: Int,
    ):
        self.frame = frame.copy()
        self.cursor = cursor
        self.ranges = ranges.copy()
        self.steps = steps.copy()
        self.expressions = expressions.copy()
        self.batch_size = batch_size
        self.parts = List[DataFrame]()
        self.sequences = List[Int]()
        self.reduction = List[_StreamReduction]()
        self.rows = 0
        self.counts = List[Int](length=2 * len(steps) + 1, fill=0)

    def run(mut self) raises:
        var count = len(self.ranges[]) // 2
        var cursor = _Cursor(self.cursor)
        while True:
            var k = cursor.take(1)
            if k >= count:
                return
            var offset = self.ranges[][2 * k]
            var length = self.ranges[][2 * k + 1]
            var morsel = Morsel(self.frame[].slice(offset, length), k)
            self.counts[0] += length
            for i in range(len(self.steps)):
                self.counts[2 * i + 1] += morsel.height()
                morsel.apply(self.steps[i], self.batch_size)
                self.counts[2 * i + 2] += morsel.height()
                if morsel.height() == 0:
                    break
            if morsel.height() == 0:
                continue
            if len(self.expressions) > 0:
                var reads = _reads(self.expressions)
                var input: DataFrame
                if reads:
                    input = morsel._compact_frame(reads.take())
                else:
                    input = morsel.materialized()
                var state = _StreamReduction(
                    input, self.expressions, List[String]()
                )
                self.rows += state.rows
                if len(self.reduction) == 0:
                    self.reduction.append(state^)
                else:
                    self.reduction[0].merge(state)
            else:
                self.parts.append(morsel.materialized())
                self.sequences.append(morsel.sequence)


def _morsel_ranges(
    frame: DataFrame, morsel_rows: Int, workers: Int
) -> List[Int]:
    """(offset, length) pairs covering the frame: each chunk of columns
    chunked at the same rows is cut into even pieces of about
    `morsel_rows`, and a frame without such chunks likewise. When that
    gives fewer than four morsels a worker, the pieces are cut finer so
    their count is a multiple of `workers` and the last round of morsels
    is as full as the first: ten even pieces of 600K rows on eight
    workers left two workers a whole second round (PDS-H q12 at scale
    0.1, 1.8 to 2.1 ms). Many morsels are left as they are: each costs
    its steps' setup (ClickBench q29's ninety sums, 36 morsels of 312K
    rows against 144 of 69K, 4 to 6 ms)."""
    var ends = List[Int]()
    if can_filter_aligned_chunks(frame._columns):
        ends = frame._columns[0]._chunked.value()[].ends.copy()
    else:
        ends.append(frame.height())
    var lanes = max(1, workers)
    var lengths = List[Int]()
    var starts = List[Int]()
    var pieces = List[Int]()
    var total = 0
    var start = 0
    for end in ends:
        var length = end - start
        if length > 0:
            starts.append(start)
            lengths.append(length)
            # Pieces of about `morsel_rows`: a chunk a little longer than
            # the target stays one morsel rather than two short ones, each
            # paying the steps' setup (TPC-DS q9's fifteen conditional
            # sums over 123K-row groups cut at 90K: 145 to 174 ms on one
            # thread).
            pieces.append(
                max(1, (2 * length + morsel_rows) // (2 * morsel_rows))
            )
            total += pieces[len(pieces) - 1]
        start = end
    if total > 1 and total < 4 * lanes and total % lanes != 0:
        # Cut every chunk finer by the same factor, then give the longest
        # chunks one more piece each until the count is a multiple.
        var wanted = total + lanes - total % lanes
        var scaled = 0
        for c in range(len(pieces)):
            pieces[c] = min(lengths[c], pieces[c] * wanted // total)
            scaled += pieces[c]
        while scaled < wanted:
            var longest = -1
            for c in range(len(pieces)):
                if pieces[c] < lengths[c] and (
                    longest < 0
                    or lengths[c] // pieces[c]
                    > lengths[longest] // pieces[longest]
                ):
                    longest = c
            if longest < 0:
                break
            pieces[longest] += 1
            scaled += 1
    # Boundaries inside a chunk fall on multiples of 64 rows, so a
    # morsel's validity bitmaps start on a byte: kernels that AND or
    # window bitmaps take their slow path from a bit offset (TPC-DS q9's
    # masked sums, 145 to 174 ms on one thread at 87,381-row morsels).
    var ranges = List[Int]()
    for c in range(len(pieces)):
        for p in range(pieces[c]):
            var low = starts[c] + lengths[c] * p // pieces[c]
            var high = starts[c] + lengths[c] * (p + 1) // pieces[c]
            if p > 0:
                low = starts[c] + (low - starts[c]) // 64 * 64
            if p + 1 < pieces[c]:
                high = starts[c] + (high - starts[c]) // 64 * 64
            if high > low:
                ranges.append(low)
                ranges.append(high - low)
    return ranges^


def run_pipeline(
    frame: DataFrame,
    steps: List[Step],
    expressions: List[Expr],
    workers: Int,
    morsel_rows: Int,
    batch_size: Int,
    mut counts: List[Int],
) raises -> DataFrame:
    """Run the steps over `frame` in morsels on `workers` threads, into a
    materialize sink, or a reduce sink when `expressions` are ungrouped
    reductions over the steps' output. `counts` receives the rows fed,
    then each step's input and output rows."""
    trace_path("pipeline.run")
    var shared = ArcPointer(frame.copy())
    var ranges = ArcPointer(
        _morsel_ranges(frame, max(1, morsel_rows), max(1, workers))
    )
    var cursor = _Cursor.new()
    var jobs = List[_PipelineJob](capacity=workers)
    for _ in range(max(1, workers)):
        jobs.append(
            _PipelineJob(
                shared, cursor.address, ranges, steps, expressions, batch_size
            )
        )
    try:
        run_jobs(jobs)
    except e:
        cursor.free()
        raise e
    cursor.free()
    counts = List[Int](length=2 * len(steps) + 1, fill=0)
    for j in range(len(jobs)):
        for i in range(len(counts)):
            counts[i] += jobs[j].counts[i]
    if len(expressions) > 0:
        var merged = List[_StreamReduction]()
        for j in range(len(jobs)):
            if len(jobs[j].reduction) == 0:
                continue
            if len(merged) == 0:
                merged.append(jobs[j].reduction.pop())
            else:
                merged[0].merge(jobs[j].reduction[0])
        if len(merged) == 0:
            # No row survived: reduce an empty frame of the right schema.
            var empty = frame.clear()
            for step in steps:
                var morsel = Morsel(empty.copy(), 0)
                morsel.apply(step, batch_size)
                empty = morsel.materialized()
            return _StreamReduction(empty, expressions, List[String]()).finish()
        return merged[0].finish()
    # Morsels in source order: each worker's parts are in the order it
    # took them, so a merge by sequence over the workers' lists orders all.
    var parts = List[DataFrame]()
    var next = List[Int](length=len(jobs), fill=0)
    while True:
        var best = -1
        var best_sequence = 0
        for j in range(len(jobs)):
            if next[j] < len(jobs[j].sequences):
                var sequence = jobs[j].sequences[next[j]]
                if best < 0 or sequence < best_sequence:
                    best = j
                    best_sequence = sequence
        if best < 0:
            break
        parts.append(jobs[best].parts[next[best]].copy())
        next[best] += 1
    if len(parts) == 0:
        var empty = frame.clear()
        for step in steps:
            var morsel = Morsel(empty.copy(), 0)
            morsel.apply(step, batch_size)
            empty = morsel.materialized()
        return empty^
    if len(parts) == 1:
        return parts[0].copy()
    return concat(parts)
