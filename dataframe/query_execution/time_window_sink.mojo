"""Time windows, `group_by_dynamic` and `rolling`, run as rounds of tasks of
the query's schedule rather than as one task that runs whole-frame kernels.

A time window aggregates the rows whose index time falls in a window: a
calendar-aligned span for `group_by_dynamic`, a span ending at each row's
time for `rolling`. Windows may overlap, so a row can belong to several of
them, and every window's rows are only known once the whole input has
been read and each group's rows are in time order. The work is in three
parts:

- the serial step first checks the request, numbers the `group_by` keys
  and puts each group's rows in time order (a group already in order is
  left as it is).
- a round lists the windows: each task takes whole groups, or for
  `rolling` ranges of one group's rows (each row's window is found by
  binary search over its group's times, independent of the others), and
  lists each window's label, first row and rows.
- a round aggregates them: the windows are cut into contiguous ranges of
  about the same number of rows, and each task gathers its range's rows
  and reduces them with the grouped reducers, a window a group. Its
  output columns are one chunk of the result's, in window order.
"""
from std.collections import Dict, Optional

from dataframe.aggregation.expressions import _bind_all
from dataframe.aggregation.keys import _expand_struct_keys
from dataframe.binding import AGGREGATE, BoundExpr
from dataframe.execution import evaluate
from dataframe.frame import DataFrame
from dataframe.kernels.hashing import encode_rows
from dataframe.parallel import (
    Rounds,
    partitions,
    round_width,
    run_rounds,
    worker_count,
)
from dataframe.planning.model import PlanNode, ROLLING
from dataframe.dtype import DataType
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.time.temporal import parse_interval, zone_of
from dataframe.time.timezone import TimeZone
from dataframe.time.windows import (
    check_time_index,
    closed_sides,
    negate_period,
    positive_period,
    search_time,
    shift_time,
    sort_time_groups,
    truncate_time,
)
from dataframe.window import partitions as window_partitions

# The round that ran last.
comptime _START = 0
comptime _WINDOWS = 1
comptime _AGGREGATE = 2
comptime _DONE = 3


def _fixed_width(interval: String, dtype: DataType) raises -> Bool:
    """Whether shifting a time of `dtype` by `interval` always moves it by
    the same amount: no months (their lengths vary), no calendar days in a
    time zone (23 or 25 hours across a clock change), and for a Date no
    unit below a day (a Date holds whole days)."""
    var parts = parse_interval(interval)
    if parts[0] != 0:
        return False
    if parts[1] != 0 and dtype.is_datetime() and dtype.time_zone() != "":
        return False
    return not (parts[2] != 0 and dtype.is_date())


def time_window_bound(
    input: DataFrame, node: PlanNode
) raises -> List[BoundExpr]:
    """Check time window node `node` (`group_by_dynamic` or `rolling`)
    against its input's columns, raising what the request gets wrong, and
    bind its aggregations."""
    var rolling = node.kind == ROLLING
    positive_period(node.window_period())
    _ = parse_interval(node.window_offset())
    _ = closed_sides(node.window_closed())
    if node.window_label() not in ["left", "right", "datapoint"]:
        raise Error("label must be left, right or datapoint")
    if node.window_batch_size() <= 0:
        raise Error("batch_size must be positive")
    var index = input.column(node.window_columns()[0])
    if not (index.dtype().is_date() or index.dtype().is_datetime()):
        raise Error(
            "time windows require a Date or Datetime index: " + index.name()
        )
    ref names = node.window_columns()
    for k in range(1, len(names)):
        for j in range(k):
            if names[j] == names[k]:
                raise Error("duplicate time grouping key: " + names[k])
        _ = input.column(names[k])
    if not rolling:
        positive_period(node.window_every())
    var bound = _bind_all(node.exprs, input._columns)
    for expression in bound:
        if expression.shape() != AGGREGATE:
            raise Error("Time grouping requires scalar aggregate expressions")
        if expression.expr._name in names:
            raise Error(
                "Aggregate output name collides with time grouping key: "
                + expression.expr._name
            )
    return bound^


@always_inline
def _skip_to(start: Int64, time: Int64, step: Int64, span: Int64) -> Int64:
    """The latest window start reached from `start` in whole steps of
    `step` whose window (of length `span`) cannot reach past `time`: every
    window skipped ends before `time`, so it holds no row at or after
    `time`, and none of the windows that can hold `time` is skipped."""
    var gap = time - start
    if gap <= step + span:
        return start
    return start + ((gap - span) // step) * step


struct TimeWindowSink(Movable, Rounds):
    """Time window node `node` over all rows of `input`, as rounds (see the
    module docstring); `take_result` once no round is left."""

    var input: DataFrame
    var node: PlanNode
    var stage: Int
    var height: Int
    var bound: List[BoundExpr]
    var index: Series
    var keys: List[Series]
    # Each group's rows in time order and, for a rolling window over
    # groups that were not, in row order (the order of its output).
    var groups: List[List[Int]]
    var inputs: List[List[Int]]
    # The window round's pieces (group, first and end position in the
    # group's output order) and by piece its windows: their labels, first
    # rows, rows, and where each window's rows start in those.
    var pieces: List[Tuple[Int, Int, Int]]
    var labels: List[List[Int64]]
    var firsts: List[List[Int]]
    var rows: List[List[Int]]
    var starts: List[List[Int]]
    # The windows before each piece, and the first window of each
    # aggregate task (then the window count); by task its output columns.
    var before: List[Int]
    var cuts: List[Int]
    var parts: List[List[Series]]
    var width: Int
    var result: DataFrame

    def __init__(out self, var input: DataFrame, node: PlanNode) raises:
        self.height = input.height()
        self.input = input^
        self.node = node.copy()
        self.stage = _START
        self.bound = List[BoundExpr]()
        self.index = Series("", Column[Int64](List[Int64]()))
        self.keys = List[Series]()
        self.groups = List[List[Int]]()
        self.inputs = List[List[Int]]()
        self.pieces = List[Tuple[Int, Int, Int]]()
        self.labels = List[List[Int64]]()
        self.firsts = List[List[Int]]()
        self.rows = List[List[Int]]()
        self.starts = List[List[Int]]()
        self.before = List[Int]()
        self.cuts = List[Int]()
        self.parts = List[List[Series]]()
        self.width = 0
        self.result = DataFrame(List[Series]())

    def next_round(mut self) raises -> Int:
        if self.stage == _START:
            return self._order()
        if self.stage == _WINDOWS:
            return self._cut()
        if self.stage == _AGGREGATE:
            self._assemble()
        return 0

    def round_task(mut self, task: Int) raises:
        if self.stage == _WINDOWS:
            for p in range(task, len(self.pieces), self.width):
                if self.node.kind == ROLLING:
                    self._rolling(p)
                else:
                    self._dynamic(p)
        elif self.stage == _AGGREGATE:
            for t in range(task, len(self.cuts) - 1, self.width):
                self._aggregate(t)

    def _order(mut self) raises -> Int:
        """The checks, each group's rows in time order, and the round that
        lists the windows."""
        self.bound = time_window_bound(self.input, self.node)
        # The aggregate tasks gather from whole arrays, none of them
        # rechunking a shared column.
        var columns = List[Series](capacity=self.input.width())
        for column in self.input._columns:
            columns.append(column.rechunk())
        self.input = DataFrame(columns^, height=self.height)
        ref names = self.node.window_columns()
        self.index = self.input.column(names[0])
        for k in range(1, len(names)):
            self.keys.append(self.input.column(names[k]))
        var ids = List[Int]()
        if len(self.keys):
            ids = encode_rows(_expand_struct_keys(self.keys), True).ids.copy()
        self.groups = window_partitions(self.height, ids)
        if not check_time_index(self.index, self.groups):
            # Rolling output rows still follow input row order within
            # each group.
            if self.node.kind == ROLLING:
                self.inputs = self.groups.copy()
            sort_time_groups(self.index, self.groups)
        for g in range(len(self.groups)):
            var n = len(self.groups[g])
            if n == 0:
                continue
            if self.node.kind != ROLLING:
                self.pieces.append((g, 0, n))
                continue
            # Each row's window is found on its own: a large group splits
            # into ranges of its rows.
            var bounds = partitions(n, worker_count(n), 64)
            for w in range(len(bounds) - 1):
                if bounds[w] < bounds[w + 1]:
                    self.pieces.append((g, bounds[w], bounds[w + 1]))
        var n = len(self.pieces)
        self.labels = List[List[Int64]](length=n, fill=List[Int64]())
        self.firsts = List[List[Int]](length=n, fill=List[Int]())
        self.rows = List[List[Int]](length=n, fill=List[Int]())
        self.starts = List[List[Int]](length=n, fill=List[Int]())
        self.stage = _WINDOWS
        self.width = round_width(n)
        if self.width == 0:
            return self._cut()
        return self.width

    def _rolling(mut self, p: Int) raises:
        """Piece `p`'s windows: one for each of its rows, from the row's
        time back by the period (or from its offset)."""
        var piece = self.pieces[p]
        ref rows = self.groups[piece[0]]
        var dtype = self.index.dtype()
        var zone = zone_of(dtype)
        ref times = self.index._data[Column[Int64]]
        var sides = closed_sides(self.node.window_closed())
        var period = self.node.window_period()
        var offset = self.node.window_offset()
        var default_offset = offset == negate_period(period)
        var labels = List[Int64](capacity=piece[2] - piece[1])
        var firsts = List[Int](capacity=piece[2] - piece[1])
        var gathered = List[Int]()
        var starts = List[Int](capacity=piece[2] - piece[1] + 1)
        starts.append(0)
        var ordered = len(self.inputs) == 0
        for j in range(piece[1], piece[2]):
            var row = rows[j] if ordered else self.inputs[piece[0]][j]
            var time = times._get(row)
            var start = shift_time(time, dtype, offset, zone)
            # Calendar month subtraction then addition need not round-trip.
            var end = time if default_offset else shift_time(
                start, dtype, period, zone
            )
            var lo = search_time(times, rows, start, not sides[0])
            var hi = search_time(times, rows, end, sides[1])
            labels.append(time)
            firsts.append(row)
            for k in range(lo, hi):
                gathered.append(rows[k])
            starts.append(len(gathered))
        self.labels[p] = labels^
        self.firsts[p] = firsts^
        self.rows[p] = gathered^
        self.starts[p] = starts^

    def _dynamic(mut self, p: Int) raises:
        """Piece `p`'s windows: the calendar-aligned windows of one group
        that hold some of its rows."""
        ref rows = self.groups[self.pieces[p][0]]
        var dtype = self.index.dtype()
        var zone = zone_of(dtype)
        ref times = self.index._data[Column[Int64]]
        var sides = closed_sides(self.node.window_closed())
        var every = self.node.window_every()
        var period = self.node.window_period()
        var label = self.node.window_label()
        var labels = List[Int64]()
        var firsts = List[Int]()
        var gathered = List[Int]()
        var starts = List[Int]()
        starts.append(0)
        var first = times._get(rows[0])
        var last = times._get(rows[len(rows) - 1])
        var origin = shift_time(
            truncate_time(first, dtype, every, zone),
            dtype,
            self.node.window_offset(),
            zone,
        )
        # Window k starts `every` times k after the origin, each computed
        # from the origin: stepping from the previous start instead would
        # carry a month-end clamp forward (Jan 30, Feb 28, Mar 28, ...).
        # When every and period move a time by a constant amount, start k
        # is the origin plus k steps, and the windows that hold no row up
        # to some time can be stepped over in one move: work then follows
        # the rows and windows found, not the length of the gaps between
        # rows.
        var fixed = _fixed_width(every, dtype) and _fixed_width(period, dtype)
        var step = Int64(0)
        var span = Int64(0)
        if fixed:
            step = shift_time(origin, dtype, every, zone) - origin
            span = shift_time(origin, dtype, period, zone) - origin
            fixed = step > 0 and span > 0
        var k = 0
        var start = origin
        if fixed and start > first:
            # Back to within one step after the first row.
            k -= Int((start - first) // step)
            start = origin + Int64(k) * step
        while start > first or (start == first and not sides[0]):
            var previous = self._window_start(
                origin, k - 1, fixed, step, every, dtype, zone
            )
            if previous >= start:
                raise Error("every does not advance the time index")
            k -= 1
            start = previous
        # Large negative offsets can start before the first useful window.
        if fixed:
            k += Int((_skip_to(start, first, step, span) - start) // step)
            start = origin + Int64(k) * step
        while shift_time(start, dtype, period, zone) < first:
            k += 1
            start = self._window_start(
                origin, k, fixed, step, every, dtype, zone
            )
        while start <= last:
            if fixed:
                # The first row this window or a later one can hold.
                var next_row = search_time(times, rows, start, not sides[0])
                if next_row == len(rows):
                    break
                var target = times._get(rows[next_row])
                k += Int((_skip_to(start, target, step, span) - start) // step)
                start = origin + Int64(k) * step
            var end = shift_time(start, dtype, period, zone)
            var lo = search_time(times, rows, start, not sides[0])
            var hi = search_time(times, rows, end, sides[1])
            if lo < hi:
                labels.append(
                    start if label
                    == "left" else end if label
                    == "right" else times._get(rows[lo])
                )
                firsts.append(rows[lo])
                for r in range(lo, hi):
                    gathered.append(rows[r])
                starts.append(len(gathered))
            var next = self._window_start(
                origin, k + 1, fixed, step, every, dtype, zone
            )
            if next <= start:
                raise Error("every does not advance the time index")
            k += 1
            start = next
        self.labels[p] = labels^
        self.firsts[p] = firsts^
        self.rows[p] = gathered^
        self.starts[p] = starts^

    def _window_start(
        self,
        origin: Int64,
        k: Int,
        fixed: Bool,
        step: Int64,
        every: String,
        dtype: DataType,
        zone: TimeZone,
    ) raises -> Int64:
        """Where window `k` starts: `every` times `k` after `origin`."""
        if fixed:
            return origin + Int64(k) * step
        if k == 0:
            return origin
        var parts = parse_interval(every)
        var scale = Int64(abs(k))
        var text = String("-") if k < 0 else String("")
        if parts[0] != 0:
            text += String(parts[0] * scale) + "mo"
        if parts[1] != 0:
            text += String(parts[1] * scale) + "d"
        if parts[2] != 0:
            text += String(parts[2] * scale) + "ns"
        return shift_time(origin, dtype, text, zone)

    def _cut(mut self) raises -> Int:
        """The aggregate round: the windows cut into ranges of about the
        same number of rows, a task a range."""
        self.before = List[Int](capacity=len(self.pieces) + 1)
        var reads = List[Int](capacity=len(self.pieces) + 1)
        self.before.append(0)
        reads.append(0)
        for p in range(len(self.pieces)):
            self.before.append(self.before[p] + len(self.labels[p]))
            reads.append(reads[p] + len(self.rows[p]))
        var windows = self.before[len(self.pieces)]
        var total = reads[len(self.pieces)]
        var tasks = max(1, min(worker_count(max(total, windows)), windows))
        self.cuts = List[Int](capacity=tasks + 1)
        self.cuts.append(0)
        var p = 0
        for k in range(1, tasks):
            # The first window whose rows start at or after this task's
            # share of them.
            var target = total * k // tasks
            while p + 1 < len(self.pieces) and reads[p + 1] <= target:
                p += 1
            var at = windows
            if p < len(self.pieces):
                ref starts = self.starts[p]
                var lo = 0
                var hi = len(self.labels[p])
                while lo < hi:
                    var mid = (lo + hi) // 2
                    if reads[p] + starts[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid
                at = self.before[p] + lo
            self.cuts.append(max(at, self.cuts[k - 1]))
        self.cuts.append(windows)
        self.parts = List[List[Series]](length=tasks, fill=List[Series]())
        self.stage = _AGGREGATE
        self.width = round_width(tasks)
        return self.width

    def _aggregate(mut self, t: Int) raises:
        """Task `t`'s windows: their keys, labels and aggregations."""
        var w0 = self.cuts[t]
        var w1 = self.cuts[t + 1]
        var rows = List[Int]()
        var ids = List[Int]()
        var firsts = List[Int](capacity=w1 - w0)
        var labels = List[Int64](capacity=w1 - w0)
        # The piece holding window w0.
        var lo = 0
        var hi = len(self.pieces)
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.before[mid + 1] <= w0:
                lo = mid + 1
            else:
                hi = mid
        var p = lo
        while p < len(self.pieces) and self.before[p] < w1:
            ref starts = self.starts[p]
            ref piece_rows = self.rows[p]
            var first = max(w0 - self.before[p], 0)
            var end = min(w1 - self.before[p], len(self.labels[p]))
            for j in range(first, end):
                var id = len(firsts)
                firsts.append(self.firsts[p][j])
                labels.append(self.labels[p][j])
                for k in range(starts[j], starts[j + 1]):
                    rows.append(piece_rows[k])
                    ids.append(id)
            p += 1
        var columns = List[Series]()
        for key in self.keys:
            columns.append(key.take(firsts))
        columns.append(
            Series(self.index.name(), Column[Int64](labels^)).with_dtype(
                self.index.dtype()
            )
        )
        # The task's rows of each column, on this task's thread.
        var gathered = List[Series](capacity=self.input.width())
        for column in self.input._columns:
            gathered.append(column.take(rows))
        for expression in self.bound:
            columns.append(
                evaluate(
                    expression,
                    gathered,
                    len(rows),
                    batch_size=self.node.window_batch_size(),
                    grouped=True,
                    groups=ids,
                    group_count=len(firsts),
                    workers=1,
                )
            )
        self.parts[t] = columns^

    def _assemble(mut self) raises:
        """The output: each task's columns, one chunk each in window
        order."""
        var columns = List[Series]()
        if len(self.parts) == 1:
            swap(columns, self.parts[0])
        else:
            for c in range(len(self.parts[0])):
                var chunks = List[Series](capacity=len(self.parts))
                for t in range(len(self.parts)):
                    if len(self.parts[t][c]) > 0:
                        chunks.append(self.parts[t][c].copy())
                if len(chunks) == 0:
                    columns.append(self.parts[0][c].copy())
                elif len(chunks) == 1:
                    columns.append(chunks[0].copy())
                else:
                    columns.append(Series._from_chunks(chunks))
        self.parts = List[List[Series]]()
        self.rows = List[List[Int]]()
        self.starts = List[List[Int]]()
        self.result = DataFrame(columns^, height=self.cuts[len(self.cuts) - 1])
        self.stage = _DONE

    def take_result(mut self) raises -> DataFrame:
        """The output, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^


def time_window(var input: DataFrame, node: PlanNode) raises -> DataFrame:
    """Time window node `node` over `input`, its rounds run here: for a
    node the plan's executor runs over its executed input, which may be
    inside a pipeline's task, where no schedule may start."""
    var sink = TimeWindowSink(input^, node)
    run_rounds(sink)
    return sink.take_result()
