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
  lists each window's label, first row and bounds: a window is a range of
  positions in its group's time-ordered rows, so listing it costs the same
  however many rows it holds.
- a round aggregates them: the windows are cut into contiguous ranges of
  about the same number of rows, and each task reduces its range's
  windows. A count of rows is read from the bounds. When windows hold
  more rows in all than the input has, a count of values or nulls is the
  difference of a running count of nulls in time order at the bounds, so
  it costs one pass over the input plus a step per window. Other
  aggregations (and those counts, when windows overlap little) gather the
  range's rows and reduce them with the grouped reducers, a window a
  group. A task's output columns are one chunk of the result's, in window
  order.
"""
from std.collections import Dict, Optional

from dataframe.aggregation.expressions import _bind_all
from dataframe.aggregation.keys import _expand_struct_keys
from dataframe.binding import AGGREGATE, BoundExpr
from dataframe.dtype import DataType
from dataframe.expr import COL, COUNT, LEN, NULL_COUNT, ROW_LEN
from dataframe.kernels.binary import validity
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
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.time.temporal import parse_interval, zone_of
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

# What a count read from a window's bounds counts.
comptime _NOT_A_COUNT = 0
comptime _ROWS = 1
comptime _VALUES = 2
comptime _NULLS = 3


def _window_count(expression: BoundExpr) -> Tuple[Int, Int]:
    """Whether `expression` is a count that a window's bounds answer: the
    kind (`_ROWS`, `_VALUES`, `_NULLS`, or `_NOT_A_COUNT`) and the input
    column it counts (-1 for rows)."""
    ref nodes = expression.expr._nodes
    if expression.dtypes[len(nodes) - 1] != DataType.INT64:
        return (_NOT_A_COUNT, -1)
    if len(nodes) == 1 and nodes[0].op == ROW_LEN:
        return (_ROWS, -1)
    if len(nodes) != 2 or nodes[0].op != COL or nodes[1].left != 0:
        return (_NOT_A_COUNT, -1)
    var source = expression.sources[0]
    if source < 0:
        return (_NOT_A_COUNT, -1)
    if nodes[1].op == LEN:
        return (_ROWS, -1)
    if nodes[1].op == COUNT:
        return (_VALUES, source)
    if nodes[1].op == NULL_COUNT:
        return (_NULLS, source)
    return (_NOT_A_COUNT, -1)


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
    # rows, and first and end positions in their group's time-ordered rows.
    var pieces: List[Tuple[Int, Int, Int]]
    var labels: List[List[Int64]]
    var firsts: List[List[Int]]
    var lows: List[List[Int]]
    var highs: List[List[Int]]
    # By expression its count kind and column (`_window_count`), and
    # whether any expression gathers rows. By input column (empty when
    # none is counted, it has no nulls, or windows hold too few rows for
    # a running count to pay) the nulls before each position of the
    # groups' time-ordered rows laid end to end (`offsets` is where each
    # group starts in that order).
    var counts: List[Tuple[Int, Int]]
    var gathers: Bool
    var offsets: List[Int]
    var nulls: List[List[Int]]
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
        self.lows = List[List[Int]]()
        self.highs = List[List[Int]]()
        self.counts = List[Tuple[Int, Int]]()
        self.gathers = False
        self.offsets = List[Int]()
        self.nulls = List[List[Int]]()
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
        self._classify()
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
        self.lows = List[List[Int]](length=n, fill=List[Int]())
        self.highs = List[List[Int]](length=n, fill=List[Int]())
        self.stage = _WINDOWS
        self.width = round_width(n)
        if self.width == 0:
            return self._cut()
        return self.width

    def _classify(mut self):
        """Each expression's count kind."""
        for expression in self.bound:
            var count = _window_count(expression)
            self.counts.append(count)
            if count[0] == _NOT_A_COUNT:
                self.gathers = True

    def _count_nulls(mut self):
        """For every counted column with nulls, the running count of its
        nulls in the groups' time order (a window's nulls are the
        difference of the counts at its bounds)."""
        self.offsets = List[Int](capacity=len(self.groups) + 1)
        self.offsets.append(0)
        for g in range(len(self.groups)):
            self.offsets.append(self.offsets[g] + len(self.groups[g]))
        self.nulls = List[List[Int]](
            length=len(self.input._columns), fill=List[Int]()
        )
        for count in self.counts:
            var c = count[1]
            if c < 0 or len(self.nulls[c]) > 0:
                continue
            if self.input._columns[c].null_count() == 0:
                continue
            var valid = validity(self.input._columns[c])
            var running = List[Int](capacity=self.height + 1)
            running.append(0)
            for rows in self.groups:
                for row in rows:
                    running.append(
                        running[len(running) - 1] + (0 if valid[row] else 1)
                    )
            self.nulls[c] = running^

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
        var lows = List[Int](capacity=piece[2] - piece[1])
        var highs = List[Int](capacity=piece[2] - piece[1])
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
            lows.append(lo)
            highs.append(hi)
        self.labels[p] = labels^
        self.firsts[p] = firsts^
        self.lows[p] = lows^
        self.highs[p] = highs^

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
        var lows = List[Int]()
        var highs = List[Int]()
        var first = times._get(rows[0])
        var last = times._get(rows[len(rows) - 1])
        var start = shift_time(
            truncate_time(first, dtype, every, zone),
            dtype,
            self.node.window_offset(),
            zone,
        )
        while start > first or (start == first and not sides[0]):
            var previous = shift_time(start, dtype, negate_period(every), zone)
            if previous >= start:
                raise Error("every does not advance the time index")
            start = previous
        # Large negative offsets can start before the first useful window.
        while shift_time(start, dtype, period, zone) < first:
            start = shift_time(start, dtype, every, zone)
        while start <= last:
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
                lows.append(lo)
                highs.append(hi)
            var next = shift_time(start, dtype, every, zone)
            if next <= start:
                raise Error("every does not advance the time index")
            start = next
        self.labels[p] = labels^
        self.firsts[p] = firsts^
        self.lows[p] = lows^
        self.highs[p] = highs^

    def _cut(mut self) raises -> Int:
        """The aggregate round: the windows cut into ranges of about the
        same work, a task a range. Gathering or reading a window's rows
        costs its row count; reading a count from its bounds costs one."""
        # A count of values or nulls costs the window's rows when its
        # rows are gathered and reduced, or one step when read from a
        # running count of nulls, which costs one pass over the input to
        # build. The running count is built when windows hold more rows
        # in all than the input has; otherwise such counts are gathered.
        var members = 0
        var counts_nulls = False
        for p in range(len(self.pieces)):
            for j in range(len(self.labels[p])):
                members += self.highs[p][j] - self.lows[p][j]
        for count in self.counts:
            if count[0] == _VALUES or count[0] == _NULLS:
                counts_nulls = counts_nulls or (
                    self.input._columns[count[1]].null_count() > 0
                )
        if counts_nulls and members > self.height:
            self._count_nulls()
        elif counts_nulls:
            for e in range(len(self.counts)):
                var kind = self.counts[e][0]
                if kind == _VALUES or kind == _NULLS:
                    self.counts[e] = (_NOT_A_COUNT, -1)
                    self.gathers = True
        var reads_rows = self.gathers
        self.before = List[Int](capacity=len(self.pieces) + 1)
        var reads = List[Int](capacity=len(self.pieces) + 1)
        # By piece, where each window's work starts in the piece's.
        var starts = List[List[Int]](capacity=len(self.pieces))
        self.before.append(0)
        reads.append(0)
        for p in range(len(self.pieces)):
            var windows = len(self.labels[p])
            self.before.append(self.before[p] + windows)
            var piece_starts = List[Int](capacity=windows + 1)
            piece_starts.append(0)
            for j in range(windows):
                var work = (
                    self.highs[p][j] - self.lows[p][j] if reads_rows else 1
                )
                piece_starts.append(piece_starts[j] + work)
            reads.append(reads[p] + piece_starts[windows])
            starts.append(piece_starts^)
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
                ref piece_starts = starts[p]
                var lo = 0
                var hi = len(self.labels[p])
                while lo < hi:
                    var mid = (lo + hi) // 2
                    if reads[p] + piece_starts[mid] < target:
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
        var firsts = List[Int](capacity=w1 - w0)
        var labels = List[Int64](capacity=w1 - w0)
        # By window its group and first and end positions in the group's
        # time-ordered rows.
        var groups = List[Int](capacity=w1 - w0)
        var lows = List[Int](capacity=w1 - w0)
        var highs = List[Int](capacity=w1 - w0)
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
            var group = self.pieces[p][0]
            var first = max(w0 - self.before[p], 0)
            var end = min(w1 - self.before[p], len(self.labels[p]))
            for j in range(first, end):
                firsts.append(self.firsts[p][j])
                labels.append(self.labels[p][j])
                groups.append(group)
                lows.append(self.lows[p][j])
                highs.append(self.highs[p][j])
            p += 1
        var columns = List[Series]()
        for key in self.keys:
            columns.append(key.take(firsts))
        columns.append(
            Series(self.index.name(), Column[Int64](labels^)).with_dtype(
                self.index.dtype()
            )
        )
        var rows = List[Int]()
        var ids = List[Int]()
        var gathered = List[Series]()
        if self.gathers:
            var members = 0
            for w in range(len(lows)):
                members += highs[w] - lows[w]
            rows = List[Int](capacity=members)
            ids = List[Int](capacity=members)
            for w in range(len(lows)):
                ref group = self.groups[groups[w]]
                for k in range(lows[w], highs[w]):
                    rows.append(group[k])
                    ids.append(w)
            # The task's rows of each column, on this task's thread.
            gathered = List[Series](capacity=self.input.width())
            for column in self.input._columns:
                gathered.append(column.take(rows))
        for e in range(len(self.bound)):
            var count = self.counts[e]
            if count[0] == _NOT_A_COUNT:
                columns.append(
                    evaluate(
                        self.bound[e],
                        gathered,
                        len(rows),
                        batch_size=self.node.window_batch_size(),
                        grouped=True,
                        groups=ids,
                        group_count=len(firsts),
                        workers=1,
                    )
                )
                continue
            var values = List[Int64](capacity=len(lows))
            var c = count[1]
            var has_nulls = (
                c >= 0 and len(self.nulls) > 0 and len(self.nulls[c]) > 0
            )
            for w in range(len(lows)):
                var total = highs[w] - lows[w]
                var missing = 0
                if has_nulls:
                    var offset = self.offsets[groups[w]]
                    missing = (
                        self.nulls[c][offset + highs[w]]
                        - self.nulls[c][offset + lows[w]]
                    )
                if count[0] == _VALUES:
                    total -= missing
                elif count[0] == _NULLS:
                    total = missing
                values.append(Int64(total))
            columns.append(
                Series(self.bound[e].expr._name, Column[Int64](values^))
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
        self.lows = List[List[Int]]()
        self.highs = List[List[Int]]()
        self.nulls = List[List[Int]]()
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
