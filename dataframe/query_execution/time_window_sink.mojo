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
  more rows in all than the input has, a count of values or nulls, and a
  sum, mean, minimum, maximum, standard deviation or variance of a
  numeric column, are read from the bounds too: from running counts and
  integer sums in time order (one step) or a segment tree of float sums,
  extremes or running moments (O(log n) steps),
  each built in one pass over the input. Other aggregations, and all of
  these when windows overlap little, gather the range's rows and reduce
  them with the grouped reducers, a window a group. A task's output
  columns are one chunk of the result's, in window order.
"""
from std.collections import Dict, Optional
from std.math import isnan, sqrt

from dataframe.aggregation.expressions import _bind_all
from dataframe.aggregation.keys import _expand_struct_keys
from dataframe.binding import AGGREGATE, BoundExpr
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.expr import (
    COL,
    COUNT,
    LEN,
    MAX,
    MEAN,
    MIN,
    NULL_COUNT,
    ROW_LEN,
    STD,
    SUM,
    VAR,
)
from dataframe.aggregation.scalar_states import VarState
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

# How an aggregation is computed: from each window's rows gathered and
# reduced, or from the window's bounds (a count of rows, values or nulls,
# or a sum, mean, minimum, maximum, standard deviation or variance of a
# numeric column).
comptime _GATHER = 0
comptime _ROWS = 1
comptime _VALUES = 2
comptime _NULLS = 3
comptime _SUM = 4
comptime _MEAN = 5
comptime _MIN = 6
comptime _MAX = 7
comptime _STD = 8
comptime _VAR = 9


def _numeric_column(column: Series) -> Bool:
    """Whether `column` holds plain numbers (no decimal, temporal or
    Boolean meaning on top of its storage)."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if column.dtype() == DataType.of(D):
            return True
    return False


def _window_reduction(
    expression: BoundExpr, columns: List[Series]
) -> Tuple[Int, Int]:
    """How `expression` is computed over windows: the kind (`_GATHER`, or
    a reduction a window's bounds answer) and the input column it reads
    (-1 for none)."""
    ref nodes = expression.expr._nodes
    var dtype = expression.dtypes[len(nodes) - 1]
    if len(nodes) == 1 and nodes[0].op == ROW_LEN:
        return (_ROWS, -1) if dtype == DataType.INT64 else (_GATHER, -1)
    if len(nodes) != 2 or nodes[0].op != COL or nodes[1].left != 0:
        return (_GATHER, -1)
    var source = expression.sources[0]
    if source < 0:
        return (_GATHER, -1)
    var op = nodes[1].op
    if op == LEN or op == COUNT or op == NULL_COUNT:
        if dtype != DataType.INT64:
            return (_GATHER, -1)
        if op == LEN:
            return (_ROWS, -1)
        return (_VALUES if op == COUNT else _NULLS, source)
    if not _numeric_column(columns[source]):
        return (_GATHER, -1)
    if op == SUM:
        return (_SUM, source)
    if op == MEAN and dtype == DataType.FLOAT64:
        return (_MEAN, source)
    if op == MIN:
        return (_MIN, source)
    if op == MAX:
        return (_MAX, source)
    if (op == STD or op == VAR) and dtype == DataType.FLOAT64:
        return (_STD if op == STD else _VAR, source)
    return (_GATHER, -1)


def _moment_tree(var leaves: List[VarState]) -> List[VarState]:
    """A segment tree of running moments (as `_tree`): node i merges
    nodes 2i and 2i + 1."""
    var size = 1
    while size < len(leaves):
        size *= 2
    var tree = List[VarState](length=2 * size, fill=VarState())
    for i in range(len(leaves)):
        tree[size + i] = leaves[i].copy()
    for i in range(size - 1, 0, -1):
        var node = tree[2 * i].copy()
        node.merge(tree[2 * i + 1])
        tree[i] = node^
    return tree^


def _moment_range(tree: List[VarState], first: Int, end: Int) -> VarState:
    """The moments of leaves [first, end) of `tree`."""
    var size = len(tree) // 2
    var lo = first + size
    var hi = end + size
    var left = VarState()
    var right = VarState()
    while lo < hi:
        if lo & 1:
            left.merge(tree[lo])
            lo += 1
        if hi & 1:
            hi -= 1
            var node = tree[hi].copy()
            node.merge(right)
            right = node^
        lo >>= 1
        hi >>= 1
    left.merge(right)
    return left^


@always_inline
def _combine[op: Int, T: DType](a: Scalar[T], b: Scalar[T]) -> Scalar[T]:
    comptime if op == _SUM:
        return a + b
    elif op == _MIN:
        return min(a, b)
    else:
        return max(a, b)


def _tree[
    op: Int, T: DType
](var leaves: List[Scalar[T]], identity: Scalar[T]) -> List[Scalar[T]]:
    """A segment tree over `leaves`: node i combines nodes 2i and 2i + 1,
    and the leaves start at the first power of two not below their
    count, so any range combines O(log n) nodes."""
    var size = 1
    while size < len(leaves):
        size *= 2
    var tree = List[Scalar[T]](length=2 * size, fill=identity)
    for i in range(len(leaves)):
        tree[size + i] = leaves[i]
    for i in range(size - 1, 0, -1):
        tree[i] = _combine[op, T](tree[2 * i], tree[2 * i + 1])
    return tree^


@always_inline
def _tree_range[
    op: Int, T: DType
](tree: List[Scalar[T]], first: Int, end: Int, identity: Scalar[T]) -> Scalar[
    T
]:
    """Leaves [first, end) of `tree` combined, left to right."""
    var size = len(tree) // 2
    var lo = first + size
    var hi = end + size
    var left = identity
    var right = identity
    while lo < hi:
        if lo & 1:
            left = _combine[op, T](left, tree[lo])
            lo += 1
        if hi & 1:
            hi -= 1
            right = _combine[op, T](tree[hi], right)
        lo >>= 1
        hi >>= 1
    return _combine[op, T](left, right)


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
    # By expression how it is computed and the column it reads
    # (`_window_reduction`), and whether any expression gathers rows.
    # Reductions read from window bounds use the groups' time-ordered
    # rows laid end to end (`offsets` is where each group starts in that
    # order): by input column (empty when it has no nulls) the nulls
    # before each position, and by expression the running Int128 sums
    # of an integer column, a segment tree of a float column's sums or of
    # a column's extremes, and the NaNs before each position of a float
    # column whose extremes are taken.
    var counts: List[Tuple[Int, Int]]
    var gathers: Bool
    var offsets: List[Int]
    var nulls: List[List[Int]]
    var int_sums: List[List[Int128]]
    var float_trees: List[List[Float64]]
    var int_trees: List[List[Int128]]
    var moment_trees: List[List[VarState]]
    var nans: List[List[Int]]
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
        self.int_sums = List[List[Int128]]()
        self.float_trees = List[List[Float64]]()
        self.int_trees = List[List[Int128]]()
        self.moment_trees = List[List[VarState]]()
        self.nans = List[List[Int]]()
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
        """How each expression is computed."""
        for expression in self.bound:
            var count = _window_reduction(expression, self.input._columns)
            self.counts.append(count)
            if count[0] == _GATHER:
                self.gathers = True

    def _prepare_bounds(mut self) raises:
        """What the reductions read from window bounds use (see the
        fields): one pass over the input for each."""
        self.offsets = List[Int](capacity=len(self.groups) + 1)
        self.offsets.append(0)
        for g in range(len(self.groups)):
            self.offsets.append(self.offsets[g] + len(self.groups[g]))
        var columns = len(self.input._columns)
        var expressions = len(self.counts)
        self.nulls = List[List[Int]](length=columns, fill=List[Int]())
        self.int_sums = List[List[Int128]](
            length=expressions, fill=List[Int128]()
        )
        self.float_trees = List[List[Float64]](
            length=expressions, fill=List[Float64]()
        )
        self.int_trees = List[List[Int128]](
            length=expressions, fill=List[Int128]()
        )
        self.moment_trees = List[List[VarState]](
            length=expressions, fill=List[VarState]()
        )
        self.nans = List[List[Int]](length=expressions, fill=List[Int]())
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
        for e in range(expressions):
            var kind = self.counts[e][0]
            if kind >= _SUM:
                self._prepare_reduction(e)

    def _prepare_reduction(mut self, e: Int) raises:
        """Expression `e`'s sums or extremes over its column, in the
        groups' time order. Null values add nothing and are never an
        extreme; NaN is never an extreme either (a window's extreme is NaN
        only when all its values are NaN, from the NaN counts)."""
        var kind = self.counts[e][0]
        ref column = self.input._columns[self.counts[e][1]]
        var valid = validity(column)
        if kind == _STD or kind == _VAR:
            var floats = column.cast(DataType.FLOAT64)
            ref values = floats._data[Column[Float64]]
            var leaves = List[VarState](capacity=self.height)
            for rows in self.groups:
                for row in rows:
                    var state = VarState()
                    if valid[row]:
                        state.add(values._get(row))
                    leaves.append(state^)
            self.moment_trees[e] = _moment_tree(leaves^)
            return
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if column._data.isa[Column[Scalar[D]]]():
                ref values = column._data[Column[Scalar[D]]]
                comptime if D.is_integral():
                    var leaves = List[Int128](capacity=self.height)
                    for rows in self.groups:
                        for row in rows:
                            leaves.append(
                                values._get(row).cast[DType.int128]() if valid[
                                    row
                                ] else 0
                            )
                    if kind == _SUM or kind == _MEAN:
                        var running = List[Int128](capacity=self.height + 1)
                        running.append(0)
                        for i in range(len(leaves)):
                            running.append(running[i] + leaves[i])
                        self.int_sums[e] = running^
                    else:
                        var low = kind == _MIN
                        var identity = Int128.MAX if low else Int128.MIN
                        var at = 0
                        for rows in self.groups:
                            for row in rows:
                                if not valid[row]:
                                    leaves[at] = identity
                                at += 1
                        if low:
                            self.int_trees[e] = _tree[_MIN, DType.int128](
                                leaves^, identity
                            )
                        else:
                            self.int_trees[e] = _tree[_MAX, DType.int128](
                                leaves^, identity
                            )
                else:
                    var extreme = kind == _MIN or kind == _MAX
                    var inf = Float64(1) / Float64(0)
                    var identity = Float64(0)
                    if kind == _MIN:
                        identity = inf
                    elif kind == _MAX:
                        identity = -inf
                    var leaves = List[Float64](capacity=self.height)
                    var nans = List[Int](capacity=self.height + 1)
                    nans.append(0)
                    for rows in self.groups:
                        for row in rows:
                            var value = values._get(row).cast[DType.float64]()
                            var is_nan = valid[row] and isnan(value)
                            if extreme:
                                nans.append(
                                    nans[len(nans) - 1] + (1 if is_nan else 0)
                                )
                            if not valid[row] or (extreme and is_nan):
                                value = identity
                            leaves.append(value)
                    if kind == _MIN:
                        self.float_trees[e] = _tree[_MIN, DType.float64](
                            leaves^, identity
                        )
                    elif kind == _MAX:
                        self.float_trees[e] = _tree[_MAX, DType.float64](
                            leaves^, identity
                        )
                    else:
                        self.float_trees[e] = _tree[_SUM, DType.float64](
                            leaves^, identity
                        )
                    if extreme:
                        self.nans[e] = nans^
                return

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
        # A reduction costs the window's rows when they are gathered and
        # reduced. Read from the window's bounds it costs one step (a
        # count) or O(log n) (an extreme or a float sum), after one pass
        # over the input to prepare. Bounds are used when windows hold
        # more rows in all than the input has; otherwise every reduction
        # but a count of rows is gathered.
        var members = 0
        for p in range(len(self.pieces)):
            for j in range(len(self.labels[p])):
                members += self.highs[p][j] - self.lows[p][j]
        if members > self.height:
            self._prepare_bounds()
        else:
            for e in range(len(self.counts)):
                if self.counts[e][0] != _ROWS:
                    self.counts[e] = (_GATHER, -1)
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
            if count[0] == _GATHER:
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
            var flat_lows = List[Int](capacity=len(lows))
            var flat_highs = List[Int](capacity=len(lows))
            for w in range(len(lows)):
                var offset = (
                    self.offsets[groups[w]] if len(self.offsets) > 0 else 0
                )
                flat_lows.append(offset + lows[w])
                flat_highs.append(offset + highs[w])
            columns.append(self._from_bounds(e, flat_lows, flat_highs))
        self.parts[t] = columns^

    def _from_bounds(
        self, e: Int, lows: List[Int], highs: List[Int]
    ) raises -> Series:
        """Expression `e` over the windows whose rows are positions
        [lows[w], highs[w]) of the groups' time-ordered rows laid end to
        end, read from what `_prepare_bounds` made."""
        var kind = self.counts[e][0]
        var c = self.counts[e][1]
        ref expression = self.bound[e]
        var name = expression.expr._name
        var dtype = expression.dtypes[len(expression.dtypes) - 1]
        var n = len(lows)
        # Each window's count of values.
        var counts = List[Int](capacity=n)
        for w in range(n):
            var total = highs[w] - lows[w]
            if c >= 0 and len(self.nulls[c]) > 0:
                total -= self.nulls[c][highs[w]] - self.nulls[c][lows[w]]
            counts.append(total)
        if kind == _ROWS or kind == _VALUES or kind == _NULLS:
            var values = List[Int64](capacity=n)
            for w in range(n):
                var rows = highs[w] - lows[w]
                if kind == _ROWS:
                    values.append(Int64(rows))
                elif kind == _VALUES:
                    values.append(Int64(counts[w]))
                else:
                    values.append(Int64(rows - counts[w]))
            return Series(name, Column[Int64](values^))
        if kind == _STD or kind == _VAR:
            var ddof = Int(expression.expr._nodes[1].integer)
            var out = List[Float64](length=n, fill=0)
            var present = List[Bool](length=n, fill=False)
            for w in range(n):
                var variance = _moment_range(
                    self.moment_trees[e], lows[w], highs[w]
                ).variance(ddof)
                if variance:
                    var value = variance.value()
                    out[w] = sqrt(value) if kind == _STD else value
                    present[w] = True
            return Series(name, Column[Float64](out^, present))
        var valid = List[Bool](capacity=n)
        var min_count = expression.expr._nodes[1].min_count
        for w in range(n):
            if kind == _SUM:
                valid.append(counts[w] >= min_count)
            else:
                valid.append(counts[w] > 0)
        if len(self.int_sums[e]) > 0:
            ref sums = self.int_sums[e]
            if kind == _MEAN:
                var means = List[Float64](length=n, fill=0)
                for w in range(n):
                    if valid[w]:
                        var total = sums[highs[w]] - sums[lows[w]]
                        means[w] = total.cast[DType.float64]() / Float64(
                            counts[w]
                        )
                return Series(name, Column[Float64](means^, valid))
            comptime for k in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[k]
                comptime if D.is_integral():
                    if dtype == DataType.of(D):
                        var out = List[Scalar[D]](length=n, fill=0)
                        for w in range(n):
                            if not valid[w]:
                                continue
                            var total = sums[highs[w]] - sums[lows[w]]
                            if (
                                total > Scalar[D].MAX.cast[DType.int128]()
                                or total < Scalar[D].MIN.cast[DType.int128]()
                            ):
                                raise Error(
                                    dtype.name() + " expression sum overflow"
                                )
                            out[w] = total.cast[D]()
                        return Series(name, Column[Scalar[D]](out^, valid))
            raise Error("Unsupported sum type " + dtype.name())
        if len(self.int_trees[e]) > 0:
            ref tree = self.int_trees[e]
            var low = kind == _MIN
            var identity = Int128.MAX if low else Int128.MIN
            comptime for k in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[k]
                comptime if D.is_integral():
                    if dtype == DataType.of(D):
                        var out = List[Scalar[D]](length=n, fill=0)
                        for w in range(n):
                            if not valid[w]:
                                continue
                            var value: Int128
                            if low:
                                value = _tree_range[_MIN, DType.int128](
                                    tree, lows[w], highs[w], identity
                                )
                            else:
                                value = _tree_range[_MAX, DType.int128](
                                    tree, lows[w], highs[w], identity
                                )
                            out[w] = value.cast[D]()
                        return Series(name, Column[Scalar[D]](out^, valid))
            raise Error("Unsupported extreme type " + dtype.name())
        # A float column: its sums or extremes from the tree.
        ref tree = self.float_trees[e]
        var inf = Float64(1) / Float64(0)
        var nan = Float64(0) / Float64(0)
        var out = List[Float64](length=n, fill=0)
        for w in range(n):
            if not valid[w]:
                continue
            if kind == _SUM or kind == _MEAN:
                var total = _tree_range[_SUM, DType.float64](
                    tree, lows[w], highs[w], 0
                )
                out[w] = total / Float64(counts[w]) if kind == _MEAN else total
            elif self.nans[e][highs[w]] - self.nans[e][lows[w]] == counts[w]:
                out[w] = nan
            elif kind == _MIN:
                out[w] = _tree_range[_MIN, DType.float64](
                    tree, lows[w], highs[w], inf
                )
            else:
                out[w] = _tree_range[_MAX, DType.float64](
                    tree, lows[w], highs[w], -inf
                )
        if dtype == DataType.FLOAT32:
            var narrow = List[Float32](capacity=n)
            for value in out:
                narrow.append(value.cast[DType.float32]())
            return Series(name, Column[Float32](narrow^, valid))
        return Series(name, Column[Float64](out^, valid))

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
        self.int_sums = List[List[Int128]]()
        self.float_trees = List[List[Float64]]()
        self.int_trees = List[List[Int128]]()
        self.moment_trees = List[List[VarState]]()
        self.nans = List[List[Int]]()
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
