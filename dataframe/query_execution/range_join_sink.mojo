"""Inequality joins run as rounds of tasks of the query's schedule.

A RANGE_JOIN node joins rows whose keys satisfy one or two comparisons of a
left key with a right key (see dataframe/joins/range_join.mojo for the two
algorithms). Both inputs are complete when its rounds start:

- A sort, as the rounds of a parallel sort: the point side's valid keys for
  a range join; for a two-key join each side's valid rows by first key.
- Matching: a range join counts each interval row's run of point rows (a
  task a range of interval rows), then writes the pairs at their offsets; a
  two-key join runs one task per pair of blocks whose first keys can meet.
- Gathering the left columns at the matched left rows, then the right
  columns at the matched right rows, as rounds of row ranges.

The output is every left column, then every right column, a right name
that a left column has taking the node's suffix. Its row order is
unspecified.
"""
from std.collections import Dict, Optional
from std.math import sqrt

from dataframe.dtype import DataType
from dataframe.frame import DataFrame
from dataframe.joins.range_join import (
    IE_GE,
    IE_GT,
    IE_LE,
    IE_LT,
    ie_blocks_overlap,
    ie_join_block,
    ie_less,
    ie_swapped,
    matchable_rows,
    nested_pairs,
    pairing_is_cheaper,
    range_fill,
    range_key_type,
    range_spans,
)
from dataframe.kernels.gather import GatherRounds
from dataframe.parallel import (
    Rounds,
    configured_workers,
    partitions,
    round_width,
    run_rounds,
    worker_count,
)
from dataframe.planning.model import PlanNode
from dataframe.series import Series
from dataframe.sorting.sort import SortRounds
from dataframe.storage.column import Column

# The stage whose rounds run next.
comptime _START = 0
comptime _SORT = 1
comptime _COUNT = 2
comptime _FILL = 3
comptime _PAIRS = 4
comptime _GATHER_LEFT = 5
comptime _GATHER_RIGHT = 6
comptime _DONE = 7
comptime _POINTS = 8


def range_join_ops(node: PlanNode) raises -> List[Int]:
    """The node's comparisons, `left_key[k] OP right_key[k]`."""
    var ops = List[Int]()
    for part in node.text.split(","):
        var op = String(part)
        if op == "<":
            ops.append(IE_LT)
        elif op == "<=":
            ops.append(IE_LE)
        elif op == ">":
            ops.append(IE_GT)
        elif op == ">=":
            ops.append(IE_GE)
        else:
            raise Error("Unknown inequality join comparison: " + op)
    return ops^


def _key_series(column: Series, target: DataType) raises -> Series:
    """`column` as the comparison type, one chunk."""
    var key = column.rechunk() if column.is_chunked() else column.copy()
    if key.dtype().is_temporal():
        return key.with_dtype(DataType.INT64)
    if key.dtype() != target:
        return key.cast(target)
    return key^


def range_join_names(
    left: DataFrame, right: DataFrame, suffix: String
) raises -> List[String]:
    """The right columns' output names: suffixed where a left column has
    the name."""
    var names = Dict[String, Bool]()
    for column in left._columns:
        names[column.name()] = True
    var left_names = names.copy()
    var out = List[String]()
    for column in right._columns:
        var name = column.name()
        if name in left_names:
            name += suffix
        if name in names:
            raise Error("Join output name collision: " + name)
        names[name] = True
        out.append(name)
    return out^


struct RangeJoinSink(Movable, Rounds):
    """RANGE_JOIN node `node` over all rows of `left` and `right`, as rounds
    (see the module docstring); `take_result` once no round is left."""

    var left: DataFrame
    var right: DataFrame
    var node: PlanNode
    var stage: Int
    var ops: List[Int]
    var floats: Bool
    var right_names: List[String]
    # Each comparison's keys, in the comparison type.
    var left_keys: List[Series]
    var right_keys: List[Series]
    # A range join: whether the left side is the point side, and the
    # comparisons as `point OP bound`.
    var range: Bool
    var point_left: Bool
    var point_ops: List[Int]
    # The parallel sort running, and the sorts still to run (by side: 0
    # left, 1 right), with the rows each sorted key comes from.
    var sorting: List[SortRounds]
    var sort_sides: List[Int]
    var sort_rows_of: List[List[Int]]
    # Per side, its valid rows in sorted order.
    var sorted_rows: List[List[Int]]
    # Range join: the point side's sorted keys, each interval row's run of
    # them, and the interval row ranges of the count and fill rounds with
    # their pairs and first output position.
    var points_int: List[Int64]
    var points_float: List[Float64]
    var starts: List[Int]
    var stops: List[Int]
    var ranges: List[Int]
    var range_pairs: List[Int]
    # Two-key join: block bounds per side (positions in sorted order), the
    # block pairs to scan and their pairs.
    var left_blocks: List[Int]
    var right_blocks: List[Int]
    var block_pairs: List[Tuple[Int, Int]]
    var pair_left: List[List[Int]]
    var pair_right: List[List[Int]]
    var scanned: Bool
    # Whether the pairs round compares every pair of a range of left rows
    # with every right row instead (one input has a handful of rows).
    var nested: Bool
    var width: Int
    # The matched rows, the gather running, and the gathered left columns.
    var left_rows: List[Int]
    var right_rows: List[Int]
    var gathering: List[GatherRounds]
    var gathered_left: List[Series]
    var result: DataFrame

    def __init__(
        out self, var left: DataFrame, var right: DataFrame, node: PlanNode
    ) raises:
        self.left = left^
        self.right = right^
        self.node = node.copy()
        self.stage = _START
        self.ops = List[Int]()
        self.floats = False
        self.right_names = List[String]()
        self.left_keys = List[Series]()
        self.right_keys = List[Series]()
        self.range = False
        self.point_left = False
        self.point_ops = List[Int]()
        self.sorting = List[SortRounds]()
        self.sort_sides = List[Int]()
        self.sort_rows_of = List[List[Int]]()
        self.sorted_rows = List[List[Int]](length=2, fill=List[Int]())
        self.points_int = List[Int64]()
        self.points_float = List[Float64]()
        self.starts = List[Int]()
        self.stops = List[Int]()
        self.ranges = List[Int]()
        self.range_pairs = List[Int]()
        self.left_blocks = List[Int]()
        self.right_blocks = List[Int]()
        self.block_pairs = List[Tuple[Int, Int]]()
        self.pair_left = List[List[Int]]()
        self.pair_right = List[List[Int]]()
        self.scanned = False
        self.nested = False
        self.width = 0
        self.left_rows = List[Int]()
        self.right_rows = List[Int]()
        self.gathering = List[GatherRounds]()
        self.gathered_left = List[Series]()
        self.result = DataFrame(List[Series]())

    def next_round(mut self) raises -> Int:
        while True:
            if self.stage == _START:
                self._start()
                continue
            if self.stage == _SORT:
                var tasks = self.sorting[0].next_round()
                if tasks > 0:
                    return tasks
                self._sorted()
                continue
            if self.stage == _POINTS:
                if not self.scanned:
                    self.scanned = True
                    if self.width > 0:
                        return self.width
                self.scanned = False
                self._counting()
                continue
            if self.stage == _COUNT:
                if self.width > 0:
                    # The count round ran: offsets, then the fill round.
                    var total = 0
                    for r in range(len(self.range_pairs)):
                        var pairs = self.range_pairs[r]
                        self.range_pairs[r] = total
                        total += pairs
                    var interval = List[Int](length=total, fill=0)
                    var point = List[Int](length=total, fill=0)
                    if self.point_left:
                        self.right_rows = interval^
                        self.left_rows = point^
                    else:
                        self.left_rows = interval^
                        self.right_rows = point^
                    self.stage = _FILL
                    self.width = round_width(len(self.ranges) - 1)
                    return self.width
                self.width = round_width(len(self.ranges) - 1)
                if self.width == 0:
                    self._begin_gather()
                    continue
                return self.width
            if self.stage == _FILL:
                self.starts = List[Int]()
                self.stops = List[Int]()
                self._begin_gather()
                continue
            if self.stage == _PAIRS:
                if not self.scanned:
                    # The pair round first.
                    self.scanned = True
                    if self.width > 0:
                        return self.width
                var total = 0
                for p in range(len(self.pair_left)):
                    total += len(self.pair_left[p])
                self.left_rows = List[Int](capacity=total)
                self.right_rows = List[Int](capacity=total)
                for p in range(len(self.pair_left)):
                    self.left_rows.extend(self.pair_left[p].copy())
                    self.right_rows.extend(self.pair_right[p].copy())
                self.pair_left = List[List[Int]]()
                self.pair_right = List[List[Int]]()
                self._begin_gather()
                continue
            if self.stage == _GATHER_LEFT or self.stage == _GATHER_RIGHT:
                var tasks = self.gathering[0].next_round()
                if tasks > 0:
                    return tasks
                var columns = self.gathering[0].take_result()
                _ = self.gathering.pop()
                if self.stage == _GATHER_LEFT:
                    self.gathered_left = columns^
                    self.gathering.append(
                        GatherRounds(
                            self.right._columns,
                            self.right_rows.copy(),
                            configured_workers(),
                        )
                    )
                    self.stage = _GATHER_RIGHT
                    continue
                self._assemble(columns^)
                continue
            return 0

    def round_task(mut self, task: Int) raises:
        if self.stage == _SORT:
            self.sorting[0].round_task(task)
        elif self.stage == _POINTS:
            for r in range(task, len(self.ranges) - 1, self.width):
                self._fill_points(self.ranges[r], self.ranges[r + 1])
        elif self.stage == _COUNT:
            for r in range(task, len(self.ranges) - 1, self.width):
                if self.floats:
                    self.range_pairs[r] = self._count[DType.float64](r)
                else:
                    self.range_pairs[r] = self._count[DType.int64](r)
        elif self.stage == _FILL:
            ref order = self.sorted_rows[0 if self.point_left else 1]
            for r in range(task, len(self.ranges) - 1, self.width):
                if self.point_left:
                    range_fill(
                        self.starts,
                        self.stops,
                        order,
                        self.ranges[r],
                        self.ranges[r + 1],
                        self.range_pairs[r],
                        self.right_rows,
                        self.left_rows,
                    )
                else:
                    range_fill(
                        self.starts,
                        self.stops,
                        order,
                        self.ranges[r],
                        self.ranges[r + 1],
                        self.range_pairs[r],
                        self.left_rows,
                        self.right_rows,
                    )
        elif self.stage == _PAIRS and self.nested:
            for r in range(task, len(self.ranges) - 1, self.width):
                if self.floats:
                    self._nested_range[DType.float64](r)
                else:
                    self._nested_range[DType.int64](r)
        elif self.stage == _PAIRS:
            for p in range(task, len(self.block_pairs), self.width):
                if self.floats:
                    self._scan_pair[DType.float64](p)
                else:
                    self._scan_pair[DType.int64](p)
        elif self.stage == _GATHER_LEFT or self.stage == _GATHER_RIGHT:
            self.gathering[0].round_task(task)

    def _start(mut self) raises:
        """The keys in their comparison type, the algorithm, and its sorts."""
        self.right_names = range_join_names(
            self.left, self.right, self.node.join_suffix()
        )
        self.ops = range_join_ops(self.node)
        ref left_on = self.node.join_left_keys()
        ref right_on = self.node.join_right_keys()
        if len(self.ops) != len(left_on) or len(left_on) != len(right_on):
            raise Error("An inequality join needs one comparison per key pair")
        for k in range(len(self.ops)):
            var lkey = self.left.column(left_on[k])
            var rkey = self.right.column(right_on[k])
            var target = range_key_type(lkey.dtype(), rkey.dtype())
            if not target:
                raise Error(
                    "An inequality join cannot compare "
                    + lkey.dtype().name()
                    + " with "
                    + rkey.dtype().name()
                )
            if target.value() == DataType.FLOAT64:
                self.floats = True
        # One number type serves every comparison: all integer or all float
        # (planning pairs only comparisons of one kind).
        for k in range(len(self.ops)):
            var t = range_key_type(
                self.left.column(left_on[k]).dtype(),
                self.right.column(right_on[k]).dtype(),
            )
            if (t.value() == DataType.FLOAT64) != self.floats:
                raise Error(
                    "An inequality join compares integer and float keys"
                    " together"
                )
        var target = DataType.FLOAT64 if self.floats else DataType.INT64
        for k in range(len(self.ops)):
            self.left_keys.append(
                _key_series(self.left.column(left_on[k]), target)
            )
            self.right_keys.append(
                _key_series(self.right.column(right_on[k]), target)
            )
        # With one input of a handful of rows, comparing every pair is less
        # work than any sort: tasks take ranges of left rows.
        if pairing_is_cheaper(self.left.height(), self.right.height()):
            self.nested = True
            var n = self.left.height()
            self.ranges = partitions(n, worker_count(n), 64)
            self.pair_left = List[List[Int]](
                length=len(self.ranges) - 1, fill=List[Int]()
            )
            self.pair_right = List[List[Int]](
                length=len(self.ranges) - 1, fill=List[Int]()
            )
            self.stage = _PAIRS
            self.width = round_width(len(self.ranges) - 1)
            return
        # A range join when every comparison reads one key of a side; the
        # smaller side is sorted when either could be.
        var left_point = len(self.ops) == 1 or left_on[0] == left_on[1]
        var right_point = len(self.ops) == 1 or right_on[0] == right_on[1]
        self.range = left_point or right_point
        if self.range:
            self.point_left = left_point and (
                not right_point or self.left.height() < self.right.height()
            )
            for op in self.ops:
                self.point_ops.append(op if self.point_left else ie_swapped(op))
            self._queue_sort(0 if self.point_left else 1, 1, False)
        else:
            var descending = not ie_less(self.ops[0])
            self._queue_sort(0, 2, descending)
            self._queue_sort(1, 2, descending)
        self.stage = _SORT

    def _queue_sort(mut self, side: Int, keys: Int, descending: Bool) raises:
        """Queue the sort of `side`'s rows whose first `keys` keys are all
        valid, by its first key."""
        ref columns = self.left_keys if side == 0 else self.right_keys
        var count = self.left.height() if side == 0 else self.right.height()
        # Every row takes part unless a key has a null (or a float key a
        # NaN): then the sort reads only the rows that can match.
        var all_rows = not self.floats
        for k in range(keys):
            all_rows = all_rows and columns[k].null_count() == 0
        var rows = List[Int]()
        if all_rows:
            self.sort_sides.append(side)
            self.sorting.append(
                SortRounds([columns[0].copy()], [descending], [True], True)
            )
            self.sort_rows_of.append(rows^)
            return
        var valid: List[Bool]
        if self.floats:
            var typed = List[Column[Float64]]()
            for k in range(keys):
                typed.append(columns[k]._data[Column[Float64]].copy())
            valid = matchable_rows[DType.float64](typed, count)
        else:
            var typed = List[Column[Int64]]()
            for k in range(keys):
                typed.append(columns[k]._data[Column[Int64]].copy())
            valid = matchable_rows[DType.int64](typed, count)
        for row in range(count):
            if valid[row]:
                rows.append(row)
        self.sort_sides.append(side)
        self.sorting.append(
            SortRounds([columns[0].take(rows)], [descending], [True], True)
        )
        self.sort_rows_of.append(rows^)

    def _sorted(mut self) raises:
        """A sort finished: its side's rows in order; then the next sort or
        the match rounds."""
        var order = self.sorting[0].take_order()
        var side = self.sort_sides[0]
        if len(self.sort_rows_of[0]) == 0:
            # Every row was sorted: the order is in input rows already.
            self.sorted_rows[side] = order^
        else:
            ref rows = self.sort_rows_of[0]
            var sorted = List[Int](capacity=len(order))
            for p in order:
                sorted.append(rows[p])
            self.sorted_rows[side] = sorted^
        _ = self.sorting.pop(0)
        _ = self.sort_sides.pop(0)
        _ = self.sort_rows_of.pop(0)
        if len(self.sorting) > 0:
            return
        if self.range:
            self._begin_count()
        else:
            self._begin_pairs()

    def _begin_count(mut self) raises:
        """The point side's sorted keys, and the count round over the
        interval side's rows."""
        var side = 0 if self.point_left else 1
        ref keys = self.left_keys if self.point_left else self.right_keys
        var count = len(self.sorted_rows[side])
        if self.floats:
            self.points_float = List[Float64](length=count, fill=0)
        else:
            self.points_int = List[Int64](length=count, fill=0)
        # The keys in sorted order, a range of positions a task.
        self.ranges = partitions(count, worker_count(count), 64)
        self.width = round_width(len(self.ranges) - 1)
        self.stage = _POINTS

    def _fill_points(mut self, first: Int, end: Int):
        """Points task: the point side's keys at sorted positions [first,
        end)."""
        var side = 0 if self.point_left else 1
        ref keys = self.left_keys if self.point_left else self.right_keys
        ref order = self.sorted_rows[side]
        if self.floats:
            ref column = keys[0]._data[Column[Float64]]
            for i in range(first, end):
                self.points_float[i] = column._get(order[i])
        else:
            ref column = keys[0]._data[Column[Int64]]
            for i in range(first, end):
                self.points_int[i] = column._get(order[i])

    def _counting(mut self) raises:
        """The count round over the interval side's rows."""
        var n = self.right.height() if self.point_left else self.left.height()
        self.starts = List[Int](length=n, fill=0)
        self.stops = List[Int](length=n, fill=0)
        self.ranges = partitions(n, worker_count(n), 64)
        self.range_pairs = List[Int](length=len(self.ranges) - 1, fill=0)
        self.width = 0
        self.stage = _COUNT

    def _count[D: DType](mut self, r: Int) raises -> Int:
        """Count round task for interval range `r`."""
        ref bounds_of = self.right_keys if self.point_left else self.left_keys
        var bounds = List[Column[Scalar[D]]]()
        for k in range(len(self.point_ops)):
            bounds.append(bounds_of[k]._data[Column[Scalar[D]]].copy())
        comptime if D == DType.float64:
            return range_spans[D](
                rebind[List[Scalar[D]]](self.points_float),
                bounds,
                self.point_ops,
                self.ranges[r],
                self.ranges[r + 1],
                self.starts,
                self.stops,
            )
        else:
            return range_spans[D](
                rebind[List[Scalar[D]]](self.points_int),
                bounds,
                self.point_ops,
                self.ranges[r],
                self.ranges[r + 1],
                self.starts,
                self.stops,
            )

    def _begin_pairs(mut self) raises:
        """Cut each sorted side into blocks and list the block pairs whose
        first keys can satisfy the first comparison."""
        var workers = configured_workers()
        var per_side = max(1, Int(sqrt(Float64(workers))) * 2)
        ref left_sorted = self.sorted_rows[0]
        ref right_sorted = self.sorted_rows[1]
        self.left_blocks = _block_bounds(len(left_sorted), per_side)
        self.right_blocks = _block_bounds(len(right_sorted), per_side)
        for a in range(len(self.left_blocks) - 1):
            for b in range(len(self.right_blocks) - 1):
                if self._blocks_meet(a, b):
                    self.block_pairs.append((a, b))
        self.pair_left = List[List[Int]](
            length=len(self.block_pairs), fill=List[Int]()
        )
        self.pair_right = List[List[Int]](
            length=len(self.block_pairs), fill=List[Int]()
        )
        self.stage = _PAIRS
        self.width = round_width(len(self.block_pairs))

    def _blocks_meet(self, a: Int, b: Int) raises -> Bool:
        var lfirst = self.sorted_rows[0][self.left_blocks[a]]
        var llast = self.sorted_rows[0][self.left_blocks[a + 1] - 1]
        var rfirst = self.sorted_rows[1][self.right_blocks[b]]
        var rlast = self.sorted_rows[1][self.right_blocks[b + 1] - 1]
        if self.floats:
            ref lx = self.left_keys[0]._data[Column[Float64]]
            ref rx = self.right_keys[0]._data[Column[Float64]]
            return ie_blocks_overlap[DType.float64](
                lx._get(lfirst),
                lx._get(llast),
                rx._get(rfirst),
                rx._get(rlast),
                self.ops[0],
            )
        ref lx = self.left_keys[0]._data[Column[Int64]]
        ref rx = self.right_keys[0]._data[Column[Int64]]
        return ie_blocks_overlap[DType.int64](
            lx._get(lfirst),
            lx._get(llast),
            rx._get(rfirst),
            rx._get(rlast),
            self.ops[0],
        )

    def _scan_pair[D: DType](mut self, p: Int) raises:
        """Pairs task: the matches of block pair `p`, as input rows."""
        var a = self.block_pairs[p][0]
        var b = self.block_pairs[p][1]
        ref lx = self.left_keys[0]._data[Column[Scalar[D]]]
        ref ly = self.left_keys[1]._data[Column[Scalar[D]]]
        ref rx = self.right_keys[0]._data[Column[Scalar[D]]]
        ref ry = self.right_keys[1]._data[Column[Scalar[D]]]
        var left_first = self.left_blocks[a]
        var left_end = self.left_blocks[a + 1]
        var right_first = self.right_blocks[b]
        var right_end = self.right_blocks[b + 1]
        var bx = List[Scalar[D]](capacity=left_end - left_first)
        var by = List[Scalar[D]](capacity=left_end - left_first)
        for i in range(left_first, left_end):
            var row = self.sorted_rows[0][i]
            bx.append(lx._get(row))
            by.append(ly._get(row))
        var cx = List[Scalar[D]](capacity=right_end - right_first)
        var cy = List[Scalar[D]](capacity=right_end - right_first)
        for i in range(right_first, right_end):
            var row = self.sorted_rows[1][i]
            cx.append(rx._get(row))
            cy.append(ry._get(row))
        var local_left = List[Int]()
        var local_right = List[Int]()
        ie_join_block[D](
            bx, by, cx, cy, self.ops[0], self.ops[1], local_left, local_right
        )
        var rows_left = List[Int](capacity=len(local_left))
        var rows_right = List[Int](capacity=len(local_right))
        for k in range(len(local_left)):
            rows_left.append(self.sorted_rows[0][left_first + local_left[k]])
            rows_right.append(self.sorted_rows[1][right_first + local_right[k]])
        self.pair_left[p] = rows_left^
        self.pair_right[p] = rows_right^

    def _nested_range[D: DType](mut self, r: Int) raises:
        """Pairs task: every pair of left rows in range `r` with the right
        rows, compared directly."""
        var left = List[Column[Scalar[D]]]()
        var right = List[Column[Scalar[D]]]()
        for k in range(len(self.ops)):
            left.append(self.left_keys[k]._data[Column[Scalar[D]]].copy())
            right.append(self.right_keys[k]._data[Column[Scalar[D]]].copy())
        var rows_left = List[Int]()
        var rows_right = List[Int]()
        nested_pairs[D](
            left,
            right,
            self.ops,
            self.ranges[r],
            self.ranges[r + 1],
            self.right.height(),
            rows_left,
            rows_right,
        )
        self.pair_left[r] = rows_left^
        self.pair_right[r] = rows_right^

    def _begin_gather(mut self) raises:
        self.gathering.append(
            GatherRounds(
                self.left._columns, self.left_rows.copy(), configured_workers()
            )
        )
        self.stage = _GATHER_LEFT

    def _assemble(mut self, var right_columns: List[Series]) raises:
        var columns = self.gathered_left^
        self.gathered_left = List[Series]()
        for k in range(len(right_columns)):
            columns.append(right_columns[k].renamed(self.right_names[k]))
        self.result = DataFrame(columns^, height=len(self.left_rows))
        self.left_rows = List[Int]()
        self.right_rows = List[Int]()
        self.stage = _DONE

    def take_result(mut self) raises -> DataFrame:
        """The output, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^


def _block_bounds(count: Int, blocks: Int) -> List[Int]:
    """Boundaries of at most `blocks` nonempty contiguous blocks covering
    [0, count)."""
    var bounds = List[Int]()
    bounds.append(0)
    var n = min(blocks, count)
    for b in range(1, n + 1):
        bounds.append(count * b // n)
    return bounds^


def range_join_schema(
    left: DataFrame, right: DataFrame, node: PlanNode
) raises -> DataFrame:
    """The node's output columns, with no rows."""
    var names = range_join_names(left, right, node.join_suffix())
    var columns = left.clear()._columns.copy()
    var empty = right.clear()
    for k in range(empty.width()):
        columns.append(empty._columns[k].renamed(names[k]))
    return DataFrame(columns^, height=0)


def range_join(
    var left: DataFrame, var right: DataFrame, node: PlanNode
) raises -> DataFrame:
    """RANGE_JOIN node `node` over `left` and `right`, its rounds run here,
    for the plan's executor."""
    var sink = RangeJoinSink(left^, right^, node)
    run_rounds(sink)
    return sink.take_result()
