"""The operations that place rows by their position in the input (tail,
reverse, shift, gather_every, with_row_index) and fills that carry values
along the rows (`fill_null` with a strategy), run as rounds of tasks of
the query's schedule rather than as one task that runs whole-frame kernels.

Each reads its whole input: a row's place in the output depends on how
many rows come before or after it, and a forward or backward fill carries
a value across any number of rows. Morsels run on any worker in any order
and only their sink puts them back in input order, so a morsel knows its
place only once every row is in. The input is therefore complete when the
sink starts, and from then on each output row's source row is known from
its own place, so the work splits by rows:

- tail keeps the input's last rows as a slice, with no copy.
- with_row_index keeps the input's columns and adds one number column, a
  task filling each range of it.
- reverse, shift and gather_every each map output row r to an input row
  (or to none, for a shifted-out row). A round of tasks lists those rows,
  a range each, then the gather's rounds (`GatherRounds`) copy every
  column's rows into place.
- a fill by strategy evaluates each filled column's expression, a column
  a task.
"""
from std.collections import Optional
from std.memory import ArcPointer

from dataframe.binding import ROWS, BoundExpr, bind
from dataframe.execution import evaluate
from dataframe.expr import Expr
from dataframe.frame import DataFrame
from dataframe.kernels.gather import GatherRounds
from dataframe.parallel import (
    Rounds,
    partitions,
    round_width,
    run_rounds,
    worker_count,
)
from dataframe.planning.model import (
    FILL_NULL_STRATEGY,
    GATHER_EVERY,
    PlanNode,
    REVERSE,
    ROW_INDEX,
    SHIFT,
    TAIL,
)
from dataframe.series import Series
from dataframe.storage.column import Column

# The round that ran last.
comptime _START = 0
comptime _ROWS = 1
comptime _GATHER = 2
comptime _INDEX = 3
comptime _FILL = 4
comptime _DONE = 5


def row_order_node(kind: Int) -> Bool:
    """Whether a plan node of `kind` runs as a `RowOrderSink`."""
    return (
        kind == TAIL
        or kind == REVERSE
        or kind == SHIFT
        or kind == GATHER_EVERY
        or kind == ROW_INDEX
        or kind == FILL_NULL_STRATEGY
    )


def gather_every_rows(height: Int, n: Int, offset: Int) raises -> Int:
    """The rows `gather_every(n, offset)` keeps of `height` rows."""
    if n <= 0 or offset < 0:
        raise Error("gather_every needs positive n and nonnegative offset")
    if offset >= height:
        return 0
    return (height - offset - 1) // n + 1


def shift_fills(input: DataFrame, fill_value: Expr) raises -> DataFrame:
    """One row of `fill_value` cast to each column's type, under the
    column's name: the value a shift puts in the rows it empties. It is a
    scalar expression evaluated against the whole input (a literal, or a
    reduction such as `col("a").max()`)."""
    if bind(fill_value, input._columns).shape() == ROWS:
        raise Error("shift fill_value must be a scalar value")
    var columns = List[Series](capacity=input.width())
    for column in input._columns:
        var expression = fill_value.cast(column.dtype()).alias(column.name())
        columns.append(
            evaluate(
                bind(expression, input._columns),
                input._columns,
                input.height(),
                batch_size=8192,
            ).renamed(column.name())
        )
    return DataFrame(columns^, height=1)


struct RowOrderSink(Movable, Rounds):
    """Plan node `node` (`row_order_node`) over all rows of `input`, as
    rounds (see the module docstring); `take_result` once no round is
    left."""

    var input: DataFrame
    var node: PlanNode
    var stage: Int
    var height: Int
    # The output rows, and the ranges of them the tasks of a round fill.
    var rows: Int
    var bounds: List[Int]
    # Each output row's input row (-1: none), as the gather reads them.
    var order: List[Int]
    # The columns gathered (a shift's with its fill row appended).
    var gathered: List[Series]
    var or_null: Bool
    var gather: List[GatherRounds]
    # The row numbers with_row_index adds.
    var index: List[Int64]
    # A fill's expressions, bound to the input, and their columns.
    var fills: List[BoundExpr]
    var filled: List[Series]
    var result: DataFrame

    def __init__(out self, var input: DataFrame, node: PlanNode) raises:
        self.height = input.height()
        self.input = input^
        self.node = node.copy()
        self.stage = _START
        self.rows = 0
        self.bounds = List[Int]()
        self.order = List[Int]()
        self.gathered = List[Series]()
        self.or_null = False
        self.gather = List[GatherRounds]()
        self.index = List[Int64]()
        self.fills = List[BoundExpr]()
        self.filled = List[Series]()
        self.result = DataFrame(List[Series]())

    def next_round(mut self) raises -> Int:
        if self.stage == _START:
            return self._start()
        if self.stage == _ROWS:
            return self._gather_rows()
        if self.stage == _GATHER:
            var tasks = self.gather[0].next_round()
            if tasks > 0:
                return tasks
            self.result = DataFrame(
                self.gather[0].take_result(), height=self.rows
            )
            _ = self.gather.pop()
            self.stage = _DONE
            return 0
        if self.stage == _INDEX:
            self._indexed()
            return 0
        if self.stage == _FILL:
            self._fills_done()
            return 0
        return 0

    def _ranges(mut self, rows: Int) -> Int:
        """Split `rows` output rows into a round's ranges; its tasks."""
        self.rows = rows
        var workers = worker_count(rows)
        self.bounds = partitions(rows, workers, 1)
        return workers

    def _start(mut self) raises -> Int:
        var kind = self.node.kind
        var height = self.height
        if kind == TAIL:
            var n = self.node.tail_count()
            if n < 0:
                self.result = self.input.slice(min(-n, height))
            else:
                self.result = self.input.slice(height - min(n, height))
            self.stage = _DONE
            return 0
        if kind == ROW_INDEX:
            var name = self.node.row_index_name()
            for column in self.input._columns:
                if column.name() == name:
                    raise Error("Row index name collides with column: " + name)
            self.index = List[Int64](unsafe_uninit_length=height)
            self.stage = _INDEX
            if height == 0:
                self._indexed()
                return 0
            return self._ranges(height)
        if kind == FILL_NULL_STRATEGY:
            var limit = Optional[Int]()
            if self.node.fill_limit() >= 0:
                limit = self.node.fill_limit()
            var run_limit = self.input._fill_strategy_limit(
                self.node.text, limit, self.node.names
            )
            var exprs = self.input._strategy_fills(
                self.node.text, run_limit, self.node.names
            )
            if len(exprs) == 0:
                self.result = self.input.copy()
                self.stage = _DONE
                return 0
            for expression in exprs:
                self.fills.append(bind(expression, self.input._columns))
            self.filled = List[Series](
                length=len(exprs), fill=self.input._columns[0].copy()
            )
            self.stage = _FILL
            return round_width(len(exprs))
        # The rows gathered: reverse, shift or gather_every.
        self.gathered = self.input._columns.copy()
        var rows = height
        if kind == SHIFT:
            var n = self.node.shift_count()
            var vacated = height if n >= height or n <= -height else abs(n)
            if vacated == 0:
                self.result = self.input.copy()
                self.stage = _DONE
                return 0
            if len(self.node.exprs) > 0 and self.input.width() > 0:
                # Vacated rows read the fill row, appended after the last.
                var fills = shift_fills(self.input, self.node.exprs[0])
                for k in range(len(self.gathered)):
                    self.gathered[k] = self.gathered[k].append(
                        fills._columns[k]
                    )
            else:
                self.or_null = True
        elif kind == GATHER_EVERY:
            rows = gather_every_rows(
                height, self.node.gather_stride(), self.node.gather_offset()
            )
        self.order = List[Int](unsafe_uninit_length=rows)
        self.stage = _ROWS
        if rows == 0:
            return self._gather_rows()
        return self._ranges(rows)

    def round_task(mut self, task: Int) raises:
        if self.stage == _ROWS:
            self._list_rows(task)
        elif self.stage == _GATHER:
            self.gather[0].round_task(task)
        elif self.stage == _INDEX:
            var start = self.bounds[task]
            var end = self.bounds[task + 1]
            var first = Int64(self.node.row_index_offset())
            var values = self.index.unsafe_ptr()
            for r in range(start, end):
                values[unsafe_offset=r] = first + Int64(r)
        elif self.stage == _FILL:
            var width = round_width(len(self.fills))
            var k = task
            while k < len(self.fills):
                self.filled[k] = evaluate(
                    self.fills[k],
                    self.input._columns,
                    self.height,
                    batch_size=8192,
                )
                k += width

    def _list_rows(mut self, task: Int):
        """Output rows [bounds[task], bounds[task + 1]) mapped to the input
        rows they read."""
        var start = self.bounds[task]
        var end = self.bounds[task + 1]
        var rows = self.order.unsafe_ptr()
        var height = self.height
        var kind = self.node.kind
        if kind == REVERSE:
            for r in range(start, end):
                rows[unsafe_offset=r] = height - 1 - r
        elif kind == GATHER_EVERY:
            var stride = self.node.gather_stride()
            var offset = self.node.gather_offset()
            for r in range(start, end):
                rows[unsafe_offset=r] = offset + r * stride
        else:
            var n = self.node.shift_count()
            var vacated = height if n >= height or n <= -height else abs(n)
            # With a fill, the fill row appended after the last.
            var empty_row = -1 if self.or_null else height
            for r in range(start, end):
                if n > 0:
                    rows[unsafe_offset=r] = (
                        r - vacated if r >= vacated else empty_row
                    )
                else:
                    rows[unsafe_offset=r] = (
                        r + vacated if r + vacated < height else empty_row
                    )

    def _gather_rows(mut self) raises -> Int:
        """Once the rows are listed: the gather, as rounds when its work
        splits (as `DataFrame.take` decides), else here."""
        var height = len(self.order)
        var width = len(self.gathered)
        var workers = worker_count(height)
        if width > 0 and (workers > 1 or (height >= 65536 and width >= 4)):
            self.gather.append(
                GatherRounds(self.gathered, self.order^, workers, self.or_null)
            )
            self.order = List[Int]()
            self.stage = _GATHER
            return self.next_round()
        var columns = List[Series](capacity=width)
        for column in self.gathered:
            if self.or_null:
                columns.append(column.take_or_null(self.order))
            else:
                columns.append(column.take(self.order))
        self.result = DataFrame(columns^, height=height)
        self.stage = _DONE
        return 0

    def _indexed(mut self) raises:
        var columns = List[Series](capacity=self.input.width() + 1)
        columns.append(
            Series(self.node.row_index_name(), Column[Int64](self.index^))
        )
        self.index = List[Int64]()
        for column in self.input._columns:
            columns.append(column.copy())
        self.result = DataFrame(columns^, height=self.height)
        self.stage = _DONE

    def _fills_done(mut self) raises:
        """The input with each filled column in its place."""
        var columns = self.input._columns.copy()
        for k in range(len(self.fills)):
            var name = self.fills[k].expr._name
            for c in range(len(columns)):
                if columns[c].name() == name:
                    columns[c] = self.filled[k].renamed(name)
                    break
        self.result = DataFrame(columns^, height=self.height)
        self.fills = List[BoundExpr]()
        self.filled = List[Series]()
        self.stage = _DONE

    def take_result(mut self) raises -> DataFrame:
        """The rows, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^


def row_order(var input: DataFrame, node: PlanNode) raises -> DataFrame:
    """Plan node `node` (`row_order_node`) over `input`, its rounds run
    here: for a node the plan's executor runs over its executed input,
    which may be inside a pipeline's task, where no schedule may start."""
    var sink = RowOrderSink(input^, node)
    run_rounds(sink)
    return sink.take_result()
