"""A projection of ordered windows over all of its input's rows, run as
rounds of tasks of the query's schedule rather than as one task that
starts its kernels' jobs.

A window ordered by some keys with no partition keys
(`f.over(order_by=...)`) reads every row in that order, so it cannot be
split by rows or by partitions: it is one window over one partition. The
columns the windows read are first made contiguous by a round (a
fixed-width number column by ranges of its rows, any other column whole).
Each window then runs as the rounds of `OverRounds` (the sort, the peer boundaries, the
gather into sorted order, one task evaluating it, the results put back in
row order), one window after another; the columns the projection passes
through are kept as they are.
"""
from dataframe.binding import bind
from dataframe.execution import OverRounds, whole_input_window
from dataframe.expr import COL, Expr, split_names
from dataframe.frame import DataFrame
from dataframe.parallel import Rounds, partitions, round_width, worker_count
from dataframe.series import ChunkMerge, Series
from dataframe.storage.column import Column

# The round that ran last.
comptime _START = 0
comptime _RECHUNK = 1
comptime _WINDOWS = 2
comptime _DONE = 3


def whole_input_windows(exprs: List[Expr]) -> Bool:
    """Whether every expression of a projection is a column passed through
    or an ordered window over all rows (`whole_input_window`), with at
    least one window."""
    var windows = 0
    for expr in exprs:
        if len(expr._nodes) == 1 and expr._nodes[0].op == COL:
            continue
        if whole_input_window(expr):
            windows += 1
            continue
        return False
    return windows > 0


struct OrderedWindowSink(Movable, Rounds):
    """A SELECT or WITH_COLUMNS of ordered windows over all rows of
    `input` (`whole_input_windows`), as rounds (see the module docstring);
    `take_result` once no round is left."""

    var input: DataFrame
    var exprs: List[Expr]
    var select: Bool
    var batch_size: Int
    var stage: Int
    # The window being computed (an index of `exprs`), its rounds, and by
    # expression its values once made.
    var at: Int
    var running: List[OverRounds]
    var values: List[Series]
    # The chunked columns the windows read (input column places), their
    # merges by row ranges (none: rechunked whole), and the rechunk
    # round's pieces (chunked column, first row, end row; -1 for whole).
    var chunked: List[Int]
    var merges: List[Optional[ChunkMerge]]
    var pieces: List[Tuple[Int, Int, Int]]
    var width: Int
    var result: DataFrame

    def __init__(
        out self,
        var input: DataFrame,
        var exprs: List[Expr],
        select: Bool,
        batch_size: Int,
    ) raises:
        self.input = input^
        self.exprs = exprs^
        self.select = select
        self.batch_size = batch_size
        self.stage = _START
        self.at = -1
        self.running = List[OverRounds]()
        self.values = List[Series](
            length=len(self.exprs),
            fill=Series("", Column[Int64](List[Int64]())),
        )
        self.chunked = List[Int]()
        self.merges = List[Optional[ChunkMerge]]()
        self.pieces = List[Tuple[Int, Int, Int]]()
        self.width = 0
        self.result = DataFrame(List[Series]())

    def _eager(self) raises -> DataFrame:
        if self.select:
            return self.input.select_exprs(self.exprs)
        return self.input.with_columns(self.exprs)

    def next_round(mut self) raises -> Int:
        if self.stage == _START:
            var names = List[String]()
            for expr in self.exprs:
                var found = False
                if len(expr._nodes) == 1:
                    found = expr._nodes[0].text in self.input.columns()
                else:
                    found = True
                    for node in expr._nodes:
                        if node.op == COL and not (
                            node.text in self.input.columns()
                        ):
                            found = False
                if expr._name in names or not found:
                    # The projection's own error.
                    self.result = self._eager()
                    self.stage = _DONE
                    return 0
                names.append(expr._name)
            # Every column a window reads: its argument's, its order
            # keys.
            for expr in self.exprs:
                if len(expr._nodes) == 1:
                    continue
                var read = List[String]()
                for node in expr._nodes:
                    if node.op == COL:
                        read.append(node.text)
                read.extend(split_names(expr._nodes[len(expr._nodes) - 1].text))
                for c in range(len(self.input._columns)):
                    if (
                        self.input._columns[c].name() in read
                        and self.input._columns[c].is_chunked()
                        and c not in self.chunked
                    ):
                        self._add_chunked(c)
            self.stage = _RECHUNK
            if len(self.pieces) > 0:
                self.width = round_width(len(self.pieces))
                return self.width
        if self.stage == _RECHUNK:
            for k in range(len(self.chunked)):
                if self.merges[k]:
                    self.input._columns[self.chunked[k]] = (
                        self.merges[k].value().merged.copy()
                    )
            self.merges = List[Optional[ChunkMerge]]()
            self.pieces = List[Tuple[Int, Int, Int]]()
            self.stage = _WINDOWS
        if self.stage == _WINDOWS:
            if len(self.running) > 0:
                var tasks = self.running[0].next_round()
                if tasks > 0:
                    return tasks
                self.values[self.at] = self.running[0].take_result()
                _ = self.running.pop()
            # The next window to compute.
            while True:
                self.at += 1
                if self.at >= len(self.exprs):
                    self._finish()
                    return 0
                ref expr = self.exprs[self.at]
                if len(expr._nodes) == 1:
                    continue
                var bound = bind(expr, self.input._columns)
                self.running.append(
                    OverRounds(
                        bound,
                        self.input._columns,
                        len(expr._nodes) - 1,
                        self.input.height(),
                        self.batch_size,
                    )
                )
                var tasks = self.running[0].next_round()
                if tasks > 0:
                    return tasks
                self.values[self.at] = self.running[0].take_result()
                _ = self.running.pop()
        return 0

    def _add_chunked(mut self, c: Int) raises:
        """Input column `c` as pieces of the rechunk round."""
        var k = len(self.chunked)
        self.chunked.append(c)
        ref column = self.input._columns[c]
        var merge = ChunkMerge.of(column)
        if merge:
            var n = len(column)
            var bounds = partitions(n, worker_count(n), 64)
            for w in range(len(bounds) - 1):
                if bounds[w] < bounds[w + 1]:
                    self.pieces.append((k, bounds[w], bounds[w + 1]))
        else:
            self.pieces.append((k, 0, -1))
        self.merges.append(merge^)

    def round_task(mut self, task: Int) raises:
        if self.stage == _RECHUNK:
            for p in range(task, len(self.pieces), self.width):
                var piece = self.pieces[p]
                var k = piece[0]
                if self.merges[k]:
                    self.merges[k].value().copy_rows(piece[1], piece[2])
                else:
                    var c = self.chunked[k]
                    self.input._columns[c] = self.input._columns[c].rechunk()
        elif self.stage == _WINDOWS:
            self.running[0].round_task(task)

    def _finish(mut self) raises:
        var columns = List[Series]()
        if not self.select:
            columns = self.input._columns.copy()
        for e in range(len(self.exprs)):
            ref expr = self.exprs[e]
            var value: Series
            if len(expr._nodes) == 1:
                value = self.input.column(expr._nodes[0].text).renamed(
                    expr._name
                )
            else:
                value = self.values[e].renamed(expr._name)
            if self.select:
                columns.append(value^)
                continue
            var replacement = -1
            for i in range(len(columns)):
                if columns[i].name() == value.name():
                    replacement = i
                    break
            if replacement < 0:
                columns.append(value^)
            else:
                columns[replacement] = value^
        self.values = List[Series]()
        self.result = DataFrame(columns^, height=self.input.height())
        self.stage = _DONE

    def take_result(mut self) raises -> DataFrame:
        """The projection's rows, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^
