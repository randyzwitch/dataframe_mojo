"""A projection that reads reductions over all rows of its input
(`with_columns(x.sum())`, `with_columns([x.sum(), y + 1])`,
`select(y - y.mean())`), run as rounds of tasks of the query's schedule
rather than as one task that starts its kernels' jobs.

A value made from reductions alone is a window over one partition with no
order: every row reads the same value, which needs every row. A row-valued
expression that reads such reductions needs them before its first row,
and after that each of its rows needs only its own row. So the work is two
rounds. The first is `ReduceRounds` for every reduction of every
expression (each task folds a range of rows into states of its own; the
states are merged in row order). Each value made from reductions alone is
then evaluated from its reductions' values, one value of work, and spread
over the input's rows as a constant column, as the projection itself
makes it. The second round evaluates the row-valued expressions, each task
a range of rows with the reductions' values read as given, and the ranges
are joined in row order as the column's chunks. The columns the projection
passes through are kept as they are.
"""
from dataframe.binding import ROWS, BoundExpr, bind
from dataframe.execution import (
    ReduceRounds,
    constant_reductions,
    evaluate,
    row_reductions,
)
from dataframe.expr import COL, ROW_LEN, Expr
from dataframe.frame import DataFrame
from dataframe.parallel import Rounds, partitions, worker_count
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.trace import trace_path
from std.collections import Dict

# The round that ran last.
comptime _START = 0
comptime _REDUCE = 1
comptime _ROWS = 2
comptime _DONE = 3

# Rows per batch an expression's argument is evaluated in, as the
# projection evaluates it.
comptime _BATCH = 8192


def whole_input_reductions(exprs: List[Expr]) -> Bool:
    """Whether every expression of a projection is a column passed through,
    one value for every row made from reductions over all rows
    (`constant_reductions`), or a row-valued expression over row-wise
    steps that may read such reductions (`row_reductions`), with at least
    one reduction among them."""
    var reductions = 0
    for expr in exprs:
        if len(expr._nodes) == 1 and expr._nodes[0].op == COL:
            continue
        if constant_reductions(expr):
            reductions += 1
            continue
        var read = row_reductions(expr)
        if read < 0:
            return False
        reductions += read
    return reductions > 0


struct ReductionSink(Movable, Rounds):
    """A SELECT or WITH_COLUMNS that reads reductions over all rows of
    `input` (`whole_input_reductions`), as rounds (see the module
    docstring); `take_result` once no round is left."""

    var input: DataFrame
    var exprs: List[Expr]
    var select: Bool
    var stage: Int
    # By expression, its place in the rounds' expressions (-1 for a
    # column passed through), and the rounds while they run.
    var at: List[Int]
    var reducing: List[ReduceRounds]
    # The row-valued expressions (places in the rounds' expressions), the
    # reductions each reads by node, the row ranges of the row round
    # (each range's first row, then the end), and by range, its rows of
    # each row-valued expression.
    var rows: List[Int]
    var reduced: List[Dict[Int, Series]]
    var bounds: List[Int]
    var pieces: List[List[Series]]
    var result: DataFrame

    def __init__(
        out self, var input: DataFrame, var exprs: List[Expr], select: Bool
    ) raises:
        self.input = input^
        self.exprs = exprs^
        self.select = select
        self.stage = _START
        self.at = List[Int]()
        self.reducing = List[ReduceRounds]()
        self.rows = List[Int]()
        self.reduced = List[Dict[Int, Series]]()
        self.bounds = List[Int]()
        self.pieces = List[List[Series]]()
        self.result = DataFrame(List[Series]())

    def _eager(self) raises -> DataFrame:
        if self.select:
            return self.input.select_exprs(self.exprs)
        return self.input.with_columns(self.exprs)

    def next_round(mut self) raises -> Int:
        if self.stage == _START:
            var names = List[String]()
            var bound = List[BoundExpr]()
            for expr in self.exprs:
                var known = True
                if len(expr._nodes) == 1:
                    known = expr._nodes[0].text in self.input.columns()
                    self.at.append(-1)
                else:
                    try:
                        bound.append(bind(expr, self.input._columns))
                        self.at.append(len(bound) - 1)
                    except:
                        known = False
                    # A categorical made row by row would take a
                    # dictionary of each range's own: the projection makes
                    # it whole.
                    if known:
                        ref last = bound[len(bound) - 1]
                        known = not (
                            last.shape() == ROWS
                            and last.dtypes[
                                len(last.dtypes) - 1
                            ].is_categorical()
                        )
                if expr._name in names or not known:
                    # The projection's own error.
                    self.result = self._eager()
                    self.stage = _DONE
                    return 0
                names.append(expr._name)
            self.reducing.append(
                ReduceRounds(
                    bound^, self.input._columns, self.input.height(), _BATCH
                )
            )
            self.stage = _REDUCE
        if self.stage == _REDUCE:
            var tasks = self.reducing[0].next_round()
            if tasks > 0:
                return tasks
            var ranges = self._begin_rows()
            if ranges > 0:
                return ranges
            self._finish()
        elif self.stage == _ROWS:
            self._finish()
        return 0

    def _begin_rows(mut self) raises -> Int:
        """The row round's tasks, one a range of rows; 0 when no
        expression is row-valued."""
        ref rounds = self.reducing[0]
        var height = self.input.height()
        for e in range(len(rounds.bound)):
            ref b = rounds.bound[e]
            if b.shape() != ROWS:
                continue
            var reduced = rounds.reduced(e)
            # The row count is every row's, not one range's.
            for i in range(len(b.expr._nodes)):
                if b.expr._nodes[i].op == ROW_LEN:
                    reduced[i] = Series("", Column[Int64]([Int64(height)]))
            self.rows.append(e)
            self.reduced.append(reduced^)
        if len(self.rows) == 0:
            return 0
        self.bounds = partitions(height, max(1, worker_count(height)), _BATCH)
        for _ in range(len(self.bounds) - 1):
            self.pieces.append(List[Series]())
        trace_path("window.reduce_row_rounds")
        self.stage = _ROWS
        return len(self.bounds) - 1

    def round_task(mut self, task: Int) raises:
        if self.stage == _REDUCE:
            self.reducing[0].round_task(task)
            return
        var start = self.bounds[task]
        var count = self.bounds[task + 1] - start
        var slices = List[Series](capacity=len(self.input._columns))
        for column in self.input._columns:
            slices.append(column.slice(start, count))
        var values = List[Series](capacity=len(self.rows))
        for r in range(len(self.rows)):
            values.append(
                evaluate(
                    self.reducing[0].bound[self.rows[r]],
                    slices,
                    count,
                    batch_size=_BATCH,
                    reduced=self.reduced[r],
                    workers=1,
                )
            )
        self.pieces[task] = values^

    def _finish(mut self) raises:
        ref rounds = self.reducing[0]
        var height = self.input.height()
        # A selection of these values alone is one row, as the projection
        # makes it; beside any column or row-valued expression, every
        # value is spread over the rows.
        var spread = not self.select or len(self.rows) > 0
        for k in range(len(self.exprs)):
            spread = spread or self.at[k] < 0
        var columns = List[Series]()
        if not self.select:
            columns = self.input._columns.copy()
        for k in range(len(self.exprs)):
            ref expr = self.exprs[k]
            var value: Series
            var row = -1
            for r in range(len(self.rows)):
                if self.at[k] >= 0 and self.rows[r] == self.at[k]:
                    row = r
            if self.at[k] < 0:
                value = self.input.column(expr._nodes[0].text).renamed(
                    expr._name
                )
            elif row >= 0:
                var parts = List[Series](capacity=len(self.pieces))
                for t in range(len(self.pieces)):
                    parts.append(self.pieces[t][row].copy())
                value = Series._from_chunks(parts)
                value._name = expr._name
            else:
                value = evaluate(
                    rounds.bound[self.at[k]],
                    rounds.columns,
                    height,
                    batch_size=_BATCH,
                    reduced=rounds.reduced(self.at[k]),
                )
                if spread:
                    value = value._broadcast(height)
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
        _ = self.reducing.pop()
        self.pieces = List[List[Series]]()
        self.result = DataFrame(columns^, height=height if spread else 1)
        self.stage = _DONE

    def take_result(mut self) raises -> DataFrame:
        """The projection's rows, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^
