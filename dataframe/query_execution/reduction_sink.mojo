"""A projection of values that each stand for every row of its input,
made from reductions over all of those rows (`with_columns(x.sum())`), run
as rounds of tasks of the query's schedule rather than as one task that
starts its kernels' jobs.

Such a value is a window over one partition with no order: every row reads
the same value, which needs every row. Its reductions run as the rounds of
`ReduceRounds` (each task folds a range of rows into states of its own;
the states are merged in row order). Each expression is then evaluated
from its reductions' values, one value of work, and spread over the
input's rows as a constant column, as the projection itself makes it. The
columns the projection passes through are kept as they are.
"""
from dataframe.binding import BoundExpr, bind
from dataframe.execution import ReduceRounds, constant_reductions, evaluate
from dataframe.expr import COL, Expr
from dataframe.frame import DataFrame
from dataframe.parallel import Rounds
from dataframe.series import Series

# The round that ran last.
comptime _START = 0
comptime _REDUCE = 1
comptime _DONE = 2

# Rows per batch an expression's argument is evaluated in, as the
# projection evaluates it.
comptime _BATCH = 8192


def whole_input_reductions(exprs: List[Expr]) -> Bool:
    """Whether every expression of a projection is a column passed through
    or one value for every row made from reductions over all rows
    (`constant_reductions`), with at least one such value."""
    var values = 0
    for expr in exprs:
        if len(expr._nodes) == 1 and expr._nodes[0].op == COL:
            continue
        if constant_reductions(expr):
            values += 1
            continue
        return False
    return values > 0


struct ReductionSink(Movable, Rounds):
    """A SELECT or WITH_COLUMNS of values made from reductions over all
    rows of `input` (`whole_input_reductions`), as rounds (see the module
    docstring); `take_result` once no round is left."""

    var input: DataFrame
    var exprs: List[Expr]
    var select: Bool
    var stage: Int
    # By expression, its place in the rounds' expressions (-1 for a
    # column passed through), and the rounds while they run.
    var at: List[Int]
    var reducing: List[ReduceRounds]
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
            self._finish()
        return 0

    def round_task(mut self, task: Int) raises:
        self.reducing[0].round_task(task)

    def _finish(mut self) raises:
        ref rounds = self.reducing[0]
        var height = self.input.height()
        # A selection of these values alone is one row, as the projection
        # makes it; beside any column, every value is spread over the
        # rows.
        var spread = not self.select
        for k in range(len(self.exprs)):
            spread = spread or self.at[k] < 0
        var columns = List[Series]()
        if not self.select:
            columns = self.input._columns.copy()
        for k in range(len(self.exprs)):
            ref expr = self.exprs[k]
            var value: Series
            if self.at[k] < 0:
                value = self.input.column(expr._nodes[0].text).renamed(
                    expr._name
                )
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
        self.result = DataFrame(columns^, height=height if spread else 1)
        self.stage = _DONE

    def take_result(mut self) raises -> DataFrame:
        """The projection's rows, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^
