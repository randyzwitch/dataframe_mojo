"""A projection of ranks over all of its input's rows, run as rounds of
tasks of the query's schedule rather than as one task that starts its
kernels' jobs.

Each rank orders every row of the input, so it cannot be split by rows or
by groups: it is one window over one partition. Its work is the rounds of
`RankRounds` (the sort, the peer boundaries, the ranges' summaries, the
ranks written), one rank after another; the columns the projection passes
through are kept as they are.
"""
from dataframe.expr import COL, RANK, Expr
from dataframe.frame import DataFrame
from dataframe.parallel import Rounds
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.window_sort import RankRounds

# The round that ran last.
comptime _START = 0
comptime _RANKS = 1
comptime _DONE = 2


def whole_input_ranks(exprs: List[Expr]) -> Bool:
    """Whether every expression of a projection is a column passed through
    or a rank of a column over all rows (`col(x).rank(...)`), with at
    least one rank."""
    var ranks = 0
    for expr in exprs:
        var n = len(expr._nodes)
        if n == 1 and expr._nodes[0].op == COL:
            continue
        if (
            n == 2
            and expr._nodes[1].op == RANK
            and expr._nodes[1].left == 0
            and expr._nodes[0].op == COL
        ):
            ranks += 1
            continue
        return False
    return ranks > 0


struct RankSink(Movable, Rounds):
    """A SELECT or WITH_COLUMNS of ranks over all rows of `input`
    (`whole_input_ranks`), as rounds (see the module docstring);
    `take_result` once no round is left."""

    var input: DataFrame
    var exprs: List[Expr]
    var select: Bool
    var stage: Int
    # The rank being computed (an index of `exprs`), its rounds, and by
    # expression its ranks once made.
    var at: Int
    var ranking: List[RankRounds]
    var ranks: List[Series]
    var result: DataFrame

    def __init__(
        out self, var input: DataFrame, var exprs: List[Expr], select: Bool
    ) raises:
        self.input = input^
        self.exprs = exprs^
        self.select = select
        self.stage = _START
        self.at = -1
        self.ranking = List[RankRounds]()
        self.ranks = List[Series](
            length=len(self.exprs), fill=Series("", Column[Int64](List[Int64]()))
        )
        self.result = DataFrame(List[Series]())

    def _eager(self) raises -> DataFrame:
        if self.select:
            return self.input.select_exprs(self.exprs)
        return self.input.with_columns(self.exprs)

    def next_round(mut self) raises -> Int:
        if self.stage == _START:
            var names = List[String]()
            for expr in self.exprs:
                if expr._name in names:
                    # The projection's own error.
                    self.result = self._eager()
                    self.stage = _DONE
                    return 0
                names.append(expr._name)
                var column = expr._nodes[0].text
                var found = False
                for c in self.input._columns:
                    if c.name() == column:
                        # A categorical is ranked by its strings, as the
                        # evaluator reads it; that path runs whole.
                        found = not c.dtype().is_categorical()
                if not found:
                    self.result = self._eager()
                    self.stage = _DONE
                    return 0
            self.stage = _RANKS
        if self.stage == _RANKS:
            if len(self.ranking) > 0:
                var tasks = self.ranking[0].next_round()
                if tasks > 0:
                    return tasks
                self.ranks[self.at] = self.ranking[0].take_result()
                _ = self.ranking.pop()
            # The next rank to compute.
            while True:
                self.at += 1
                if self.at >= len(self.exprs):
                    self._finish()
                    return 0
                ref expr = self.exprs[self.at]
                if len(expr._nodes) != 2:
                    continue
                ref rank = expr._nodes[1]
                self.ranking.append(
                    RankRounds(
                        self.input.column(expr._nodes[0].text),
                        rank.min_count == 1,
                        rank.text,
                    )
                )
                var tasks = self.ranking[0].next_round()
                if tasks > 0:
                    return tasks
                self.ranks[self.at] = self.ranking[0].take_result()
                _ = self.ranking.pop()
        return 0

    def round_task(mut self, task: Int) raises:
        if self.stage == _RANKS:
            self.ranking[0].round_task(task)

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
                value = self.ranks[e].renamed(expr._name)
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
        self.ranks = List[Series]()
        self.result = DataFrame(columns^, height=self.input.height())
        self.stage = _DONE

    def take_result(mut self) raises -> DataFrame:
        """The projection's rows, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^
