"""A projection of ranks over all of its input's rows, run as rounds of
tasks of the query's schedule rather than as one task that starts its
kernels' jobs.

Each rank orders every row of the input, so it cannot be split by rows or
by groups: it is one window over one partition. Its work is a round making
the ranked columns contiguous (a fixed-width number column by ranges of its
rows, any other column whole), then the rounds of `RankRounds` (the sort,
the peer boundaries, the ranges' summaries, the ranks written), one rank
after another; the columns the projection passes through are kept as they
are.
"""
from dataframe.expr import (
    COL,
    RANK,
    STRUCT_PACK,
    Expr,
    struct_pack_children,
)
from dataframe.frame import DataFrame
from dataframe.parallel import Rounds, partitions, round_width, worker_count
from dataframe.series import ChunkMerge, Series
from dataframe.storage.column import Column
from dataframe.window_sort import RankRounds

# The round that ran last.
comptime _START = 0
comptime _RECHUNK = 1
comptime _RANKS = 2
comptime _DONE = 3


def ranked_columns(expr: Expr) -> List[String]:
    """The columns a rank over all rows orders by: one for `col(x).rank()`,
    each field's in turn for `as_struct([col(a), col(b), ...]).rank()`;
    none when `expr` is not such a rank."""
    var names = List[String]()
    var n = len(expr._nodes)
    if n < 2 or expr._nodes[n - 1].op != RANK:
        return names^
    ref ranked = expr._nodes[expr._nodes[n - 1].left]
    if expr._nodes[n - 1].left == n - 2 and ranked.op == COL and n == 2:
        names.append(ranked.text)
        return names^
    if expr._nodes[n - 1].left != n - 2 or ranked.op != STRUCT_PACK:
        return names^
    var fields = struct_pack_children(ranked)
    if len(fields) != n - 2:
        return List[String]()
    for f in fields:
        if f < 0 or f >= n - 2 or expr._nodes[f].op != COL:
            return List[String]()
        names.append(expr._nodes[f].text)
    return names^


def whole_input_ranks(exprs: List[Expr]) -> Bool:
    """Whether every expression of a projection is a column passed through
    or a rank over all rows of a column or a struct of columns
    (`ranked_columns`), with at least one rank."""
    var ranks = 0
    for expr in exprs:
        var n = len(expr._nodes)
        if n == 1 and expr._nodes[0].op == COL:
            continue
        if len(ranked_columns(expr)) > 0:
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
    # The ranked columns that are chunked (input column places), their
    # merges by row ranges (none: rechunked whole), and the rechunk
    # round's pieces (chunked column, first row, end row; -1 for whole).
    var chunked: List[Int]
    var merges: List[Optional[ChunkMerge]]
    var pieces: List[Tuple[Int, Int, Int]]
    var width: Int
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
                if expr._name in names:
                    # The projection's own error.
                    self.result = self._eager()
                    self.stage = _DONE
                    return 0
                names.append(expr._name)
                var columns = List[String]()
                if len(expr._nodes) == 1:
                    columns.append(expr._nodes[0].text)
                else:
                    columns = ranked_columns(expr)
                for column in columns:
                    var found = -1
                    for c in range(len(self.input._columns)):
                        ref series = self.input._columns[c]
                        # A categorical is ranked by its strings, as the
                        # evaluator reads it; that path runs whole.
                        if (
                            series.name() == column
                            and not series.dtype().is_categorical()
                        ):
                            found = c
                    if found < 0:
                        self.result = self._eager()
                        self.stage = _DONE
                        return 0
                    if (
                        len(expr._nodes) > 1
                        and self.input._columns[found].is_chunked()
                        and found not in self.chunked
                    ):
                        self._add_chunked(found)
            self.stage = _RECHUNK
            if len(self.pieces) > 0:
                self.width = round_width(len(self.pieces))
                return self.width
        if self.stage == _RECHUNK:
            for k in range(len(self.chunked)):
                if self.merges[k]:
                    self.input._columns[self.chunked[k]] = (
                        self.merges[k].value().result()
                    )
            self.merges = List[Optional[ChunkMerge]]()
            self.pieces = List[Tuple[Int, Int, Int]]()
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
                if len(expr._nodes) == 1:
                    continue
                ref rank = expr._nodes[len(expr._nodes) - 1]
                if len(expr._nodes) == 2:
                    self.ranking.append(
                        RankRounds(
                            self.input.column(expr._nodes[0].text),
                            rank.min_count == 1,
                            rank.text,
                        )
                    )
                else:
                    var fields = List[Series]()
                    for name in ranked_columns(expr):
                        fields.append(self.input.column(name))
                    self.ranking.append(
                        RankRounds(fields, rank.min_count == 1, rank.text)
                    )
                var tasks = self.ranking[0].next_round()
                if tasks > 0:
                    return tasks
                self.ranks[self.at] = self.ranking[0].take_result()
                _ = self.ranking.pop()
        return 0

    def _add_chunked(mut self, c: Int) raises:
        """Input column `c` as pieces of the rechunk round."""
        var k = len(self.chunked)
        self.chunked.append(c)
        ref column = self.input._columns[c]
        var merge = ChunkMerge.of(column)
        if merge:
            # Its rows split over the workers, as a column copied on its
            # own would split them.
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
        elif self.stage == _RANKS:
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
