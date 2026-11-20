"""A sort of a whole input, or its first rows (a top-N), run as rounds of
tasks of the query's schedule rather than as one task that starts its
kernels' jobs.

The rounds are the key sort's (`SortRounds`: runs, merge boundaries,
merged partitions; or `TopNRounds`: a heap per range of rows, combined)
followed by the gather's (`GatherRounds`: chunked columns made
contiguous, then every column's rows gathered into sorted order, one
range of output rows a task). Each round needs only the one before it,
so a pipeline's finish runs them one after another. The gathered columns
may be fewer than the input's: a projection of columns above the sort
gathers only its own, so keys it drops are never copied.
"""
from std.collections import Optional

from dataframe.frame import DataFrame
from dataframe.kernels.gather import GatherRounds
from dataframe.parallel import Rounds, worker_count
from dataframe.series import Series
from dataframe.sorting.sort import SortRounds, TopNRounds, top_n

# The round that ran last.
comptime _KEYS = 0
comptime _ROWS = 1
comptime _DONE = 2


struct SortSink(Movable, Rounds):
    """The rows of `input` sorted by key columns, all of them or the first
    `limit`, as rounds (see the module docstring); `take_result` once no
    round is left."""

    # The columns gathered into sorted order.
    var input: DataFrame
    var limit: Int
    # The rows of the result.
    var rows: Int
    # The key sort: a full sort, or (`top_n`) a top-N.
    var keys: List[SortRounds]
    var top: List[TopNRounds]
    var gather: List[GatherRounds]
    var stage: Int
    var result: DataFrame

    def __init__(
        out self,
        var input: DataFrame,
        by: List[String],
        descending: List[Bool],
        nulls_last: List[Bool],
        maintain_order: Bool,
        limit: Int = -1,
        gathered: Optional[DataFrame] = None,
    ) raises:
        """`gathered`: the columns to gather, with `input`'s rows (none:
        every column of `input`)."""
        var keys = List[Series](capacity=len(by))
        for name in by:
            keys.append(input.column(name))
        self.limit = limit
        self.rows = 0
        self.keys = List[SortRounds]()
        self.top = List[TopNRounds]()
        if top_n(input.height(), limit):
            self.top.append(
                TopNRounds(keys, descending, nulls_last, maintain_order, limit)
            )
        else:
            self.keys.append(
                SortRounds(keys, descending, nulls_last, maintain_order)
            )
        if gathered:
            self.input = gathered.value().copy()
        else:
            self.input = input^
        self.gather = List[GatherRounds]()
        self.stage = _KEYS
        self.result = DataFrame(List[Series]())

    def next_round(mut self) raises -> Int:
        if self.stage == _KEYS:
            var order: List[Int]
            if len(self.top) > 0:
                var tasks = self.top[0].next_round()
                if tasks > 0:
                    return tasks
                order = self.top[0].take_order()
                _ = self.top.pop()
            else:
                var tasks = self.keys[0].next_round()
                if tasks > 0:
                    return tasks
                order = self.keys[0].take_order()
                _ = self.keys.pop()
                if self.limit >= 0 and self.limit < len(order):
                    order.shrink(self.limit)
            var height = len(order)
            self.rows = height
            var workers = worker_count(height)
            var width = self.input.width()
            # As `DataFrame.take`: columns gather in parallel with rows to
            # split, or several columns of a medium height; else here.
            if width > 0 and (workers > 1 or (height >= 65536 and width >= 4)):
                self.gather.append(
                    GatherRounds(self.input._columns, order^, workers)
                )
                self.stage = _ROWS
            else:
                var columns = List[Series](capacity=width)
                for column in self.input._columns:
                    columns.append(column.take(order))
                self.result = DataFrame(columns^, height=height)
                self.stage = _DONE
                return 0
        if self.stage == _ROWS:
            var tasks = self.gather[0].next_round()
            if tasks > 0:
                return tasks
            var height = self.rows
            self.result = DataFrame(self.gather[0].take_result(), height=height)
            _ = self.gather.pop()
            self.stage = _DONE
        return 0

    def round_task(mut self, task: Int) raises:
        if self.stage == _KEYS:
            if len(self.top) > 0:
                self.top[0].round_task(task)
            else:
                self.keys[0].round_task(task)
        elif self.stage == _ROWS:
            self.gather[0].round_task(task)

    def take_result(mut self) raises -> DataFrame:
        """The sorted rows, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^
