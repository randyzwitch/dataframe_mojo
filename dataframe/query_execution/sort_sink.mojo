"""A sort of a whole input, run as rounds of tasks of the query's schedule
rather than as one task that starts its kernels' jobs.

The rounds are the key sort's (`SortRounds`: runs, merge boundaries,
merged partitions) followed by the gather's (`GatherRounds`: chunked
columns made contiguous, then every column's rows gathered into sorted
order, one range of output rows a task). Each round needs only the one
before it, so a pipeline's finish runs them one after another.
"""
from dataframe.frame import DataFrame
from dataframe.kernels.gather import GatherRounds
from dataframe.parallel import Rounds, worker_count
from dataframe.series import Series
from dataframe.sorting.sort import SortRounds

# The round that ran last.
comptime _KEYS = 0
comptime _ROWS = 1
comptime _DONE = 2


struct SortSink(Movable, Rounds):
    """The rows of `input` sorted by key columns, as rounds (see the
    module docstring); `take_result` once no round is left."""

    var input: DataFrame
    var keys: SortRounds
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
    ) raises:
        var keys = List[Series](capacity=len(by))
        for name in by:
            keys.append(input.column(name))
        self.keys = SortRounds(keys, descending, nulls_last, maintain_order)
        self.input = input^
        self.gather = List[GatherRounds]()
        self.stage = _KEYS
        self.result = DataFrame(List[Series]())

    def next_round(mut self) raises -> Int:
        if self.stage == _KEYS:
            var tasks = self.keys.next_round()
            if tasks > 0:
                return tasks
            var order = self.keys.take_order()
            var height = len(order)
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
            var height = self.input.height()
            self.result = DataFrame(self.gather[0].take_result(), height=height)
            _ = self.gather.pop()
            self.stage = _DONE
        return 0

    def round_task(mut self, task: Int) raises:
        if self.stage == _KEYS:
            self.keys.round_task(task)
        elif self.stage == _ROWS:
            self.gather[0].round_task(task)

    def take_result(mut self) raises -> DataFrame:
        """The sorted rows, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^
