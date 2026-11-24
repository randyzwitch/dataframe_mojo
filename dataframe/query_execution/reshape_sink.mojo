"""The reshapes that read their whole input, unpivot and pivot, run as
rounds of tasks of the query's schedule rather than as one task that runs
whole-frame kernels.

Unpivot puts every input row once per `on` column, those of the first
`on` column first: its output is ordered by `on` column, then by input
row. A morsel's rows would land in as many places as there are `on`
columns, between other morsels' rows, while the pipeline's sink keeps
each morsel's output together in morsel order. So unpivot takes its
whole input, and then needs no row-level decisions at all: each index
column is itself repeated once per `on` column, the value column is the
`on` columns one after another, and the variable column is each `on`
column's name repeated. Those columns are made of the input's own arrays
and copied into one array each as a round of tasks: a fixed-width column
by ranges of its rows, any other column whole, and the variable column
written by a task of its own.

A pivot's output has a column for each `on` value its rows hold and a row
for each index key, so nothing is known about the output until every row
has been read. Its work is a grouping: each cell, an index key and `on`
value, reduces its rows. The plan's executor gives the pivot its cells
from the parallel aggregate table (`pivot_cells_plan`), one row per cell
in order of first appearance, with a count of each cell's rows when no
aggregate function allows several. The sink then has one row per cell to
spread out:

- a round of two tasks numbers the cells' index keys and `on` values,
  each in order of first appearance. A key's first cell holds its first
  row, so that is also the order of first appearance in the input.
- the serial step between rounds checks the names, picks the output
  columns and lists, for each one, the cell each output row reads.
- a round of tasks gathers the columns, a column a task.
"""
from std.collections import Dict, Optional

from dataframe.frame import DataFrame
from dataframe.kernels.hashing import encode_rows
from dataframe.parallel import (
    Rounds,
    partitions,
    round_width,
    run_rounds,
    worker_count,
)
from dataframe.planning.model import PIVOT, PlanNode, UNPIVOT
from dataframe.query_execution.operators import _pivoted
from dataframe.series import ChunkMerge, Series
from dataframe.storage.string import StringBuilder

# The round that ran last.
comptime _START = 0
comptime _MERGE = 1
comptime _KEYS = 2
comptime _GATHER = 3
comptime _DONE = 4


def reshape_node(kind: Int) -> Bool:
    """Whether a plan node of `kind` runs as a `ReshapeSink`."""
    return kind == UNPIVOT or kind == PIVOT


def pivot_cell_rows_name(node: PlanNode) -> String:
    """A name for the count of each pivot cell's rows that no column the
    pivot reads has."""
    var name = String("__pivot_cell_rows")
    while name in node.names:
        name += "_"
    return name


struct ReshapeSink(Movable, Rounds):
    """Plan node `node` (`reshape_node`) over all rows of `input`, as
    rounds (see the module docstring); `take_result` once no round is
    left. A pivot's input is its cells (`PlanNode.pivot_cells`), or else
    its rows, which the pivot kernel reduces in the first step."""

    var input: DataFrame
    var node: PlanNode
    var stage: Int
    var height: Int
    var result: DataFrame
    # Unpivot: the output columns, those still in pieces, their merges by
    # row ranges (none: rechunked whole), and the round's pieces (merged
    # column, first row, end row; -1 for the whole column). The variable
    # column's names, written by the round's last task.
    var columns: List[Series]
    var merged: List[Int]
    var merges: List[Optional[ChunkMerge]]
    var pieces: List[Tuple[Int, Int, Int]]
    var names: List[String]
    var width: Int
    # Pivot: the cells' index keys and `on` values numbered (each cell's
    # number, and each number's first cell), the output columns' names,
    # whether some cell has each, the cell each of their rows reads (-1:
    # none), and the columns gathered.
    var row_ids: List[Int]
    var row_firsts: List[Int]
    var column_ids: List[Int]
    var column_firsts: List[Int]
    var labels: List[String]
    var present: List[Bool]
    var picks: List[List[Int]]
    var gathered: List[Series]

    def __init__(out self, var input: DataFrame, node: PlanNode) raises:
        self.height = input.height()
        self.input = input^
        self.node = node.copy()
        self.stage = _START
        self.result = DataFrame(List[Series]())
        self.columns = List[Series]()
        self.merged = List[Int]()
        self.merges = List[Optional[ChunkMerge]]()
        self.pieces = List[Tuple[Int, Int, Int]]()
        self.names = List[String]()
        self.width = 0
        self.row_ids = List[Int]()
        self.row_firsts = List[Int]()
        self.column_ids = List[Int]()
        self.column_firsts = List[Int]()
        self.labels = List[String]()
        self.present = List[Bool]()
        self.picks = List[List[Int]]()
        self.gathered = List[Series]()

    def next_round(mut self) raises -> Int:
        if self.stage == _START:
            if self.node.kind == UNPIVOT:
                return self._unpivot()
            return self._pivot()
        if self.stage == _MERGE:
            self._merged()
            return 0
        if self.stage == _KEYS:
            return self._spread()
        if self.stage == _GATHER:
            self._gathered()
            return 0
        return 0

    def round_task(mut self, task: Int) raises:
        if self.stage == _MERGE:
            for p in range(task, len(self.pieces) + 1, self.width):
                if p == len(self.pieces):
                    self._variable()
                    continue
                var piece = self.pieces[p]
                var k = piece[0]
                if self.merges[k]:
                    self.merges[k].value().copy_rows(piece[1], piece[2])
                else:
                    var c = self.merged[k]
                    self.columns[c] = self.columns[c].rechunk()
        elif self.stage == _KEYS:
            # Job 0 numbers the `on` values, job 1 the index keys (none
            # with no index).
            var indexed = len(self._index()) > 0
            for job in range(task, 2 if indexed else 1, self.width):
                if job == 1:
                    var keys = List[Series]()
                    for name in self._index():
                        keys.append(self.input.column(name))
                    var encoded = encode_rows(keys, nulls_equal=True)
                    swap(self.row_ids, encoded.ids)
                    swap(self.row_firsts, encoded.representatives)
                else:
                    var encoded = encode_rows(
                        [self.input.column(self.node.pivot_on())],
                        nulls_equal=True,
                    )
                    swap(self.column_ids, encoded.ids)
                    swap(self.column_firsts, encoded.representatives)
        elif self.stage == _GATHER:
            var index = self._index()
            for c in range(task, len(self.gathered), self.width):
                if c < len(index):
                    self.gathered[c] = self.input.column(index[c]).take(
                        self.row_firsts
                    )
                elif self.present[c - len(index)]:
                    self.gathered[c] = (
                        self.input.column(self.node.pivot_values())
                        .take_or_null(self.picks[c - len(index)])
                        .renamed(self.labels[c - len(index)])
                    )

    # Unpivot.

    def _unpivot(mut self) raises -> Int:
        """The output columns made of the input's arrays, and the round
        that copies them into one array each."""
        var index = self.node.unpivot_index().copy()
        var output = self.node.unpivot_output_names().copy()
        self.names = self.input._unpivot_on(
            self.node.unpivot_columns(), index, output[0], output[1]
        )
        var repeats = len(self.names)
        for name in index:
            var column = self.input.column(name)
            self.columns.append(
                Series._from_chunks(List[Series](length=repeats, fill=column))
            )
        # The variable column, written by the round's last task.
        self.columns.append(Series(output[0], StringBuilder().finish()))
        var values = List[Series](capacity=repeats)
        for name in self.names:
            values.append(self.input.column(name).renamed(output[1]))
        self.columns.append(Series._from_chunks(values))
        for c in range(len(self.columns)):
            if not self.columns[c].is_chunked():
                continue
            var k = len(self.merged)
            self.merged.append(c)
            var merge = ChunkMerge.of(self.columns[c])
            if merge:
                var n = len(self.columns[c])
                var bounds = partitions(n, worker_count(n), 64)
                for w in range(len(bounds) - 1):
                    if bounds[w] < bounds[w + 1]:
                        self.pieces.append((k, bounds[w], bounds[w + 1]))
            else:
                self.pieces.append((k, 0, -1))
            self.merges.append(merge^)
        self.stage = _MERGE
        self.width = round_width(len(self.pieces) + 1)
        return self.width

    def _variable(mut self) raises:
        """Each `on` column's name once per input row, in `on` order."""
        var bytes = 0
        for name in self.names:
            bytes += name.byte_length() * self.height
        var builder = StringBuilder(self.height * len(self.names), bytes)
        for name in self.names:
            for _ in range(self.height):
                builder.append(name)
        var at = len(self.node.unpivot_index())
        self.columns[at] = Series(self.columns[at].name(), builder^.finish())

    def _merged(mut self) raises:
        for k in range(len(self.merged)):
            if self.merges[k]:
                self.columns[self.merged[k]] = self.merges[k].value().result()
        self.merges = List[Optional[ChunkMerge]]()
        self.pieces = List[Tuple[Int, Int, Int]]()
        var columns = List[Series]()
        swap(columns, self.columns)
        self.result = DataFrame(columns^, height=self.height * len(self.names))
        self.stage = _DONE

    # Pivot.

    def _index(self) -> List[String]:
        return List[String](
            self.node.pivot_columns()[: self.node.pivot_index_count()]
        )

    def _pivot(mut self) raises -> Int:
        """The round that numbers the cells' index keys and `on` values."""
        if not self.node.pivot_cells():
            # Rows, not cells: the kernel reduces and spreads them here.
            self.result = _pivoted(self.input, self.node)
            self.stage = _DONE
            return 0
        var counted = self.node.pivot_cell_rows()
        if counted.byte_length() > 0 and self.height > 0:
            var most = self.input.column(counted).max()
            if not most.is_null() and most.int64() > 1:
                raise Error(
                    "pivot found several rows for one cell; pass an"
                    " aggregate_function"
                )
        self.stage = _KEYS
        if len(self._index()) == 0:
            # No index: one row, every cell's, once there is a cell.
            self.row_ids = List[Int](length=self.height, fill=0)
            if self.height > 0:
                self.row_firsts.append(0)
            self.width = 1
        else:
            self.width = round_width(2)
        return self.width

    def _spread(mut self) raises -> Int:
        """The output columns chosen and checked, the cell each of their
        rows reads, and the round that gathers them."""
        var index = self._index()
        var on = self.input.column(self.node.pivot_on())
        var order = List[Int](capacity=len(self.column_firsts))
        for k in range(len(self.column_firsts)):
            order.append(k)
        if self.node.pivot_sorts_columns():
            order = on.take(self.column_firsts).argsort()
        # Every `on` value's name, checked against the index and the
        # names before it.
        var taken = Dict[String, Bool]()
        for name in index:
            taken[name] = True
        var named = Dict[String, Int]()
        var found = List[String](capacity=len(order))
        for k in order:
            var label = String(on.get(self.column_firsts[k]))
            if label in taken:
                raise Error("pivot column name collides: " + label)
            taken[label] = True
            named[label] = k
            found.append(label)
        if self.node.pivot_found():
            self.labels = found^
        else:
            self.labels = self.node.pivot_output_columns().copy()
        # The output column of each `on` value's cells (-1: none), and the
        # cell each output row reads.
        var n_rows = len(self.row_firsts)
        var output = List[Int](length=len(self.column_firsts), fill=-1)
        self.present = List[Bool](length=len(self.labels), fill=False)
        self.picks = List[List[Int]](length=len(self.labels), fill=List[Int]())
        for c in range(len(self.labels)):
            var k = named.get(self.labels[c])
            if k:
                output[k.value()] = c
                self.present[c] = True
                self.picks[c] = List[Int](length=n_rows, fill=-1)
        for cell in range(self.height):
            var c = output[self.column_ids[cell]]
            if c >= 0:
                self.picks[c][self.row_ids[cell]] = cell
        # A name no cell has: nulls of the reduction's type.
        var dtype = self.input.column(self.node.pivot_values()).dtype()
        self.gathered = List[Series](capacity=len(index) + len(self.labels))
        for name in index:
            self.gathered.append(Series.full_null(name, dtype, 0))
        for c in range(len(self.labels)):
            self.gathered.append(
                Series.full_null(
                    self.labels[c], dtype, 0 if self.present[c] else n_rows
                )
            )
        self.stage = _GATHER
        self.width = round_width(len(self.gathered))
        if self.width == 0:
            self._gathered()
            return 0
        return self.width

    def _gathered(mut self) raises:
        var columns = List[Series]()
        swap(columns, self.gathered)
        self.result = DataFrame(columns^, height=len(self.row_firsts))
        self.picks = List[List[Int]]()
        self.row_ids = List[Int]()
        self.column_ids = List[Int]()
        self.stage = _DONE

    def take_result(mut self) raises -> DataFrame:
        """The output, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^


def reshape(var input: DataFrame, node: PlanNode) raises -> DataFrame:
    """Plan node `node` (`reshape_node`) over `input`, its rounds run
    here: for a node the plan's executor runs over its executed input,
    which may be inside a pipeline's task, where no schedule may start."""
    var sink = ReshapeSink(input^, node)
    run_rounds(sink)
    return sink.take_result()
