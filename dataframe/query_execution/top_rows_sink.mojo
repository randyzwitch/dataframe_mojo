"""A projection whose one computed column keeps each partition's first k
rows by ordinal rank (`col(v).rank("ordinal").over(keys) <= k`), run as
rounds of tasks of the query's schedule.

Ranking every row would sort every partition and scatter the whole
column. A row is kept only when it is among its partition's k best, so
the partitions are groups of an aggregate whose state per group holds its
k best (value, row) pairs (`TopRowsRounds`): each task of the first round
hashes its morsels' keys where they are, into a table of its own, and the
table's finish merges the tables partition by partition. The kept rows
mark the column's true values. A row whose value is null has a null rank,
so its comparison is null too; every other row not kept is false.

A filter by such a predicate needs no column at all: the kept rows, put in
input order, are gathered from every column in one more round, each task
gathering a share of the columns, a chunked column chunk by chunk.
"""
from std.collections import Optional

from dataframe.aggregation.hash_table import TOP_ROWS_INLINE, key_supported
from dataframe.aggregation.table_sink import TopRowsRounds
from dataframe.binding import BoundExpr, bind
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.expr import COL, Expr, split_names
from dataframe.frame import DataFrame
from dataframe.kernels.gather import can_filter_aligned_chunks
from dataframe.kernels.top_k import _pattern, top_rows_ranges
from dataframe.parallel import Rounds, configured_workers
from dataframe.series import Series
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column


struct TopRowsInputs(Movable):
    """What the aggregate table's top-rows state reads for a
    top-k-per-partition predicate: the partition keys and the ranked
    value, cut into chunks at the same rows (`ends`, each chunk's end),
    k, and whether the rank is descending."""

    var keys: List[Series]
    var value: Series
    var ends: List[Int]
    var k: Int
    var descending: Bool

    def __init__(
        out self,
        var keys: List[Series],
        var value: Series,
        var ends: List[Int],
        k: Int,
        descending: Bool,
    ):
        self.keys = keys^
        self.value = value^
        self.ends = ends^
        self.k = k
        self.descending = descending


def top_rows_inputs(
    expr: Expr, columns: List[Series]
) -> Optional[TopRowsInputs]:
    """The inputs of `expr` over `columns` when it is a top-k-per-partition
    predicate whose k best pairs fit inline in a group's state, its value
    a plain number and its keys columns with key words; None otherwise.
    Columns cut at the same rows stay as they are; others are joined into
    one chunk each."""
    var bound: BoundExpr
    try:
        bound = bind(expr, columns)
    except:
        return None
    var found = _pattern(bound)
    var column_node = found[0]
    if column_node < 0 or found[2] > TOP_ROWS_INLINE:
        return None
    ref nodes = bound.expr._nodes
    var value = columns[bound.sources[column_node]].copy()
    var numeric = False
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        numeric = numeric or value._data.isa[Column[Scalar[D]]]()
    if not numeric or value.dtype().is_decimal():
        return None
    var read = List[Series]()
    for name in split_names(nodes[found[1]].text2):
        var at = -1
        for c in range(len(columns)):
            if columns[c].name() == name:
                at = c
        if at < 0 or not key_supported(columns[at]):
            return None
        read.append(columns[at].copy())
    read.append(value^)
    var ends = List[Int]()
    if can_filter_aligned_chunks(read):
        ends = read[0]._chunked.value()[].ends.copy()
    else:
        try:
            for c in range(len(read)):
                if read[c].is_chunked():
                    read[c] = read[c].rechunk()
        except:
            return None
        ends.append(len(read[0]))
    var whole = read.pop()
    return TopRowsInputs(
        read^,
        whole^,
        ends^,
        found[2],
        nodes[nodes[found[1]].left].min_count == 1,
    )


def top_rows_projection(exprs: List[Expr], columns: List[Series]) -> Int:
    """The place of the projection's one computed expression when it is a
    top-k-per-partition predicate the aggregate table answers
    (`top_rows_inputs`) and every other expression passes a column of
    `columns` through under its own name; -1 otherwise."""
    var found = -1
    var names = List[String]()
    for e in range(len(exprs)):
        ref expr = exprs[e]
        if expr._name in names:
            return -1
        names.append(expr._name)
        if len(expr._nodes) == 1 and expr._nodes[0].op == COL:
            var passed = expr._nodes[0].text == expr._name
            var known = False
            for column in columns:
                known = known or column.name() == expr._name
            if passed and known:
                continue
        if found >= 0:
            return -1
        found = e
    if found < 0 or not top_rows_inputs(exprs[found], columns):
        return -1
    return found


def _gather_rows(column: Series, rows: List[Int]) raises -> Series:
    """`column`'s values at `rows` (ascending), on the calling thread: a
    chunked column gathers each chunk's own rows where they lie, and the
    pieces are the result's chunks."""
    if not column.is_chunked() or column.is_constant():
        return column.take(rows)
    var pieces = List[Series]()
    var at = 0
    var start = 0
    for part in column.chunks():
        var end = start + len(part)
        var local = List[Int]()
        while at < len(rows) and rows[at] < end:
            local.append(rows[at] - start)
            at += 1
        if len(local) > 0:
            pieces.append(part.take(local))
        start = end
    if len(pieces) == 0:
        return column.slice(0, 0)
    return Series._from_chunks(pieces)


struct TopRowsSink(Movable, Rounds):
    """A SELECT or WITH_COLUMNS over `input` whose expression at `at` is a
    top-k-per-partition predicate (`top_rows_projection`), or a filter by
    such a predicate (`filtering`), as rounds (see the module docstring);
    `take_result` once no round is left."""

    var input: DataFrame
    var exprs: List[Expr]
    var select: Bool
    var at: Int
    # Whether it filters: the kept rows are gathered from every column,
    # in input order, one more round with a task per share of the
    # columns, rather than marked in a column.
    var keep: Bool
    var value: Series
    var rounds: List[TopRowsRounds]
    # The kept rows in order and, by column, its gathered rows, while the
    # gather round runs.
    var rows: List[Int]
    var tasks: Int
    var gathered: List[Series]
    var result: DataFrame

    @staticmethod
    def filtering(var input: DataFrame, var predicate: Expr) raises -> Self:
        """The rows of `input` that `predicate`, a top-k-per-partition
        predicate the table answers, keeps."""
        var sink = Self(input^, [predicate^], False, 0)
        sink.keep = True
        return sink^

    def __init__(
        out self,
        var input: DataFrame,
        var exprs: List[Expr],
        select: Bool,
        at: Int,
    ) raises:
        var inputs = top_rows_inputs(exprs[at], input._columns)
        if not inputs:
            raise Error("not a top-k-per-partition predicate")
        ref found = inputs.value()
        self.value = found.value.copy()
        self.rounds = List[TopRowsRounds]()
        var height = input.height()
        if found.k > 0 and height > 0:
            var workers = configured_workers()
            self.rounds.append(
                TopRowsRounds(
                    found.keys,
                    found.value,
                    found.k,
                    found.descending,
                    top_rows_ranges(found.ends, workers),
                    workers,
                )
            )
        self.input = input^
        self.exprs = exprs^
        self.select = select
        self.at = at
        self.keep = False
        self.rows = List[Int]()
        self.tasks = 0
        self.gathered = List[Series]()
        self.result = DataFrame(List[Series]())

    def next_round(mut self) raises -> Int:
        if self.tasks > 0:
            # The gather round ran.
            var columns = List[Series]()
            swap(columns, self.gathered)
            self.result = DataFrame(columns^, height=len(self.rows))
            self.tasks = 0
            self._release()
            return 0
        var rows = List[Int]()
        if len(self.rounds) > 0:
            var tasks = self.rounds[0].next_round()
            if tasks > 0:
                return tasks
            swap(rows, self.rounds[0].rows)
            _ = self.rounds.pop()
        if not self.keep:
            self._finish(rows)
            return 0
        sort(rows)
        self.rows = rows^
        var width = self.input.width()
        self.gathered = List[Series](
            length=width, fill=Series("", BoolColumn(List[Bool]()))
        )
        self.tasks = max(1, min(width, configured_workers()))
        return self.tasks

    def round_task(mut self, task: Int) raises:
        if self.tasks == 0:
            self.rounds[0].round_task(task)
            return
        for c in range(task, self.input.width(), self.tasks):
            self.gathered[c] = _gather_rows(self.input._columns[c], self.rows)

    def _release(mut self) raises:
        self.input = DataFrame(List[Series]())
        self.value = Series("", BoolColumn(List[Bool]()))

    def _finish(mut self, rows: List[Int]) raises:
        var height = self.input.height()
        # The kept rows' bits set; the rest false, or null where the value
        # is null.
        var bits = List[UInt8](length=(height + 7) // 8, fill=0)
        for row in rows:
            bits[row >> 3] |= UInt8(1) << UInt8(row & 7)
        var valid = List[UInt8]()
        if self.value.null_count() > 0:
            valid = List[UInt8](length=(height + 7) // 8, fill=0)
            # The value's validity, read chunk by chunk.
            var base = 0
            for part in self.value.chunks():
                comptime for t in range(len(NUMERIC_DTYPES)):
                    comptime D = NUMERIC_DTYPES[t]
                    if part._data.isa[Column[Scalar[D]]]():
                        ref column = part._data[Column[Scalar[D]]]
                        for i in range(len(column)):
                            if column._valid(i):
                                var r = base + i
                                valid[r >> 3] |= UInt8(1) << UInt8(r & 7)
                base += len(part)
        var mask = Series(
            self.exprs[self.at]._name,
            BoolColumn(values=bits^, bits=valid^, length=height),
        )
        if can_filter_aligned_chunks(self.input._columns):
            # Cut where the input's columns are cut, so the frame's columns
            # stay aligned and a step reading it takes each chunk as it is.
            var pieces = List[Series]()
            var start = 0
            for end in self.input._columns[0]._chunked.value()[].ends:
                if end > start:
                    pieces.append(mask.slice(start, end - start))
                start = end
            mask = Series._from_chunks(pieces)
        var columns = List[Series]()
        if self.select:
            for e in range(len(self.exprs)):
                if e == self.at:
                    columns.append(mask.copy())
                else:
                    columns.append(
                        self.input._columns[
                            self.input._index(self.exprs[e]._name)
                        ].copy()
                    )
        else:
            var placed = False
            for c in self.input._columns:
                if c.name() == mask.name():
                    columns.append(mask.copy())
                    placed = True
                else:
                    columns.append(c.copy())
            if not placed:
                columns.append(mask.copy())
        self.result = DataFrame(columns^)
        self._release()

    def take_result(mut self) raises -> DataFrame:
        var result = DataFrame(List[Series]())
        swap(result, self.result)
        return result^
