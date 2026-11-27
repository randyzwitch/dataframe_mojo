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

A filter by such a predicate needs no column at all. The table hands back
the kept rows partition by partition, in no row order, and a filter must
keep its input's order. Sorting them would compare every pair on one
thread; but they are distinct row numbers below the input's height, so a
flag per row puts them in order with no comparison. One round sets the
flags, each task for a share of the kept rows (distinct rows, so distinct
bytes: no two tasks write the same one). The next round cuts the input
into row ranges, none crossing a chunk's end in any column; each task
reads its ranges' flags in row order and gathers those rows from every
column, and each range's rows become one chunk of every result column, so
the chunks in range order are the kept rows in input order.
"""
from std.bit import count_trailing_zeros
from std.collections import Optional
from std.memory import bitcast

from dataframe.aggregation.hash_table import TOP_ROWS_INLINE, key_supported
from dataframe.aggregation.table_sink import TopRowsRounds
from dataframe.binding import BoundExpr, bind
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.expr import COL, Expr, split_names
from dataframe.frame import DataFrame
from dataframe.kernels.gather import can_filter_aligned_chunks
from dataframe.kernels.top_k import _pattern, top_rows_ranges
from dataframe.parallel import Rounds, configured_workers, task_count
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


def _gather_ranges(columns: List[Series], height: Int, tasks: Int) -> List[Int]:
    """Bounds of the row ranges a filter's gather round splits [0, height)
    into: cut at every column's chunk ends, so each range lies in one chunk
    of every column, and cut further into pieces of about a quarter of a
    task's share, so a task whose rows hold more kept rows than the others
    is not the one every other task waits for. Each range's kept rows
    become one chunk of the result, which is why the pieces are not cut
    finer than that. Returns the ranges' starts followed by `height`."""
    var cuts = List[Int]()
    for column in columns:
        if column.is_chunked():
            for end in column._chunked.value()[].ends:
                if end > 0 and end < height:
                    cuts.append(end)
    cuts.append(height)
    sort(cuts)
    var piece = max(1, (height + 4 * tasks - 1) // (4 * tasks))
    var bounds: List[Int] = [0]
    for end in cuts:
        var start = bounds[len(bounds) - 1]
        if end <= start:
            continue
        var count = (end - start + piece - 1) // piece
        for p in range(1, count):
            bounds.append(start + (end - start) * p // count)
        bounds.append(end)
    return bounds^


def _gather_range(column: Series, rows: List[Int], start: Int) raises -> Series:
    """`column`'s values at `rows` (ascending, all in one chunk of
    `column` at or after row `start`), on the calling thread."""
    if not column.is_chunked() or column.is_constant():
        return column.take(rows)
    ref ends = column._chunked.value()[].ends
    var chunk = 0
    while ends[chunk] <= start:
        chunk += 1
    var offset = ends[chunk - 1] if chunk > 0 else 0
    var local = List[Int](capacity=len(rows))
    for row in rows:
        local.append(row - offset)
    return Series(
        column._name,
        column._chunked.value()[].arrays[chunk].copy(),
        column._dtype,
        column._field_metadata,
    ).take(local)


# The filter's rounds after the table's: setting the kept rows' flags,
# then gathering the flagged rows range by range.
comptime _TABLE = 0
comptime _FLAG_ROWS = 1
comptime _GATHER_RANGES = 2


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
    # in input order, rather than marked in a column (two more rounds,
    # see the module docstring).
    var keep: Bool
    var value: Series
    var rounds: List[TopRowsRounds]
    # While the filter's rounds run: the round (_FLAG_ROWS or
    # _GATHER_RANGES), its task count, the kept rows in the table's order,
    # a flag per input row, the gather's range bounds (`_gather_ranges`),
    # and by range, its gathered piece of every column.
    var stage: Int
    var tasks: Int
    var rows: List[Int]
    var marks: List[UInt8]
    var bounds: List[Int]
    var gathered: List[List[Series]]
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
        self.stage = _TABLE
        self.tasks = 0
        self.rows = List[Int]()
        self.marks = List[UInt8]()
        self.bounds = List[Int]()
        self.gathered = List[List[Series]]()
        self.result = DataFrame(List[Series]())

    def next_round(mut self) raises -> Int:
        if self.stage == _FLAG_ROWS:
            return self._start_gather()
        if self.stage == _GATHER_RANGES:
            self._finish_gather()
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
        self.rows = rows^
        self.marks = List[UInt8](length=self.input.height(), fill=0)
        self.stage = _FLAG_ROWS
        self.tasks = task_count(len(self.rows), configured_workers())
        return self.tasks

    def round_task(mut self, task: Int) raises:
        if self.stage == _FLAG_ROWS:
            # A share of the kept rows; every row is kept once, so no other
            # task writes these bytes.
            var n = len(self.rows)
            var rows = self.rows.unsafe_ptr()
            var flags = self.marks.unsafe_ptr()
            for i in range(
                n * task // self.tasks, n * (task + 1) // self.tasks
            ):
                flags[rows[i]] = 1
            return
        if self.stage == _GATHER_RANGES:
            var flags = self.marks.unsafe_ptr()
            for r in range(task, len(self.bounds) - 1, self.tasks):
                var start = self.bounds[r]
                var end = self.bounds[r + 1]
                var rows = List[Int]()
                var i = start
                # Eight flags a load: most loads in a sparse filter are
                # all zero and skip their rows at once.
                while i + 8 <= end:
                    var word = bitcast[DType.uint64, 1](
                        flags.unsafe_offset(i).unsafe_load[width=8]()
                    )
                    while word != 0:
                        rows.append(i + Int(count_trailing_zeros(word)) // 8)
                        word &= word - 1
                    i += 8
                while i < end:
                    if flags[i] != 0:
                        rows.append(i)
                    i += 1
                if len(rows) == 0:
                    continue
                var pieces = List[Series](capacity=self.input.width())
                for column in self.input._columns:
                    pieces.append(_gather_range(column, rows, start))
                self.gathered[r] = pieces^
            return
        self.rounds[0].round_task(task)

    def _start_gather(mut self) raises -> Int:
        """The gather round, over ranges of the input's rows."""
        self.rows = List[Int]()
        var height = self.input.height()
        var tasks = task_count(height, configured_workers())
        self.bounds = _gather_ranges(self.input._columns, height, tasks)
        var ranges = len(self.bounds) - 1
        self.gathered = List[List[Series]](length=ranges, fill=List[Series]())
        self.stage = _GATHER_RANGES
        self.tasks = max(1, min(tasks, ranges))
        return self.tasks

    def _finish_gather(mut self) raises:
        """Every column's pieces, in range order, as its chunks."""
        var height = 0
        var columns = List[Series](capacity=self.input.width())
        for c in range(self.input.width()):
            var pieces = List[Series]()
            for r in range(len(self.gathered)):
                if len(self.gathered[r]) > 0:
                    pieces.append(self.gathered[r][c].copy())
            if len(pieces) == 0:
                columns.append(self.input._columns[c].slice(0, 0))
            elif len(pieces) == 1:
                columns.append(pieces[0].copy())
            else:
                columns.append(Series._from_chunks(pieces))
            height = len(columns[c])
        self.result = DataFrame(columns^, height=height)
        self.gathered = List[List[Series]]()
        self.marks = List[UInt8]()
        self.bounds = List[Int]()
        self.stage = _TABLE
        self.tasks = 0
        self._release()

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
