"""A projection holding window functions, run as rounds of tasks of the
query's schedule rather than as one task that starts its kernels' jobs.

Every window of such a projection partitions its rows by named keys. The
keys all of them share split the rows into groups that no partition of any
window crosses: two rows with different shared keys are never in one
partition. So the projection gives the same rows when it runs on each such
group apart from the others, and the groups can run at the same time, each
on one thread, with no shared state between them.

The work is four rounds:

1. Sink: each task hashes the shared keys of one range of rows and copies
   the range's row numbers into bins by the top bits of the hash, keeping
   row order within each bin.
2. Merge: each task takes whole bins, largest first, until none is left;
   for a bin it gathers the rows of the columns the projection reads and
   evaluates the projection over them. Everything the evaluation starts
   runs on the task's own thread, since the other tasks keep the other
   threads busy.
3. Source: each task takes whole bins again and writes each bin's results
   to the bin's rows of the output columns, so the output is in input row
   order.
4. Pack: each task packs one range of rows' validity into bits, and
   gathers by row range the result columns whose values cannot be written
   by row (strings, booleans, lists...).

A bin's rows keep their input order, so every window sees each partition's
rows in the same order as over the whole input, and the results equal the
whole-input evaluation's.
"""
from std.atomic import Atomic
from std.memory import Pointer

from dataframe.execution import _used_columns
from dataframe.expr import (
    Expr,
    COL,
    IMPLODE,
    OVER,
    OVER_GLOBAL,
    OVER_MAPPING_MASK,
    _SCALAR_SOURCE,
    is_reduction,
    is_sequence,
    is_window,
    node_children,
    split_names,
)
from dataframe.frame import DataFrame
from dataframe.kernels.gather import (
    GATHER_DTYPES,
    _allocate,
    _payload_address,
    _set_bits,
)
from dataframe.kernels.partition import _hash_column
from dataframe.parallel import Rounds, configured_workers, partitions
from dataframe.selectors import expand_all
from dataframe.series import Series
from dataframe.sorting.sort_key import STANDARD_VECTOR_SIZE
from dataframe.storage.allocation import huge_uninit, release_huge
from dataframe.storage.column import Column

# The round that ran last.
comptime _START = 0
comptime _SINK = 1
comptime _MERGE = 2
comptime _SOURCE = 3
comptime _PACK = 4
comptime _DONE = 5

# Bins are the top bits of the key hash. There are at least 16, so the
# threads have bins to pick from when partitions differ in size, and at
# most 256, so a bin's share of every sink task's histogram stays small.
# Between those, the bits grow two at a time while a bin would average
# more rows than one row group (122,880 rows): a bin's rows are gathered
# and evaluated by one thread, and bins of bounded size keep any one of
# them from holding the last thread long after the others finish.
comptime _FIRST_BITS = 4
comptime _MOST_BITS = 8
comptime _BIN_ROWS = 122880


def shared_partition_keys(exprs: List[Expr]) -> List[String]:
    """The partition keys every window of `exprs` partitions by, in the
    order the first window names them; empty unless every expression is
    row-local outside its over() windows and every window partitions by
    named keys (none over all rows, none mapping results back by join or
    explode). Rows that differ on these keys are in different partitions
    of every window."""
    var shared = List[String]()
    var found = False
    for expr in exprs:
        var n = len(expr._nodes)
        if n == 0:
            continue
        # Nodes reached from the root without passing an over(): their
        # value at a row may depend only on that row.
        var outside = List[Bool](length=n, fill=False)
        outside[n - 1] = True
        for i in range(n - 1, -1, -1):
            ref node = expr._nodes[i]
            if node.op == OVER:
                if (node.integer & OVER_GLOBAL) != 0 or (
                    node.integer & OVER_MAPPING_MASK
                ) != 0:
                    return List[String]()
                var keys = List[String]()
                for name in split_names(node.text2):
                    if name.byte_length() > 0:
                        keys.append(name)
                if not found:
                    shared = keys^
                    found = True
                else:
                    var kept = List[String]()
                    for name in shared:
                        if name in keys:
                            kept.append(name)
                    shared = kept^
                if len(shared) == 0:
                    return List[String]()
            elif outside[i]:
                if (
                    is_reduction(node.op)
                    or is_sequence(node.op)
                    or is_window(node.op)
                    or node.op == IMPLODE
                    or node.op == _SCALAR_SOURCE
                ):
                    return List[String]()
            for child in node_children(node):
                if child >= i:
                    # Not a tree in evaluation order: not split.
                    return List[String]()
                if node.op != OVER and outside[i]:
                    outside[child] = True
    return shared^ if found else List[String]()


def _column_at(columns: List[Series], name: String) -> Int:
    """The place of the column named `name`, or -1."""
    for c in range(len(columns)):
        if columns[c].name() == name:
            return c
    return -1


def _fixed_width(series: Series) -> Bool:
    """Whether the column's values are one fixed-width word per row that a
    task can write at any row."""
    if series.dtype().is_categorical():
        return False
    comptime for d in range(len(GATHER_DTYPES)):
        comptime D = GATHER_DTYPES[d]
        if series._data.isa[Column[Scalar[D]]]():
            return True
    return False


def _write_rows(value: Series, rows: List[Int], output: Int, valid: Int) raises:
    """Row i of `value` to output row rows[i], and its validity to the
    byte for that row when `valid` holds an address."""
    comptime for d in range(len(GATHER_DTYPES)):
        comptime D = GATHER_DTYPES[d]
        if value._data.isa[Column[Scalar[D]]]():
            ref column = value._data[Column[Scalar[D]]]
            var source = column._ptr()
            var target = Pointer[Scalar[D], MutAnyOrigin](
                unsafe_from_address=output
            )
            for i in range(len(rows)):
                target[unsafe_offset=rows[i]] = source[unsafe_offset=i]
            if valid != 0:
                var bytes = Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=valid
                )
                if len(column._bits[]) == 0:
                    for i in range(len(rows)):
                        bytes[unsafe_offset=rows[i]] = 1
                else:
                    for i in range(len(rows)):
                        bytes[unsafe_offset=rows[i]] = UInt8(
                            1
                        ) if column._valid(i) else UInt8(0)
            return
    raise Error("window result column has no fixed-width storage")


struct WindowSink(Movable, Rounds):
    """A SELECT or WITH_COLUMNS of window functions over `input`, split by
    the windows' shared partition keys (`shared_partition_keys`) and run
    as rounds (see the module docstring); `result` once no round is
    left."""

    var input: DataFrame
    var exprs: List[Expr]
    var select: Bool
    var keys: List[String]
    var stage: Int
    var height: Int
    var bits: Int
    # Row ranges of the sink tasks, then of the pack tasks.
    var bounds: List[Int]
    # The key columns and the columns the evaluated expressions read.
    var key_columns: List[Series]
    var reads: List[Series]
    # The expressions after expanding selectors; those evaluated per bin
    # (the others are input columns passed through), and by expression,
    # its place among the evaluated ones (-1 for passed through).
    var expanded: List[Expr]
    var evaluated: List[Expr]
    var place: List[Int]
    # Row numbers by sink task, then bin: task t's rows in bin b are
    # rows[starts[t][b]:starts[t][b + 1]].
    var rows: List[Int]
    var starts: List[List[Int]]
    # The bins holding rows, largest first, and by bin: its row count,
    # its first place in that order of all bins' rows, its rows and its
    # evaluated columns.
    var order: List[Int]
    var sizes: List[Int]
    var base: List[Int]
    var bin_rows: List[List[Int]]
    var results: List[List[Series]]
    # The next bin of `order` a task takes.
    var next: Atomic[Int64]
    # By evaluated expression: its output, whether it is written by row,
    # where its payload is, and its validity as a byte a row (empty when
    # every row is valid), then as bits.
    var outputs: List[Series]
    var scatter: List[Bool]
    var payloads: List[Int]
    var valid: List[List[UInt8]]
    var bits_out: List[List[UInt8]]
    # Each row's place in the bins' order, and the output pieces each
    # pack task gathered for the columns not written by row.
    var inverse: List[Int]
    var combined: List[Series]
    var pieces: List[List[Series]]
    var result: DataFrame

    def __init__(
        out self,
        var input: DataFrame,
        var exprs: List[Expr],
        select: Bool,
        var keys: List[String],
    ) raises:
        self.input = input^
        self.exprs = exprs^
        self.select = select
        self.keys = keys^
        self.stage = _START
        self.height = 0
        self.bits = _FIRST_BITS
        self.bounds = List[Int]()
        self.key_columns = List[Series]()
        self.reads = List[Series]()
        self.expanded = List[Expr]()
        self.evaluated = List[Expr]()
        self.place = List[Int]()
        self.rows = List[Int]()
        self.starts = List[List[Int]]()
        self.order = List[Int]()
        self.sizes = List[Int]()
        self.base = List[Int]()
        self.bin_rows = List[List[Int]]()
        self.results = List[List[Series]]()
        self.next = Atomic[Int64](0)
        self.outputs = List[Series]()
        self.scatter = List[Bool]()
        self.payloads = List[Int]()
        self.valid = List[List[UInt8]]()
        self.bits_out = List[List[UInt8]]()
        self.inverse = List[Int]()
        self.combined = List[Series]()
        self.pieces = List[List[Series]]()
        self.result = DataFrame(List[Series]())

    def next_round(mut self) raises -> Int:
        if self.stage == _START:
            return self._start()
        if self.stage == _SINK:
            return self._start_merge()
        if self.stage == _MERGE:
            return self._start_source()
        if self.stage == _SOURCE:
            var tasks = self._start_pack()
            if tasks > 0:
                return tasks
        if self.stage != _DONE:
            self._finish()
        return 0

    def round_task(mut self, task: Int) raises:
        if self.stage == _SINK:
            self._sink(task)
        elif self.stage == _MERGE:
            self._merge()
        elif self.stage == _SOURCE:
            self._source()
        elif self.stage == _PACK:
            self._pack(task)

    def _eager(self) raises -> DataFrame:
        if self.select:
            return self.input.select_exprs(self.exprs)
        return self.input.with_columns(self.exprs)

    def _start(mut self) raises -> Int:
        self.height = self.input.height()
        ref columns = self.input._columns
        for key in self.keys:
            var at = -1
            for c in range(len(columns)):
                if columns[c].name() == key:
                    at = c
            if at < 0:
                # Not a column of the input: the projection's own error.
                self.result = self._eager()
                self.stage = _DONE
                return 0
            self.key_columns.append(columns[at].copy())
        if self.height == 0:
            self.result = self._eager()
            self.stage = _DONE
            return 0
        self.expanded = expand_all(self.exprs, columns)
        var names = List[String]()
        var used = List[Bool](length=len(columns), fill=False)
        for expr in self.expanded:
            if expr._name in names:
                raise Error("Duplicate expression output name: " + expr._name)
            names.append(expr._name)
            if (
                len(expr._nodes) == 1
                and expr._nodes[0].op == COL
                and _column_at(self.input._columns, expr._nodes[0].text) >= 0
            ):
                self.place.append(-1)
                continue
            self.place.append(len(self.evaluated))
            self.evaluated.append(expr.copy())
            var reads = _used_columns(expr, columns)
            for c in range(len(columns)):
                used[c] = used[c] or reads[c]
        for c in range(len(columns)):
            if used[c]:
                self.reads.append(columns[c].copy())
        while self.bits < _MOST_BITS and (self.height >> self.bits) > _BIN_ROWS:
            self.bits = min(self.bits + 2, _MOST_BITS)
        var blocks = (
            self.height + STANDARD_VECTOR_SIZE - 1
        ) // STANDARD_VECTOR_SIZE
        var tasks = max(1, min(configured_workers(), blocks))
        self.bounds = partitions(self.height, tasks, 1)
        self.rows = huge_uninit[Int](self.height)
        self.starts = List[List[Int]](length=tasks, fill=List[Int]())
        self.stage = _SINK
        return tasks

    def _sink(mut self, task: Int) raises:
        var lo = self.bounds[task]
        var hi = self.bounds[task + 1]
        var n = hi - lo
        if n <= 0:
            self.starts[task] = List[Int](length=(1 << self.bits) + 1, fill=lo)
            return
        var hashes = List[UInt64](length=n, fill=0)
        var out = Int(hashes.unsafe_ptr())
        for k in range(len(self.key_columns)):
            ref key = self.key_columns[k]
            if key.is_chunked():
                var piece = key.slice(lo, n).rechunk()
                _hash_column(piece, 0, n, out, k == 0)
            else:
                _hash_column(key, lo, hi, out, k == 0, lo)
        var bins = 1 << self.bits
        var shift = UInt64(64 - self.bits)
        var at = List[Int](length=bins + 1, fill=0)
        var counts = at.unsafe_ptr()
        for i in range(n):
            counts[unsafe_offset=Int(hashes[i] >> shift) + 1] += 1
        counts[unsafe_offset=0] = lo
        for b in range(bins):
            counts[unsafe_offset=b + 1] += counts[unsafe_offset=b]
        self.starts[task] = at.copy()
        var rows = self.rows.unsafe_ptr()
        for i in range(n):
            var b = Int(hashes[i] >> shift)
            rows[unsafe_offset=counts[unsafe_offset=b]] = lo + i
            counts[unsafe_offset=b] += 1

    def _start_merge(mut self) raises -> Int:
        var bins = 1 << self.bits
        self.sizes = List[Int](length=bins, fill=0)
        for t in range(len(self.starts)):
            ref at = self.starts[t]
            for b in range(bins):
                self.sizes[b] += at[b + 1] - at[b]
        # Largest bins first, so the last bins taken are the smallest.
        for b in range(bins):
            if self.sizes[b] == 0:
                continue
            var i = len(self.order)
            self.order.append(b)
            while i > 0 and self.sizes[self.order[i - 1]] < self.sizes[b]:
                self.order[i] = self.order[i - 1]
                i -= 1
            self.order[i] = b
        self.base = List[Int](length=bins, fill=0)
        var at = 0
        for b in self.order:
            self.base[b] = at
            at += self.sizes[b]
        self.bin_rows = List[List[Int]](length=bins, fill=List[Int]())
        self.results = List[List[Series]](length=bins, fill=List[Series]())
        self.next.store(0)
        self.stage = _MERGE
        return min(len(self.order), configured_workers())

    def _merge(mut self) raises:
        while True:
            var i = Int(self.next.fetch_add(1))
            if i >= len(self.order):
                return
            var b = self.order[i]
            var rows = List[Int](capacity=self.sizes[b])
            var all_rows = self.rows.unsafe_ptr()
            for t in range(len(self.starts)):
                for r in range(self.starts[t][b], self.starts[t][b + 1]):
                    rows.append(all_rows[unsafe_offset=r])
            var part = List[Series](capacity=len(self.reads))
            for column in self.reads:
                part.append(column.take(rows))
            var frame = DataFrame(part^, height=len(rows))
            var out = frame.select_exprs(self.evaluated)
            var columns = List[Series](capacity=len(out._columns))
            for column in out._columns:
                columns.append(
                    column.rechunk() if column.is_chunked() else column.copy()
                )
            self.results[b] = columns^
            self.bin_rows[b] = rows^

    def _start_source(mut self) raises -> Int:
        var first = self.order[0]
        var inverse = False
        for k in range(len(self.evaluated)):
            ref sample = self.results[first][k]
            var fixed = _fixed_width(sample)
            self.scatter.append(fixed)
            var nulls = False
            for b in self.order:
                nulls = nulls or self.results[b][k].null_count() > 0
            if fixed:
                var output = _allocate(sample, self.height)
                self.payloads.append(_payload_address(output))
                self.outputs.append(output^)
            else:
                inverse = True
                self.payloads.append(0)
                self.outputs.append(sample.copy())
            self.valid.append(
                List[UInt8](length=self.height, fill=0) if fixed
                and nulls else List[UInt8]()
            )
        if inverse:
            self.inverse = huge_uninit[Int](self.height)
        self.next.store(0)
        self.stage = _SOURCE
        return min(len(self.order), configured_workers())

    def _source(mut self) raises:
        while True:
            var i = Int(self.next.fetch_add(1))
            if i >= len(self.order):
                return
            var b = self.order[i]
            for k in range(len(self.evaluated)):
                var dtype = self.outputs[k].dtype()
                if self.results[b][k].dtype() != dtype:
                    self.results[b][k] = self.results[b][k].cast(dtype)
                if self.scatter[k]:
                    var valid = 0
                    if len(self.valid[k]) > 0:
                        valid = Int(self.valid[k].unsafe_ptr())
                    _write_rows(
                        self.results[b][k],
                        self.bin_rows[b],
                        self.payloads[k],
                        valid,
                    )
            if len(self.inverse) > 0:
                var inverse = self.inverse.unsafe_ptr()
                ref rows = self.bin_rows[b]
                var base = self.base[b]
                for r in range(len(rows)):
                    inverse[unsafe_offset=rows[r]] = base + r

    def _start_pack(mut self) raises -> Int:
        var work = len(self.inverse) > 0
        for k in range(len(self.evaluated)):
            work = work or len(self.valid[k]) > 0
        if not work:
            return 0
        for k in range(len(self.evaluated)):
            self.bits_out.append(
                List[UInt8](length=(self.height + 7) // 8, fill=0) if len(
                    self.valid[k]
                )
                > 0 else List[UInt8]()
            )
            if self.scatter[k]:
                self.combined.append(self.outputs[k].copy())
            else:
                var parts = List[Series](capacity=len(self.order))
                for b in self.order:
                    parts.append(self.results[b][k].copy())
                self.combined.append(
                    parts[0].copy() if len(parts)
                    == 1 else Series._from_chunks(parts)
                )
        var tasks = len(self.bounds) - 1
        self.bounds = partitions(self.height, tasks, 8)
        self.pieces = List[List[Series]](length=tasks, fill=List[Series]())
        self.stage = _PACK
        return tasks

    def _pack(mut self, task: Int) raises:
        var lo = self.bounds[task]
        var hi = self.bounds[task + 1]
        for k in range(len(self.evaluated)):
            if len(self.valid[k]) > 0:
                var bytes = self.valid[k].unsafe_ptr()
                var bits = self.bits_out[k].unsafe_ptr()
                for r in range(lo, hi):
                    if bytes[unsafe_offset=r] != 0:
                        bits[unsafe_offset=r // 8] |= UInt8(1) << UInt8(r % 8)
            if not self.scatter[k] and hi > lo:
                var rows = List[Int](capacity=hi - lo)
                var inverse = self.inverse.unsafe_ptr()
                for r in range(lo, hi):
                    rows.append(inverse[unsafe_offset=r])
                ref whole = self.combined[k]
                self.pieces[task].append(
                    whole._take_from_chunks(
                        rows
                    ) if whole.is_chunked() else whole.take(rows)
                )

    def _finish(mut self) raises:
        if self.stage == _START:
            return
        for k in range(len(self.evaluated)):
            if self.scatter[k]:
                if len(self.valid[k]) > 0:
                    var bits = List[UInt8]()
                    swap(bits, self.bits_out[k])
                    _set_bits(self.outputs[k], bits^)
                    self.valid[k] = List[UInt8]()
            else:
                # Each pack task gathered the columns not written by row
                # in order; this column's piece is the nth of them.
                var parts = List[Series]()
                var nth = 0
                for j in range(k):
                    if not self.scatter[j]:
                        nth += 1
                for t in range(len(self.pieces)):
                    if nth < len(self.pieces[t]):
                        parts.append(self.pieces[t][nth].copy())
                self.outputs[k] = parts[0].copy() if len(
                    parts
                ) == 1 else Series._from_chunks(parts)
        release_huge(self.rows^)
        self.rows = List[Int]()
        if len(self.inverse) > 0:
            release_huge(self.inverse^)
            self.inverse = List[Int]()
        self.bin_rows = List[List[Int]]()
        self.results = List[List[Series]]()
        self.combined = List[Series]()
        self.pieces = List[List[Series]]()
        var columns = List[Series]()
        if not self.select:
            columns = self.input._columns.copy()
        for e in range(len(self.expanded)):
            ref expr = self.expanded[e]
            var value: Series
            if self.place[e] < 0:
                value = self.input._columns[
                    _column_at(self.input._columns, expr._nodes[0].text)
                ].renamed(expr._name)
            else:
                value = self.outputs[self.place[e]].copy()
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
        self.outputs = List[Series]()
        self.result = DataFrame(columns^, height=self.height)
        self.stage = _DONE

    def take_result(mut self) raises -> DataFrame:
        """The projection's rows, moved out once no round is left."""
        var taken = DataFrame(List[Series]())
        swap(taken, self.result)
        return taken^
