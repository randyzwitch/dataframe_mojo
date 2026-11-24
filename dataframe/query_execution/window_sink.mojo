"""A projection holding window functions, run as rounds of tasks of the
query's schedule rather than as one task that starts its kernels' jobs.

Every window of such a projection partitions its rows by named keys. The
keys all of them share split the rows into groups that no partition of any
window crosses: two rows with different shared keys are never in one
partition. So the projection gives the same rows when it runs on each such
group apart from the others, and the groups can run at the same time, each
on one thread, with no shared state between them.

The work is three rounds:

1. Sink: each task takes one range of rows, hashes their shared keys,
   counts the range's rows in each bin (the top bits of the hash), and
   copies the range's values of the columns the projection reads into its
   own stretch of one array per column, ordered by bin. The range is read
   in order and each value moved once: a bin's rows gathered later from
   all over each column would read every part of it once per bin.
2. Merge: each task takes whole bins, largest first, until none is left;
   it joins a bin's stretches from every sink task into one column each
   and evaluates the projection over them. A bin is one hash partition of
   the shared keys, so the windows over exactly those keys take their
   partitions from it: the bin's keys are numbered into groups once (by
   value for one integer key, else probing with the hashes the sink kept),
   with no second hash pass or scatter. Everything the evaluation
   starts runs on the task's own thread, since the other tasks keep the
   other threads busy.
3. Gather: each task fills the rows of its sink range in every output
   column, each row's value read from its bin's results.

A bin's rows keep their input order, so every window sees each partition's
rows in the same order as over the whole input, and the results equal the
whole-input evaluation's.
"""
from std.atomic import Atomic
from std.memory import ArcPointer, Pointer, unsafe_memcpy

from dataframe.aggregation.expressions import _bind_all
from dataframe.binding import ROWS
from dataframe.execution import WindowLayouts, _used_columns, evaluate
from dataframe.expr import (
    Expr,
    COL,
    OVER,
    split_names,
)
from dataframe.frame import DataFrame
from dataframe.kernels.gather import (
    GATHER_DTYPES,
    _allocate,
    _payload_address,
    _set_bits,
)
from dataframe.kernels.group_rows import wide_row_indices
from dataframe.kernels.partition import _hash_column
from dataframe.parallel import Rounds, configured_workers, partitions
from dataframe.selectors import expand_all
from dataframe.series import Series
from dataframe.sorting.sort_key import STANDARD_VECTOR_SIZE
from dataframe.storage.allocation import huge_uninit, release_huge
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.window_sort import _integer_key, bin_partition_groups

# The round that ran last.
comptime _START = 0
comptime _SINK = 1
comptime _MERGE = 2
comptime _GATHER = 3
comptime _DONE = 4

# Bins are the top bits of the key hash. There are at least 16, so the
# threads have bins to pick from when partitions differ in size, and at
# most 256, so a bin fits in a byte a row and a task's per-bin cursors
# stay small. Between those, the bits grow two at a time while a bin
# would average more rows than one row group (122,880 rows): a bin's rows
# are evaluated by one thread, and bins of bounded size keep any one of
# them from holding the last thread long after the others finish.
comptime _FIRST_BITS = 4
comptime _MOST_BITS = 8
comptime _BIN_ROWS = 122880


def _column_at(columns: List[Series], name: String) -> Int:
    """The place of the column named `name`, or -1."""
    for c in range(len(columns)):
        if columns[c].name() == name:
            return c
    return -1


def _fixed_width(series: Series) -> Bool:
    """Whether the column's values are one fixed-width word per row that a
    task can read or write at any row."""
    if series.dtype().is_categorical():
        return False
    comptime for d in range(len(GATHER_DTYPES)):
        comptime D = GATHER_DTYPES[d]
        if series._data.isa[Column[Scalar[D]]]():
            return True
    return False


def _contiguous(series: Series, lo: Int, hi: Int) raises -> Series:
    """Rows [lo, hi) of `series` in one chunk; a column already in one
    chunk is returned whole (read at its own row numbers)."""
    if series.is_chunked():
        return series.slice(lo, hi - lo).rechunk()
    return series.copy()


def _place_values(
    source: Series,
    shift: Int,
    lo: Int,
    hi: Int,
    bins: Int,
    inverse: Int,
    starts: Int,
    target: Int,
    valid: Int,
) raises:
    """Rows [lo, hi) of `source` (row r at r - shift) to their places in
    the sink task's stretch: starts[bin of r] + inverse[r]. The validity
    of each, when `valid` holds an address, to the byte at the same
    place."""
    var bin_of = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bins)
    var within = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=inverse)
    var first = Pointer[Int, MutAnyOrigin](unsafe_from_address=starts)
    comptime for d in range(len(GATHER_DTYPES)):
        comptime D = GATHER_DTYPES[d]
        if source._data.isa[Column[Scalar[D]]]():
            ref column = source._data[Column[Scalar[D]]]
            var values = column._ptr()
            var out = Pointer[Scalar[D], MutAnyOrigin](
                unsafe_from_address=target
            )
            for r in range(lo, hi):
                var at = first[
                    unsafe_offset=Int(bin_of[unsafe_offset=r])
                ] + Int(within[unsafe_offset=r])
                out[unsafe_offset=at] = values[unsafe_offset=r - shift]
            if valid != 0:
                var bytes = Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=valid
                )
                var nulls = len(column._bits) > 0
                for r in range(lo, hi):
                    var at = first[
                        unsafe_offset=Int(bin_of[unsafe_offset=r])
                    ] + Int(within[unsafe_offset=r])
                    bytes[unsafe_offset=at] = UInt8(
                        1
                    ) if not nulls or column._valid(r - shift) else UInt8(0)
            return
    raise Error("window input column has no fixed-width storage")


def _join_stretches(
    placed: Series, valid: Int, starts: List[Int], counts: List[Int]
) raises -> Series:
    """One bin's values of a column the sink tasks placed: each task's
    stretch (counts[t] values from starts[t]) in task order, as a column
    of its own, with validity from the bytes at `valid` when it holds an
    address."""
    var rows = 0
    for count in counts:
        rows += count
    comptime for d in range(len(GATHER_DTYPES)):
        comptime D = GATHER_DTYPES[d]
        if placed._data.isa[Column[Scalar[D]]]():
            var source = placed._data[Column[Scalar[D]]]._ptr()
            var values = List[Scalar[D]](unsafe_uninit_length=rows)
            var to = values.unsafe_ptr()
            var at = 0
            for t in range(len(starts)):
                if counts[t] > 0:
                    unsafe_memcpy(
                        dest=to.unsafe_offset(at),
                        src=source.unsafe_offset(starts[t]),
                        count=counts[t],
                    )
                at += counts[t]
            var bits = List[UInt8]()
            if valid != 0:
                var bytes = Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=valid
                )
                var nulls = False
                var packed = List[UInt8](length=(rows + 7) // 8, fill=0)
                var i = 0
                for t in range(len(starts)):
                    for k in range(starts[t], starts[t] + counts[t]):
                        if bytes[unsafe_offset=k] != 0:
                            packed[i >> 3] |= UInt8(1) << UInt8(i & 7)
                        else:
                            nulls = True
                        i += 1
                if nulls:
                    bits = packed^
            var piece = Series(
                placed.name(), Column[Scalar[D]](values=values^, bits=bits^)
            )
            piece._dtype = placed.dtype()
            piece._field_metadata = placed._field_metadata
            return piece^
    raise Error("window input column has no fixed-width storage")


def _value_parts(series: Series) -> Tuple[Int, Int, Int]:
    """A fixed-width column's payload address (row 0 of its window), its
    validity bits' address (0 when every row is valid) and the bit of its
    row 0."""
    comptime for d in range(len(GATHER_DTYPES)):
        comptime D = GATHER_DTYPES[d]
        if series._data.isa[Column[Scalar[D]]]():
            ref column = series._data[Column[Scalar[D]]]
            var bits = 0
            if len(column._bits) > 0:
                bits = Int(column._bits.unsafe_ptr())
            return (Int(column._ptr()), bits, column._offset)
    return (0, 0, 0)


def _gather_values(
    output: Series,
    lo: Int,
    hi: Int,
    bins: Int,
    inverse: Int,
    before: Int,
    values: List[Int],
    valid: List[Int],
    first_bit: List[Int],
    out_bits: Int,
) raises:
    """Output rows [lo, hi) of one sink task: row r's value is place
    before[bin] + inverse[r] of its bin's results (payload at
    values[bin]); its validity bit, when `out_bits` holds an address, from
    that bin's bits (valid[bin], 0 for none)."""
    var bin_of = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bins)
    var within = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=inverse)
    var skip = Pointer[Int, MutAnyOrigin](unsafe_from_address=before)
    var table = values.unsafe_ptr()
    comptime for d in range(len(GATHER_DTYPES)):
        comptime D = GATHER_DTYPES[d]
        if output._data.isa[Column[Scalar[D]]]():
            var out = Pointer[Scalar[D], MutAnyOrigin](
                unsafe_from_address=_payload_address(output)
            )
            for r in range(lo, hi):
                var b = Int(bin_of[unsafe_offset=r])
                out[unsafe_offset=r] = Pointer[Scalar[D], MutAnyOrigin](
                    unsafe_from_address=table[unsafe_offset=b]
                )[
                    unsafe_offset=skip[unsafe_offset=b]
                    + Int(within[unsafe_offset=r])
                ]
            if out_bits != 0:
                var bits = Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=out_bits
                )
                for r in range(lo, hi):
                    var b = Int(bin_of[unsafe_offset=r])
                    var ok = True
                    if valid[b] != 0:
                        var at = (
                            first_bit[b]
                            + skip[unsafe_offset=b]
                            + Int(within[unsafe_offset=r])
                        )
                        ok = (
                            Pointer[UInt8, MutAnyOrigin](
                                unsafe_from_address=valid[b]
                            )[unsafe_offset=at // 8]
                            >> UInt8(at % 8)
                        ) & 1 == 1
                    if ok:
                        bits[unsafe_offset=r // 8] |= UInt8(1) << UInt8(r % 8)
            return
    raise Error("window result column has no fixed-width storage")


def _boolean(series: Series) -> Bool:
    """Whether the column's values are bits, one a row, that a task can
    read at any row (a column of one repeated value aside)."""
    return series._data.isa[BoolColumn]() and not (
        series._data[BoolColumn].is_constant()
    )


def _gather_bits(
    output: Int,
    lo: Int,
    hi: Int,
    bins: Int,
    inverse: Int,
    before: Int,
    values: List[Int],
    valid: List[Int],
    first_bit: List[Int],
    out_bits: Int,
):
    """Output rows [lo, hi) of one sink task for a Boolean result: row r's
    value is bit first_bit[bin] + before[bin] + inverse[r] of its bin's
    value bits (at values[bin]), set in the output bits at `output`; its
    validity, when `out_bits` holds an address, from the same bit of that
    bin's validity bits (valid[bin], 0 for none). `lo` is a multiple of 8,
    so the task writes whole bytes no other task writes."""
    var bin_of = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bins)
    var within = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=inverse)
    var skip = Pointer[Int, MutAnyOrigin](unsafe_from_address=before)
    var out = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=output)
    for r in range(lo, hi):
        var b = Int(bin_of[unsafe_offset=r])
        var at = (
            first_bit[b] + skip[unsafe_offset=b] + Int(within[unsafe_offset=r])
        )
        var bit = (
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=values[b])[
                unsafe_offset=at // 8
            ]
            >> UInt8(at % 8)
        ) & 1
        out[unsafe_offset=r // 8] |= bit << UInt8(r % 8)
    if out_bits == 0:
        return
    var bits = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=out_bits)
    for r in range(lo, hi):
        var b = Int(bin_of[unsafe_offset=r])
        var ok = True
        if valid[b] != 0:
            var at = (
                first_bit[b]
                + skip[unsafe_offset=b]
                + Int(within[unsafe_offset=r])
            )
            ok = (
                Pointer[UInt8, MutAnyOrigin](unsafe_from_address=valid[b])[
                    unsafe_offset=at // 8
                ]
                >> UInt8(at % 8)
            ) & 1 == 1
        if ok:
            bits[unsafe_offset=r // 8] |= UInt8(1) << UInt8(r % 8)


struct WindowSink(Movable, Rounds):
    """A SELECT or WITH_COLUMNS of window functions over `input`, split by
    the windows' shared partition keys (`shared_partition_keys` in
    dataframe.optimizer.expressions) and run
    as rounds (see the module docstring); `result` once no round is
    left."""

    var input: DataFrame
    var exprs: List[Expr]
    var select: Bool
    var keys: List[String]
    var stage: Int
    var height: Int
    var bits: Int
    # The sink tasks' row ranges, which the gather tasks fill again.
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
    # By row: its bin, and its place among its sink task's rows in that
    # bin (32 bits: a taller input runs whole, see `_start`).
    var bin_of: List[UInt8]
    var inverse: List[UInt32]
    # By sink task, by bin: where the task's stretch of the bin starts in
    # the placed arrays, its row count, and the rows earlier tasks put in
    # the bin.
    var starts: List[List[Int]]
    var counts: List[List[Int]]
    var before: List[List[Int]]
    # By bin: its rows; the bins holding rows, largest first.
    var sizes: List[Int]
    var order: List[Int]
    # By read column: its values as the sink tasks placed them (fixed-width
    # columns), and their validity as a byte a row (empty when every row
    # is valid); the rows as placed, when some read column is not
    # fixed-width.
    var placed: List[Series]
    var placed_fixed: List[Bool]
    var placed_valid: List[List[UInt8]]
    var placed_rows: List[Int]
    # The key hashes as the sink tasks placed them, when the keys are not
    # one integer column; by shared key, its place among the read
    # columns; the spellings of the over() partitions whose keys are
    # exactly the shared keys.
    var placed_hashes: List[UInt64]
    var key_reads: List[Int]
    var shared_texts: List[String]
    # By bin: its evaluated columns.
    var results: List[List[Series]]
    # The next bin of `order` a merge task takes.
    var next: Atomic[Int64]
    # By evaluated expression: its output, whether it is filled by row
    # from fixed-width results, then by bin those results' payloads,
    # validity bits and first bit (for the other columns, the bin's first
    # place among the chunked results); the output's validity bits (empty
    # when no result has a null); the bins' results as one chunked column
    # and each gather task's piece, for the columns not filled by row.
    var outputs: List[Series]
    var fixed: List[Bool]
    # By evaluated expression: whether it is filled by row from Boolean
    # results' bits, then its output's value bits.
    var bits_out: List[Bool]
    var out_values: List[List[UInt8]]
    var values_at: List[List[Int]]
    var valid_at: List[List[Int]]
    var first_bit: List[List[Int]]
    var out_bits: List[List[UInt8]]
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
        self.bin_of = List[UInt8]()
        self.inverse = List[UInt32]()
        self.starts = List[List[Int]]()
        self.counts = List[List[Int]]()
        self.before = List[List[Int]]()
        self.sizes = List[Int]()
        self.order = List[Int]()
        self.placed = List[Series]()
        self.placed_fixed = List[Bool]()
        self.placed_valid = List[List[UInt8]]()
        self.placed_rows = List[Int]()
        self.placed_hashes = List[UInt64]()
        self.key_reads = List[Int]()
        self.shared_texts = List[String]()
        self.results = List[List[Series]]()
        self.next = Atomic[Int64](0)
        self.outputs = List[Series]()
        self.fixed = List[Bool]()
        self.bits_out = List[Bool]()
        self.out_values = List[List[UInt8]]()
        self.values_at = List[List[Int]]()
        self.valid_at = List[List[Int]]()
        self.first_bit = List[List[Int]]()
        self.out_bits = List[List[UInt8]]()
        self.combined = List[Series]()
        self.pieces = List[List[Series]]()
        self.result = DataFrame(List[Series]())

    def next_round(mut self) raises -> Int:
        if self.stage == _START:
            return self._start()
        if self.stage == _SINK:
            return self._start_merge()
        if self.stage == _MERGE:
            return self._start_gather()
        if self.stage == _GATHER:
            self._finish()
        return 0

    def round_task(mut self, task: Int) raises:
        if self.stage == _SINK:
            self._sink(task)
        elif self.stage == _MERGE:
            self._merge()
        elif self.stage == _GATHER:
            self._gather(task)

    def _eager(self) raises -> DataFrame:
        if self.select:
            return self.input._select_exprs(self.exprs)
        return self.input._with_columns(self.exprs)

    def _start(mut self) raises -> Int:
        self.height = self.input.height()
        ref columns = self.input._columns
        for key in self.keys:
            var at = _column_at(columns, key)
            if at < 0:
                # Not a column of the input: the projection's own error.
                self.result = self._eager()
                self.stage = _DONE
                return 0
            self.key_columns.append(columns[at].copy())
        if self.height == 0 or wide_row_indices(self.height):
            # No rows, or more than a 32-bit place within a bin numbers:
            # the projection over the whole input.
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
                and _column_at(columns, expr._nodes[0].text) >= 0
            ):
                self.place.append(-1)
                continue
            self.place.append(len(self.evaluated))
            self.evaluated.append(expr.copy())
            var reads = _used_columns(expr, columns)
            for c in range(len(columns)):
                used[c] = used[c] or reads[c]
        var rows_needed = False
        for c in range(len(columns)):
            if not used[c]:
                continue
            ref column = columns[c]
            self.reads.append(column.copy())
            var sample = _contiguous(column, 0, 1)
            var fixed = _fixed_width(sample)
            self.placed_fixed.append(fixed)
            rows_needed = rows_needed or not fixed
            self.placed.append(
                _allocate(sample, self.height) if fixed else sample.copy()
            )
            # Every place gets its byte from the sink task that owns it.
            self.placed_valid.append(
                List[UInt8](unsafe_uninit_length=self.height) if fixed
                and column.null_count() > 0 else List[UInt8]()
            )
        if rows_needed:
            self.placed_rows = huge_uninit[Int](self.height)
        for key in self.keys:
            self.key_reads.append(_column_at(self.reads, key))
        for expr in self.evaluated:
            for node in expr._nodes:
                if node.op != OVER or node.text2 in self.shared_texts:
                    continue
                var names = List[String]()
                for name in split_names(node.text2):
                    if name.byte_length() > 0 and name not in names:
                        names.append(name)
                var same = len(names) == len(self.keys)
                for name in names:
                    same = same and name in self.keys
                if same:
                    self.shared_texts.append(node.text2)
        var samples = List[Series](capacity=len(self.key_columns))
        for key in self.key_columns:
            samples.append(_contiguous(key, 0, 1))
        if len(self.shared_texts) > 0 and not _integer_key(samples):
            self.placed_hashes = huge_uninit[UInt64](self.height)
        while self.bits < _MOST_BITS and (self.height >> self.bits) > _BIN_ROWS:
            self.bits = min(self.bits + 2, _MOST_BITS)
        var blocks = (
            self.height + STANDARD_VECTOR_SIZE - 1
        ) // STANDARD_VECTOR_SIZE
        var tasks = max(1, min(configured_workers(), blocks))
        # Ranges start at multiples of 8 rows, so the gather tasks that
        # fill them again write disjoint bytes of validity bits.
        self.bounds = partitions(self.height, tasks, 8)
        self.bin_of = huge_uninit[UInt8](self.height)
        self.inverse = huge_uninit[UInt32](self.height)
        self.starts = List[List[Int]](length=tasks, fill=List[Int]())
        self.counts = List[List[Int]](length=tasks, fill=List[Int]())
        self.stage = _SINK
        return tasks

    def _sink(mut self, task: Int) raises:
        var lo = self.bounds[task]
        var hi = self.bounds[task + 1]
        var n = hi - lo
        var bins = 1 << self.bits
        var counts = List[Int](length=bins, fill=0)
        var starts = List[Int](length=bins, fill=lo)
        if n <= 0:
            self.counts[task] = counts^
            self.starts[task] = starts^
            return
        # Every row is hashed: the first key column writes each.
        var hashes = List[UInt64](unsafe_uninit_length=n)
        var out = Int(hashes.unsafe_ptr())
        for k in range(len(self.key_columns)):
            ref key = self.key_columns[k]
            if key.is_chunked():
                var piece = key.slice(lo, n).rechunk()
                _hash_column(piece, 0, n, out, k == 0)
            else:
                _hash_column(key, lo, hi, out, k == 0, lo)
        var shift = UInt64(64 - self.bits)
        var hash = hashes.unsafe_ptr()
        var bin_of = self.bin_of.unsafe_ptr()
        var inverse = self.inverse.unsafe_ptr()
        var count = counts.unsafe_ptr()
        for i in range(n):
            var b = Int(hash[unsafe_offset=i] >> shift)
            bin_of[unsafe_offset=lo + i] = UInt8(b)
            inverse[unsafe_offset=lo + i] = UInt32(count[unsafe_offset=b])
            count[unsafe_offset=b] += 1
        var at = lo
        for b in range(bins):
            starts[b] = at
            at += counts[b]
        if len(self.placed_hashes) > 0:
            var placed = self.placed_hashes.unsafe_ptr()
            for i in range(n):
                var r = lo + i
                placed[
                    unsafe_offset=starts[Int(bin_of[unsafe_offset=r])]
                    + Int(inverse[unsafe_offset=r])
                ] = hash[unsafe_offset=i]
        if len(self.placed_rows) > 0:
            var rows = self.placed_rows.unsafe_ptr()
            for r in range(lo, hi):
                rows[
                    unsafe_offset=starts[Int(bin_of[unsafe_offset=r])]
                    + Int(inverse[unsafe_offset=r])
                ] = r
        for c in range(len(self.reads)):
            if not self.placed_fixed[c]:
                continue
            var source = _contiguous(self.reads[c], lo, hi)
            var valid = 0
            if len(self.placed_valid[c]) > 0:
                valid = Int(self.placed_valid[c].unsafe_ptr())
            _place_values(
                source,
                lo if self.reads[c].is_chunked() else 0,
                lo,
                hi,
                Int(bin_of),
                Int(inverse),
                Int(starts.unsafe_ptr()),
                _payload_address(self.placed[c]),
                valid,
            )
        self.counts[task] = counts^
        self.starts[task] = starts^

    def _start_merge(mut self) raises -> Int:
        var bins = 1 << self.bits
        var tasks = len(self.counts)
        self.sizes = List[Int](length=bins, fill=0)
        self.before = List[List[Int]](
            length=tasks, fill=List[Int](length=bins, fill=0)
        )
        for t in range(tasks):
            for b in range(bins):
                self.before[t][b] = self.sizes[b]
                self.sizes[b] += self.counts[t][b]
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
        self.results = List[List[Series]](length=bins, fill=List[Series]())
        self.next.store(0)
        self.stage = _MERGE
        return min(len(self.order), configured_workers())

    def _merge(mut self) raises:
        var tasks = len(self.counts)
        while True:
            var i = Int(self.next.fetch_add(1))
            if i >= len(self.order):
                return
            var b = self.order[i]
            var starts = List[Int](capacity=tasks)
            var counts = List[Int](capacity=tasks)
            for t in range(tasks):
                starts.append(self.starts[t][b])
                counts.append(self.counts[t][b])
            var part = List[Series](capacity=len(self.reads))
            for c in range(len(self.reads)):
                if self.placed_fixed[c]:
                    var valid = 0
                    if len(self.placed_valid[c]) > 0:
                        valid = Int(self.placed_valid[c].unsafe_ptr())
                    part.append(
                        _join_stretches(self.placed[c], valid, starts, counts)
                    )
                else:
                    var picks = List[Int](capacity=self.sizes[b])
                    for t in range(tasks):
                        for k in range(starts[t], starts[t] + counts[t]):
                            picks.append(self.placed_rows[k])
                    part.append(self.reads[c].take(picks))
            var rows = self.sizes[b]
            var layouts = WindowLayouts()
            if len(self.shared_texts) > 0:
                var keys = List[Series](capacity=len(self.key_reads))
                for at in self.key_reads:
                    keys.append(part[at].copy())
                var hashes = List[UInt64]()
                if len(self.placed_hashes) > 0:
                    hashes = List[UInt64](unsafe_uninit_length=rows)
                    var to = hashes.unsafe_ptr()
                    var source = self.placed_hashes.unsafe_ptr()
                    var at = 0
                    for t in range(tasks):
                        if counts[t] > 0:
                            unsafe_memcpy(
                                dest=to.unsafe_offset(at),
                                src=source.unsafe_offset(starts[t]),
                                count=counts[t],
                            )
                        at += counts[t]
                var groups = ArcPointer(
                    bin_partition_groups(keys, Int(hashes.unsafe_ptr()), rows)
                )
                _ = hashes^
                for text in self.shared_texts:
                    layouts.id_keys.append(text)
                    layouts.groups.append(groups)
                    layouts.ids.append(List[Int]())
            var bound = _bind_all(self.evaluated, part)
            var columns = List[Series](capacity=len(bound))
            for expression in bound:
                var value = evaluate(
                    expression,
                    part,
                    rows,
                    batch_size=8192,
                    layouts=Int(Pointer(to=layouts)),
                )
                if expression.shape() != ROWS:
                    value = value._broadcast(rows)
                columns.append(
                    value.rechunk() if value.is_chunked() else value^
                )
            _ = layouts^
            self.results[b] = columns^

    def _start_gather(mut self) raises -> Int:
        # The placed columns are no longer read.
        self.placed = List[Series]()
        if len(self.placed_hashes) > 0:
            release_huge(self.placed_hashes^)
            self.placed_hashes = List[UInt64]()
        self.placed_valid = List[List[UInt8]]()
        var bins = 1 << self.bits
        # Each bin's first place among all rows in the order its results
        # are chunked (`order`), for the columns gathered from chunks.
        var chunk_base = List[Int](length=bins, fill=0)
        var at = 0
        for b in self.order:
            chunk_base[b] = at
            at += self.sizes[b]
        var first = self.order[0]
        for k in range(len(self.evaluated)):
            var dtype = self.results[first][k].dtype()
            var nulls = False
            for b in self.order:
                if self.results[b][k].dtype() != dtype:
                    self.results[b][k] = self.results[b][k].cast(dtype)
                nulls = nulls or self.results[b][k].null_count() > 0
            ref sample = self.results[first][k]
            var fixed = _fixed_width(sample)
            var boolean = not fixed
            for b in self.order:
                boolean = boolean and _boolean(self.results[b][k])
            self.fixed.append(fixed)
            self.bits_out.append(boolean)
            self.out_values.append(
                List[UInt8](
                    length=(self.height + 7) // 8, fill=0
                ) if boolean else List[UInt8]()
            )
            var values = List[Int](length=bins, fill=0)
            var valid = List[Int](length=bins, fill=0)
            var first_bit = List[Int](length=bins, fill=0)
            if boolean:
                for b in self.order:
                    ref column = self.results[b][k]._data[BoolColumn]
                    values[b] = Int(column.unsafe_values())
                    if column.null_count() > 0:
                        valid[b] = Int(column.unsafe_validity())
                    first_bit[b] = column.validity_offset()
                self.outputs.append(sample.copy())
                self.combined.append(sample.copy())
            elif fixed:
                for b in self.order:
                    var parts = _value_parts(self.results[b][k])
                    values[b] = parts[0]
                    valid[b] = parts[1]
                    first_bit[b] = parts[2]
                self.outputs.append(_allocate(sample, self.height))
                self.combined.append(sample.copy())
            else:
                self.outputs.append(sample.copy())
                var chunks = List[Series](capacity=len(self.order))
                for b in self.order:
                    chunks.append(self.results[b][k].copy())
                self.combined.append(
                    chunks[0].copy() if len(chunks)
                    == 1 else Series._from_chunks(chunks)
                )
                for b in self.order:
                    values[b] = chunk_base[b]
            self.values_at.append(values^)
            self.valid_at.append(valid^)
            self.first_bit.append(first_bit^)
            self.out_bits.append(
                List[UInt8](length=(self.height + 7) // 8, fill=0) if (
                    fixed or boolean
                )
                and nulls else List[UInt8]()
            )
        var tasks = len(self.bounds) - 1
        self.pieces = List[List[Series]](length=tasks, fill=List[Series]())
        self.stage = _GATHER
        return tasks

    def _gather(mut self, task: Int) raises:
        var lo = self.bounds[task]
        var hi = self.bounds[task + 1]
        if hi <= lo:
            return
        var bin_of = self.bin_of.unsafe_ptr()
        var inverse = self.inverse.unsafe_ptr()
        var before = self.before[task].unsafe_ptr()
        for k in range(len(self.evaluated)):
            if self.fixed[k]:
                var bits = 0
                if len(self.out_bits[k]) > 0:
                    bits = Int(self.out_bits[k].unsafe_ptr())
                _gather_values(
                    self.outputs[k],
                    lo,
                    hi,
                    Int(bin_of),
                    Int(inverse),
                    Int(before),
                    self.values_at[k],
                    self.valid_at[k],
                    self.first_bit[k],
                    bits,
                )
                continue
            if self.bits_out[k]:
                var bits = 0
                if len(self.out_bits[k]) > 0:
                    bits = Int(self.out_bits[k].unsafe_ptr())
                _gather_bits(
                    Int(self.out_values[k].unsafe_ptr()),
                    lo,
                    hi,
                    Int(bin_of),
                    Int(inverse),
                    Int(before),
                    self.values_at[k],
                    self.valid_at[k],
                    self.first_bit[k],
                    bits,
                )
                continue
            # The places, among the chunked results, of the range's rows.
            var picks = List[Int](capacity=hi - lo)
            for r in range(lo, hi):
                var b = Int(bin_of[unsafe_offset=r])
                picks.append(
                    self.values_at[k][b]
                    + before[unsafe_offset=b]
                    + Int(inverse[unsafe_offset=r])
                )
            ref whole = self.combined[k]
            self.pieces[task].append(
                whole._take_from_chunks(
                    picks
                ) if whole.is_chunked() else whole.take(picks)
            )

    def _finish(mut self) raises:
        var nth = 0
        for k in range(len(self.evaluated)):
            if self.fixed[k]:
                if len(self.out_bits[k]) > 0:
                    var bits = List[UInt8]()
                    swap(bits, self.out_bits[k])
                    _set_bits(self.outputs[k], bits^)
                continue
            if self.bits_out[k]:
                var values = List[UInt8]()
                swap(values, self.out_values[k])
                var bits = List[UInt8]()
                swap(bits, self.out_bits[k])
                var output = Series(
                    self.outputs[k].name(),
                    BoolColumn(values=values^, bits=bits^, length=self.height),
                )
                output._field_metadata = self.outputs[k]._field_metadata
                self.outputs[k] = output^
                continue
            # Each gather task made one piece per column not filled by
            # row, in column order.
            var parts = List[Series]()
            for t in range(len(self.pieces)):
                if nth < len(self.pieces[t]):
                    parts.append(self.pieces[t][nth].copy())
            nth += 1
            self.outputs[k] = parts[0].copy() if len(
                parts
            ) == 1 else Series._from_chunks(parts)
        release_huge(self.bin_of^)
        self.bin_of = List[UInt8]()
        release_huge(self.inverse^)
        self.inverse = List[UInt32]()
        if len(self.placed_rows) > 0:
            release_huge(self.placed_rows^)
            self.placed_rows = List[Int]()
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
