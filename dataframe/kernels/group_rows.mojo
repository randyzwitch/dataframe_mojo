"""Each group's row list, and writing one value per group to its rows.

`GroupRows` holds, for every group, the input rows it contains in
ascending order. With the rows of each group at hand, a grouped reduction
can split its work by group: each thread owns whole groups, so it keeps
state only for them and no partial states are merged afterwards, however
many groups there are. The same lists let a per-group result be written
straight to the group's rows, with no per-row gather of the results. The
lists are kept as one array cut by offsets rather than one list per
group.
"""
from std.memory import Pointer

from dataframe.dtype import NUMERIC_DTYPES
from dataframe.parallel import Job, run_jobs
from dataframe.series import Series
from dataframe.storage.allocation import huge_uninit
from dataframe.storage.column import Column


comptime ROW_INDEX = DType.uint32
"""Row indices in group lists when the input's rows all fit 32 bits."""
comptime WIDE_ROW_INDEX = DType.uint64
"""Row indices in group lists for inputs of more rows than 32 bits
number."""


def wide_row_indices(height: Int) -> Bool:
    """Whether `height` rows need 64-bit row indices: a 32-bit index
    names rows 0 to 2^32 - 1, and a group list's end offset can equal the
    row count, so the narrow type holds every index only while the count
    itself fits in 32 bits. The choice follows the input's actual row
    count, not a size picked for speed."""
    return UInt64(height) > UInt64(UInt32.MAX)


struct GroupRows(Copyable, Movable):
    """Group g's rows are `rows[offsets[g]:offsets[g + 1]]`, ascending.

    Offsets and rows are stored as 32-bit indices, or as 64-bit ones
    (`wide`) when the input has more rows than 32 bits can number. The
    lists are read through every grouped reduction and every write back to
    rows, so their width is most of the memory traffic of those passes:
    the narrow form halves it. Kernels read them through `offsets_address`
    and `rows_address`, typed by `index_dtype` as a compile-time
    parameter, so neither width costs a branch per row."""

    var wide: Bool
    var offsets: List[UInt32]
    var rows: List[UInt32]
    var wide_offsets: List[UInt64]
    var wide_rows: List[UInt64]

    def __init__(out self):
        self.wide = False
        self.offsets = List[UInt32](length=1, fill=0)
        self.rows = List[UInt32]()
        self.wide_offsets = List[UInt64]()
        self.wide_rows = List[UInt64]()

    def __init__(out self, var offsets: List[UInt32], var rows: List[UInt32]):
        self.wide = False
        self.offsets = offsets^
        self.rows = rows^
        self.wide_offsets = List[UInt64]()
        self.wide_rows = List[UInt64]()

    def __init__(out self, var offsets: List[UInt64], var rows: List[UInt64]):
        self.wide = True
        self.offsets = List[UInt32]()
        self.rows = List[UInt32]()
        self.wide_offsets = offsets^
        self.wide_rows = rows^

    def count(self) -> Int:
        """The number of groups."""
        var n = len(self.wide_offsets) if self.wide else len(self.offsets)
        return max(0, n - 1)

    @always_inline
    def offset(self, g: Int) -> Int:
        """Where group g's rows start in `rows` (`offset(count())` is the
        row total)."""
        if self.wide:
            return Int(self.wide_offsets[g])
        return Int(self.offsets[g])

    def offsets_address(self) -> Int:
        """The address of the offsets, `index_dtype` values."""
        if self.wide:
            return Int(self.wide_offsets.unsafe_ptr())
        return Int(self.offsets.unsafe_ptr())

    def rows_address(self) -> Int:
        """The address of the rows, `index_dtype` values."""
        if self.wide:
            return Int(self.wide_rows.unsafe_ptr())
        return Int(self.rows.unsafe_ptr())

    def slices(self, workers: Int) -> List[Int]:
        """Cut the groups into at most `workers` ranges of whole groups
        holding about equal rows; returns the range bounds, by group.
        Ranges are sized by rows rather than group counts, since one big
        group is as much work as many small ones."""
        var count = self.count()
        var bounds = List[Int]()
        bounds.append(0)
        var total = self.offset(count) if count > 0 else 0
        var share = max(1, (total + workers - 1) // max(1, workers))
        var g = 0
        for w in range(1, workers):
            var target = w * share
            while g < count and self.offset(g) < target:
                g += 1
            if g > bounds[len(bounds) - 1] and g < count:
                bounds.append(g)
        if count > bounds[len(bounds) - 1]:
            bounds.append(count)
        return bounds^


def group_rows_from_ids(ids: List[Int], count: Int) -> GroupRows:
    """The row lists of groups numbered by row (`ids[row]` in [0, count)),
    each list ascending: a counting sort of the rows by group."""
    if wide_row_indices(len(ids)):
        return _group_rows_from_ids[WIDE_ROW_INDEX](ids, count)
    return _group_rows_from_ids[ROW_INDEX](ids, count)


def _group_rows_from_ids[I: DType](ids: List[Int], count: Int) -> GroupRows:
    var offsets = List[Scalar[I]](length=count + 1, fill=0)
    var height = len(ids)
    var at = offsets.unsafe_ptr()
    var group = ids.unsafe_ptr()
    for row in range(height):
        at[unsafe_offset=group[unsafe_offset=row] + 1] += 1
    for g in range(count):
        at[unsafe_offset=g + 1] += at[unsafe_offset=g]
    var cursors = offsets.copy()
    var next = cursors.unsafe_ptr()
    var rows = huge_uninit[Scalar[I]](height)
    var dst = rows.unsafe_ptr()
    for row in range(height):
        var g = group[unsafe_offset=row]
        dst[unsafe_offset=Int(next[unsafe_offset=g])] = Scalar[I](row)
        next[unsafe_offset=g] += 1
    return group_rows_of[I](offsets^, rows^)


def consecutive_group_rows(offsets: List[Int], height: Int) -> GroupRows:
    """GroupRows of groups that each hold consecutive rows: group g holds
    rows [offsets[g], offsets[g + 1]) of `height`."""
    if wide_row_indices(height):
        return _consecutive_group_rows[WIDE_ROW_INDEX](offsets, height)
    return _consecutive_group_rows[ROW_INDEX](offsets, height)


def _consecutive_group_rows[
    I: DType
](offsets: List[Int], height: Int) -> GroupRows:
    var starts = List[Scalar[I]](capacity=len(offsets))
    for offset in offsets:
        starts.append(Scalar[I](offset))
    var rows = huge_uninit[Scalar[I]](height)
    var at = rows.unsafe_ptr()
    for k in range(height):
        at[unsafe_offset=k] = Scalar[I](k)
    return group_rows_of[I](starts^, rows^)


def group_rows_of[
    I: DType
](var offsets: List[Scalar[I]], var rows: List[Scalar[I]]) -> GroupRows:
    """GroupRows over lists of either index type."""
    comptime if I == WIDE_ROW_INDEX:
        return GroupRows(
            rebind_var[List[UInt64]](offsets^), rebind_var[List[UInt64]](rows^)
        )
    else:
        comptime assert I == ROW_INDEX, "row indices are 32 or 64 bits"
        return GroupRows(
            rebind_var[List[UInt32]](offsets^), rebind_var[List[UInt32]](rows^)
        )


struct _GroupIdsJob[I: DType](Job):
    """Groups [first, last): write each group's number to its rows."""

    var offsets: Int
    var rows: Int
    var first: Int
    var last: Int
    var ids: Int

    def __init__(
        out self, offsets: Int, rows: Int, first: Int, last: Int, ids: Int
    ):
        self.offsets = offsets
        self.rows = rows
        self.first = first
        self.last = last
        self.ids = ids

    def run(mut self) raises:
        var offsets = Pointer[Scalar[Self.I], MutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        var rows = Pointer[Scalar[Self.I], MutAnyOrigin](
            unsafe_from_address=self.rows
        )
        var ids = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ids)
        for g in range(self.first, self.last):
            for k in range(
                Int(offsets[unsafe_offset=g]), Int(offsets[unsafe_offset=g + 1])
            ):
                ids[unsafe_offset=Int(rows[unsafe_offset=k])] = g


def group_ids(groups: GroupRows, height: Int, workers: Int) raises -> List[Int]:
    """Each row's group number (by input row), from the groups' row lists,
    for the consumers that read groups row by row. Groups own disjoint
    rows, so slices of groups are written in parallel."""
    if groups.wide:
        return _group_ids[WIDE_ROW_INDEX](groups, height, workers)
    return _group_ids[ROW_INDEX](groups, height, workers)


def _group_ids[
    I: DType
](groups: GroupRows, height: Int, workers: Int) raises -> List[Int]:
    var ids = huge_uninit[Int](height)
    var bounds = groups.slices(workers)
    var jobs = List[_GroupIdsJob[I]](capacity=len(bounds))
    for s in range(len(bounds) - 1):
        jobs.append(
            _GroupIdsJob[I](
                groups.offsets_address(),
                groups.rows_address(),
                bounds[s],
                bounds[s + 1],
                Int(ids.unsafe_ptr()),
            )
        )
    if len(jobs) == 1:
        jobs[0].run()
    elif len(jobs) > 1:
        run_jobs(jobs)
    return ids^


struct _WriteNumericJob[D: DType, I: DType](Job):
    """Groups [first, last): write each group's value to every one of its
    rows, and its validity when the values have nulls."""

    var values: Int
    var source_bits: Int
    # Bit index of value 0 in `source_bits`.
    var bit_offset: Int
    var nulls: Bool
    var offsets: Int
    var rows: Int
    var first: Int
    var last: Int
    var dest: Int
    var valid: Int

    def __init__(
        out self,
        values: Int,
        source_bits: Int,
        bit_offset: Int,
        nulls: Bool,
        offsets: Int,
        rows: Int,
        first: Int,
        last: Int,
        dest: Int,
        valid: Int,
    ):
        self.values = values
        self.source_bits = source_bits
        self.bit_offset = bit_offset
        self.nulls = nulls
        self.offsets = offsets
        self.rows = rows
        self.first = first
        self.last = last
        self.dest = dest
        self.valid = valid

    def run(mut self) raises:
        var values = Pointer[Scalar[Self.D], MutAnyOrigin](
            unsafe_from_address=self.values
        )
        var offsets = Pointer[Scalar[Self.I], MutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        var rows = Pointer[Scalar[Self.I], MutAnyOrigin](
            unsafe_from_address=self.rows
        )
        var dst = Pointer[Scalar[Self.D], MutAnyOrigin](
            unsafe_from_address=self.dest
        )
        if not self.nulls:
            for g in range(self.first, self.last):
                var value = values[unsafe_offset=g]
                for k in range(
                    Int(offsets[unsafe_offset=g]),
                    Int(offsets[unsafe_offset=g + 1]),
                ):
                    dst[unsafe_offset=Int(rows[unsafe_offset=k])] = value
            return
        # A bitmap shared by threads would race within a byte (two groups'
        # rows can share one), so validity is written one Bool per row and
        # packed afterwards.
        var bits = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.source_bits
        )
        var valid = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.valid)
        for g in range(self.first, self.last):
            var bit = self.bit_offset + g
            var ok = ((bits[unsafe_offset=bit >> 3] >> UInt8(bit & 7)) & 1) != 0
            var value = values[unsafe_offset=g] if ok else Scalar[Self.D](0)
            for k in range(
                Int(offsets[unsafe_offset=g]), Int(offsets[unsafe_offset=g + 1])
            ):
                var row = Int(rows[unsafe_offset=k])
                dst[unsafe_offset=row] = value
                valid[unsafe_offset=row] = ok


def write_by_groups(
    values: Series, groups: GroupRows, height: Int, workers: Int
) raises -> Optional[Series]:
    """Spread one value per group (`values`, in group order) to the
    `height` rows the groups hold, writing each group's rows directly, in
    parallel over slices of groups. Groups own disjoint rows, so threads
    never write the same slot. Only fixed-width numeric values are written
    this way; any other type gives None, and the caller gathers by row
    instead."""
    var flat = values.rechunk() if values.is_chunked() else values.copy()
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if flat._data.isa[Column[Scalar[D]]]():
            return _write_numeric_rows[D](flat, groups, height, workers)
    if flat._data.isa[Column[Int128]]():
        return _write_numeric_rows[DType.int128](flat, groups, height, workers)
    return None


def _write_numeric_rows[
    D: DType
](
    values: Series, groups: GroupRows, height: Int, workers: Int
) raises -> Series:
    if groups.wide:
        return _write_numeric[D, WIDE_ROW_INDEX](
            values, groups, height, workers
        )
    return _write_numeric[D, ROW_INDEX](values, groups, height, workers)


def _write_numeric[
    D: DType, I: DType
](
    values: Series, groups: GroupRows, height: Int, workers: Int
) raises -> Series:
    ref column = values._data[Column[Scalar[D]]]
    var nulls = column.null_count() > 0
    var dst = huge_uninit[Scalar[D]](height)
    var valid = List[Bool](length=height if nulls else 0, fill=False)
    var source = column.copy()
    var bounds = groups.slices(workers)
    var jobs = List[_WriteNumericJob[D, I]](capacity=len(bounds))
    for s in range(len(bounds) - 1):
        jobs.append(
            _WriteNumericJob[D, I](
                Int(source._ptr()),
                Int(source.unsafe_validity()) if nulls else 0,
                source.validity_offset(),
                nulls,
                groups.offsets_address(),
                groups.rows_address(),
                bounds[s],
                bounds[s + 1],
                Int(dst.unsafe_ptr()),
                Int(valid.unsafe_ptr()),
            )
        )
    if len(jobs) == 1:
        jobs[0].run()
    elif len(jobs) > 1:
        run_jobs(jobs)
    _ = source^
    var result: Series
    if nulls:
        result = Series(values.name(), Column[Scalar[D]](dst^, valid))
    else:
        result = Series(values.name(), Column[Scalar[D]](dst^))
    result._dtype = values.dtype()
    result._field_metadata = values._field_metadata
    return result^
