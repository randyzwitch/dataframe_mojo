"""Each group's row list, and writing one value per group to its rows.

`GroupRows` is Polars' `GroupsIdx` (polars-core frame/group_by): for every
group the input rows it holds, in ascending order, so a grouped reduction
runs in parallel over slices of groups (`_agg_helper_idx`, group_by
aggregations/mod.rs) and a per-group result is written straight to the
group's rows (`set_numeric`, polars-expr expressions/window.rs). The lists
are kept as one array cut by offsets rather than one vector per group.
"""
from std.memory import Pointer

from dataframe.dtype import NUMERIC_DTYPES
from dataframe.parallel import Job, run_jobs
from dataframe.series import Series
from dataframe.storage.allocation import huge_uninit
from dataframe.storage.column import Column


struct GroupRows(Copyable, Movable):
    """Group g's rows are `rows[offsets[g]:offsets[g + 1]]`, ascending."""

    var offsets: List[Int]
    var rows: List[Int]

    def __init__(out self):
        self.offsets = List[Int]()
        self.rows = List[Int]()

    def __init__(out self, var offsets: List[Int], var rows: List[Int]):
        self.offsets = offsets^
        self.rows = rows^

    def count(self) -> Int:
        """The number of groups."""
        return max(0, len(self.offsets) - 1)

    def slices(self, workers: Int) -> List[Int]:
        """Cut the groups into at most `workers` ranges of whole groups
        holding about equal rows; returns the range bounds, by group. This
        is how the group par_iter of `_agg_helper_idx` divides work, sized
        by rows instead of group counts, since one big group is as much
        work as many small ones."""
        var count = self.count()
        var bounds = List[Int]()
        bounds.append(0)
        var total = self.offsets[count] if count > 0 else 0
        var share = max(1, (total + workers - 1) // max(1, workers))
        var offsets = self.offsets.unsafe_ptr()
        var g = 0
        for w in range(1, workers):
            var target = w * share
            while g < count and offsets[unsafe_offset=g] < target:
                g += 1
            if g > bounds[len(bounds) - 1] and g < count:
                bounds.append(g)
        if count > bounds[len(bounds) - 1]:
            bounds.append(count)
        return bounds^


def group_rows_from_ids(ids: List[Int], count: Int) -> GroupRows:
    """The row lists of groups numbered by row (`ids[row]` in [0, count)),
    each list ascending: a counting sort of the rows by group."""
    var offsets = List[Int](length=count + 1, fill=0)
    var height = len(ids)
    var at = offsets.unsafe_ptr()
    var group = ids.unsafe_ptr()
    for row in range(height):
        at[unsafe_offset=group[unsafe_offset=row] + 1] += 1
    for g in range(count):
        at[unsafe_offset=g + 1] += at[unsafe_offset=g]
    var cursors = offsets.copy()
    var next = cursors.unsafe_ptr()
    var rows = huge_uninit[Int](height)
    var dst = rows.unsafe_ptr()
    for row in range(height):
        var g = group[unsafe_offset=row]
        dst[unsafe_offset=next[unsafe_offset=g]] = row
        next[unsafe_offset=g] += 1
    return GroupRows(offsets^, rows^)


struct _SetNumericJob[D: DType](Job):
    """Groups [first, last): write each group's value to every one of its
    rows, and its validity when the values have nulls (`set_numeric`)."""

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
        var offsets = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        var rows = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.rows)
        var dst = Pointer[Scalar[Self.D], MutAnyOrigin](
            unsafe_from_address=self.dest
        )
        if not self.nulls:
            for g in range(self.first, self.last):
                var value = values[unsafe_offset=g]
                for k in range(
                    offsets[unsafe_offset=g], offsets[unsafe_offset=g + 1]
                ):
                    dst[unsafe_offset=rows[unsafe_offset=k]] = value
            return
        # A bitmap shared by threads would race within a byte, so validity
        # is written one Bool per row, as Polars writes a Vec<bool>.
        var bits = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.source_bits
        )
        var valid = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.valid)
        for g in range(self.first, self.last):
            var bit = self.bit_offset + g
            var ok = ((bits[unsafe_offset=bit >> 3] >> UInt8(bit & 7)) & 1) != 0
            var value = values[unsafe_offset=g] if ok else Scalar[Self.D](0)
            for k in range(
                offsets[unsafe_offset=g], offsets[unsafe_offset=g + 1]
            ):
                var row = rows[unsafe_offset=k]
                dst[unsafe_offset=row] = value
                valid[unsafe_offset=row] = ok


def set_by_groups(
    values: Series, groups: GroupRows, height: Int, workers: Int
) raises -> Optional[Series]:
    """Spread one value per group (`values`, in group order) to the
    `height` rows the groups hold, writing each group's rows directly, in
    parallel over slices of groups (Polars `set_by_groups`). Only
    fixed-width numeric values are written this way; any other type gives
    None, and the caller gathers by row instead, as Polars falls back to
    its join mapping."""
    var flat = values.rechunk() if values.is_chunked() else values.copy()
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if flat._data.isa[Column[Scalar[D]]]():
            return _set_numeric[D](flat, groups, height, workers)
    if flat._data.isa[Column[Int128]]():
        return _set_numeric[DType.int128](flat, groups, height, workers)
    return None


def _set_numeric[
    D: DType
](
    values: Series, groups: GroupRows, height: Int, workers: Int
) raises -> Series:
    ref column = values._data[Column[Scalar[D]]]
    var nulls = column.null_count() > 0
    var dst = huge_uninit[Scalar[D]](height)
    var valid = List[Bool](length=height if nulls else 0, fill=False)
    var source = column.copy()
    var bounds = groups.slices(workers)
    var jobs = List[_SetNumericJob[D]](capacity=len(bounds))
    for s in range(len(bounds) - 1):
        jobs.append(
            _SetNumericJob[D](
                Int(source._ptr()),
                Int(source.unsafe_validity()) if nulls else 0,
                source.validity_offset(),
                nulls,
                Int(groups.offsets.unsafe_ptr()),
                Int(groups.rows.unsafe_ptr()),
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
