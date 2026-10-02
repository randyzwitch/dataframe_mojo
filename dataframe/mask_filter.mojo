"""Filtering columns straight from a Boolean mask (#376).

`DataFrame.filter` used to turn the mask into a `List[Int]` of kept row
numbers (8 bytes a row, built per worker and concatenated) and then gather
every column from that list. Polars filters each column from the mask
itself (polars-compute's filter/primitive.rs): it reads the mask 64 bits at
a time, skips empty words, copies full words in bulk and walks the set bits
of the rest. This does the same:

1. The mask's values and validity are ANDed into 64-bit words (a null
   entry drops its row), and a popcount per word gives every word's first
   output position.
2. The output is cut into one range per worker at rows where the output
   position is a multiple of 64, so each worker writes whole words of the
   output validity (and Boolean value) bitmaps and no two workers share one.
3. Each worker copies its rows of every fixed-width and Boolean column,
   reading chunked (Parquet) columns chunk by chunk.

String and nested columns still go through a row list (`take`); #375 makes
string gathers produce views.
"""
from std.bit import count_trailing_zeros, pop_count
from std.memory import ArcPointer, Pointer, unsafe_memcpy

from .bool_column import BoolColumn
from .column import Column
from .gather import GATHER_DTYPES
from .parallel import Job, run_jobs, worker_count
from .series import Series

comptime _FULL = UInt64(0xFFFFFFFFFFFFFFFF)


@always_inline
def _bits64(bits: List[UInt8], bit: Int, length: Int) -> UInt64:
    """64 bits of an LSB-first bitmap starting at bit `bit`; bits past
    `length` (the bitmap's last valid bit, exclusive) read as 0. An empty
    bitmap is all ones (Arrow's absent validity)."""
    if len(bits) == 0:
        var left = length - bit
        return _FULL if left >= 64 else (UInt64(1) << UInt64(max(left, 0))) - 1
    var byte = bit >> 3
    var shift = UInt64(bit & 7)
    var word = UInt64(0)
    var last = min(byte + 9, len(bits))
    var k = 0
    var value = UInt64(0)
    for b in range(byte, last):
        if k < 8:
            word |= UInt64(bits[b]) << UInt64(8 * k)
        else:
            value = UInt64(bits[b])
        k += 1
    word >>= shift
    if shift > 0 and k == 9:
        word |= value << (64 - shift)
    var left = length - bit
    if left < 64:
        word &= (UInt64(1) << UInt64(max(left, 0))) - 1
    return word


struct _Selection(Copyable, Movable):
    """The mask as words aligned to row 0, each word's first output
    position, and the kept-row count."""

    var words: List[UInt64]
    var starts: List[Int]
    var count: Int
    var rows: Int

    def __init__(out self, mask: BoolColumn):
        var n = len(mask)
        self.rows = n
        var nwords = (n + 63) // 64
        self.words = List[UInt64](capacity=nwords)
        self.starts = List[Int](capacity=nwords + 1)
        ref values = mask._data[]
        ref valid = mask._bits[]
        var end = mask._offset + n
        var total = 0
        for w in range(nwords):
            var bit = mask._offset + 64 * w
            var word = _bits64(values, bit, end) & _bits64(valid, bit, end)
            self.words.append(word)
            self.starts.append(total)
            total += Int(pop_count(word))
        self.starts.append(total)
        self.count = total

    def row_of(self, position: Int) -> Int:
        """The row holding output position `position` (0 <= position <
        count): the first row whose kept rows before it number `position`."""
        var lo = 0
        var hi = len(self.words) - 1
        # The last word whose start is <= position.
        while lo < hi:
            var mid = (lo + hi + 1) // 2
            if self.starts[mid] <= position:
                lo = mid
            else:
                hi = mid - 1
        var word = self.words[lo]
        var skip = position - self.starts[lo]
        for _ in range(skip):
            word &= word - 1
        return 64 * lo + Int(count_trailing_zeros(word))


struct _Chunks(Copyable, Movable):
    """One column's arrays: where each starts, its values and validity."""

    var parts: List[Series]
    var ends: List[Int]

    def __init__(out self, column: Series):
        self.parts = column.chunks()
        self.ends = List[Int](capacity=len(self.parts))
        var end = 0
        for part in self.parts:
            end += len(part)
            self.ends.append(end)


struct _CompressJob(Job):
    """Output positions [first, last) of every fixed-width and Boolean
    column; `first` is a multiple of 64 and so is `last` unless it is the
    end. Input rows run from `row` on."""

    var selection: _Selection
    var columns: List[_Chunks]
    var values: List[Int]
    var valid: List[Int]
    var first: Int
    var last: Int
    var row: Int

    def __init__(
        out self,
        selection: _Selection,
        columns: List[_Chunks],
        values: List[Int],
        valid: List[Int],
        first: Int,
        last: Int,
        row: Int,
    ):
        self.selection = selection.copy()
        self.columns = columns.copy()
        self.values = values.copy()
        self.valid = valid.copy()
        self.first = first
        self.last = last
        self.row = row

    def run(mut self) raises:
        for c in range(len(self.columns)):
            var boolean = self.columns[c].parts[0]._data.isa[BoolColumn]()
            if boolean:
                self._booleans(c)
                continue
            comptime for d in range(len(GATHER_DTYPES)):
                comptime D = GATHER_DTYPES[d]
                if self.columns[c].parts[0]._data.isa[Column[Scalar[D]]]():
                    self._fixed[D](c)

    def _fixed[D: DType](mut self, c: Int):
        ref chunks = self.columns[c]
        var out = Pointer[List[Scalar[D]], MutAnyOrigin](
            unsafe_from_address=self.values[c]
        )[].unsafe_ptr()
        var out_valid_address = self.valid[c]
        var k = self.first
        var row = self.row
        var chunk = 0
        while chunk < len(chunks.ends) and chunks.ends[chunk] <= row:
            chunk += 1
        while k < self.last:
            # Next word of the mask from `row` on.
            var w = row >> 6
            var word = self.selection.words[w] & (_FULL << UInt64(row & 63))
            row = 64 * (w + 1)
            if word == 0:
                continue
            var base = 64 * w
            while chunk < len(chunks.ends) and chunks.ends[chunk] <= base:
                chunk += 1
            ref part = chunks.parts[chunk]._data[Column[Scalar[D]]]
            var start = chunks.ends[chunk] - len(part)
            var src = part._ptr()
            # A full word inside one array, with room before `last`: one
            # copy, and its validity bits moved in one piece.
            if (
                word == _FULL
                and base + 64 <= chunks.ends[chunk]
                and k + 64 <= self.last
            ):
                unsafe_memcpy(
                    dest=out.unsafe_offset(k),
                    src=src.unsafe_offset(base - start),
                    count=64,
                )
                if out_valid_address != 0:
                    var bits = Pointer[List[UInt8], MutAnyOrigin](
                        unsafe_from_address=out_valid_address
                    )[].unsafe_ptr()
                    var vword = _bits64(
                        part._bits[],
                        part._offset + base - start,
                        part._offset + len(part),
                    )
                    # OR the 64 bits in at bit k, which need not start a
                    # byte; bytes past this job's range are never touched,
                    # since k + 64 <= last and `last` starts a word.
                    var shift = UInt64(k & 7)
                    var at = k >> 3
                    for b in range(8):
                        var byte = (vword >> UInt64(8 * b)) & 0xFF
                        bits.unsafe_offset(at + b)[] |= UInt8(
                            (byte << shift) & 0xFF
                        )
                        if shift > 0:
                            bits.unsafe_offset(at + b + 1)[] |= UInt8(
                                byte >> (8 - shift)
                            )
                k += 64
                continue
            if base + 64 <= chunks.ends[chunk]:
                # The word lies in one array: copy each run of kept rows
                # with one memcpy, reading through this array's pointer
                # (#395; a filter keeping most rows leaves long runs).
                var local = base - start
                var nulls = out_valid_address != 0 and len(part._bits[]) > 0
                while word != 0 and k < self.last:
                    var b0 = Int(count_trailing_zeros(word))
                    var rest = word >> UInt64(b0)
                    var ones = 64 - b0 if rest == (
                        _FULL >> UInt64(b0)
                    ) else Int(count_trailing_zeros(~rest))
                    var run = min(ones, self.last - k)
                    unsafe_memcpy(
                        dest=out.unsafe_offset(k),
                        src=src.unsafe_offset(local + b0),
                        count=run,
                    )
                    if out_valid_address != 0:
                        var bits = Pointer[List[UInt8], MutAnyOrigin](
                            unsafe_from_address=out_valid_address
                        )[].unsafe_ptr()
                        for i in range(run):
                            if not nulls or part._valid(local + b0 + i):
                                var at = k + i
                                bits.unsafe_offset(at >> 3)[] |= UInt8(
                                    1
                                ) << UInt8(at & 7)
                    k += run
                    if b0 + run >= 64:
                        word = 0
                    else:
                        word &= ~(
                            ((UInt64(1) << UInt64(run)) - 1) << UInt64(b0)
                        )
                continue
            while word != 0 and k < self.last:
                var r = base + Int(count_trailing_zeros(word))
                word &= word - 1
                while chunks.ends[chunk] <= r:
                    chunk += 1
                ref piece = chunks.parts[chunk]._data[Column[Scalar[D]]]
                var local = r - (chunks.ends[chunk] - len(piece))
                out.unsafe_offset(k)[] = piece._ptr().unsafe_offset(local)[]
                if out_valid_address != 0 and piece._valid(local):
                    var bits = Pointer[List[UInt8], MutAnyOrigin](
                        unsafe_from_address=out_valid_address
                    )[].unsafe_ptr()
                    bits.unsafe_offset(k >> 3)[] |= UInt8(1) << UInt8(k & 7)
                k += 1

    def _booleans(mut self, c: Int):
        ref chunks = self.columns[c]
        var out = Pointer[List[UInt8], MutAnyOrigin](
            unsafe_from_address=self.values[c]
        )[].unsafe_ptr()
        var out_valid_address = self.valid[c]
        var k = self.first
        var row = self.row
        var chunk = 0
        while k < self.last:
            var w = row >> 6
            var word = self.selection.words[w] & (_FULL << UInt64(row & 63))
            row = 64 * (w + 1)
            var base = 64 * w
            while word != 0 and k < self.last:
                var r = base + Int(count_trailing_zeros(word))
                word &= word - 1
                while chunks.ends[chunk] <= r:
                    chunk += 1
                ref piece = chunks.parts[chunk]._data[BoolColumn]
                var local = r - (chunks.ends[chunk] - len(piece))
                if piece._get(local):
                    out.unsafe_offset(k >> 3)[] |= UInt8(1) << UInt8(k & 7)
                if out_valid_address != 0 and piece._valid(local):
                    var bits = Pointer[List[UInt8], MutAnyOrigin](
                        unsafe_from_address=out_valid_address
                    )[].unsafe_ptr()
                    bits.unsafe_offset(k >> 3)[] |= UInt8(1) << UInt8(k & 7)
                k += 1


def _fixed_width(column: Series) -> Bool:
    if column.dtype().is_nested():
        return False
    for part in column.chunks():
        if part._data.isa[BoolColumn]():
            continue
        var found = False
        comptime for d in range(len(GATHER_DTYPES)):
            comptime D = GATHER_DTYPES[d]
            if part._data.isa[Column[Scalar[D]]]():
                found = True
        if not found:
            return False
    return True


def filter_columns(
    columns: List[Series], mask: BoolColumn
) raises -> Tuple[List[Series], List[Int], Int]:
    """The fixed-width and Boolean `columns` filtered by `mask`, the indices
    (into `columns`) of the columns that were not (strings and nested
    values, left to the caller), and the kept-row count."""
    var selection = _Selection(mask)
    var count = selection.count
    var chunked = List[_Chunks]()
    var outputs = List[Series]()
    var others = List[Int]()
    var value_lists_fixed = List[Int]()
    var valid_lists = List[Int]()
    # Output buffers, kept in typed lists so their addresses are stable.
    var bool_values = List[List[UInt8]]()
    var bitmaps = List[List[UInt8]]()
    var fixed_index = List[Int]()
    for i in range(len(columns)):
        if _fixed_width(columns[i]):
            fixed_index.append(i)
        else:
            others.append(i)
    bitmaps.reserve(len(fixed_index))
    bool_values.reserve(len(fixed_index))
    for i in fixed_index:
        var has_nulls = columns[i].null_count() > 0
        bitmaps.append(
            List[UInt8](length=(count + 7) // 8 if has_nulls else 0, fill=0)
        )
    # Typed value buffers.
    var typed = List[Series]()
    for j in range(len(fixed_index)):
        ref column = columns[fixed_index[j]]
        chunked.append(_Chunks(column))
        ref first = chunked[j].parts[0]
        if first._data.isa[BoolColumn]():
            bool_values.append(List[UInt8](length=(count + 7) // 8, fill=0))
            typed.append(Series.full_null(column.name(), column.dtype(), 0))
            continue
        bool_values.append(List[UInt8]())
        comptime for d in range(len(GATHER_DTYPES)):
            comptime D = GATHER_DTYPES[d]
            if first._data.isa[Column[Scalar[D]]]():
                var output = Series(
                    column.name(),
                    Column[Scalar[D]](
                        List[Scalar[D]](unsafe_uninit_length=count)
                    ),
                )
                output._dtype = column.dtype()
                typed.append(output^)
    # Addresses only once every list has stopped moving.
    for j in range(len(fixed_index)):
        ref first = chunked[j].parts[0]
        if first._data.isa[BoolColumn]():
            value_lists_fixed.append(Int(Pointer(to=bool_values[j])))
        else:
            var address = 0
            comptime for d in range(len(GATHER_DTYPES)):
                comptime D = GATHER_DTYPES[d]
                if typed[j]._data.isa[Column[Scalar[D]]]():
                    address = Int(
                        Pointer(to=typed[j]._data[Column[Scalar[D]]]._data[])
                    )
            value_lists_fixed.append(address)
        valid_lists.append(
            Int(Pointer(to=bitmaps[j])) if len(bitmaps[j]) > 0 else 0
        )
    if len(fixed_index) > 0 and count > 0:
        var workers = worker_count(count)
        var jobs = List[_CompressJob](capacity=workers)
        for w in range(workers):
            var first = (count * w // workers) // 64 * 64
            var last = (
                count if w
                == workers - 1 else (count * (w + 1) // workers) // 64 * 64
            )
            if last <= first:
                continue
            jobs.append(
                _CompressJob(
                    selection,
                    chunked,
                    value_lists_fixed,
                    valid_lists,
                    first,
                    last,
                    selection.row_of(first),
                )
            )
        if len(jobs) == 1:
            jobs[0].run()
        else:
            run_jobs(jobs)
    for j in range(len(fixed_index)):
        ref column = columns[fixed_index[j]]
        var bits = bitmaps[j].copy()
        if chunked[j].parts[0]._data.isa[BoolColumn]():
            var result = Series(
                column.name(),
                BoolColumn(
                    values=bool_values[j].copy(), bits=bits^, length=count
                ),
            )
            result._dtype = column.dtype()
            outputs.append(result^)
            continue
        var result = typed[j].copy()
        comptime for d in range(len(GATHER_DTYPES)):
            comptime D = GATHER_DTYPES[d]
            if result._data.isa[Column[Scalar[D]]]():
                result._data[Column[Scalar[D]]]._bits = ArcPointer(
                    bitmaps[j].copy()
                )
        outputs.append(result^)
    return (outputs^, others^, count)
