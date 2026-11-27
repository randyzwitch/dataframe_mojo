"""Element transforms on list columns: unique, sort, reverse, slices,
gathers, drop_nulls, shift, diff, and concatenation.

Every transform builds row i's list from the elements of input row i, so
the result keeps the input's row validity: a null row gives a null row with
no elements (whatever payload is stored beneath it) and an empty row gives
an empty row.

Most transforms only choose child positions. One pass counts each row's
output elements, the counts become offsets, a second pass writes the chosen
positions, and one gather over the child copies the elements, so these
transforms work for every element type. Both passes split the rows into
contiguous ranges holding about the same number of elements, one range per
worker.

unique and sort compare elements. Numeric and string elements are compared
in place, one row at a time, each worker taking a range of rows. A row long
enough to be split across workers by itself, and every row of any other
element type, goes through the general sort and row-key encoders instead,
which split their input across workers.
"""
from std.math import isnan
from std.memory import Pointer, unsafe_memcmp
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.expr import (
    LIST_CONCAT,
    LIST_DIFF,
    LIST_DROP_NULLS,
    LIST_GATHER,
    LIST_GATHER_EVERY,
    LIST_GET,
    LIST_JOIN,
    LIST_LEN,
    LIST_MAX,
    LIST_MEAN,
    LIST_MIN,
    LIST_N_UNIQUE,
    LIST_REVERSE,
    LIST_SHIFT,
    LIST_SLICE,
    LIST_SORT,
    LIST_SUM,
    LIST_UNIQUE,
    LIST_CONTAINS,
    Node,
    split_names,
)
from dataframe.parallel import Job, run_jobs, worker_count
from dataframe.series import Series
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column, _append_validity
from dataframe.storage.nested import ListColumn
from dataframe.storage.string import StringColumn
from dataframe.text.bytes import _Bytes

# Positions every element of a valid row, in order (no public operation).
comptime _IDENTITY = -1
# Rows this short are sorted by insertion and deduplicated by comparing
# each element with the ones kept so far.
comptime _SMALL = 16


def list_op_name(op: Int) -> String:
    """The method name of a list operation, for error messages."""
    if op == LIST_LEN:
        return "list.len"
    if op == LIST_GET:
        return "list.get"
    if op == LIST_CONTAINS:
        return "list.contains"
    if op == LIST_JOIN:
        return "list.join"
    if op == LIST_SUM:
        return "list.sum"
    if op == LIST_MIN:
        return "list.min"
    if op == LIST_MAX:
        return "list.max"
    if op == LIST_MEAN:
        return "list.mean"
    if op == LIST_UNIQUE:
        return "list.unique"
    if op == LIST_N_UNIQUE:
        return "list.n_unique"
    if op == LIST_SORT:
        return "list.sort"
    if op == LIST_REVERSE:
        return "list.reverse"
    if op == LIST_SLICE:
        return "list.slice"
    if op == LIST_GATHER:
        return "list.gather"
    if op == LIST_GATHER_EVERY:
        return "list.gather_every"
    if op == LIST_DROP_NULLS:
        return "list.drop_nulls"
    if op == LIST_SHIFT:
        return "list.shift"
    if op == LIST_DIFF:
        return "list.diff"
    if op == LIST_CONCAT:
        return "list.concat"
    return "list operation " + String(op)


def list_diff_inner(inner: DataType) raises -> DataType:
    """The element type of list.diff: a signed integer wide enough for the
    difference of two unsigned values, otherwise the input type."""
    if inner.is_categorical() or not (inner.is_integer() or inner.is_float()):
        raise Error(
            "list.diff needs an integer or float list, found " + inner.name()
        )
    if inner.is_unsigned():
        if inner.bit_width() == 8:
            return DataType.INT16
        if inner.bit_width() == 16:
            return DataType.INT32
        return DataType.INT64
    return inner


def list_concat_inner(left: DataType, right: DataType) raises -> DataType:
    """The common element type of two concatenated lists: the type both
    share, or the numeric type that holds the values of each."""
    if left == right:
        return left
    if left == DataType.NULL:
        return right
    if right == DataType.NULL:
        return left
    if (
        (left.is_integer() or left.is_float())
        and (right.is_integer() or right.is_float())
        and not left.is_categorical()
        and not right.is_categorical()
    ):
        if left.is_float() or right.is_float():
            if left == DataType.FLOAT64 or right == DataType.FLOAT64:
                return DataType.FLOAT64
            var integer = right if left.is_float() else left
            return (
                DataType.FLOAT32 if integer.bit_width()
                <= 16 else DataType.FLOAT64
            )
        if left.is_unsigned() == right.is_unsigned():
            return left if left.bit_width() > right.bit_width() else right
        var signed = right if left.is_unsigned() else left
        var unsigned = left if left.is_unsigned() else right
        if signed.bit_width() > unsigned.bit_width():
            return signed
        if unsigned.bit_width() == 64:
            return DataType.FLOAT64
        return (
            DataType.INT16 if unsigned.bit_width()
            == 8 else DataType.INT32 if unsigned.bit_width()
            == 16 else DataType.INT64
        )
    raise Error(
        "list.concat needs elements with a common type, found "
        + left.name()
        + " and "
        + right.name()
    )


# --- Raw reads shared by the jobs ------------------------------------------


struct _View(Copyable, Movable):
    """A list column's offsets and row validity read through addresses, so
    a job never copies the column's buffer handles per row."""

    var offsets: Int
    var bits: Int
    var bit0: Int
    # The stored offset of row 0, so positions count from the first element
    # the rows span (see `_elements`).
    var base: Int

    def __init__(out self, lists: ListColumn):
        self.base = lists._start(0)
        self.offsets = Int(lists._offsets.unsafe_ptr()) + 8 * lists._offset
        self.bits = 0
        if lists.null_count() > 0:
            self.bits = Int(lists._bits.unsafe_ptr())
        self.bit0 = lists._offset

    @always_inline
    def start(self, i: Int) -> Int:
        return (
            Int(
                Pointer[Int64, MutAnyOrigin](unsafe_from_address=self.offsets)[
                    unsafe_offset=i
                ]
            )
            - self.base
        )

    @always_inline
    def valid(self, i: Int) -> Bool:
        if self.bits == 0:
            return True
        var bit = self.bit0 + i
        var byte = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.bits)[
            unsafe_offset=bit >> 3
        ]
        return ((byte >> UInt8(bit & 7)) & 1) != 0

    @always_inline
    def count(self, i: Int) -> Int:
        """Elements of row i, 0 for a null row."""
        return self.start(i + 1) - self.start(i) if self.valid(i) else 0


struct _Present(Copyable, Movable):
    """Which child elements hold a value: a bitmap read by address, or no
    bitmap when every element does. Holds the bitmap's owner."""

    var bits: Int
    var bit0: Int
    var owner: Optional[Series]

    def __init__(out self):
        """Every element present."""
        self.bits = 0
        self.bit0 = 0
        self.owner = None

    def __init__(out self, child: Series) raises:
        self = Self()
        if child.null_count() == 0:
            return
        var values = child.rechunk() if child.is_chunked() else child.copy()
        if values._data.isa[StringColumn]():
            ref column = values._data[StringColumn]
            self.bits = Int(column._bits.unsafe_ptr())
            self.bit0 = column._offset
        elif values._data.isa[BoolColumn]():
            ref column = values._data[BoolColumn]
            self.bits = Int(column._bits.unsafe_ptr())
            self.bit0 = column._offset
        elif values._data.isa[ListColumn]():
            ref column = values._data[ListColumn]
            self.bits = Int(column._bits.unsafe_ptr())
            self.bit0 = column._offset
        elif values._data.isa[Column[Int128]]():
            ref column = values._data[Column[Int128]]
            self.bits = Int(column.unsafe_validity())
            self.bit0 = column.validity_offset()
        else:
            comptime for k in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[k]
                if values._data.isa[Column[Scalar[D]]]():
                    ref column = values._data[Column[Scalar[D]]]
                    self.bits = Int(column.unsafe_validity())
                    self.bit0 = column.validity_offset()
            if self.bits == 0:
                # Other storage (structs): its validity as a Bool column.
                values = child.is_not_null().rechunk()
                ref column = values._data[BoolColumn]
                self.bits = Int(column._data.unsafe_ptr())
                self.bit0 = column._offset
        self.owner = values^

    @always_inline
    def all(self) -> Bool:
        return self.bits == 0

    @always_inline
    def at(self, r: Int) -> Bool:
        if self.bits == 0:
            return True
        var bit = self.bit0 + r
        var byte = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.bits)[
            unsafe_offset=bit >> 3
        ]
        return ((byte >> UInt8(bit & 7)) & 1) != 0


def _ranges(view: _View, rows: Int, elements: Int) -> List[Int]:
    """Row boundaries of contiguous ranges holding about the same number of
    stored elements, one range per worker the elements warrant."""
    var workers = min(worker_count(elements), max(rows, 1))
    var bounds = List[Int](capacity=workers + 1)
    bounds.append(0)
    var first = view.start(0)
    for w in range(1, workers):
        var target = first + (elements * w) // workers
        # The first row starting at or after target.
        var lo = bounds[len(bounds) - 1]
        var hi = rows
        while lo < hi:
            var mid = (lo + hi) // 2
            if view.start(mid) < target:
                lo = mid + 1
            else:
                hi = mid
        bounds.append(lo)
    bounds.append(rows)
    return bounds^


def _elements(lists: ListColumn) raises -> Series:
    """The child elements the rows span, as a window without a copy. A
    column that is itself a window onto a longer child (one batch of rows)
    then costs work in proportion to its own elements, not the child's."""
    var child = lists.child()
    var first = lists._start(0)
    var last = lists._start(len(lists))
    if first == 0 and last == len(child):
        return child^
    return child.slice(first, last - first)


def _row_bits(lists: ListColumn) -> List[UInt8]:
    """The column's row validity as a bitmap starting at row 0 (empty when
    every row is valid)."""
    var bits = List[UInt8]()
    if lists.null_count() > 0:
        _append_validity(bits, 0, lists._bits[], lists._offset, len(lists))
    return bits^


def _slice_bounds(offset: Int, length: Int, count: Int) -> Tuple[Int, Int]:
    """[begin, end) of a slice of a `count`-element list; a negative offset
    counts from the end, length -1 runs to the end, and both ends are cut
    to the list."""
    var begin = offset + count if offset < 0 else offset
    var end = count if length < 0 else begin + length
    return (max(0, min(begin, count)), max(0, min(end, count)))


# --- Position transforms ---------------------------------------------------


struct _PositionsJob(Job):
    """One range of rows. The counting pass stores each row's output count
    at offsets[i + 1] and the range total; the filling pass turns those
    counts into offsets from `base` and writes the chosen child positions
    (-1 for a null element)."""

    var op: Int
    var filling: Bool
    var view: _View
    var present: _Present
    var a: Int
    var b: Int
    var indices: List[Int]
    var lo: Int
    var hi: Int
    var offsets: Int
    var positions: Int
    var base: Int
    var total: Int
    var nulls: Bool

    def __init__(
        out self,
        op: Int,
        view: _View,
        present: _Present,
        a: Int,
        b: Int,
        indices: List[Int],
        lo: Int,
        hi: Int,
        offsets: Int,
    ):
        self.op = op
        self.filling = False
        self.view = view.copy()
        self.present = present.copy()
        self.a = a
        self.b = b
        self.indices = indices.copy()
        self.lo = lo
        self.hi = hi
        self.offsets = offsets
        self.positions = 0
        self.base = 0
        self.total = 0
        self.nulls = False

    def _count(self, i: Int) -> Int:
        var count = self.view.count(i)
        if self.op == LIST_SLICE:
            var bounds = _slice_bounds(self.a, self.b, count)
            return bounds[1] - bounds[0]
        if self.op == LIST_GATHER:
            return len(self.indices) if self.view.valid(i) else 0
        if self.op == LIST_GATHER_EVERY:
            if self.b >= count:
                return 0
            return (count - self.b + self.a - 1) // self.a
        if self.op == LIST_DROP_NULLS and not self.present.all():
            var kept = 0
            var start = self.view.start(i)
            for r in range(start, start + count):
                kept += Int(self.present.at(r))
            return kept
        return count

    def run(mut self) raises:
        var offsets = Pointer[Int64, MutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        if not self.filling:
            var total = 0
            for i in range(self.lo, self.hi):
                var count = self._count(i)
                offsets[unsafe_offset=i + 1] = Int64(count)
                total += count
            self.total = total
            return
        var out = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.positions)
        var at = self.base
        for i in range(self.lo, self.hi):
            var produced = Int(offsets[unsafe_offset=i + 1])
            offsets[unsafe_offset=i + 1] = Int64(at + produced)
            if produced == 0:
                continue
            var start = self.view.start(i)
            var count = self.view.count(i)
            if self.op == LIST_REVERSE:
                for j in range(count):
                    out[unsafe_offset=at + j] = start + count - 1 - j
            elif self.op == LIST_SLICE:
                var bounds = _slice_bounds(self.a, self.b, count)
                for j in range(bounds[1] - bounds[0]):
                    out[unsafe_offset=at + j] = start + bounds[0] + j
            elif self.op == LIST_GATHER:
                if not self.view.valid(i):
                    continue
                for j in range(len(self.indices)):
                    var index = self.indices[j]
                    var position = index + count if index < 0 else index
                    if position < 0 or position >= count:
                        if self.a == 0:
                            raise Error(
                                "list.gather index "
                                + String(index)
                                + " is out of bounds for a list of length "
                                + String(count)
                            )
                        out[unsafe_offset=at + j] = -1
                        self.nulls = True
                    else:
                        out[unsafe_offset=at + j] = start + position
            elif self.op == LIST_GATHER_EVERY:
                for j in range(produced):
                    out[unsafe_offset=at + j] = start + self.b + j * self.a
            elif self.op == LIST_DROP_NULLS:
                var k = 0
                for r in range(start, start + count):
                    if self.present.at(r):
                        out[unsafe_offset=at + k] = r
                        k += 1
            elif self.op == LIST_SHIFT:
                for j in range(count):
                    var source = j - self.a
                    if source < 0 or source >= count:
                        out[unsafe_offset=at + j] = -1
                        self.nulls = True
                    else:
                        out[unsafe_offset=at + j] = start + source
            else:
                for j in range(count):
                    out[unsafe_offset=at + j] = start + j
            at += produced


struct _Chosen(Movable):
    """Output offsets and child positions of a position transform."""

    var offsets: List[Int64]
    var positions: List[Int]
    var nulls: Bool

    def __init__(
        out self,
        var offsets: List[Int64],
        var positions: List[Int],
        nulls: Bool,
    ):
        self.offsets = offsets^
        self.positions = positions^
        self.nulls = nulls

    def into_offsets(deinit self) -> List[Int64]:
        return self.offsets^


def _choose(
    lists: ListColumn,
    op: Int,
    a: Int = 0,
    b: Int = 0,
    indices: List[Int] = List[Int](),
) raises -> _Chosen:
    """Run both passes of a position transform over every row."""
    var n = len(lists)
    var view = _View(lists)
    var present = _Present()
    if op == LIST_DROP_NULLS:
        present = _Present(_elements(lists))
    var offsets = List[Int64](length=n + 1, fill=0)
    if n == 0:
        return _Chosen(offsets^, List[Int](), False)
    var stored = view.start(n) - view.start(0)
    var bounds = _ranges(view, n, stored)
    var jobs = List[_PositionsJob](capacity=len(bounds) - 1)
    for w in range(len(bounds) - 1):
        jobs.append(
            _PositionsJob(
                op,
                view,
                present,
                a,
                b,
                indices,
                bounds[w],
                bounds[w + 1],
                Int(offsets.unsafe_ptr()),
            )
        )
    _run(jobs)
    var total = 0
    for w in range(len(jobs)):
        jobs[w].filling = True
        jobs[w].base = total
        total += jobs[w].total
    var positions = List[Int](unsafe_uninit_length=total)
    for w in range(len(jobs)):
        jobs[w].positions = Int(positions.unsafe_ptr())
    _run(jobs)
    var nulls = False
    for w in range(len(jobs)):
        nulls = nulls or jobs[w].nulls
    _ = present^
    return _Chosen(offsets^, positions^, nulls)


def _run[J: Job](mut jobs: List[J]) raises:
    if len(jobs) == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)


def _gathered(lists: ListColumn, var chosen: _Chosen) raises -> Series:
    """The list column whose rows hold the chosen child positions."""
    var child = _elements(lists)
    var elements = child.take_or_null(
        chosen.positions
    ) if chosen.nulls else child.take(chosen.positions)
    return Series(
        "", ListColumn(chosen^.into_offsets(), elements^, _row_bits(lists))
    )


# --- Comparing elements ----------------------------------------------------


trait _Order(Copyable, Deinitable, Movable):
    """Comparisons between two present child elements, by position."""

    def less(self, a: Int, b: Int) -> Bool:
        ...

    def same(self, a: Int, b: Int) -> Bool:
        ...


struct _NumberOrder[D: DType](_Order):
    """Numeric order with NaN after every number and equal to itself."""

    var values: Int

    def __init__(out self, values: Int):
        self.values = values

    @always_inline
    def less(self, a: Int, b: Int) -> Bool:
        var p = Pointer[Scalar[Self.D], MutAnyOrigin](
            unsafe_from_address=self.values
        )
        var x = p[unsafe_offset=a]
        var y = p[unsafe_offset=b]
        comptime if Self.D.is_floating_point():
            if isnan(x):
                return False
            if isnan(y):
                return True
        return x < y

    @always_inline
    def same(self, a: Int, b: Int) -> Bool:
        var p = Pointer[Scalar[Self.D], MutAnyOrigin](
            unsafe_from_address=self.values
        )
        var x = p[unsafe_offset=a]
        var y = p[unsafe_offset=b]
        comptime if Self.D.is_floating_point():
            if isnan(x):
                return isnan(y)
        return x == y


struct _BytesOrder(_Order):
    """Byte-wise order of strings or binary values (shorter first on a
    common prefix)."""

    var rows: _Bytes

    def __init__(out self, column: StringColumn):
        self.rows = _Bytes(column)

    @always_inline
    def less(self, a: Int, b: Int) -> Bool:
        var x = self.rows.get(a)
        var y = self.rows.get(b)
        var common = min(len(x), len(y))
        var order = unsafe_memcmp(
            Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=Int(x.unsafe_ptr())
            ),
            Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=Int(y.unsafe_ptr())
            ),
            common,
        )
        return order < 0 or (order == 0 and len(x) < len(y))

    @always_inline
    def same(self, a: Int, b: Int) -> Bool:
        var x = self.rows.get(a)
        var y = self.rows.get(b)
        return len(x) == len(y) and (
            unsafe_memcmp(
                Pointer[UInt8, ImmutAnyOrigin](
                    unsafe_from_address=Int(x.unsafe_ptr())
                ),
                Pointer[UInt8, ImmutAnyOrigin](
                    unsafe_from_address=Int(y.unsafe_ptr())
                ),
                len(x),
            )
            == 0
        )


@always_inline
def _swap(p: Pointer[Int, MutAnyOrigin], i: Int, j: Int):
    var t = p[unsafe_offset=i]
    p[unsafe_offset=i] = p[unsafe_offset=j]
    p[unsafe_offset=j] = t


def _sort_positions[
    O: _Order
](order: O, p: Pointer[Int, MutAnyOrigin], first: Int, last: Int):
    """Sort positions p[first:last) by their elements: quicksort with a
    three-way partition, so runs of equal elements end a branch at once,
    and insertion sort for short spans."""
    var lo = first
    var hi = last
    while hi - lo > _SMALL:
        var mid = lo + (hi - lo) // 2
        # Median of three as the pivot.
        if order.less(p[unsafe_offset=mid], p[unsafe_offset=lo]):
            _swap(p, mid, lo)
        if order.less(p[unsafe_offset=hi - 1], p[unsafe_offset=lo]):
            _swap(p, hi - 1, lo)
        if order.less(p[unsafe_offset=hi - 1], p[unsafe_offset=mid]):
            _swap(p, hi - 1, mid)
        var pivot = p[unsafe_offset=mid]
        var lt = lo
        var i = lo
        var gt = hi - 1
        while i <= gt:
            var x = p[unsafe_offset=i]
            if order.less(x, pivot):
                _swap(p, lt, i)
                lt += 1
                i += 1
            elif order.less(pivot, x):
                _swap(p, i, gt)
                gt -= 1
            else:
                i += 1
        # [lo, lt) is below the pivot, (gt, hi) above: recurse into the
        # smaller side and loop on the larger, so the depth stays
        # logarithmic.
        if lt - lo < hi - gt - 1:
            _sort_positions(order, p, lo, lt)
            lo = gt + 1
        else:
            _sort_positions(order, p, gt + 1, hi)
            hi = lt
    for i in range(lo + 1, hi):
        var x = p[unsafe_offset=i]
        var j = i
        while j > lo and order.less(x, p[unsafe_offset=j - 1]):
            p[unsafe_offset=j] = p[unsafe_offset=j - 1]
            j -= 1
        p[unsafe_offset=j] = x


def _long(count: Int) -> Bool:
    """Whether one row holds enough elements to be split across workers."""
    return worker_count(count) > 1


struct _SortJob[O: _Order](Job):
    """Sort each row of a range into its output span, nulls first or last;
    rows long enough for several workers are left to the caller."""

    var order: Self.O
    var view: _View
    var present: _Present
    var lo: Int
    var hi: Int
    var offsets: Int
    var positions: Int
    var descending: Bool
    var nulls_last: Bool

    def __init__(
        out self,
        order: Self.O,
        view: _View,
        present: _Present,
        lo: Int,
        hi: Int,
        offsets: Int,
        positions: Int,
        descending: Bool,
        nulls_last: Bool,
    ):
        self.order = order.copy()
        self.view = view.copy()
        self.present = present.copy()
        self.lo = lo
        self.hi = hi
        self.offsets = offsets
        self.positions = positions
        self.descending = descending
        self.nulls_last = nulls_last

    def run(mut self) raises:
        var offsets = Pointer[Int64, MutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        var out = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.positions)
        for i in range(self.lo, self.hi):
            var count = self.view.count(i)
            if count == 0 or _long(count):
                continue
            var start = self.view.start(i)
            var at = Int(offsets[unsafe_offset=i])
            var first = at
            var last = at + count
            if self.present.all():
                for j in range(count):
                    out[unsafe_offset=at + j] = start + j
            else:
                var missing = 0
                for r in range(start, start + count):
                    missing += Int(not self.present.at(r))
                # Present elements fill [first, last); nulls the rest.
                var null_at = at + count - missing if self.nulls_last else at
                var value_at = at if self.nulls_last else at + missing
                first = value_at
                last = value_at + count - missing
                for r in range(start, start + count):
                    if self.present.at(r):
                        out[unsafe_offset=value_at] = r
                        value_at += 1
                    else:
                        out[unsafe_offset=null_at] = r
                        null_at += 1
            _sort_positions(self.order, out, first, last)
            if self.descending:
                var a = first
                var b = last - 1
                while a < b:
                    _swap(out, a, b)
                    a += 1
                    b -= 1


struct _UniqueJob[O: _Order](Job):
    """Keep the first occurrence of each distinct element of each row of a
    range, in position order, appending the kept positions to `kept` and
    each row's count to offsets[i + 1]; with `counting`, only the counts.
    Rows long enough for several workers are left to the caller in
    `deferred`."""

    var order: Self.O
    var view: _View
    var present: _Present
    var lo: Int
    var hi: Int
    var offsets: Int
    var counting: Bool
    var kept: List[Int]
    var deferred: List[Int]

    def __init__(
        out self,
        order: Self.O,
        view: _View,
        present: _Present,
        lo: Int,
        hi: Int,
        offsets: Int,
        counting: Bool,
    ):
        self.order = order.copy()
        self.view = view.copy()
        self.present = present.copy()
        self.lo = lo
        self.hi = hi
        self.offsets = offsets
        self.counting = counting
        self.kept = List[Int]()
        self.deferred = List[Int]()

    def run(mut self) raises:
        var offsets = Pointer[Int64, MutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        var scratch = List[Int]()
        var flags = List[Bool]()
        for i in range(self.lo, self.hi):
            var count = self.view.count(i)
            if count == 0:
                offsets[unsafe_offset=i + 1] = 0
                continue
            if _long(count):
                self.deferred.append(i)
                continue
            var start = self.view.start(i)
            var row_first = len(self.kept)
            if count <= _SMALL:
                var seen_null = False
                for r in range(start, start + count):
                    if not self.present.at(r):
                        if not seen_null:
                            seen_null = True
                            self.kept.append(r)
                        continue
                    var repeated = False
                    for k in range(row_first, len(self.kept)):
                        var q = self.kept[k]
                        if self.present.at(q) and self.order.same(q, r):
                            repeated = True
                            break
                    if not repeated:
                        self.kept.append(r)
            else:
                # Sort the present positions, mark the first position of
                # each run of equal elements, then keep the marks in order.
                scratch.clear()
                flags.clear()
                flags.resize(count, False)
                var first_null = -1
                for r in range(start, start + count):
                    if self.present.at(r):
                        scratch.append(r)
                    elif first_null < 0:
                        first_null = r
                if first_null >= 0:
                    flags[first_null - start] = True
                var p = Pointer[Int, MutAnyOrigin](
                    unsafe_from_address=Int(scratch.unsafe_ptr())
                )
                _sort_positions(self.order, p, 0, len(scratch))
                var k = 0
                while k < len(scratch):
                    var smallest = scratch[k]
                    var run_end = k + 1
                    while run_end < len(scratch) and self.order.same(
                        scratch[k], scratch[run_end]
                    ):
                        smallest = min(smallest, scratch[run_end])
                        run_end += 1
                    flags[smallest - start] = True
                    k = run_end
                for r in range(count):
                    if flags[r]:
                        self.kept.append(start + r)
            var produced = len(self.kept) - row_first
            offsets[unsafe_offset=i + 1] = Int64(produced)
            if self.counting:
                self.kept.shrink(row_first)


def _ordered[
    O: _Order
](order: O, lists: ListColumn, op: Int, flag: Int,) raises -> Series:
    """sort, unique, or n_unique for element types compared in place."""
    var n = len(lists)
    var view = _View(lists)
    var child = _elements(lists)
    var present = _Present(child)
    var stored = view.start(n) - view.start(0)
    var bounds = _ranges(view, n, stored)
    if op == LIST_SORT:
        var chosen = _choose(lists, LIST_REVERSE)
        var descending = (flag & 1) != 0
        var nulls_last = (flag & 2) != 0
        var jobs = List[_SortJob[O]](capacity=len(bounds) - 1)
        for w in range(len(bounds) - 1):
            jobs.append(
                _SortJob[O](
                    order,
                    view,
                    present,
                    bounds[w],
                    bounds[w + 1],
                    Int(chosen.offsets.unsafe_ptr()),
                    Int(chosen.positions.unsafe_ptr()),
                    descending,
                    nulls_last,
                )
            )
        _run(jobs)
        for i in range(n):
            var count = view.count(i)
            if _long(count):
                var start = view.start(i)
                var at = Int(chosen.offsets[i])
                var rows = child.slice(start, count).argsort(
                    descending, nulls_last
                )
                for j in range(count):
                    chosen.positions[at + j] = start + rows[j]
        _ = present^
        return _gathered(lists, chosen^)
    var counting = op == LIST_N_UNIQUE
    var offsets = List[Int64](length=n + 1, fill=0)
    var jobs = List[_UniqueJob[O]](capacity=len(bounds) - 1)
    for w in range(len(bounds) - 1):
        jobs.append(
            _UniqueJob[O](
                order,
                view,
                present,
                bounds[w],
                bounds[w + 1],
                Int(offsets.unsafe_ptr()),
                counting,
            )
        )
    if n > 0:
        _run(jobs)
    _ = present^
    # Long rows: their distinct elements through the row-key encoder.
    var long_kept = List[List[Int]]()
    for w in range(len(jobs)):
        for i in jobs[w].deferred:
            var start = view.start(i)
            var rows = _first_rows(child.slice(start, view.count(i)))
            offsets[i + 1] = Int64(len(rows))
            if not counting:
                for k in range(len(rows)):
                    rows[k] += start
                long_kept.append(rows^)
    if counting:
        var counts = List[Int64](unsafe_uninit_length=n)
        for i in range(n):
            counts[i] = offsets[i + 1]
        return Series("", Column[Int64](values=counts^, bits=_row_bits(lists)))
    for i in range(n):
        offsets[i + 1] += offsets[i]
    var positions = List[Int](capacity=Int(offsets[n]))
    var next_long = 0
    for w in range(len(jobs)):
        var taken = 0
        for i in jobs[w].deferred:
            var before = Int(offsets[i]) - len(positions)
            positions.extend(Span(jobs[w].kept)[taken : taken + before])
            taken += before
            positions.extend(Span(long_kept[next_long]))
            next_long += 1
        positions.extend(Span(jobs[w].kept)[taken:])
    return _gathered(lists, _Chosen(offsets^, positions^, False))


def _first_rows(values: Series) raises -> List[Int]:
    """Positions of the first occurrence of each distinct value (null
    counted once), in position order."""
    from dataframe.frame import DataFrame

    var keys = DataFrame([values.renamed("v")])._subset_row_keys(List[String]())
    var chosen = List[Bool](length=len(values), fill=False)
    for row in keys.representatives:
        chosen[row] = True
    var rows = List[Int](capacity=len(keys.representatives))
    for i in range(len(values)):
        if chosen[i]:
            rows.append(i)
    return rows^


def _encoded(lists: ListColumn, op: Int, flag: Int) raises -> Series:
    """sort, unique, or n_unique for any element type, through the general
    encoders keyed by (row, element)."""
    from dataframe.frame import DataFrame
    from dataframe.sorting.sort import sort_rows

    var n = len(lists)
    var view = _View(lists)
    var every = _choose(lists, _IDENTITY)
    var rows = List[Int64](unsafe_uninit_length=len(every.positions))
    for i in range(n):
        for j in range(Int(every.offsets[i]), Int(every.offsets[i + 1])):
            rows[j] = Int64(i)
    if len(every.positions) == 0:
        if op == LIST_N_UNIQUE:
            return Series(
                "",
                Column[Int64](
                    values=List[Int64](length=n, fill=0), bits=_row_bits(lists)
                ),
            )
        return _gathered(lists, every^)
    var owners = Series("row", Column[Int64](rows^))
    var elements = _elements(lists).take(every.positions).renamed("v")
    if op == LIST_SORT:
        var order = sort_rows(
            [owners^, elements^],
            [False, (flag & 1) != 0],
            [False, (flag & 2) != 0],
            maintain_order=True,
        )
        var positions = List[Int](unsafe_uninit_length=len(order))
        for j in range(len(order)):
            positions[j] = every.positions[order[j]]
        return _gathered(
            lists, _Chosen(every^.into_offsets(), positions^, False)
        )
    var keys = DataFrame([owners^, elements^])._subset_row_keys(List[String]())
    var chosen = List[Bool](length=len(every.positions), fill=False)
    for row in keys.representatives:
        chosen[row] = True
    var offsets = List[Int64](length=n + 1, fill=0)
    var positions = List[Int](capacity=len(keys.representatives))
    for i in range(n):
        var kept = 0
        for j in range(Int(every.offsets[i]), Int(every.offsets[i + 1])):
            if chosen[j]:
                kept += 1
                if op == LIST_UNIQUE:
                    positions.append(every.positions[j])
        offsets[i + 1] = offsets[i] + Int64(kept)
    if op == LIST_N_UNIQUE:
        var counts = List[Int64](unsafe_uninit_length=n)
        for i in range(n):
            counts[i] = offsets[i + 1] - offsets[i]
        return Series("", Column[Int64](values=counts^, bits=_row_bits(lists)))
    _ = view^
    return _gathered(lists, _Chosen(offsets^, positions^, False))


def _compared(lists: ListColumn, op: Int, flag: Int) raises -> Series:
    """Dispatch sort/unique/n_unique on the child's storage."""
    var child = _elements(lists)
    if child.is_chunked():
        child = child.rechunk()
    var dtype = child.dtype()
    if not dtype.is_categorical():
        if child._data.isa[StringColumn]():
            return _ordered(
                _BytesOrder(child._data[StringColumn]), lists, op, flag
            )
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if child._data.isa[Column[Scalar[D]]]():
                ref column = child._data[Column[Scalar[D]]]
                var order = _NumberOrder[D](Int(column.unsafe_values()))
                var result = _ordered(order, lists, op, flag)
                _ = child^
                return result^
    return _encoded(lists, op, flag)


# --- diff and concat -------------------------------------------------------


def _diff(lists: ListColumn, n: Int) raises -> Series:
    """Each element minus the element n positions earlier in its row."""
    var every = _choose(lists, _IDENTITY)
    var earlier = _choose(lists, LIST_SHIFT, n)
    var child = _elements(lists)
    var target = list_diff_inner(child.dtype())
    var current = child.take(every.positions).cast(target)
    var previous = child.take_or_null(earlier.positions).cast(target)
    var difference = (current - previous).renamed(child.name())
    return Series(
        "", ListColumn(every^.into_offsets(), difference^, _row_bits(lists))
    )


def _as_lists(value: Series, inner: DataType, rows: Int) raises -> ListColumn:
    """A concat operand as a list column of `rows` rows with `inner`
    elements: a non-list value becomes a one-element list, and a single
    row is repeated."""
    var lists: ListColumn
    if value.dtype().is_list():
        lists = value.list_column()
    else:
        var count = len(value)
        var offsets = List[Int64](unsafe_uninit_length=count + 1)
        for i in range(count + 1):
            offsets[i] = Int64(i)
        var child = value.rechunk() if value.is_chunked() else value.copy()
        lists = ListColumn(offsets^, child.renamed("item"))
    if lists.child().dtype() != inner:
        var count = len(lists)
        var base = lists._start(0)
        var offsets = List[Int64](unsafe_uninit_length=count + 1)
        for i in range(count + 1):
            offsets[i] = Int64(lists._start(i) - base)
        lists = ListColumn(
            offsets^, _elements(lists).cast(inner), _row_bits(lists)
        )
    if len(lists) == 1 and rows != 1:
        lists = lists._broadcast(rows)
    if len(lists) != rows:
        raise Error(
            "list.concat operands have different lengths: "
            + String(len(lists))
            + " and "
            + String(rows)
        )
    return lists^


struct _ConcatJob(Job):
    """One range of rows of a concatenation: counts in the first pass,
    offsets and positions into the stacked children in the second."""

    var views: List[_View]
    var shifts: List[Int]
    var lo: Int
    var hi: Int
    var offsets: Int
    var positions: Int
    var filling: Bool
    var base: Int
    var total: Int

    def __init__(
        out self,
        views: List[_View],
        shifts: List[Int],
        lo: Int,
        hi: Int,
        offsets: Int,
    ):
        self.views = views.copy()
        self.shifts = shifts.copy()
        self.lo = lo
        self.hi = hi
        self.offsets = offsets
        self.positions = 0
        self.filling = False
        self.base = 0
        self.total = 0

    def run(mut self) raises:
        var offsets = Pointer[Int64, MutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        if not self.filling:
            var total = 0
            for i in range(self.lo, self.hi):
                var count = 0
                var valid = True
                for v in range(len(self.views)):
                    valid = valid and self.views[v].valid(i)
                    count += self.views[v].count(i)
                count = count if valid else 0
                offsets[unsafe_offset=i + 1] = Int64(count)
                total += count
            self.total = total
            return
        var out = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.positions)
        var at = self.base
        for i in range(self.lo, self.hi):
            var produced = Int(offsets[unsafe_offset=i + 1])
            offsets[unsafe_offset=i + 1] = Int64(at + produced)
            if produced == 0:
                continue
            for v in range(len(self.views)):
                var start = self.views[v].start(i)
                var shift = self.shifts[v]
                for j in range(self.views[v].count(i)):
                    out[unsafe_offset=at] = shift + start + j
                    at += 1


def list_concat(operands: List[Series], dtype: DataType) raises -> Series:
    """Row-wise concatenation of list (or element) operands into `dtype`
    lists; a null list operand makes the row null."""
    var inner = dtype.inner()
    var rows = 0
    for operand in operands:
        rows = max(rows, len(operand))
    var columns = List[ListColumn](capacity=len(operands))
    for operand in operands:
        columns.append(_as_lists(operand, inner, rows))
    if len(columns) == 1:
        return Series("", columns[0].copy())
    var views = List[_View](capacity=len(columns))
    var shifts = List[Int](capacity=len(columns))
    var children = List[Series](capacity=len(columns))
    var stored = 0
    var all_valid = True
    for column in columns:
        views.append(_View(column))
        shifts.append(stored)
        var child = _elements(column)
        stored += len(child)
        all_valid = all_valid and column.null_count() == 0
        children.append(child^)
    var offsets = List[Int64](length=rows + 1, fill=0)
    var workers = min(worker_count(stored), max(rows, 1))
    var jobs = List[_ConcatJob](capacity=workers)
    for w in range(workers):
        jobs.append(
            _ConcatJob(
                views,
                shifts,
                rows * w // workers,
                rows * (w + 1) // workers,
                Int(offsets.unsafe_ptr()),
            )
        )
    if rows > 0:
        _run(jobs)
    var total = 0
    for w in range(len(jobs)):
        jobs[w].filling = True
        jobs[w].base = total
        total += jobs[w].total
    var positions = List[Int](unsafe_uninit_length=total)
    for w in range(len(jobs)):
        jobs[w].positions = Int(positions.unsafe_ptr())
    if rows > 0:
        _run(jobs)
    var name = children[0].name()
    var stacked = Series._from_chunks(children).rechunk()
    var elements = stacked.take(positions).renamed(name)
    var bits = List[UInt8]()
    if not all_valid:
        var valid = List[Bool](length=rows, fill=True)
        for column in columns:
            for i in range(rows):
                if not column._valid(i):
                    valid[i] = False
        bits = _pack(valid)
    _ = columns^
    return Series("", ListColumn(offsets^, elements^, bits^))


def _pack(valid: List[Bool]) -> List[UInt8]:
    var bits = List[UInt8](length=(len(valid) + 7) // 8, fill=0)
    for i in range(len(valid)):
        if valid[i]:
            bits[i // 8] |= UInt8(1) << UInt8(i % 8)
    return bits^


# --- Entry -----------------------------------------------------------------


def list_transform(node: Node, lists: ListColumn) raises -> Series:
    """Evaluate a one-input list transform (see `is_list_transform`)."""
    var op = node.op
    if op == LIST_SORT or op == LIST_UNIQUE or op == LIST_N_UNIQUE:
        return _compared(lists, op, Int(node.integer))
    if op == LIST_DIFF:
        return _diff(lists, Int(node.integer))
    if op == LIST_GATHER_EVERY:
        if node.integer <= 0:
            raise Error("list.gather_every needs a positive step")
        if node.min_count < 0:
            raise Error("list.gather_every needs a nonnegative offset")
    if op == LIST_SLICE and node.min_count < -1:
        raise Error("list.slice needs a nonnegative length")
    var indices = List[Int]()
    if op == LIST_GATHER and node.min_count > 0:
        for item in split_names(node.text):
            indices.append(Int(item))
    var chosen = _choose(lists, op, Int(node.integer), node.min_count, indices)
    return _gathered(lists, chosen^)
