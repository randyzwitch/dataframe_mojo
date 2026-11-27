"""List reductions: median, std, var, any, all, arg_min, arg_max,
count_matches, and contains with an expression. Each gives one value per
row from that row's elements, and a null row gives null.

Rows are split into contiguous ranges holding about the same number of
elements, one range per worker (see `list_transforms`), and each worker
writes its rows' results into shared output arrays.
"""
from std.math import sqrt
from std.memory import Pointer, unsafe_memcmp
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.expr import (
    LIST_ALL,
    LIST_ANY,
    LIST_ARG_MAX,
    LIST_ARG_MIN,
    LIST_COUNT_MATCHES,
    LIST_HAS,
    LIST_MEDIAN,
    LIST_STD,
    LIST_VAR,
    Node,
)
from dataframe.kernels.list_transforms import (
    _BytesOrder,
    _IDENTITY,
    _NumberOrder,
    _Order,
    _Present,
    _View,
    _choose,
    _elements,
    _ranges,
    _run,
    _sort_positions,
    row_keys,
    list_concat_inner,
    list_op_name,
)
from dataframe.parallel import Job
from dataframe.series import Series
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.storage.nested import ListColumn
from dataframe.storage.string import StringColumn
from dataframe.text.bytes import _Bytes


def list_match_check(op: Int, inner: DataType, item: DataType) raises:
    """Raise unless item can be compared with the list's elements: the same
    type, a null, or two numeric types (compared by value)."""
    if item == inner or item == DataType.NULL:
        return
    if (
        (inner.is_integer() or inner.is_float())
        and (item.is_integer() or item.is_float())
        and not inner.is_categorical()
        and not item.is_categorical()
    ):
        return
    raise Error(
        list_op_name(op)
        + " needs an item of the element type "
        + inner.name()
        + ", found "
        + item.name()
    )


# --- median, std, var ------------------------------------------------------


struct _MomentJob(Job):
    """median, std, or var of each row of a range over Float64 elements."""

    var op: Int
    var ddof: Int
    var view: _View
    var present: _Present
    var values: Int
    var lo: Int
    var hi: Int
    var dest: Int
    var ok: Int

    def __init__(
        out self,
        op: Int,
        ddof: Int,
        view: _View,
        present: _Present,
        values: Int,
        lo: Int,
        hi: Int,
        dest: Int,
        ok: Int,
    ):
        self.op = op
        self.ddof = ddof
        self.view = view.copy()
        self.present = present.copy()
        self.values = values
        self.lo = lo
        self.hi = hi
        self.dest = dest
        self.ok = ok

    def run(mut self) raises:
        var values = Pointer[Float64, MutAnyOrigin](
            unsafe_from_address=self.values
        )
        var out = Pointer[Float64, MutAnyOrigin](unsafe_from_address=self.dest)
        var ok = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.ok)
        var order = _NumberOrder[DType.float64](self.values)
        var scratch = List[Int]()
        for i in range(self.lo, self.hi):
            out[unsafe_offset=i] = 0
            ok[unsafe_offset=i] = False
            var start = self.view.start(i)
            var count = self.view.count(i)
            scratch.clear()
            for r in range(start, start + count):
                if self.present.at(r):
                    scratch.append(r)
            var m = len(scratch)
            if self.op == LIST_MEDIAN:
                if m == 0:
                    continue
                var p = Pointer[Int, MutAnyOrigin](
                    unsafe_from_address=Int(scratch.unsafe_ptr())
                )
                # NaN sorts last, so it lands in the middle (and makes the
                # median NaN) only when it is that common.
                _sort_positions(order, p, 0, m)
                var upper = values[unsafe_offset=scratch[m // 2]]
                if m % 2 == 1:
                    out[unsafe_offset=i] = upper
                else:
                    out[unsafe_offset=i] = (
                        values[unsafe_offset=scratch[m // 2 - 1]] + upper
                    ) / 2
                ok[unsafe_offset=i] = True
                continue
            if m <= self.ddof:
                continue
            var total = Float64(0)
            for k in range(m):
                total += values[unsafe_offset=scratch[k]]
            var mean = total / Float64(m)
            var squares = Float64(0)
            for k in range(m):
                var d = values[unsafe_offset=scratch[k]] - mean
                squares += d * d
            var variance = squares / Float64(m - self.ddof)
            out[unsafe_offset=i] = (
                sqrt(variance) if self.op == LIST_STD else variance
            )
            ok[unsafe_offset=i] = True


def _moments(lists: ListColumn, op: Int, ddof: Int) raises -> Series:
    var n = len(lists)
    var elements = _elements(lists)
    if elements.dtype() != DataType.FLOAT64:
        elements = elements.cast(DataType.FLOAT64)
    if elements.is_chunked():
        elements = elements.rechunk()
    var view = _View(lists)
    var present = _Present(elements)
    ref column = elements._data[Column[Float64]]
    var out = List[Float64](unsafe_uninit_length=n)
    var ok = List[Bool](unsafe_uninit_length=n)
    var bounds = _ranges(view, n, len(elements))
    var jobs = List[_MomentJob](capacity=len(bounds) - 1)
    for w in range(len(bounds) - 1):
        jobs.append(
            _MomentJob(
                op,
                ddof,
                view,
                present,
                Int(column.unsafe_values()),
                bounds[w],
                bounds[w + 1],
                Int(out.unsafe_ptr()),
                Int(ok.unsafe_ptr()),
            )
        )
    if n > 0:
        _run(jobs)
    _ = present^
    _ = elements^
    return Series("", Column[Float64](out^, ok))


# --- any, all --------------------------------------------------------------


struct _TruthJob(Job):
    """any or all of each row of a range over Bool elements."""

    var all: Bool
    var view: _View
    var present: _Present
    var values: Int
    var bit0: Int
    var lo: Int
    var hi: Int
    var dest: Int
    var ok: Int

    def __init__(
        out self,
        all: Bool,
        view: _View,
        present: _Present,
        values: Int,
        bit0: Int,
        lo: Int,
        hi: Int,
        dest: Int,
        ok: Int,
    ):
        self.all = all
        self.view = view.copy()
        self.present = present.copy()
        self.values = values
        self.bit0 = bit0
        self.lo = lo
        self.hi = hi
        self.dest = dest
        self.ok = ok

    def run(mut self) raises:
        var bytes = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.values
        )
        var out = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.dest)
        var ok = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.ok)
        for i in range(self.lo, self.hi):
            ok[unsafe_offset=i] = self.view.valid(i)
            var start = self.view.start(i)
            # any looks for a true element, all for a false one.
            var result = self.all
            for r in range(start, start + self.view.count(i)):
                if not self.present.at(r):
                    continue
                var bit = self.bit0 + r
                var value = (
                    (bytes[unsafe_offset=bit >> 3] >> UInt8(bit & 7)) & 1
                ) != 0
                if value != self.all:
                    result = not self.all
                    break
            out[unsafe_offset=i] = result


def _truth(lists: ListColumn, all: Bool) raises -> Series:
    var n = len(lists)
    var elements = _elements(lists)
    if elements.is_chunked():
        elements = elements.rechunk()
    var view = _View(lists)
    var present = _Present(elements)
    ref column = elements._data[BoolColumn]
    var out = List[Bool](unsafe_uninit_length=n)
    var ok = List[Bool](unsafe_uninit_length=n)
    var bounds = _ranges(view, n, len(elements))
    var jobs = List[_TruthJob](capacity=len(bounds) - 1)
    for w in range(len(bounds) - 1):
        jobs.append(
            _TruthJob(
                all,
                view,
                present,
                Int(column._data.unsafe_ptr()),
                column._offset,
                bounds[w],
                bounds[w + 1],
                Int(out.unsafe_ptr()),
                Int(ok.unsafe_ptr()),
            )
        )
    if n > 0:
        _run(jobs)
    _ = present^
    _ = elements^
    return Series("", BoolColumn(out, ok))


# --- arg_min, arg_max ------------------------------------------------------


struct _ArgJob[O: _Order](Job):
    """The position of each row's smallest or largest valid element in a
    range; the first one on ties, and a number before NaN."""

    var order: Self.O
    var largest: Bool
    var view: _View
    var present: _Present
    var lo: Int
    var hi: Int
    var dest: Int
    var ok: Int

    def __init__(
        out self,
        order: Self.O,
        largest: Bool,
        view: _View,
        present: _Present,
        lo: Int,
        hi: Int,
        dest: Int,
        ok: Int,
    ):
        self.order = order.copy()
        self.largest = largest
        self.view = view.copy()
        self.present = present.copy()
        self.lo = lo
        self.hi = hi
        self.dest = dest
        self.ok = ok

    def run(mut self) raises:
        var out = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=self.dest)
        var ok = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.ok)
        for i in range(self.lo, self.hi):
            var start = self.view.start(i)
            var best = -1
            for j in range(self.view.count(i)):
                var r = start + j
                if not self.present.at(r):
                    continue
                if best < 0:
                    best = j
                    continue
                var held = start + best
                if self.largest:
                    # NaN orders last; prefer any number to it.
                    if self.order.nan(r):
                        continue
                    if self.order.nan(held) or self.order.less(held, r):
                        best = j
                elif self.order.less(r, held):
                    best = j
            out[unsafe_offset=i] = UInt32(max(best, 0))
            ok[unsafe_offset=i] = best >= 0


def _argument[
    O: _Order
](
    order: O, lists: ListColumn, elements: Series, largest: Bool
) raises -> Series:
    var n = len(lists)
    var view = _View(lists)
    var present = _Present(elements)
    var out = List[UInt32](unsafe_uninit_length=n)
    var ok = List[Bool](unsafe_uninit_length=n)
    var bounds = _ranges(view, n, len(elements))
    var jobs = List[_ArgJob[O]](capacity=len(bounds) - 1)
    for w in range(len(bounds) - 1):
        jobs.append(
            _ArgJob[O](
                order,
                largest,
                view,
                present,
                bounds[w],
                bounds[w + 1],
                Int(out.unsafe_ptr()),
                Int(ok.unsafe_ptr()),
            )
        )
    if n > 0:
        _run(jobs)
    _ = present^
    return Series("", Column[UInt32](out^, ok))


def _arg(lists: ListColumn, largest: Bool) raises -> Series:
    var elements = _elements(lists)
    if elements.is_chunked():
        elements = elements.rechunk()
    if elements._data.isa[StringColumn]():
        return _argument(
            _BytesOrder(elements._data[StringColumn]), lists, elements, largest
        )
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if elements._data.isa[Column[Scalar[D]]]():
            ref column = elements._data[Column[Scalar[D]]]
            var order = _NumberOrder[D](Int(column.unsafe_values()))
            var result = _argument(order, lists, elements, largest)
            _ = elements^
            return result^
    # Wide (Int128) decimals have no in-place order here yet.
    raise Error(
        list_op_name(LIST_ARG_MAX if largest else LIST_ARG_MIN)
        + " does not support "
        + elements.dtype().name()
        + " elements"
    )


# --- count_matches and contains --------------------------------------------


struct _MatchJob[D: DType](Job):
    """Count the elements of each row of a range that equal one value (or
    that are null, when `null` is set)."""

    var view: _View
    var present: _Present
    var values: Int
    var target: Scalar[Self.D]
    var null: Bool
    var lo: Int
    var hi: Int
    var dest: Int

    def __init__(
        out self,
        view: _View,
        present: _Present,
        values: Int,
        target: Scalar[Self.D],
        null: Bool,
        lo: Int,
        hi: Int,
        dest: Int,
    ):
        self.view = view.copy()
        self.present = present.copy()
        self.values = values
        self.target = target
        self.null = null
        self.lo = lo
        self.hi = hi
        self.dest = dest

    def run(mut self) raises:
        var values = Pointer[Scalar[Self.D], MutAnyOrigin](
            unsafe_from_address=self.values
        )
        var out = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=self.dest)
        for i in range(self.lo, self.hi):
            var start = self.view.start(i)
            var found = 0
            for r in range(start, start + self.view.count(i)):
                var here = self.present.at(r)
                if self.null:
                    found += Int(not here)
                else:
                    found += Int(
                        here and values[unsafe_offset=r] == self.target
                    )
            out[unsafe_offset=i] = UInt32(found)


struct _TextMatchJob(Job):
    """Count the string elements of each row of a range equal to one
    value."""

    var view: _View
    var present: _Present
    var rows: _Bytes
    var target: String
    var lo: Int
    var hi: Int
    var dest: Int

    def __init__(
        out self,
        view: _View,
        present: _Present,
        column: StringColumn,
        target: String,
        lo: Int,
        hi: Int,
        dest: Int,
    ):
        self.view = view.copy()
        self.present = present.copy()
        self.rows = _Bytes(column)
        self.target = target
        self.lo = lo
        self.hi = hi
        self.dest = dest

    def run(mut self) raises:
        var out = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=self.dest)
        var wanted = self.target.as_bytes()
        var size = len(wanted)
        for i in range(self.lo, self.hi):
            var start = self.view.start(i)
            var found = 0
            for r in range(start, start + self.view.count(i)):
                if not self.present.at(r):
                    continue
                var row = self.rows.get(r)
                if len(row) == size and (
                    size == 0
                    or unsafe_memcmp(
                        Pointer[UInt8, ImmutAnyOrigin](
                            unsafe_from_address=Int(row.unsafe_ptr())
                        ),
                        Pointer[UInt8, ImmutAnyOrigin](
                            unsafe_from_address=Int(wanted.unsafe_ptr())
                        ),
                        size,
                    )
                    == 0
                ):
                    found += 1
            out[unsafe_offset=i] = UInt32(found)


def _literal_counts(
    lists: ListColumn, elements: Series, value: Series
) raises -> Optional[List[UInt32]]:
    """Per-row counts of one value among numeric or string elements, read
    in place; None for other element types."""
    var n = len(lists)
    var view = _View(lists)
    var present = _Present(elements)
    var counts = List[UInt32](unsafe_uninit_length=n)
    var bounds = _ranges(view, n, len(elements))
    var null = value.null_count() == 1
    if elements._data.isa[StringColumn]():
        if null:
            return None
        var target = String(value._data[StringColumn]._get(0))
        var jobs = List[_TextMatchJob](capacity=len(bounds) - 1)
        for w in range(len(bounds) - 1):
            jobs.append(
                _TextMatchJob(
                    view,
                    present,
                    elements._data[StringColumn],
                    target,
                    bounds[w],
                    bounds[w + 1],
                    Int(counts.unsafe_ptr()),
                )
            )
        if n > 0:
            _run(jobs)
        _ = present^
        return counts^
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if (
            elements._data.isa[Column[Scalar[D]]]()
            and value._data.isa[Column[Scalar[D]]]()
        ):
            ref column = elements._data[Column[Scalar[D]]]
            var target = Scalar[D](0)
            if not null:
                target = value._data[Column[Scalar[D]]]._get(0)
            var jobs = List[_MatchJob[D]](capacity=len(bounds) - 1)
            for w in range(len(bounds) - 1):
                jobs.append(
                    _MatchJob[D](
                        view,
                        present,
                        Int(column.unsafe_values()),
                        target,
                        null,
                        bounds[w],
                        bounds[w + 1],
                        Int(counts.unsafe_ptr()),
                    )
                )
            if n > 0:
                _run(jobs)
            _ = present^
            return counts^
    return None


def _matched_counts(
    lists: ListColumn, elements: Series, value: Series
) raises -> List[UInt32]:
    """Per-row counts of elements equal to the row's value (a null value
    matching null elements), for any element type: each element is
    compared with its row's value, broadcast to it."""
    var n = len(lists)
    var every = _choose(lists, _IDENTITY)
    var m = len(every.positions)
    var owners = List[Int](unsafe_uninit_length=m)
    for i in range(n):
        for j in range(Int(every.offsets[i]), Int(every.offsets[i + 1])):
            owners[j] = i if len(value) != 1 else 0
    if m == 0:
        return List[UInt32](length=n, fill=0)
    # Element j and its row's value get the same id exactly when they are
    # equal (two nulls included), as rows j and m + j of one stacked
    # column.
    var stacked = Series._from_chunks(
        [
            elements.take(every.positions).renamed("v"),
            value.take(owners).renamed("v"),
        ]
    ).rechunk()
    var keys = row_keys([stacked^])
    var counts = List[UInt32](length=n, fill=0)
    for i in range(n):
        var found = 0
        for j in range(Int(every.offsets[i]), Int(every.offsets[i + 1])):
            found += Int(keys.ids[j] == keys.ids[m + j])
        counts[i] = UInt32(found)
    return counts^


def list_match(left: Series, item: Series, op: Int) raises -> Series:
    """count_matches (UInt32) or contains (Bool) of each row's item value
    among its list's elements; a null list gives null."""
    var lists = left.list_column()
    var n = len(lists)
    var elements = _elements(lists)
    var value = item.rechunk() if item.is_chunked() else item.copy()
    if value.dtype() == DataType.NULL:
        value = value.cast(elements.dtype())
    if value.dtype() != elements.dtype():
        # Numbers of two types compare in the type that holds both.
        var common = list_concat_inner(elements.dtype(), value.dtype())
        if elements.dtype() != common:
            elements = elements.cast(common)
        if value.dtype() != common:
            value = value.cast(common)
    if elements.is_chunked():
        elements = elements.rechunk()
    if len(value) != 1 and len(value) != n:
        raise Error(
            list_op_name(op)
            + " item has "
            + String(len(value))
            + " rows for "
            + String(n)
            + " lists"
        )
    var counts: List[UInt32]
    var literal = Optional[List[UInt32]](None)
    if len(value) == 1 and not elements.dtype().is_categorical():
        literal = _literal_counts(lists, elements, value)
    if literal:
        counts = literal.take()
    else:
        counts = _matched_counts(lists, elements, value)
    var valid = List[Bool](unsafe_uninit_length=n)
    var view = _View(lists)
    for i in range(n):
        valid[i] = view.valid(i)
    if op == LIST_HAS:
        var flags = List[Bool](unsafe_uninit_length=n)
        for i in range(n):
            flags[i] = counts[i] > 0
        return Series("", BoolColumn(flags, valid))
    return Series("", Column[UInt32](counts^, valid))


# --- Entry -----------------------------------------------------------------


def list_reduce(node: Node, lists: ListColumn) raises -> Series:
    """Evaluate a one-input list reduction."""
    var op = node.op
    if op == LIST_MEDIAN or op == LIST_STD or op == LIST_VAR:
        return _moments(lists, op, node.min_count)
    if op == LIST_ANY or op == LIST_ALL:
        return _truth(lists, op == LIST_ALL)
    if op == LIST_ARG_MIN or op == LIST_ARG_MAX:
        return _arg(lists, op == LIST_ARG_MAX)
    raise Error(list_op_name(op) + " needs an item")
