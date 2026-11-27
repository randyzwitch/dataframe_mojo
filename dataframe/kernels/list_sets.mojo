"""Set operations between two lists in the same row: union, intersection,
difference, and symmetric difference.

Both lists are first concatenated row by row (the left list's elements,
then the right's), so a null list on either side makes the row null. Each
distinct element then keeps its first position in that row, which is in
the left list whenever the left list holds it, and the operation keeps the
distinct elements by where they occur: union keeps all of them,
intersection those in both lists, difference those only in the left one,
and symmetric difference those in exactly one. A null element is one
value, equal to another null. The result therefore lists the left list's
elements first, in their order, then the right list's.

Numeric and string elements are compared in place, one row at a time, as
`list.unique` does. Other element types, and a row long enough to be split
across workers by itself, go through the row-key encoder keyed by (row,
element).
"""
from std.memory import Pointer
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.expr import (
    LIST_SET_DIFFERENCE,
    LIST_SET_INTERSECTION,
    LIST_SET_SYMMETRIC_DIFFERENCE,
    LIST_SET_UNION,
)
from dataframe.kernels.list_transforms import (
    _BytesOrder,
    _Chosen,
    _NumberOrder,
    _Order,
    _Present,
    _View,
    _as_lists,
    _gathered,
    _long,
    _ranges,
    _run,
    _sort_positions,
    _SMALL,
    list_concat,
    row_keys,
)
from dataframe.parallel import Job
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.storage.nested import ListColumn
from dataframe.storage.string import StringColumn

# Where a distinct element occurs: bit 0 in the left list, bit 1 in the
# right list.
comptime _LEFT = 1
comptime _RIGHT = 2


@always_inline
def _keeps(op: Int, sides: Int) -> Bool:
    """Whether the operation keeps a distinct element found on `sides`."""
    if op == LIST_SET_UNION:
        return True
    if op == LIST_SET_INTERSECTION:
        return sides == _LEFT | _RIGHT
    if op == LIST_SET_DIFFERENCE:
        return sides == _LEFT
    return sides != _LEFT | _RIGHT


struct _SetJob[O: _Order](Job):
    """One range of rows of the concatenated lists: append each kept
    element's position to `kept` and each row's count to offsets[i + 1].
    Rows long enough for several workers are left in `deferred`."""

    var op: Int
    var order: Self.O
    var view: _View
    var present: _Present
    var split: Int
    var lo: Int
    var hi: Int
    var offsets: Int
    var kept: List[Int]
    var deferred: List[Int]

    def __init__(
        out self,
        op: Int,
        order: Self.O,
        view: _View,
        present: _Present,
        split: Int,
        lo: Int,
        hi: Int,
        offsets: Int,
    ):
        self.op = op
        self.order = order.copy()
        self.view = view.copy()
        self.present = present.copy()
        self.split = split
        self.lo = lo
        self.hi = hi
        self.offsets = offsets
        self.kept = List[Int]()
        self.deferred = List[Int]()

    def run(mut self) raises:
        var offsets = Pointer[Int64, MutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        # Elements of row i's left list (the rest are the right list's).
        var split = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.split)
        var firsts = List[Int]()
        var sides = List[Int]()
        var scratch = List[Int]()
        var marks = List[Int]()
        for i in range(self.lo, self.hi):
            var count = self.view.count(i)
            if count == 0:
                offsets[unsafe_offset=i + 1] = 0
                continue
            if _long(count):
                self.deferred.append(i)
                continue
            var start = self.view.start(i)
            var boundary = start + split[unsafe_offset=i]
            var row_first = len(self.kept)
            if count <= _SMALL:
                firsts.clear()
                sides.clear()
                for r in range(start, start + count):
                    var side = _LEFT if r < boundary else _RIGHT
                    var here = self.present.at(r)
                    var found = -1
                    for k in range(len(firsts)):
                        var q = firsts[k]
                        var there = self.present.at(q)
                        if here == there and (
                            not here or self.order.same(q, r)
                        ):
                            found = k
                            break
                    if found >= 0:
                        sides[found] |= side
                    else:
                        firsts.append(r)
                        sides.append(side)
                for k in range(len(firsts)):
                    if _keeps(self.op, sides[k]):
                        self.kept.append(firsts[k])
            else:
                # Sort the present positions; in each run of equal elements
                # the smallest position stands for the run, marked with the
                # sides the run touches. Nulls form one more run.
                scratch.clear()
                marks.clear()
                marks.resize(count, 0)
                var first_null = -1
                var null_sides = 0
                for r in range(start, start + count):
                    if self.present.at(r):
                        scratch.append(r)
                    else:
                        if first_null < 0:
                            first_null = r
                        null_sides |= _LEFT if r < boundary else _RIGHT
                if first_null >= 0:
                    marks[first_null - start] = null_sides
                var p = Pointer[Int, MutAnyOrigin](
                    unsafe_from_address=Int(scratch.unsafe_ptr())
                )
                _sort_positions(self.order, p, 0, len(scratch))
                var k = 0
                while k < len(scratch):
                    var smallest = scratch[k]
                    var touched = _LEFT if smallest < boundary else _RIGHT
                    var run_end = k + 1
                    while run_end < len(scratch) and self.order.same(
                        scratch[k], scratch[run_end]
                    ):
                        var r = scratch[run_end]
                        smallest = min(smallest, r)
                        touched |= _LEFT if r < boundary else _RIGHT
                        run_end += 1
                    marks[smallest - start] = touched
                    k = run_end
                for r in range(count):
                    if marks[r] != 0 and _keeps(self.op, marks[r]):
                        self.kept.append(start + r)
            offsets[unsafe_offset=i + 1] = Int64(len(self.kept) - row_first)


def _encoded_rows(
    lists: ListColumn, split: List[Int], rows: List[Int], op: Int
) raises -> List[List[Int]]:
    """The kept positions of each of `rows`, through the row-key encoder
    keyed by (row, element)."""
    var view = _View(lists)
    var owners = List[Int64]()
    var positions = List[Int]()
    for k in range(len(rows)):
        var i = rows[k]
        var start = view.start(i)
        for r in range(start, start + view.count(i)):
            owners.append(Int64(k))
            positions.append(r)
    var result = List[List[Int]](capacity=len(rows))
    for _ in range(len(rows)):
        result.append(List[Int]())
    if len(positions) == 0:
        return result^
    var elements = lists.child().take(positions)
    var keys = row_keys(
        [Series("row", Column[Int64](owners.copy())), elements^]
    )
    var sides = List[Int](length=keys.count(), fill=0)
    for j in range(len(positions)):
        var i = rows[Int(owners[j])]
        var left = positions[j] < view.start(i) + split[i]
        sides[keys.ids[j]] |= _LEFT if left else _RIGHT
    # Representatives are each id's first row, and rows are in position
    # order, so a sorted walk of them lists each row's kept elements.
    var first = List[Bool](length=len(positions), fill=False)
    for g in range(keys.count()):
        if _keeps(op, sides[g]):
            first[keys.representatives[g]] = True
    for j in range(len(positions)):
        if first[j]:
            result[Int(owners[j])].append(positions[j])
    return result^


def _combined[
    O: _Order
](
    order: O,
    lists: ListColumn,
    elements: Series,
    split: List[Int],
    op: Int,
) raises -> Series:
    var n = len(lists)
    var view = _View(lists)
    var present = _Present(elements)
    var offsets = List[Int64](length=n + 1, fill=0)
    var bounds = _ranges(view, n, len(elements))
    var jobs = List[_SetJob[O]](capacity=len(bounds) - 1)
    for w in range(len(bounds) - 1):
        jobs.append(
            _SetJob[O](
                op,
                order,
                view,
                present,
                Int(split.unsafe_ptr()),
                bounds[w],
                bounds[w + 1],
                Int(offsets.unsafe_ptr()),
            )
        )
    if n > 0:
        _run(jobs)
    _ = present^
    var deferred = List[Int]()
    for w in range(len(jobs)):
        deferred.extend(Span(jobs[w].deferred))
    var long_kept = _encoded_rows(lists, split, deferred, op)
    for k in range(len(deferred)):
        offsets[deferred[k] + 1] = Int64(len(long_kept[k]))
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


def list_set(operands: List[Series], dtype: DataType, op: Int) raises -> Series:
    """A set operation between the two list operands, row by row, giving
    `dtype` lists."""
    var rows = max(len(operands[0]), len(operands[1]))
    var left = _as_lists(operands[0], dtype.inner(), rows)
    var left_view = _View(left)
    var split = List[Int](unsafe_uninit_length=rows)
    for i in range(rows):
        split[i] = left_view.count(i)
    var lists = list_concat(operands, dtype).list_column()
    var elements = lists.child()
    if elements.is_chunked():
        elements = elements.rechunk()
    var n = len(lists)
    if not elements.dtype().is_categorical():
        if elements._data.isa[StringColumn]():
            return _combined(
                _BytesOrder(elements._data[StringColumn]),
                lists,
                elements,
                split,
                op,
            )
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if elements._data.isa[Column[Scalar[D]]]():
                ref column = elements._data[Column[Scalar[D]]]
                var order = _NumberOrder[D](Int(column.unsafe_values()))
                var result = _combined(order, lists, elements, split, op)
                _ = elements^
                return result^
    var every = List[Int](capacity=n)
    for i in range(n):
        every.append(i)
    var kept = _encoded_rows(lists, split, every, op)
    var offsets = List[Int64](length=n + 1, fill=0)
    var positions = List[Int]()
    for i in range(n):
        positions.extend(Span(kept[i]))
        offsets[i + 1] = Int64(len(positions))
    return _gathered(lists, _Chosen(offsets^, positions^, False))
