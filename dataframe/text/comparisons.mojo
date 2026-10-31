"""Ordinary String comparisons over owned physical spans without merging."""
from dataframe.series import Series
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.string import StringColumn
from dataframe.parallel import Job, run_jobs, worker_count
from dataframe.expr import EQ, NE, LT, LE, GT, GE
from dataframe.text.bytes import _Bytes, _same_bytes, _order_bytes
from dataframe.text.predicates import compare_with_literal
from std.memory import ArcPointer, Pointer


def _ready_strings(input: Series) raises -> Series:
    if input.is_chunked():
        if input._chunked.value()[].cache[].state.load() == 2:
            return input.rechunk()
    return input.copy()


def _string_spans(input: Series) raises -> ArcPointer[List[_Bytes]]:
    var spans = List[_Bytes]()
    if input.is_chunked():
        for array in input._chunked.value()[].arrays:
            spans.append(_Bytes(array[StringColumn]))
    else:
        spans.append(_Bytes(input._data[StringColumn]))
    return ArcPointer(spans^)


struct _StringComparisonJob[op: Int](Job):
    """Each job owns source handles and writes disjoint global bitmap bytes."""

    var left: ArcPointer[List[_Bytes]]
    var right: ArcPointer[List[_Bytes]]
    var left_scalar: Bool
    var right_scalar: Bool
    var nulls: Bool
    var values: Int
    var valid: Int
    var first: Int
    var last: Int

    def __init__(
        out self,
        left: ArcPointer[List[_Bytes]],
        right: ArcPointer[List[_Bytes]],
        left_scalar: Bool,
        right_scalar: Bool,
        nulls: Bool,
        values: Int,
        valid: Int,
        first: Int,
        last: Int,
    ):
        self.left = left
        self.right = right
        self.left_scalar = left_scalar
        self.right_scalar = right_scalar
        self.nulls = nulls
        self.values = values
        self.valid = valid
        self.first = first
        self.last = last

    def run(mut self) raises:
        if self.first == self.last:
            return
        var a = self.left[].unsafe_ptr()
        var b = self.right[].unsafe_ptr()
        var abase = 0
        var bbase = 0
        var aend = len(a[].column)
        var bend = len(b[].column)
        if not self.left_scalar:
            while aend <= self.first:
                abase = aend
                a = a.unsafe_offset(1)
                aend += len(a[].column)
        if not self.right_scalar:
            while bend <= self.first:
                bbase = bend
                b = b.unsafe_offset(1)
                bend += len(b[].column)
        var output = Pointer[List[UInt8], MutAnyOrigin](
            unsafe_from_address=self.values
        )[].unsafe_ptr()
        var validity = Pointer[List[UInt8], MutAnyOrigin](
            unsafe_from_address=self.valid
        )[].unsafe_ptr()
        var row = self.first
        while row < self.last:
            var end = min(row + 8, self.last)
            var data_byte = UInt8(0)
            var valid_byte = UInt8(0)
            var at = row
            while at < end:
                if not self.left_scalar:
                    while at >= aend:
                        abase = aend
                        a = a.unsafe_offset(1)
                        aend += len(a[].column)
                if not self.right_scalar:
                    while at >= bend:
                        bbase = bend
                        b = b.unsafe_offset(1)
                        bend += len(b[].column)
                var stop = min(
                    end,
                    min(
                        end if self.left_scalar else aend,
                        end if self.right_scalar else bend,
                    ),
                )
                for i in range(at, stop):
                    var ai = 0 if self.left_scalar else i - abase
                    var bi = 0 if self.right_scalar else i - bbase
                    if self.nulls and not (
                        a[].column._valid(ai) and b[].column._valid(bi)
                    ):
                        continue
                    valid_byte |= UInt8(1) << UInt8(i - row)
                    var x = a[].get(ai)
                    var y = b[].get(bi)
                    var hit: Bool
                    comptime if Self.op == EQ:
                        hit = _same_bytes(x, y)
                    elif Self.op == NE:
                        hit = not _same_bytes(x, y)
                    elif Self.op == LT:
                        hit = _order_bytes(x, y) < 0
                    elif Self.op == LE:
                        hit = _order_bytes(x, y) <= 0
                    elif Self.op == GT:
                        hit = _order_bytes(x, y) > 0
                    else:
                        hit = _order_bytes(x, y) >= 0
                    if hit:
                        data_byte |= UInt8(1) << UInt8(i - row)
                at = stop
            output.unsafe_offset(row // 8)[] = data_byte
            if self.nulls:
                validity.unsafe_offset(row // 8)[] = valid_byte
            row = end


def compare_chunked_strings[
    op: Int
](left: Series, right: Series) raises -> Series:
    """Caller retains logical/physical dispatch; observation masks are ignored.

    Preserve the contiguous String kernel's valid literal shortcuts before
    its length check. Null scalars use the general conjunction-of-validity.
    """
    var a = _ready_strings(left)
    var b = _ready_strings(right)
    if len(b) == 1 and len(a) != 1 and b._data[StringColumn]._valid(0):
        return compare_with_literal(a, op, b._data[StringColumn]._get(0), False)
    if len(a) == 1 and len(b) != 1 and a._data[StringColumn]._valid(0):
        return compare_with_literal(b, op, a._data[StringColumn]._get(0), True)
    var na = len(a)
    var nb = len(b)
    if na != nb and na != 1 and nb != 1:
        raise Error("Incompatible expression lengths")
    var n = 0 if na == 0 or nb == 0 else max(na, nb)
    var values = List[UInt8](unsafe_uninit_length=(n + 7) // 8)
    if n == 0:
        return Series(
            "", BoolColumn(values=values^, bits=List[UInt8](), length=0)
        )
    if (na == 1 and not a._data[StringColumn]._valid(0)) or (
        nb == 1 and not b._data[StringColumn]._valid(0)
    ):
        for i in range(len(values)):
            values[i] = 0
        return Series(
            "",
            BoolColumn(
                values=values^,
                bits=List[UInt8](length=(n + 7) // 8, fill=0),
                length=n,
            ),
        )
    var nulls = a.null_count() > 0 or b.null_count() > 0
    var valid = List[UInt8](length=(n + 7) // 8, fill=0) if nulls else List[
        UInt8
    ]()
    var left_spans = _string_spans(a)
    var right_spans = _string_spans(b)
    var workers = worker_count(n)
    var jobs = List[_StringComparisonJob[op]](capacity=workers)
    for w in range(workers):
        var first = (n * w // workers) // 8 * 8
        var last = n if w == workers - 1 else (n * (w + 1) // workers) // 8 * 8
        jobs.append(
            _StringComparisonJob[op](
                left_spans,
                right_spans,
                na == 1,
                nb == 1,
                nulls,
                Int(Pointer(to=values)),
                Int(Pointer(to=valid)),
                first,
                last,
            )
        )
    if workers == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    return Series("", BoolColumn(values=values^, bits=valid^, length=n))
