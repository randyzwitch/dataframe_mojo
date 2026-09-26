"""Kernels for list and struct expressions: str.split, the .list namespace,
and struct field access. Nulls propagate: a null list row gives a null
result, and a null struct row makes every field null."""
from .bool_column import BoolColumn
from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .expr import (
    LIST_CONTAINS,
    LIST_GET,
    LIST_JOIN,
    LIST_LEN,
    LIST_MAX,
    LIST_MEAN,
    LIST_MIN,
    LIST_SUM,
    STR_SPLIT,
    STRUCT_FIELD,
    Node,
)
from .nested_column import ListColumn
from .series import Series
from .string_column import StringBuilder, StringColumn


def nested_op(
    node: Node, input: Series, input_dtype: DataType
) raises -> Series:
    if input.is_chunked():
        return nested_op(node, input.rechunk(), input_dtype)
    if node.op == STR_SPLIT:
        return _split(input, node.text, node.integer == 1)
    if node.op == STRUCT_FIELD:
        return _struct_field(input, node.text)
    var lists = input.list_column()
    if node.op == LIST_LEN:
        var lengths = lists.lengths()
        return Series("", Column[Int64](lengths^, lists.validity()))
    if node.op == LIST_GET:
        return _get(lists, Int(node.integer))
    if node.op == LIST_CONTAINS:
        return _contains(lists, node)
    if node.op == LIST_JOIN:
        return _join(lists, node.text)
    return _reduce(lists, node.op)


def _split(input: Series, by: String, inclusive: Bool) raises -> Series:
    """One list of parts per string; a null string gives a null list."""
    if by == "":
        raise Error("str.split needs a non-empty separator")
    ref column = input._data[StringColumn]
    var n = len(column)
    var offsets = List[Int64](capacity=n + 1)
    offsets.append(0)
    var parts = StringBuilder(n, column._value_bytes())
    var valid = List[Bool](capacity=n)
    var total = 0
    for i in range(n):
        if not column._valid(i):
            valid.append(False)
            offsets.append(Int64(total))
            continue
        valid.append(True)
        var text = String(column._get(i))
        var pieces = text.split(by)
        for k in range(len(pieces)):
            if inclusive and k + 1 < len(pieces):
                parts.append(String(pieces[k]) + by)
            else:
                parts.append(String(pieces[k]))
            total += 1
        offsets.append(Int64(total))
    var child = Series("item", parts^.finish())
    return Series("", ListColumn(offsets^, child^, _bits_of(valid)))


def _bits_of(valid: List[Bool]) -> List[UInt8]:
    for flag in valid:
        if not flag:
            return _pack(valid)
    return List[UInt8]()


def _pack(valid: List[Bool]) -> List[UInt8]:
    var bits = List[UInt8](length=(len(valid) + 7) // 8, fill=0)
    for i in range(len(valid)):
        if valid[i]:
            bits[i // 8] |= UInt8(1) << UInt8(i % 8)
    return bits^


def _get(lists: ListColumn, index: Int) raises -> Series:
    """Element `index` of each list (negative counts from the end); null
    when the list is null or the index is out of range."""
    var rows = List[Int](capacity=len(lists))
    for i in range(len(lists)):
        var count = lists.element_count(i)
        var at = index + count if index < 0 else index
        if not lists._valid(i) or at < 0 or at >= count:
            rows.append(-1)
        else:
            rows.append(lists._start(i) + at)
    return lists.child().take_or_null(rows).renamed("")


def _contains(lists: ListColumn, node: Node) raises -> Series:
    """Whether each list holds the literal (null list gives null; a null
    element never matches)."""
    var child = lists.child()
    var flags = List[Bool](capacity=len(lists))
    var valid = lists.validity()
    if node.min_count == 0:
        var column = child.string()
        for i in range(len(lists)):
            var found = False
            for r in range(lists._start(i), lists._end(i)):
                if column._valid(r) and column._get(r) == node.text:
                    found = True
                    break
            flags.append(found)
        return Series("", BoolColumn(flags^, valid))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if child._data.isa[Column[Scalar[D]]]():
            ref column = child._data[Column[Scalar[D]]]
            var target: Scalar[D]
            comptime if D.is_floating_point():
                target = Scalar[D](node.floating)
            else:
                target = node.integer.cast[D]()
            for i in range(len(lists)):
                var found = False
                for r in range(lists._start(i), lists._end(i)):
                    if column._valid(r) and column._get(r) == target:
                        found = True
                        break
                flags.append(found)
            return Series("", BoolColumn(flags^, valid))
    raise Error("list.contains needs a string or numeric list")


def _join(lists: ListColumn, separator: String) raises -> Series:
    """Join string elements with a separator, skipping null elements."""
    var column = lists.child().string()
    var out = StringBuilder(len(lists))
    for i in range(len(lists)):
        if not lists._valid(i):
            out.append_null()
            continue
        var text = String()
        var first = True
        for r in range(lists._start(i), lists._end(i)):
            if not column._valid(r):
                continue
            if not first:
                text += separator
            text += String(column._get(r))
            first = False
        out.append(text)
    return Series("", out^.finish())


def _reduce(lists: ListColumn, op: Int) raises -> Series:
    """sum/min/max/mean over each list's valid elements; a list with no
    valid element gives null (sum gives 0 for an empty valid list, as
    Polars does, but null for a null list)."""
    var child = lists.child()
    var n = len(lists)
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if child._data.isa[Column[Scalar[D]]]():
            ref column = child._data[Column[Scalar[D]]]
            var valid = List[Bool](capacity=n)
            if op == LIST_MEAN:
                var means = List[Float64](capacity=n)
                for i in range(n):
                    var total = Float64(0)
                    var count = 0
                    for r in range(lists._start(i), lists._end(i)):
                        if column._valid(r):
                            total += Float64(column._get(r))
                            count += 1
                    valid.append(lists._valid(i) and count > 0)
                    means.append(total / Float64(count) if count > 0 else 0)
                return Series("", Column[Float64](means^, valid))
            if op == LIST_SUM:
                comptime if D == DType.float32 or D == DType.float64:
                    var sums = List[Scalar[D]](capacity=n)
                    for i in range(n):
                        var total = Scalar[D](0)
                        for r in range(lists._start(i), lists._end(i)):
                            if column._valid(r):
                                total += column._get(r)
                        valid.append(lists._valid(i))
                        sums.append(total)
                    return Series("", Column[Scalar[D]](sums^, valid))
                else:
                    var sums = List[Int64](capacity=n)
                    for i in range(n):
                        var total = Int64(0)
                        for r in range(lists._start(i), lists._end(i)):
                            if column._valid(r):
                                total += column._get(r).cast[DType.int64]()
                        valid.append(lists._valid(i))
                        sums.append(total)
                    return Series("", Column[Int64](sums^, valid)).cast(
                        DataType.of(D).sum_type()
                    )
            var extremes = List[Scalar[D]](capacity=n)
            for i in range(n):
                var found = False
                var best = Scalar[D](0)
                for r in range(lists._start(i), lists._end(i)):
                    if not column._valid(r):
                        continue
                    var value = column._get(r)
                    if not found:
                        best = value
                        found = True
                    elif op == LIST_MIN:
                        if value < best:
                            best = value
                    elif value > best:
                        best = value
                valid.append(lists._valid(i) and found)
                extremes.append(best)
            return Series("", Column[Scalar[D]](extremes^, valid))
    raise Error("list reductions need a numeric list")


def _struct_field(input: Series, name: String) raises -> Series:
    """One field, null wherever the struct row itself is null."""
    var structs = input.struct_column()
    var field = structs.field(name)
    if structs.null_count() == 0:
        return field.renamed("")
    var rows = List[Int](capacity=len(structs))
    for i in range(len(structs)):
        rows.append(i if structs._valid(i) else -1)
    return field.take_or_null(rows).renamed("")
