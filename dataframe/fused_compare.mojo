"""The AND of a filter's column-versus-constant comparisons in one pass.

A filter like PDS-H q6's (a date range, a discount range and a quantity
bound: five comparisons on three columns) evaluated each comparison as its
own expression, each writing a full-length bitmap that the next AND read
back, about 2 ns a row a comparison. Here every comparison of a numeric
column against a constant of the column's own type runs over
cache-sized blocks of rows, and each block's bits are ANDed in a buffer that
stays in L1, so each column is read from memory once and one bitmap is
written. The combined bitmap is the selection for the next pipeline step.
Semantics are the comparison kernels' (`_compare_lanes`): IEEE for
floats, so a NaN compares false except with `!=`.
"""
from ._buffer import _BufferData
from std.memory import pack_bits

from .expr import node_children
from .binding import bind
from .bool_column import BoolColumn
from .column import Column
from .dtype import NUMERIC_DTYPES
from .execution import _numeric_literal, evaluate
from .expr import (
    AND,
    COL,
    EQ,
    GE,
    GT,
    IS_NOT_NULL,
    IS_NULL,
    LE,
    LIT_FLOAT,
    LIT_INT,
    LT,
    NE,
    OR,
    UNTYPED,
    Expr,
    subtree,
)
from .expr_kernels import _compare_lanes, _validity_bytes
from .parallel import Job, partitions, run_jobs, worker_count
from .series import Series

# Rows a block: each column's slice of a block stays in L1 (8 KB of an
# 8-byte column) while every comparison on it runs.
comptime _BLOCK_ROWS = 1024


@fieldwise_init
struct _Comparison(Copyable, Movable):
    var column: Series
    var op: Int
    # The constant, one row of the column's dtype.
    var value: Series
    # Whether any row is null: those compare false, which costs a pass
    # over the validity bits that an all-valid column skips.
    var nulls: Bool


@fieldwise_init
struct FusedComparisons(Movable):
    """The mask of the parts marked in `used` (empty when none is)."""

    var mask: List[UInt8]
    var used: List[Bool]


def _flip(op: Int) -> Int:
    if op == LT:
        return GT
    if op == LE:
        return GE
    if op == GT:
        return LT
    if op == GE:
        return LE
    return op


def _is_constant(expr: Expr, index: Int) -> Bool:
    var stack: List[Int] = [index]
    while len(stack) > 0:
        var i = stack.pop()
        ref node = expr._nodes[i]
        if node.op == COL:
            return False
        for child in node_children(node):
            stack.append(child)
    return True


def _comparison(
    part: Expr, columns: List[Series]
) raises -> Optional[_Comparison]:
    """The part as `column op constant`, when the column is numeric and
    the constant evaluates to one valid value of the column's exact
    dtype; None for any other part. A null row compares false, as a
    filter drops it."""
    var root = len(part._nodes) - 1
    ref node = part._nodes[root]
    var op = node.op
    if not (
        op == EQ or op == NE or op == LT or op == LE or op == GT or op == GE
    ):
        return None
    if node.left < 0 or node.right < 0:
        return None
    var column_at: Int
    var constant_at: Int
    if part._nodes[node.left].op == COL and _is_constant(part, node.right):
        column_at = node.left
        constant_at = node.right
    elif part._nodes[node.right].op == COL and _is_constant(part, node.left):
        column_at = node.right
        constant_at = node.left
        op = _flip(op)
    else:
        return None
    var name = part._nodes[column_at].text
    var found = -1
    for i in range(len(columns)):
        if columns[i].name() == name:
            found = i
    if found < 0:
        return None
    ref column = columns[found]
    if column.is_chunked():
        return None
    # A validity bitmap with no null in it (Parquet imports carry one) is
    # as good as none: every row's value is read.
    var numeric = False
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if column._data.isa[Column[Scalar[D]]]():
            numeric = True
    if not numeric:
        return None
    # A typed numeric literal (what the optimizer folds a constant cast
    # into) is read straight from its node; anything else is bound and
    # evaluated. Filters build these comparisons for every morsel.
    var value: Series
    ref constant = part._nodes[constant_at]
    if (
        (constant.op == LIT_INT or constant.op == LIT_FLOAT)
        and constant.text != UNTYPED
        and len(constant.dtypes) == 1
        and constant.dtypes[0]
    ):
        value = _numeric_literal(constant)
    else:
        var no_columns = List[Series]()
        value = evaluate(
            bind(subtree(part, constant_at), no_columns), no_columns, 1
        )
    if len(value) != 1 or value.is_chunked() or value.null_count() > 0:
        return None
    if value.dtype() != column.dtype():
        return None
    return Optional(
        _Comparison(column.copy(), op, value^, column.null_count() > 0)
    )


@always_inline
def _block[
    D: DType, op: Int
](
    values: Pointer[Scalar[D], MutAnyOrigin],
    constant: Scalar[D],
    first: Int,
    rows: Int,
    bits: Pointer[UInt8, MutAnyOrigin],
    initial: Bool,
):
    """Bits of `values[first : first + rows] op constant` into `bits`, one
    byte per eight rows (`rows` a multiple of 8 except at the end), or
    ANDed into what is there unless `initial`."""
    var splat = SIMD[D, 8](constant)
    var full = rows // 8
    for k in range(full):
        var x = values.unsafe_load[width=8](first + 8 * k)
        var byte = pack_bits[DType.uint8](_compare_lanes[op, D, 8](x, splat))
        if initial:
            bits[unsafe_offset=k] = byte
        else:
            bits[unsafe_offset=k] &= byte
    if full * 8 < rows:
        var byte = UInt8(0)
        for i in range(full * 8, rows):
            if _compare_lanes[op, D, 1](
                values[unsafe_offset=first + i], constant
            )[0]:
                byte |= UInt8(1) << UInt8(i - full * 8)
        if initial:
            bits[unsafe_offset=full] = byte
        else:
            bits[unsafe_offset=full] &= byte


@always_inline
def _and_valid(
    valid: _BufferData[UInt8],
    offset: Int,
    rows: Int,
    bits: Pointer[UInt8, MutAnyOrigin],
):
    """AND the validity of `rows` rows from bit `offset` into `bits`, whose
    bit 0 is the first of those rows."""
    var count = (rows + 7) // 8
    var byte = offset >> 3
    var shift = UInt8(offset & 7)
    var src = valid.unsafe_ptr().unsafe_offset(byte)
    if shift == 0:
        for k in range(count):
            bits[unsafe_offset=k] &= src[unsafe_offset=k]
        return
    # The last byte's upper bits may lie past the bitmap when the rows past
    # `rows` do; those bits of the mask are zero already.
    var whole = count if byte + count < len(valid) else count - 1
    for k in range(whole):
        bits[unsafe_offset=k] &= (src[unsafe_offset=k] >> shift) | (
            src[unsafe_offset=k + 1] << (8 - shift)
        )
    if whole < count:
        bits[unsafe_offset=whole] &= src[unsafe_offset=whole] >> shift


def _apply[
    D: DType
](
    comparison: _Comparison,
    first: Int,
    rows: Int,
    bits: Pointer[UInt8, MutAnyOrigin],
    initial: Bool,
):
    ref column = comparison.column._data[Column[Scalar[D]]]
    var values = column._ptr()
    var constant = comparison.value._data[Column[Scalar[D]]]._get(0)
    var op = comparison.op
    if op == LT:
        _block[D, LT](values, constant, first, rows, bits, initial)
    elif op == LE:
        _block[D, LE](values, constant, first, rows, bits, initial)
    elif op == GT:
        _block[D, GT](values, constant, first, rows, bits, initial)
    elif op == GE:
        _block[D, GE](values, constant, first, rows, bits, initial)
    elif op == EQ:
        _block[D, EQ](values, constant, first, rows, bits, initial)
    else:
        _block[D, NE](values, constant, first, rows, bits, initial)
    if comparison.nulls:
        # Null rows compare false: AND the validity of these rows in.
        _and_valid(column._bits[], column._offset + first, rows, bits)


struct _FusedJob(Job):
    var comparisons: List[_Comparison]
    var start: Int
    var end: Int
    var out: Int

    def __init__(
        out self,
        comparisons: List[_Comparison],
        start: Int,
        end: Int,
        out_address: Int,
    ):
        self.comparisons = comparisons.copy()
        self.start = start
        self.end = end
        self.out = out_address

    def run(mut self) raises:
        var mask = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.out)
        var row = self.start
        while row < self.end:
            var rows = min(_BLOCK_ROWS, self.end - row)
            var bits = mask.unsafe_offset(row // 8)
            for c in range(len(self.comparisons)):
                ref comparison = self.comparisons[c]
                comptime for d in range(len(NUMERIC_DTYPES)):
                    comptime D = NUMERIC_DTYPES[d]
                    if comparison.column._data.isa[Column[Scalar[D]]]():
                        _apply[D](comparison, row, rows, bits, c == 0)
            row += rows


def fused_comparisons(
    columns: List[Series], height: Int, parts: List[Expr]
) raises -> FusedComparisons:
    """The AND of the parts that compare a numeric column without nulls
    with a constant of its dtype, as a bitmap of `height` rows, and which
    parts it answers. No part answered: an empty bitmap."""
    var used = List[Bool](length=len(parts), fill=False)
    var comparisons = List[_Comparison]()
    for k in range(len(parts)):
        var found = _comparison(parts[k], columns)
        if found:
            if len(found.value().column) != height:
                continue
            comparisons.append(found.value().copy())
            used[k] = True
    if len(comparisons) == 0 or height == 0:
        return FusedComparisons(
            List[UInt8](), List[Bool](length=len(parts), fill=False)
        )
    var mask = List[UInt8](unsafe_uninit_length=(height + 7) // 8)
    var workers = worker_count(height)
    var bounds = partitions(height, workers, 64)
    var jobs = List[_FusedJob](capacity=workers)
    for w in range(workers):
        if bounds[w] < bounds[w + 1]:
            jobs.append(
                _FusedJob(
                    comparisons,
                    bounds[w],
                    bounds[w + 1],
                    Int(mask.unsafe_ptr()),
                )
            )
    run_jobs(jobs)
    return FusedComparisons(mask^, used^)


def fused_mask(mask: List[UInt8], height: Int) -> BoolColumn:
    """The bitmap as an all-valid Boolean column."""
    return BoolColumn(values=mask.copy(), bits=List[UInt8](), length=height)


# Selection trees: a filter part's AND/OR of comparisons and null tests,
# evaluated on a list of row positions rather than on every row, the way
# DuckDB's ConjunctionAnd/Or select narrow one selection vector. AND hands
# each child the rows the earlier ones kept; OR keeps a child's matches and
# offers the next child only the rows still unmatched. A comparison or test
# reads the source column at each listed row, so nothing is gathered.
comptime _COMPARE = 0
comptime _ALL = 1
comptime _ANY = 2
comptime _VALID = 3
comptime _MISSING = 4


@fieldwise_init
struct _TreeNode(Copyable, Movable):
    var kind: Int
    var left: Int
    var right: Int
    # The comparison (`_COMPARE`) or tested column (`_VALID`, `_MISSING`).
    var leaf: Int


struct SelectionTree(Movable):
    """One filter part as an AND/OR tree over comparisons and null tests.
    A row is selected when the part is true there; null counts as false,
    which is exact for AND and OR of these leaves under a filter."""

    var nodes: List[_TreeNode]
    var comparisons: List[_Comparison]
    # Each null test's column validity, one bit a row from row 0.
    var tested: List[List[UInt8]]

    def __init__(out self):
        self.nodes = List[_TreeNode]()
        self.comparisons = List[_Comparison]()
        self.tested = List[List[UInt8]]()

    def select(self, rows: List[Int]) raises -> List[Int]:
        """The rows of `rows` (ascending) where the part is true."""
        return self._select(len(self.nodes) - 1, rows)

    def _select(self, at: Int, rows: List[Int]) raises -> List[Int]:
        ref node = self.nodes[at]
        if node.kind == _ALL:
            var kept = self._select(node.left, rows)
            return self._select(node.right, kept)
        if node.kind == _ANY:
            var first = self._select(node.left, rows)
            var rest = _without(rows, first)
            var second = self._select(node.right, rest)
            return _merged(first, second)
        if node.kind == _COMPARE:
            ref comparison = self.comparisons[node.leaf]
            comptime for d in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[d]
                if comparison.column._data.isa[Column[Scalar[D]]]():
                    return _select_compare[D](comparison, rows)
            raise Error("selection tree: unsupported comparison column")
        var out = List[Int](capacity=len(rows))
        var want = UInt8(1) if node.kind == _VALID else UInt8(0)
        ref bits = self.tested[node.leaf]
        for row in rows:
            if (bits[row >> 3] >> UInt8(row & 7)) & 1 == want:
                out.append(row)
        return out^


def selection_tree(
    part: Expr, columns: List[Series]
) raises -> Optional[SelectionTree]:
    """`part` as a selection tree when every leaf is a comparison of a
    numeric column with a constant of its dtype, or a null test of a
    column; None otherwise."""
    var tree = SelectionTree()
    if not _add_tree(part, len(part._nodes) - 1, columns, tree):
        return None
    return Optional(tree^)


def _add_tree(
    part: Expr, at: Int, columns: List[Series], mut tree: SelectionTree
) raises -> Bool:
    """Append the subtree rooted at `at`, its root last; False when some
    leaf is unsupported."""
    ref node = part._nodes[at]
    if (node.op == AND or node.op == OR) and node.left >= 0 and node.right >= 0:
        if not _add_tree(part, node.left, columns, tree):
            return False
        var left = len(tree.nodes) - 1
        if not _add_tree(part, node.right, columns, tree):
            return False
        var right = len(tree.nodes) - 1
        tree.nodes.append(
            _TreeNode(_ALL if node.op == AND else _ANY, left, right, -1)
        )
        return True
    if (node.op == IS_NULL or node.op == IS_NOT_NULL) and node.left >= 0:
        ref child = part._nodes[node.left]
        if child.op != COL:
            return False
        for column in columns:
            if column.name() == child.text and not column.is_chunked():
                tree.tested.append(_validity_bytes(column, len(column)))
                tree.nodes.append(
                    _TreeNode(
                        _VALID if node.op == IS_NOT_NULL else _MISSING,
                        -1,
                        -1,
                        len(tree.tested) - 1,
                    )
                )
                return True
        return False
    var found = _comparison(subtree(part, at), columns)
    if not found:
        return False
    tree.comparisons.append(found.value().copy())
    tree.nodes.append(_TreeNode(_COMPARE, -1, -1, len(tree.comparisons) - 1))
    return True


def _select_compare[
    D: DType
](comparison: _Comparison, rows: List[Int]) -> List[Int]:
    var op = comparison.op
    if op == LT:
        return _select_rows[D, LT](comparison, rows)
    if op == LE:
        return _select_rows[D, LE](comparison, rows)
    if op == GT:
        return _select_rows[D, GT](comparison, rows)
    if op == GE:
        return _select_rows[D, GE](comparison, rows)
    if op == EQ:
        return _select_rows[D, EQ](comparison, rows)
    return _select_rows[D, NE](comparison, rows)


def _select_rows[
    D: DType, op: Int
](comparison: _Comparison, rows: List[Int]) -> List[Int]:
    """The rows whose value compares true; a null row compares false. The
    output is written branch-free: each row is stored, and the cursor moves
    past it only when it matched."""
    ref column = comparison.column._data[Column[Scalar[D]]]
    var values = column._ptr()
    var constant = comparison.value._data[Column[Scalar[D]]]._get(0)
    var out = List[Int](unsafe_uninit_length=len(rows))
    var dst = out.unsafe_ptr()
    var src = rows.unsafe_ptr()
    var count = 0
    if comparison.nulls:
        for i in range(len(rows)):
            var row = src[unsafe_offset=i]
            dst[unsafe_offset=count] = row
            var hit = _compare_lanes[op, D, 1](
                values[unsafe_offset=row], constant
            )[0] and column._valid(row)
            count += Int(hit)
    else:
        for i in range(len(rows)):
            var row = src[unsafe_offset=i]
            dst[unsafe_offset=count] = row
            count += Int(
                _compare_lanes[op, D, 1](values[unsafe_offset=row], constant)[0]
            )
    out.shrink(count)
    return out^


def _without(rows: List[Int], taken: List[Int]) -> List[Int]:
    """`rows` minus `taken`, a subsequence of it; both ascending."""
    var out = List[Int](capacity=len(rows) - len(taken))
    var t = 0
    for row in rows:
        if t < len(taken) and taken[t] == row:
            t += 1
        else:
            out.append(row)
    return out^


def _merged(a: List[Int], b: List[Int]) -> List[Int]:
    """The union of two disjoint ascending lists, ascending."""
    var out = List[Int](capacity=len(a) + len(b))
    var i = 0
    var j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    while i < len(a):
        out.append(a[i])
        i += 1
    while j < len(b):
        out.append(b[j])
        j += 1
    return out^
