"""Inequality joins: the pairs of rows whose keys satisfy one or two
comparisons (<, <=, >, >=) of a left key with a right key, found from sorted
keys rather than by comparing every pair of rows.

Range join. When every comparison reads the same key on one side (the point
side), that side's valid keys are sorted once. A row of the other side (the
interval side) then matches one contiguous run of the sorted keys: each
comparison bounds the run from below or from above, and binary search finds
the bound. Interval rows are matched independently, so tasks take ranges of
them; a count pass sizes the output and a fill pass writes each row's pairs
at its offset.

Two-key join. With a different key in each comparison on both sides, both
sides are sorted by their first key and cut into blocks. A pair of blocks
whose first-key ranges cannot satisfy the first comparison is skipped; every
other pair is a task. The task merges its two blocks in first-key order and
visits the rows in second-key order (the IEJoin of Khayyat et al., 2015,
extended to repeated keys): a bit per merged row marks the right rows
visited so far, which are those satisfying the second comparison, and a
left row's matches are the marked rows from its place in first-key order
on, which satisfy the first.

Keys arrive as one number type for both sides (Int64 or Float64). A null
key, or a NaN (no comparison with NaN holds), matches nothing.
"""
from std.bit import count_trailing_zeros
from std.math import isnan

from std.collections import Optional

from dataframe.dtype import DataType
from dataframe.series import Series
from dataframe.sorting.sort import sort_rows
from dataframe.storage.column import Column

# A comparison `a OP b`.
comptime IE_LT = 0
comptime IE_LE = 1
comptime IE_GT = 2
comptime IE_GE = 3

# Rows per filter bit of the visited-row bit array: a scan for set bits
# skips a whole stretch when its filter bit is clear.
comptime _STRETCH = 1024


def ie_swapped(op: Int) -> Int:
    """The comparison with its operands swapped: `a < b` is `b > a`."""
    if op == IE_LT:
        return IE_GT
    if op == IE_LE:
        return IE_GE
    if op == IE_GT:
        return IE_LT
    return IE_LE


def ie_less(op: Int) -> Bool:
    return op == IE_LT or op == IE_LE


def ie_strict(op: Int) -> Bool:
    return op == IE_LT or op == IE_GT


@always_inline
def ie_holds[D: DType](op: Int, a: Scalar[D], b: Scalar[D]) -> Bool:
    """Whether `a OP b`."""
    if op == IE_LT:
        return a < b
    if op == IE_LE:
        return a <= b
    if op == IE_GT:
        return a > b
    return a >= b


def pairing_is_cheaper(left_rows: Int, right_rows: Int) -> Bool:
    """Whether comparing every pair of rows is no more work than sorting:
    n * m comparisons against about (n + m) * log2(n + m) steps for a sort
    of the inputs. It holds when one input has a handful of rows."""
    var total = left_rows + right_rows
    var steps = 1
    while (1 << steps) < total + 1:
        steps += 1
    return Float64(left_rows) * Float64(right_rows) <= Float64(total) * Float64(
        steps
    )


def nested_pairs[
    D: DType
](
    left: List[Column[Scalar[D]]],
    right: List[Column[Scalar[D]]],
    ops: List[Int],
    first: Int,
    end: Int,
    right_count: Int,
    mut left_out: List[Int],
    mut right_out: List[Int],
):
    """Append every pair of a left row in [first, end) and a right row whose
    keys satisfy every `left[k] OP right[k]`, comparing each pair."""
    for i in range(first, end):
        var valid = True
        for k in range(len(ops)):
            if not _matchable(left[k], i):
                valid = False
                break
        if not valid:
            continue
        for j in range(right_count):
            var holds = True
            for k in range(len(ops)):
                if not _matchable(right[k], j) or not ie_holds[D](
                    ops[k], left[k]._get(i), right[k]._get(j)
                ):
                    holds = False
                    break
            if holds:
                left_out.append(i)
                right_out.append(j)


def ie_name(op: Int) -> String:
    if op == IE_LT:
        return "<"
    if op == IE_LE:
        return "<="
    if op == IE_GT:
        return ">"
    return ">="


@always_inline
def _matchable[D: DType](column: Column[Scalar[D]], row: Int) -> Bool:
    if not column._valid(row):
        return False
    comptime if D.is_floating_point():
        return not isnan(column._get(row))
    return True


def matchable_rows[
    D: DType
](keys: List[Column[Scalar[D]]], count: Int) -> List[Bool]:
    """Per row, whether every key is valid and not NaN."""
    var rows = List[Bool](length=count, fill=True)
    for k in range(len(keys)):
        for row in range(count):
            if rows[row] and not _matchable(keys[k], row):
                rows[row] = False
    return rows^


@always_inline
def _first_at_least[
    D: DType
](values: List[Scalar[D]], lo: Int, hi: Int, v: Scalar[D]) -> Int:
    """The first position in [lo, hi) of ascending `values` holding v or
    more, else hi."""
    var a = lo
    var b = hi
    while a < b:
        var m = (a + b) >> 1
        if values[m] < v:
            a = m + 1
        else:
            b = m
    return a


@always_inline
def _first_above[
    D: DType
](values: List[Scalar[D]], lo: Int, hi: Int, v: Scalar[D]) -> Int:
    """The first position in [lo, hi) of ascending `values` holding more
    than v, else hi."""
    var a = lo
    var b = hi
    while a < b:
        var m = (a + b) >> 1
        if values[m] <= v:
            a = m + 1
        else:
            b = m
    return a


def range_spans[
    D: DType
](
    points: List[Scalar[D]],
    bounds: List[Column[Scalar[D]]],
    ops: List[Int],
    first: Int,
    end: Int,
    mut starts: List[Int],
    mut stops: List[Int],
) -> Int:
    """For interval rows [first, end), the run [starts[row], stops[row]) of
    ascending `points` (the point side's valid keys) satisfying every
    `point OP bound` (`ops[k]`, `bounds[k]` the interval side's key), and
    the pairs they make. A row with a null or NaN bound matches nothing."""
    var n = len(points)
    var total = 0
    for row in range(first, end):
        var lo = 0
        var hi = n
        var valid = True
        for k in range(len(ops)):
            if not _matchable(bounds[k], row):
                valid = False
                break
            var b = bounds[k]._get(row)
            var op = ops[k]
            if op == IE_GT:
                lo = _first_above(points, lo, hi, b)
            elif op == IE_GE:
                lo = _first_at_least(points, lo, hi, b)
            elif op == IE_LT:
                hi = _first_at_least(points, lo, hi, b)
            else:
                hi = _first_above(points, lo, hi, b)
        if not valid or hi <= lo:
            starts[row] = 0
            stops[row] = 0
            continue
        starts[row] = lo
        stops[row] = hi
        total += hi - lo
    return total


def range_fill(
    starts: List[Int],
    stops: List[Int],
    order: List[Int],
    first: Int,
    end: Int,
    offset: Int,
    mut interval_rows: List[Int],
    mut point_rows: List[Int],
):
    """Write the pairs of interval rows [first, end) from `offset` on: each
    row with each point row of its run (`order` maps sorted positions to
    point rows)."""
    var at = offset
    for row in range(first, end):
        for p in range(starts[row], stops[row]):
            interval_rows[at] = row
            point_rows[at] = order[p]
            at += 1


struct _VisitedRows(Movable):
    """A bit per merged row, set once the row is visited, and a filter bit
    per stretch of rows, set when any row of the stretch is. A scan for set
    bits reads the filter a word (64 stretches) at a time and skips every
    stretch whose bit is clear, so a scan over a mostly unmarked tail costs
    little."""

    var words: List[UInt64]
    var filter: List[UInt64]
    var count: Int

    def __init__(out self, count: Int):
        self.count = count
        self.words = List[UInt64](length=(count + 63) // 64, fill=0)
        var stretches = (count + _STRETCH - 1) // _STRETCH
        self.filter = List[UInt64](length=(stretches + 63) // 64, fill=0)

    @always_inline
    def set(mut self, i: Int):
        self.words[i >> 6] |= UInt64(1) << UInt64(i & 63)
        var stretch = i // _STRETCH
        self.filter[stretch >> 6] |= UInt64(1) << UInt64(stretch & 63)


def _match_starts[D: DType](xs: List[Scalar[D]], op: Int) -> List[Int]:
    """Per merged position p, the first position from p whose first key k
    satisfies `xs[p] OP k`. The merged keys run in op's order (ascending
    for < and <=, descending for > and >=), so the satisfying keys are a
    suffix: from p itself when the comparison admits equal keys, else from
    the end of p's run of equal keys."""
    var count = len(xs)
    var starts = List[Int](length=count, fill=count)
    if ie_strict(op):
        var run_end = count
        var p = count - 1
        while p >= 0:
            if p + 1 < count and xs[p + 1] != xs[p]:
                run_end = p + 1
            starts[p] = run_end
            p -= 1
    else:
        for p in range(count):
            starts[p] = p
    return starts^


def _emit_marked[
    D: DType
](
    visited: _VisitedRows,
    rows: List[Int],
    start: Int,
    left_row: Int,
    mut left_out: List[Int],
    mut right_out: List[Int],
):
    """Pair `left_row` with every visited (right) merged row from `start`
    on; `rows` holds -1 - right row at each right position."""
    if start >= visited.count:
        return
    var first_stretch = start // _STRETCH
    var f = first_stretch >> 6
    var fmask = ~UInt64(0) << UInt64(first_stretch & 63)
    while f < len(visited.filter):
        var stretches = visited.filter[f] & fmask
        fmask = ~UInt64(0)
        while stretches != 0:
            var stretch = f * 64 + Int(count_trailing_zeros(stretches))
            stretches &= stretches - 1
            var first = max(start, stretch * _STRETCH)
            var end = min((stretch + 1) * _STRETCH, visited.count)
            var w = first >> 6
            var mask = ~UInt64(0) << UInt64(first & 63)
            while w * 64 < end:
                var word = visited.words[w] & mask
                mask = ~UInt64(0)
                while word != 0:
                    var bit = w * 64 + Int(count_trailing_zeros(word))
                    if bit >= end:
                        break
                    left_out.append(left_row)
                    right_out.append(-1 - rows[bit])
                    word &= word - 1
                w += 1
        f += 1


def ie_join_block[
    D: DType
](
    lx: List[Scalar[D]],
    ly: List[Scalar[D]],
    rx: List[Scalar[D]],
    ry: List[Scalar[D]],
    op1: Int,
    op2: Int,
    mut left_out: List[Int],
    mut right_out: List[Int],
) raises:
    """Append the (left, right) positions, within these blocks, of every
    pair with `lx OP1 rx` and `ly OP2 ry`. Each side's rows come sorted by
    their first key in op1's order: ascending for < and <=, descending for
    > and >=, so a left row's first-key matches follow it once the sides
    are merged (left rows first among equal keys)."""
    var nl = len(lx)
    var nr = len(rx)
    var count = nl + nr
    if nl == 0 or nr == 0:
        return
    # Merge by first key; `rows` holds left row l as l, right row r as
    # -1 - r.
    var ascending = ie_less(op1)
    var xs = List[Scalar[D]](capacity=count)
    var ys = List[Scalar[D]](capacity=count)
    var rows = List[Int](capacity=count)
    var i = 0
    var j = 0
    while i < nl or j < nr:
        var take_left: Bool
        if i == nl:
            take_left = False
        elif j == nr:
            take_left = True
        elif ascending:
            take_left = lx[i] <= rx[j]
        else:
            take_left = lx[i] >= rx[j]
        if take_left:
            xs.append(lx[i])
            ys.append(ly[i])
            rows.append(i)
            i += 1
        else:
            xs.append(rx[j])
            ys.append(ry[j])
            rows.append(-1 - j)
            j += 1
    # The merged positions in second-key order, stable: descending for <
    # and <=, so the right rows visited before a left row hold greater
    # second keys; ascending for > and >=.
    var order = sort_rows(
        [Series("y", Column[Scalar[D]](ys.copy()))],
        [ie_less(op2)],
        [True],
        maintain_order=True,
        threads=1,
    )
    var visited = _VisitedRows(count)
    var starts = _match_starts(xs, op1)
    if ie_strict(op2):
        # A strict second comparison: stable order puts equal second keys
        # in merged order, so an equal right row visited before a left row
        # sits before it in first-key order, where no match is sought.
        for p in order:
            if rows[p] >= 0:
                _emit_marked[D](
                    visited,
                    rows,
                    starts[p],
                    rows[p],
                    left_out,
                    right_out,
                )
            else:
                visited.set(p)
        return
    # An inclusive second comparison: a run of equal second keys is marked
    # whole before any of its left rows looks for matches.
    var run = 0
    for k in range(len(order)):
        var p = order[k]
        if rows[p] < 0:
            visited.set(p)
        if k + 1 < len(order) and ys[order[k + 1]] == ys[p]:
            continue
        for q in range(run, k + 1):
            var r = order[q]
            if rows[r] >= 0:
                _emit_marked[D](
                    visited,
                    rows,
                    starts[r],
                    rows[r],
                    left_out,
                    right_out,
                )
        run = k + 1


def ie_blocks_overlap[
    D: DType
](
    left_first: Scalar[D],
    left_last: Scalar[D],
    right_first: Scalar[D],
    right_last: Scalar[D],
    op1: Int,
) -> Bool:
    """Whether a left block and a right block, each sorted by its first key
    in op1's order (first and last keys given), can hold a pair satisfying
    `left OP1 right`."""
    var left_low = min(left_first, left_last)
    var left_high = max(left_first, left_last)
    var right_low = min(right_first, right_last)
    var right_high = max(right_first, right_last)
    if op1 == IE_LT:
        return left_low < right_high
    if op1 == IE_LE:
        return left_low <= right_high
    if op1 == IE_GT:
        return left_high > right_low
    return left_high >= right_low


def range_key_type(left: DataType, right: DataType) -> Optional[DataType]:
    """The number type both keys of a comparison are compared in, when an
    inequality join can compare them exactly: Int64 for integers that fit
    it and for one temporal type on both sides, Float64 for floats; none
    for anything else (the join's caller filters pairs instead)."""
    if left.is_temporal() or right.is_temporal():
        if left == right:
            return DataType.INT64
        return None
    if left.is_integer() and right.is_integer():
        if left == DataType.UINT64 or right == DataType.UINT64:
            return None
        return DataType.INT64
    if left.is_float() and right.is_float():
        return DataType.FLOAT64
    return None
