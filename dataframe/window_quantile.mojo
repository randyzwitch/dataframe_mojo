"""Quantiles of sliding frames.

Each output row has a frame, a range of rows inside its partition, and the
frames of consecutive rows move forward (a fixed window, or a duration
window over rows in time order). A quantile needs the frame's values in
order, but sorting every frame costs O(w log w) per row for a frame of w
rows. Instead the partition is sorted once, which gives every row its
rank. A fixed window of w rows is then read with a sorted-block filter
(`fixed_quantiles`, below), which works within two blocks of w rows at a
time. Frames of varying size (duration windows, time windows) are kept as
the set of ranks they hold:

- a bitset with one bit per rank, so a row entering or leaving the frame
  sets or clears its bit;
- counts of the frame's ranks by block of the bitset, and a Fenwick tree
  of counts by group of blocks, so the k-th smallest value is found by a
  descent through the tree, a short scan of block counts and words, and
  a select within one word (`_RankFrame`).

Each row entering, each row leaving and each quantile costs O(log(n /
8192)) plus a scan of at most 24 counts for a partition of n rows,
whatever the frame's size, and the arrays are flat (about n / 7 bytes in
all) with the tree small enough to stay in cache. Frames that move
backwards are handled too, at a cost proportional to how far the bounds
move.

Nulls take no rank and are never in a frame. NaN sorts above every number,
as in `sort`. Positions follow the interpolation method on the frame's
count of valid values c and quantile q, with f = (c - 1) * q:

- lower, higher: floor(f), ceil(f); nearest: f rounded half away from zero;
- equiprobable: ceil(c * q) - 1 (at least 0);
- linear: the values at floor(f) and ceil(f), joined in proportion to the
  fraction of f; midpoint: their average.
"""
from std.bit import pop_count, count_trailing_zeros
from std.math import floor, ceil
from std.memory import Pointer
from dataframe.storage.column import Column
from .dtype import DataType
from .series import Series


@always_inline
def _select_in_word(word: UInt64, nth: Int) -> Int:
    """The bit position of the nth (from 1) set bit of `word`, found by
    halving: each step counts the set bits of the low half and moves to the
    half holding the nth."""
    var w = word
    var rest = UInt64(nth)
    var base = 0
    # Each step's direction depends on the data, so it is taken without a
    # branch.
    comptime for half in [32, 16, 8, 4, 2, 1]:
        var mask = (UInt64(1) << UInt64(half)) - 1
        var low = pop_count(w & mask)
        var up = rest > low
        rest -= low if up else 0
        w >>= UInt64(half) if up else 0
        base += half if up else 0
    return base


struct _RankFrame:
    """The ranks in one partition's current frame, in three levels held at
    offsets in arrays shared by all partitions (by address):

    - `bits`: one bit per rank;
    - `blocks`: how many ranks each block of 8 words (512 ranks) holds;
    - `tree`: a Fenwick tree of how many each superblock of 16 blocks
      (8,192 ranks) holds, `size` (a power of two) entries.

    Adding or removing a rank updates its bit, its block and O(log(n /
    8192)) tree entries. Finding the k-th rank descends the tree to its
    superblock, then scans at most 16 block counts and 8 words: the tree
    is small enough to stay in the nearest cache, and each scan reads one
    or two cache lines, where a tree over single words would chain a
    dependent load from a farther cache per level."""

    var bits: Int
    var blocks: Int
    var tree: Int
    var size: Int
    var words: Int
    var block_count: Int

    def __init__(
        out self,
        bits: Int,
        blocks: Int,
        tree: Int,
        size: Int,
        words: Int,
        block_count: Int,
    ):
        self.bits = bits
        self.blocks = blocks
        self.tree = tree
        self.size = size
        self.words = words
        self.block_count = block_count

    @always_inline
    def _bits(self) -> Pointer[UInt64, MutAnyOrigin]:
        return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.bits)

    @always_inline
    def _blocks(self) -> Pointer[Int32, MutAnyOrigin]:
        return Pointer[Int32, MutAnyOrigin](unsafe_from_address=self.blocks)

    @always_inline
    def _tree(self) -> Pointer[Int32, MutAnyOrigin]:
        return Pointer[Int32, MutAnyOrigin](unsafe_from_address=self.tree)

    @always_inline
    def add(self, rank: Int, delta: Int32):
        """Put `rank` in the frame (delta 1) or take it out (delta -1)."""
        var bits = self._bits()
        var bit = UInt64(1) << UInt64(rank & 63)
        if delta > 0:
            bits[unsafe_offset=rank >> 6] |= bit
        else:
            bits[unsafe_offset=rank >> 6] &= ~bit
        self._blocks()[unsafe_offset=rank >> 9] += delta
        var tree = self._tree()
        var i = (rank >> 13) + 1
        while i <= self.size:
            tree[unsafe_offset=i - 1] += delta
            i += i & -i

    @always_inline
    def select(self, k: Int) -> Int:
        """The k-th (from 0) smallest rank in the frame."""
        var tree = self._tree()
        var at = 0
        var rest = Int32(k + 1)
        var step = self.size
        # Which way each step goes depends on the data, so it is taken
        # without a branch.
        while step > 0:
            var count = tree[unsafe_offset=at + step - 1]
            var right = count < rest
            at += step if right else 0
            rest -= count if right else 0
            step >>= 1
        var blocks = self._blocks()
        var block = at << 4
        while blocks[unsafe_offset=block] < rest:
            rest -= blocks[unsafe_offset=block]
            block += 1
        var bits = self._bits()
        var word = block << 3
        while True:
            var count = Int32(pop_count(bits[unsafe_offset=word]))
            if count >= rest:
                break
            rest -= count
            word += 1
        return (word << 6) + _select_in_word(
            bits[unsafe_offset=word], Int(rest)
        )

    @always_inline
    def after(self, rank: Int, k: Int) -> Int:
        """The rank following `rank`, the k-th, in the frame: from the same
        word when it holds one, else by a descent."""
        var bits = self._bits()
        var shift = UInt64((rank & 63) + 1)
        if shift < 64:
            var higher = bits[unsafe_offset=rank >> 6] >> shift
            if higher != 0:
                return rank + 1 + Int(count_trailing_zeros(higher))
        # The rest of the rank's block, then the rest of its superblock,
        # before a descent from the top.
        var word = (rank >> 6) + 1
        var block_end = ((rank >> 9) + 1) << 3
        while word < block_end and word < self.words:
            var w = bits[unsafe_offset=word]
            if w != 0:
                return (word << 6) + Int(count_trailing_zeros(w))
            word += 1
        var blocks = self._blocks()
        var block = (rank >> 9) + 1
        var super_end = ((rank >> 13) + 1) << 4
        while block < super_end and block < self.block_count:
            if blocks[unsafe_offset=block] != 0:
                word = block << 3
                while bits[unsafe_offset=word] == 0:
                    word += 1
                return (word << 6) + Int(
                    count_trailing_zeros(bits[unsafe_offset=word])
                )
            block += 1
        return self.select(k + 1)


struct _Layout(Copyable, Movable):
    """Where each partition's `_RankFrame` levels start in the shared
    arrays, and the arrays' lengths."""

    var words: List[Int]
    var blocks: List[Int]
    var trees: List[Int]
    var sizes: List[Int]

    def __init__(out self, starts: List[Int]):
        var parts = len(starts) - 1
        self.words = List[Int](capacity=parts + 1)
        self.blocks = List[Int](capacity=parts + 1)
        self.trees = List[Int](capacity=parts + 1)
        self.sizes = List[Int](capacity=parts)
        var words = 0
        var blocks = 0
        var trees = 0
        for p in range(parts):
            self.words.append(words)
            self.blocks.append(blocks)
            self.trees.append(trees)
            var ranks = starts[p + 1] - starts[p]
            var w = (ranks + 63) // 64
            var b = (w + 7) // 8
            var supers = (b + 15) // 16
            var size = 1
            while size < supers:
                size *= 2
            self.sizes.append(size)
            words += w
            blocks += b
            trees += size
        self.words.append(words)
        self.blocks.append(blocks)
        self.trees.append(trees)


@always_inline
def _positions(
    count: Int, quantile: Float64, method: Int
) -> Tuple[Int, Int, Float64]:
    """The positions (from 0) of the values a quantile reads among `count`
    sorted values, and the weight of the second (0 when one is read)."""
    var last = count - 1
    var f = Float64(last) * quantile
    if method == _LINEAR or method == _MIDPOINT:
        var lo = min(Int(floor(f)), last)
        var hi = min(Int(ceil(f)), last)
        if lo == hi:
            return (lo, lo, 0.0)
        return (lo, hi, 0.5 if method == _MIDPOINT else f - Float64(lo))
    if method == _LOWER:
        return (min(Int(floor(f)), last), -1, 0.0)
    if method == _HIGHER:
        return (min(Int(ceil(f)), last), -1, 0.0)
    if method == _EQUIPROBABLE:
        var at = max(Int(ceil(Float64(count) * quantile)) - 1, 0)
        return (min(at, last), -1, 0.0)
    # Nearest, rounding halves away from zero (f is never negative).
    return (min(Int(floor(f + 0.5)), last), -1, 0.0)


comptime _NEAREST = 0
comptime _LOWER = 1
comptime _HIGHER = 2
comptime _MIDPOINT = 3
comptime _LINEAR = 4
comptime _EQUIPROBABLE = 5


def _method_code(method: String) raises -> Int:
    if method == "nearest":
        return _NEAREST
    if method == "lower":
        return _LOWER
    if method == "higher":
        return _HIGHER
    if method == "midpoint":
        return _MIDPOINT
    if method == "linear":
        return _LINEAR
    if method == "equiprobable":
        return _EQUIPROBABLE
    raise Error(
        "interpolation must be nearest, lower, higher, midpoint, linear or"
        " equiprobable"
    )


def quantile_frames(
    input: Series,
    begins: List[Int],
    ends: List[Int],
    starts: List[Int],
    needed: Int,
    quantile: Float64,
    method: String,
) raises -> Series:
    """Each row's quantile of `input` over its frame [begins, ends), inside
    the partitions at `starts` (row ranges): `frame_quantiles` with a frame
    per row."""
    return frame_quantiles(
        input, starts, begins, ends, starts, needed, quantile, method
    )


struct _Ranked(Movable):
    """Each valid row's rank within its partition (-1 for a null) and the
    rows by rank: one sort of the column gives every valid row its place,
    and the rows of each partition then take ranks 0, 1, ... in that
    order (a counting pass). Rank r of partition p is at row_at[starts[p]
    + r], and fill[p] is where partition p's ranks end."""

    var rank: List[Int32]
    var row_at: List[Int]
    var fill: List[Int]

    def __init__(out self, column: Series, starts: List[Int]) raises:
        var n = len(column)
        var parts = len(starts) - 1
        var order = column.argsort()
        var valid_rows = n - column.null_count()
        var part = List[Int32]()
        if parts > 1:
            part = List[Int32](unsafe_uninit_length=n)
            for p in range(parts):
                for j in range(starts[p], starts[p + 1]):
                    part[j] = Int32(p)
        self.rank = List[Int32](length=n, fill=-1)
        self.row_at = List[Int](unsafe_uninit_length=n)
        self.fill = List[Int](capacity=max(parts, 1))
        for p in range(parts):
            self.fill.append(starts[p])
        for k in range(valid_rows):
            var row = order[k]
            var p = Int(part[row]) if parts > 1 else 0
            var slot = self.fill[p]
            self.fill[p] = slot + 1
            self.rank[row] = Int32(slot - starts[p])
            self.row_at[slot] = row


def frame_quantiles(
    input: Series,
    starts: List[Int],
    begins: List[Int],
    ends: List[Int],
    frame_starts: List[Int],
    needed: Int,
    quantile: Float64,
    method: String,
) raises -> Series:
    """The quantile of `input` over each frame [begins, ends), positions of
    `input`. The rows form partitions at `starts` (row ranges), and frames
    [frame_starts[p], frame_starts[p + 1]) lie in partition p. Fewer than
    `needed` valid values, or none, give null. Float32 and decimals keep
    their type; other numbers give Float64."""
    var column = input.rechunk()
    var code = _method_code(method)
    var n = len(column)
    var parts = len(starts) - 1
    var ranked = _Ranked(column, starts)
    ref rank = ranked.rank
    ref row_at = ranked.row_at
    ref fill = ranked.fill
    if column.dtype().is_decimal():
        var wide = column._decimal128()
        ref stored = wide._data[Column[Int128]]
        var sorted = List[Int128](unsafe_uninit_length=n)
        var from_wide = stored._ptr()
        for p in range(parts):
            for slot in range(starts[p], fill[p]):
                sorted[slot] = from_wide[unsafe_offset=row_at[slot]]
        return _frame_quantiles[DType.int128](
            sorted,
            rank,
            begins,
            ends,
            starts,
            frame_starts,
            needed,
            quantile,
            code,
        ).with_dtype(column.dtype())
    var floats = column.cast(DataType.FLOAT64)
    ref stored = floats._data[Column[Float64]]
    var sorted = List[Float64](unsafe_uninit_length=n)
    var from_floats = stored._ptr()
    for p in range(parts):
        for slot in range(starts[p], fill[p]):
            sorted[slot] = from_floats[unsafe_offset=row_at[slot]]
    if column.dtype() == DataType.FLOAT32:
        # Float32 values join in Float32 arithmetic, so an interpolated
        # quantile rounds as the values' own type does.
        return _frame_quantiles[DType.float64, True](
            sorted,
            rank,
            begins,
            ends,
            starts,
            frame_starts,
            needed,
            quantile,
            code,
        ).cast(DataType.FLOAT32)
    return _frame_quantiles[DType.float64](
        sorted,
        rank,
        begins,
        ends,
        starts,
        frame_starts,
        needed,
        quantile,
        code,
    )


def _visit_order(
    begins: List[Int], ends: List[Int], f0: Int, f1: Int, lo: Int, hi: Int
) raises -> List[Int]:
    """Frames [f0, f1) in order of (begin, end), or none when they already
    are: then each frame's bounds are at or after the previous one's."""
    var ordered = True
    for j in range(f0 + 1, f1):
        if begins[j] < begins[j - 1] or ends[j] < ends[j - 1]:
            ordered = False
            break
    if ordered:
        return List[Int]()
    var span = Int64(hi - lo + 1)
    var keys = List[Int64](capacity=f1 - f0)
    for j in range(f0, f1):
        keys.append(Int64(begins[j] - lo) * span + Int64(ends[j] - lo))
    var order = Series("", Column[Int64](keys^)).argsort()
    for k in range(len(order)):
        order[k] += f0
    return order^


def _frame_quantiles[
    D: DType, narrow: Bool = False
](
    sorted: List[Scalar[D]],
    rank: List[Int32],
    begins: List[Int],
    ends: List[Int],
    starts: List[Int],
    frame_starts: List[Int],
    needed: Int,
    quantile: Float64,
    method: Int,
) raises -> Series:
    """`frame_quantiles` over values laid out by rank (Float64, or a
    decimal's unscaled Int128, which interpolates as Float64 and truncates
    toward zero back to an integer). A partition's frames are taken in
    order of their bounds, so each moves the frame forward; frames listed
    out of that order (rows not in time order) are sorted first."""
    var n = len(begins)
    var parts = len(starts) - 1
    var layout = _Layout(starts)
    var bits = List[UInt64](length=max(layout.words[parts], 1), fill=0)
    var blocks = List[Int32](length=max(layout.blocks[parts], 1), fill=0)
    var tree = List[Int32](length=max(layout.trees[parts], 1), fill=0)
    var out = List[Scalar[D]](unsafe_uninit_length=n)
    var ok = List[Bool](unsafe_uninit_length=n)
    var any_null = False
    var ranks = rank.unsafe_ptr()
    var values = sorted.unsafe_ptr()
    var bp = begins.unsafe_ptr()
    var ep = ends.unsafe_ptr()
    var op = out.unsafe_ptr()
    var okp = ok.unsafe_ptr()
    for p in range(parts):
        var lo = starts[p]
        var hi = starts[p + 1]
        var frame = _RankFrame(
            Int(bits.unsafe_ptr()) + 8 * layout.words[p],
            Int(blocks.unsafe_ptr()) + 4 * layout.blocks[p],
            Int(tree.unsafe_ptr()) + 4 * layout.trees[p],
            layout.sizes[p],
            layout.words[p + 1] - layout.words[p],
            layout.blocks[p + 1] - layout.blocks[p],
        )
        var first = lo
        var last = lo
        var count = 0
        var f0 = frame_starts[p]
        var f1 = frame_starts[p + 1]
        var visit = _visit_order(begins, ends, f0, f1, lo, hi)
        for step in range(f0, f1):
            var j = step if len(visit) == 0 else visit[step - f0]
            var b = bp[unsafe_offset=j]
            var e = ep[unsafe_offset=j]
            if b >= last or e <= first:
                # No overlap with the previous frame: empty it.
                for row in range(first, last):
                    var r = Int(ranks[unsafe_offset=row])
                    if r >= 0:
                        frame.add(r, -1)
                count = 0
                first = b
                last = b
            while last < e:
                var r = Int(ranks[unsafe_offset=last])
                if r >= 0:
                    frame.add(r, 1)
                    count += 1
                last += 1
            while first > b:
                first -= 1
                var r = Int(ranks[unsafe_offset=first])
                if r >= 0:
                    frame.add(r, 1)
                    count += 1
            while last > e:
                last -= 1
                var r = Int(ranks[unsafe_offset=last])
                if r >= 0:
                    frame.add(r, -1)
                    count -= 1
            while first < b:
                var r = Int(ranks[unsafe_offset=first])
                if r >= 0:
                    frame.add(r, -1)
                    count -= 1
                first += 1
            if count < needed or count == 0:
                okp[unsafe_offset=j] = False
                op[unsafe_offset=j] = Scalar[D](0)
                any_null = True
                continue
            okp[unsafe_offset=j] = True
            var at = _positions(count, quantile, method)
            var low_rank = frame.select(at[0])
            var low = values[unsafe_offset=lo + low_rank]
            if at[1] < 0 or at[1] == at[0]:
                op[unsafe_offset=j] = low
                continue
            var high = values[unsafe_offset=lo + frame.after(low_rank, at[0])]
            op[unsafe_offset=j] = _join[D, narrow](low, high, at[2], method)
    _ = bits^
    _ = blocks^
    _ = tree^
    _ = rank
    _ = sorted
    if any_null:
        return Series("", Column[Scalar[D]](out^, ok))
    return Series("", Column[Scalar[D]](out^))


# Fixed windows: a sorted-block filter.
#
# A trailing window of k rows always lies within two consecutive blocks of
# k rows: a suffix of the left block and a prefix of the right one. Each
# block's valid rows are kept in value order as a doubly linked list over
# the block (from the partition's one sort, no sorting per block). As the
# window moves one row, the row leaving is unlinked from the left block
# and the row entering is linked back into the right block. The right
# block starts empty after unlinking its rows from last to first, and is
# relinked from first to last; links undone in the reverse order of their
# removal are restored exactly. Each block keeps a cursor on one element
# and its position, so the k-th value of the window is found by walking
# the two cursors from where they were, merging the two lists: the
# position asked for moves little from one window to the next, so a row
# costs O(1) steps on average, all within the 2k rows of the two blocks.


struct _Block(Movable):
    """One block's valid rows in value order, as links over the block's
    positions (`tail`, the block's length, ends the list both ways), with
    a cursor on element `m` at position `current` of `count`."""

    var ranks: Int
    var start: Int
    var tail: Int
    var prev: List[Int32]
    var next: List[Int32]
    var m: Int
    var current: Int
    var count: Int

    def __init__(out self, ranks: Int):
        self.ranks = ranks
        self.start = 0
        self.tail = 0
        self.prev = List[Int32]()
        self.next = List[Int32]()
        self.m = 0
        self.current = 0
        self.count = 0

    @always_inline
    def key(self, i: Int) -> Int32:
        """Element i's rank (-1 for a null, which is never linked)."""
        return Pointer[Int32, MutAnyOrigin](unsafe_from_address=self.ranks)[
            unsafe_offset=self.start + i
        ]

    def reset(
        mut self,
        start: Int,
        length: Int,
        order: Pointer[Int32, MutAnyOrigin],
        valid: Int,
    ):
        """The block of `length` rows at `start`, its `valid` valid rows
        (`order`, by value) linked, then unlinked from last to first so
        the block starts empty."""
        self.start = start
        self.tail = length
        self.prev.resize(length + 1, 0)
        self.next.resize(length + 1, 0)
        var p = length
        for k in range(valid):
            var q = Int(order[unsafe_offset=k])
            self.next[p] = Int32(q)
            self.prev[q] = Int32(p)
            p = q
        self.next[p] = Int32(length)
        self.prev[length] = Int32(p)
        for i in range(length - 1, -1, -1):
            if self.key(i) >= 0:
                self._unlink(i)
        self.m = length
        self.current = 0
        self.count = 0

    @always_inline
    def _unlink(mut self, i: Int):
        var before = Int(self.prev[i])
        var after = Int(self.next[i])
        self.next[before] = Int32(after)
        self.prev[after] = Int32(before)

    @always_inline
    def _relink(mut self, i: Int):
        self.next[Int(self.prev[i])] = Int32(i)
        self.prev[Int(self.next[i])] = Int32(i)

    @always_inline
    def at_end(self) -> Bool:
        return self.m == self.tail

    @always_inline
    def back(mut self):
        if self.current > 0:
            self.current -= 1
            self.m = Int(self.prev[self.m])

    @always_inline
    def forward(mut self):
        if self.current < self.count:
            self.current += 1
            self.m = Int(self.next[self.m])

    @always_inline
    def peek(self) -> Int:
        """The cursor's element, or -1 at the end."""
        return -1 if self.m == self.tail else self.m

    @always_inline
    def peek_previous(self) -> Int:
        var p = Int(self.prev[self.m])
        return -1 if p == self.tail else p

    def move_to(mut self, position: Int):
        while self.current > position:
            self.current -= 1
            self.m = Int(self.prev[self.m])
        while self.current < position:
            self.current += 1
            self.m = Int(self.next[self.m])

    def remove(mut self, i: Int):
        """Unlink valid element i, keeping the cursor on the same element
        (or a neighbor when it is the one removed)."""
        if self.at_end():
            self.back()
        var mine = self.key(i)
        var cursor = self.key(self.m)
        self._unlink(i)
        self.count -= 1
        if mine < cursor:
            self.current -= 1
        elif mine == cursor:
            if self.count >= self.current:
                var next_m = Int(self.next[self.m])
                if next_m == self.tail and self.count > 0:
                    self.current -= 1
                    self.m = Int(self.prev[self.m])
                else:
                    self.m = next_m
            else:
                self.m = Int(self.prev[self.m])

    def restore(mut self, i: Int):
        """Link valid element i back, keeping the cursor on its element."""
        if self.count > 0 and self.at_end():
            self.back()
        self._relink(i)
        if self.count == 0:
            self.m = Int(self.prev[self.m])
            self.count = 1
            self.current = 0
            return
        self.count += 1
        if self.key(i) < self.key(self.m):
            self.current += 1


def _union_back(mut left: _Block, mut right: _Block):
    """Step back over the greater of the two elements before the cursors."""
    var l = left.peek_previous()
    var r = right.peek_previous()
    if l >= 0 and r < 0:
        left.back()
    elif l < 0 and r >= 0:
        right.back()
    elif l >= 0 and r >= 0:
        if left.key(l) <= right.key(r):
            right.back()
        else:
            left.back()


def _union_get(mut left: _Block, mut right: _Block, i: Int) -> Int:
    """The position (in the partition's rows) of the i-th (from 0) value
    of the two blocks' linked elements together."""
    if right.count == 0:
        left.move_to(i)
        return left.start + left.m
    if left.count == 0:
        right.move_to(i)
        return right.start + right.m
    # A cursor can point too far after the other block changed: step back
    # until both are before position i, then merge forward.
    var peek_index = left.current + right.current + 1
    while i <= peek_index:
        _union_back(left, right)
        peek_index = left.current + right.current + 1
        if peek_index <= 1 and i <= 1:
            break
    while True:
        var s = left.current + right.current
        var l = left.peek()
        var r = right.peek()
        if l >= 0 and (r < 0 or left.key(l) <= right.key(r)):
            if s == i:
                return left.start + l
            left.forward()
        elif r >= 0:
            if s == i:
                return right.start + r
            right.forward()
        else:
            return left.start + left.m


def fixed_quantiles(
    input: Series,
    starts: List[Int],
    window: Int,
    center: Bool,
    needed: Int,
    quantile: Float64,
    method: String,
) raises -> Series:
    """Each row's quantile of `input` over its fixed window inside the
    partitions at `starts` (row ranges): the window rows ending at the
    row, or centered on it (the extra row of an even window before it),
    cut short at the partition's edges. Fewer than `needed` valid values,
    or none, give null; types as in `frame_quantiles`."""
    var column = input.rechunk()
    var code = _method_code(method)
    var ranked = _Ranked(column, starts)
    if column.dtype().is_decimal():
        var wide = column._decimal128()
        var values = wide._data[Column[Int128]].to_list()
        return _fixed_quantiles[DType.int128](
            values, ranked, starts, window, center, needed, quantile, code
        ).with_dtype(column.dtype())
    var floats = column.cast(DataType.FLOAT64)
    var values = floats._data[Column[Float64]].to_list()
    if column.dtype() == DataType.FLOAT32:
        return _fixed_quantiles[DType.float64, True](
            values, ranked, starts, window, center, needed, quantile, code
        ).cast(DataType.FLOAT32)
    return _fixed_quantiles[DType.float64](
        values, ranked, starts, window, center, needed, quantile, code
    )


@always_inline
def _join[
    D: DType, narrow: Bool
](low: Scalar[D], high: Scalar[D], weight: Float64, method: Int) -> Scalar[D]:
    """The value between `low` and `high` a linear or midpoint quantile
    takes: Float32 values in Float32 arithmetic; a decimal's unscaled
    integers in Float64, truncated toward zero."""
    comptime if narrow:
        var lf = rebind[Float64](low).cast[DType.float32]()
        var hf = rebind[Float64](high).cast[DType.float32]()
        var joined = (lf + hf) / 2 if method == _MIDPOINT else Float32(
            weight
        ) * (hf - lf) + lf
        return rebind[Scalar[D]](joined.cast[DType.float64]())
    elif D == DType.float64:
        var lf = rebind[Float64](low)
        var hf = rebind[Float64](high)
        return rebind[Scalar[D]](
            (lf + hf) / 2 if method == _MIDPOINT else weight * (hf - lf) + lf
        )
    else:
        var lf = low.cast[DType.float64]()
        var hf = high.cast[DType.float64]()
        return (lf + (hf - lf) * weight).cast[D]()


@always_inline
def _block_order(
    base: Int, at: List[Int], b: Int
) -> Pointer[Int32, MutAnyOrigin]:
    """Block b's valid elements in value order."""
    return Pointer[Int32, MutAnyOrigin](unsafe_from_address=base + 4 * at[b])


@always_inline
def _emit[
    D: DType, narrow: Bool
](
    row: Int,
    mut left: _Block,
    mut right: _Block,
    needed: Int,
    quantile: Float64,
    method: Int,
    values: Pointer[Scalar[D], _],
    results: Pointer[Scalar[D], MutAnyOrigin],
    ok: Pointer[Bool, MutAnyOrigin],
) -> Bool:
    """Write row `row`'s quantile of the two blocks' linked elements;
    whether it is null (fewer than `needed` valid values, or none)."""
    var count = left.count + right.count
    if count == 0:
        ok[unsafe_offset=row] = False
        return True
    var picks = _positions(count, quantile, method)
    # The cursors move with every window, even one too small to give a
    # value: stepping back to a position assumes the cursors were placed
    # by the window before, and rows enter and leave unevenly around
    # nulls.
    var low = values[unsafe_offset=_union_get(left, right, picks[0])]
    if count < needed:
        ok[unsafe_offset=row] = False
        return True
    ok[unsafe_offset=row] = True
    if picks[1] < 0 or picks[1] == picks[0]:
        results[unsafe_offset=row] = low
        return False
    var high = values[unsafe_offset=_union_get(left, right, picks[1])]
    results[unsafe_offset=row] = _join[D, narrow](low, high, picks[2], method)
    return False


def _fixed_quantiles[
    D: DType, narrow: Bool = False
](
    values: List[Scalar[D]],
    ranked: _Ranked,
    starts: List[Int],
    window: Int,
    center: Bool,
    needed: Int,
    quantile: Float64,
    method: Int,
) raises -> Series:
    """`fixed_quantiles` over values by row, by the sorted-block filter: a
    trailing window ending at each row of a partition in turn, then for a
    centered window the windows at the partition's end, which only shrink
    from the left. A centered row's window is the trailing window ending
    `after` rows past it."""
    var n = len(values)
    var out = List[Scalar[D]](length=n, fill=Scalar[D](0))
    var ok = List[Bool](length=n, fill=False)
    var any_null = False
    var after = (window + 1) // 2 if center else 1
    var before = window - after
    var ranks_address = Int(ranked.rank.unsafe_ptr())
    var rank = ranked.rank.unsafe_ptr()
    var vp = values.unsafe_ptr()
    var op = out.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var okp = ok.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var left = _Block(ranks_address)
    var right = _Block(ranks_address)
    # Each block's valid elements in value order, block after block.
    var order = List[Int32](length=max(n, 1), fill=0)
    var staged = List[Int32](length=max(n, 1), fill=0)
    var at = List[Int]()
    for p in range(len(starts) - 1):
        var lo = starts[p]
        var hi = starts[p + 1]
        var m = hi - lo
        if m == 0:
            continue
        var k = min(window, m)
        var blocks = (m + k - 1) // k
        at.clear()
        for _ in range(blocks + 1):
            at.append(0)
        for row in range(lo, hi):
            if rank[unsafe_offset=row] >= 0:
                at[(row - lo) // k + 1] += 1
        for b in range(blocks):
            at[b + 1] += at[b]
        # Rows in value order are dealt to their blocks in two passes,
        # first to spans of about 65,536 rows and then to the blocks of
        # each span, so each pass writes to few places at once and its
        # counters stay in cache, rather than to every block at random.
        var per_span = max(1, 65536 // k)
        var spans = (blocks + per_span - 1) // per_span
        var span_at = List[Int](length=spans + 1, fill=0)
        for b in range(blocks):
            span_at[b // per_span + 1] += at[b + 1] - at[b]
        for g in range(spans):
            span_at[g + 1] += span_at[g]
        var filled = span_at.copy()
        for slot in range(lo, ranked.fill[p]):
            var row = ranked.row_at[slot]
            var g = (row - lo) // k // per_span
            staged[lo + filled[g]] = Int32(row - lo)
            filled[g] += 1
        var counts = at.copy()
        for slot in range(span_at[0], span_at[spans]):
            var offset = Int(staged[lo + slot])
            var b = offset // k
            order[lo + counts[b]] = Int32(offset - b * k)
            counts[b] += 1
        var base = Int(order.unsafe_ptr()) + 4 * lo
        left.reset(lo, k, _block_order(base, at, 0), at[1] - at[0])
        right.reset(lo, 0, _block_order(base, at, 0), 0)
        var right_active = False
        # Windows growing within the first block.
        for i in range(k):
            if rank[unsafe_offset=lo + i] >= 0:
                left.restore(i)
            var row = lo + i + 1 - after
            if row >= lo:
                if _emit[D, narrow](
                    row, left, right, needed, quantile, method, vp, op, okp
                ):
                    any_null = True
        for b in range(1, blocks):
            var start = lo + b * k
            var length = min(k, hi - start)
            right.reset(
                start, length, _block_order(base, at, b), at[b + 1] - at[b]
            )
            for j in range(length):
                if rank[unsafe_offset=start - k + j] >= 0:
                    left.remove(j)
                if rank[unsafe_offset=start + j] >= 0:
                    right.restore(j)
                var row = start + j + 1 - after
                if row >= lo:
                    if _emit[D, narrow](
                        row, left, right, needed, quantile, method, vp, op, okp
                    ):
                        any_null = True
            if length < k:
                right_active = True
                break
            swap(left, right)
        # A centered window's rows near the end: the window ending at the
        # partition's end, shrinking from the left.
        var first = max(lo, hi - k)
        for row in range(max(lo, hi - after + 1), hi):
            var begin = max(lo, row - before)
            while first < begin:
                if rank[unsafe_offset=first] >= 0:
                    if right_active and first >= right.start:
                        right.remove(first - right.start)
                    else:
                        left.remove(first - left.start)
                first += 1
            if _emit[D, narrow](
                row, left, right, needed, quantile, method, vp, op, okp
            ):
                any_null = True
    _ = order^
    _ = staged^
    _ = values
    if any_null:
        return Series("", Column[Scalar[D]](out^, ok))
    return Series("", Column[Scalar[D]](out^))
