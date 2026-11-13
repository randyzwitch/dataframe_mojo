"""Comparison and radix sorts over normalized key records.

Ported from DuckDB's third_party sources over this package's key records
(`SortKeys`): pdqsort_branchless (pdqsort.h, Orson Peters), vergesort
(vergesort.h, Morwenn) with the merge policy of
common/sorting/buffered_merge.hpp, and the in-place MSD radix sort of
ska_sort.hpp (Malte Skarupke, with DuckDB's changes: radix passes over the first
key word only, skippable bytes, and a vergesort over the full comparison for a
bucket whose first word is equal). The constants are the original authors'
empirically tuned values: pdqsort's insertion threshold 24, ninther threshold
128, block 64; vergesort's minimum 128; ska_sort's comparison threshold 128 and
American flag threshold 1024.

vergesort finds the input's existing sorted runs in linear time and sorts only
the stretches between them with its fallback. A run of unsorted rows falls back
to the radix sort, which beats comparison sorts on many fixed-width keys; a
merge partition, a few already sorted pieces, falls back to
`pdqsort_branchless`. `vergesort` takes the choice as `RADIX`.
"""
from std.bit import byte_swap, log2_floor
from std.memory import Pointer, unsafe_memcmp

# A key record held in registers: up to four 64-bit words (the widest
# layout, `PAYLOAD_VARIABLE_32` or `PAYLOAD_FIXED_32`, is 32 bytes).
comptime Rec = SIMD[DType.uint64, 4]

comptime _INSERTION_SORT_THRESHOLD = 24
comptime _NINTHER_THRESHOLD = 128
comptime _PARTIAL_INSERTION_SORT_LIMIT = 8
comptime _BLOCK_SIZE = 64
comptime _VERGESORT_MINIMUM = 128
comptime _STD_SORT_THRESHOLD = 128
comptime _AMERICAN_FLAG_SORT_THRESHOLD = 1024


struct SortKeys[W: Int, VAR: Bool](ImplicitlyCopyable, Movable):
    """Key records at `base`, `R` words each (a normalized key with a payload).
    Fixed: `W` key words, then the payload (the row id). Variable (`VAR`): the
    key's first 8 bytes, its byte size, the address of the whole key, then the
    payload. Key words are stored so that unsigned word order is the key bytes'
    order."""

    comptime R = 4 if Self.VAR else Self.W + 1
    var base: Int

    def __init__(out self, base: Int):
        self.base = base

    @always_inline
    def addr(self, i: Int) -> Int:
        return self.base + i * (Self.R * 8)

    @always_inline
    def get(self, i: Int) -> Rec:
        var p = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.addr(i))
        var r = Rec(0)
        comptime for k in range(Self.R):
            r[k] = p[unsafe_offset=k]
        return r

    @always_inline
    def put(self, i: Int, r: Rec):
        var p = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.addr(i))
        comptime for k in range(Self.R):
            p[unsafe_offset=k] = r[k]

    @always_inline
    def swap(self, i: Int, j: Int):
        var a = self.get(i)
        self.put(i, self.get(j))
        self.put(j, a)

    @always_inline
    def part0(self, i: Int) -> UInt64:
        return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.addr(i))[]

    @always_inline
    def payload(self, i: Int) -> Int:
        return Int(
            Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.addr(i))[
                unsafe_offset=Self.R - 1
            ]
        )

    @staticmethod
    @always_inline
    def less(a: Rec, b: Rec) -> Bool:
        """`operator<` of the key (the payload never takes part)."""
        comptime if Self.VAR:
            # Variable-size keys: the inlined bytes first; when they are equal
            # and either key is no longer than them, the shorter one is less;
            # otherwise the whole keys.
            if a[0] != b[0]:
                return a[0] < b[0]
            if a[1] <= 8 or b[1] <= 8:
                return a[1] < b[1]
            var n = min(a[1], b[1])
            var c = unsafe_memcmp(
                Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=Int(a[2])),
                Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=Int(b[2])),
                Int(n),
            )
            return c < 0 or (c == 0 and a[1] < b[1])
        else:
            comptime for k in range(Self.W):
                if a[k] != b[k]:
                    return a[k] < b[k]
            return False

    @always_inline
    def less_at(self, i: Int, j: Int) -> Bool:
        return Self.less(self.get(i), self.get(j))


struct MergeBuffer(Movable):
    """Scratch for half the keys of a sort plus one, so every merge of two
    sorted ranges moves its smaller side out and merges in one pass. In memory
    the allocation cannot come up short, so no in-place rotation fallback is
    needed."""

    var data: List[UInt64]

    def __init__(out self, key_count: Int, words: Int):
        self.data = List[UInt64](
            unsafe_uninit_length=max(1, (key_count // 2 + 1) * words)
        )

    def address(self) -> Int:
        return Int(self.data.unsafe_ptr())


# ===----------------------------------------------------------------------=== #
# pdqsort_branchless
# ===----------------------------------------------------------------------=== #


def _insertion_sort[
    W: Int, VAR: Bool
](k: SortKeys[W, VAR], begin: Int, end: Int, guarded: Bool):
    if begin == end:
        return
    for cur in range(begin + 1, end):
        var sift = cur
        var tmp = k.get(cur)
        if SortKeys[W, VAR].less(tmp, k.get(cur - 1)):
            while True:
                k.put(sift, k.get(sift - 1))
                sift -= 1
                if guarded and sift == begin:
                    break
                if not SortKeys[W, VAR].less(tmp, k.get(sift - 1)):
                    break
            k.put(sift, tmp)


def _partial_insertion_sort[
    W: Int, VAR: Bool
](k: SortKeys[W, VAR], begin: Int, end: Int) -> Bool:
    if begin == end:
        return True
    var limit = 0
    for cur in range(begin + 1, end):
        var sift = cur
        var tmp = k.get(cur)
        if SortKeys[W, VAR].less(tmp, k.get(cur - 1)):
            while True:
                k.put(sift, k.get(sift - 1))
                sift -= 1
                if sift == begin:
                    break
                if not SortKeys[W, VAR].less(tmp, k.get(sift - 1)):
                    break
            k.put(sift, tmp)
            limit += cur - sift
        if limit > _PARTIAL_INSERTION_SORT_LIMIT:
            return False
    return True


@always_inline
def _sort2[W: Int, VAR: Bool](k: SortKeys[W, VAR], a: Int, b: Int):
    if k.less_at(b, a):
        k.swap(a, b)


@always_inline
def _sort3[W: Int, VAR: Bool](k: SortKeys[W, VAR], a: Int, b: Int, c: Int):
    _sort2(k, a, b)
    _sort2(k, b, c)
    _sort2(k, a, b)


def _swap_offsets[
    W: Int, VAR: Bool
](
    k: SortKeys[W, VAR],
    first: Int,
    last: Int,
    offsets_l: Pointer[UInt8, MutAnyOrigin],
    offsets_r: Pointer[UInt8, MutAnyOrigin],
    num: Int,
    use_swaps: Bool,
):
    if use_swaps:
        for i in range(num):
            k.swap(
                first + Int(offsets_l[unsafe_offset=i]),
                last - Int(offsets_r[unsafe_offset=i]),
            )
    elif num > 0:
        var l = first + Int(offsets_l[unsafe_offset=0])
        var r = last - Int(offsets_r[unsafe_offset=0])
        var tmp = k.get(l)
        k.put(l, k.get(r))
        for i in range(1, num):
            l = first + Int(offsets_l[unsafe_offset=i])
            k.put(r, k.get(l))
            r = last - Int(offsets_r[unsafe_offset=i])
            k.put(l, k.get(r))
        k.put(r, tmp)


def _partition_right_branchless[
    W: Int, VAR: Bool
](k: SortKeys[W, VAR], begin: Int, end: Int) -> Tuple[Int, Bool]:
    comptime K = SortKeys[W, VAR]
    var pivot = k.get(begin)
    var first = begin
    var last = end
    first += 1
    while K.less(k.get(first), pivot):
        first += 1
    if first - 1 == begin:
        while first < last:
            last -= 1
            if K.less(k.get(last), pivot):
                break
    else:
        while True:
            last -= 1
            if K.less(k.get(last), pivot):
                break
    var already_partitioned = first >= last
    if not already_partitioned:
        k.swap(first, last)
        first += 1
        # The offset blocks live on the stack, as in pdqsort.h.
        var storage_l = SIMD[DType.uint8, 2 * _BLOCK_SIZE](0)
        var storage_r = SIMD[DType.uint8, 2 * _BLOCK_SIZE](0)
        var offsets_l = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(Pointer(to=storage_l))
        )
        var offsets_r = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(Pointer(to=storage_r))
        )
        var offsets_l_base = first
        var offsets_r_base = last
        var num_l = 0
        var num_r = 0
        var start_l = 0
        var start_r = 0
        while first < last:
            var num_unknown = last - first
            var left_split = (
                num_unknown // 2 if num_r == 0 else num_unknown
            ) if num_l == 0 else 0
            var right_split = (num_unknown - left_split) if num_r == 0 else 0
            if left_split >= _BLOCK_SIZE:
                for i in range(_BLOCK_SIZE):
                    offsets_l[unsafe_offset=num_l] = UInt8(i)
                    num_l += Int(not K.less(k.get(first), pivot))
                    first += 1
            else:
                for i in range(left_split):
                    offsets_l[unsafe_offset=num_l] = UInt8(i)
                    num_l += Int(not K.less(k.get(first), pivot))
                    first += 1
            if right_split >= _BLOCK_SIZE:
                for i in range(_BLOCK_SIZE):
                    offsets_r[unsafe_offset=num_r] = UInt8(i + 1)
                    last -= 1
                    num_r += Int(K.less(k.get(last), pivot))
            else:
                for i in range(right_split):
                    offsets_r[unsafe_offset=num_r] = UInt8(i + 1)
                    last -= 1
                    num_r += Int(K.less(k.get(last), pivot))
            var num = min(num_l, num_r)
            _swap_offsets(
                k,
                offsets_l_base,
                offsets_r_base,
                offsets_l.unsafe_offset(start_l),
                offsets_r.unsafe_offset(start_r),
                num,
                num_l == num_r,
            )
            num_l -= num
            num_r -= num
            start_l += num
            start_r += num
            if num_l == 0:
                start_l = 0
                offsets_l_base = first
            if num_r == 0:
                start_r = 0
                offsets_r_base = last
        if num_l > 0:
            var ol = offsets_l.unsafe_offset(start_l)
            while num_l > 0:
                num_l -= 1
                last -= 1
                k.swap(offsets_l_base + Int(ol[unsafe_offset=num_l]), last)
            first = last
        if num_r > 0:
            var orr = offsets_r.unsafe_offset(start_r)
            while num_r > 0:
                num_r -= 1
                k.swap(offsets_r_base - Int(orr[unsafe_offset=num_r]), first)
                first += 1
        _ = storage_l
        _ = storage_r
    var pivot_pos = first - 1
    k.put(begin, k.get(pivot_pos))
    k.put(pivot_pos, pivot)
    return (pivot_pos, already_partitioned)


def _partition_left[
    W: Int, VAR: Bool
](k: SortKeys[W, VAR], begin: Int, end: Int) -> Int:
    comptime K = SortKeys[W, VAR]
    var pivot = k.get(begin)
    var first = begin
    var last = end
    last -= 1
    while K.less(pivot, k.get(last)):
        last -= 1
    if last + 1 == end:
        while first < last:
            first += 1
            if K.less(pivot, k.get(first)):
                break
    else:
        while True:
            first += 1
            if K.less(pivot, k.get(first)):
                break
    while first < last:
        k.swap(first, last)
        last -= 1
        while K.less(pivot, k.get(last)):
            last -= 1
        first += 1
        while not K.less(pivot, k.get(first)):
            first += 1
    var pivot_pos = last
    k.put(begin, k.get(pivot_pos))
    k.put(pivot_pos, pivot)
    return pivot_pos


def _pdqsort_loop[
    W: Int, VAR: Bool
](k: SortKeys[W, VAR], begin_in: Int, end: Int, leftmost_in: Bool):
    var begin = begin_in
    var leftmost = leftmost_in
    while True:
        var size = end - begin
        if size < _INSERTION_SORT_THRESHOLD:
            _insertion_sort(k, begin, end, leftmost)
            return
        var s2 = size // 2
        if size > _NINTHER_THRESHOLD:
            _sort3(k, begin, begin + s2, end - 1)
            _sort3(k, begin + 1, begin + (s2 - 1), end - 2)
            _sort3(k, begin + 2, begin + (s2 + 1), end - 3)
            _sort3(k, begin + (s2 - 1), begin + s2, begin + (s2 + 1))
            k.swap(begin, begin + s2)
        else:
            _sort3(k, begin + s2, begin, end - 1)
        if not leftmost and not k.less_at(begin - 1, begin):
            begin = _partition_left(k, begin, end) + 1
            continue
        var part = _partition_right_branchless(k, begin, end)
        var pivot_pos = part[0]
        var already_partitioned = part[1]
        var l_size = pivot_pos - begin
        var r_size = end - (pivot_pos + 1)
        var highly_unbalanced = l_size < size // 8 or r_size < size // 8
        if highly_unbalanced:
            # Heapsort bounds worst-case work when partitioning stops making progress.
            if l_size >= _INSERTION_SORT_THRESHOLD:
                k.swap(begin, begin + l_size // 4)
                k.swap(pivot_pos - 1, pivot_pos - l_size // 4)
                if l_size > _NINTHER_THRESHOLD:
                    k.swap(begin + 1, begin + (l_size // 4 + 1))
                    k.swap(begin + 2, begin + (l_size // 4 + 2))
                    k.swap(pivot_pos - 2, pivot_pos - (l_size // 4 + 1))
                    k.swap(pivot_pos - 3, pivot_pos - (l_size // 4 + 2))
            if r_size >= _INSERTION_SORT_THRESHOLD:
                k.swap(pivot_pos + 1, pivot_pos + (1 + r_size // 4))
                k.swap(end - 1, end - r_size // 4)
                if r_size > _NINTHER_THRESHOLD:
                    k.swap(pivot_pos + 2, pivot_pos + (2 + r_size // 4))
                    k.swap(pivot_pos + 3, pivot_pos + (3 + r_size // 4))
                    k.swap(end - 2, end - (1 + r_size // 4))
                    k.swap(end - 3, end - (2 + r_size // 4))
        else:
            if (
                already_partitioned
                and _partial_insertion_sort(k, begin, pivot_pos)
                and _partial_insertion_sort(k, pivot_pos + 1, end)
            ):
                return
        _pdqsort_loop(k, begin, pivot_pos, leftmost)
        begin = pivot_pos + 1
        leftmost = False


def pdqsort_branchless[
    W: Int, VAR: Bool
](k: SortKeys[W, VAR], begin: Int, end: Int):
    if begin == end:
        return
    _pdqsort_loop(k, begin, end, True)


# ===----------------------------------------------------------------------=== #
# Buffered merge and vergesort
# ===----------------------------------------------------------------------=== #


def buffered_merge[
    W: Int, VAR: Bool
](k: SortKeys[W, VAR], first: Int, middle: Int, last: Int, buffer: Int):
    """Merge sorted [first, middle) and [middle, last) through `buffer`, which
    holds the smaller side: only the smaller side is copied out, and the merge
    writes into the space it left."""
    comptime K = SortKeys[W, VAR]
    var left_count = middle - first
    var right_count = last - middle
    if left_count == 0 or right_count == 0:
        return
    if not k.less_at(middle, middle - 1):
        return
    if left_count + right_count == 2:
        k.swap(first, middle)
        return
    var b = K(buffer)
    if left_count <= right_count:
        for i in range(left_count):
            b.put(i, k.get(first + i))
        var buffer_it = 0
        var right_it = middle
        var out = first
        while buffer_it != left_count and right_it != last:
            var r = k.get(right_it)
            var l = b.get(buffer_it)
            if K.less(r, l):
                k.put(out, r)
                right_it += 1
            else:
                k.put(out, l)
                buffer_it += 1
            out += 1
        while buffer_it != left_count:
            k.put(out, b.get(buffer_it))
            buffer_it += 1
            out += 1
        return
    for i in range(right_count):
        b.put(i, k.get(middle + i))
    var buffer_it = right_count
    var left_it = middle
    var out = last
    while buffer_it != 0 and left_it != first:
        out -= 1
        var r = b.get(buffer_it - 1)
        var l = k.get(left_it - 1)
        if K.less(r, l):
            left_it -= 1
            k.put(out, l)
        else:
            buffer_it -= 1
            k.put(out, r)
    while buffer_it != 0:
        out -= 1
        buffer_it -= 1
        k.put(out, b.get(buffer_it))


def _reverse[W: Int, VAR: Bool](k: SortKeys[W, VAR], first: Int, last: Int):
    var i = first
    var j = last - 1
    while i < j:
        k.swap(i, j)
        i += 1
        j -= 1


def _fallback[
    W: Int, VAR: Bool, RADIX: Bool
](k: SortKeys[W, VAR], begin: Int, end: Int, info: SkaInfo, buffer: Int):
    comptime if RADIX:
        ska_sort(k, begin, end, info, buffer)
    else:
        pdqsort_branchless(k, begin, end)


def vergesort[
    W: Int, VAR: Bool, RADIX: Bool
](k: SortKeys[W, VAR], first: Int, last: Int, info: SkaInfo, buffer: Int):
    """Vergesort's random-access version: find runs at least
    n / log2(n) long, sort what lies between them with the fallback
    (`ska_sort` when `RADIX`, else `pdqsort_branchless`), then merge the
    runs pairwise."""
    var dist = last - first
    if dist < _VERGESORT_MINIMUM:
        _fallback[W, VAR, RADIX](k, first, last, info, buffer)
        return
    var unstable_limit = dist // log2_floor(dist)
    var runs = List[Int]()
    var begin_unstable = last
    var current = first
    var next = first + 1
    while True:
        var begin_range = current
        if (last - next) <= unstable_limit:
            if begin_unstable == last:
                begin_unstable = begin_range
            break
        current += unstable_limit
        next += unstable_limit
        var current2 = current
        var next2 = next
        if k.less_at(next, current):
            # A decreasing sequence.
            while True:
                current -= 1
                next -= 1
                if k.less_at(current, next):
                    break
                if current == begin_range:
                    break
            if k.less_at(current, next):
                current += 1
            current2 += 1
            next2 += 1
            while next2 != last:
                if k.less_at(current2, next2):
                    break
                current2 += 1
                next2 += 1
            if (next2 - current) >= unstable_limit:
                _reverse(k, current, next2)
                if (current - begin_range) != 0 and begin_unstable == last:
                    begin_unstable = begin_range
                if begin_unstable != last:
                    _fallback[W, VAR, RADIX](
                        k, begin_unstable, current, info, buffer
                    )
                    runs.append(current)
                    begin_unstable = last
                runs.append(next2)
            else:
                if begin_unstable == last:
                    begin_unstable = begin_range
        else:
            # An increasing sequence.
            while True:
                current -= 1
                next -= 1
                if k.less_at(next, current):
                    break
                if current == begin_range:
                    break
            if k.less_at(next, current):
                current += 1
            current2 += 1
            next2 += 1
            while next2 != last:
                if k.less_at(next2, current2):
                    break
                current2 += 1
                next2 += 1
            if (next2 - current) >= unstable_limit:
                if (current - begin_range) != 0 and begin_unstable == last:
                    begin_unstable = begin_range
                if begin_unstable != last:
                    _fallback[W, VAR, RADIX](
                        k, begin_unstable, current, info, buffer
                    )
                    runs.append(current)
                    begin_unstable = last
                runs.append(next2)
            else:
                if begin_unstable == last:
                    begin_unstable = begin_range
        if next2 == last:
            break
        current = current2 + 1
        next = next2 + 1
    if begin_unstable != last:
        runs.append(last)
        _fallback[W, VAR, RADIX](k, begin_unstable, last, info, buffer)
    if len(runs) < 2:
        return
    while len(runs) > 1:
        var kept = List[Int](capacity=len(runs) // 2 + 1)
        var begin = first
        var i = 0
        while i + 1 < len(runs):
            buffered_merge(k, begin, runs[i], runs[i + 1], buffer)
            kept.append(runs[i + 1])
            begin = runs[i + 1]
            i += 2
        if i < len(runs):
            kept.append(runs[i])
        runs = kept^


# ===----------------------------------------------------------------------=== #
# ska_sort
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct SkaInfo(ImplicitlyCopyable, Movable):
    """The radix sort's settings: whether a bucket equal in the first key word
    still needs the full comparison, how many bytes of that word the radix
    passes read (`ska_sort_width`), and the bytes every key shares
    (`sort_skippable_bytes`: null bytes of columns that hold no null), as a bit
    mask over the eight byte offsets."""

    var requires_next_sort: Bool
    var ska_sort_width: Int
    var skippable: Int

    @always_inline
    def byte_is_skippable(self, offset: Int) -> Bool:
        return (self.skippable >> offset) & 1 == 1


@fieldwise_init
struct _scratch[D: DType](ImplicitlyCopyable):
    """An array on the caller's stack (one entry per byte value)."""

    var address: Int

    @always_inline
    def __getitem__(self, i: Int) -> Int:
        return Int(
            Pointer[Scalar[Self.D], MutAnyOrigin](
                unsafe_from_address=self.address
            )[unsafe_offset=i]
        )

    @always_inline
    def __setitem__(self, i: Int, value: Int):
        Pointer[Scalar[Self.D], MutAnyOrigin](unsafe_from_address=self.address)[
            unsafe_offset=i
        ] = Scalar[Self.D](value)


@always_inline
def _current_byte[
    W: Int, VAR: Bool
](k: SortKeys[W, VAR], i: Int, offset: Int) -> Int:
    return Int((k.part0(i) >> UInt64((7 - offset) * 8)) & 0xFF)


def _std_sort_if_less_than_threshold[
    W: Int, VAR: Bool
](k: SortKeys[W, VAR], begin: Int, end: Int) -> Bool:
    var n = end - begin
    if n <= 1:
        return True
    if n >= _STD_SORT_THRESHOLD:
        return False
    pdqsort_branchless(k, begin, end)
    return True


def _next_sort[
    W: Int, VAR: Bool
](k: SortKeys[W, VAR], begin: Int, end: Int, info: SkaInfo, buffer: Int):
    """A bucket equal in the first key word, by the full comparison: the radix
    passes cannot tell its keys apart."""
    vergesort[W, VAR, False](k, begin, end, info, buffer)


def _inplace_sort[
    W: Int, VAR: Bool
](
    k: SortKeys[W, VAR],
    begin: Int,
    end: Int,
    offset: Int,
    info: SkaInfo,
    buffer: Int,
):
    """Sort [begin, end) in place by byte `offset` of the first key word onward:
    skip a byte every key shares, use an American flag sort on a short range,
    else a ska_sort pass."""
    if offset == 8:
        if info.requires_next_sort:
            _next_sort(k, begin, end, info, buffer)
        return
    if info.byte_is_skippable(offset):
        _inplace_sort(k, begin, end, offset + 1, info, buffer)
    elif end - begin < _AMERICAN_FLAG_SORT_THRESHOLD:
        _american_flag_sort(k, begin, end, offset, info, buffer)
    else:
        _ska_byte_sort(k, begin, end, offset, info, buffer)


def _american_flag_sort[
    W: Int, VAR: Bool
](
    k: SortKeys[W, VAR],
    begin: Int,
    end: Int,
    offset: Int,
    info: SkaInfo,
    buffer: Int,
):
    var counts_storage = SIMD[DType.int64, 256](0)
    var next_storage = SIMD[DType.int64, 256](0)
    var remaining_storage = SIMD[DType.uint8, 256](0)
    var counts = _scratch[DType.int64](Int(Pointer(to=counts_storage)))
    var next_offset = _scratch[DType.int64](Int(Pointer(to=next_storage)))
    var remaining = _scratch[DType.uint8](Int(Pointer(to=remaining_storage)))
    for it in range(begin, end):
        counts[_current_byte(k, it, offset)] += 1
    var total = 0
    var num_partitions = 0
    for i in range(256):
        var count = counts[i]
        if count == 0:
            continue
        counts[i] = total
        total += count
        next_offset[i] = total
        remaining[num_partitions] = i
        num_partitions += 1
    if num_partitions > 1:
        var current_block_ptr = 0
        var current_block = Int(remaining[current_block_ptr])
        var last_block = num_partitions - 1
        var it = begin
        var block_end = begin + next_offset[current_block]
        var last_element = end - 1
        var done = False
        while not done:
            var block = _current_byte(k, it, offset)
            if block == current_block:
                it += 1
                if it == last_element:
                    break
                elif it == block_end:
                    while True:
                        current_block_ptr += 1
                        if current_block_ptr == last_block:
                            done = True
                            break
                        current_block = Int(remaining[current_block_ptr])
                        if counts[current_block] != next_offset[current_block]:
                            break
                    if done:
                        break
                    it = begin + counts[current_block]
                    block_end = begin + next_offset[current_block]
            else:
                var at = counts[block]
                counts[block] = at + 1
                k.swap(it, begin + at)
    if offset + 1 != info.ska_sort_width or info.requires_next_sort:
        var partition_begin = begin
        for p in range(num_partitions):
            var partition_end = begin + next_offset[Int(remaining[p])]
            if not _std_sort_if_less_than_threshold(
                k, partition_begin, partition_end
            ):
                _inplace_sort(
                    k, partition_begin, partition_end, offset + 1, info, buffer
                )
            partition_begin = partition_end
    # The arrays are read only through their addresses: keep them alive.
    _ = counts_storage
    _ = next_storage
    _ = remaining_storage


def _ska_byte_sort[
    W: Int, VAR: Bool
](
    k: SortKeys[W, VAR],
    begin: Int,
    end: Int,
    offset: Int,
    info: SkaInfo,
    buffer: Int,
):
    var offsets_storage = SIMD[DType.int64, 256](0)
    var next_storage = SIMD[DType.int64, 256](0)
    var remaining_storage = SIMD[DType.uint8, 256](0)
    var offsets = _scratch[DType.int64](Int(Pointer(to=offsets_storage)))
    var next_offset = _scratch[DType.int64](Int(Pointer(to=next_storage)))
    var remaining = _scratch[DType.uint8](Int(Pointer(to=remaining_storage)))
    for it in range(begin, end):
        offsets[_current_byte(k, it, offset)] += 1
    var total = 0
    var num_partitions = 0
    for i in range(256):
        var count = offsets[i]
        if count != 0:
            offsets[i] = total
            total += count
            remaining[num_partitions] = i
            num_partitions += 1
        next_offset[i] = total
    var last_remaining = num_partitions
    while last_remaining > 1:
        # custom_std_partition over the partitions still holding misplaced
        # rows: each pass moves every row of a partition to its target,
        # and keeps the partitions that are not yet done.
        var keep = 0
        for p in range(last_remaining):
            var partition = Int(remaining[p])
            var begin_offset = offsets[partition]
            var end_offset = next_offset[partition]
            var unfinished = False
            if begin_offset != end_offset:
                var it = begin + begin_offset
                for _ in range(end_offset - begin_offset):
                    var this_partition = _current_byte(k, it, offset)
                    var at = offsets[this_partition]
                    offsets[this_partition] = at + 1
                    k.swap(it, begin + at)
                    it += 1
                unfinished = offsets[partition] != next_offset[partition]
            if unfinished:
                var held = remaining[keep]
                remaining[keep] = remaining[p]
                remaining[p] = held
                keep += 1
        last_remaining = keep
    if offset + 1 != info.ska_sort_width or info.requires_next_sort:
        for i in range(255, -1, -1):
            var start_offset = 0 if i == 0 else next_offset[i - 1]
            var end_offset = next_offset[i]
            if end_offset == start_offset:
                continue
            var partition_begin = begin + start_offset
            var partition_end = begin + end_offset
            if not _std_sort_if_less_than_threshold(
                k, partition_begin, partition_end
            ):
                _inplace_sort(
                    k, partition_begin, partition_end, offset + 1, info, buffer
                )
    _ = offsets_storage
    _ = next_storage
    _ = remaining_storage


def ska_sort[
    W: Int, VAR: Bool
](k: SortKeys[W, VAR], begin: Int, end: Int, info: SkaInfo, buffer: Int):
    """`ska_sort(begin, end, extract_key, merge)`: inplace_radix_sort<128,
    1024> over the first key word."""
    if _std_sort_if_less_than_threshold(k, begin, end):
        return
    _inplace_sort(k, begin, end, 0, info, buffer)
