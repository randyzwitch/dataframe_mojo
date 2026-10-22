# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# Index-based adaptation of std/builtin/sort.mojo at Modular commit
# 0eeb9c145e2539cb5aab8a3085b4853d27440d26. The bounded local stack and
# source-word indirection replace upstream heap-owned Span sorting.
# Full source/license attribution is retained in THIRD_PARTY_NOTICES.md.
"""CPU quantile's index permutation using software IEEE comparisons.

The source words remain immutable. Scratch contains source row indices, with
finite valid rows in their original visit order. Candidate sort fallback is
ported from upstream 0eeb9c14; pinned-build oracle validation is still required.
"""
from std.memory import Pointer, unsafe_stack_allocation
from . import floating as f


def _less(
    words: Pointer[UInt64, ImmutAnyOrigin], base: Int, a: UInt32, b: UInt32
) -> Bool:
    return f.less(
        words[unsafe_offset=base + Int(a)], words[unsafe_offset=base + Int(b)]
    )


def _swap(index: Pointer[UInt32, MutAnyOrigin], a: Int, b: Int):
    var held = index[unsafe_offset=a]
    index[unsafe_offset=a] = index[unsafe_offset=b]
    index[unsafe_offset=b] = held


def _insertion(
    words: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    index: Pointer[UInt32, MutAnyOrigin],
    lo: Int,
    hi: Int,
):
    for i in range(lo + 1, hi):
        var value = index[unsafe_offset=i]
        var j = i
        while j > lo and _less(words, base, value, index[unsafe_offset=j - 1]):
            index[unsafe_offset=j] = index[unsafe_offset=j - 1]
            j -= 1
        index[unsafe_offset=j] = value


def _sort2(
    words: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    index: Pointer[UInt32, MutAnyOrigin],
    a: Int,
    b: Int,
):
    # Upstream deliberately swaps EQUAL elements as well.
    if not _less(words, base, index[unsafe_offset=a], index[unsafe_offset=b]):
        _swap(index, a, b)


def _partition(
    words: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    index: Pointer[UInt32, MutAnyOrigin],
    lo: Int,
    hi: Int,
    equal_left: Bool,
) -> Int:
    var pivot = index[unsafe_offset=lo]
    var left = lo + 1
    var right = hi - 1
    while True:
        if equal_left:
            while left < right and not _less(
                words, base, pivot, index[unsafe_offset=left]
            ):
                left += 1
            while _less(words, base, pivot, index[unsafe_offset=right]):
                right -= 1
        else:
            while _less(words, base, index[unsafe_offset=left], pivot):
                left += 1
            while left < right and not _less(
                words, base, index[unsafe_offset=right], pivot
            ):
                right -= 1
        if left >= right:
            _swap(index, left - 1, lo)
            return left - 1
        _swap(index, left, right)
        left += 1
        right -= 1


def fallback_sort(
    words: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    index: Pointer[UInt32, MutAnyOrigin],
    begin: Int,
    end: Int,
):
    """Candidate upstream generic unstable sort with a bounded local stack.

    Process the smaller child first; disjoint child permutations commute.
    Each right child's predecessor is the immutable parent pivot, so visiting
    the other child first cannot change the duplicate-pivot shortcut. At most
    31 intervals are pending with the runtime's Int32 row bound. No allocation,
    floating arithmetic, source writes or global scratch beyond the indices.
    """
    var starts = unsafe_stack_allocation[32, DType.int64]()
    var ends = unsafe_stack_allocation[32, DType.int64]()
    var pending = 0
    var lo = begin
    var hi = end
    while True:
        if hi - lo < 32:
            _insertion(words, base, index, lo, hi)
        else:
            var middle = lo + ((hi - lo) >> 1)
            _sort2(words, base, index, middle, lo)
            _sort2(words, base, index, lo, hi - 1)
            _sort2(words, base, index, middle, lo)
            var equal_left = lo > begin and not _less(
                words,
                base,
                index[unsafe_offset=lo - 1],
                index[unsafe_offset=lo],
            )
            var pivot = _partition(words, base, index, lo, hi, equal_left)
            var left_end = pivot if not equal_left else lo
            var right_start = pivot + 1
            var left_size = left_end - lo
            var right_size = hi - right_start
            if left_size > 1 and right_size > 1:
                if left_size <= right_size:
                    starts[unsafe_offset=pending] = Int64(right_start)
                    ends[unsafe_offset=pending] = Int64(hi)
                    hi = left_end
                else:
                    starts[unsafe_offset=pending] = Int64(lo)
                    ends[unsafe_offset=pending] = Int64(left_end)
                    lo = right_start
                pending += 1
                continue
            elif left_size > 1:
                hi = left_end
                continue
            elif right_size > 1:
                lo = right_start
                continue
        if pending == 0:
            return
        pending -= 1
        lo = Int(starts[unsafe_offset=pending])
        hi = Int(ends[unsafe_offset=pending])


def quickselect(
    words: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    index: Pointer[UInt32, MutAnyOrigin],
    begin: Int,
    end: Int,
    rank: Int,
) -> Bool:
    """Same swaps as aggregate._select; return whether fallback was reached."""
    var lo = begin
    var hi = end
    var bits = 0
    var n = end - begin
    while n > 0:
        bits += 1
        n >>= 1
    var rounds = 0
    var limit = 2 * bits + 8
    var wanted = begin + rank
    while hi - lo > 24:
        if rounds == limit:
            fallback_sort(words, base, index, lo, hi)
            return True
        rounds += 1
        var a = index[unsafe_offset=lo]
        var b = index[unsafe_offset=(lo + hi) // 2]
        var c = index[unsafe_offset=hi - 1]
        # Only pivot's numerical value matters, so equal representatives can
        # be picked without reproducing CPU max/min tie choice.
        if _less(words, base, b, a):
            var held = a
            a = b
            b = held
        if _less(words, base, c, b):
            b = c
        if _less(words, base, b, a):
            b = a
        var lt = lo
        var i = lo
        var gt = hi
        while i < gt:
            var x = index[unsafe_offset=i]
            if _less(words, base, x, b):
                index[unsafe_offset=i] = index[unsafe_offset=lt]
                index[unsafe_offset=lt] = x
                lt += 1
                i += 1
            elif _less(words, base, b, x):
                gt -= 1
                index[unsafe_offset=i] = index[unsafe_offset=gt]
                index[unsafe_offset=gt] = x
            else:
                i += 1
        if wanted < lt:
            hi = lt
        elif wanted >= gt:
            lo = gt
        else:
            return False
    _insertion(words, base, index, lo, hi)
    return False


def initialize[
    Indexed: Bool
](
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    source_index: Pointer[UInt32, ImmutAnyOrigin],
    scratch: Pointer[UInt32, MutAnyOrigin],
    base: Int,
    begin: Int,
    end: Int,
    scratch_start: Int,
) -> Tuple[Int, Int]:
    """Return finite/non-NaN count and total valid count; ignore invalid words."""
    var numbers = 0
    var count = 0
    for at in range(begin, end):
        var row = at
        comptime if Indexed:
            row = Int(source_index[unsafe_offset=at])
        var source = base + row
        if validity[unsafe_offset=source] == 0:
            continue
        count += 1
        if not f.is_nan(words[unsafe_offset=source]):
            scratch[unsafe_offset=scratch_start + numbers] = UInt32(row)
            numbers += 1
    return (numbers, count)


def selected_pair(
    words: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    scratch: Pointer[UInt32, MutAnyOrigin],
    start: Int,
    numbers: Int,
    wanted: Int,
    upper: Int,
) -> Tuple[UInt64, UInt64, Bool]:
    """CPU-selected lower/wanted and upper words, including canonical NaNs."""
    var fallback = False
    if wanted < numbers:
        fallback = quickselect(
            words, base, scratch, start, start + numbers, wanted
        )
        if upper == wanted + 1 and upper < numbers:
            var least = start + upper
            for at in range(least + 1, start + numbers):
                if _less(
                    words,
                    base,
                    scratch[unsafe_offset=at],
                    scratch[unsafe_offset=least],
                ):
                    least = at
            _swap(scratch, start + upper, least)
    var lo = UInt64(0x7FF8000000000000)
    var hi = lo
    if wanted < numbers:
        lo = words[
            unsafe_offset=base + Int(scratch[unsafe_offset=start + wanted])
        ]
    if upper < numbers:
        hi = words[
            unsafe_offset=base + Int(scratch[unsafe_offset=start + upper])
        ]
    return (lo, hi, fallback)
