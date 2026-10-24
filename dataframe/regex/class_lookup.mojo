"""Bounded Unicode class directories over canonical interval lists.

The packed ASCII/two-byte prefix remains first in the membership buffer.
Above it, a class either holds one or two intervals directly or indexes a
plane/page directory. Uniform pages answer immediately; other pages narrow
the binary search to the intervals intersecting 256 code points.
"""
from .charclass import MAX_CODEPOINT, class_contains, class_contains_bounds

comptime _SINGLE: UInt64 = UInt64(1) << 62
comptime _PAIR: UInt64 = UInt64(2) << 62
comptime _PLANES: UInt64 = UInt64(3) << 62
comptime _CP_MASK: UInt64 = (UInt64(1) << 21) - 1
# Extra UInt64 entries, beyond the existing membership prefix and roots.
comptime _LOOKUP_WORDS = 1 << 17


@always_inline
def _interval(low: UInt32, high: UInt32) -> UInt64:
    return UInt64(low) | (UInt64(high - low) << 21)


@always_inline
def _contains(interval: UInt64, cp: UInt32) -> Bool:
    return cp - UInt32(interval & _CP_MASK) <= UInt32(
        (interval >> 21) & _CP_MASK
    )


def _set_index(mut words: List[UInt64], base: Int, slot: Int, value: UInt32):
    var at = base + (slot >> 1)
    var shift = UInt64((slot & 1) * 32)
    var mask = UInt64(0xFFFFFFFF) << shift
    words[at] = (words[at] & ~mask) | (UInt64(value) << shift)


def _build(mut words: List[UInt64], ranges: List[UInt32], limit: Int) -> UInt64:
    var count = len(ranges) >> 1
    var first = 0
    while first < count and ranges[2 * first + 1] < 0x800:
        first += 1
    if first == count:
        return 1
    if count - first == 1:
        return _SINGLE | _interval(
            max(ranges[2 * first], UInt32(0x800)), ranges[2 * first + 1]
        )
    var mark = len(words)
    if count - first == 2:
        if mark + 2 > limit:
            return 0
        for k in range(first, count):
            words.append(
                _interval(max(ranges[2 * k], UInt32(0x800)), ranges[2 * k + 1])
            )
        return _PAIR | UInt64(mark)
    if mark + 9 > limit:
        return 0
    # Seventeen packed plane entries, with an unused final half-word.
    words.resize(mark + 9, 0)
    var cursor = first
    for plane in range(17):
        var low = max(UInt32(plane << 16), UInt32(0x800))
        var high = UInt32(((plane + 1) << 16) - 1)
        while cursor < count and ranges[2 * cursor + 1] < low:
            cursor += 1
        if cursor == count or ranges[2 * cursor] > high:
            continue
        if ranges[2 * cursor] <= low and ranges[2 * cursor + 1] >= high:
            _set_index(words, mark, plane, 1)
            continue
        var pages = len(words)
        if pages + 128 > limit:
            words.resize(mark, 0)
            return 0
        words.resize(pages + 128, 0)
        _set_index(words, mark, plane, UInt32(pages))
        for page in range(256):
            var start = UInt32((plane << 16) | (page << 8))
            var end = start + 255
            if end < 0x800:
                continue
            while cursor < count and ranges[2 * cursor + 1] < start:
                cursor += 1
            if cursor == count or ranges[2 * cursor] > end:
                continue
            if ranges[2 * cursor] <= start and ranges[2 * cursor + 1] >= end:
                _set_index(words, pages, page, 1)
                continue
            var after = cursor + 1
            while after < count and ranges[2 * after] <= end:
                after += 1
            # At most 256 nonempty disjoint ranges intersect a page.
            var info = (UInt32(cursor + 1) << 8) | UInt32(after - cursor - 1)
            _set_index(words, pages, page, info)
    return _PLANES | UInt64(mark)


def build_unicode_membership(
    mut words: List[UInt64],
    classes: List[List[UInt32]],
    word: List[UInt32],
    word_assertions: Bool,
    budget: Int = _LOOKUP_WORDS,
):
    var roots = 32 * len(classes)
    words.resize(roots + len(classes) + Int(word_assertions), 0)
    var limit = len(words) + budget
    for id in range(len(classes)):
        var root = _build(words, classes[id], limit)
        words[roots + id] = root
    if word_assertions:
        var root: UInt64 = 0
        var found = False
        for id in range(len(classes)):
            if classes[id] == word:
                root = words[roots + id]
                found = True
                break
        if not found:
            root = _build(words, word, limit)
        words[roots + len(classes)] = root


@always_inline
def unicode_class_has(
    words: List[UInt64],
    ranges: List[UInt32],
    classes: Int,
    id: Int,
    cp: UInt32,
) -> Bool:
    # A single canonical range needs no lookup directory or binary search.
    if len(ranges) == 2:
        var points = ranges.unsafe_ptr()
        var low = points[unsafe_offset=0]
        return cp - low <= points[unsafe_offset=1] - low
    var p = words.unsafe_ptr()
    var root = p[unsafe_offset=32 * classes + id]
    if root == 1:
        return False
    var kind = root >> 62
    if kind == 1:
        return _contains(root, cp)
    if kind == 2:
        var at = Int(UInt32(root))
        return _contains(p[unsafe_offset=at], cp) or _contains(
            p[unsafe_offset=at + 1], cp
        )
    if kind == 0:
        return class_contains(ranges, cp)
    if cp > MAX_CODEPOINT:
        return False
    var plane = Int(cp >> 16)
    var pages = UInt32(
        p[unsafe_offset=Int(UInt32(root)) + (plane >> 1)]
        >> UInt64((plane & 1) * 32)
    )
    if pages <= 1:
        return pages == 1
    var page = Int((cp >> 8) & 255)
    var info = UInt32(
        p[unsafe_offset=Int(pages) + (page >> 1)] >> UInt64((page & 1) * 32)
    )
    if info <= 1:
        return info == 1
    var first = Int(info >> 8) - 1
    var after = first + Int(info & 255) + 1
    return class_contains_bounds(ranges, cp, first, after)
