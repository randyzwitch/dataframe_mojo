"""Bounded plane/page directories for code-point alphabet classes.

The canonical interval-to-class prefix stays intact. Negative directory
entries encode a uniform class, plane offsets index page directories, and
mixed pages encode a bounded slice of the canonical interval table.
"""

comptime _PLANES = 17
comptime _PAGES = 256
comptime _LOOKUP_ENTRIES = _PLANES + _PLANES * _PAGES


def build_alphabet_lookup(
    points: List[UInt32],
    mut classes: List[Int32],
    budget: Int = _LOOKUP_ENTRIES,
):
    var count = len(points)
    classes.resize(count, 0)
    # The existing direct map and trailing-class check cover this domain.
    if count == 0 or points[count - 1] <= 0x800 or budget < _PLANES:
        return
    var limit = count + min(budget, _LOOKUP_ENTRIES)
    classes.resize(count + _PLANES, 0)
    var cursor = 0
    for plane in range(_PLANES):
        var low = max(UInt32(plane << 16), UInt32(0x800))
        if low >= points[count - 1]:
            break
        var high = UInt32(((plane + 1) << 16) - 1)
        while cursor + 1 < count and points[cursor + 1] <= low:
            cursor += 1
        var last = cursor
        while last + 1 < count and points[last + 1] <= high:
            last += 1
        if cursor == last:
            classes[count + plane] = Int32(-Int(classes[cursor]) - 1)
            continue
        var base = len(classes)
        if base + _PAGES > limit:
            cursor = last
            continue
        classes.resize(base + _PAGES, 0)
        classes[count + plane] = Int32(base + 1)
        for page in range(_PAGES):
            var start = UInt32((plane << 16) | (page << 8))
            if start < 0x800:
                continue
            while cursor + 1 < count and points[cursor + 1] <= start:
                cursor += 1
            var after = cursor + 1
            while after < count and points[after] <= start + 255:
                after += 1
            if after == cursor + 1:
                classes[base + page] = Int32(-Int(classes[cursor]) - 1)
            else:
                # At most 256 nonempty intervals intersect one page. Cut
                # indices fit in 21 bits over the Unicode scalar domain.
                classes[base + page] = Int32(
                    (cursor << 8) | (after - cursor - 1)
                )
            cursor = after - 1


@no_inline
def wide_alphabet_class(
    points: List[UInt32], classes: List[Int32], cp: UInt32
) -> Int:
    # The caller already handled the direct map and trailing interval, so
    # cp is within the Unicode domain and below the final boundary.
    var count = len(points)
    var lo = 0
    var hi = count
    var ids = classes.unsafe_ptr()
    if len(classes) >= count + _PLANES:
        var entry = Int(ids[unsafe_offset=count + Int(cp >> 16)])
        if entry < 0:
            return -entry - 1
        if entry > 0:
            entry = Int(ids[unsafe_offset=entry - 1 + Int((cp >> 8) & 255)])
            if entry < 0:
                return -entry - 1
            if entry > 0:
                lo = entry >> 8
                hi = lo + (entry & 255) + 1
    var cuts = points.unsafe_ptr()
    while hi - lo > 1:
        var mid = (lo + hi) >> 1
        if cuts[unsafe_offset=mid] <= cp:
            lo = mid
        else:
            hi = mid
    return Int(ids[unsafe_offset=lo])
