"""Exact DFA alphabet classification through the existing lookup directories."""
from .program import Program


@always_inline
def alphabet_batch[
    lanes: Int
](program: Program, cp: SIMD[DType.uint32, lanes]) -> SIMD[DType.int32, lanes]:
    """Classify code points, including directory-budget fallbacks.
    Mask every indirect read to the lanes whose directory entry owns it."""
    var count = len(program.boundaries)
    var cuts = program.boundaries.unsafe_ptr()
    var ids = program.interval_class.unsafe_ptr()
    var direct = cp.lt(UInt32(len(program.ascii_class)))
    var trailing = cp.ge(cuts[unsafe_offset=count - 1])
    var result = program.ascii_class.unsafe_ptr().unsafe_gather[width=lanes](
        direct.select(cp.cast[DType.int](), SIMD[DType.int, lanes](0)),
        direct,
        ids[unsafe_offset=count - 1],
    )
    var active = ~direct & ~trailing
    if not active.reduce_or():
        return result
    var lo = SIMD[DType.uint32, lanes](0)
    var hi = SIMD[DType.uint32, lanes](UInt32(count))
    if len(program.interval_class) >= count + 17:
        var indices = active.select(
            (cp >> 16).cast[DType.int]() + count, SIMD[DType.int, lanes](0)
        )
        var entries = ids.unsafe_gather[width=lanes](indices, active, 0)
        var uniform = active & entries.lt(0)
        result = uniform.select(-entries - 1, result)
        var has_pages = active & entries.gt(0)
        indices = has_pages.select(
            entries.cast[DType.int]() - 1 + ((cp >> 8) & 255).cast[DType.int](),
            SIMD[DType.int, lanes](0),
        )
        entries = ids.unsafe_gather[width=lanes](indices, has_pages, 0)
        var page_uniform = has_pages & entries.lt(0)
        result = page_uniform.select(-entries - 1, result)
        var mixed = has_pages & entries.gt(0)
        var info = entries.cast[DType.uint32]()
        lo = mixed.select(info >> 8, lo)
        hi = mixed.select(lo + (info & 255) + 1, hi)
        active &= ~uniform & ~page_uniform
    var search = active & (hi - lo).gt(1)
    while search.reduce_or():
        var mid = (lo + hi) >> 1
        var indices = search.select(
            mid.cast[DType.int](), SIMD[DType.int, lanes](0)
        )
        var point = cuts.unsafe_gather[width=lanes](indices, search, 0)
        lo = (search & point.le(cp)).select(mid, lo)
        hi = (search & point.gt(cp)).select(mid, hi)
        search = active & (hi - lo).gt(1)
    var indices = active.select(lo.cast[DType.int](), SIMD[DType.int, lanes](0))
    var found = ids.unsafe_gather[width=lanes](indices, active, 0)
    return active.select(found, result)
