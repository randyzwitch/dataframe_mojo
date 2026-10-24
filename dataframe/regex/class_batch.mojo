"""Prove Unicode class membership in batches using existing directories.

False may mean that the lookup budget left no directory. True always means
canonical membership, so a caller can safely consume that character.
Inputs must be Unicode scalars in the selected UTF8 width.
"""
from .program import Program

comptime _CP_MASK: UInt64 = (UInt64(1) << 21) - 1


@always_inline
def _contains[
    lanes: Int
](interval: UInt64, cp: SIMD[DType.uint32, lanes]) -> SIMD[DType.bool, lanes]:
    return (cp - UInt32(interval & _CP_MASK)).le(
        UInt32((interval >> 21) & _CP_MASK)
    )


@always_inline
def class_batch[
    width: Int, lanes: Int
](program: Program, id: Int, cp: SIMD[DType.uint32, lanes]) -> SIMD[
    DType.bool, lanes
]:
    var p = program.class_ascii.unsafe_ptr()
    var classes = len(program.class_ranges)
    comptime if width == 2:
        var bit = cp - 128
        var indices = (bit >> 6).cast[DType.int]() + 2 * classes + 30 * id
        var words = p.unsafe_gather[width=lanes](indices)
        return ((words >> (bit & 63).cast[DType.uint64]()) & 1).ne(0)
    else:
        var root = p[unsafe_offset=32 * classes + id]
        var kind = root >> 62
        if root == 1 or kind == 0:
            return SIMD[DType.bool, lanes]()
        if kind == 1:
            return _contains(root, cp)
        if kind == 2:
            var base = Int(UInt32(root))
            return _contains(p[unsafe_offset=base], cp) | _contains(
                p[unsafe_offset=base + 1], cp
            )
        var plane = cp >> 16
        var entries = p.unsafe_gather[width=lanes](
            (plane >> 1).cast[DType.int]() + Int(UInt32(root))
        )
        var pages = (entries >> ((plane & 1) * 32).cast[DType.uint64]()).cast[
            DType.uint32
        ]()
        var page = (cp >> 8) & 255
        var has_table = pages.gt(1)
        # A zero/one plane has no page table. Mask these loads explicitly.
        entries = p.unsafe_gather[width=lanes](
            pages.cast[DType.int]() + (page >> 1).cast[DType.int](),
            has_table,
            0,
        )
        var info = (entries >> ((page & 1) * 32).cast[DType.uint64]()).cast[
            DType.uint32
        ]()
        var uniform = pages.eq(1) | (has_table & info.eq(1))
        var mixed = has_table & info.gt(1)
        if not mixed.reduce_or():
            return uniform
        var lo = mixed.select((info >> 8) - 1, SIMD[DType.uint32, lanes](0))
        var hi = mixed.select(
            lo + (info & 255) + 1, SIMD[DType.uint32, lanes](0)
        )
        var ranges = program.class_ranges.unsafe_ptr()[
            unsafe_offset=id
        ].unsafe_ptr()
        var active = (hi - lo).gt(1) & mixed
        # Each mixed page intersects at most 256 canonical intervals. Find
        # the last lower endpoint <= cp, independently in each active lane.
        while active.reduce_or():
            var mid = (lo + hi) >> 1
            var point = ranges.unsafe_gather[width=lanes](
                mid.cast[DType.int]() * 2, active, 0
            )
            lo = (active & point.le(cp)).select(mid, lo)
            hi = (active & point.gt(cp)).select(mid, hi)
            active = (hi - lo).gt(1) & mixed
        var indices = lo.cast[DType.int]() * 2
        var low = ranges.unsafe_gather[width=lanes](indices, mixed, 0)
        var high = ranges.unsafe_gather[width=lanes](indices + 1, mixed, 0)
        return uniform | (mixed & (cp - low).le(high - low))
