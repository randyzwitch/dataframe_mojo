"""Large tables on transparent huge pages.

A hash table of a few MB is allocated fresh for every build: the runtime's
allocator returns freed spans of that size to the kernel, so each build
first-touches every 4 KB page of its tables, about 12K page faults for a
1.5M-key index at 8 threads, most of the build's time once hashing and
scatter were cheap (#525). With transparent huge pages in `madvise` mode
(the Linux default), asking for them on the table's range before the first
touch makes the kernel back it with 2 MB pages: two faults for a 4 MB
table instead of 1024, and a probe's scattered reads miss the TLB less.
"""
from std.ffi import external_call
from std.sys import CompilationTarget, size_of

# Linux `MADV_HUGEPAGE`.
comptime _MADV_HUGEPAGE = 14
comptime _HUGE_PAGE = 2 << 20
comptime _PAGE = 4096


def advise_huge_pages(address: Int, bytes: Int):
    """Ask for huge pages on the whole pages within `bytes` from `address`.
    Only a range of at least one huge page can hold one; elsewhere, or
    where the kernel declines, this changes nothing."""
    comptime if not CompilationTarget.is_linux():
        return
    if bytes < _HUGE_PAGE:
        return
    var start = (address + _PAGE - 1) & ~(_PAGE - 1)
    var end = (address + bytes) & ~(_PAGE - 1)
    if end - start >= _HUGE_PAGE:
        _ = external_call["madvise", Int32](start, end - start, _MADV_HUGEPAGE)


def huge_list[T: Copyable & Movable](length: Int, fill: T) -> List[T]:
    """`length` copies of `fill`, on huge pages when the list is large
    enough to hold one and the kernel grants them."""
    var out = List[T](unsafe_uninit_length=length)
    var data = out.unsafe_ptr()
    advise_huge_pages(Int(data), length * size_of[T]())
    for i in range(length):
        data.unsafe_offset(i).unsafe_write(fill.copy())
    return out^


def huge_uninit[T: Copyable & Movable](length: Int) -> List[T]:
    """`length` uninitialized elements, every one of which the caller
    writes, on huge pages when the list is large enough to hold one and
    the kernel grants them."""
    var out = List[T](unsafe_uninit_length=length)
    advise_huge_pages(Int(out.unsafe_ptr()), length * size_of[T]())
    return out^
