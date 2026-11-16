"""Large tables on transparent huge pages, kept for reuse once freed.

A hash table of a few MB is allocated fresh for every build: the runtime's
allocator returns freed spans of that size to the kernel, so each build
first-touches every 4 KB page of its tables, about 12K page faults for a
1.5M-key index at 8 threads, most of the build's time once hashing and
scatter were cheap (#525). With transparent huge pages in `madvise` mode
(the Linux default), asking for them on the table's range before the first
touch makes the kernel back it with 2 MB pages: two faults for a 4 MB
table instead of 1024, and a probe's scattered reads miss the TLB less.

Huge pages make the faults fewer, not free: the kernel still zeroes every
byte of a fresh range before the first write lands, and a table that is
built, probed and dropped is often written once, so zeroing costs about as
much as the build's own stores. The next operator, or the next run of the
same query, then asks for a table of about the same size and pays again.

So freed large tables do not go back to the runtime's allocator at once.
`huge_list`, `huge_uninit` and `huge_capacity` take a retained block that
fits before asking for fresh memory, and `release_huge` (also run when a
column's buffer is dropped) hands storage they gave out back to a
process-wide pool of retained blocks, still mapped and already faulted in.
A block left unused for `RETAIN_NANOSECONDS` is freed, so an idle process
gives its memory back, and the pool never holds more than `_retain_limit`
bytes. Storage that is never released is freed the ordinary way: the pool
is an optimization that a caller can always skip.
"""
from std.atomic import Atomic
from std.collections import Dict
from std.ffi import _get_global, external_call
from std.math import align_down, align_up
from std.memory import Allocation, Layout, Pointer, alloc, dealloc
from std.sys import CompilationTarget, size_of
from std.time import monotonic

# Linux `MADV_HUGEPAGE`.
comptime _MADV_HUGEPAGE = 14
comptime HUGE_PAGE_BYTES = 2 << 20
comptime _HUGE_PAGE = HUGE_PAGE_BYTES
comptime _PAGE = 4096

# How long a freed block stays mapped for reuse: long enough to cover the
# gap between one operator freeing its tables and the next one (or the
# next query) asking, short enough that an idle process soon returns the
# memory.
comptime RETAIN_NANOSECONDS = 1_000_000_000

# Linux `sysconf` names.
comptime _SC_PAGESIZE = 30
comptime _SC_PHYS_PAGES = 85


def advise_huge_pages(address: Int, bytes: Int):
    """Ask for huge pages on the whole pages within `bytes` from `address`.
    Only a range of at least one huge page can hold one; elsewhere, or
    where the kernel declines, this changes nothing."""
    comptime if not CompilationTarget.is_linux():
        return
    if bytes < _HUGE_PAGE:
        return
    var start = align_up(address, _PAGE)
    var end = align_down(address + bytes, _PAGE)
    if end - start >= _HUGE_PAGE:
        _ = external_call["madvise", Int32](start, end - start, _MADV_HUGEPAGE)


@fieldwise_init
struct _Retained(Copyable, ImplicitlyCopyable, Movable):
    """A freed block: its address, its size in bytes and when it was
    freed (`monotonic` nanoseconds)."""

    var address: Int
    var bytes: Int
    var freed_at: Int


def _retain_limit() -> Int:
    """The most bytes the pool keeps: a quarter of physical memory, so
    retained blocks never crowd out the tables that are live."""
    comptime if not CompilationTarget.is_linux():
        return 0
    var pages = Int(external_call["sysconf", Int64](Int32(_SC_PHYS_PAGES)))
    var size = Int(external_call["sysconf", Int64](Int32(_SC_PAGESIZE)))
    if pages <= 0 or size <= 0:
        return 0
    return pages * size // 4


def _free_block(address: Int, bytes: Int):
    """Give a retained block back to the runtime's allocator. Its size is
    only bookkeeping: the allocator frees by address."""
    dealloc(
        Allocation[UInt8](
            unsafe_owned_ptr=Pointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=address
            ),
            layout=Layout[UInt8](length=bytes),
        )
    )


struct _BlockPool(Movable):
    """Retained blocks, the live blocks handed out, and a spin lock over
    them. A block is a whole table, taken and freed a few times per
    operator rather than per row, so one short critical section serves
    every thread.

    Only storage that `huge_list`, `huge_uninit` or `huge_capacity` handed
    out comes back (`owned`): those are the requests that look here first,
    so what is retained is what the next request of the same kind reuses.
    Retaining storage that ordinary allocations would have reused instead
    takes it from the runtime's allocator, which then faults fresh pages
    in for them."""

    var lock: Atomic[Int64]
    var blocks: List[_Retained]
    # Address to bytes of every live block handed out.
    var owned: Dict[Int, Int]
    var retained: Int
    var limit: Int

    def __init__(out self):
        self.lock = Atomic[Int64](0)
        self.blocks = List[_Retained]()
        self.owned = Dict[Int, Int]()
        self.retained = 0
        self.limit = _retain_limit()

    def _acquire(mut self):
        while True:
            var expected = Int64(0)
            if self.lock.compare_exchange(expected, 1):
                return
            _ = external_call["sched_yield", Int32]()

    def _release(mut self):
        self.lock.store(0)

    def _drop_first(mut self, count: Int, mut out: List[_Retained]):
        """Move the `count` oldest blocks to `out`, for the caller to free
        after unlocking."""
        if count == 0:
            return
        for i in range(count):
            out.append(self.blocks[i])
            self.retained -= self.blocks[i].bytes
        var kept = List[_Retained](capacity=len(self.blocks) - count)
        for i in range(count, len(self.blocks)):
            kept.append(self.blocks[i])
        self.blocks = kept^

    def _expire(mut self, now: Int, mut out: List[_Retained]):
        """Move blocks unused for RETAIN_NANOSECONDS to `out`. Blocks are
        kept in the order freed, so the expired ones lead."""
        var cut = 0
        while (
            cut < len(self.blocks)
            and now - self.blocks[cut].freed_at > RETAIN_NANOSECONDS
        ):
            cut += 1
        self._drop_first(cut, out)

    def take(mut self, bytes: Int) -> _Retained:
        """The smallest retained block of at least `bytes` and at most
        twice that (larger blocks are left for larger tables), or an empty
        record (address 0)."""
        var expired = List[_Retained]()
        var found = _Retained(0, 0, 0)
        self._acquire()
        self._expire(monotonic(), expired)
        var best = -1
        for i in range(len(self.blocks)):
            var size = self.blocks[i].bytes
            if size >= bytes and size <= 2 * bytes:
                if best < 0 or size < self.blocks[best].bytes:
                    best = i
        if best >= 0:
            found = self.blocks.pop(best)
            self.retained -= found.bytes
            self.owned[found.address] = found.bytes
        self._release()
        for block in expired:
            _free_block(block.address, block.bytes)
        return found

    def hand_out(mut self, address: Int, bytes: Int):
        """Record fresh storage handed out, to retain once freed."""
        self._acquire()
        self.owned[address] = bytes
        self._release()

    def give(mut self, address: Int, bytes: Int) -> Bool:
        """Retain a freed block this pool handed out, and whether it did;
        the oldest blocks are freed when the pool would exceed its limit.
        """
        var expired = List[_Retained]()
        self._acquire()
        if self.owned.pop(address, 0) == 0:
            self._release()
            return False
        var now = monotonic()
        self._expire(now, expired)
        if bytes > self.limit:
            expired.append(_Retained(address, bytes, now))
        else:
            self.blocks.append(_Retained(address, bytes, now))
            self.retained += bytes
            var cut = 0
            var over = self.retained - self.limit
            while over > 0:
                over -= self.blocks[cut].bytes
                cut += 1
            self._drop_first(cut, expired)
        self._release()
        for block in expired:
            _free_block(block.address, block.bytes)
        return True


def _pool_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    var pointer = alloc(Layout[_BlockPool](length=1)).unsafe_leak()
    pointer.unsafe_write(_BlockPool())
    return pointer.unsafe_bitcast[NoneType]()


def _pool_keep(address: Optional[Pointer[NoneType, MutUntrackedOrigin]]):
    # Tables may still be released while the process tears down.
    pass


def _block_pool() -> ref[MutAnyOrigin] _BlockPool:
    var address = _get_global["DATAFRAME_BLOCK_POOL", _pool_init, _pool_keep]()
    return Pointer[_BlockPool, MutAnyOrigin](
        unsafe_from_address=Int(address.value())
    )[]


def _adopt[T: Copyable](block: _Retained, length: Int) -> List[T]:
    """A list of `length` uninitialized elements over a retained block,
    which it owns from now on (freed by the runtime's allocator, or
    released to the pool again). The block's bytes become the list's
    capacity; the runtime's allocator frees by address, so a block freed
    by one table's type serves another's."""
    var out = List[T]()
    # An empty list holds a placeholder for storage; swap the block in.
    _ = out._storage^.unsafe_leak()
    out._storage = Allocation[T](
        unsafe_owned_ptr=Pointer[T, MutUntrackedOrigin](
            unsafe_from_address=block.address
        ),
        layout=Layout[T](length=block.bytes // size_of[T]()),
    )
    out._len = length
    return out^


def release_huge[T: Copyable & Deinitable](var table: List[T]):
    """Done with `table`: destroy its elements and, when its storage came
    from `huge_list`, `huge_uninit` or `huge_capacity`, keep it for the
    next such request instead of freeing it. Other storage is freed as
    usual."""
    var bytes = table.capacity() * size_of[T]()
    if bytes < _HUGE_PAGE:
        return
    table.clear()
    if _block_pool().give(Int(table.unsafe_ptr()), bytes):
        # The pool owns the storage now; the list must not free it.
        _ = table.unsafe_take_allocation().unsafe_leak()


def _fresh(address: Int, bytes: Int):
    """Fresh storage of `bytes` at `address`: ask for huge pages before
    its first touch, and record it for retaining once freed."""
    if bytes < _HUGE_PAGE:
        return
    advise_huge_pages(address, bytes)
    _block_pool().hand_out(address, bytes)


def huge_list[T: Copyable](length: Int, fill: T) -> List[T]:
    """`length` copies of `fill`, on huge pages when the list is large
    enough to hold one and the kernel grants them."""
    var out = huge_uninit[T](length)
    var data = out.unsafe_ptr()
    for i in range(length):
        data.unsafe_offset(i).unsafe_write(fill.copy())
    return out^


def huge_uninit[T: Copyable](length: Int) -> List[T]:
    """`length` uninitialized elements, every one of which the caller
    writes, on huge pages when the list is large enough to hold one and
    the kernel grants them."""
    var bytes = length * size_of[T]()
    var block = _block_pool().take(bytes) if bytes >= _HUGE_PAGE else (
        _Retained(0, 0, 0)
    )
    if block.address != 0:
        return _adopt[T](block, length)
    var out = List[T](unsafe_uninit_length=length)
    _fresh(Int(out.unsafe_ptr()), out.capacity() * size_of[T]())
    return out^


def huge_capacity[T: Copyable](length: Int) -> List[T]:
    """An empty list with room for `length` elements that are appended or
    written through its pointer, on huge pages when the room can hold one
    and the kernel grants them."""
    var bytes = length * size_of[T]()
    var block = _block_pool().take(bytes) if bytes >= _HUGE_PAGE else (
        _Retained(0, 0, 0)
    )
    if block.address != 0:
        return _adopt[T](block, 0)
    var out = List[T](capacity=length)
    _fresh(Int(out.unsafe_ptr()), out.capacity() * size_of[T]())
    return out^
