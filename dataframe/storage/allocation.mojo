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

Retained blocks are still memory the process holds, so they count toward
the memory limit (`memory_limit`). When the limit would be crossed, every
retained block is freed
before an `out of memory` error is raised (`check_memory`): memory kept
only for speed never decides whether a query fits. The row blocks of hash
tables and aggregates have a pool of their own here (`_RowBlockPool`),
under the same rules.
"""
from std.atomic import Atomic
from std.collections import Dict
from std.ffi import _get_global, external_call
from std.math import align_down, align_up
from std.memory import Allocation, Layout, Pointer, alloc, dealloc
from std.sys import CompilationTarget, size_of
from std.time import monotonic

from dataframe.storage.memory_limit import (
    charge,
    out_of_memory,
    over_limit,
    over_limit_reserved,
    refresh_resident,
    uncharge,
)

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

    # Every thread shares the one pool, so its methods reach it by address
    # (`_block_pool`) rather than as a `mut self` argument: a `mut` argument
    # promises the callee sole access to the value's memory, which lets the
    # compiler move the list and map reads and writes across the lock's
    # atomics, and two threads could then change them at once. The helpers
    # that take `mut self` run only under the lock and are not inlined, so
    # their changes are complete before it is released.

    @staticmethod
    def _acquire():
        ref lock = _block_pool().lock
        while True:
            var expected = Int64(0)
            if lock.compare_exchange(expected, 1):
                return
            _ = external_call["sched_yield", Int32]()

    @staticmethod
    def _release():
        _block_pool().lock.store(0)

    def _drop_first(mut self, count: Int, mut out: List[_Retained]):
        """Move the `count` oldest blocks to `out`, for the caller to free
        after unlocking."""
        if count == 0:
            return
        for i in range(count):
            out.append(self.blocks[i])
            self.retained -= self.blocks[i].bytes
            uncharge(self.blocks[i].bytes)
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

    @staticmethod
    def take(bytes: Int) -> _Retained:
        """The smallest retained block of at least `bytes` and at most
        twice that (larger blocks are left for larger tables), or an empty
        record (address 0)."""
        var expired = List[_Retained]()
        _BlockPool._acquire()
        var found = _block_pool()._take_locked(bytes, expired)
        _BlockPool._release()
        for block in expired:
            _free_block(block.address, block.bytes)
        return found

    @no_inline
    def _take_locked(
        mut self, bytes: Int, mut expired: List[_Retained]
    ) -> _Retained:
        self._expire(monotonic(), expired)
        var best = -1
        for i in range(len(self.blocks)):
            var size = self.blocks[i].bytes
            if size >= bytes and size <= 2 * bytes:
                if best < 0 or size < self.blocks[best].bytes:
                    best = i
        if best < 0:
            return _Retained(0, 0, 0)
        var found = self.blocks.pop(best)
        self.retained -= found.bytes
        uncharge(found.bytes)
        self.owned[found.address] = found.bytes
        return found

    @staticmethod
    def hand_out(address: Int, bytes: Int):
        """Record fresh storage handed out, to retain once freed."""
        _BlockPool._acquire()
        _block_pool()._hand_out_locked(address, bytes)
        _BlockPool._release()

    @no_inline
    def _hand_out_locked(mut self, address: Int, bytes: Int):
        self.owned[address] = bytes

    @staticmethod
    def give(address: Int, bytes: Int) -> Bool:
        """Retain a freed block this pool handed out, and whether it did;
        the oldest blocks are freed when the pool would exceed its limit.
        """
        var expired = List[_Retained]()
        _BlockPool._acquire()
        var kept = _block_pool()._give_locked(address, bytes, expired)
        _BlockPool._release()
        for block in expired:
            _free_block(block.address, block.bytes)
        return kept

    @no_inline
    def _give_locked(
        mut self, address: Int, bytes: Int, mut expired: List[_Retained]
    ) -> Bool:
        if self.owned.pop(address, 0) == 0:
            return False
        var now = monotonic()
        self._expire(now, expired)
        var limit = self.limit
        if bytes > limit:
            expired.append(_Retained(address, bytes, now))
        else:
            self.blocks.append(_Retained(address, bytes, now))
            self.retained += bytes
            charge(bytes)
            var cut = 0
            var over = self.retained - limit
            while over > 0:
                over -= self.blocks[cut].bytes
                cut += 1
            self._drop_first(cut, expired)
        return True

    @staticmethod
    def drain():
        """Free every retained block."""
        var freed = List[_Retained]()
        _BlockPool._acquire()
        ref pool = _block_pool()
        pool._drop_first(len(pool.blocks), freed)
        _BlockPool._release()
        for block in freed:
            _free_block(block.address, block.bytes)


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
    if _BlockPool.give(Int(table.unsafe_ptr()), bytes):
        # The pool owns the storage now; the list must not free it.
        _ = table.unsafe_take_allocation().unsafe_leak()


def _fresh(address: Int, bytes: Int):
    """Fresh storage of `bytes` at `address`: ask for huge pages before
    its first touch, and record it for retaining once freed."""
    if bytes < _HUGE_PAGE:
        return
    advise_huge_pages(address, bytes)
    _BlockPool.hand_out(address, bytes)


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
    var block = _BlockPool.take(bytes) if bytes >= _HUGE_PAGE else (
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
    var block = _BlockPool.take(bytes) if bytes >= _HUGE_PAGE else (
        _Retained(0, 0, 0)
    )
    if block.address != 0:
        return _adopt[T](block, 0)
    var out = List[T](capacity=length)
    _fresh(Int(out.unsafe_ptr()), out.capacity() * size_of[T]())
    return out^


# -- Row blocks ---------------------------------------------------------------

# Freed row blocks the pool keeps per worker thread: one batch of blocks,
# what a thread's cache of freed blocks holds before it passes them on.
comptime ROW_BLOCKS_KEPT_PER_THREAD = 128


struct _RowBlockPool(Movable):
    """Freed row blocks (all one size, pages already touched) and a spin
    lock over them. Taking or returning a block holds the lock for one list
    push or pop.

    Row blocks are all one size, so a freed one can serve the next request
    as it is. Freed row blocks come here instead of going back to the
    general allocator, which would return their pages to the system and
    take a global lock for each block: many threads each starting a few
    partitions would wait on that lock, then fault the pages in again. The
    pool keeps at most ROW_BLOCKS_KEPT_PER_THREAD blocks for each worker
    thread, the most freed memory a thread holds before handing it on, and
    frees the rest. Kept blocks count toward the limit."""

    var lock: Atomic[Int64]
    var free: List[List[UInt8]]

    def __init__(out self):
        self.lock = Atomic[Int64](0)
        self.free = List[List[UInt8]]()

    # Every thread shares the one pool, so its methods reach it by address
    # (`_row_block_pool`) rather than as a `mut self` argument: a `mut`
    # argument promises the callee sole access to the value's memory, which
    # lets the compiler move the list's reads and writes across the lock's
    # atomics, and two threads could then change it at once. The list
    # changes run under the lock in functions that are not inlined, so they
    # are complete before it is released.

    @staticmethod
    def _acquire():
        ref lock = _row_block_pool().lock
        while True:
            var expected = Int64(0)
            if lock.compare_exchange(expected, 1):
                return

    @staticmethod
    def _release():
        _row_block_pool().lock.store(0)

    @staticmethod
    def take(bytes: Int) -> List[UInt8]:
        """A free block of `bytes`, or a new one when the pool has none."""
        _RowBlockPool._acquire()
        var block = _row_block_pool()._pop_locked(bytes)
        _RowBlockPool._release()
        if block:
            return block.take()
        return List[UInt8](unsafe_uninit_length=bytes)

    @no_inline
    def _pop_locked(mut self, bytes: Int) -> Optional[List[UInt8]]:
        if len(self.free) > 0 and len(self.free[len(self.free) - 1]) == bytes:
            var block = self.free.pop()
            uncharge(len(block))
            return block^
        return None

    @staticmethod
    def give(var block: List[UInt8], workers: Int):
        """Keep a freed block for reuse, unless the pool is full (then it
        is freed when `block` goes out of scope). `workers` is the worker
        thread count the pool's size follows."""
        var bytes = len(block)
        if bytes == 0:
            return
        var most = ROW_BLOCKS_KEPT_PER_THREAD * workers
        _RowBlockPool._acquire()
        _row_block_pool()._push_locked(block^, most)
        _RowBlockPool._release()

    @no_inline
    def _push_locked(mut self, var block: List[UInt8], most: Int):
        if len(self.free) < most and (
            len(self.free) == 0 or len(self.free[0]) == len(block)
        ):
            charge(len(block))
            self.free.append(block^)

    @staticmethod
    def drain():
        """Free every kept block."""
        _RowBlockPool._acquire()
        var freed = _row_block_pool()._take_all_locked()
        _RowBlockPool._release()
        _ = freed^

    @no_inline
    def _take_all_locked(mut self) -> List[List[UInt8]]:
        var freed = List[List[UInt8]]()
        swap(freed, self.free)
        for block in freed:
            uncharge(len(block))
        return freed^


def _row_pool_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    var pointer = alloc(Layout[_RowBlockPool](length=1)).unsafe_leak()
    pointer.unsafe_write(_RowBlockPool())
    return pointer.unsafe_bitcast[NoneType]()


def _row_block_pool() -> ref[MutAnyOrigin] _RowBlockPool:
    var address = _get_global[
        "DATAFRAME_ROW_BLOCK_POOL", _row_pool_init, _pool_keep
    ]()
    return Pointer[_RowBlockPool, MutAnyOrigin](
        unsafe_from_address=Int(address.value())
    )[]


def take_row_block(bytes: Int) -> List[UInt8]:
    """A row block of `bytes`: a freed one when the pool has one, else new
    (uninitialized either way)."""
    return _RowBlockPool.take(bytes)


def give_row_block(var block: List[UInt8], workers: Int):
    """Done with a row block nothing else references: keep it for the next
    `take_row_block`, or free it."""
    _RowBlockPool.give(block^, workers)


# -- The memory limit ------------------------------------------------------------


def release_retained():
    """Free every block the pools keep for reuse."""
    _BlockPool.drain()
    _RowBlockPool.drain()


@always_inline
def check_memory(extra: Int = 0) raises:
    """Raise an `out of memory` error when holding `extra` more bytes would
    go over the memory limit, after freeing the blocks kept for reuse.
    Call it before allocating `extra` bytes whose size is known, and with
    no argument between pieces of work."""
    if over_limit(extra):
        _check_memory_slow(extra)


@no_inline
def _check_memory_slow(extra: Int) raises:
    release_retained()
    refresh_resident()
    if over_limit(extra):
        raise out_of_memory(extra)


# -- Tables counted toward the memory limit -------------------------------------


def counted_huge_list[T: Copyable](length: Int, fill: T) -> List[T]:
    """`huge_list`, with the list's bytes counted toward the memory limit
    until it is given to `release_counted_huge`, which must free it."""
    var out = huge_list[T](length, fill)
    charge(out.capacity() * size_of[T]())
    return out^


def counted_huge_uninit[T: Copyable](length: Int) -> List[T]:
    """`huge_uninit`, with the list's bytes counted toward the memory
    limit until it is given to `release_counted_huge`, which must free it.
    """
    var out = huge_uninit[T](length)
    charge(out.capacity() * size_of[T]())
    return out^


def release_counted_huge[T: Copyable & Deinitable](var table: List[T]):
    """`release_huge` for a list from `counted_huge_list` or
    `counted_huge_uninit` (or an empty one): its bytes no longer count."""
    uncharge(table.capacity() * size_of[T]())
    release_huge(table^)


def reserve_memory(bytes: Int) raises:
    """Count `bytes` toward the memory limit before allocating them, or
    raise an `out of memory` error when they do not fit (after freeing the
    blocks kept for reuse). Unlike `check_memory`, the bytes are counted at
    once, so threads reserving at the same time cannot all fit under the
    limit by each seeing room for its own share. The caller allocates them
    and gives them back with `uncharge` when freed."""
    charge(bytes)
    if not over_limit_reserved(bytes):
        return
    uncharge(bytes)
    release_retained()
    refresh_resident()
    charge(bytes)
    if over_limit_reserved(bytes):
        uncharge(bytes)
        raise out_of_memory(bytes)


struct MemoryReservation(Movable):
    """Bytes reserved against the memory limit for the life of this value
    (`reserve_memory`), for work whose allocations count themselves once
    made but are all made at once by several workers: the reservation
    keeps the workers from each seeing room for its own share. Keep it
    alive to the end of the work (`_ = reservation^`): a value is freed
    after its last use."""

    var bytes: Int

    def __init__(out self, bytes: Int) raises:
        reserve_memory(bytes)
        self.bytes = bytes

    def __deinit__(deinit self):
        uncharge(self.bytes)
