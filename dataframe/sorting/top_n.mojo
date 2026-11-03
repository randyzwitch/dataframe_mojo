"""DuckDB 2.0's top-N (execution/operator/order/physical_top_n.cpp).

`TopNHeap` keeps the best `heap_size` key records seen, the worst on top
(a max-heap under the key's order, as `std::push_heap` over
`TopNEntry::operator<`). A record enters when the heap has room or when
it is better than the top (`EntryShouldBeAdded`).

`TopNBoundary` is TopNBoundaryValue: every thread whose heap is full
publishes its worst key, and the shared boundary keeps the tightest one.
Before a thread adds a chunk's rows it takes the boundary when it changed
(a try-lock, as `TryGetBoundaryValue`), and a row whose key is not better
than it cannot enter any heap's final top-N, so it is dropped
(`CheckBoundaryValues`). The boundary is compared as the normalized key,
with the same result as DuckDB's column-by-column comparison.

The operator's sink over a stream of chunks is `top_n_sink.TopNSink`.
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import Pointer, unsafe_memcpy

from dataframe.sorting.algorithms import Rec, SortKeys, pdqsort_branchless
from dataframe.sorting.sort_key import KeyBlock


struct _KeyArena(Movable):
    """StringHeap for variable keys: blocks never move, so a key's address
    stays valid while the arena lives."""

    var blocks: List[List[UInt8]]
    var used: Int

    def __init__(out self):
        self.blocks = List[List[UInt8]]()
        self.used = 0

    def add(mut self, address: Int, size: Int) -> Int:
        comptime BLOCK = 1 << 16
        if len(self.blocks) == 0 or self.used + size > len(
            self.blocks[len(self.blocks) - 1]
        ):
            self.blocks.append(
                List[UInt8](unsafe_uninit_length=max(BLOCK, size))
            )
            self.used = 0
        var block = Int(self.blocks[len(self.blocks) - 1].unsafe_ptr())
        var at = block + self.used
        unsafe_memcpy(
            dest=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at),
            src=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address),
            count=size,
        )
        self.used += size
        return at


# Cell layout of a TopNBoundary, in 64-bit words.
comptime _LOCK = 0
comptime _VERSION = 1
comptime _WORDS = 2
comptime _BLOB = 6
comptime _CAPACITY = 7
comptime _CELL_WORDS = 8


struct TopNBoundary(ImplicitlyCopyable, Movable):
    """TopNBoundaryValue: the tightest worst key among the threads' full
    heaps, shared by address (0: none). A variable key's bytes are kept
    in the cell's own buffer."""

    var address: Int

    def __init__(out self, address: Int):
        self.address = address

    @staticmethod
    def new() -> TopNBoundary:
        return TopNBoundary(external_call["calloc", Int](_CELL_WORDS, 8))

    @staticmethod
    def none() -> TopNBoundary:
        return TopNBoundary(0)

    def free(self):
        if self.address == 0:
            return
        var blob = self._word(_BLOB)[]
        if blob != 0:
            _ = external_call["free", NoneType](Int(blob))
        _ = external_call["free", NoneType](self.address)

    @always_inline
    def _word(self, k: Int) -> Pointer[UInt64, MutAnyOrigin]:
        return Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.address + 8 * k
        )

    @always_inline
    def _atomic(self, k: Int) -> Pointer[Atomic[Int64], MutAnyOrigin]:
        return Pointer[Atomic[Int64], MutAnyOrigin](
            unsafe_from_address=self.address + 8 * k
        )

    def _lock(self):
        while True:
            var free = Int64(0)
            if self._atomic(_LOCK)[].compare_exchange(free, Int64(1)):
                return

    def _try_lock(self) -> Bool:
        var free = Int64(0)
        return self._atomic(_LOCK)[].compare_exchange(free, Int64(1))

    def _unlock(self):
        self._atomic(_LOCK)[].store(0)

    def version(self) -> Int64:
        return self._atomic(_VERSION)[].load()

    def publish[W: Int, VAR: Bool](self, rec: Rec):
        """UpdateValue: take `rec` when there is no boundary yet or it is
        better than the boundary."""
        if self.address == 0:
            return
        comptime K = SortKeys[W, VAR]
        self._lock()
        if self.version() != 0:
            var current = Rec(0)
            comptime for k in range(K.R):
                current[k] = self._word(_WORDS + k)[]
            if not K.less(rec, current):
                self._unlock()
                return
        var stored = rec
        comptime if VAR:
            var size = Int(rec[1])
            if Int(self._word(_CAPACITY)[]) < size:
                var old = self._word(_BLOB)[]
                if old != 0:
                    _ = external_call["free", NoneType](Int(old))
                self._word(_BLOB)[] = UInt64(
                    external_call["malloc", Int](max(size, 64))
                )
                self._word(_CAPACITY)[] = UInt64(max(size, 64))
            var blob = Int(self._word(_BLOB)[])
            unsafe_memcpy(
                dest=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=blob),
                src=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(rec[2])
                ),
                count=size,
            )
            stored[2] = UInt64(blob)
        comptime for k in range(K.R):
            self._word(_WORDS + k)[] = stored[k]
        _ = self._atomic(_VERSION)[].fetch_add(1)
        self._unlock()


struct _BoundaryView(Movable):
    """A thread's copy of the shared boundary (`boundary_val`)."""

    var version: Int64
    var rec: Rec
    var blob: List[UInt8]

    def __init__(out self):
        self.version = 0
        self.rec = Rec(0)
        self.blob = List[UInt8]()

    def refresh[W: Int, VAR: Bool](mut self, shared: TopNBoundary):
        """TryGetBoundaryValue: an older copy stays valid when taking the
        new one would block."""
        if shared.address == 0 or shared.version() == self.version:
            return
        if not shared._try_lock():
            return
        comptime K = SortKeys[W, VAR]
        self.version = shared.version()
        comptime for k in range(K.R):
            self.rec[k] = shared._word(_WORDS + k)[]
        comptime if VAR:
            var size = Int(self.rec[1])
            if len(self.blob) < size:
                self.blob = List[UInt8](unsafe_uninit_length=size)
            unsafe_memcpy(
                dest=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(self.blob.unsafe_ptr())
                ),
                src=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(shared._word(_BLOB)[])
                ),
                count=size,
            )
            self.rec[2] = UInt64(Int(self.blob.unsafe_ptr()))
        shared._unlock()


struct TopNHeap(Movable):
    """TopNHeap: a max-heap of the best `heap_size` key records seen (the
    worst on top). The record layout is a parameter of each method."""

    var heap_size: Int
    var records: List[UInt64]
    var size: Int
    var arena: _KeyArena
    var view: _BoundaryView
    # Entries added since the keys were last compacted (heap_data's size).
    var added: Int

    def __init__(out self, heap_size: Int, record_words: Int):
        self.heap_size = heap_size
        self.records = List[UInt64](
            length=max(1, (heap_size + 1) * record_words), fill=0
        )
        self.size = 0
        self.arena = _KeyArena()
        self.view = _BoundaryView()
        self.added = 0

    def keys[W: Int, VAR: Bool](self) -> SortKeys[W, VAR]:
        return SortKeys[W, VAR](Int(self.records.unsafe_ptr()))

    def reduce_threshold(self) -> Int:
        """ReduceThreshold."""
        return max(5 * 2048, 2 * self.heap_size)

    def full(self) -> Bool:
        return self.heap_size > 0 and self.size >= self.heap_size

    @always_inline
    def should_add[W: Int, VAR: Bool](self, rec: Rec) -> Bool:
        """EntryShouldBeAdded, after the boundary check."""
        comptime K = SortKeys[W, VAR]
        if self.heap_size == 0:
            return False
        if self.view.version != 0 and not K.less(rec, self.view.rec):
            return False
        if self.size < self.heap_size:
            return True
        return K.less(rec, self.keys[W, VAR]().get(0))

    def _sift_up[W: Int, VAR: Bool](self, i_in: Int):
        comptime K = SortKeys[W, VAR]
        var k = self.keys[W, VAR]()
        var i = i_in
        var value = k.get(i)
        while i > 0:
            var parent = (i - 1) // 2
            var p = k.get(parent)
            if not K.less(p, value):
                break
            k.put(i, p)
            i = parent
        k.put(i, value)

    def _sift_down[W: Int, VAR: Bool](self, i_in: Int):
        comptime K = SortKeys[W, VAR]
        var k = self.keys[W, VAR]()
        var i = i_in
        var value = k.get(i)
        while True:
            var child = 2 * i + 1
            if child >= self.size:
                break
            var c = k.get(child)
            if child + 1 < self.size:
                var right = k.get(child + 1)
                if K.less(c, right):
                    child += 1
                    c = right
            if not K.less(value, c):
                break
            k.put(i, c)
            i = child
        k.put(i, value)

    def add[W: Int, VAR: Bool](mut self, var rec: Rec):
        """AddEntryToHeap: replace the worst entry once full."""
        comptime if VAR:
            rec[2] = UInt64(self.arena.add(Int(rec[2]), Int(rec[1])))
        self.added += 1
        if self.size >= self.heap_size:
            self.keys[W, VAR]().put(0, rec)
            self._sift_down[W, VAR](0)
            return
        self.keys[W, VAR]().put(self.size, rec)
        self.size += 1
        self._sift_up[W, VAR](self.size - 1)

    def sink[W: Int, VAR: Bool](mut self, block: KeyBlock):
        """Every record of a chunk that can enter, with its own row as the
        payload."""
        var k = SortKeys[W, VAR](block.address())
        for i in range(block.count):
            var rec = k.get(i)
            if self.should_add[W, VAR](rec):
                self.add[W, VAR](rec)

    def compact_keys[W: Int, VAR: Bool](mut self):
        """Reduce's key half: only the live entries' key bytes stay."""
        comptime if VAR:
            var arena = _KeyArena()
            var k = self.keys[W, VAR]()
            for i in range(self.size):
                var rec = k.get(i)
                rec[2] = UInt64(arena.add(Int(rec[2]), Int(rec[1])))
                k.put(i, rec)
            # The old arena goes with `arena` once every key is copied.
            swap(self.arena, arena)
        self.added = self.size

    def finalize[W: Int, VAR: Bool](self):
        """Finalize: the entries in key order."""
        pdqsort_branchless(self.keys[W, VAR](), 0, self.size)

    def combine[W: Int, VAR: Bool](mut self, other: TopNHeap):
        """Combine: another thread's finalized heap, best first, until an
        entry can no longer enter."""
        var k = other.keys[W, VAR]()
        for i in range(other.size):
            var rec = k.get(i)
            if not self.should_add[W, VAR](rec):
                break
            self.add[W, VAR](rec)
        self.compact_keys[W, VAR]()

    def publish[W: Int, VAR: Bool](mut self, shared: TopNBoundary):
        """A full heap's worst key to the shared boundary, then the
        boundary back (it may be another thread's, tighter)."""
        if self.full():
            shared.publish[W, VAR](self.keys[W, VAR]().get(0))
        self.view.refresh[W, VAR](shared)

    def payloads[W: Int, VAR: Bool](self) -> List[Int]:
        var k = self.keys[W, VAR]()
        var rows = List[Int](capacity=self.size)
        for i in range(self.size):
            rows.append(k.payload(i))
        return rows^

    def renumber[W: Int, VAR: Bool](self):
        """Each entry's payload becomes its position."""
        comptime K = SortKeys[W, VAR]
        var k = self.keys[W, VAR]()
        for i in range(self.size):
            Pointer[UInt64, MutAnyOrigin](unsafe_from_address=k.addr(i))[
                unsafe_offset=K.R - 1
            ] = UInt64(i)
