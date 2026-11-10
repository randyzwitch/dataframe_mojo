"""Keep the best k normalized keys in a bounded heap.

The worst retained row forms the heap boundary. Reject candidates that
cannot improve it; replace the boundary when a better candidate arrives.
Sorting the retained keys produces the requested prefix. Keys include
null placement and, when stable ordering is requested, input row ids.
"""
from std.collections import BinaryHeap, Optional
from std.utils import Variant
from std.sys import size_of
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

    def publish[W: Int, VAR: Bool](self, rec: Rec) -> Bool:
        """UpdateValue: take `rec` when there is no boundary yet or it is
        better than the boundary; whether it was taken."""
        if self.address == 0:
            return False
        comptime K = SortKeys[W, VAR]
        self._lock()
        if self.version() != 0:
            var current = Rec(0)
            comptime for k in range(K.R):
                current[k] = self._word(_WORDS + k)[]
            if not K.less(rec, current):
                self._unlock()
                return False
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
        return True


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


struct _HeapRecord[W: Int, VAR: Bool](Comparable, Copyable, Movable):
    """A packed normalized key with the same ordering as its sort span."""

    comptime R = SortKeys[Self.W, Self.VAR].R
    var words: Array[UInt64, Self.R]

    def __init__(out self, rec: Rec):
        self.words = Array[UInt64, Self.R](fill=0)
        comptime for i in range(Self.R):
            self.words[i] = rec[i]

    def record(self) -> Rec:
        var rec = Rec(0)
        comptime for i in range(Self.R):
            rec[i] = self.words[i]
        return rec

    def __lt__(self, other: Self) -> Bool:
        return SortKeys[Self.W, Self.VAR].less(self.record(), other.record())

    def __le__(self, other: Self) -> Bool:
        return not other < self

    def __gt__(self, other: Self) -> Bool:
        return other < self

    def __ge__(self, other: Self) -> Bool:
        return not self < other

    def __eq__(self, other: Self) -> Bool:
        return not self < other and not other < self

    def __ne__(self, other: Self) -> Bool:
        return not self == other


comptime _RecordHeaps = Variant[
    BinaryHeap[_HeapRecord[1, False]],
    BinaryHeap[_HeapRecord[2, False]],
    BinaryHeap[_HeapRecord[3, False]],
    BinaryHeap[_HeapRecord[1, True]],
]


struct TopNHeap(Movable):
    """TopNHeap: a max-heap of the best `heap_size` key records seen (the
    worst on top). The record layout is a parameter of each method."""

    var heap_size: Int
    var records: _RecordHeaps
    var size: Int
    var arena: _KeyArena
    var view: _BoundaryView
    # Entries added since the keys were last compacted (heap_data's size).
    var added: Int

    def __init__(
        out self, heap_size: Int, record_words: Int, *, variable: Bool = False
    ):
        self.heap_size = heap_size
        if variable:
            self.records = _RecordHeaps(
                BinaryHeap[_HeapRecord[1, True]](capacity=heap_size)
            )
        elif record_words == 2:
            self.records = _RecordHeaps(
                BinaryHeap[_HeapRecord[1, False]](capacity=heap_size)
            )
        elif record_words == 3:
            self.records = _RecordHeaps(
                BinaryHeap[_HeapRecord[2, False]](capacity=heap_size)
            )
        else:
            self.records = _RecordHeaps(
                BinaryHeap[_HeapRecord[3, False]](capacity=heap_size)
            )
        self.size = 0
        self.arena = _KeyArena()
        self.view = _BoundaryView()
        self.added = 0

    @always_inline
    def keys[W: Int, VAR: Bool](self) -> SortKeys[W, VAR]:
        # The standard heap owns these records. This single storage adapter
        # exposes them for payload renumbering, arena compaction, and final
        # sorting; none of those operations changes key order before finalize.
        comptime Entry = _HeapRecord[1 if VAR else W, VAR]
        comptime assert size_of[Entry]() == SortKeys[W, VAR].R * 8
        ref heap = self.records[BinaryHeap[Entry]]
        return SortKeys[W, VAR](Int(heap._data.unsafe_ptr()))

    def reduce_threshold(self) -> Int:
        """ReduceThreshold."""
        return max(5 * 2048, 2 * self.heap_size)

    def full(self) -> Bool:
        return self.heap_size > 0 and self.size >= self.heap_size

    @always_inline
    def entry_should_be_added[W: Int, VAR: Bool](self, rec: Rec) -> Bool:
        """EntryShouldBeAdded: room in the heap, or better than its worst."""
        comptime K = SortKeys[W, VAR]
        if self.heap_size == 0:
            return False
        if self.size < self.heap_size:
            return True
        return K.less(rec, self.keys[W, VAR]().get(0))

    @always_inline
    def should_add[W: Int, VAR: Bool](self, rec: Rec) -> Bool:
        """A sinking row: CheckBoundaryValues, then EntryShouldBeAdded."""
        comptime K = SortKeys[W, VAR]
        if self.view.version != 0 and not K.less(rec, self.view.rec):
            return False
        return self.entry_should_be_added[W, VAR](rec)

    def add[W: Int, VAR: Bool](mut self, var rec: Rec):
        """Keep a candidate using the standard heap's push/pop operations."""
        if self.heap_size == 0:
            return
        comptime if VAR:
            rec[2] = UInt64(self.arena.add(Int(rec[2]), Int(rec[1])))
        self.added += 1
        comptime Entry = _HeapRecord[1 if VAR else W, VAR]
        ref heap = self.records[BinaryHeap[Entry]]
        if len(heap) >= self.heap_size:
            _ = heap.pop()
        heap.push(Entry(rec))
        self.size = len(heap)

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
        entry can no longer enter. Only EntryShouldBeAdded decides: the
        boundary is a full heap's worst key, and that key's own rows are
        in the result."""
        var k = other.keys[W, VAR]()
        for i in range(other.size):
            var rec = k.get(i)
            if not self.entry_should_be_added[W, VAR](rec):
                break
            self.add[W, VAR](rec)
        self.compact_keys[W, VAR]()

    def publish[W: Int, VAR: Bool](mut self, shared: TopNBoundary) -> Bool:
        """A full heap's worst key to the shared boundary, then the
        boundary back (it may be another thread's, tighter); whether this
        heap's key became the boundary."""
        var taken = False
        if self.full():
            taken = shared.publish[W, VAR](self.keys[W, VAR]().get(0))
        self.view.refresh[W, VAR](shared)
        return taken

    def top_payload[W: Int, VAR: Bool](self) -> Int:
        """The worst entry's payload."""
        return self.keys[W, VAR]().payload(0)

    def bound_word[W: Int, VAR: Bool](self) -> Tuple[Bool, UInt64]:
        """The first key word no entering row can exceed: the full heap's
        worst key's, or the shared boundary's when tighter."""
        var has = False
        var word = UInt64.MAX
        if self.full():
            has = True
            word = self.keys[W, VAR]().part0(0)
        if self.view.version != 0:
            has = True
            word = min(word, self.view.rec[0])
        return (has, word)

    def entering[
        W: Int, VAR: Bool
    ](self, prefixes: List[UInt64], count: Int) -> Optional[List[Int]]:
        """CheckBoundaryValues on a vector's leading sort columns: the rows
        whose first key word can be at or below the bound (`prefixes`
        bound each row's from below); None when there is no bound yet."""
        var bound = self.bound_word[W, VAR]()
        if not bound[0]:
            return None
        var rows = List[Int]()
        for i in range(count):
            if prefixes[i] <= bound[1]:
                rows.append(i)
        return rows^

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
