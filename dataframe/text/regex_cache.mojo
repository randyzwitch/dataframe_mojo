"""Bounded reuse of immutable compiled patterns across expression bindings."""
from std.ffi import _get_global
from std.utils.lock import BlockingSpinLock
from std.memory import Allocation, Layout, Pointer, alloc, dealloc

from dataframe.regex import Regex

comptime _CAPACITY = 32


struct _PatternCache(Deinitable, Movable):
    var entries: List[Regex]
    var lock: BlockingSpinLock

    def __init__(out self):
        self.entries = List[Regex](capacity=_CAPACITY)
        self.lock = BlockingSpinLock()

    @staticmethod
    def _lookup(mut entries: List[Regex], pattern: String) -> Optional[Regex]:
        # Called under the lock. Most-recent entries are at the end.
        var at = len(entries) - 1
        while at >= 0:
            if entries[at]._compiled[].pattern == pattern:
                var result = entries[at].copy()
                if at + 1 != len(entries):
                    var entry = entries.pop(at)
                    entries.append(entry^)
                return result^
            at -= 1
        return None

    def compile(mut self, pattern: String) raises -> Regex:
        var cached: Optional[Regex] = None
        # Lookup/insertion cannot raise. Keep explicit acquisition pairs:
        # BlockingScopedLock adds measurable overhead to contended hot hits.
        var owner = Int(Pointer(to=cached))
        self.lock.lock(owner)
        cached = Self._lookup(self.entries, pattern)
        _ = self.lock.unlock(owner)
        if cached:
            return cached.take()
        # Parsing may raise and stays outside the lock.
        var compiled = Regex(pattern)
        var evicted: Optional[Regex] = None
        # Another caller may have cached this pattern while parsing.
        self.lock.lock(owner)
        cached = Self._lookup(self.entries, pattern)
        if not cached:
            cached = compiled.copy()
            if len(self.entries) == _CAPACITY:
                evicted = self.entries.pop(0)
            self.entries.append(compiled^)
        _ = self.lock.unlock(owner)
        # Destruction of evicted programs stays outside the lock.
        _ = evicted^
        return cached.take()


def _init_cache() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    var pointer = alloc(Layout[_PatternCache](length=1)).unsafe_leak()
    pointer.unsafe_write(_PatternCache())
    return Pointer[NoneType, MutUntrackedOrigin](
        unsafe_from_address=Int(pointer)
    )


def _destroy_cache(address: Optional[Pointer[NoneType, MutUntrackedOrigin]]):
    if not address:
        return
    var pointer = Pointer[_PatternCache, MutUntrackedOrigin](
        unsafe_from_address=Int(address.value())
    )
    var cache = pointer.unsafe_take_pointee()
    dealloc(
        Allocation[_PatternCache](
            unsafe_owned_ptr=pointer, layout=Layout[_PatternCache](length=1)
        )
    )
    _ = cache^


def cached_pattern(pattern: String) raises -> Regex:
    var address = _get_global[
        "dataframe.regex.pattern_cache.v1", _init_cache, _destroy_cache
    ]()
    ref cache = Pointer[_PatternCache, MutAnyOrigin](
        unsafe_from_address=Int(address.value())
    )[]
    return cache.compile(pattern)
