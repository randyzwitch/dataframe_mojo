"""Bounded reuse of immutable compiled patterns across expression bindings."""
from std.ffi import _get_global, external_call
from std.memory import Allocation, Layout, Pointer, alloc, dealloc

from .regex import Regex

comptime _CAPACITY = 32


struct _PatternCache(Deinitable, Movable):
    var entries: List[Regex]
    var mutex: List[UInt64]
    var ready: Bool

    def __init__(out self):
        self.entries = List[Regex](capacity=_CAPACITY)
        # Aligned storage for pthread_mutex_t on the supported hosts.
        self.mutex = List[UInt64](length=16, fill=0)
        self.ready = False
        self.ready = (
            external_call["pthread_mutex_init", Int32](self._mutex(), 0) == 0
        )

    def __deinit__(deinit self):
        if self.ready:
            _ = external_call["pthread_mutex_destroy", Int32](self._mutex())

    def _mutex(self) -> Int:
        return Int(self.mutex.unsafe_ptr())

    def _lookup(mut self, pattern: String) -> Optional[Regex]:
        # Called under the mutex. Most-recent entries are at the end.
        var at = len(self.entries) - 1
        while at >= 0:
            if self.entries[at]._compiled[].pattern == pattern:
                var result = self.entries[at].copy()
                if at + 1 != len(self.entries):
                    var entry = self.entries.pop(at)
                    self.entries.append(entry^)
                return result^
            at -= 1
        return None

    def compile(mut self, pattern: String) raises -> Regex:
        if not self.ready:
            return Regex(pattern)
        _ = external_call["pthread_mutex_lock", Int32](self._mutex())
        var cached = self._lookup(pattern)
        _ = external_call["pthread_mutex_unlock", Int32](self._mutex())
        if cached:
            return cached.take()
        # Construction can raise and takes much longer than a lookup. Keep
        # it outside the lock, including on invalid-pattern paths.
        var compiled = Regex(pattern)
        _ = external_call["pthread_mutex_lock", Int32](self._mutex())
        # Another caller may have finished constructing the same pattern.
        cached = self._lookup(pattern)
        if cached:
            _ = external_call["pthread_mutex_unlock", Int32](self._mutex())
            return cached.take()
        var result = compiled.copy()
        var evicted: Optional[Regex] = None
        if len(self.entries) == _CAPACITY:
            evicted = self.entries.pop(0)
        self.entries.append(compiled^)
        _ = external_call["pthread_mutex_unlock", Int32](self._mutex())
        # Drop potentially large evicted programs after unlocking.
        _ = evicted^
        return result^


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
