"""Synchronized facts about an immutable column buffer's declared window.

Order/progression are independent atomic, best-effort caches: concurrent callers
may compute the same fact, then publish the same answer. They never hold a lock
while running worker jobs. Zone maps publish one immutable list; a reader first
acquires its ready state and never touches a list being initialized. Cache cells
are shared by column windows and replaced before owned buffer mutation.
"""
from std.atomic import Atomic
from std.memory import Pointer


struct _BufferFacts(Movable):
    var _zones: List[Int64]
    var _zone_state: Atomic[Int64]
    var _progression: Atomic[Int64]
    var _first: Int
    var _last: Int
    # Deliberately not the first field: clients never depend on field offsets.
    var _order: Atomic[Int64]

    def __init__(out self, first: Int = 0, last: Int = -1):
        self._zones = List[Int64]()
        self._zone_state = Atomic[Int64](0)
        self._progression = Atomic[Int64](0)
        self._first = first
        self._last = last
        self._order = Atomic[Int64](0)

    def initialize_window(mut self, first: Int, last: Int):
        """Only before the freshly allocated cell is shared (Column factory).

        This keeps construction at one facts allocation without mutating bounds
        after publication. Column checks sole ownership before calling this.
        """
        debug_assert(self.order_state() == 0 and self.progression_state() == 0)
        debug_assert(self._zone_state.load() == 0)
        self._first = first
        self._last = last

    def first(self) -> Int:
        return self._first

    def last(self, buffer_length: Int) -> Int:
        return buffer_length if self._last < 0 else self._last

    def order_state(self) -> Int:
        return Int(self._order.load())

    def publish_order(self, ascending: Bool):
        # A typed field pointer provides interior mutability without assuming
        # the layout of this struct. It is used only within the live cell.
        var cell = Pointer(to=self._order).unsafe_mut_cast[True]()
        cell[].store(Int64(1 if ascending else 2))

    def progression_state(self) -> Int:
        return Int(self._progression.load())

    def publish_progression(self, progression: Bool):
        var cell = Pointer(to=self._progression).unsafe_mut_cast[True]()
        cell[].store(Int64(1 if progression else 2))

    def zones_ready(self) -> Bool:
        # Default sequentially consistent atomics include acquire/release.
        return self._zone_state.load() == 2

    def publish_zones(self, var values: List[Int64]):
        """Publish at most once; a losing initializer drops its private list.

        0 = empty, 1 = publishing, 2 = ready. Readers seeing 0 or 1 use their
        ordinary bounds scan; they never borrow the unpublished list. The
        existing pipeline initialization lock still avoids duplicate scans.
        """
        var state = Pointer(to=self._zone_state).unsafe_mut_cast[True]()
        while True:
            var expected = Int64(0)
            if state[].compare_exchange(expected, Int64(1)):
                break
            if expected != 0:
                return
        var target = Pointer(to=self._zones).unsafe_mut_cast[True]()
        target[] = values^
        state[].store(2)

    def zones(self) -> ref[self._zones] List[Int64]:
        """Borrow the immutable published map, after zones_ready is true."""
        debug_assert(self.zones_ready(), "zone map is not published")
        return self._zones
