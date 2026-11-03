"""Synchronized facts about an immutable column buffer's declared window.

Progression is an atomic, best-effort cache: concurrent callers may compute
the same fact, then publish the same answer. They never hold a lock while
running worker jobs. Cache cells are shared by column windows and replaced
before owned buffer mutation. A buffer's segment statistics live with the
buffer itself (`_BufferData.publish_statistics`).
"""
from std.atomic import Atomic
from std.memory import Pointer


struct _BufferFacts(Movable):
    var _progression: Atomic[Int64]
    var _first: Int
    var _last: Int

    def __init__(out self, first: Int = 0, last: Int = -1):
        self._progression = Atomic[Int64](0)
        self._first = first
        self._last = last

    def initialize_window(mut self, first: Int, last: Int):
        """Only before the freshly allocated cell is shared (Column factory).

        This keeps construction at one facts allocation without mutating bounds
        after publication. Column checks sole ownership before calling this.
        """
        debug_assert(self.progression_state() == 0)
        self._first = first
        self._last = last

    def first(self) -> Int:
        return self._first

    def last(self, buffer_length: Int) -> Int:
        return buffer_length if self._last < 0 else self._last

    def progression_state(self) -> Int:
        return Int(self._progression.load())

    def publish_progression(self, progression: Bool):
        # A typed field pointer provides interior mutability without assuming
        # the layout of this struct. It is used only within the live cell.
        var cell = Pointer(to=self._progression).unsafe_mut_cast[True]()
        cell[].store(Int64(1 if progression else 2))
