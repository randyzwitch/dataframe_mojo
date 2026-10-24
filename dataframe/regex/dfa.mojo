"""A lazily built DFA, ported from RE2's `DFA` (dfa.cc); regex-automata's
`hybrid` DFA is the same design.

A DFA state is the ordered list of NFA instructions the Pike VM would hold
at that point, plus a flag word: the zero-width assertions known to hold
before the next code point, whether the previous code point was a word
character, and whether the state matches. States are built only when the
search first reaches them and their transitions are cached per alphabet
class, so a search costs one table load per code point once warm.

As in RE2, matches are reported one code point late: a state is "matching"
when a match ended just before the code point that led into it. That
delay is what lets `$`, `\\b` and friends look at the next code point. A
final transition on the end-of-text class (or the code point just past the
searched range) reveals a match ending at the end.

Leftmost-first semantics come from the instruction order: once a matching
instruction is seen, the lower priority ones are dropped, which also
removes the unanchored `.*?` loop, so the search stops when the match can
no longer grow. A reverse DFA run backwards from a match's end with all
instructions kept (longest match) finds where the match starts.

Each worker owns its DFA (the cache is mutable); the program is shared.
When the cache grows past its budget it is cleared and the search goes on
from the current state, as RE2 does; if that happens too often, the search
gives up and the caller falls back to the Pike VM.
"""
from std.memory import Pointer, pack_bits
from std.bit import count_trailing_zeros
from .pikevm import decode, previous_start, _ascii_word
from .program import (
    OP_FAIL,
    OP_LOOK,
    OP_MATCH,
    OP_NOP,
    OP_SAVE,
    OP_SPLIT,
    Program,
    SUFFIX_LIMIT,
    consumes,
)
from .suffix import suffix_end
from .unicode_dfa_run import scan_unicode_self_loop
from .syntax import (
    LOOK_END_LINE,
    LOOK_END_TEXT,
    LOOK_MASK,
    LOOK_NOT_WORD,
    LOOK_START_LINE,
    LOOK_START_TEXT,
    LOOK_WORD,
    LOOK_WORD_START,
    LOOK_WORD_END,
    LOOK_ASCII_WORD,
    LOOK_ASCII_NOT_WORD,
    LOOK_ASCII_WORD_START,
    LOOK_ASCII_WORD_END,
    LOOK_UNICODE_WORD,
    LOOK_ASCII,
)

comptime FLAG_MATCH: UInt32 = 1 << 8
comptime FLAG_LAST_WORD: UInt32 = 1 << 9
comptime FLAG_LAST_ASCII_WORD: UInt32 = 1 << 14
comptime NEED_SHIFT: UInt32 = 16

comptime UNKNOWN: Int32 = -1
comptime DEAD: Int32 = -2

comptime GAVE_UP = -2

# Cache budget, in transition table entries (4 MB), and the progress a
# reset must buy: RE2 gives up when fewer than ten bytes were searched per
# state built since the last reset.
comptime _MAX_TRANSITIONS = 1 << 20
comptime _MIN_BYTES_PER_STATE = 10


struct _Workq(Movable):
    """An insertion-ordered sparse set of instruction ids."""

    var dense: List[Int32]
    var sparse: List[Int32]
    var size: Int

    def __init__(out self, length: Int):
        self.dense = List[Int32](length=length, fill=0)
        self.sparse = List[Int32](length=length, fill=0)
        self.size = 0

    @always_inline
    def contains(self, pc: Int) -> Bool:
        var i = Int(self.sparse[pc])
        return i < self.size and Int(self.dense[i]) == pc

    @always_inline
    def insert(mut self, pc: Int):
        self.sparse[pc] = Int32(self.size)
        self.dense[self.size] = Int32(pc)
        self.size += 1


@always_inline
def _mix(h: UInt64, v: UInt64) -> UInt64:
    return (h ^ v) * 0x100000001B3


struct LazyDFA(Movable):
    var reverse: Bool
    var longest: Bool
    var stride: Int
    var eot: Int
    # State s has instructions pool[starts[s] : starts[s + 1]] and flags.
    var pool: List[Int32]
    var starts: List[Int32]
    var flags: List[UInt32]
    var hashes: List[UInt64]
    var transitions: List[Int32]
    var table: List[Int32]
    var q0: _Workq
    var q1: _Workq
    var stack: List[Int32]
    var start_states: List[Int32]
    var accel: List[Bool]
    var accel_enabled: Bool
    var resets: Int

    def __init__(out self, program: Program, reverse: Bool):
        """A DFA over `program`: forward leftmost-first, or (reverse) the
        longest match backwards."""
        self.reverse = reverse
        self.longest = reverse
        self.stride = program.alphabet + 1
        self.eot = program.alphabet
        self.pool = List[Int32]()
        self.starts = [0]
        self.flags = List[UInt32]()
        self.hashes = List[UInt64]()
        self.transitions = List[Int32]()
        self.table = List[Int32]()
        self.q0 = _Workq(0)
        self.q1 = _Workq(0)
        self.stack = List[Int32]()
        self.start_states = List[Int32](length=10, fill=UNKNOWN)
        self.accel = List[Bool]()
        self.accel_enabled = False
        self.resets = 0

    def state_count(self) -> Int:
        return len(self.flags)

    def clear(mut self):
        self.pool.clear()
        self.starts = [0]
        self.flags.clear()
        self.hashes.clear()
        self.accel.clear()
        self.transitions.clear()
        for i in range(len(self.table)):
            self.table[i] = -1
        for i in range(len(self.start_states)):
            self.start_states[i] = UNKNOWN

    def _add_to_queue(
        mut self, program: Program, into_q1: Bool, start_pc: Int, flag: UInt32
    ):
        """Add `start_pc` and everything reachable from it without
        consuming input, following assertions that `flag` satisfies
        (RE2's `AddToQueue`)."""
        self.stack.append(Int32(start_pc))
        while len(self.stack) > 0:
            var pc = Int(self.stack.pop())
            while True:
                if into_q1:
                    if self.q1.contains(pc):
                        break
                    self.q1.insert(pc)
                else:
                    if self.q0.contains(pc):
                        break
                    self.q0.insert(pc)
                var op = program.op[pc]
                if op == OP_SPLIT:
                    self.stack.append(program.arg[pc])
                    pc = Int(program.out[pc])
                    continue
                if op == OP_NOP or op == OP_SAVE:
                    pc = Int(program.out[pc])
                    continue
                if op == OP_LOOK:
                    if UInt32(program.arg[pc]) & ~flag:
                        break
                    pc = Int(program.out[pc])
                    continue
                break

    def _cached_state(
        mut self, program: Program, from_q1: Bool, var flag: UInt32
    ) -> Int32:
        """The state for a work queue (RE2's `WorkqToCachedState`): keep the
        instructions that act on input or assertions, cut everything after
        a match in leftmost-first mode, and look the state up."""
        var mark = len(self.pool)
        var need: UInt32 = 0
        var size = self.q1.size if from_q1 else self.q0.size
        for i in range(size):
            var pc = Int(self.q1.dense[i]) if from_q1 else Int(self.q0.dense[i])
            var op = program.op[pc]
            if consumes(op) or op == OP_MATCH or op == OP_LOOK:
                self.pool.append(Int32(pc))
                if op == OP_LOOK:
                    need |= UInt32(program.arg[pc])
                if op == OP_MATCH and not self.longest:
                    break
        var count = len(self.pool) - mark
        if need == 0:
            flag &= FLAG_MATCH
        if count == 0 and flag == 0:
            self._truncate(mark)
            return DEAD
        if program.has_suffix and flag & FLAG_MATCH and count == 2:
            if program.op[Int(self.pool[mark + 1])] == OP_MATCH:
                var code = program._greedy_suffix_code(Int(self.pool[mark]))
                if code != 0:
                    self._truncate(mark)
                    return code
        flag |= need << NEED_SHIFT
        var h: UInt64 = 0xCBF29CE484222325
        h = _mix(h, UInt64(flag))
        for i in range(mark, len(self.pool)):
            h = _mix(h, UInt64(self.pool[i]))
        var mask = len(self.table) - 1
        var slot = Int(h ^ (h >> 29)) & mask
        while True:
            var s = Int(self.table[slot])
            if s < 0:
                break
            if self.hashes[s] == h and self.flags[s] == flag:
                var a = Int(self.starts[s])
                if Int(self.starts[s + 1]) - a == count:
                    var same = True
                    for k in range(count):
                        if self.pool[a + k] != self.pool[mark + k]:
                            same = False
                            break
                    if same:
                        self._truncate(mark)
                        return Int32(s)
            slot = (slot + 1) & mask
        var id = len(self.flags)
        self.starts.append(Int32(len(self.pool)))
        self.flags.append(flag)
        self.hashes.append(h)
        self.accel.append(False)
        for _ in range(self.stride):
            self.transitions.append(UNKNOWN)
        self.table[slot] = Int32(id)
        if 2 * len(self.flags) > len(self.table):
            self._grow_table()
        return Int32(id)

    def _truncate(mut self, size: Int):
        while len(self.pool) > size:
            _ = self.pool.pop()

    def _full(self) -> Bool:
        return len(self.transitions) + self.stride > _MAX_TRANSITIONS

    def _grow_table(mut self):
        var size = 2 * len(self.table)
        self.table = List[Int32](length=size, fill=-1)
        var mask = size - 1
        for s in range(len(self.flags)):
            var h = self.hashes[s]
            var slot = Int(h ^ (h >> 29)) & mask
            while self.table[slot] >= 0:
                slot = (slot + 1) & mask
            self.table[slot] = Int32(s)

    def _state_to_q0(mut self, program: Program, s: Int):
        self.q0.size = 0
        var flag = self.flags[s] & LOOK_MASK
        for i in range(Int(self.starts[s]), Int(self.starts[s + 1])):
            self._add_to_queue(program, False, Int(self.pool[i]), flag)

    def _run_on_empty(mut self, program: Program, flag: UInt32):
        """Re-follow q0 with more assertions holding, into q0 (RE2's
        `RunWorkqOnEmptyString`)."""
        self.q1.size = 0
        for i in range(self.q0.size):
            self._add_to_queue(program, True, Int(self.q0.dense[i]), flag)
        swap(self.q0, self.q1)

    def _run_on_class(
        mut self, program: Program, cls: Int, after: UInt32
    ) -> Bool:
        """Step q0 over one code point of class `cls` into q0 (RE2's
        `RunWorkqOnByte`); returns whether a match instruction was
        reached first."""
        self.q1.size = 0
        var matched = False
        var cp = program.class_rep[cls] if cls < self.eot else UInt32(0)
        for i in range(self.q0.size):
            var pc = Int(self.q0.dense[i])
            var op = program.op[pc]
            if op == OP_MATCH:
                matched = True
                if not self.longest:
                    break
            elif consumes(op) and cls < self.eot and program.step(pc, cp):
                self._add_to_queue(program, True, Int(program.out[pc]), after)
        swap(self.q0, self.q1)
        return matched

    def _compute(mut self, program: Program, s: Int, cls: Int) -> Int32:
        """The transition from state `s` on class `cls` (RE2's
        `RunStateOnByte`)."""
        self._state_to_q0(program, s)
        var state_flag = self.flags[s]
        var need = state_flag >> NEED_SHIFT
        var before = state_flag & LOOK_MASK
        var old_before = before
        var after: UInt32 = 0
        var newline = cls < self.eot and program.class_rep[cls] == 10
        if not self.reverse:
            if newline:
                before |= LOOK_END_LINE
                after |= LOOK_START_LINE
            if cls == self.eot:
                before |= LOOK_END_LINE | LOOK_END_TEXT
        else:
            if newline:
                before |= LOOK_START_LINE
                after |= LOOK_END_LINE
            if cls == self.eot:
                before |= LOOK_START_LINE | LOOK_START_TEXT
        var last_word = (state_flag & FLAG_LAST_WORD) != 0
        var word = cls < self.eot and program.class_word[cls]
        if word == last_word:
            before |= LOOK_NOT_WORD
        else:
            before |= LOOK_WORD
        var left = word if self.reverse else last_word
        var right = last_word if self.reverse else word
        if not left and right:
            before |= LOOK_WORD_START
        if left and not right:
            before |= LOOK_WORD_END
        var ascii_word = cls < self.eot and _ascii_word(program.class_rep[cls])
        var last_ascii = (state_flag & FLAG_LAST_ASCII_WORD) != 0
        before |= (
            LOOK_ASCII_NOT_WORD if ascii_word == last_ascii else LOOK_ASCII_WORD
        )
        left = ascii_word if self.reverse else last_ascii
        right = last_ascii if self.reverse else ascii_word
        if not left and right:
            before |= LOOK_ASCII_WORD_START
        if left and not right:
            before |= LOOK_ASCII_WORD_END
        if before & ~old_before & need:
            self._run_on_empty(program, before)
        var matched = self._run_on_class(program, cls, after)
        var flag = after
        if matched:
            flag |= FLAG_MATCH
        if word:
            flag |= FLAG_LAST_WORD
        if ascii_word:
            flag |= FLAG_LAST_ASCII_WORD
        var next = self._cached_state(program, False, flag)
        self.transitions[s * self.stride + cls] = self._entry(next)
        return next

    @always_inline
    def _entry(self, s: Int32) -> Int32:
        """A transition table entry: the target's row offset, or for a
        matching state or a start state the prefilter may skip from, the
        offset tagged negative (`-(offset << 2 | tags) - 3`), the way
        regex-automata tags lazy DFA state ids so the search loop tests one
        sign per code point."""
        if s < 0:
            return s
        var tags = 0
        if self.flags[Int(s)] & FLAG_MATCH:
            tags |= 1
        if self.accel[Int(s)]:
            tags |= 2
        var offset = Int(s) * self.stride
        if tags == 0:
            return Int32(offset)
        return Int32(-((offset << 2) | tags) - 3)

    def _context(
        self, program: Program, text: Span[UInt8, ImmutAnyOrigin], at: Int
    ) -> Int:
        """The search context at `at`: 0 at the text's edge, 1 next to a
        newline, 2 next to an ASCII word character, 3 next to a Unicode-only
        word character, 4 otherwise. Forward looks at
        the code point before `at`; reverse at the one after."""
        if not self.reverse:
            if at == 0:
                return 0
            var cp = UInt32(decode(text, previous_start(text, at)) >> 3)
            if cp == 10:
                return 1
            return 2 if _ascii_word(cp) else 3 if program.is_word(cp) else 4
        if at == len(text):
            return 0
        var cp = UInt32(decode(text, at) >> 3)
        if cp == 10:
            return 1
        return 2 if _ascii_word(cp) else 3 if program.is_word(cp) else 4

    @always_inline
    def start_state(
        mut self,
        program: Program,
        text: Span[UInt8, ImmutAnyOrigin],
        at: Int,
        anchored: Bool,
    ) -> Int32:
        # Without assertions, surrounding characters cannot affect the
        # epsilon closure. Share one context instead of decoding a neighbor
        # at every short match. Keep cache misses out of this hot lookup.
        var context = 0
        if program.looks != 0:
            context = self._context(program, text, at)
        var key = context + (5 if anchored else 0)
        var cached = self.start_states[key]
        if cached != UNKNOWN:
            return cached
        return self._new_start_state(program, context, key, anchored)

    def _new_start_state(
        mut self, program: Program, context: Int, key: Int, anchored: Bool
    ) -> Int32:
        # Literal, single-class and anchored one-pass searches need no DFA
        # buffers. Allocate them on the first cold start-state lookup.
        if len(self.table) == 0:
            self.table = List[Int32](length=1024, fill=-1)
            self.q0 = _Workq(len(program))
            self.q1 = _Workq(len(program))
        var flag: UInt32 = 0
        if context == 0:
            if self.reverse:
                flag = LOOK_END_TEXT | LOOK_END_LINE
            else:
                flag = LOOK_START_TEXT | LOOK_START_LINE
        elif context == 1:
            flag = LOOK_END_LINE if self.reverse else LOOK_START_LINE
        elif context == 2:
            flag = FLAG_LAST_WORD | FLAG_LAST_ASCII_WORD
        elif context == 3:
            flag = FLAG_LAST_WORD
        self.q0.size = 0
        var pc = program.start if anchored else program.start_unanchored
        self._add_to_queue(program, False, pc, flag & LOOK_MASK)
        var s = self._cached_state(program, False, flag)
        if s >= 0 and not anchored and self.accel_enabled:
            # A start state of an unanchored search: from here the
            # prefilter may skip to the next place a match can begin.
            self.accel[Int(s)] = True
        self.start_states[key] = s
        return s

    def _reset(mut self, program: Program, s: Int) -> Int32:
        """Clear the cache, keeping state `s` (renumbered)."""
        var insts = List[Int32]()
        for i in range(Int(self.starts[s]), Int(self.starts[s + 1])):
            insts.append(self.pool[i])
        var flag = self.flags[s]
        self.clear()
        self.resets += 1
        for pc in insts:
            self.pool.append(pc)
        self.starts.append(Int32(len(self.pool)))
        self.flags.append(flag)
        self.accel.append(False)
        var h: UInt64 = 0xCBF29CE484222325
        h = _mix(h, UInt64(flag))
        for pc in insts:
            h = _mix(h, UInt64(pc))
        self.hashes.append(h)
        for _ in range(self.stride):
            self.transitions.append(UNKNOWN)
        var slot = Int(h ^ (h >> 29)) & (len(self.table) - 1)
        self.table[slot] = 0
        return 0

    @always_inline
    def _class_at(
        self, program: Program, text: Span[UInt8, ImmutAnyOrigin], at: Int
    ) -> Tuple[Int, Int]:
        """The class of the code point at `at` and its byte width."""
        var b = text.unsafe_ptr()[unsafe_offset=at]
        if b < 0x80:
            return Int(program.ascii_class[Int(b)]), 1
        var packed = decode(text, at)
        return program.class_of(UInt32(packed >> 3)), packed & 7

    @no_inline
    def _advance_cached(
        self,
        program: Program,
        text: Span[UInt8, ImmutAnyOrigin],
        var off: Int,
        var at: Int,
        end: Int,
    ) -> Tuple[Int, Int]:
        """Group four untagged transitions without enlarging tag dispatch.
        Sixteen remaining bytes bound four UTF8 decodes. The first tagged
        edge leaves its code point untouched for ordinary dispatch. Only
        the state and cursor cross the call boundary."""
        var table = self.transitions.unsafe_ptr()
        var cls = 0
        var width = 0
        var e = 0
        while end - at >= 16:
            if text.unsafe_ptr()[unsafe_offset=at] >= 128:
                return self._advance_unicode_cached(program, text, off, at, end)
            cls, width = self._class_at(program, text, at)
            e = Int(table[unsafe_offset=off + cls])
            if e < 0:
                return off, at
            off = e
            at += width
            cls, width = self._class_at(program, text, at)
            e = Int(table[unsafe_offset=off + cls])
            if e < 0:
                return off, at
            off = e
            at += width
            cls, width = self._class_at(program, text, at)
            e = Int(table[unsafe_offset=off + cls])
            if e < 0:
                return off, at
            off = e
            at += width
            cls, width = self._class_at(program, text, at)
            e = Int(table[unsafe_offset=off + cls])
            if e < 0:
                return off, at
            off = e
            at += width
        return off, at

    @no_inline
    def _advance_unicode_cached(
        self,
        program: Program,
        text: Span[UInt8, ImmutAnyOrigin],
        var off: Int,
        var at: Int,
        end: Int,
    ) -> Tuple[Int, Int]:
        """Consume cached Unicode packets and scalar groups until a tag.
        An ASCII group returns to the caller's ordinary cached loop."""
        var table = self.transitions.unsafe_ptr()
        var cls = 0
        var width = 0
        var e = 0
        while end - at >= 16:
            if text.unsafe_ptr()[unsafe_offset=at] < 128:
                return off, at
            var stop = scan_unicode_self_loop(
                program, self.transitions, off, text, at, end, Int32(off)
            )
            if stop > at:
                at = stop
                continue
            cls, width = self._class_at(program, text, at)
            e = Int(table[unsafe_offset=off + cls])
            if e < 0:
                return off, at
            off = e
            at += width
            cls, width = self._class_at(program, text, at)
            e = Int(table[unsafe_offset=off + cls])
            if e < 0:
                return off, at
            off = e
            at += width
            cls, width = self._class_at(program, text, at)
            e = Int(table[unsafe_offset=off + cls])
            if e < 0:
                return off, at
            off = e
            at += width
            cls, width = self._class_at(program, text, at)
            e = Int(table[unsafe_offset=off + cls])
            if e < 0:
                return off, at
            off = e
            at += width
        return off, at

    @always_inline
    def find_forward(
        mut self,
        program: Program,
        prefilter: Prefilter,
        text: Span[UInt8, ImmutAnyOrigin],
        start: Int,
        anchored: Bool,
        earliest: Bool,
    ) -> Int:
        if program.has_suffix:
            return self._find_forward_suffix(
                program, prefilter, text, start, anchored, earliest
            )
        return self._find_forward[False](
            program, prefilter, text, start, anchored, earliest
        )

    @no_inline
    def _find_forward_suffix(
        mut self,
        program: Program,
        prefilter: Prefilter,
        text: Span[UInt8, ImmutAnyOrigin],
        start: Int,
        anchored: Bool,
        earliest: Bool,
    ) -> Int:
        return self._find_forward[True](
            program, prefilter, text, start, anchored, earliest
        )

    @always_inline
    def _find_forward[
        use_suffix: Bool
    ](
        mut self,
        program: Program,
        prefilter: Prefilter,
        text: Span[UInt8, ImmutAnyOrigin],
        start: Int,
        anchored: Bool,
        earliest: Bool,
    ) -> Int:
        """Where the leftmost-first match starting at or after `start`
        ends (with `earliest`, the end of the first match seen instead),
        -1 for no match, or GAVE_UP."""
        var n = len(text)
        var p = text.unsafe_ptr()
        var ascii = program.ascii_class.unsafe_ptr()
        var stride = self.stride
        var at = start
        # Literal sets extend coverage when no cheap start-state scanner is
        # available. Skip once before entering the DFA, leaving its cached
        # transitions unchanged. Small inputs use the ordinary DFA directly.
        if not anchored and len(prefilter.literals) > 0 and n - at >= 64:
            at = prefilter._find_literals(text, at)
            if at < 0:
                return -1
        var s0 = Int(self.start_state(program, text, at, anchored))
        if s0 == Int(DEAD):
            return -1
        var accel = not anchored and prefilter.active()
        var off = s0 * stride
        if accel and self.accel[s0] and at < n:
            var candidate = prefilter.find(text, at)
            if candidate < 0:
                return -1
            if candidate != at:
                at = candidate
                off = Int(self.start_state(program, text, at, False)) * stride
        var last = -1
        var budget_mark = start
        var resets = self.resets
        var table = self.transitions.unsafe_ptr()
        while at < n:
            var b = p[unsafe_offset=at]
            var cls: Int
            var width = 1
            if b < 0x80:
                cls = Int(ascii[unsafe_offset=Int(b)])
            else:
                var packed = decode(text, at)
                cls = program.class_of(UInt32(packed >> 3))
                width = packed & 7
            var e = Int(table[unsafe_offset=off + cls])
            if e >= 0:
                off = e
                at += width
                if n - at >= 16:
                    off, at = self._advance_cached(program, text, off, at, n)
                continue
            if e == Int(UNKNOWN):
                var s = off // stride
                if self._full():
                    if (
                        self.resets - resets >= 1
                        and at - budget_mark
                        < _MIN_BYTES_PER_STATE * self.state_count()
                    ):
                        return GAVE_UP
                    budget_mark = at
                    s = Int(self._reset(program, s))
                var next = self._compute(program, s, cls)
                table = self.transitions.unsafe_ptr()
                if next == DEAD:
                    return last
                e = Int(self._entry(next))
                if e >= 0:
                    off = e
                    at += width
                    continue
            elif e == Int(DEAD):
                return last
            comptime if use_suffix:
                if e < SUFFIX_LIMIT:
                    if earliest:
                        return at
                    # The transition already consumed this code point.
                    return suffix_end[False](text, at + width, n, Int32(e))
            var v = -e - 3
            var next_offset = v >> 2
            if v & 1:
                last = at
                if earliest:
                    return last
                if (
                    width > 1
                    and next_offset == off
                    and not (v & 2)
                    and n - at >= 16
                ):
                    var stop = scan_unicode_self_loop(
                        program, self.transitions, off, text, at, n, Int32(e)
                    )
                    if stop > at:
                        last = stop - width
                        at = stop
                        continue
            off = next_offset
            at += width
            if (v & 2) and accel and at < n:
                var candidate = prefilter.find(text, at)
                if candidate < 0:
                    return last
                if candidate != at:
                    at = candidate
                    if self.flags[off // stride] >> NEED_SHIFT:
                        # The start state depends on what precedes it.
                        off = (
                            Int(self.start_state(program, text, at, False))
                            * stride
                        )
                        table = self.transitions.unsafe_ptr()
        var s = off // stride
        var e = Int(self.transitions[off + self.eot])
        var next: Int
        if e == Int(UNKNOWN):
            if self._full():
                s = Int(self._reset(program, s))
            next = Int(self._compute(program, s, self.eot))
        elif e == Int(DEAD):
            next = -1
        elif e >= 0:
            next = e // stride
        else:
            next = ((-e - 3) >> 2) // stride
        if next >= 0 and self.flags[next] & FLAG_MATCH:
            last = n
        return last

    @always_inline
    def find_reverse(
        mut self,
        program: Program,
        text: Span[UInt8, ImmutAnyOrigin],
        end: Int,
        floor: Int,
    ) -> Int:
        if program.has_suffix:
            return self._find_reverse_suffix(program, text, end, floor)
        return self._find_reverse[False](program, text, end, floor)

    @no_inline
    def _find_reverse_suffix(
        mut self,
        program: Program,
        text: Span[UInt8, ImmutAnyOrigin],
        end: Int,
        floor: Int,
    ) -> Int:
        return self._find_reverse[True](program, text, end, floor)

    @always_inline
    def _find_reverse[
        use_suffix: Bool
    ](
        mut self,
        program: Program,
        text: Span[UInt8, ImmutAnyOrigin],
        end: Int,
        floor: Int,
    ) -> Int:
        """The smallest start in [floor, end] of a match of the reversed
        program ending at `end`, -1 for none, or GAVE_UP."""
        var p = text.unsafe_ptr()
        var ascii = program.ascii_class.unsafe_ptr()
        var stride = self.stride
        var s0 = Int(self.start_state(program, text, end, True))
        if s0 == Int(DEAD):
            return -1
        var off = s0 * stride
        var last = -1
        var at = end
        var budget_mark = end
        var resets = self.resets
        var table = self.transitions.unsafe_ptr()
        while True:
            var cls: Int
            var prev = at
            if at > 0:
                # At `floor`, one code point past the range, only to see
                # whether a match starts at `floor`.
                prev = at - 1
                var b = p[unsafe_offset=prev]
                if b < 0x80:
                    cls = Int(ascii[unsafe_offset=Int(b)])
                else:
                    prev = previous_start(text, at)
                    cls = program.class_of(UInt32(decode(text, prev) >> 3))
            else:
                cls = self.eot
            var e = Int(table[unsafe_offset=off + cls])
            if e == Int(UNKNOWN):
                var s = off // stride
                if self._full():
                    if (
                        self.resets - resets >= 1
                        and budget_mark - at
                        < _MIN_BYTES_PER_STATE * self.state_count()
                    ):
                        return GAVE_UP
                    budget_mark = at
                    s = Int(self._reset(program, s))
                var next = self._compute(program, s, cls)
                table = self.transitions.unsafe_ptr()
                e = Int(self._entry(next))
            if e == Int(DEAD):
                return last
            comptime if use_suffix:
                if e < SUFFIX_LIMIT:
                    if at <= floor:
                        return floor
                    return suffix_end[True](text, prev, floor, Int32(e))
            if e >= 0:
                off = e
            else:
                var v = -e - 3
                off = v >> 2
                if v & 1:
                    last = at
            if at <= floor:
                return last
            at = prev


struct Prefilter(Copyable, Movable):
    """Finds where a match can begin, for skipping input while the DFA sits
    in its start state (RE2's prefix accel, regex-automata's prefilters):
    a literal every match starts with, or a sequence of small byte sets
    every match starts with (`(?i)ab` is {A, a} then {B, b}; `(fox|dog)`
    is {f, d}, {o}, {x, g}). Literal alternatives preserve correlations
    between positions and allow unequal lengths."""

    var prefix: List[UInt8]
    # Four bytes per position (a set of fewer repeats its first byte).
    var sets: List[UInt8]
    var literals: List[List[UInt8]]
    var literal_width: Int
    var masks: List[SIMD[DType.uint8, 16]]

    def __init__(out self):
        self.prefix = List[UInt8]()
        self.sets = List[UInt8]()
        self.literals = List[List[UInt8]]()
        self.literal_width = 0
        self.masks = List[SIMD[DType.uint8, 16]]()

    def __init__(
        out self,
        var prefix: List[UInt8],
        var sets: List[UInt8],
        var literals: List[List[UInt8]],
    ):
        self.prefix = prefix^
        self.sets = sets^
        self.literals = List[List[UInt8]]()
        self.literal_width = 0
        self.masks = List[SIMD[DType.uint8, 16]]()
        if len(self.sets) <= 4 * len(self.prefix):
            self.sets.clear()
        else:
            self.prefix.clear()
        # Preserve the cheaper single-literal and byte-set scanners.
        # The new path covers alternatives those analyses cannot represent.
        if len(self.prefix) == 0 and len(self.sets) == 0 and len(literals) > 1:
            var width = len(literals[0])
            for literal in literals:
                width = min(width, len(literal))
            if width > 1:
                self.literals = literals^
                self.literal_width = min(width, 3)
                for _ in range(6):
                    self.masks.append(SIMD[DType.uint8, 16]())
                for k in range(len(self.literals)):
                    var bit = UInt8(1 << k)
                    for j in range(self.literal_width):
                        var byte = self.literals[k][j]
                        self.masks[j * 2][Int(byte & 15)] |= bit
                        self.masks[j * 2 + 1][Int(byte >> 4)] |= bit

    @always_inline
    def active(self) -> Bool:
        return len(self.prefix) > 0 or len(self.sets) > 0

    def find(self, text: Span[UInt8, ImmutAnyOrigin], at: Int) -> Int:
        if len(self.prefix) > 0:
            return find_literal(text, self.prefix, at)
        return find_sets(text, self.sets, at)

    @no_inline
    def _find_literals(self, text: Span[UInt8, ImmutAnyOrigin], at: Int) -> Int:
        if self.literal_width == 3:
            return self._find_teddy[3](text, at)
        return self._find_teddy[2](text, at)

    def _find_teddy[
        depth: Int
    ](self, text: Span[UInt8, ImmutAnyOrigin], at: Int) -> Int:
        """Nibble lookup masks test up to eight prefixes together. Each
        bit names one literal, so intersecting all masks preserves byte
        correlations without a loop over literals in the vector scan.
        Full verification and the DFA retain regex priority/semantics."""
        var n = len(text)
        var p = text.unsafe_ptr()
        var lo0 = self.masks[0]
        var hi0 = self.masks[1]
        var lo1 = self.masks[2]
        var hi1 = self.masks[3]
        var lo2 = self.masks[4]
        var hi2 = self.masks[5]
        var i = at
        comptime lanes = 16
        while i + lanes + depth - 1 <= n:
            var a = p.unsafe_load[width=lanes](i)
            var b = p.unsafe_load[width=lanes](i + 1)
            var hits = (
                lo0._dynamic_shuffle(a & 15)
                & hi0._dynamic_shuffle(a >> 4)
                & lo1._dynamic_shuffle(b & 15)
                & hi1._dynamic_shuffle(b >> 4)
            )
            comptime if depth == 3:
                var c = p.unsafe_load[width=lanes](i + 2)
                hits &= lo2._dynamic_shuffle(c & 15) & hi2._dynamic_shuffle(
                    c >> 4
                )
            if hits.ne(SIMD[DType.uint8, lanes]()).reduce_or():
                for k in range(lanes):
                    if hits[k] != 0:
                        for literal in range(len(self.literals)):
                            if hits[k] & UInt8(1 << literal):
                                if _literal_matches(
                                    text, self.literals[literal], i + k
                                ):
                                    return i + k
            i += lanes
        while i + depth <= n:
            if _literal_at(text, self.literals, i):
                return i
            i += 1
        return -1


def _literal_at(
    text: Span[UInt8, ImmutAnyOrigin], literals: List[List[UInt8]], at: Int
) -> Bool:
    for literal in literals:
        if _literal_matches(text, literal, at):
            return True
    return False


@always_inline
def _literal_matches(
    text: Span[UInt8, ImmutAnyOrigin], literal: List[UInt8], at: Int
) -> Bool:
    if len(literal) > len(text) - at:
        return False
    var p = text.unsafe_ptr()
    for j in range(len(literal)):
        if p[unsafe_offset=at + j] != literal[j]:
            return False
    return True


@always_inline
def _in_set(sets: Pointer[UInt8, ImmutAnyOrigin], k: Int, b: UInt8) -> Bool:
    var q = sets.unsafe_offset(4 * k)
    return (
        b == q[unsafe_offset=0]
        or b == q[unsafe_offset=1]
        or b == q[unsafe_offset=2]
        or b == q[unsafe_offset=3]
    )


def find_sets(
    text: Span[UInt8, ImmutAnyOrigin], sets: List[UInt8], at: Int
) -> Int:
    """The first position at or after `at` where every byte set of the
    sequence matches in turn, or -1. Candidates come 32 at a time from the
    first and last sets, then the middle is checked."""
    var m = len(sets) // 4
    var n = len(text)
    if m > n - at:
        return -1
    var h = text.unsafe_ptr()
    var q = (
        sets.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    comptime lanes = 32
    var f0 = SIMD[DType.uint8, lanes](sets[0])
    var f1 = SIMD[DType.uint8, lanes](sets[1])
    var f2 = SIMD[DType.uint8, lanes](sets[2])
    var f3 = SIMD[DType.uint8, lanes](sets[3])
    var e = 4 * (m - 1)
    var l0 = SIMD[DType.uint8, lanes](sets[e])
    var l1 = SIMD[DType.uint8, lanes](sets[e + 1])
    var l2 = SIMD[DType.uint8, lanes](sets[e + 2])
    var l3 = SIMD[DType.uint8, lanes](sets[e + 3])
    var i = at
    while i + m - 1 + lanes <= n:
        var a = h.unsafe_load[width=lanes](i)
        var z = h.unsafe_load[width=lanes](i + m - 1)
        var hits = (a.eq(f0) | a.eq(f1) | a.eq(f2) | a.eq(f3)) & (
            z.eq(l0) | z.eq(l1) | z.eq(l2) | z.eq(l3)
        )
        if hits.reduce_or():
            for k in range(lanes):
                if hits[k]:
                    var same = True
                    for j in range(1, m - 1):
                        if not _in_set(q, j, h[unsafe_offset=i + k + j]):
                            same = False
                            break
                    if same:
                        return i + k
        i += lanes
    while i + m <= n:
        var same = True
        for j in range(m):
            if not _in_set(q, j, h[unsafe_offset=i + j]):
                same = False
                break
        if same:
            return i
        i += 1
    return -1


@always_inline
def _verify_literal(
    hay: Pointer[UInt8, ImmutAnyOrigin],
    needle: Pointer[UInt8, ImmutAnyOrigin],
    length: Int,
    base: Int,
    var candidates: UInt64,
) -> Int:
    while candidates:
        var k = Int(count_trailing_zeros(candidates))
        var same = True
        for j in range(1, length - 1):
            if hay[unsafe_offset=base + k + j] != needle[unsafe_offset=j]:
                same = False
                break
        if same:
            return base + k
        candidates &= candidates - 1
    return -1


@always_inline
def _literal_chunk[
    lanes: Int
](
    hay: Pointer[UInt8, ImmutAnyOrigin],
    needle: Pointer[UInt8, ImmutAnyOrigin],
    length: Int,
    at: Int,
    firsts: SIMD[DType.uint8, lanes],
    lasts: SIMD[DType.uint8, lanes],
) -> Int:
    var hits = hay.unsafe_load[width=lanes](at).eq(firsts) & hay.unsafe_load[
        width=lanes
    ](at + length - 1).eq(lasts)
    comptime Mask = (
        DType.uint64 if lanes
        == 64 else DType.uint32 if lanes
        == 32 else DType.uint16
    )
    var candidates = UInt64(pack_bits[Mask](hits))
    if not candidates:
        return -1
    # A one- or two-byte needle has no interior bytes to verify. Let the
    # empty verification loop handle it, keeping miss masks off that path.
    return _verify_literal(hay, needle, length, at, candidates)


@no_inline
def find_literal(
    text: Span[UInt8, ImmutAnyOrigin], needle: List[UInt8], at: Int
) -> Int:
    """memmem: the first occurrence of `needle` at or after `at`, or -1.
    Check the first 32 candidates before scanning long misses 64 at a time,
    filtering on the needle's first and last bytes."""
    var m = len(needle)
    var n = len(text)
    if m == 0:
        return at if at <= n else -1
    if m > n - at:
        return -1
    var h = text.unsafe_ptr()
    var p = needle.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin]()
    var first = p[]
    var last = p[unsafe_offset=m - 1]
    comptime lanes = 32
    var firsts = SIMD[DType.uint8, lanes](first)
    var lasts = SIMD[DType.uint8, lanes](last)
    # Both loads must end inside the haystack. Hoist the last legal
    # start so the miss loop only advances and compares one position.
    var last_start = n - m
    var vector_end = last_start - (lanes - 1)
    var i = at
    if i <= vector_end:
        var found = _literal_chunk[32](h, p, m, i, firsts, lasts)
        if found >= 0:
            return found
        i += lanes
    # Two vectors per load amortize the cursor, mask and branch work over
    # twice as many bytes. Verification runs only for candidate positions.
    var wide_firsts = SIMD[DType.uint8, 64](first)
    var wide_lasts = SIMD[DType.uint8, 64](last)
    var bulk_end = last_start - 63
    while i <= bulk_end:
        var found = _literal_chunk[64](h, p, m, i, wide_firsts, wide_lasts)
        if found >= 0:
            return found
        i += 64
    if i <= vector_end:
        var found = _literal_chunk[32](h, p, m, i, firsts, lasts)
        if found >= 0:
            return found
        i += lanes
    if i <= last_start - 15:
        var found = _literal_chunk[16](
            h,
            p,
            m,
            i,
            SIMD[DType.uint8, 16](first),
            SIMD[DType.uint8, 16](last),
        )
        if found >= 0:
            return found
        i += 16
    while i <= last_start:
        if h[unsafe_offset=i] == first and h[unsafe_offset=i + m - 1] == last:
            var same = True
            for j in range(1, m - 1):
                if h[unsafe_offset=i + j] != p[unsafe_offset=j]:
                    same = False
                    break
            if same:
                return i
        i += 1
    return -1
