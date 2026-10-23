"""The Pike VM: an NFA simulation that tracks capture positions, as in RE2's
`NFA` and regex-automata's `PikeVM`.

Every live thread is one program counter with its own capture slots; all of
them advance together one code point at a time, and a sparse set drops
threads that reach an instruction another (higher priority) thread already
holds, so the work per code point is bounded by the program size. Threads
are kept in priority order, which gives leftmost-first ("Perl") match
semantics: once a thread matches, the lower priority ones are cut.

The VM is scratch space for one worker; the program it runs is shared.
"""
from .program import (
    OP_FAIL,
    OP_LOOK,
    OP_MATCH,
    OP_NOP,
    OP_SAVE,
    OP_SPLIT,
    Program,
    consumes,
)
from .syntax import (
    LOOK_END_LINE,
    LOOK_END_TEXT,
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


@always_inline
def decode(text: Span[UInt8, ImmutAnyOrigin], at: Int) -> Int:
    """The code point at `at` and its width, packed as `cp << 3 | width`.
    Truncated sequences decode as U+FFFD, one byte wide."""
    var p = text.unsafe_ptr()
    var b0 = Int(p[unsafe_offset=at])
    if b0 < 0x80:
        return (b0 << 3) | 1
    var n = len(text)
    if b0 < 0xE0:
        if at + 1 < n:
            return (
                (((b0 & 0x1F) << 6) | (Int(p[unsafe_offset=at + 1]) & 0x3F))
                << 3
            ) | 2
    elif b0 < 0xF0:
        if at + 2 < n:
            return (
                (
                    ((b0 & 0x0F) << 12)
                    | ((Int(p[unsafe_offset=at + 1]) & 0x3F) << 6)
                    | (Int(p[unsafe_offset=at + 2]) & 0x3F)
                )
                << 3
            ) | 3
    elif at + 3 < n:
        return (
            (
                ((b0 & 0x07) << 18)
                | ((Int(p[unsafe_offset=at + 1]) & 0x3F) << 12)
                | ((Int(p[unsafe_offset=at + 2]) & 0x3F) << 6)
                | (Int(p[unsafe_offset=at + 3]) & 0x3F)
            )
            << 3
        ) | 4
    return (0xFFFD << 3) | 1


@always_inline
def previous_start(text: Span[UInt8, ImmutAnyOrigin], at: Int) -> Int:
    """Where the code point ending at `at` begins."""
    var p = text.unsafe_ptr()
    var i = at - 1
    while i > 0 and at - i < 4 and (Int(p[unsafe_offset=i]) & 0xC0) == 0x80:
        i -= 1
    return i


@always_inline
def _ascii_word(cp: UInt32) -> Bool:
    return (
        (cp >= 48 and cp <= 57)
        or (cp >= 65 and cp <= 90)
        or cp == 95
        or (cp >= 97 and cp <= 122)
    )


def look_holds(
    program: Program, look: UInt32, text: Span[UInt8, ImmutAnyOrigin], at: Int
) -> Bool:
    """Whether a zero-width assertion holds at byte offset `at`."""
    var n = len(text)
    var p = text.unsafe_ptr()
    if look == LOOK_START_TEXT:
        return at == 0
    if look == LOOK_END_TEXT:
        return at == n
    if look == LOOK_START_LINE:
        return at == 0 or p[unsafe_offset=at - 1] == 10
    if look == LOOK_END_LINE:
        return at == n or p[unsafe_offset=at] == 10
    var before = False
    if at > 0:
        before = program.is_word(
            UInt32(decode(text, previous_start(text, at)) >> 3)
        )
    var after = False
    if at < n:
        after = program.is_word(UInt32(decode(text, at) >> 3))
    if look & LOOK_ASCII:
        before = at > 0 and _ascii_word(
            UInt32(text.unsafe_ptr()[unsafe_offset=at - 1])
        )
        after = at < n and _ascii_word(
            UInt32(text.unsafe_ptr()[unsafe_offset=at])
        )
    if look == LOOK_WORD or look == LOOK_ASCII_WORD:
        return before != after
    if look == LOOK_WORD_START or look == LOOK_ASCII_WORD_START:
        return not before and after
    if look == LOOK_WORD_END or look == LOOK_ASCII_WORD_END:
        return before and not after
    return before == after


struct _Threads(Movable):
    """Threads in priority order (`dense`), a sparse index for membership,
    and each thread's capture slots in `slots[pc * width ...]`."""

    var dense: List[Int32]
    var sparse: List[Int32]
    var size: Int
    var slots: List[Int]
    var width: Int

    def __init__(out self, length: Int, width: Int):
        self.dense = List[Int32](length=length, fill=0)
        self.sparse = List[Int32](length=length, fill=0)
        self.size = 0
        self.slots = List[Int](length=length * width, fill=-1)
        self.width = width

    @always_inline
    def contains(self, pc: Int) -> Bool:
        var i = Int(self.sparse[pc])
        return i < self.size and Int(self.dense[i]) == pc

    @always_inline
    def insert(mut self, pc: Int):
        self.sparse[pc] = Int32(self.size)
        self.dense[self.size] = Int32(pc)
        self.size += 1


struct PikeVM(Movable):
    var current: _Threads
    var next: _Threads
    var stack: List[Int]
    var scratch: List[Int]
    var width: Int

    def __init__(out self, program: Program, slots: Int):
        """`slots` is how many capture slots to track: 2 for the whole
        match, up to `program.slot_count` for every group."""
        self.width = slots
        self.current = _Threads(len(program), slots)
        self.next = _Threads(len(program), slots)
        self.stack = List[Int]()
        self.scratch = List[Int](length=slots, fill=-1)

    def _follow(
        mut self,
        program: Program,
        text: Span[UInt8, ImmutAnyOrigin],
        start_pc: Int,
        at: Int,
        into_next: Bool,
    ):
        """Add the thread at `start_pc` (with slots `scratch`) and every
        thread reachable from it without consuming input, in priority
        order. SAVE writes `scratch` and pushes a frame that restores it,
        as regex-automata's `epsilon_closure` does."""
        self.stack.append(start_pc)
        while len(self.stack) > 0:
            var pc = self.stack.pop()
            if pc < 0:
                # A restore frame: -(slot + 1) above the old value.
                var old = self.stack.pop()
                self.scratch[-pc - 1] = old
                continue
            while True:
                if into_next:
                    if self.next.contains(pc):
                        break
                    self.next.insert(pc)
                else:
                    if self.current.contains(pc):
                        break
                    self.current.insert(pc)
                var op = program.op[pc]
                if op == OP_SPLIT:
                    self.stack.append(Int(program.arg[pc]))
                    pc = Int(program.out[pc])
                    continue
                if op == OP_NOP:
                    pc = Int(program.out[pc])
                    continue
                if op == OP_SAVE:
                    var slot = Int(program.arg[pc])
                    if slot < self.width:
                        self.stack.append(self.scratch[slot])
                        self.stack.append(-slot - 1)
                        self.scratch[slot] = at
                    pc = Int(program.out[pc])
                    continue
                if op == OP_LOOK:
                    if look_holds(program, UInt32(program.arg[pc]), text, at):
                        pc = Int(program.out[pc])
                        continue
                    break
                if op == OP_FAIL:
                    break
                # A consuming instruction or MATCH: record its slots.
                var base = pc * self.width
                for s in range(self.width):
                    if into_next:
                        self.next.slots[base + s] = self.scratch[s]
                    else:
                        self.current.slots[base + s] = self.scratch[s]
                break

    def search(
        mut self,
        program: Program,
        text: Span[UInt8, ImmutAnyOrigin],
        start: Int,
        end: Int,
        anchored: Bool,
        mut slots: List[Int],
    ) -> Bool:
        """The leftmost-first match starting at or after `start` and ending
        at or before `end` (assertions see the whole text). Fills `slots`
        (length >= the VM's slot count) and returns whether one was
        found."""
        self.current.size = 0
        self.next.size = 0
        var matched = False
        var at = start
        while True:
            if self.current.size == 0 and (
                matched or (anchored and at > start)
            ):
                break
            if not matched and (not anchored or at == start):
                for s in range(self.width):
                    self.scratch[s] = -1
                self._follow(program, text, program.start, at, False)
            var cp: UInt32 = 0
            var width = 0
            if at < end:
                var packed = decode(text, at)
                cp = UInt32(packed >> 3)
                width = packed & 7
            self.next.size = 0
            for i in range(self.current.size):
                var pc = Int(self.current.dense[i])
                var op = program.op[pc]
                if op == OP_MATCH:
                    var base = pc * self.width
                    for s in range(self.width):
                        slots[s] = self.current.slots[base + s]
                    matched = True
                    break
                if width > 0 and consumes(op) and program.step(pc, cp):
                    var base = pc * self.width
                    for s in range(self.width):
                        self.scratch[s] = self.current.slots[base + s]
                    self._follow(
                        program,
                        text,
                        Int(program.out[pc]),
                        at + width,
                        True,
                    )
            if at >= end:
                break
            swap(self.current, self.next)
            at += width
        return matched
