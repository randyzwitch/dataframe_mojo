"""A bounded backtracker for capture groups, as RE2's `BitState` and
regex-automata's `BoundedBacktracker`.

It explores the NFA depth first in priority order, so the first match it
reaches is the leftmost-first one, and it never visits the same
(instruction, position) pair twice: a bitset of `len(program)` x
(positions + 1) bits keeps the work linear in the span searched. For the
short spans a dataframe cell holds this is much cheaper than the Pike VM,
which copies every thread's slots at every step; `Matcher` uses it when the
bitset stays within `MAX_VISITED_BITS` and the Pike VM otherwise.
"""
from .pikevm import decode, look_holds
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

# regex-automata's default visited capacity: 256 KiB of bits.
comptime MAX_VISITED_BITS = 256 * 1024 * 8


struct Backtracker(Movable):
    var visited: List[UInt64]
    # Frames: an instruction and a position, or a slot to restore (the
    # instruction is then -(slot + 1)).
    var stack_pc: List[Int]
    var stack_at: List[Int]

    def __init__(out self):
        self.visited = List[UInt64]()
        self.stack_pc = List[Int]()
        self.stack_at = List[Int]()

    @staticmethod
    def fits(program: Program, start: Int, end: Int) -> Bool:
        return len(program) * (end - start + 1) <= MAX_VISITED_BITS

    def search(
        mut self,
        program: Program,
        text: Span[UInt8, ImmutAnyOrigin],
        start: Int,
        end: Int,
        mut slots: List[Int],
    ) -> Bool:
        """The leftmost-first match anchored at `start` and ending at or
        before `end` (assertions see the whole text), filling `slots`."""
        var width = end - start + 1
        var words = (len(program) * width + 63) // 64
        if len(self.visited) < words:
            self.visited = List[UInt64](length=words, fill=0)
        else:
            for i in range(words):
                self.visited[i] = 0
        for s in range(len(slots)):
            slots[s] = -1
        self.stack_pc.clear()
        self.stack_at.clear()
        self.stack_pc.append(program.start)
        self.stack_at.append(start)
        while len(self.stack_pc) > 0:
            var pc = self.stack_pc.pop()
            var at = self.stack_at.pop()
            if pc < 0:
                slots[-pc - 1] = at
                continue
            while True:
                var bit = pc * width + (at - start)
                var word = bit >> 6
                var mask = UInt64(1) << UInt64(bit & 63)
                if self.visited[word] & mask:
                    break
                self.visited[word] |= mask
                var op = program.op[pc]
                if op == OP_MATCH:
                    return True
                if consumes(op):
                    if at >= end:
                        break
                    var packed = decode(text, at)
                    if not program.step(pc, UInt32(packed >> 3)):
                        break
                    at += packed & 7
                    pc = Int(program.out[pc])
                    continue
                if op == OP_SPLIT:
                    self.stack_pc.append(Int(program.arg[pc]))
                    self.stack_at.append(at)
                    pc = Int(program.out[pc])
                    continue
                if op == OP_SAVE:
                    var slot = Int(program.arg[pc])
                    if slot < len(slots):
                        self.stack_pc.append(-slot - 1)
                        self.stack_at.append(slots[slot])
                        slots[slot] = at
                    pc = Int(program.out[pc])
                    continue
                if op == OP_LOOK:
                    if not look_holds(
                        program, UInt32(program.arg[pc]), text, at
                    ):
                        break
                    pc = Int(program.out[pc])
                    continue
                if op == OP_NOP:
                    pc = Int(program.out[pc])
                    continue
                # OP_FAIL
                break
        return False
