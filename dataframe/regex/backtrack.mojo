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
    # Frames as (instruction, position) pairs, or a slot to restore (the
    # instruction is then -(slot + 1) and the position the old value).
    var stack: List[Int]

    def __init__(out self):
        self.visited = List[UInt64]()
        self.stack = List[Int]()

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
            var clear = self.visited.unsafe_ptr()
            for i in range(words):
                clear[unsafe_offset=i] = 0
        for s in range(len(slots)):
            slots[s] = -1
        # The loop below reads everything through raw pointers: the
        # program and the scratch lists outlive it and do not move.
        var visited = self.visited.unsafe_ptr()
        var ops = program.op.unsafe_ptr()
        var outs = program.out.unsafe_ptr()
        var args = program.arg.unsafe_ptr()
        var slot_count = len(slots)
        self.stack.clear()
        self.stack.append(program.start)
        self.stack.append(start)
        while len(self.stack) > 0:
            var at = self.stack.pop()
            var pc = self.stack.pop()
            if pc < 0:
                slots[-pc - 1] = at
                continue
            while True:
                var bit = pc * width + (at - start)
                var word = visited.unsafe_offset(bit >> 6)
                var mask = UInt64(1) << UInt64(bit & 63)
                if word[] & mask:
                    break
                word[] |= mask
                var op = ops[unsafe_offset=pc]
                if op == OP_MATCH:
                    return True
                if consumes(op):
                    if at >= end:
                        break
                    var packed = decode(text, at)
                    if not program.step(pc, UInt32(packed >> 3)):
                        break
                    at += packed & 7
                    pc = Int(outs[unsafe_offset=pc])
                    continue
                if op == OP_SPLIT:
                    self.stack.append(Int(args[unsafe_offset=pc]))
                    self.stack.append(at)
                    pc = Int(outs[unsafe_offset=pc])
                    continue
                if op == OP_SAVE:
                    var slot = Int(args[unsafe_offset=pc])
                    if slot < slot_count:
                        self.stack.append(-slot - 1)
                        self.stack.append(slots[slot])
                        slots[slot] = at
                    pc = Int(outs[unsafe_offset=pc])
                    continue
                if op == OP_LOOK:
                    if not look_holds(
                        program, UInt32(args[unsafe_offset=pc]), text, at
                    ):
                        break
                    pc = Int(outs[unsafe_offset=pc])
                    continue
                if op == OP_NOP:
                    pc = Int(outs[unsafe_offset=pc])
                    continue
                # OP_FAIL
                break
        return False
