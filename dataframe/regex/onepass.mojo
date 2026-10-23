"""Bounded deterministic capture tables for unambiguous Thompson programs.

Each alphabet class may reach at most one consuming instruction from a state.
SAVE operations become a bit mask; assertions remain conditions on that edge.
Ambiguity, epsilon cycles and resource limits leave the existing engines in
charge. The construction follows the one-pass property described by RE2;
this implementation compiles this engine's code-point alphabet and uses
its ASCII classes for execution. Unicode spans retain the original capture
engines; assertions may inspect Unicode outside the captured ASCII span.
"""
from std.bit import count_trailing_zeros
from .program import (
    Program,
    consumes,
    OP_MATCH,
    OP_SPLIT,
    OP_SAVE,
    OP_LOOK,
    OP_NOP,
)
from .pikevm import look_holds

comptime _MAX_STATES = 256
comptime _MAX_CELLS = 32768
comptime _MAX_PROGRAM = 512
comptime _MAX_SLOTS = 64


struct OnePass(Movable):
    var next: List[Int32]
    var save: List[UInt64]
    var looks: List[UInt32]
    var match_first: List[Bool]
    var matched: List[Bool]
    var match_save: List[UInt64]
    var match_looks: List[UInt32]

    def __init__(out self):
        self.next = List[Int32]()
        self.save = List[UInt64]()
        self.looks = List[UInt32]()
        self.match_first = List[Bool]()
        self.matched = List[Bool]()
        self.match_save = List[UInt64]()
        self.match_looks = List[UInt32]()

    @always_inline
    def active(self) -> Bool:
        return len(self.matched) > 0

    def search(
        self,
        program: Program,
        text: Span[UInt8, ImmutAnyOrigin],
        start: Int,
        end: Int,
        mut current: List[Int],
        mut slots: List[Int],
    ) -> Bool:
        # Caller verified [start, end) is ASCII. Assertions still inspect
        # the full text and can see Unicode outside this captured span.
        for k in range(len(slots)):
            current[k] = -1
            slots[k] = -1
        var state = 0
        var at = start
        var found = False
        var alphabet = program.alphabet
        while True:
            var edge = -1
            if at < end:
                var cls = Int(program.ascii_class[Int(text[at])])
                edge = state * alphabet + cls
            var accepts = self.matched[state] and _holds(
                program, self.match_looks[state], text, at
            )
            var advances = edge >= 0
            if advances:
                advances = self.next[edge] >= 0 and _holds(
                    program, self.looks[edge], text, at
                )
            if accepts and advances and not self.match_first[edge]:
                var next = Int(self.next[edge])
                # A certain, higher-priority acceptance after consuming
                # this byte dominates the current match. Avoid copying a
                # snapshot that the next iteration must supersede.
                if self.matched[next] and self.match_looks[next] == 0:
                    accepts = False
            if accepts:
                for k in range(len(slots)):
                    slots[k] = current[k]
                _apply(self.match_save[state], at, slots)
                found = True
                if not advances or self.match_first[edge]:
                    return True
            if not advances:
                return found
            _apply(self.save[edge], at, current)
            state = Int(self.next[edge])
            at += 1


@always_inline
def _apply(var mask: UInt64, at: Int, mut slots: List[Int]):
    while mask:
        var k = Int(count_trailing_zeros(mask))
        slots[k] = at
        mask &= mask - 1


@always_inline
def _holds(
    program: Program,
    var flags: UInt32,
    text: Span[UInt8, ImmutAnyOrigin],
    at: Int,
) -> Bool:
    while flags:
        var bit = UInt32(1) << count_trailing_zeros(flags)
        if not look_holds(program, bit, text, at):
            return False
        flags &= flags - 1
    return True


def compile_onepass(program: Program) -> OnePass:
    var result = OnePass()
    if (
        program.slot_count <= 2
        or program.slot_count > _MAX_SLOTS
        or len(program) > _MAX_PROGRAM
    ):
        return result^
    var alphabet = program.alphabet
    if alphabet > _MAX_CELLS:
        return result^
    var starts = List[Int]()
    starts.append(program.start)
    var state_of = List[Int](length=len(program), fill=-1)
    state_of[program.start] = 0
    var state = 0
    while state < len(starts):
        if len(starts) > _MAX_STATES or len(starts) * alphabet > _MAX_CELLS:
            return OnePass()
        var visited = List[Bool](length=len(program), fill=False)
        var pcs = List[Int]()
        var saves = List[UInt64]()
        var looks = List[UInt32]()
        pcs.append(starts[state])
        saves.append(0)
        looks.append(0)
        var terminals = List[Int]()
        var terminal_save = List[UInt64]()
        var terminal_looks = List[UInt32]()
        var match_rank = -1
        var match_save: UInt64 = 0
        var match_looks: UInt32 = 0
        while len(pcs):
            var pc = pcs.pop()
            var save = saves.pop()
            var flags = looks.pop()
            if visited[pc]:
                # Merging epsilon paths can carry different captures or
                # priorities. Reject conservatively, including cycles.
                return OnePass()
            visited[pc] = True
            var op = program.op[pc]
            if consumes(op):
                terminals.append(pc)
                terminal_save.append(save)
                terminal_looks.append(flags)
            elif op == OP_MATCH:
                if match_rank >= 0:
                    return OnePass()
                match_rank = len(terminals)
                match_save = save
                match_looks = flags
            elif op == OP_SPLIT:
                pcs.append(Int(program.arg[pc]))
                saves.append(save)
                looks.append(flags)
                pcs.append(Int(program.out[pc]))
                saves.append(save)
                looks.append(flags)
            elif op == OP_SAVE or op == OP_LOOK or op == OP_NOP:
                if op == OP_SAVE:
                    save |= UInt64(1) << UInt64(program.arg[pc])
                elif op == OP_LOOK:
                    flags |= UInt32(program.arg[pc])
                pcs.append(Int(program.out[pc]))
                saves.append(save)
                looks.append(flags)
        result.matched.append(match_rank >= 0)
        result.match_save.append(match_save)
        result.match_looks.append(match_looks)
        for cls in range(alphabet):
            var terminal = -1
            for k in range(len(terminals)):
                if program.step(terminals[k], program.class_rep[cls]):
                    if terminal >= 0:
                        return OnePass()
                    terminal = k
            var next = -1
            var save: UInt64 = 0
            var flags: UInt32 = 0
            var first = False
            if terminal >= 0:
                var pc = Int(program.out[terminals[terminal]])
                next = state_of[pc]
                if next < 0:
                    next = len(starts)
                    state_of[pc] = next
                    starts.append(pc)
                save = terminal_save[terminal]
                flags = terminal_looks[terminal]
                first = match_rank >= 0 and match_rank <= terminal
            result.next.append(Int32(next))
            result.save.append(save)
            result.looks.append(flags)
            result.match_first.append(first)
        state += 1
    return result^


def ascii_span(text: Span[UInt8, ImmutAnyOrigin], start: Int, end: Int) -> Bool:
    """Whether a span can use byte-indexed one-pass capture transitions.
    Check before executing, so Unicode never requires replaying partial
    deterministic work in a fallback engine."""
    var p = text.unsafe_ptr()
    var at = start
    while at + 16 <= end:
        if (
            (p.unsafe_load[width=16](at) & 128)
            .ne(SIMD[DType.uint8, 16]())
            .reduce_or()
        ):
            return False
        at += 16
    while at < end:
        if p[unsafe_offset=at] >= 128:
            return False
        at += 1
    return True
