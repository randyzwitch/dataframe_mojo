"""Bounded deterministic capture tables for unambiguous Thompson programs.

Each alphabet class may reach at most one consuming instruction from a state.
SAVE operations become a bit mask; assertions remain conditions on that edge.
Ambiguity, epsilon cycles and resource limits leave the existing engines in
charge. The construction follows the one-pass property described by RE2;
this implementation compiles this engine's code-point alphabet. ASCII uses
dense class tables; Unicode tests the state's disjoint consuming predicates
without searching the complete alphabet. Assertions see the full input.
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
from .pikevm import decode, look_holds
from .syntax import LOOK_MASK

comptime _MAX_STATES = 256
comptime _MAX_CELLS = 32768
comptime _MAX_PROGRAM = 512
comptime _MAX_SLOTS = 64
# LOOK_MASK occupies bits 0..13. An edge's next state plus one occupies
# the upper bits, leaving zero as a dead edge. The state's match tag uses
# the same spare bit as an edge's priority tag, in a separate table.
comptime _MATCH_TAG: UInt32 = 1 << 14
comptime _NEXT_SHIFT = 15


struct OnePass(Movable):
    var actions: List[UInt32]
    var save: List[UInt64]
    var match_cond: List[UInt32]
    var match_save: List[UInt64]
    # Unicode edges refer to the same actions as the dense ASCII table.
    # Their predicates are disjoint, as checked during construction.
    var terminal_starts: List[Int32]
    var terminal_pc: List[Int32]
    var terminal_edge: List[Int32]

    def __init__(out self):
        self.actions = List[UInt32]()
        self.save = List[UInt64]()
        self.match_cond = List[UInt32]()
        self.match_save = List[UInt64]()
        self.terminal_starts = List[Int32]()
        self.terminal_pc = List[Int32]()
        self.terminal_edge = List[Int32]()

    @always_inline
    def active(self) -> Bool:
        return len(self.match_cond) > 0

    def search[
        unicode_input: Bool
    ](
        self,
        program: Program,
        text: Span[UInt8, ImmutAnyOrigin],
        start: Int,
        end: Int,
        mut current: List[Int],
        mut slots: List[Int],
    ) -> Bool:
        # The ASCII instantiation's caller verified [start, end). Both
        # instantiations inspect the full text when checking assertions.
        # All tables are immutable and all scratch storage is fixed for
        # this search. Hoist their addresses out of the per-byte loop.
        var actions = self.actions.unsafe_ptr()
        var saves = self.save.unsafe_ptr()
        var conditions = self.match_cond.unsafe_ptr()
        var match_saves = self.match_save.unsafe_ptr()
        var classes = program.ascii_class.unsafe_ptr()
        var terminal_starts = self.terminal_starts.unsafe_ptr()
        var terminal_pc = self.terminal_pc.unsafe_ptr()
        var terminal_edge = self.terminal_edge.unsafe_ptr()
        var bytes = text.unsafe_ptr()
        var count = len(slots)
        var captures = current.unsafe_ptr()
        var output = slots.unsafe_ptr()
        for k in range(count):
            captures[unsafe_offset=k] = -1
        var state = 0
        var at = start
        var found = False
        var alphabet = program.alphabet
        while True:
            var edge = -1
            var width = 1
            if at < end:
                comptime if unicode_input:
                    var packed = decode(text, at)
                    var cp = UInt32(packed >> 3)
                    width = packed & 7
                    if cp < 128:
                        edge = state * alphabet + Int(
                            classes[unsafe_offset=Int(cp)]
                        )
                    else:
                        var first_terminal = Int(
                            terminal_starts[unsafe_offset=state]
                        )
                        var last_terminal = Int(
                            terminal_starts[unsafe_offset=state + 1]
                        )
                        for k in range(first_terminal, last_terminal):
                            if program.step(
                                Int(terminal_pc[unsafe_offset=k]), cp
                            ):
                                edge = Int(terminal_edge[unsafe_offset=k])
                                break
                else:
                    var cls = Int(
                        classes[unsafe_offset=Int(bytes[unsafe_offset=at])]
                    )
                    edge = state * alphabet + cls
            var condition = conditions[unsafe_offset=state]
            var accepts = condition & _MATCH_TAG != 0 and _holds(
                program, condition & LOOK_MASK, text, at
            )
            var advances = edge >= 0
            var next = -1
            var action: UInt32 = 0
            if advances:
                action = actions[unsafe_offset=edge]
                next = Int(action >> _NEXT_SHIFT) - 1
                advances = next >= 0 and _holds(
                    program, action & LOOK_MASK, text, at
                )
            if accepts and advances and action & _MATCH_TAG == 0:
                # A certain higher-priority acceptance supersedes this
                # snapshot on the next iteration.
                if conditions[unsafe_offset=next] == _MATCH_TAG:
                    accepts = False
            if accepts:
                for k in range(count):
                    output[unsafe_offset=k] = captures[unsafe_offset=k]
                var mask = match_saves[unsafe_offset=state]
                while mask:
                    var k = Int(count_trailing_zeros(mask))
                    output[unsafe_offset=k] = at
                    mask &= mask - 1
                found = True
                if not advances or action & _MATCH_TAG != 0:
                    return True
            if not advances:
                return found
            var mask = saves[unsafe_offset=edge]
            while mask:
                var k = Int(count_trailing_zeros(mask))
                captures[unsafe_offset=k] = at
                mask &= mask - 1
            state = next
            at += width


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
        result.match_cond.append(
            (_MATCH_TAG | match_looks) if match_rank >= 0 else UInt32(0)
        )
        result.match_save.append(match_save)
        result.terminal_starts.append(Int32(len(result.terminal_pc)))
        var edges = List[Int](length=len(terminals), fill=-1)
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
                if edges[terminal] < 0:
                    edges[terminal] = len(result.actions)
                var pc = Int(program.out[terminals[terminal]])
                next = state_of[pc]
                if next < 0:
                    next = len(starts)
                    state_of[pc] = next
                    starts.append(pc)
                save = terminal_save[terminal]
                flags = terminal_looks[terminal]
                first = match_rank >= 0 and match_rank <= terminal
            var action = UInt32(next + 1) << _NEXT_SHIFT
            action |= flags
            if first:
                action |= _MATCH_TAG
            result.actions.append(action)
            result.save.append(save)
        for k in range(len(terminals)):
            if edges[k] >= 0:
                result.terminal_pc.append(Int32(terminals[k]))
                result.terminal_edge.append(Int32(edges[k]))
        state += 1
    result.terminal_starts.append(Int32(len(result.terminal_pc)))
    return result^


def ascii_span(text: Span[UInt8, ImmutAnyOrigin], start: Int, end: Int) -> Bool:
    """Whether a span can use byte-indexed one-pass capture transitions."""
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
