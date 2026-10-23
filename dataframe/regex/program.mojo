"""Thompson construction: a syntax tree compiled to an NFA program over code
points, as RE2's `Compiler` and regex-automata's `thompson::Compiler` do
(they compile to bytes; this program steps one code point at a time, so
`.` and classes are whole code points and every match starts and ends on a
character boundary).

The program also carries the DFA's alphabet: code points partitioned into
classes that no instruction distinguishes, like RE2's bytemap and
regex-automata's `ByteClasses`, so a DFA state has one transition per class
instead of one per code point.
"""
from .charclass import (
    MAX_CODEPOINT,
    _sort,
    class_contains,
    word_class,
    ascii_class,
)
from .syntax import (
    Ast,
    LOOK_NOT_WORD,
    LOOK_WORD,
    LOOK_WORD_START,
    LOOK_WORD_END,
    LOOK_ASCII_WORD,
    LOOK_ASCII_NOT_WORD,
    LOOK_ASCII_WORD_START,
    LOOK_ASCII_WORD_END,
    LOOK_UNICODE_WORD,
    LOOK_ASCII,
    NODE_ALTERNATE,
    NODE_CLASS,
    NODE_CONCAT,
    NODE_EMPTY,
    NODE_GROUP,
    NODE_LITERAL,
    NODE_LOOK,
    NODE_REPEAT,
)

comptime OP_MATCH: UInt8 = 0
comptime OP_CHAR: UInt8 = 1
comptime OP_CLASS: UInt8 = 2
comptime OP_ANY: UInt8 = 3
comptime OP_ANY_NOT_NL: UInt8 = 4
comptime OP_SPLIT: UInt8 = 5
comptime OP_SAVE: UInt8 = 6
comptime OP_LOOK: UInt8 = 7
comptime OP_NOP: UInt8 = 8
comptime OP_FAIL: UInt8 = 9

# Programs larger than this are rejected, like Rust's size limit.
comptime MAX_INSTRUCTIONS = 500_000


@always_inline
def consumes(op: UInt8) -> Bool:
    return op >= OP_CHAR and op <= OP_ANY_NOT_NL


struct Program(Movable, Sized):
    """An NFA. Instruction `pc` has opcode `op[pc]`, successor `out[pc]` and
    argument `arg[pc]`: the code point of OP_CHAR, the class of OP_CLASS,
    the second (lower priority) successor of OP_SPLIT, the slot of OP_SAVE
    or the assertion of OP_LOOK."""

    var op: List[UInt8]
    var out: List[Int32]
    var arg: List[Int32]
    var class_ranges: List[List[UInt32]]
    # 128 bits of ASCII membership per class.
    var class_ascii: List[UInt64]
    var start: Int
    var start_unanchored: Int
    var slot_count: Int
    var looks: UInt32
    # The DFA alphabet: `ascii_class` for code points below 128, else the
    # interval of `boundaries` holding the code point names its class in
    # `interval_class`. `class_rep` is one member of each class.
    var ascii_class: List[Int32]
    var boundaries: List[UInt32]
    var interval_class: List[Int32]
    var class_rep: List[UInt32]
    var class_word: List[Bool]
    var alphabet: Int
    var word: List[UInt32]
    var word_ascii: List[Bool]

    def __init__(out self):
        self.op = List[UInt8]()
        self.out = List[Int32]()
        self.arg = List[Int32]()
        self.class_ranges = List[List[UInt32]]()
        self.class_ascii = List[UInt64]()
        self.start = 0
        self.start_unanchored = 0
        self.slot_count = 2
        self.looks = 0
        self.ascii_class = List[Int32]()
        self.boundaries = List[UInt32]()
        self.interval_class = List[Int32]()
        self.class_rep = List[UInt32]()
        self.class_word = List[Bool]()
        self.alphabet = 0
        self.word = List[UInt32]()
        self.word_ascii = List[Bool](length=128, fill=False)

    def __len__(self) -> Int:
        return len(self.op)

    def emit(mut self, op: UInt8, next: Int = -1, arg: Int = 0) raises -> Int:
        if len(self.op) >= MAX_INSTRUCTIONS:
            raise Error(
                "regex is too large: the compiled program exceeds "
                + String(MAX_INSTRUCTIONS)
                + " instructions"
            )
        self.op.append(op)
        self.out.append(Int32(next))
        self.arg.append(Int32(arg))
        return len(self.op) - 1

    @always_inline
    def class_has(self, id: Int, cp: UInt32) -> Bool:
        if cp < 128:
            var word = self.class_ascii[2 * id + Int(cp >> 6)]
            return (word >> UInt64(cp & 63)) & 1 == 1
        return class_contains(self.class_ranges[id], cp)

    @always_inline
    def step(self, pc: Int, cp: UInt32) -> Bool:
        """Whether consuming instruction `pc` accepts `cp`."""
        var op = self.op[pc]
        if op == OP_CHAR:
            return UInt32(self.arg[pc]) == cp
        if op == OP_CLASS:
            return self.class_has(Int(self.arg[pc]), cp)
        if op == OP_ANY:
            return True
        return cp != 10

    @always_inline
    def is_word(self, cp: UInt32) -> Bool:
        if cp < 128:
            return self.word_ascii[Int(cp)]
        return class_contains(self.word, cp)

    @always_inline
    def class_of(self, cp: UInt32) -> Int:
        """The DFA alphabet class of a code point."""
        if cp < 128:
            return Int(self.ascii_class[Int(cp)])
        var lo = 0
        var hi = len(self.boundaries)
        while hi - lo > 1:
            var mid = (lo + hi) // 2
            if self.boundaries[mid] <= cp:
                lo = mid
            else:
                hi = mid
        return Int(self.interval_class[lo])


def _nullable(ast: Ast, node: Int) -> Bool:
    """Whether a node can match the empty string."""
    var kind = ast.kind[node]
    if kind == NODE_LITERAL or kind == NODE_CLASS:
        return False
    if kind == NODE_GROUP:
        return _nullable(ast, ast.kid(node, 0))
    if kind == NODE_CONCAT:
        for k in range(ast.count[node]):
            if not _nullable(ast, ast.kid(node, k)):
                return False
        return True
    if kind == NODE_ALTERNATE:
        for k in range(ast.count[node]):
            if _nullable(ast, ast.kid(node, k)):
                return True
        return False
    if kind == NODE_REPEAT:
        return ast.a[node] == 0 or _nullable(ast, ast.kid(node, 0))
    return True


struct _Frag(Movable):
    """A compiled piece: its entry and the successor fields still to be
    patched (`pc * 2` for `out`, `pc * 2 + 1` for `arg`)."""

    var start: Int
    var holes: List[Int]

    def __init__(out self, start: Int, var holes: List[Int]):
        self.start = start
        self.holes = holes^


struct _Compiler(Movable):
    var program: Program
    var reverse: Bool

    def __init__(out self, reverse: Bool):
        self.program = Program()
        self.reverse = reverse

    def finish(deinit self) -> Program:
        return self.program^

    def patch(mut self, holes: List[Int], target: Int):
        for hole in holes:
            if hole & 1:
                self.program.arg[hole >> 1] = Int32(target)
            else:
                self.program.out[hole >> 1] = Int32(target)

    def class_id(mut self, ranges: List[UInt32]) -> Int:
        for k in range(len(self.program.class_ranges)):
            if self.program.class_ranges[k] == ranges:
                return k
        var low: UInt64 = 0
        var high: UInt64 = 0
        for i in range(0, len(ranges), 2):
            var lo = ranges[i]
            while lo <= ranges[i + 1] and lo < 128:
                if lo < 64:
                    low |= UInt64(1) << UInt64(lo)
                else:
                    high |= UInt64(1) << UInt64(lo - 64)
                lo += 1
        self.program.class_ranges.append(ranges.copy())
        self.program.class_ascii.append(low)
        self.program.class_ascii.append(high)
        return len(self.program.class_ranges) - 1

    def one(mut self, op: UInt8, arg: Int = 0) raises -> _Frag:
        var pc = self.program.emit(op, -1, arg)
        return _Frag(pc, [pc * 2])

    def compile(mut self, ast: Ast, node: Int) raises -> _Frag:
        var kind = ast.kind[node]
        if kind == NODE_EMPTY:
            return self.one(OP_NOP)
        if kind == NODE_LITERAL:
            return self.one(OP_CHAR, ast.a[node])
        if kind == NODE_CLASS:
            ref ranges = ast.classes[ast.a[node]]
            if len(ranges) == 0:
                return _Frag(self.program.emit(OP_FAIL), List[Int]())
            if len(ranges) == 2 and ranges[0] == 0:
                if ranges[1] == MAX_CODEPOINT:
                    return self.one(OP_ANY)
            if (
                len(ranges) == 4
                and ranges[0] == 0
                and ranges[1] == 9
                and ranges[2] == 11
                and ranges[3] == MAX_CODEPOINT
            ):
                return self.one(OP_ANY_NOT_NL)
            if len(ranges) == 2 and ranges[0] == ranges[1]:
                return self.one(OP_CHAR, Int(ranges[0]))
            return self.one(OP_CLASS, self.class_id(ranges))
        if kind == NODE_LOOK:
            self.program.looks |= UInt32(ast.a[node])
            return self.one(OP_LOOK, ast.a[node])
        if kind == NODE_GROUP:
            if self.reverse:
                return self.compile(ast, ast.kid(node, 0))
            var index = ast.a[node]
            var open = self.program.emit(OP_SAVE, -1, 2 * index)
            var body = self.compile(ast, ast.kid(node, 0))
            self.program.out[open] = Int32(body.start)
            var close = self.program.emit(OP_SAVE, -1, 2 * index + 1)
            self.patch(body.holes, close)
            return _Frag(open, [close * 2])
        if kind == NODE_CONCAT:
            var n = ast.count[node]
            var first = self.compile(
                ast, ast.kid(node, n - 1 if self.reverse else 0)
            )
            var start = first.start
            var holes = first.holes.copy()
            for k in range(1, n):
                var next = self.compile(
                    ast, ast.kid(node, n - 1 - k if self.reverse else k)
                )
                self.patch(holes, next.start)
                holes = next.holes.copy()
            return _Frag(start, holes^)
        if kind == NODE_ALTERNATE:
            var n = ast.count[node]
            var holes = List[Int]()
            var start = -1
            var pending = -1
            for k in range(n):
                var entry: Int
                if k < n - 1:
                    entry = self.program.emit(OP_SPLIT)
                    var branch = self.compile(ast, ast.kid(node, k))
                    self.program.out[entry] = Int32(branch.start)
                    holes.extend(branch.holes.copy())
                else:
                    var branch = self.compile(ast, ast.kid(node, k))
                    entry = branch.start
                    holes.extend(branch.holes.copy())
                if pending >= 0:
                    self.program.arg[pending] = Int32(entry)
                else:
                    start = entry
                pending = entry if k < n - 1 else -1
            return _Frag(start, holes^)
        return self.repeat(ast, node)

    def star(mut self, ast: Ast, child: Int, greedy: Bool) raises -> _Frag:
        if _nullable(ast, child):
            # x* as (x+)? when x can match empty, as regex-automata does: a
            # loop straight back would end the empty iteration before its
            # captures are recorded.
            return self.quest(self.plus(ast, child, greedy), greedy)
        var split = self.program.emit(OP_SPLIT)
        var body = self.compile(ast, child)
        self.patch(body.holes, split)
        if greedy:
            self.program.out[split] = Int32(body.start)
            return _Frag(split, [split * 2 + 1])
        self.program.arg[split] = Int32(body.start)
        return _Frag(split, [split * 2])

    def plus(mut self, ast: Ast, child: Int, greedy: Bool) raises -> _Frag:
        var body = self.compile(ast, child)
        var split = self.program.emit(OP_SPLIT)
        self.patch(body.holes, split)
        if greedy:
            self.program.out[split] = Int32(body.start)
            return _Frag(body.start, [split * 2 + 1])
        self.program.arg[split] = Int32(body.start)
        return _Frag(body.start, [split * 2])

    def quest(mut self, var body: _Frag, greedy: Bool) raises -> _Frag:
        var split = self.program.emit(OP_SPLIT)
        var holes = body.holes.copy()
        if greedy:
            self.program.out[split] = Int32(body.start)
            holes.append(split * 2 + 1)
        else:
            self.program.arg[split] = Int32(body.start)
            holes.append(split * 2)
        return _Frag(split, holes^)

    def repeat(mut self, ast: Ast, node: Int) raises -> _Frag:
        var low = ast.a[node]
        var high = ast.b[node]
        var greedy = ast.greedy[node]
        var child = ast.kid(node, 0)
        if high == 0:
            return self.one(OP_NOP)
        if low == 0 and high == -1:
            return self.star(ast, child, greedy)
        if low == 0 and high == 1:
            return self.quest(self.compile(ast, child), greedy)
        if low == 1 and high == -1:
            return self.plus(ast, child, greedy)
        # x{n,m}: n copies, then x* or (x(x(x)?)?)? for the optional rest,
        # nested as RE2 does so failed optional copies stop early.
        var start = -1
        var holes = List[Int]()
        for _ in range(low):
            var copy = self.compile(ast, child)
            if start < 0:
                start = copy.start
            else:
                self.patch(holes, copy.start)
            holes = copy.holes.copy()
        if high == -1:
            var rest = self.star(ast, child, greedy)
            if start < 0:
                return rest^
            self.patch(holes, rest.start)
            return _Frag(start, rest.holes.copy())
        var optional = high - low
        if optional > 0:
            # Build innermost first: x?, then (x x?)?, ...
            var inner = self.quest(self.compile(ast, child), greedy)
            for _ in range(optional - 1):
                var body = self.compile(ast, child)
                self.patch(body.holes, inner.start)
                inner = self.quest(
                    _Frag(body.start, inner.holes.copy()), greedy
                )
            if start < 0:
                return inner^
            self.patch(holes, inner.start)
            holes = inner.holes.copy()
        return _Frag(start, holes^)


def _build_alphabet(mut program: Program):
    """Partition code points into classes no instruction tells apart: cut
    at every range edge, then give intervals that belong to the same sets
    the same class (regex-automata's `ByteClassSet`, over code points)."""
    var sets = List[List[UInt32]]()
    for pc in range(len(program)):
        var op = program.op[pc]
        if op == OP_CHAR:
            var cp = UInt32(program.arg[pc])
            sets.append([cp, cp])
        elif op == OP_CLASS:
            sets.append(program.class_ranges[Int(program.arg[pc])].copy())
    sets.append([10, 10])
    if program.looks & LOOK_UNICODE_WORD:
        sets.append(program.word.copy())
    if program.looks & LOOK_ASCII:
        sets.append(ascii_class("word"))
    var cuts = List[UInt32]()
    cuts.append(0)
    for s in sets:
        for i in range(0, len(s), 2):
            cuts.append(s[i])
            if s[i + 1] < MAX_CODEPOINT:
                cuts.append(s[i + 1] + 1)
    var keys = List[UInt64]()
    for c in cuts:
        keys.append(UInt64(c))
    _sort(keys)
    program.boundaries = List[UInt32]()
    for k in keys:
        var cut = UInt32(k)
        if (
            len(program.boundaries) == 0
            or program.boundaries[len(program.boundaries) - 1] != cut
        ):
            program.boundaries.append(cut)
    var intervals = len(program.boundaries)
    # Membership signature per interval, as the list of sets holding it.
    var signatures = List[String](length=intervals, fill=String())
    for k in range(len(sets)):
        ref s = sets[k]
        for i in range(0, len(s), 2):
            # The interval starting at s[i] and every one up to s[i + 1].
            var lo = 0
            var hi = intervals
            while hi - lo > 1:
                var mid = (lo + hi) // 2
                if program.boundaries[mid] <= s[i]:
                    lo = mid
                else:
                    hi = mid
            var j = lo
            while j < intervals and program.boundaries[j] <= s[i + 1]:
                signatures[j] += String(k) + ","
                j += 1
    var ids = Dict[String, Int]()
    program.interval_class = List[Int32](capacity=intervals)
    program.class_rep = List[UInt32]()
    for j in range(intervals):
        var id = ids.get(signatures[j], -1)
        if id < 0:
            id = len(program.class_rep)
            ids[signatures[j]] = id
            program.class_rep.append(program.boundaries[j])
        program.interval_class.append(Int32(id))
    program.alphabet = len(program.class_rep)
    program.ascii_class = List[Int32](length=128, fill=0)
    for cp in range(128):
        var lo = 0
        var hi = intervals
        while hi - lo > 1:
            var mid = (lo + hi) // 2
            if program.boundaries[mid] <= UInt32(cp):
                lo = mid
            else:
                hi = mid
        program.ascii_class[cp] = program.interval_class[lo]
    program.class_word = List[Bool](capacity=program.alphabet)
    for c in range(program.alphabet):
        program.class_word.append(program.is_word(program.class_rep[c]))


def compile_program(ast: Ast, reverse: Bool) raises -> Program:
    """Compile a syntax tree. A forward program brackets the pattern with
    the whole-match slots and adds the unanchored entry `(?s:.)*?`; a
    reverse program matches the pattern backwards (concatenations
    reversed) and keeps no slots, for finding where a match starts."""
    var compiler = _Compiler(reverse)
    compiler.program.word = word_class()
    for cp in range(128):
        compiler.program.word_ascii[cp] = class_contains(
            compiler.program.word, UInt32(cp)
        )
    compiler.program.slot_count = 2 * (ast.group_count() + 1)
    if reverse:
        var body = compiler.compile(ast, ast.root)
        var done = compiler.program.emit(OP_MATCH)
        compiler.patch(body.holes, done)
        compiler.program.start = body.start
    else:
        var open = compiler.program.emit(OP_SAVE, -1, 0)
        var body = compiler.compile(ast, ast.root)
        compiler.program.out[open] = Int32(body.start)
        var close = compiler.program.emit(OP_SAVE, -1, 1)
        compiler.patch(body.holes, close)
        var done = compiler.program.emit(OP_MATCH)
        compiler.program.out[close] = Int32(done)
        compiler.program.start = open
    # Unanchored entry: prefer starting the match here over skipping a
    # code point.
    var loop = compiler.program.emit(OP_SPLIT, compiler.program.start)
    var skip = compiler.program.emit(OP_ANY, loop)
    compiler.program.arg[loop] = Int32(skip)
    compiler.program.start_unanchored = loop
    var program = compiler^.finish()
    _build_alphabet(program)
    return program^
