"""Regex syntax: a recursive-descent parser producing a flat syntax tree.

Case-insensitive literals become classes closed under simple case folding,
`.` and the Perl classes are Unicode-aware, and look-around and backreferences
are rejected: the engine has no backtracking, so every search stays linear
in the haystack. Nodes live in parallel lists and refer to their children
by index, so the tree has no recursive types.
"""
from .charclass import (
    CaseFolds,
    MAX_CODEPOINT,
    ascii_class,
    canonical,
    class_contains,
    difference,
    digit_class,
    intersect,
    negate,
    space_class,
    symmetric_difference,
    union,
    word_class,
    property_class,
)

comptime NODE_EMPTY = 0
comptime NODE_LITERAL = 1
comptime NODE_CLASS = 2
comptime NODE_LOOK = 3
comptime NODE_GROUP = 4
comptime NODE_CONCAT = 5
comptime NODE_ALTERNATE = 6
comptime NODE_REPEAT = 7

# Zero-width assertions. The values double as the DFA's context flags.
comptime LOOK_START_TEXT: UInt32 = 1
comptime LOOK_END_TEXT: UInt32 = 2
comptime LOOK_START_LINE: UInt32 = 4
comptime LOOK_END_LINE: UInt32 = 8
comptime LOOK_WORD: UInt32 = 16
comptime LOOK_NOT_WORD: UInt32 = 32
comptime LOOK_WORD_START: UInt32 = 1 << 6
comptime LOOK_WORD_END: UInt32 = 1 << 7
# Bits 8 and 9 are DFA state flags, so additional look flags skip them.
comptime LOOK_ASCII_WORD: UInt32 = 1 << 10
comptime LOOK_ASCII_NOT_WORD: UInt32 = 1 << 11
comptime LOOK_ASCII_WORD_START: UInt32 = 1 << 12
comptime LOOK_ASCII_WORD_END: UInt32 = 1 << 13
comptime LOOK_UNICODE_WORD: UInt32 = LOOK_WORD | LOOK_NOT_WORD | LOOK_WORD_START | LOOK_WORD_END
comptime LOOK_ASCII: UInt32 = LOOK_ASCII_WORD | LOOK_ASCII_NOT_WORD | LOOK_ASCII_WORD_START | LOOK_ASCII_WORD_END
comptime LOOK_MASK: UInt32 = 0x3CFF

comptime _FLAG_CASE = 1
comptime _FLAG_MULTILINE = 2
comptime _FLAG_DOT_NEWLINE = 4
comptime _FLAG_SWAP_GREED = 8
comptime _FLAG_VERBOSE = 16
comptime _FLAG_ASCII = 32

# regex-syntax's default nesting limit.
comptime _NEST_LIMIT = 250


struct Ast(Movable):
    """A parsed pattern. Node `i` has kind `kind[i]`; `a[i]` is a literal's
    code point, a class's index into `classes`, an assertion, a group's
    capture index (-1 for none) or a repetition's minimum; `b[i]` is a
    repetition's maximum (-1 for unbounded) and `greedy[i]` its greed.
    Children are `kids[first[i] : first[i] + count[i]]`."""

    var kind: List[Int]
    var a: List[Int]
    var b: List[Int]
    var greedy: List[Bool]
    var first: List[Int]
    var count: List[Int]
    var kids: List[Int]
    var classes: List[List[UInt32]]
    var names: List[String]
    var root: Int

    def __init__(out self):
        self.kind = List[Int]()
        self.a = List[Int]()
        self.b = List[Int]()
        self.greedy = List[Bool]()
        self.first = List[Int]()
        self.count = List[Int]()
        self.kids = List[Int]()
        self.classes = List[List[UInt32]]()
        # Group 0 is the whole match.
        self.names = [String()]
        self.root = 0

    def add(
        mut self,
        kind: Int,
        a: Int = 0,
        b: Int = 0,
        greedy: Bool = True,
        kids: List[Int] = List[Int](),
    ) -> Int:
        self.kind.append(kind)
        self.a.append(a)
        self.b.append(b)
        self.greedy.append(greedy)
        self.first.append(len(self.kids))
        self.count.append(len(kids))
        self.kids.extend(kids.copy())
        return len(self.kind) - 1

    def kid(self, node: Int, k: Int) -> Int:
        return self.kids[self.first[node] + k]

    def group_count(self) -> Int:
        """Capture groups, not counting the whole match."""
        return len(self.names) - 1


struct _Escape(Movable):
    """What an escape sequence denotes: a code point, a set or an
    assertion."""

    var kind: Int
    var cp: UInt32
    var ranges: List[UInt32]
    var look: UInt32

    def __init__(out self, kind: Int, cp: UInt32 = 0, look: UInt32 = 0):
        self.kind = kind
        self.cp = cp
        self.ranges = List[UInt32]()
        self.look = look


comptime _ESC_LITERAL = 0
comptime _ESC_CLASS = 1
comptime _ESC_LOOK = 2


@always_inline
def _is_digit(byte: Int) -> Bool:
    return byte >= 48 and byte <= 57


@always_inline
def _is_alnum(byte: Int) -> Bool:
    return (
        _is_digit(byte)
        or (byte >= 65 and byte <= 90)
        or (byte >= 97 and byte <= 122)
    )


@always_inline
def _is_space(byte: Int) -> Bool:
    return byte == 32 or (byte >= 9 and byte <= 13)


def _hex_value(byte: Int) -> Int:
    if _is_digit(byte):
        return byte - 48
    if byte >= 65 and byte <= 70:
        return byte - 55
    if byte >= 97 and byte <= 102:
        return byte - 87
    return -1


struct _Parser(Movable):
    var bytes: List[UInt8]
    var pattern: String
    var pos: Int
    var flags: Int
    var depth: Int
    var ast: Ast
    var folds: List[CaseFolds]

    def __init__(out self, pattern: String):
        self.pattern = pattern
        self.bytes = List[UInt8](pattern.as_bytes())
        self.pos = 0
        self.flags = 0
        self.depth = 0
        self.ast = Ast()
        self.folds = List[CaseFolds]()

    def finish(deinit self) -> Ast:
        return self.ast^

    def fail(self, message: String) raises:
        raise Error(
            "regex parse error: "
            + message
            + " (at byte "
            + String(self.pos)
            + " of pattern '"
            + self.pattern
            + "')"
        )

    @always_inline
    def peek(self, ahead: Int = 0) -> Int:
        var at = self.pos + ahead
        if at >= len(self.bytes):
            return -1
        return Int(self.bytes[at])

    def at_end(self) -> Bool:
        return self.pos >= len(self.bytes)

    def next_codepoint(mut self) -> UInt32:
        """Decode the code point at `pos` and step past it. The pattern is
        a String, so valid UTF-8."""
        var b0 = UInt32(self.bytes[self.pos])
        if b0 < 0x80:
            self.pos += 1
            return b0
        var width = 2 if b0 < 0xE0 else (3 if b0 < 0xF0 else 4)
        var mask = UInt32(0x1F) if width == 2 else (
            UInt32(0x0F) if width == 3 else UInt32(0x07)
        )
        var cp = b0 & mask
        for k in range(1, width):
            cp = (cp << 6) | (UInt32(self.bytes[self.pos + k]) & 0x3F)
        self.pos += width
        return cp

    def skip_verbose(mut self):
        """In verbose mode `(?x)`, whitespace and `#` comments are
        ignored."""
        if not (self.flags & _FLAG_VERBOSE):
            return
        while not self.at_end():
            var c = self.peek()
            if _is_space(c):
                self.pos += 1
            elif c == 35:
                while not self.at_end() and self.peek() != 10:
                    self.pos += 1
            else:
                return

    def fold(mut self, ranges: List[UInt32]) -> List[UInt32]:
        """Close a set under simple case folding, loading the orbit table
        on first use."""
        if self.flags & _FLAG_ASCII:
            var out = ranges.copy()
            for cp in range(65, 91):
                if class_contains(ranges, UInt32(cp)):
                    out.extend([UInt32(cp + 32), UInt32(cp + 32)])
                if class_contains(ranges, UInt32(cp + 32)):
                    out.extend([UInt32(cp), UInt32(cp)])
            return canonical(out)
        if len(self.folds) == 0:
            self.folds.append(CaseFolds())
        return self.folds[0].fold(ranges)

    def class_node(mut self, var ranges: List[UInt32]) -> Int:
        self.ast.classes.append(ranges^)
        return self.ast.add(NODE_CLASS, len(self.ast.classes) - 1)

    def literal(mut self, cp: UInt32) -> Int:
        if self.flags & _FLAG_CASE:
            var single: List[UInt32] = [cp, cp]
            var folded = self.fold(single)
            if len(folded) > 2 or folded[0] != folded[1]:
                return self.class_node(folded^)
        return self.ast.add(NODE_LITERAL, Int(cp))

    def parse_alternation(mut self) raises -> Int:
        self.depth += 1
        if self.depth > _NEST_LIMIT:
            self.fail("exceeds the nesting limit of " + String(_NEST_LIMIT))
        var branches = List[Int]()
        branches.append(self.parse_concat())
        while self.peek() == 124:
            self.pos += 1
            branches.append(self.parse_concat())
        self.depth -= 1
        if len(branches) == 1:
            return branches[0]
        return self.ast.add(NODE_ALTERNATE, kids=branches)

    def parse_concat(mut self) raises -> Int:
        var items = List[Int]()
        while True:
            self.skip_verbose()
            var c = self.peek()
            if c < 0 or c == 124 or c == 41:
                break
            var atom = self.parse_atom()
            if atom < 0:
                continue
            items.append(self.parse_repeats(atom))
        if len(items) == 0:
            return self.ast.add(NODE_EMPTY)
        if len(items) == 1:
            return items[0]
        return self.ast.add(NODE_CONCAT, kids=items)

    def parse_atom(mut self) raises -> Int:
        """One atom, or -1 for a flag group like `(?i)` that only changes
        the flags."""
        var c = self.peek()
        if c == 40:
            return self.parse_group()
        if c == 91:
            var ranges = self.parse_class()
            self.check_ascii(ranges)
            return self.class_node(ranges^)
        if c == 46:
            self.pos += 1
            if self.flags & _FLAG_ASCII:
                self.fail("pattern can match invalid UTF-8")
            var any: List[UInt32] = [0, MAX_CODEPOINT]
            if not (self.flags & _FLAG_DOT_NEWLINE):
                any = [0, 9, 11, MAX_CODEPOINT]
            return self.class_node(any^)
        if c == 94:
            self.pos += 1
            var look = (
                LOOK_START_LINE if self.flags
                & _FLAG_MULTILINE else LOOK_START_TEXT
            )
            return self.ast.add(NODE_LOOK, Int(look))
        if c == 36:
            self.pos += 1
            var look = (
                LOOK_END_LINE if self.flags & _FLAG_MULTILINE else LOOK_END_TEXT
            )
            return self.ast.add(NODE_LOOK, Int(look))
        if c == 92:
            var escape = self.parse_escape(False)
            if escape.kind == _ESC_LOOK:
                return self.ast.add(NODE_LOOK, Int(escape.look))
            if escape.kind == _ESC_CLASS:
                self.check_ascii(escape.ranges)
                return self.class_node(escape.ranges.copy())
            return self.literal(escape.cp)
        if c == 42 or c == 43 or c == 63 or c == 123:
            self.fail("repetition operator missing expression")
        return self.literal(self.next_codepoint())

    def skip_brace_space(mut self):
        """Whitespace inside `{m,n}` is allowed even outside verbose
        mode, as in regex-syntax."""
        while _is_space(self.peek()):
            self.pos += 1

    def parse_decimal(mut self) raises -> Int:
        self.skip_brace_space()
        var start = self.pos
        var value = 0
        while _is_digit(self.peek()):
            value = value * 10 + (self.peek() - 48)
            if value > 1_000_000:
                self.fail("repetition count is too large")
            self.pos += 1
        if self.pos == start:
            self.fail("invalid decimal in counted repetition")
        self.skip_brace_space()
        return value

    def parse_repeats(mut self, var atom: Int) raises -> Int:
        while True:
            self.skip_verbose()
            var c = self.peek()
            var low: Int
            var high: Int
            if c == 42:
                low = 0
                high = -1
                self.pos += 1
            elif c == 43:
                low = 1
                high = -1
                self.pos += 1
            elif c == 63:
                low = 0
                high = 1
                self.pos += 1
            elif c == 123:
                self.pos += 1
                low = self.parse_decimal()
                high = low
                if self.peek() == 44:
                    self.pos += 1
                    self.skip_brace_space()
                    if self.peek() == 125:
                        high = -1
                    else:
                        high = self.parse_decimal()
                if self.peek() != 125:
                    self.fail("unclosed counted repetition")
                self.pos += 1
                if high >= 0 and low > high:
                    self.fail("invalid repetition count range")
            else:
                return atom
            var greedy = True
            if self.peek() == 63:
                greedy = False
                self.pos += 1
            if self.flags & _FLAG_SWAP_GREED:
                greedy = not greedy
            atom = self.ast.add(
                NODE_REPEAT, low, high, greedy=greedy, kids=[atom]
            )

    def parse_group(mut self) raises -> Int:
        var open_at = self.pos
        self.pos += 1
        var capture = True
        var name = String()
        var saved = self.flags
        if self.peek() == 63:
            self.pos += 1
            var c = self.peek()
            if c == 61 or c == 33:
                self.fail(
                    "look-around, including look-ahead and look-behind, is"
                    " not supported"
                )
            if c == 60 and (self.peek(1) == 61 or self.peek(1) == 33):
                self.fail(
                    "look-around, including look-ahead and look-behind, is"
                    " not supported"
                )
            if c == 80 and self.peek(1) == 61:
                self.fail("backreferences are not supported")
            if c == 60 or (c == 80 and self.peek(1) == 60):
                self.pos += 1 if c == 60 else 2
                name = self.parse_group_name()
            else:
                capture = False
                if self.parse_flags():
                    # `(?flags)`: applies to the rest of the enclosing
                    # group.
                    return -1
        var index = -1
        if capture:
            index = len(self.ast.names)
            self.ast.names.append(name)
        var child = self.parse_alternation()
        if self.peek() != 41:
            self.pos = open_at
            self.fail("unclosed group")
        self.pos += 1
        self.flags = saved
        if not capture:
            return child
        return self.ast.add(NODE_GROUP, index, kids=[child])

    def parse_group_name(mut self) raises -> String:
        var start = self.pos
        while True:
            var c = self.peek()
            if c < 0:
                self.fail("unclosed capture group name")
            if c == 62:
                break
            var ok = (
                c == 95
                or (c >= 65 and c <= 90)
                or (c >= 97 and c <= 122)
                or (
                    self.pos > start
                    and (_is_digit(c) or c == 46 or c == 91 or c == 93)
                )
            )
            if not ok:
                self.fail("invalid capture group character")
            self.pos += 1
        if self.pos == start:
            self.fail("empty capture group name")
        var name = String(
            StringSlice(unsafe_from_utf8=Span(self.bytes)[start : self.pos])
        )
        self.pos += 1
        for existing in self.ast.names:
            if existing == name:
                self.fail("duplicate capture group name")
        return name^

    def parse_flags(mut self) raises -> Bool:
        """Flags after `(?`: true for `(?flags)`, false for `(?flags:`."""
        var negate = False
        var any = False
        var flags = self.flags
        while True:
            var c = self.peek()
            if c == 41 or c == 58:
                if c == 41 and not any and not negate:
                    self.fail("empty flag group")
                if negate and not any:
                    self.fail("expected flag but got end of flag group")
                self.pos += 1
                self.flags = flags
                return c == 41
            var bit = 0
            if c == 105:
                bit = _FLAG_CASE
            elif c == 109:
                bit = _FLAG_MULTILINE
            elif c == 115:
                bit = _FLAG_DOT_NEWLINE
            elif c == 85:
                bit = _FLAG_SWAP_GREED
            elif c == 120:
                bit = _FLAG_VERBOSE
            elif c == 117:
                # Store the inverse flag so Unicode remains the default.
                if negate:
                    flags |= _FLAG_ASCII
                else:
                    flags &= ~_FLAG_ASCII
                self.pos += 1
                any = True
                continue
            elif c == 45:
                if negate:
                    self.fail("repeated negation in flag group")
                negate = True
                any = False
                self.pos += 1
                continue
            elif c == 82:
                self.fail("CRLF mode (?R) is not supported")
            elif c < 0:
                self.fail("unclosed group")
            else:
                self.fail("unrecognized flag")
            if negate:
                flags &= ~bit
            else:
                flags |= bit
            any = True
            self.pos += 1

    def parse_hex(
        mut self, digits: Int, in_class: Bool = False
    ) raises -> UInt32:
        """`\\x`, `\\u` and `\\U` escapes: a fixed number of hex digits, or
        any number in braces."""
        var value = 0
        var unicode_escape = digits != 2 or self.peek() == 123
        if self.peek() == 123:
            self.pos += 1
            var count = 0
            while self.peek() != 125:
                var d = _hex_value(self.peek())
                if d < 0:
                    self.fail("invalid hexadecimal digit")
                value = value * 16 + d
                if value > 0x10FFFF:
                    self.fail("invalid Unicode scalar value")
                count += 1
                self.pos += 1
            if count == 0:
                self.fail("empty hexadecimal literal")
            self.pos += 1
        else:
            for _ in range(digits):
                var d = _hex_value(self.peek())
                if d < 0:
                    self.fail("invalid hexadecimal digit")
                value = value * 16 + d
                self.pos += 1
        if value > 0x10FFFF or (value >= 0xD800 and value <= 0xDFFF):
            self.fail("invalid Unicode scalar value")
        if self.flags & _FLAG_ASCII and value > 127:
            if not unicode_escape:
                self.fail("pattern can match invalid UTF-8")
            if in_class:
                self.fail("Unicode not allowed here")
        return UInt32(value)

    def parse_escape(mut self, in_class: Bool) raises -> _Escape:
        self.pos += 1
        var c = self.peek()
        if c < 0:
            self.fail("incomplete escape sequence")
        self.pos += 1
        if c == 100 or c == 68 or c == 119 or c == 87 or c == 115 or c == 83:
            var result = _Escape(_ESC_CLASS)
            var lower = c | 32
            var ranges = digit_class() if lower == 100 else (
                word_class() if lower == 119 else space_class()
            )
            if self.flags & _FLAG_ASCII:
                ranges = ascii_class(
                    "digit" if lower
                    == 100 else "word" if lower
                    == 119 else "space"
                )
            result.ranges = negate(ranges) if c < 97 else ranges^
            self.check_ascii(result.ranges)
            return result^
        if c == 98 or c == 66 or c == 65 or c == 122:
            if in_class:
                self.fail("unrecognized escape sequence in a character class")
            var look = LOOK_WORD if c == 98 else (
                LOOK_NOT_WORD if c
                == 66 else (LOOK_START_TEXT if c == 65 else LOOK_END_TEXT)
            )
            if self.flags & _FLAG_ASCII and (c == 98 or c == 66):
                look = LOOK_ASCII_WORD if c == 98 else LOOK_ASCII_NOT_WORD
            return _Escape(_ESC_LOOK, look=look)
        if c == 112 or c == 80:
            if self.flags & _FLAG_ASCII:
                self.fail("Unicode property not allowed with Unicode disabled")
            var inverse = c == 80
            var name: String
            if self.peek() == 123:
                self.pos += 1
                var start = self.pos
                while self.peek() >= 0 and self.peek() != 125:
                    self.pos += 1
                if self.peek() != 125:
                    self.fail("unclosed Unicode property")
                name = String(
                    StringSlice(
                        unsafe_from_utf8=Span(self.bytes)[start : self.pos]
                    )
                )
                self.pos += 1
            else:
                if self.peek() < 0:
                    self.fail("missing Unicode property")
                name = chr(self.peek())
                self.pos += 1
            var unequal = "!=" in name
            if unequal:
                inverse = not inverse
                name = name.replace("!=", "=")
            else:
                name = name.replace(":", "=")
            var ranges = property_class(name)
            if self.flags & _FLAG_CASE:
                ranges = self.fold(ranges)
            var result = _Escape(_ESC_CLASS)
            result.ranges = negate(ranges) if inverse else ranges^
            return result^
        if c == 120:
            return _Escape(_ESC_LITERAL, self.parse_hex(2, in_class))
        if c == 117:
            return _Escape(_ESC_LITERAL, self.parse_hex(4, in_class))
        if c == 85:
            return _Escape(_ESC_LITERAL, self.parse_hex(8, in_class))
        if c == 110:
            return _Escape(_ESC_LITERAL, 10)
        if c == 116:
            return _Escape(_ESC_LITERAL, 9)
        if c == 114:
            return _Escape(_ESC_LITERAL, 13)
        if c == 102:
            return _Escape(_ESC_LITERAL, 12)
        if c == 118:
            return _Escape(_ESC_LITERAL, 11)
        if c == 97:
            return _Escape(_ESC_LITERAL, 7)
        if _is_digit(c):
            self.pos -= 2
            self.fail("backreferences are not supported")
        if c == 60 or c == 62:
            if in_class:
                self.fail("unrecognized escape sequence in a character class")
            var look = LOOK_WORD_START if c == 60 else LOOK_WORD_END
            if self.flags & _FLAG_ASCII:
                look = LOOK_ASCII_WORD_START if c == 60 else LOOK_ASCII_WORD_END
            return _Escape(_ESC_LOOK, look=look)
        if c < 128 and not _is_alnum(c):
            # Any ASCII character other than [0-9A-Za-z<>] may be escaped.
            return _Escape(_ESC_LITERAL, UInt32(c))
        self.pos -= 2
        self.fail("unrecognized escape sequence")
        return _Escape(_ESC_LITERAL)

    def parse_class(mut self) raises -> List[UInt32]:
        """A bracketed class from its `[` through its `]`, with nested
        classes, ranges, `&&`, `--` and `~~`. As in regex-syntax, a
        case-insensitive class is folded before it is negated."""
        var open_at = self.pos
        self.pos += 1
        var negated = False
        if self.peek() == 94:
            negated = True
            self.pos += 1
        var ranges = self.parse_class_union(True, open_at)
        while True:
            var c = self.peek()
            if c == 93:
                break
            var op = c
            self.pos += 2
            var rhs = self.parse_class_union(False, open_at)
            if op == 38:
                ranges = intersect(ranges, rhs)
            elif op == 45:
                ranges = difference(ranges, rhs)
            else:
                ranges = symmetric_difference(ranges, rhs)
        self.pos += 1
        if self.flags & _FLAG_CASE:
            ranges = self.fold(ranges)
        if negated:
            ranges = negate(ranges)
        self.check_ascii(ranges)
        return ranges^

    def check_ascii(self, ranges: List[UInt32]) raises:
        if (
            self.flags & _FLAG_ASCII
            and len(ranges)
            and ranges[len(ranges) - 1] > 127
        ):
            self.fail("pattern can match invalid UTF-8")

    def class_operator_ahead(self) -> Bool:
        var c = self.peek()
        return (c == 38 or c == 45 or c == 126) and self.peek(1) == c

    def skip_class_space(mut self):
        if self.flags & _FLAG_VERBOSE:
            while _is_space(self.peek()):
                self.pos += 1

    def class_item(mut self) raises -> _Escape:
        """One class member: an escape or a literal code point."""
        if self.peek() == 92:
            return self.parse_escape(True)
        var cp = self.next_codepoint()
        if self.flags & _FLAG_ASCII and cp > 127:
            self.fail("Unicode not allowed here")
        return _Escape(_ESC_LITERAL, cp)

    def parse_class_union(
        mut self, first: Bool, open_at: Int
    ) raises -> List[UInt32]:
        var items = List[UInt32]()
        var empty = True
        while True:
            self.skip_class_space()
            var c = self.peek()
            if c < 0:
                self.pos = open_at
                self.fail("unclosed character class")
            if c == 93 and not (first and empty):
                break
            if not empty and self.class_operator_ahead():
                break
            empty = False
            if c == 91:
                if self.peek(1) == 58:
                    var ascii = self.try_ascii_class()
                    if ascii:
                        items.extend(ascii.value().copy())
                        continue
                items.extend(self.parse_class())
                continue
            var item = self.class_item()
            if item.kind == _ESC_CLASS:
                items.extend(item.ranges.copy())
                continue
            var low = item.cp
            var high = low
            self.skip_class_space()
            if (
                self.peek() == 45
                and self.peek(1) != 93
                and self.peek(1) != 45
                and self.peek(1) >= 0
            ):
                self.pos += 1
                self.skip_class_space()
                var end = self.class_item()
                if end.kind != _ESC_LITERAL:
                    self.fail("invalid range boundary in character class")
                high = end.cp
                if high < low:
                    self.fail("invalid range in character class")
            items.append(low)
            items.append(high)
        return canonical(items)

    def try_ascii_class(mut self) raises -> Optional[List[UInt32]]:
        """`[:name:]` or `[:^name:]` at `pos`; None, with `pos` unchanged,
        when the text is not one."""
        var at = self.pos + 2
        var negated = False
        if at < len(self.bytes) and self.bytes[at] == 94:
            negated = True
            at += 1
        var name_start = at
        while at < len(self.bytes) and (
            (self.bytes[at] >= 97 and self.bytes[at] <= 122)
        ):
            at += 1
        if (
            at + 1 >= len(self.bytes)
            or self.bytes[at] != 58
            or self.bytes[at + 1] != 93
        ):
            return None
        var name = String(
            StringSlice(unsafe_from_utf8=Span(self.bytes)[name_start:at])
        )
        var ranges = ascii_class(name)
        if len(ranges) == 0:
            # Not an ASCII class: `[[:foo:]]` is a nested class.
            return None
        self.pos = at + 2
        var result = negate(ranges) if negated else ranges^
        self.check_ascii(result)
        return result^


def parse(pattern: String) raises -> Ast:
    """Parse a pattern in the syntax of Rust's `regex` crate."""
    var parser = _Parser(pattern)
    var root = parser.parse_alternation()
    if not parser.at_end():
        # Only an unmatched `)` stops the top-level alternation early.
        parser.fail("unopened group")
    parser.ast.root = root
    return parser^.finish()
