"""Compiled regular expressions and per-worker matchers.

`Regex` compiles a pattern once into immutable programs shared by every
worker (an `ArcPointer`); `Matcher` is one worker's mutable scratch space:
its two lazy DFAs and Pike VM. A search goes the way Rust's `regex` meta
engine goes:

1. A pattern that is one literal string is a substring search. A single
   character class tests its membership bitmap directly.
2. Otherwise a literal every match must contain rejects most
   non-matching rows with one substring search (RE2's prefilter idea, for
   a single pattern).
3. The forward DFA finds where the leftmost-first match ends, skipping
   input with literal prefixes or a byte-set prefilter while it sits in its
   start state.
4. The reverse DFA, run back from that end, finds where it starts. Anchored
   patterns and patterns with a fixed code-point length need no reverse DFA.
5. Only when capture groups are needed does a capture engine run.
   Eligible deterministic programs use one-pass tables; other programs use
   the bounded backtracker when its visited set fits in 256 KiB, the Pike VM
   otherwise. Anchored one-pass captures skip the DFAs. As in RE2, anchored
   captures also skip the DFAs when the
   backtracker can cover the whole text.

If a DFA gives up (its cache keeps overflowing), the Pike VM answers
instead, so every search stays linear in the haystack.
"""
from std.memory import ArcPointer

from .backtrack import Backtracker
from .onepass import OnePass, ascii_span, compile_onepass
from .charclass import class_size
from .dfa import GAVE_UP, LazyDFA, Prefilter, find_literal
from .pikevm import PikeVM, decode, previous_start
from .program import Program, compile_program
from .syntax import (
    Ast,
    LOOK_START_TEXT,
    NODE_ALTERNATE,
    NODE_CLASS,
    NODE_CONCAT,
    NODE_EMPTY,
    NODE_GROUP,
    NODE_LITERAL,
    NODE_LOOK,
    NODE_REPEAT,
    parse,
)

# Longest literal tracked by the prefilter analysis.
comptime _LITERAL_CAP = 256


def _utf8(cp: Int) -> List[UInt8]:
    var out = List[UInt8]()
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    return out^


@fieldwise_init
struct _Literals(Copyable, Movable):
    """What literal text a node's matches must hold: `text` is the only
    string it matches when `exact`; every match starts with `prefix`, ends
    with `suffix` and contains `required`."""

    var exact: Bool
    var text: List[UInt8]
    var prefix: List[UInt8]
    var suffix: List[UInt8]
    var required: List[UInt8]

    @staticmethod
    def of(var text: List[UInt8]) -> _Literals:
        return _Literals(True, text.copy(), text.copy(), text.copy(), text^)

    @staticmethod
    def nothing() -> _Literals:
        return _Literals(
            False, List[UInt8](), List[UInt8](), List[UInt8](), List[UInt8]()
        )


def _longer(a: List[UInt8], b: List[UInt8]) -> List[UInt8]:
    return b.copy() if len(b) > len(a) else a.copy()


def _cap(mut info: _Literals):
    if info.exact and len(info.text) > _LITERAL_CAP:
        info.exact = False
    if len(info.prefix) > _LITERAL_CAP:
        info.prefix.resize(_LITERAL_CAP, 0)
    if len(info.suffix) > _LITERAL_CAP:
        var keep = List[UInt8]()
        for i in range(len(info.suffix) - _LITERAL_CAP, len(info.suffix)):
            keep.append(info.suffix[i])
        info.suffix = keep^
    if len(info.required) > _LITERAL_CAP:
        info.required.resize(_LITERAL_CAP, 0)


def _concat(a: _Literals, b: _Literals) -> _Literals:
    var out = _Literals.nothing()
    out.exact = a.exact and b.exact
    if out.exact:
        out.text = a.text.copy()
        out.text.extend(b.text.copy())
    if a.exact:
        out.prefix = a.text.copy()
        out.prefix.extend(b.prefix.copy())
    else:
        out.prefix = a.prefix.copy()
    if b.exact:
        out.suffix = a.suffix.copy()
        out.suffix.extend(b.text.copy())
    else:
        out.suffix = b.suffix.copy()
    # A literal can also span the seam: a's suffix then b's prefix.
    var seam = a.suffix.copy()
    seam.extend(b.prefix.copy())
    out.required = _longer(_longer(a.required, b.required), seam)
    out.required = _longer(out.required, out.prefix)
    out.required = _longer(out.required, out.suffix)
    if out.exact:
        out.required = out.text.copy()
    _cap(out)
    return out^


def _analyze(ast: Ast, node: Int) -> _Literals:
    var kind = ast.kind[node]
    if kind == NODE_LITERAL:
        return _Literals.of(_utf8(ast.a[node]))
    if kind == NODE_CLASS:
        ref ranges = ast.classes[ast.a[node]]
        if len(ranges) == 2 and ranges[0] == ranges[1]:
            return _Literals.of(_utf8(Int(ranges[0])))
        return _Literals.nothing()
    if kind == NODE_LOOK or kind == NODE_EMPTY:
        return _Literals.of(List[UInt8]())
    if kind == NODE_GROUP:
        return _analyze(ast, ast.kid(node, 0))
    if kind == NODE_CONCAT:
        var acc = _Literals.of(List[UInt8]())
        for k in range(ast.count[node]):
            acc = _concat(acc, _analyze(ast, ast.kid(node, k)))
        return acc^
    if kind == NODE_ALTERNATE:
        # Only the common prefix and suffix survive an alternation.
        var first = _analyze(ast, ast.kid(node, 0))
        var prefix = (
            first.prefix.copy() if not first.exact else first.text.copy()
        )
        var suffix = (
            first.suffix.copy() if not first.exact else first.text.copy()
        )
        var exact = first.exact
        for k in range(1, ast.count[node]):
            var other = _analyze(ast, ast.kid(node, k))
            var p = other.text.copy() if other.exact else other.prefix.copy()
            var s = other.text.copy() if other.exact else other.suffix.copy()
            if not other.exact or other.text != first.text:
                exact = False
            var keep = 0
            while (
                keep < len(prefix) and keep < len(p) and prefix[keep] == p[keep]
            ):
                keep += 1
            prefix.resize(keep, 0)
            var tail = 0
            while (
                tail < len(suffix)
                and tail < len(s)
                and suffix[len(suffix) - 1 - tail] == s[len(s) - 1 - tail]
            ):
                tail += 1
            var kept = List[UInt8]()
            for i in range(len(suffix) - tail, len(suffix)):
                kept.append(suffix[i])
            suffix = kept^
        if exact:
            return first^
        var required = _longer(prefix, suffix)
        return _Literals(False, List[UInt8](), prefix^, suffix^, required^)
    # NODE_REPEAT
    var low = ast.a[node]
    var high = ast.b[node]
    if high == 0:
        return _Literals.of(List[UInt8]())
    if low == 0:
        return _Literals.nothing()
    var child = _analyze(ast, ast.kid(node, 0))
    if child.exact:
        var text = List[UInt8]()
        for _ in range(low):
            text.extend(child.text.copy())
            if len(text) > _LITERAL_CAP:
                break
        if low == high:
            var out = _Literals.of(text^)
            _cap(out)
            return out^
        var out = _Literals(
            False, List[UInt8](), text.copy(), child.text.copy(), text^
        )
        _cap(out)
        return out^
    return _Literals(
        False,
        List[UInt8](),
        child.prefix.copy(),
        child.suffix.copy(),
        child.required.copy(),
    )


# Bounded literal alternatives. An empty prefix represents an unknown (or
# empty) beginning, so callers must keep the ordinary DFA in that case.
comptime _PREFIX_COUNT = 8


@fieldwise_init
struct _Prefixes(Movable):
    var values: List[List[UInt8]]
    var full: Bool


def _unknown_prefix(full: Bool = False) -> _Prefixes:
    var values = List[List[UInt8]]()
    values.append(List[UInt8]())
    return _Prefixes(values^, full)


def _join_prefixes(left: _Prefixes, right: _Prefixes) -> _Prefixes:
    if not left.full:
        return _Prefixes(left.values.copy(), False)
    if len(left.values) * len(right.values) > _PREFIX_COUNT:
        # Retain all left alternatives; dropping right only broadens them.
        return _Prefixes(left.values.copy(), False)
    var out = List[List[UInt8]]()
    var full = right.full
    for a in left.values:
        for b in right.values:
            var value = a.copy()
            for byte in b:
                if len(value) == _LITERAL_CAP:
                    full = False
                    break
                value.append(byte)
            out.append(value^)
    return _Prefixes(out^, full)


def _literal_prefixes(ast: Ast, node: Int) -> _Prefixes:
    """A covering set of UTF-8 prefixes, never an incomplete set of
    alternatives. At most eight 256-byte strings survive any node."""
    var kind = ast.kind[node]
    if kind == NODE_LITERAL:
        var values = List[List[UInt8]]()
        values.append(_utf8(ast.a[node]))
        return _Prefixes(values^, True)
    if kind == NODE_CLASS:
        ref ranges = ast.classes[ast.a[node]]
        if class_size(ranges) == 1:
            var values = List[List[UInt8]]()
            values.append(_utf8(Int(ranges[0])))
            return _Prefixes(values^, True)
        return _unknown_prefix()
    if kind == NODE_EMPTY or kind == NODE_LOOK:
        return _unknown_prefix(True)
    if kind == NODE_GROUP:
        return _literal_prefixes(ast, ast.kid(node, 0))
    if kind == NODE_CONCAT:
        var out = _unknown_prefix(True)
        for k in range(ast.count[node]):
            var part = _literal_prefixes(ast, ast.kid(node, k))
            out = _join_prefixes(out, part)
            if not out.full:
                break
        return out^
    if kind == NODE_ALTERNATE:
        var values = List[List[UInt8]]()
        var full = True
        for k in range(ast.count[node]):
            var part = _literal_prefixes(ast, ast.kid(node, k))
            if len(values) + len(part.values) > _PREFIX_COUNT:
                return _unknown_prefix()
            full = full and part.full
            for value in part.values:
                values.append(value.copy())
        return _Prefixes(values^, full)
    # Repetitions contribute only their mandatory copies. Optional copies
    # prevent concatenation with following nodes, including lazy repeats.
    var low = ast.a[node]
    var high = ast.b[node]
    if high == 0:
        return _unknown_prefix(True)
    if low == 0:
        return _unknown_prefix()
    var child = _literal_prefixes(ast, ast.kid(node, 0))
    var out = _unknown_prefix(True)
    for _ in range(min(low, _LITERAL_CAP)):
        out = _join_prefixes(out, child)
        if not out.full:
            break
    out.full = out.full and low == high and low <= _LITERAL_CAP
    return out^


# Longest byte-set prefix, and the most bytes one position may hold.
comptime _SET_POSITIONS = 16
comptime _SET_BYTES = 4


@fieldwise_init
struct _Sets(Copyable, Movable):
    """Byte sets every match of a node starts with, position by position;
    `full` when they cover the whole match, so what follows the node
    continues the sequence."""

    var sets: List[List[UInt8]]
    var full: Bool


def _add_byte(mut set: List[UInt8], byte: UInt8):
    for b in set:
        if b == byte:
            return
    set.append(byte)


def _byte_sets(ast: Ast, node: Int) -> _Sets:
    var kind = ast.kind[node]
    if kind == NODE_LITERAL:
        var out = List[List[UInt8]]()
        for b in _utf8(ast.a[node]):
            out.append([b])
        return _Sets(out^, True)
    if kind == NODE_CLASS:
        ref ranges = ast.classes[ast.a[node]]
        if class_size(ranges) > _SET_BYTES or len(ranges) == 0:
            return _Sets(List[List[UInt8]](), False)
        var width = -1
        var out = List[List[UInt8]]()
        for i in range(0, len(ranges), 2):
            for cp in range(Int(ranges[i]), Int(ranges[i + 1]) + 1):
                var bytes = _utf8(cp)
                if width < 0:
                    width = len(bytes)
                    for _ in range(width):
                        out.append(List[UInt8]())
                if len(bytes) != width:
                    return _Sets(List[List[UInt8]](), False)
                for k in range(width):
                    _add_byte(out[k], bytes[k])
        return _Sets(out^, True)
    if kind == NODE_LOOK or kind == NODE_EMPTY:
        return _Sets(List[List[UInt8]](), True)
    if kind == NODE_GROUP:
        return _byte_sets(ast, ast.kid(node, 0))
    if kind == NODE_CONCAT:
        var out = List[List[UInt8]]()
        for k in range(ast.count[node]):
            var part = _byte_sets(ast, ast.kid(node, k))
            for s in part.sets:
                if len(out) < _SET_POSITIONS:
                    out.append(s.copy())
            if not part.full or len(out) >= _SET_POSITIONS:
                return _Sets(out^, False)
        return _Sets(out^, True)
    if kind == NODE_ALTERNATE:
        var first = _byte_sets(ast, ast.kid(node, 0))
        var out = first.sets.copy()
        var full = first.full
        for k in range(1, ast.count[node]):
            var other = _byte_sets(ast, ast.kid(node, k))
            if len(other.sets) != len(out):
                full = False
            full = full and other.full
            var keep = min(len(out), len(other.sets))
            while len(out) > keep:
                _ = out.pop()
            for i in range(keep):
                for b in other.sets[i]:
                    _add_byte(out[i], b)
        # Positions past one holding too many bytes say nothing useful.
        for i in range(len(out)):
            if len(out[i]) > _SET_BYTES:
                while len(out) > i:
                    _ = out.pop()
                return _Sets(out^, False)
        return _Sets(out^, full)
    # NODE_REPEAT
    var low = ast.a[node]
    var high = ast.b[node]
    if high == 0:
        return _Sets(List[List[UInt8]](), True)
    if low == 0:
        return _Sets(List[List[UInt8]](), False)
    var child = _byte_sets(ast, ast.kid(node, 0))
    if not child.full:
        return _Sets(child.sets.copy(), False)
    var out = List[List[UInt8]]()
    for _ in range(low):
        for s in child.sets:
            if len(out) < _SET_POSITIONS:
                out.append(s.copy())
    return _Sets(out^, low == high and len(out) < _SET_POSITIONS)


def _flat_sets(sets: _Sets) -> List[UInt8]:
    """Four bytes per position, a smaller set repeating its first byte;
    empty when the sequence is empty or the pattern can match empty."""
    var out = List[UInt8]()
    for s in sets.sets:
        for k in range(4):
            out.append(s[k] if k < len(s) else s[0])
    return out^


def _fixed_width(ast: Ast, node: Int) -> Int:
    """The number of code points in every match, or -1 when it varies."""
    var kind = ast.kind[node]
    if kind == NODE_LITERAL or kind == NODE_CLASS:
        return 1
    if kind == NODE_LOOK or kind == NODE_EMPTY:
        return 0
    if kind == NODE_GROUP:
        return _fixed_width(ast, ast.kid(node, 0))
    if kind == NODE_CONCAT:
        var total = 0
        for k in range(ast.count[node]):
            var width = _fixed_width(ast, ast.kid(node, k))
            if width < 0:
                return -1
            total += width
        return total
    if kind == NODE_ALTERNATE:
        var width = _fixed_width(ast, ast.kid(node, 0))
        for k in range(1, ast.count[node]):
            if _fixed_width(ast, ast.kid(node, k)) != width:
                return -1
        return width
    if ast.b[node] == 0:
        return 0
    var width = _fixed_width(ast, ast.kid(node, 0))
    if width == 0:
        return 0
    if width < 0 or ast.a[node] != ast.b[node]:
        return -1
    return width * ast.a[node]


def _anchored(ast: Ast, node: Int) -> Bool:
    """Whether every match starts at the start of the text."""
    var kind = ast.kind[node]
    if kind == NODE_LOOK:
        return UInt32(ast.a[node]) == LOOK_START_TEXT
    if kind == NODE_GROUP or kind == NODE_CONCAT:
        return _anchored(ast, ast.kid(node, 0))
    if kind == NODE_ALTERNATE:
        for k in range(ast.count[node]):
            if not _anchored(ast, ast.kid(node, k)):
                return False
        return True
    if kind == NODE_REPEAT:
        return ast.a[node] >= 1 and _anchored(ast, ast.kid(node, 0))
    return False


struct _Compiled(Movable):
    var pattern: String
    var forward: Program
    var reverse: Program
    var names: List[String]
    var anchored: Bool
    var literal: Bool
    var single_class: Bool
    var fixed_width: Int32
    var text: List[UInt8]
    var required: List[UInt8]
    var prefilter: Prefilter
    var onepass: Optional[ArcPointer[OnePass]]

    def __init__(out self, pattern: String) raises:
        var ast = parse(pattern)
        self.pattern = pattern
        self.forward = compile_program(ast, False)
        self.reverse = compile_program(ast, True)
        self.names = ast.names.copy()
        self.anchored = _anchored(ast, ast.root)
        var root = ast.root
        while ast.kind[root] == NODE_GROUP:
            root = ast.kid(root, 0)
        self.single_class = (
            ast.kind[root] == NODE_CLASS and len(self.forward.class_ranges) == 1
        )
        self.fixed_width = Int32(_fixed_width(ast, ast.root))
        var info = _analyze(ast, ast.root)
        # A pattern with assertions is not a plain substring search even
        # when it matches only one string.
        self.literal = info.exact and self.forward.looks == 0
        self.text = info.text.copy()
        self.required = info.required.copy()
        var sets = _flat_sets(_byte_sets(ast, ast.root))
        var prefixes = List[List[UInt8]]()
        if not self.anchored and len(info.prefix) == 0 and len(sets) == 0:
            var extracted = _literal_prefixes(ast, ast.root)
            prefixes = extracted.values.copy()
        self.prefilter = Prefilter(info.prefix.copy(), sets^, prefixes^)
        # Capture tables are cold for ordinary searches. Keep their
        # vectors behind a pointer instead of widening the hot program object.
        var onepass = compile_onepass(self.forward)
        self.onepass = None
        if onepass.active():
            self.onepass = ArcPointer(onepass^)


@no_inline
def _find_class(
    program: Program, text: Span[UInt8, ImmutAnyOrigin], at: Int
) -> Tuple[Int, Int]:
    """A whole pattern consisting of one class, with optional captures.
    Its only consuming instruction uses class zero; no assertion or
    alternative can change where that code point matches."""
    var bytes = text.unsafe_ptr()
    var membership = program.class_ascii.unsafe_ptr()
    var i = at
    while i < len(text):
        var b = bytes[unsafe_offset=i]
        if b < 128:
            var word = membership[unsafe_offset=Int(b >> 6)]
            if (word >> UInt64(b & 63)) & 1 != 0:
                return (i, i + 1)
            i += 1
        else:
            var packed = decode(text, i)
            var width = packed & 7
            if program.class_has(0, UInt32(packed >> 3)):
                return (i, i + width)
            i += width
    return (-1, -1)


def _may_match(c: _Compiled, text: Span[UInt8, ImmutAnyOrigin]) -> Bool:
    """False when a match is impossible: an anchored pattern whose prefix
    does not start the text, or a required literal the text lacks."""
    ref prefix = c.prefilter.prefix
    if c.anchored and len(prefix) > 0:
        if len(text) < len(prefix):
            return False
        var p = text.unsafe_ptr()
        for i in range(len(prefix)):
            if p[unsafe_offset=i] != prefix[i]:
                return False
        return (
            len(c.required) <= len(prefix)
            or find_literal(text, c.required, 0) >= 0
        )
    return len(c.required) == 0 or find_literal(text, c.required, 0) >= 0


struct Regex(Copyable, Movable):
    """A compiled regular expression. Cheap to copy: the compiled programs
    are shared and never change, so any number of workers can match with one
    `Regex`, each through its own `Matcher`.

    Supported: literals and escapes (`\\n`, `\\x7F`, `\\u{1F600}`, escaped
    punctuation), `.`, classes with ranges, negation, nesting, `&&`, `--`,
    `~~` and `[[:alpha:]]`-style ASCII classes, Unicode-aware `\\d \\w \\s`
    and their negations, the anchors `^ $ \\A \\z \\b \\B`, alternation,
    capturing, named (`(?P<name>...)`, `(?<name>...)`) and non-capturing
    groups, the repetitions `* + ? {m} {m,} {m,n}` and their lazy forms,
    and the flags `i m s U x` (as `(?flags)` or `(?flags:...)`).

    Rejected with an error, as Rust does: look-around and backreferences.
    Unicode properties (`\\p{Greek}`, `\\pL`, `\\P{...}`) and word-start/end
    anchors (`\\<`, `\\>`) are supported. `(?-u)` selects ASCII classes,
    boundaries and case folding, provided the pattern cannot match invalid
    UTF-8, as in Rust's string regex. CRLF mode `(?R)` is not supported.
    """

    var _compiled: ArcPointer[_Compiled]

    def __init__(out self, pattern: String) raises:
        """Compile `pattern`; raises on invalid syntax or an unsupported
        feature."""
        self._compiled = ArcPointer(_Compiled(pattern))

    def pattern(self) -> String:
        return self._compiled[].pattern

    def group_count(self) -> Int:
        """Capture groups, not counting the whole match (group 0)."""
        return len(self._compiled[].names) - 1

    def group_names(self) -> List[String]:
        """Capture field names in group order, using the index when unnamed."""
        var out = List[String]()
        ref names = self._compiled[].names
        for i in range(1, len(names)):
            out.append(names[i] if names[i] != "" else String(i))
        return out^

    def group_index(self, name: String) -> Int:
        """The index of a named group, or -1."""
        ref names = self._compiled[].names
        for i in range(1, len(names)):
            if names[i] == name:
                return i
        return -1

    def literal(self) -> Optional[String]:
        """The one string the pattern matches, when it is a plain literal
        (no assertions or case-insensitivity): searches are then substring
        searches."""
        ref c = self._compiled[]
        if not c.literal:
            return None
        return String(StringSlice(unsafe_from_utf8=Span(c.text)))

    def template(self, replacement: String, expand: Bool = True) -> Template:
        """A replacement string, with `$1`, `${1}`, `$name` and `${name}`
        group references (and `$$` for a dollar sign) when `expand`."""
        return Template(self, replacement, expand)


struct Template(Copyable, Movable):
    """A parsed replacement: literal bytes and group references, following
    `regex::Captures::expand`. `$name` takes the longest run of
    `[0-9A-Za-z_]`; an all-digit name is a group number. A reference to a
    group that does not exist or did not take part in the match inserts
    nothing; a `$` that starts no reference is literal."""

    var bytes: List[UInt8]
    # Pieces as (start, end) into `bytes`, or (-1, group) for a group.
    var pieces: List[Int]
    var needs_groups: Bool

    def __init__(out self, regex: Regex, replacement: String, expand: Bool):
        self.bytes = List[UInt8](replacement.as_bytes())
        self.pieces = List[Int]()
        self.needs_groups = False
        var n = len(self.bytes)
        if not expand:
            self.pieces.append(0)
            self.pieces.append(n)
            return
        var literal_start = 0
        var i = 0
        while i < n:
            if self.bytes[i] != 36 or i + 1 >= n:
                i += 1
                continue
            if self.bytes[i + 1] == 36:
                self._literal(literal_start, i + 1)
                i += 2
                literal_start = i
                continue
            var name_start: Int
            var name_end: Int
            var after: Int
            if self.bytes[i + 1] == 123:
                name_start = i + 2
                var j = name_start
                while j < n and self.bytes[j] != 125:
                    j += 1
                if j >= n:
                    i += 1
                    continue
                name_end = j
                after = j + 1
            else:
                name_start = i + 1
                var j = name_start
                while j < n and _is_name_byte(self.bytes[j]):
                    j += 1
                if j == name_start:
                    i += 1
                    continue
                name_end = j
                after = j
            self._literal(literal_start, i)
            var group = _resolve(regex, self.bytes, name_start, name_end)
            self.pieces.append(-1)
            self.pieces.append(group)
            if group > 0:
                self.needs_groups = True
            i = after
            literal_start = i
        self._literal(literal_start, n)

    def _literal(mut self, start: Int, end: Int):
        if end > start:
            self.pieces.append(start)
            self.pieces.append(end)


@always_inline
def _is_name_byte(b: UInt8) -> Bool:
    return (
        (b >= 48 and b <= 57)
        or (b >= 65 and b <= 90)
        or (b >= 97 and b <= 122)
        or b == 95
    )


def _resolve(regex: Regex, bytes: List[UInt8], start: Int, end: Int) -> Int:
    """A reference's group index, or -1 when there is no such group."""
    var number = 0
    var digits = end > start
    for i in range(start, end):
        var b = bytes[i]
        if b < 48 or b > 57:
            digits = False
            break
        number = number * 10 + Int(b - 48)
        if number > 1_000_000:
            return -1
    if digits:
        return number if number <= regex.group_count() else -1
    var name = String(StringSlice(unsafe_from_utf8=Span(bytes)[start:end]))
    return regex.group_index(name)


struct Matcher(Movable):
    """One worker's searcher for a `Regex`: the caches of its forward and
    reverse DFAs, a Pike VM, and capture slots. Not shared between
    threads."""

    var regex: Regex
    var forward: LazyDFA
    var reverse: LazyDFA
    var pike: PikeVM
    var backtracker: Backtracker
    var slots: List[Int]
    var _position: Int
    var _last_end: Int

    def __init__(out self, regex: Regex):
        self.regex = regex.copy()
        ref c = regex._compiled[]
        self.forward = LazyDFA(c.forward, False)
        self.forward.accel_enabled = c.prefilter.active() and not c.anchored
        self.reverse = LazyDFA(c.reverse, True)
        self.pike = PikeVM(c.forward, c.forward.slot_count)
        self.backtracker = Backtracker()
        self.slots = List[Int](length=c.forward.slot_count, fill=-1)
        self._position = 0
        self._last_end = -1

    def is_match(mut self, text: Span[UInt8, ImmutAnyOrigin]) -> Bool:
        """Whether the pattern matches anywhere in `text`."""
        ref c = self.regex._compiled[]
        if c.literal:
            return find_literal(text, c.text, 0) >= 0
        if c.single_class:
            return _find_class(c.forward, text, 0)[0] >= 0
        if not _may_match(c, text):
            return False
        var end = self.forward.find_forward(
            c.forward, c.prefilter, text, 0, c.anchored, True
        )
        if end == GAVE_UP:
            return self.pike.search(
                c.forward, text, 0, len(text), c.anchored, self.slots
            )
        return end >= 0

    def find(
        mut self, text: Span[UInt8, ImmutAnyOrigin], start: Int = 0
    ) -> Tuple[Int, Int]:
        """The leftmost-first match starting at or after `start`, as byte
        offsets (start, end), or (-1, -1). Assertions see the whole text."""
        ref c = self.regex._compiled[]
        if start > len(text):
            return (-1, -1)
        if c.literal:
            var at = find_literal(text, c.text, start)
            if at < 0:
                return (-1, -1)
            return (at, at + len(c.text))
        if c.single_class:
            return _find_class(c.forward, text, start)
        if c.anchored and start > 0:
            # Every match starts at the start of the text.
            return (-1, -1)
        if start == 0 and not _may_match(c, text):
            return (-1, -1)
        var end = self.forward.find_forward(
            c.forward, c.prefilter, text, start, c.anchored, False
        )
        if end == -1:
            return (-1, -1)
        if end != GAVE_UP and c.anchored:
            # The anchored forward search already knows where the match
            # starts; RE2 runs the reverse DFA only for unanchored ones.
            return (start, end)
        if end != GAVE_UP and c.fixed_width >= 0:
            # The forward match's end and its code-point count determine
            # the start. Walk UTF-8 boundaries instead of replaying the NFA.
            var begin = end
            for _ in range(Int(c.fixed_width)):
                begin = previous_start(text, begin)
            return (begin, end)
        if end != GAVE_UP:
            var begin = self.reverse.find_reverse(c.reverse, text, end, start)
            if begin >= 0:
                return (begin, end)
        if not self.pike.search(
            c.forward, text, start, len(text), c.anchored, self.slots
        ):
            return (-1, -1)
        return (self.slots[0], self.slots[1])

    def captures(
        mut self, text: Span[UInt8, ImmutAnyOrigin], start: Int = 0
    ) -> Bool:
        """Find the leftmost-first match at or after `start` and fill
        `slots`: group g spans `slots[2g] : slots[2g + 1]`, -1 when it took
        no part."""
        ref c = self.regex._compiled[]
        if c.anchored and start == 0 and Bool(c.onepass):
            # The start state's ASCII table can reject impossible first
            # bytes before scanning a whole row for ASCII eligibility.
            if (
                len(text) > 0
                and text[0] < 128
                and not c.onepass.value()[].matched[0]
            ):
                var cls = Int(c.forward.ascii_class[Int(text[0])])
                if c.onepass.value()[].next[cls] < 0:
                    return False
            if not _may_match(c, text):
                return False
            if ascii_span(text, 0, len(text)):
                # The capture engines run sequentially and initialize their
                # scratch before use, so they can share this worker's slots.
                return c.onepass.value()[].search[False](
                    c.forward,
                    text,
                    0,
                    len(text),
                    self.pike.scratch,
                    self.slots,
                )
            return c.onepass.value()[].search[True](
                c.forward,
                text,
                0,
                len(text),
                self.pike.scratch,
                self.slots,
            )
        if self._direct_captures(text):
            return start == 0 and self._anchored_captures(text)
        var m = self.find(text, start)
        if m[0] < 0:
            return False
        return self._groups(text, m[0], m[1])

    def _direct_captures(self, text: Span[UInt8, ImmutAnyOrigin]) -> Bool:
        """Whether a capture search should skip the DFAs: RE2's `Match`
        sends an anchored pattern with groups over a text the bounded
        backtracker covers straight to it, since the DFA pass would only
        find the span the backtracker then walks again."""
        ref c = self.regex._compiled[]
        return (
            c.anchored
            and len(self.slots) > 2
            and Backtracker.fits(c.forward, 0, len(text))
        )

    def _anchored_captures(mut self, text: Span[UInt8, ImmutAnyOrigin]) -> Bool:
        """The match of an anchored pattern, with its groups in `slots`,
        found by the backtracker alone."""
        ref c = self.regex._compiled[]
        if not _may_match(c, text):
            return False
        return self.backtracker.search(
            c.forward, text, 0, len(text), self.slots
        )

    def _groups(
        mut self, text: Span[UInt8, ImmutAnyOrigin], begin: Int, end: Int
    ) -> Bool:
        """Capture positions for the match known to span [begin, end)."""
        ref c = self.regex._compiled[]
        if len(self.slots) == 2:
            self.slots[0] = begin
            self.slots[1] = end
            return True
        if Bool(c.onepass):
            if ascii_span(text, begin, end):
                return c.onepass.value()[].search[False](
                    c.forward, text, begin, end, self.pike.scratch, self.slots
                )
            return c.onepass.value()[].search[True](
                c.forward, text, begin, end, self.pike.scratch, self.slots
            )
        if Backtracker.fits(c.forward, begin, end):
            return self.backtracker.search(
                c.forward, text, begin, end, self.slots
            )
        return self.pike.search(c.forward, text, begin, end, True, self.slots)

    def reset(mut self):
        """Start iterating matches from the beginning of a text."""
        self._position = 0
        self._last_end = -1

    def next(mut self, text: Span[UInt8, ImmutAnyOrigin]) -> Tuple[Int, Int]:
        """The next non-overlapping match after `reset`, or (-1, -1), with
        Rust's rule for empty matches: an empty match that ends where the
        previous match ended is skipped by searching again one code point
        later."""
        var n = len(text)
        if self._position > n:
            return (-1, -1)
        var m = self.find(text, self._position)
        if m[0] < 0:
            self._position = n + 1
            return (-1, -1)
        if m[0] == m[1] and m[1] == self._last_end:
            if self._position >= n:
                self._position = n + 1
                return (-1, -1)
            var width = decode(text, self._position) & 7
            m = self.find(text, self._position + width)
            if m[0] < 0:
                self._position = n + 1
                return (-1, -1)
        self._position = m[1]
        self._last_end = m[1]
        return m

    def count(mut self, text: Span[UInt8, ImmutAnyOrigin]) -> Int:
        """Non-overlapping matches, as `regex::Regex::find_iter` counts
        them."""
        self.reset()
        var total = 0
        while self.next(text)[0] >= 0:
            total += 1
        return total

    def replace(
        mut self,
        text: Span[UInt8, ImmutAnyOrigin],
        template: Template,
        limit: Int,
        mut out: List[UInt8],
    ) -> Bool:
        """Append `text` to `out` with up to `limit` matches (all when
        negative) replaced by `template`; returns whether anything
        matched."""
        self.reset()
        var copied = 0
        var done = 0
        var p = text.unsafe_ptr()
        if template.needs_groups and self._direct_captures(text):
            # An anchored pattern matches at most once, at the start.
            if limit != 0 and self._anchored_captures(text):
                self.expand(text, template, out)
                copied = self.slots[1]
                done = 1
            for i in range(copied, len(text)):
                out.append(p[unsafe_offset=i])
            return done > 0
        while limit < 0 or done < limit:
            var m = self.next(text)
            if m[0] < 0:
                break
            for i in range(copied, m[0]):
                out.append(p[unsafe_offset=i])
            if template.needs_groups:
                _ = self._groups(text, m[0], m[1])
            else:
                self.slots[0] = m[0]
                self.slots[1] = m[1]
            self.expand(text, template, out)
            copied = m[1]
            done += 1
        for i in range(copied, len(text)):
            out.append(p[unsafe_offset=i])
        return done > 0

    def expand(
        self,
        text: Span[UInt8, ImmutAnyOrigin],
        template: Template,
        mut out: List[UInt8],
    ):
        """Append the template with the groups in `slots`."""
        var p = text.unsafe_ptr()
        for k in range(0, len(template.pieces), 2):
            var a = template.pieces[k]
            var b = template.pieces[k + 1]
            if a >= 0:
                for i in range(a, b):
                    out.append(template.bytes[i])
            elif b >= 0 and 2 * b + 1 < len(self.slots):
                var s = self.slots[2 * b]
                var e = self.slots[2 * b + 1]
                if s >= 0 and e >= s:
                    for i in range(s, e):
                        out.append(p[unsafe_offset=i])


def escape(text: String) -> String:
    """Escape every regex metacharacter (`\\.+*?()|[]{}^$#&-~`), so the
    result matches `text` literally."""
    var out = String()
    for cp in text.codepoints():
        var c = Int(cp)
        if (
            c == 92
            or c == 46
            or c == 43
            or c == 42
            or c == 63
            or c == 40
            or c == 41
            or c == 124
            or c == 91
            or c == 93
            or c == 123
            or c == 125
            or c == 94
            or c == 36
            or c == 35
            or c == 38
            or c == 45
            or c == 126
        ):
            out += "\\"
        out += String(cp)
    return out^
