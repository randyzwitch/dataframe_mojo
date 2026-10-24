"""Code point sets for the regex engine.

A set is a flat list of inclusive ranges `lo0, hi0, lo1, hi1, ...`, sorted,
disjoint and not adjacent, as regex-syntax's `ClassUnicode` keeps them; the
set operations below keep that form. Case-insensitive classes are closed
under simple case folding with the orbit table in `unicode_tables`.
"""
from .unicode_tables import FOLD, digit_ranges, space_ranges, word_ranges
from .property_tables import PROPERTY_INDEX, PROPERTY_RANGES

comptime MAX_CODEPOINT: UInt32 = 0x10FFFF


def _sort(mut keys: List[UInt64]):
    """Sort packed ranges and alphabet cuts with the standard sorter."""
    sort(Span(keys))


def canonical(ranges: List[UInt32]) -> List[UInt32]:
    """Sort ranges by start and merge overlapping or adjacent ones."""
    var keys = List[UInt64](capacity=len(ranges) // 2)
    for i in range(0, len(ranges), 2):
        keys.append((UInt64(ranges[i]) << 32) | UInt64(ranges[i + 1]))
    _sort(keys)
    var out = List[UInt32](capacity=len(ranges))
    for key in keys:
        var lo = UInt32(key >> 32)
        var hi = UInt32(key & 0xFFFFFFFF)
        var n = len(out)
        if n > 0 and lo <= out[n - 1] + 1:
            if hi > out[n - 1]:
                out[n - 1] = hi
        else:
            out.append(lo)
            out.append(hi)
    return out^


def negate(ranges: List[UInt32]) -> List[UInt32]:
    var out = List[UInt32]()
    var next: UInt32 = 0
    for i in range(0, len(ranges), 2):
        if ranges[i] > next:
            out.append(next)
            out.append(ranges[i] - 1)
        next = ranges[i + 1] + 1
    if next <= MAX_CODEPOINT:
        out.append(next)
        out.append(MAX_CODEPOINT)
    return out^


def union(a: List[UInt32], b: List[UInt32]) -> List[UInt32]:
    var both = a.copy()
    both.extend(b.copy())
    return canonical(both)


def intersect(a: List[UInt32], b: List[UInt32]) -> List[UInt32]:
    var out = List[UInt32]()
    var i = 0
    var j = 0
    while i < len(a) and j < len(b):
        var lo = max(a[i], b[j])
        var hi = min(a[i + 1], b[j + 1])
        if lo <= hi:
            out.append(lo)
            out.append(hi)
        if a[i + 1] < b[j + 1]:
            i += 2
        else:
            j += 2
    return out^


def difference(a: List[UInt32], b: List[UInt32]) -> List[UInt32]:
    return intersect(a, negate(b))


def symmetric_difference(a: List[UInt32], b: List[UInt32]) -> List[UInt32]:
    return union(difference(a, b), difference(b, a))


@always_inline
def class_contains_bounds(
    ranges: List[UInt32], cp: UInt32, var lo: Int, var hi: Int
) -> Bool:
    """Search a bounded slice of canonical range pairs."""
    var points = ranges.unsafe_ptr()
    while lo < hi:
        # lo and hi index range pairs; their sum is nonnegative and
        # bounded by the Unicode domain. Both endpoints are in bounds.
        var mid = (lo + hi) >> 1
        if cp < points[unsafe_offset=2 * mid]:
            hi = mid
        elif cp > points[unsafe_offset=2 * mid + 1]:
            lo = mid + 1
        else:
            return True
    return False


def class_contains(ranges: List[UInt32], cp: UInt32) -> Bool:
    """Binary search over the ranges."""
    return class_contains_bounds(ranges, cp, 0, len(ranges) >> 1)


def class_size(ranges: List[UInt32]) -> Int:
    var total = 0
    for i in range(0, len(ranges), 2):
        total += Int(ranges[i + 1] - ranges[i]) + 1
    return total


def _hex_digit(byte: UInt8) -> UInt32:
    if byte <= 57:
        return UInt32(byte - 48)
    if byte <= 70:
        return UInt32(byte - 55)
    return UInt32(byte - 87)


def parse_ranges(
    table: StaticString, start: Int = 0, end: Int = -1
) -> List[UInt32]:
    """Read a table of whitespace-separated hex code points and `lo-hi`
    ranges."""
    var out = List[UInt32]()
    var value: UInt32 = 0
    var low: UInt32 = 0
    var in_token = False
    var ranged = False
    var bytes = table.as_bytes()
    for i in range(start, len(bytes) if end < 0 else end):
        var byte = bytes[i]
        if byte == 32 or byte == 10:
            if in_token:
                out.append(low if ranged else value)
                out.append(value)
            value = 0
            in_token = False
            ranged = False
        elif byte == 45:
            low = value
            value = 0
            ranged = True
        else:
            value = value * 16 + _hex_digit(byte)
            in_token = True
    if in_token:
        out.append(low if ranged else value)
        out.append(value)
    return out^


def word_class() -> List[UInt32]:
    """Unicode `\\w`: alphabetic, marks, decimal numbers, connector
    punctuation and join controls."""
    return word_ranges()


def digit_class() -> List[UInt32]:
    """Unicode `\\d`: general category Nd."""
    return digit_ranges()


def space_class() -> List[UInt32]:
    """Unicode `\\s`: the White_Space property."""
    return space_ranges()


def ascii_class(name: String) -> List[UInt32]:
    """`[[:name:]]` ASCII classes; empty for an unknown name."""
    var r: List[UInt32]
    if name == "alnum":
        r = [48, 57, 65, 90, 97, 122]
    elif name == "alpha":
        r = [65, 90, 97, 122]
    elif name == "ascii":
        r = [0, 127]
    elif name == "blank":
        r = [9, 9, 32, 32]
    elif name == "cntrl":
        r = [0, 31, 127, 127]
    elif name == "digit":
        r = [48, 57]
    elif name == "graph":
        r = [33, 126]
    elif name == "lower":
        r = [97, 122]
    elif name == "print":
        r = [32, 126]
    elif name == "punct":
        r = [33, 47, 58, 64, 91, 96, 123, 126]
    elif name == "space":
        r = [9, 13, 32, 32]
    elif name == "upper":
        r = [65, 90]
    elif name == "word":
        r = [48, 57, 65, 90, 95, 95, 97, 122]
    elif name == "xdigit":
        r = [48, 57, 65, 70, 97, 102]
    else:
        r = List[UInt32]()
    return r^


struct CaseFolds(Copyable, Movable):
    """Simple case-folding orbits: `next[k]` is the member after `keys[k]`
    in its orbit, cyclically. Only code points with a folding appear."""

    var keys: List[UInt32]
    var next: List[UInt32]

    def __init__(out self):
        self.keys = List[UInt32]()
        self.next = List[UInt32]()
        var value: UInt32 = 0
        var key: UInt32 = 0
        var in_token = False
        for byte in FOLD.as_bytes():
            if byte == 32 or byte == 10:
                if in_token:
                    self.keys.append(key)
                    self.next.append(value)
                value = 0
                in_token = False
            elif byte == 58:
                key = value
                value = 0
            else:
                value = value * 16 + _hex_digit(byte)
                in_token = True
        if in_token:
            self.keys.append(key)
            self.next.append(value)

    def _lower_bound(self, cp: UInt32) -> Int:
        var lo = 0
        var hi = len(self.keys)
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.keys[mid] < cp:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def fold(self, ranges: List[UInt32]) -> List[UInt32]:
        """The set closed under simple case folding."""
        var out = ranges.copy()
        for i in range(0, len(ranges), 2):
            var k = self._lower_bound(ranges[i])
            while k < len(self.keys) and self.keys[k] <= ranges[i + 1]:
                var member = self.next[k]
                while member != self.keys[k]:
                    out.append(member)
                    out.append(member)
                    member = self.next[self._lower_bound(member)]
                k += 1
        return canonical(out)


def property_class(name: String) raises -> List[UInt32]:
    """Unicode property names and values use loose ASCII matching."""
    var key = String()
    for byte in name.as_bytes():
        if byte == 95 or byte == 45 or byte == 32 or (byte >= 9 and byte <= 13):
            continue
        key += chr(Int(byte + 32 if byte >= 65 and byte <= 90 else byte))
    if key.startswith("is"):
        var trimmed = String(key[byte=2:])
        key = trimmed^
    var table = PROPERTY_INDEX.as_bytes()
    var wanted = key.as_bytes()
    var at = 0
    while at < len(table):
        var equal = True
        var size = 0
        while table[at] != 58:
            if size >= len(wanted):
                equal = False
            elif table[at] != wanted[size]:
                equal = False
            at += 1
            size += 1
        equal = equal and size == len(wanted)
        at += 1
        var start = 0
        while table[at] != 58:
            start = start * 10 + Int(table[at]) - 48
            at += 1
        at += 1
        var end = 0
        while table[at] != 10:
            end = end * 10 + Int(table[at]) - 48
            at += 1
        at += 1
        if equal:
            return parse_ranges(PROPERTY_RANGES, start, end)
    raise Error("Unknown Unicode property: " + name)
