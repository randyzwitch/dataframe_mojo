"""Shared strict JSON byte reader and optional flat value tape.

The caller keeps the source bytes alive. Scanning can validate a value without
allocating a tape (metadata), or reuse one tape across column rows. Strings
are validated without constructing per-character or unselected value objects.
"""
from std.memory import Pointer

comptime JsonBytes = Span[UInt8, ImmutAnyOrigin]


@always_inline
def json_bytes(text: String) -> JsonBytes:
    return JsonBytes(
        unsafe_ptr=text.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        length=text.byte_length(),
    )


@always_inline
def _utf8(mut out: List[UInt8], cp: Int):
    if cp < 128:
        out.append(UInt8(cp))
    elif cp < 2048:
        out.append(UInt8(192 | (cp >> 6)))
        out.append(UInt8(128 | (cp & 63)))
    elif cp < 65536:
        out.append(UInt8(224 | (cp >> 12)))
        out.append(UInt8(128 | ((cp >> 6) & 63)))
        out.append(UInt8(128 | (cp & 63)))
    else:
        out.append(UInt8(240 | (cp >> 18)))
        out.append(UInt8(128 | ((cp >> 12) & 63)))
        out.append(UInt8(128 | ((cp >> 6) & 63)))
        out.append(UInt8(128 | (cp & 63)))


@fieldwise_init
struct JsonNode(Copyable, Movable):
    var kind: Int
    var start: Int
    var end: Int
    # First node after this complete subtree; children begin at index + 1.
    var after: Int
    var key_start: Int
    var key_end: Int
    var key_escaped: Bool


struct JsonReader(Movable):
    var address: Int
    var size: Int
    var pos: Int
    var nodes: List[JsonNode]

    def __init__(out self, data: Span[UInt8, _], pos: Int = 0):
        self.address = Int(data.unsafe_ptr())
        self.size = len(data)
        self.pos = pos
        self.nodes = List[JsonNode]()

    @always_inline
    def byte(self, at: Int) -> Int:
        return Int(
            Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=self.address)[
                unsafe_offset=at
            ]
        )

    def space(mut self):
        while self.pos < self.size:
            var b = self.byte(self.pos)
            if b != 32 and b != 9 and b != 10 and b != 13:
                break
            self.pos += 1

    def hex4(mut self) raises -> Int:
        var value = 0
        for _ in range(4):
            if self.pos >= self.size:
                raise Error("Truncated JSON Unicode escape")
            var b = self.byte(self.pos)
            self.pos += 1
            var digit = b - 48 if b >= 48 and b <= 57 else (
                b - 87 if b >= 97
                and b <= 102 else (b - 55 if b >= 65 and b <= 70 else -1)
            )
            if digit < 0:
                raise Error("Invalid JSON Unicode escape")
            value = value * 16 + digit
        return value

    def escaped_codepoint(mut self) raises -> Int:
        var cp = self.hex4()
        if cp >= 0xD800 and cp <= 0xDBFF:
            if (
                self.pos + 2 > self.size
                or self.byte(self.pos) != 92
                or self.byte(self.pos + 1) != 117
            ):
                raise Error("Missing JSON low surrogate")
            self.pos += 2
            var low = self.hex4()
            if low < 0xDC00 or low > 0xDFFF:
                raise Error("Invalid JSON low surrogate")
            return 0x10000 + (cp - 0xD800) * 1024 + low - 0xDC00
        if cp >= 0xDC00 and cp <= 0xDFFF:
            raise Error("Unexpected JSON low surrogate")
        return cp

    def scan_string(mut self) raises -> Bool:
        if self.pos >= self.size or self.byte(self.pos) != 34:
            raise Error("Expected JSON string")
        self.pos += 1
        var escaped = False
        while self.pos < self.size:
            var b = self.byte(self.pos)
            self.pos += 1
            if b == 34:
                return escaped
            if b < 32:
                raise Error("Unescaped control character in JSON string")
            if b == 92:
                escaped = True
                if self.pos >= self.size:
                    break
                b = self.byte(self.pos)
                self.pos += 1
                if b == 117:
                    _ = self.escaped_codepoint()
                elif (
                    b != 34
                    and b != 92
                    and b != 47
                    and b != 98
                    and b != 102
                    and b != 110
                    and b != 114
                    and b != 116
                ):
                    raise Error("Invalid JSON escape")
            elif b >= 128:
                var extra = 1 if b >= 194 and b <= 223 else (
                    2 if b >= 224
                    and b <= 239 else (3 if b >= 240 and b <= 244 else -1)
                )
                if extra < 0 or self.pos + extra > self.size:
                    raise Error("Invalid UTF-8 in JSON string")
                var first = self.byte(self.pos)
                if (
                    (b == 224 and first < 160)
                    or (b == 237 and first >= 160)
                    or (b == 240 and first < 144)
                    or (b == 244 and first >= 144)
                ):
                    raise Error("Invalid UTF-8 in JSON string")
                for _ in range(extra):
                    var next = self.byte(self.pos)
                    self.pos += 1
                    if next < 128 or next > 191:
                        raise Error("Invalid UTF-8 in JSON string")
        raise Error("Unterminated JSON string")

    def scan_number(mut self) raises:
        var start = self.pos
        if self.byte(self.pos) == 45:
            self.pos += 1
        if self.pos >= self.size:
            raise Error("Invalid JSON number")
        var b = self.byte(self.pos)
        if b == 48:
            self.pos += 1
        elif b >= 49 and b <= 57:
            while (
                self.pos < self.size
                and self.byte(self.pos) >= 48
                and self.byte(self.pos) <= 57
            ):
                self.pos += 1
        else:
            raise Error("Invalid JSON value")
        if self.pos < self.size and self.byte(self.pos) == 46:
            self.pos += 1
            var digits = self.pos
            while (
                self.pos < self.size
                and self.byte(self.pos) >= 48
                and self.byte(self.pos) <= 57
            ):
                self.pos += 1
            if self.pos == digits:
                raise Error("Invalid JSON fraction")
        if self.pos < self.size and (
            self.byte(self.pos) == 101 or self.byte(self.pos) == 69
        ):
            self.pos += 1
            if self.pos < self.size and (
                self.byte(self.pos) == 43 or self.byte(self.pos) == 45
            ):
                self.pos += 1
            var digits = self.pos
            while (
                self.pos < self.size
                and self.byte(self.pos) >= 48
                and self.byte(self.pos) <= 57
            ):
                self.pos += 1
            if self.pos == digits:
                raise Error("Invalid JSON exponent")
        if self.pos == start:
            raise Error("Invalid JSON value")

    def scan(
        mut self,
        depth: Int = 0,
        store: Bool = False,
        unique_keys: Bool = True,
        key_start: Int = -1,
        key_end: Int = -1,
        key_escaped: Bool = False,
    ) raises -> Int:
        if depth > 64:
            raise Error("JSON nesting exceeds 64")
        self.space()
        if self.pos >= self.size:
            raise Error("Expected JSON value")
        var start = self.pos
        var b = self.byte(start)
        var index = len(self.nodes)
        if store:
            self.nodes.append(
                JsonNode(
                    b, start, start, index + 1, key_start, key_end, key_escaped
                )
            )
        if b == 34:
            _ = self.scan_string()
        elif b == 123 or b == 91:
            var object = b == 123
            var close = 125 if object else 93
            self.pos += 1
            self.space()
            var keys = Dict[String, Bool]()
            if self.pos < self.size and self.byte(self.pos) == close:
                self.pos += 1
            else:
                while True:
                    var ks = -1
                    var ke = -1
                    var escaped = False
                    if object:
                        ks = self.pos
                        escaped = self.scan_string()
                        ke = self.pos
                        if unique_keys:
                            var name = self.string_at(ks)
                            if name in keys:
                                raise Error("Duplicate JSON key: " + name)
                            keys[name] = True
                        self.space()
                        if self.pos >= self.size or self.byte(self.pos) != 58:
                            raise Error("Expected JSON colon")
                        self.pos += 1
                    _ = self.scan(
                        depth + 1, store, unique_keys, ks, ke, escaped
                    )
                    self.space()
                    if self.pos >= self.size:
                        raise Error("Unterminated JSON container")
                    b = self.byte(self.pos)
                    self.pos += 1
                    if b == close:
                        break
                    if b != 44:
                        raise Error("Expected JSON comma")
                    self.space()
        elif b == 110 or b == 116 or b == 102:
            var literal = StringSlice("null") if b == 110 else (
                StringSlice("true") if b == 116 else StringSlice("false")
            )
            if self.pos + literal.byte_length() > self.size:
                raise Error("Invalid JSON literal")
            for byte in literal.as_bytes():
                if self.byte(self.pos) != Int(byte):
                    raise Error("Invalid JSON literal")
                self.pos += 1
        else:
            self.scan_number()
        if store:
            self.nodes[index].end = self.pos
            self.nodes[index].after = len(self.nodes)
        return index

    def parse(mut self, data: Span[UInt8, _]) raises:
        self.address = Int(data.unsafe_ptr())
        self.size = len(data)
        self.pos = 0
        self.nodes.clear()
        _ = self.scan(store=True, unique_keys=False)
        self.space()
        if self.pos != self.size:
            raise Error("Trailing JSON content")

    @always_inline
    def span(self) -> JsonBytes:
        return JsonBytes(
            unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=self.address
            ),
            length=self.size,
        )

    def slice(self, start: Int, end: Int) -> StringSlice[ImmutAnyOrigin]:
        return StringSlice(
            unsafe_from_utf8=JsonBytes(
                unsafe_ptr=self.span()
                .unsafe_ptr()
                .unsafe_offset(start)
                .unsafe_origin_cast[ImmutAnyOrigin](),
                length=end - start,
            )
        )

    def raw(self, index: Int) -> StringSlice[ImmutAnyOrigin]:
        return self.slice(self.nodes[index].start, self.nodes[index].end)

    def append_string(self, start: Int, mut out: List[UInt8]) raises:
        var reader = JsonReader(self.span(), start + 1)
        while reader.pos < reader.size:
            var b = reader.byte(reader.pos)
            reader.pos += 1
            if b == 34:
                return
            if b != 92:
                out.append(UInt8(b))
                continue
            b = reader.byte(reader.pos)
            reader.pos += 1
            if b == 117:
                _utf8(out, reader.escaped_codepoint())
            else:
                out.append(
                    UInt8(
                        8 if b
                        == 98 else (
                            12 if b
                            == 102 else (
                                10 if b
                                == 110 else (
                                    13 if b == 114 else (9 if b == 116 else b)
                                )
                            )
                        )
                    )
                )
        raise Error("Unterminated JSON string")

    def string_at(self, start: Int) raises -> String:
        var bytes = List[UInt8]()
        self.append_string(start, bytes)
        return String(unsafe_from_utf8=bytes)
