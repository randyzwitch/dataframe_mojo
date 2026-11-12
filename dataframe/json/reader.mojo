"""EmberJSON cursor adapter retaining source spans for typed column decoding.

EmberJSON validates tokens, strings, and container separators. This adapter
indexes their original byte spans so Int128, decimal conversion, duplicate-key
policy, and JSONPath continue to use the exact input spelling. Callers own the
source bytes; the reusable node list never owns or converts scalar values.
"""
from std.memory import Pointer
from std.collections import Set
from emberjson import ParseOptions, is_valid_utf8
from emberjson._deserialize import Parser
from emberjson._deserialize._parser_helper import decode_codepoint
from emberjson._serde.deserializer import EmberJsonCursor
from unicode_mojo.utf8 import unsafe_append_codepoint

comptime JsonBytes = Span[UInt8, ImmutAnyOrigin]
comptime _Options = ParseOptions(max_depth=1025)
comptime _Cursor = EmberJsonCursor[ImmutAnyOrigin, _Options]
comptime _Parser = Parser[ImmutAnyOrigin, _Options]


@always_inline
def json_bytes(text: String) -> JsonBytes:
    return JsonBytes(
        unsafe_ptr=text.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        length=text.byte_length(),
    )


@always_inline
def _utf8(mut out: List[UInt8], cp: Int):
    unsafe_append_codepoint(out, cp)


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
    var max_depth: Int
    var nodes: List[JsonNode]

    def __init__(
        out self, data: Span[UInt8, _], pos: Int = 0, max_depth: Int = 64
    ):
        self.address = Int(data.unsafe_ptr())
        self.size = len(data)
        self.pos = pos
        self.max_depth = max_depth
        self.nodes = List[JsonNode]()

    @always_inline
    def byte(self, at: Int) -> Int:
        return Int(self.span().unsafe_ptr()[unsafe_offset=at])

    def space(mut self) raises:
        var parser = _Parser(self.span()[self.pos :])
        parser.skip_whitespace()
        self.pos = self.size - parser.bytes_remaining()

    def escaped_codepoint(mut self) raises -> Int:
        var pointer = self.span().unsafe_ptr().unsafe_offset(self.pos)
        var end = self.span().unsafe_ptr().unsafe_offset(self.size)
        var value = decode_codepoint(pointer, end)
        self.pos = Int(pointer) - self.address
        return Int(value)

    def scan_number(mut self) raises:
        var parser = _Parser(self.span()[self.pos :])
        _ = parser.expect_float_bytes()
        self.pos += parser.size - parser.bytes_remaining()

    def _walk[
        store: Bool, unique_keys: Bool
    ](
        mut self,
        mut cursor: _Cursor,
        base: Int,
        depth: Int,
        key_start: Int = -1,
        key_end: Int = -1,
        key_escaped: Bool = False,
    ) raises -> Int:
        if depth > self.max_depth:
            raise Error("JSON nesting exceeds " + String(self.max_depth))
        var kind = Int(cursor.peek())
        var start = base + cursor.peek_off()
        var index = len(self.nodes)
        if store:
            self.nodes.append(
                JsonNode(
                    kind,
                    start,
                    start,
                    index + 1,
                    key_start,
                    key_end,
                    key_escaped,
                )
            )
        var end = start
        if kind == 123 or kind == 91:
            var object = kind == 123
            var close = UInt8(125 if object else 93)
            cursor.expect_open(UInt8(kind))
            var first = True
            var keys = Set[String]()
            while True:
                var ks = -1
                var ke = -1
                var escaped = False
                if object:
                    var key = cursor.next_field_key(first)
                    if key[0] < 0:
                        break
                    ks = base + key[0]
                    ke = base + key[1] + 1
                    escaped = not cursor.plain(key[0] + 1, key[1])
                    if unique_keys:
                        var name = cursor.decode_string(key[0], key[1])
                        if name in keys:
                            raise Error("Duplicate JSON key: " + name)
                        keys.add(name^)
                    else:
                        cursor.check_string(key[0], key[1])
                elif not cursor.list_next(first, close):
                    break
                _ = self._walk[store, unique_keys](
                    cursor,
                    base,
                    depth + 1,
                    ks,
                    ke,
                    escaped,
                )
            end = base + cursor.peek_off() + 1
            cursor.expect(close, close)
        elif kind == 34:
            var value = cursor.take_string_span()
            cursor.check_string(value[0], value[1])
            end = base + value[1] + 1
        else:
            cursor.take_scalar()
            if kind == 110:
                cursor.p.expect_null()
            elif kind == 116 or kind == 102:
                _ = cursor.p.expect_bool()
            else:
                _ = cursor.p.expect_float_bytes()
            end = base + cursor.p.size - cursor.p.bytes_remaining()
            cursor.check_token_end()
        if store:
            self.nodes[index].end = end
            self.nodes[index].after = len(self.nodes)
        self.pos = end
        return index

    def _read[
        store: Bool, unique_keys: Bool
    ](mut self, bytes: JsonBytes, base: Int, depth: Int) raises -> Int:
        # The upstream structural index stores byte offsets in UInt32 and
        # rounds its last block to 64 bytes. Reject overflow before reading.
        if len(bytes) > Int(UInt32.MAX) - 64:
            raise Error("JSON value exceeds EmberJSON's 32-bit index")
        if not is_valid_utf8(StringSlice(unsafe_from_utf8=bytes)):
            raise Error("Invalid UTF-8 in JSON string")
        var cursor = _Cursor(bytes)
        var index = self._walk[store, unique_keys](cursor, base, depth)
        if cursor.i != cursor.n:
            raise cursor.trailing_error()
        return index

    def scan(
        mut self,
        depth: Int = 0,
        store: Bool = False,
        unique_keys: Bool = True,
    ) raises -> Int:
        # Frame one value before indexing it. JSON table arrays retain only
        # one row's index, even when the mapped input contains millions of rows.
        self.space()
        var base = self.pos
        var parser = _Parser(self.span()[base:])
        var value = parser.expect_value_bytes()
        if store:
            if unique_keys:
                return self._read[True, True](value, base, depth)
            return self._read[True, False](value, base, depth)
        if unique_keys:
            return self._read[False, True](value, base, depth)
        return self._read[False, False](value, base, depth)

    def parse(
        mut self,
        data: Span[UInt8, _],
        max_depth: Int = 1024,
        *,
        unique_keys: Bool = False,
    ) raises:
        self.address = Int(data.unsafe_ptr())
        self.size = len(data)
        self.pos = 0
        self.max_depth = max_depth
        self.nodes.clear()
        if unique_keys:
            _ = self._read[True, True](self.span(), 0, 0)
        else:
            _ = self._read[True, False](self.span(), 0, 0)
        self.space()

    @always_inline
    def span(self) -> JsonBytes:
        return JsonBytes(
            unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=self.address
            ),
            length=self.size,
        )

    def slice(self, start: Int, end: Int) -> StringSlice[ImmutAnyOrigin]:
        return StringSlice(unsafe_from_utf8=self.span()[start:end])

    def raw(self, index: Int) -> StringSlice[ImmutAnyOrigin]:
        return self.slice(self.nodes[index].start, self.nodes[index].end)

    def append_string(self, start: Int, mut out: List[UInt8]) raises:
        var value = self.string_at(start)
        out.extend(value.as_bytes())

    def string_at(self, start: Int) raises -> String:
        var parser = _Parser(self.span()[start:])
        return parser.expect_string()
