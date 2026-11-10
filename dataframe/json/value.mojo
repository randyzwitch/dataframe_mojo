"""Small strict JSON reader for geospatial metadata and GeoJSON.

Values own their source text. Nesting is bounded and duplicate object keys
are rejected, so metadata cannot have ambiguous interpretations.
"""

from dataframe.json.reader import JsonNode, JsonReader, json_bytes
from std.memory import ArcPointer


def json_quote(text: String) -> String:
    var out = String('"')
    for c in text.codepoints():
        var s = String(c)
        if s == '"' or s == "\\":
            out += "\\" + s
        elif s == "\n":
            out += "\\n"
        elif s == "\r":
            out += "\\r"
        elif s == "\t":
            out += "\\t"
        elif s == "\b":
            out += "\\b"
        elif s == "\f":
            out += "\\f"
        elif s.as_bytes()[0] < 32:
            var b = Int(s.as_bytes()[0])
            var hex = String("0123456789abcdef")
            out += (
                "\\u00" + String(hex[byte=b // 16]) + String(hex[byte=b % 16])
            )
        else:
            out += s
    return out + '"'


def _space(text: String, mut pos: Int):
    var reader = JsonReader(json_bytes(text), pos)
    reader.space()
    pos = reader.pos


def _hex4(text: String, mut pos: Int) raises -> Int:
    var reader = JsonReader(json_bytes(text), pos)
    var value = reader.hex4()
    pos = reader.pos
    return value


def _string(text: String, mut pos: Int) raises -> String:
    var reader = JsonReader(json_bytes(text), pos)
    _ = reader.scan_string()
    var value = reader.string_at(pos)
    pos = reader.pos
    return value^


def _scan(text: String, mut pos: Int, depth: Int = 0) raises:
    var reader = JsonReader(json_bytes(text), pos)
    _ = reader.scan(depth)
    pos = reader.pos


struct _JsonObject(Copyable, Movable):
    var keys: List[String]
    var values: Dict[String, Int]

    def __init__(out self):
        self.keys = List[String]()
        self.values = Dict[String, Int]()


struct _JsonIndex(Movable):
    """One validated value tape and decoded field index shared by its views."""

    var nodes: List[JsonNode]
    var objects: Dict[Int, _JsonObject]

    def __init__(out self, text: String) raises:
        var reader = JsonReader(json_bytes(text))
        _ = reader.scan(store=True)
        reader.space()
        if reader.pos != text.byte_length():
            raise Error("Trailing JSON content")
        self.objects = Dict[Int, _JsonObject]()
        for index in range(len(reader.nodes)):
            if reader.nodes[index].kind != 123:
                continue
            var object = _JsonObject()
            var child = index + 1
            while child < reader.nodes[index].after:
                var key = reader.string_at(reader.nodes[child].key_start)
                object.values[key] = child
                object.keys.append(key^)
                child = reader.nodes[child].after
            self.objects[index] = object^
        self.nodes = List[JsonNode]()
        swap(self.nodes, reader.nodes)


struct JsonValue(Copyable, Movable):
    """Validated JSON with indexed fields and owned source text.

    Construction validates the complete document once. Child values share its
    immutable index and own only their source substring, so they outlive their
    parent without rescanning containers or retaining its complete source text.
    """

    var text: String
    var _index: ArcPointer[_JsonIndex]
    var _node: Int

    def __init__(out self, text: String) raises:
        self._index = ArcPointer(_JsonIndex(text))
        ref root = self._index[].nodes[0]
        self.text = String(text[byte = root.start : root.end])
        self._node = 0

    def __init__(
        out self, var text: String, index: ArcPointer[_JsonIndex], node: Int
    ):
        self.text = text^
        self._index = index
        self._node = node

    def kind(self) -> UInt8:
        return UInt8(self._index[].nodes[self._node].kind)

    def string(self) raises -> String:
        if self.kind() != 34:
            raise Error("Expected JSON string")
        return JsonReader(json_bytes(self.text)).string_at(0)

    def keys(self) raises -> List[String]:
        if self.kind() != 123:
            raise Error("Expected JSON object")
        return self._index[].objects[self._node].keys.copy()

    def _child(self, node: Int) -> Self:
        var base = self._index[].nodes[self._node].start
        ref value = self._index[].nodes[node]
        return Self(
            String(self.text[byte = value.start - base : value.end - base]),
            self._index,
            node,
        )

    def get(self, key: String) raises -> Self:
        if self.kind() != 123:
            raise Error("Expected JSON object")
        var child = self._index[].objects[self._node].values.get(key, -1)
        if child < 0:
            return Self("null")
        return self._child(child)

    def has(self, key: String) raises -> Bool:
        if self.kind() != 123:
            raise Error("Expected JSON object")
        return key in self._index[].objects[self._node].values

    def _canonical_at(self, index: Int, mut out: List[UInt8]) raises:
        ref node = self._index[].nodes[index]
        var base = self._index[].nodes[self._node].start
        if node.kind == 34:
            var reader = JsonReader(json_bytes(self.text))
            var quoted = json_quote(reader.string_at(node.start - base))
            out.extend(quoted.as_bytes())
        elif node.kind == 123:
            ref object = self._index[].objects[index]
            var keys = object.keys.copy()
            sort(keys)
            out.append(123)
            for i in range(len(keys)):
                if i:
                    out.append(44)
                var quoted = json_quote(keys[i])
                out.extend(quoted.as_bytes())
                out.append(58)
                self._canonical_at(object.values[keys[i]], out)
            out.append(125)
        elif node.kind == 91:
            out.append(91)
            var child = index + 1
            while child < node.after:
                if child != index + 1:
                    out.append(44)
                self._canonical_at(child, out)
                child = self._index[].nodes[child].after
            out.append(93)
        else:
            out.extend(
                self.text.as_bytes()[node.start - base : node.end - base]
            )

    def canonical(self) raises -> String:
        """Stable JSON spelling with sorted object keys (numbers unchanged)."""
        var bytes = List[UInt8](capacity=self.text.byte_length())
        self._canonical_at(self._node, bytes)
        return String(unsafe_from_utf8=bytes)

    def items(self) raises -> List[Self]:
        if self.kind() != 91:
            raise Error("Expected JSON array")
        var values = List[Self]()
        var child = self._node + 1
        while child < self._index[].nodes[self._node].after:
            values.append(self._child(child))
            child = self._index[].nodes[child].after
        return values^
