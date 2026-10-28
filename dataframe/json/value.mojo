"""Small strict JSON reader for geospatial metadata and GeoJSON.

Values own their source text. Nesting is bounded and duplicate object keys
are rejected, so metadata cannot have ambiguous interpretations.
"""

from dataframe.json.reader import JsonReader, json_bytes


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


struct JsonValue(Copyable, Movable):
    var text: String

    def __init__(out self, text: String) raises:
        var pos = 0
        _scan(text, pos)
        _space(text, pos)
        if pos != text.byte_length():
            raise Error("Trailing JSON content")
        self.text = String(text.strip())

    def kind(self) -> UInt8:
        return self.text.as_bytes()[0]

    def string(self) raises -> String:
        var pos = 0
        return _string(self.text, pos)

    def keys(self) raises -> List[String]:
        if self.kind() != 123:
            raise Error("Expected JSON object")
        var keys = List[String]()
        var pos = 1
        _space(self.text, pos)
        while self.text.as_bytes()[pos] != 125:
            keys.append(_string(self.text, pos))
            _space(self.text, pos)
            pos += 1
            _scan(self.text, pos)
            _space(self.text, pos)
            if self.text.as_bytes()[pos] == 44:
                pos += 1
                _space(self.text, pos)
        return keys^

    def get(self, key: String) raises -> Self:
        if self.kind() != 123:
            raise Error("Expected JSON object")
        var pos = 1
        _space(self.text, pos)
        while self.text.as_bytes()[pos] != 125:
            var name = _string(self.text, pos)
            _space(self.text, pos)
            pos += 1
            _space(self.text, pos)
            var start = pos
            _scan(self.text, pos)
            if name == key:
                return Self(String(self.text[byte=start:pos]))
            _space(self.text, pos)
            if self.text.as_bytes()[pos] == 44:
                pos += 1
                _space(self.text, pos)
        return Self("null")

    def has(self, key: String) raises -> Bool:
        for name in self.keys():
            if name == key:
                return True
        return False

    def canonical(self) raises -> String:
        """Stable JSON spelling with sorted object keys (numbers unchanged)."""
        if self.kind() == 34:
            return json_quote(self.string())
        if self.kind() == 123:
            var keys = self.keys()
            for i in range(1, len(keys)):
                var j = i
                while j > 0 and keys[j] < keys[j - 1]:
                    var saved = keys[j]
                    keys[j] = keys[j - 1]
                    keys[j - 1] = saved
                    j -= 1
            var result = String("{")
            for i in range(len(keys)):
                if i > 0:
                    result += ","
                result += (
                    json_quote(keys[i]) + ":" + self.get(keys[i]).canonical()
                )
            return result + "}"
        if self.kind() == 91:
            var result = String("[")
            var items = self.items()
            for i in range(len(items)):
                if i > 0:
                    result += ","
                result += items[i].canonical()
            return result + "]"
        return self.text

    def items(self) raises -> List[Self]:
        if self.kind() != 91:
            raise Error("Expected JSON array")
        var values = List[Self]()
        var pos = 1
        _space(self.text, pos)
        while self.text.as_bytes()[pos] != 93:
            var start = pos
            _scan(self.text, pos)
            values.append(Self(String(self.text[byte=start:pos])))
            _space(self.text, pos)
            if self.text.as_bytes()[pos] == 44:
                pos += 1
                _space(self.text, pos)
        return values^
