"""Small strict JSON reader for geospatial metadata and GeoJSON.

Values own their source text. Nesting is bounded and duplicate object keys
are rejected, so metadata cannot have ambiguous interpretations.
"""


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
    while pos < text.byte_length():
        var b = text.as_bytes()[pos]
        if b != 32 and b != 9 and b != 10 and b != 13:
            break
        pos += 1


def _hex4(text: String, mut pos: Int) raises -> Int:
    var value = 0
    for _ in range(4):
        if pos >= text.byte_length():
            raise Error("Truncated JSON Unicode escape")
        var b = Int(text.as_bytes()[pos])
        pos += 1
        var d = b - 48 if b >= 48 and b <= 57 else (
            b - 87 if b >= 97
            and b <= 102 else (b - 55 if b >= 65 and b <= 70 else -1)
        )
        if d < 0:
            raise Error("Invalid JSON Unicode escape")
        value = value * 16 + d
    return value


def _string(text: String, mut pos: Int) raises -> String:
    if pos >= text.byte_length() or text.as_bytes()[pos] != 34:
        raise Error("Expected JSON string")
    pos += 1
    var bytes = List[UInt8]()
    while pos < text.byte_length():
        var b = text.as_bytes()[pos]
        pos += 1
        if b == 34:
            return String(StringSlice(from_utf8=Span(bytes)))
        if b < 32:
            raise Error("Unescaped control character in JSON string")
        if b != 92:
            bytes.append(b)
            continue
        if pos >= text.byte_length():
            break
        b = text.as_bytes()[pos]
        pos += 1
        if b == 34 or b == 92 or b == 47:
            bytes.append(b)
        elif b == 98:
            bytes.append(8)
        elif b == 102:
            bytes.append(12)
        elif b == 110:
            bytes.append(10)
        elif b == 114:
            bytes.append(13)
        elif b == 116:
            bytes.append(9)
        elif b == 117:
            var cp = _hex4(text, pos)
            if cp >= 0xD800 and cp <= 0xDBFF:
                if (
                    pos + 2 > text.byte_length()
                    or text.as_bytes()[pos] != 92
                    or text.as_bytes()[pos + 1] != 117
                ):
                    raise Error("Missing JSON low surrogate")
                pos += 2
                var low = _hex4(text, pos)
                if low < 0xDC00 or low > 0xDFFF:
                    raise Error("Invalid JSON low surrogate")
                cp = 0x10000 + (cp - 0xD800) * 1024 + low - 0xDC00
            elif cp >= 0xDC00 and cp <= 0xDFFF:
                raise Error("Unexpected JSON low surrogate")
            bytes.extend(chr(cp).as_bytes())
        else:
            raise Error("Invalid JSON escape")
    raise Error("Unterminated JSON string")


def _scan(text: String, mut pos: Int, depth: Int = 0) raises:
    if depth > 64:
        raise Error("JSON nesting exceeds 64")
    _space(text, pos)
    if pos >= text.byte_length():
        raise Error("Expected JSON value")
    var b = text.as_bytes()[pos]
    if b == 34:
        _ = _string(text, pos)
        return
    if b == 123 or b == 91:
        var object = b == 123
        var close = 125 if object else 93
        pos += 1
        _space(text, pos)
        if pos < text.byte_length() and Int(text.as_bytes()[pos]) == close:
            pos += 1
            return
        var keys = List[String]()
        while True:
            if object:
                var key = _string(text, pos)
                for previous in keys:
                    if previous == key:
                        raise Error("Duplicate JSON key: " + key)
                keys.append(key^)
                _space(text, pos)
                if pos >= text.byte_length() or text.as_bytes()[pos] != 58:
                    raise Error("Expected JSON colon")
                pos += 1
            _scan(text, pos, depth + 1)
            _space(text, pos)
            if pos >= text.byte_length():
                raise Error("Unterminated JSON container")
            b = text.as_bytes()[pos]
            pos += 1
            if Int(b) == close:
                return
            if b != 44:
                raise Error("Expected JSON comma")
            _space(text, pos)
    if b == 110 or b == 116 or b == 102:
        var literal = "null" if b == 110 else ("true" if b == 116 else "false")
        var end = pos + literal.byte_length()
        if end > text.byte_length() or String(text[byte=pos:end]) != literal:
            raise Error("Invalid JSON literal")
        pos = end
        return
    var start = pos
    if b == 45:
        pos += 1
    if pos >= text.byte_length():
        raise Error("Invalid JSON number")
    b = text.as_bytes()[pos]
    if b == 48:
        pos += 1
    elif b >= 49 and b <= 57:
        while (
            pos < text.byte_length()
            and text.as_bytes()[pos] >= 48
            and text.as_bytes()[pos] <= 57
        ):
            pos += 1
    else:
        raise Error("Invalid JSON value")
    if pos < text.byte_length() and text.as_bytes()[pos] == 46:
        pos += 1
        var digits = pos
        while (
            pos < text.byte_length()
            and text.as_bytes()[pos] >= 48
            and text.as_bytes()[pos] <= 57
        ):
            pos += 1
        if pos == digits:
            raise Error("Invalid JSON fraction")
    if pos < text.byte_length() and (
        text.as_bytes()[pos] == 101 or text.as_bytes()[pos] == 69
    ):
        pos += 1
        if pos < text.byte_length() and (
            text.as_bytes()[pos] == 43 or text.as_bytes()[pos] == 45
        ):
            pos += 1
        var digits = pos
        while (
            pos < text.byte_length()
            and text.as_bytes()[pos] >= 48
            and text.as_bytes()[pos] <= 57
        ):
            pos += 1
        if pos == digits:
            raise Error("Invalid JSON exponent")
    if pos == start:
        raise Error("Invalid JSON value")


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
