"""Borrowed NDJSON records over mapped files, with bounded row framing."""
from std.bit import count_trailing_zeros
from std.collections import Optional
from std.memory import Pointer
from .csv_bits import _mask64
from .csv_types import _Mapping, _map_file
from ..json_reader import JsonBytes, JsonReader


struct JsonInput(Movable):
    """Own the source while record spans and parser tapes borrow its bytes."""

    var mapping: _Mapping
    var owned: List[UInt8]
    var address: Int
    var size: Int

    def __init__(out self, path: String) raises:
        self.mapping = _map_file(path)
        self.owned = List[UInt8]()
        self.address = self.mapping.address
        self.size = self.mapping.length
        if self.address == 0:
            with open(path, "r") as file:
                self.owned = file.read_bytes()
            self.address = Int(self.owned.unsafe_ptr())
            self.size = len(self.owned)
        var bytes = self.span()
        if (
            self.size >= 3
            and bytes[0] == 239
            and bytes[1] == 187
            and bytes[2] == 191
        ):
            self.address += 3
            self.size -= 3

    def span(self) -> JsonBytes:
        return JsonBytes(
            unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=self.address
            ),
            length=self.size,
        )


@fieldwise_init
struct JsonRecord(Copyable, Movable):
    var start: Int
    var end: Int
    var line: Int
    var row: Int


struct JsonLines(Movable):
    """Find LF boundaries without allocating a String or offsets per row.

    JSON escapes contain the two bytes backslash/n, never a literal LF.
    The JSON parser validates each complete record separately. Whitespace
    lines do not contribute rows; physical line numbers are retained.
    """

    var address: Int
    var size: Int
    var pos: Int
    var line: Int
    var rows: Int

    def __init__(out self, bytes: JsonBytes):
        self.address = Int(bytes.unsafe_ptr())
        self.size = len(bytes)
        self.pos = 0
        self.line = 0
        self.rows = 0

    def span(self) -> JsonBytes:
        return JsonBytes(
            unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=self.address
            ),
            length=self.size,
        )

    def next(mut self) -> Optional[JsonRecord]:
        var bytes = self.span()
        while self.pos < self.size:
            var start = self.pos
            while self.pos + 64 <= self.size:
                var block = bytes.unsafe_ptr().unsafe_load[width=64](self.pos)
                var mask = _mask64(block.eq(SIMD[DType.uint8, 64](10)))
                if mask != 0:
                    self.pos += Int(count_trailing_zeros(mask))
                    break
                self.pos += 64
            while self.pos < self.size and bytes[self.pos] != 10:
                self.pos += 1
            var end = self.pos
            self.pos += Int(self.pos < self.size)
            var line = self.line
            self.line += 1
            var at = start
            while at < end and (
                bytes[at] == 32 or bytes[at] == 9 or bytes[at] == 13
            ):
                at += 1
            if at == end:
                continue
            var row = self.rows
            self.rows += 1
            return JsonRecord(start, end, line, row)
        return None


struct JsonArrayRows(Movable):
    """Reuse the shared parser's tape for one array object at a time."""

    var reader: JsonReader
    var rows: Int
    var ended: Bool

    def __init__(out self, bytes: JsonBytes) raises:
        self.reader = JsonReader(bytes, max_depth=1024)
        self.rows = 0
        self.ended = False
        self.reader.space()
        if (
            self.reader.pos >= self.reader.size
            or self.reader.byte(self.reader.pos) != 91
        ):
            raise Error("JSON table input must be an array of objects")
        self.reader.pos += 1

    def next(mut self) raises -> Bool:
        if self.ended:
            return False
        self.reader.nodes.clear()
        self.reader.space()
        if (
            self.reader.pos < self.reader.size
            and self.reader.byte(self.reader.pos) == 93
        ):
            self.reader.pos += 1
            self.reader.space()
            if self.reader.pos != self.reader.size:
                raise Error("Unexpected content after JSON array")
            self.ended = True
            return False
        if self.rows > 0:
            if (
                self.reader.pos >= self.reader.size
                or self.reader.byte(self.reader.pos) != 44
            ):
                raise Error("Expected JSON array comma or closing bracket")
            self.reader.pos += 1
            self.reader.space()
        _ = self.reader.scan(store=True, unique_keys=False)
        if self.reader.nodes[0].kind != 123:
            raise Error("JSON table rows must contain objects")
        self.rows += 1
        return True
