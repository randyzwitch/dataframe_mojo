"""Owned incremental gzip/zstd input and quote-aware CSV record framing.

The native stream borrows compressed CsvBuffer bytes; this holder retains their
Arc owner until after native close. Decompressed records own their bytes before
being submitted to workers. No temporary decompressed file is created.
"""
from std.collections import Optional
from std.memory import Pointer
from .csv_input import CsvBuffer
from .native_library import _Library, _load_library, _raise_backend_error

comptime _OpenPath = def(Int, Int32, Int, Int) thin abi("C") -> Int32
comptime _OpenBuffer = def(Int, Int64, Int32, Int, Int) thin abi("C") -> Int32
comptime _Read = def(Int, Int, Int64, Int, Int) thin abi("C") -> Int32
comptime _Close = def(Int) thin abi("C") -> None


def _open_path(
    entry: Int, path: String, codec: Int, mut handle: Int, mut error: Int
) -> Int32:
    var name = List[UInt8]()
    name.extend(path.as_bytes())
    name.append(0)
    var status = Pointer(to=entry).unsafe_bitcast[_OpenPath]()[](
        Int(name.unsafe_ptr()),
        Int32(codec),
        Int(Pointer(to=handle)),
        Int(Pointer(to=error)),
    )
    # Raw integer addresses do not retain the filename's byte owner.
    _ = name^
    return status


def _open_buffer(
    entry: Int,
    source: CsvBuffer,
    codec: Int,
    mut handle: Int,
    mut error: Int,
) -> Int32:
    var bytes = source._span()
    return Pointer(to=entry).unsafe_bitcast[_OpenBuffer]()[](
        Int(bytes.unsafe_ptr()),
        Int64(len(bytes)),
        Int32(codec),
        Int(Pointer(to=handle)),
        Int(Pointer(to=error)),
    )


def _read_input(
    entry: Int,
    handle: Int,
    mut bytes: List[UInt8],
    mut count: Int64,
    mut error: Int,
) -> Int32:
    # Explicit mut boundaries expose C's output writes to Mojo's optimizer.
    return Pointer(to=entry).unsafe_bitcast[_Read]()[](
        handle,
        Int(bytes.unsafe_ptr()),
        Int64(len(bytes)),
        Int(Pointer(to=count)),
        Int(Pointer(to=error)),
    )


def _csv_compression(
    bytes: Span[UInt8, ImmutAnyOrigin], path: String = ""
) -> Int:
    """Content wins over extension, including a conflicting known extension."""
    if len(bytes) >= 2 and bytes[0] == 31 and bytes[1] == 139:
        return 1
    if len(bytes) >= 4:
        if (
            bytes[0] == 40
            and bytes[1] == 181
            and bytes[2] == 47
            and bytes[3] == 253
        ):
            return 2
        # Zstd permits skippable frames before a data frame.
        if (
            bytes[0] >= 80
            and bytes[0] <= 95
            and bytes[1] == 42
            and bytes[2] == 77
            and bytes[3] == 24
        ):
            return 2
    var lower = path.lower()
    if lower.endswith(".gz") or lower.endswith(".gzip"):
        return 1
    if lower.endswith(".zst") or lower.endswith(".zstd"):
        return 2
    return 0


def _csv_span(bytes: List[UInt8]) -> Span[UInt8, ImmutAnyOrigin]:
    # The caller retains the byte owner for the borrowed synchronous operation.
    return Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(bytes),
    )


struct _CsvCompressedInput(Movable):
    var library: _Library
    var handle: Int
    var read_entry: Int
    var close_entry: Int
    var path: String
    var owner: List[CsvBuffer]
    var codec: Int
    var eof: Bool

    def __init__(out self, path: String, codec: Int) raises:
        self.library = _load_library()
        self.handle = 0
        self.read_entry = _Library._symbol(
            self.library.handle, "dfq_csv_input_read"
        )
        self.close_entry = _Library._symbol(
            self.library.handle, "dfq_csv_input_close"
        )
        self.path = path.copy()
        self.owner = List[CsvBuffer]()
        self.codec = codec
        self.eof = False
        self._open()

    def __init__(out self, source: CsvBuffer, codec: Int) raises:
        self.library = _load_library()
        self.handle = 0
        self.read_entry = _Library._symbol(
            self.library.handle, "dfq_csv_input_read"
        )
        self.close_entry = _Library._symbol(
            self.library.handle, "dfq_csv_input_close"
        )
        self.path = ""
        self.owner = [source.copy()]
        self.codec = codec
        self.eof = False
        self._open()

    def _open(mut self) raises:
        var error = 0
        var status: Int32
        if len(self.owner):
            var entry = _Library._symbol(
                self.library.handle, "dfq_csv_input_open_buffer"
            )
            status = _open_buffer(
                entry, self.owner[0], self.codec, self.handle, error
            )
        else:
            if "\x00" in self.path:
                raise Error("CSV path must not contain NUL bytes")
            var entry = _Library._symbol(
                self.library.handle, "dfq_csv_input_open_path"
            )
            status = _open_path(
                entry, self.path, self.codec, self.handle, error
            )
        if status != 0:
            _raise_backend_error(
                self.library, error, "read_csv compressed input"
            )

    def __deinit__(deinit self):
        if self.handle != 0:
            var entry = self.close_entry
            Pointer(to=entry).unsafe_bitcast[_Close]()[](self.handle)
        # Native buffer streams borrow these bytes until after close.
        _ = self.owner^

    def reset(mut self) raises:
        var entry = self.close_entry
        Pointer(to=entry).unsafe_bitcast[_Close]()[](self.handle)
        self.handle = 0
        self.eof = False
        self._open()

    def read(mut self) raises -> List[UInt8]:
        if self.eof:
            return List[UInt8]()
        var bytes = List[UInt8](length=65536, fill=0)
        var count = Int64(0)
        var error = 0
        var entry = self.read_entry
        var status = _read_input(entry, self.handle, bytes, count, error)
        if status != 0:
            _raise_backend_error(
                self.library, error, "read_csv compressed input"
            )
        bytes.resize(Int(count), 0)
        self.eof = count == 0
        return bytes^

    def close(mut self):
        # A pushed-down row limit does not consume the unread compressed tail.
        if self.handle != 0:
            var entry = self.close_entry
            Pointer(to=entry).unsafe_bitcast[_Close]()[](self.handle)
            self.handle = 0
        self.eof = True


struct _CsvRecords(Movable):
    var input: _CsvCompressedInput
    var chunk: List[UInt8]
    var offset: Int
    var quote: UInt8
    var quoting: Bool

    def __init__(out self, var input: _CsvCompressedInput, quote_char: String):
        self.input = input^
        self.chunk = List[UInt8]()
        self.offset = 0
        self.quoting = quote_char.byte_length() != 0
        self.quote = quote_char.as_bytes()[0] if self.quoting else UInt8(34)

    def reset(mut self) raises:
        self.input.reset()
        self.chunk.clear()
        self.offset = 0

    def next(mut self) raises -> Optional[List[UInt8]]:
        var record = List[UInt8]()
        var inside = False
        while True:
            if self.offset == len(self.chunk):
                self.chunk = self.input.read()
                self.offset = 0
                if len(self.chunk) == 0:
                    if len(record):
                        return record^
                    return None
            var start = self.offset
            while self.offset < len(self.chunk):
                var byte = self.chunk[self.offset]
                self.offset += 1
                if self.quoting and byte == self.quote:
                    inside = not inside
                elif byte == 10 and not inside:
                    record.extend(_csv_span(self.chunk)[start : self.offset])
                    return record^
            record.extend(_csv_span(self.chunk)[start : self.offset])

    def close(mut self):
        self.chunk.clear()
        self.offset = 0
        self.input.close()
