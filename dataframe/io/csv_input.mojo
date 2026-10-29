"""An owned CSV byte source, distinct from a filesystem path."""
from std.memory import ArcPointer


struct CsvBuffer(Copyable, Sized):
    """Reusable CSV bytes. A String constructor treats its text as CSV data.

    Byte lists convert implicitly; String paths remain unambiguous because
    the String constructor requires an explicit CsvBuffer(...). Copies share
    immutable bytes. Readers borrow them only while decoding; output columns
    own their values independently.
    """

    var _bytes: ArcPointer[List[UInt8]]

    @implicit
    def __init__(out self, bytes: List[UInt8]):
        self._bytes = ArcPointer(bytes.copy())

    def __init__(out self, text: String):
        var bytes = List[UInt8](capacity=text.byte_length())
        for byte in text.as_bytes():
            bytes.append(byte)
        self._bytes = ArcPointer(bytes^)

    def __len__(self) -> Int:
        return len(self._bytes[])

    def _span(self) -> Span[UInt8, ImmutAnyOrigin]:
        return Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=self._bytes[]
            .unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            length=len(self),
        )
