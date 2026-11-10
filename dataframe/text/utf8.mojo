"""Append validated Unicode scalars to shared output byte buffers."""
from std.collections.string import Codepoint


@always_inline
def append_codepoint(mut out: List[UInt8], cp: Int):
    """The caller supplies a valid Unicode scalar, after input validation."""
    var scalar = Codepoint(unsafe_unchecked_codepoint=UInt32(cp))
    var start = len(out)
    var size = scalar.utf8_byte_length()
    out.resize(start + size, 0)
    _ = scalar.unsafe_write_utf8(out.unsafe_ptr().unsafe_offset(start))
