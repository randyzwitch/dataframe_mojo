"""Row adapters for hex and standard-library base64 codecs."""
from std.base64 import b64encode, b64decode, b16decode

comptime _HEX = "0123456789abcdef"


def encode_bytes(row: Span[UInt8, _], hex: Bool, mut out: List[UInt8]):
    """Append one byte span as hex/base64 without materializing a column."""
    var hex_digits = StringSlice(_HEX).as_bytes()
    var data = row.unsafe_ptr()
    if hex:
        for b in range(len(row)):
            var byte = Int(data[unsafe_offset=b])
            out.append(hex_digits[byte >> 4])
            out.append(hex_digits[byte & 15])
    else:
        var encoded = b64encode(row)
        out.extend(encoded.as_bytes())


@always_inline
def digit(byte: UInt8) -> Int:
    """A byte's value as a digit in bases up to 36, or 99."""
    if byte >= 48 and byte <= 57:
        return Int(byte) - 48
    var lower = byte | 32
    if lower >= 97 and lower <= 122:
        return Int(lower) - 87
    return 99


def decode_bytes(row: Span[UInt8, _], hex: Bool, mut out: List[UInt8]) -> Bool:
    """Append decoded bytes; return False for invalid input.

    Base64 follows the standard library, including whitespace and nonzero
    padding bits. Hex accepts either case and requires pairs of digits.
    """
    try:
        var text = String(StringSpan(from_utf8=row))
        if hex:
            for byte in row:
                if byte >= 128:
                    return False
            var upper = text.upper()
            out.extend(b16decode(upper))
        else:
            out.extend(b64decode(text))
        return True
    except:
        return False
