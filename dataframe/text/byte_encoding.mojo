"""Shared row-wise hex and canonical padded base64 codecs."""

comptime _HEX = "0123456789abcdef"
comptime _BASE64 = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
)


def encode_bytes(row: Span[UInt8, _], hex: Bool, mut out: List[UInt8]):
    """Append one byte span as hex/base64 without materializing a column."""
    var hex_digits = StringSlice(_HEX).as_bytes()
    var alphabet = StringSlice(_BASE64).as_bytes()
    var data = row.unsafe_ptr()
    if hex:
        for b in range(len(row)):
            var byte = Int(data[unsafe_offset=b])
            out.append(hex_digits[byte >> 4])
            out.append(hex_digits[byte & 15])
    else:
        var b = 0
        while b < len(row):
            var left = len(row) - b
            var word = Int(data[unsafe_offset=b]) << 16
            if left > 1:
                word |= Int(data[unsafe_offset=b + 1]) << 8
            if left > 2:
                word |= Int(data[unsafe_offset=b + 2])
            out.append(alphabet[(word >> 18) & 63])
            out.append(alphabet[(word >> 12) & 63])
            out.append(alphabet[(word >> 6) & 63] if left > 1 else 61)
            out.append(alphabet[word & 63] if left > 2 else 61)
            b += 3


@always_inline
def digit(byte: UInt8) -> Int:
    """A byte's value as a digit in bases up to 36, or 99."""
    if byte >= 48 and byte <= 57:
        return Int(byte) - 48
    var lower = byte | 32
    if lower >= 97 and lower <= 122:
        return Int(lower) - 87
    return 99


@always_inline
def _sextet(byte: UInt8) -> Int:
    """A base64 character's value, or -1."""
    if byte >= 65 and byte <= 90:
        return Int(byte) - 65
    if byte >= 97 and byte <= 122:
        return Int(byte) - 71
    if byte >= 48 and byte <= 57:
        return Int(byte) + 4
    if byte == 43:
        return 62
    if byte == 47:
        return 63
    return -1


def decode_bytes(row: Span[UInt8, _], hex: Bool, mut out: List[UInt8]) -> Bool:
    """Append the bytes a row encodes; False (with `out` partly written) if
    the row is not valid hex (an even number of digits) or canonical padded
    base64 (Rust's `base64` STANDARD engine: padding required, unused bits
    zero)."""
    var size = len(row)
    var data = row.unsafe_ptr()
    if hex:
        if size % 2 != 0:
            return False
        for b in range(0, size, 2):
            var high = digit(data[unsafe_offset=b])
            var low = digit(data[unsafe_offset=b + 1])
            if high > 15 or low > 15:
                return False
            out.append(UInt8(high * 16 + low))
        return True
    if size % 4 != 0:
        return False
    for b in range(0, size, 4):
        var last = b + 4 == size
        var values = SIMD[DType.int32, 4](0)
        var padding = 0
        for j in range(4):
            var byte = data[unsafe_offset=b + j]
            if byte == 61 and last and j >= 2:
                padding += 1
                continue
            if padding > 0:
                return False
            var value = _sextet(byte)
            if value < 0:
                return False
            values[j] = Int32(value)
        var word = Int(
            (values[0] << 18) | (values[1] << 12) | (values[2] << 6) | values[3]
        )
        out.append(UInt8((word >> 16) & 255))
        if padding == 2:
            if (word & 0xFFFF) != 0:
                return False
        elif padding == 1:
            if (word & 0xFF) != 0:
                return False
            out.append(UInt8((word >> 8) & 255))
        else:
            out.append(UInt8((word >> 8) & 255))
            out.append(UInt8(word & 255))
    return True
