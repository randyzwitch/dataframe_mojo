"""CSV integer adapters: decimal grammar and destination-range validation."""
from dataframe.text.parse import parse_int64


def parse_csv_uint64(text: StringSlice) raises -> UInt64:
    """Parse UInt64's full range, which does not fit the standard Int parser."""
    var bytes = text.as_bytes()
    var start = 1 if len(bytes) and bytes[0] == 43 else 0
    if start == len(bytes):
        raise Error("CSV integer requires digits")
    var value = UInt64(0)
    for i in range(start, len(bytes)):
        var byte = bytes[i]
        if byte < 48 or byte > 57:
            raise Error("invalid CSV integer byte")
        var digit = UInt64(byte - 48)
        if value > (UInt64.MAX - digit) // 10:
            raise Error("CSV unsigned integer overflow")
        value = value * 10 + digit
    return value


def parse_csv_int64(text: StringSlice) raises -> Int64:
    """Parse signed CSV integers with strict full consumption."""
    return parse_int64(text)


def parse_csv_integer[D: DType](text: StringSlice) raises -> Scalar[D]:
    """Parse CSV integers, preserving unsigned rejection of a minus sign."""
    comptime assert D.is_integral(), "parse_csv_integer needs an integer dtype"
    comptime if D.is_signed():
        var value = parse_csv_int64(text)
        if value < Int64(Scalar[D].MIN) or value > Int64(Scalar[D].MAX):
            raise Error("CSV signed integer overflow")
        return value.cast[D]()
    else:
        var value = parse_csv_uint64(text)
        if value > UInt64(Scalar[D].MAX):
            raise Error("CSV unsigned integer overflow")
        return value.cast[D]()
