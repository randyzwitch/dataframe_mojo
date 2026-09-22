from std.time import monotonic

from dataframe.parse import parse_integer


comptime ITERATIONS = 20000000


def timed_u8(fields: List[String]) raises -> Int:
    var total = UInt64(0)
    var start = monotonic()
    for i in range(ITERATIONS):
        total ^= parse_integer[DType.uint8](StringSlice(fields[i & 15])).cast[
            DType.uint64
        ]()
    if total == UInt64.MAX:
        raise Error("unexpected checksum")
    return monotonic() - start


def timed_i16(fields: List[String]) raises -> Int:
    var total = UInt64(0)
    var start = monotonic()
    for i in range(ITERATIONS):
        total ^= parse_integer[DType.int16](StringSlice(fields[i & 15])).cast[
            DType.uint64
        ]()
    if total == UInt64.MAX:
        raise Error("unexpected checksum")
    return monotonic() - start


def timed_i32(fields: List[String]) raises -> Int:
    var total = UInt64(0)
    var start = monotonic()
    for i in range(ITERATIONS):
        total ^= parse_integer[DType.int32](StringSlice(fields[i & 15])).cast[
            DType.uint64
        ]()
    if total == UInt64.MAX:
        raise Error("unexpected checksum")
    return monotonic() - start


def timed_u64(fields: List[String]) raises -> Int:
    var total = UInt64(0)
    var start = monotonic()
    for i in range(ITERATIONS):
        total ^= parse_integer[DType.uint64](StringSlice(fields[i & 15])).cast[
            DType.uint64
        ]()
    if total == UInt64.MAX:
        raise Error("unexpected checksum")
    return monotonic() - start


def main() raises:
    var u8: List[String] = [
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
    ]
    var i16: List[String] = [
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "6789",
        "7890",
        "8912",
        "9123",
        "1235",
        "2346",
        "3457",
        "4568",
        "5679",
        "6781",
        "7892",
    ]
    var i32: List[String] = [
        "123456789",
        "223456789",
        "323456789",
        "423456789",
        "523456789",
        "623456789",
        "723456789",
        "823456789",
        "123456781",
        "223456781",
        "323456781",
        "423456781",
        "523456781",
        "623456781",
        "723456781",
        "823456781",
    ]
    var u64: List[String] = [
        "1234567890123456789",
        "2234567890123456789",
        "3234567890123456789",
        "4234567890123456789",
        "5234567890123456789",
        "6234567890123456789",
        "7234567890123456789",
        "8234567890123456789",
        "1234567891123456789",
        "2234567891123456789",
        "3234567891123456789",
        "4234567891123456789",
        "5234567891123456789",
        "6234567891123456789",
        "7234567891123456789",
        "8234567891123456789",
    ]
    print("u8_ns,", timed_u8(u8), sep="")
    print("i16_ns,", timed_i16(i16), sep="")
    print("i32_ns,", timed_i32(i32), sep="")
    print("u64_ns,", timed_u64(u64), sep="")
