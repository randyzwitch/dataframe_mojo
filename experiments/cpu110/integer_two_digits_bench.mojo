from std.time import monotonic

from dataframe.parse import parse_integer


comptime ITERATIONS = 20000000


def timed[D: DType](fields: List[String]) raises -> Int:
    var total = UInt64(0)
    var start = monotonic()
    for i in range(ITERATIONS):
        total ^= parse_integer[D](StringSlice(fields[i & 15])).cast[
            DType.uint64
        ]()
    if total == UInt64.MAX:
        raise Error("unexpected checksum")
    return monotonic() - start


def main() raises:
    var fields: List[String] = [
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
    print("u8_2_ns,", timed[DType.uint8](fields), sep="")
    print("i16_2_ns,", timed[DType.int16](fields), sep="")
    print("i32_2_ns,", timed[DType.int32](fields), sep="")
    print("u64_2_ns,", timed[DType.uint64](fields), sep="")
