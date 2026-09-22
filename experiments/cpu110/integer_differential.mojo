from dataframe.parse import parse_int64, parse_integer


def report_int64(text: String) raises:
    try:
        print("int64,", text, ",", parse_int64(StringSlice(text)), sep="")
    except:
        print("int64,", text, ",ERR", sep="")


def report_u8(text: String) raises:
    try:
        print(
            "u8,",
            text,
            ",",
            parse_integer[DType.uint8](StringSlice(text)),
            sep="",
        )
    except:
        print("u8,", text, ",ERR", sep="")


def report_i16(text: String) raises:
    try:
        print(
            "i16,",
            text,
            ",",
            parse_integer[DType.int16](StringSlice(text)),
            sep="",
        )
    except:
        print("i16,", text, ",ERR", sep="")


def report_i32(text: String) raises:
    try:
        print(
            "i32,",
            text,
            ",",
            parse_integer[DType.int32](StringSlice(text)),
            sep="",
        )
    except:
        print("i32,", text, ",ERR", sep="")


def report_u64(text: String) raises:
    try:
        print(
            "u64,",
            text,
            ",",
            parse_integer[DType.uint64](StringSlice(text)),
            sep="",
        )
    except:
        print("u64,", text, ",ERR", sep="")


def report(text: String) raises:
    report_int64(text)
    report_u8(text)
    report_i16(text)
    report_i32(text)
    report_u64(text)


def main() raises:
    for text in [
        "",
        "+",
        "-",
        "+0",
        "-0",
        "0",
        "1",
        "12",
        "123",
        "1234",
        "0000",
        "0000000000000000001",
        "127",
        "128",
        "255",
        "256",
        "32767",
        "32768",
        "-32768",
        "-32769",
        "2147483647",
        "2147483648",
        "-2147483648",
        "-2147483649",
        "9223372036854775807",
        "9223372036854775808",
        "18446744073709551615",
        "18446744073709551616",
        "123x",
        "1234x",
        "999x",
        "9999x",
        "12 34",
    ]:
        report(text)
    var seed = UInt64(0x926B3CE45F0187DA)
    for _ in range(1000):
        seed = seed * 6364136223846793005 + 1442695040888963407
        report(String(seed))
        report("-" + String(seed))
