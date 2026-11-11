"""CSV Float32 conversion keeps exact rounding and bounded-field grammar."""
# contract-group: kernel.parse
from std.memory import bitcast
from std.testing import assert_equal, assert_raises
from dataframe.io.csv_numeric import parse_csv_float32, parse_csv_float64


def _bits(text: String, expected: UInt32) raises:
    for offset in [0, 1, 7, 15]:
        var prefix = String()
        for _ in range(offset):
            prefix += "9"
        var owner = prefix + text + "987e42"
        var field = StringSlice(
            unsafe_from_utf8=owner.as_bytes()[
                offset : offset + text.byte_length()
            ]
        )
        assert_equal(bitcast[DType.uint32](parse_csv_float32(field)), expected)
        _ = owner^


def _invalid(text: String) raises:
    var owner = "9" + text + "123"
    var field = StringSlice(
        unsafe_from_utf8=owner.as_bytes()[1 : 1 + text.byte_length()]
    )
    with assert_raises():
        _ = parse_csv_float32(field)
    with assert_raises():
        _ = parse_csv_float64(field)
    _ = owner^


def test_csv_float32_halfway_residuals_round_at_target_precision() raises:
    _bits("1.000000059604644775390625", 0x3F800000)
    _bits("1.000000059604644775390625000001", 0x3F800001)
    _bits("1.000000059604644775390624999999", 0x3F800000)
    _bits("-1.000000059604644775390625000001", 0xBF800001)
    _bits("16777217", 0x4B800000)
    _bits("16777217.00000000000000000001", 0x4B800001)
    # Half the minimum subnormal, exactly and on either side.
    _bits(
        "7.00649232162408535461864791644958065640130970938257885878534141944895541342930300743319094181060791015625e-46",
        0,
    )
    _bits("7.006492321624085354618647916449580656401309709382579e-46", 1)
    _bits("7.006492321624085354618647916449580656401309709382578e-46", 0)
    _bits("1.17549435082228750796873653722224568e-38", 0x00800000)


def test_csv_float32_range_boundaries_and_overflow_do_not_hide_bad_suffixes() raises:
    _bits("3.40282346638528859811704183484516925440e38", 0x7F7FFFFF)
    # Exact midpoint between maximum finite and the hypothetical 2^128.
    _bits("340282356779733661637539395458142568448", 0x7F800000)
    _bits("340282356779733661637539395458142568447", 0x7F7FFFFF)
    _bits("-340282356779733661637539395458142568447", 0xFF7FFFFF)
    for text in ["1e40", "1e400", "+001e99999"]:
        _bits(text, 0x7F800000)
        _bits("-" + text.lstrip("+"), 0xFF800000)
    for text in ["1e-99999", "0e99999", "0e-99999"]:
        _bits(text, 0)
        _bits("-" + text, 0x80000000)
    for text in ["1e400x", "1e400,", "1e400\x00", "1e400.", "1e400e1"]:
        _invalid(text)


def test_csv_float32_csv_spellings_specials_and_signed_zero() raises:
    for text in ["1.5", "+1.5", "001.5", "+0001.50", "1.5e0"]:
        _bits(text, 0x3FC00000)
    for text in [".5", "+.5", "000.5", ".5e0"]:
        _bits(text, 0x3F000000)
        _bits("-" + text.lstrip("+"), 0xBF000000)
    for text in ["1.", "+0001.", "1.e0", "001.e+0"]:
        _bits(text, 0x3F800000)
    for text in ["-0", "-000", "-0.", "-.0", "-0.e100"]:
        _bits(text, 0x80000000)
    for text in ["nan", "NaN", "NAN", "nAn"]:
        _bits(text, 0x7FC00000)
        _bits("+" + text, 0x7FC00000)
        _bits("-" + text, 0xFFC00000)
    for text in ["inf", "INF", "Infinity", "iNfInItY"]:
        _bits(text, 0x7F800000)
        _bits("+" + text, 0x7F800000)
        _bits("-" + text, 0xFF800000)


def test_csv_float_fields_reject_non_numeric_bytes_and_incomplete_grammar() raises:
    for text in [
        "",
        "+",
        "-",
        ".",
        "--1",
        "+-1",
        "-+1",
        "++1",
        "1e",
        "1e+",
        "1e-",
        "1.2.",
        "1e2.",
        "1..",
        "1.e.",
        "1e1.0",
        "1e1e1",
        " 1",
        "1 ",
        "1\t",
        "nan(1)",
        "nanx",
        "Infinityx",
        "0x1p2",
        "1_0",
    ]:
        _invalid(text)
    # A decimal digit extends the fraction; every other byte is invalid here.
    for byte in range(256):
        if 48 <= byte <= 57:
            continue
        var data = List[UInt8](length=4, fill=49)
        data[1] = 46
        data[2] = 53
        data[3] = UInt8(byte)
        _invalid(String(unsafe_from_utf8=data))
