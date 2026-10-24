"""Unicode vector runs preserve scalar decoding, range bounds and captures."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_true
from dataframe.regex import Matcher, Regex
from dataframe.regex.onepass import _UNICODE_RUN, _LOOP_PC
from dataframe.regex.pikevm import decode, previous_start
from dataframe.regex.program import consumes
from dataframe.regex.unicode_run import _scan_width, scan_unicode_run
from .test_regex_engine import _span


def test_unicode_run_metadata_requires_a_unicode_predicate() raises:
    var patterns: List[String] = [
        "^(一+)(!*)",
        "^([一-丂]+)(!*)",
        "(?s)^(.+)",
        "^(.+)",
        "^(a+)(b*)",
        "^([一-丂丅-丆]+)(!*)",
        "^(\\w+) (\\w+) (\\w+)",
    ]
    var expected: List[Bool] = [True, True, True, True, False, True, True]
    for i in range(len(patterns)):
        var regex = Regex(patterns[i])
        assert_true(Bool(regex._compiled[].onepass), patterns[i])
        ref onepass = regex._compiled[].onepass.value()[].forward
        var found = False
        for encoded in onepass.loop_pc:
            if encoded < 0:
                continue
            var pc = Int(encoded) & _LOOP_PC
            assert_true(consumes(regex._compiled[].forward.op[pc]))
            found = found or (Int(encoded) & _UNICODE_RUN) != 0
        assert_equal(found, expected[i], patterns[i])


def _check_scalar[width: Int](text: String, cp: Int) raises:
    var span = _span(text)
    var end = len(span) - 4
    var distance = 12 if width == 3 else 8 * width
    assert_equal(
        _scan_width[width, False](span, 4, end, UInt32(cp), UInt32(cp)),
        4 + distance,
    )
    assert_equal(
        _scan_width[width, True](span, end, 4, UInt32(cp), UInt32(cp)),
        end - distance,
    )
    var other = UInt32(cp + 1 if cp < 0x10FFFF else cp - 1)
    assert_equal(_scan_width[width, False](span, 4, end, other, other), 4)
    assert_equal(_scan_width[width, True](span, end, 4, other, other), end)


def test_unicode_runs_decode_every_unicode_scalar_in_both_directions() raises:
    for cp in range(128, 0x110000):
        if cp >= 0xD800 and cp <= 0xDFFF:
            continue
        var text = "!!!!" + chr(cp) * 8 + "!!!!"
        if cp < 0x800:
            _check_scalar[2](text, cp)
        elif cp < 0x10000:
            _check_scalar[3](text, cp)
        else:
            _check_scalar[4](text, cp)


def _check_consumed[width: Int](text: Span[UInt8, ImmutAnyOrigin]) raises:
    var end = _scan_width[width, False](text, 0, len(text), 128, 0x10FFFF)
    assert_true(end >= 0 and end <= len(text))
    var at = 0
    while at < end:
        var packed = decode(text, at)
        assert_equal(packed & 7, width)
        assert_true((packed >> 3) >= 128 and (packed >> 3) <= 0x10FFFF)
        at += width
    assert_equal(at, end)
    var begin = _scan_width[width, True](text, len(text), 0, 128, 0x10FFFF)
    assert_true(begin >= 0 and begin <= len(text))
    at = len(text)
    while at > begin:
        var prev = previous_start(text, at)
        var packed = decode(text, prev)
        assert_equal(at - prev, width)
        assert_equal(packed & 7, width)
        assert_true((packed >> 3) >= 128 and (packed >> 3) <= 0x10FFFF)
        at = prev
    assert_equal(at, begin)


def test_unicode_runs_preserve_malformed_bytes_truncation_and_slices() raises:
    var bytes = List[UInt8](length=96, fill=0)
    var ptr = (
        bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    for a in range(256):
        for b in range(256):
            for k in range(0, 64, 2):
                bytes[k] = UInt8(a)
                bytes[k + 1] = UInt8(b)
            var span = Span[UInt8, ImmutAnyOrigin](unsafe_ptr=ptr, length=64)
            _check_consumed[2](span)
            _check_consumed[3](span)
            _check_consumed[4](span)
    for value in [String("α"), "中", "𐐀"]:
        var encoded = _span(value)
        for k in range(len(bytes)):
            bytes[k] = encoded[k % len(encoded)]
        for bad in range(64):
            var saved = bytes[bad]
            for replacement in range(256):
                bytes[bad] = UInt8(replacement)
                for offset in [0, 1, 3, 15]:
                    for length in [15, 16, 17, 31, 32, 33, 64]:
                        var span = Span[UInt8, ImmutAnyOrigin](
                            unsafe_ptr=ptr.unsafe_offset(offset), length=length
                        )
                        _check_consumed[2](span)
                        _check_consumed[3](span)
                        _check_consumed[4](span)
            bytes[bad] = saved
        _ = value
    _ = bytes


def test_unicode_vector_capture_runs_match_vm_at_range_and_utf8_boundaries() raises:
    var patterns: List[String] = [
        "^([α-ω]+)(!*)",
        "([α-ω]+)(!*)",
        "^(中+)(!*)",
        "(𐐀+)(!*)",
        "^([一-丂]+)([𐐀-𐐂]*)",
        "([一-丂]+) ([𐐀-𐐂]+) ([अ-आ]+)",
        "^(α+?)(!*)",
        "^((α)+)(!*)",
        "(?s)^(.+)",
        "^(.+)",
    ]
    var words: List[String] = ["α", "ω", "中", "一丂", "𐐀", "一丂 𐐀𐐂 अआ"]
    var suffixes: List[String] = ["", "!", "\n", "a", "中", "𐐀"]
    for pattern in patterns:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        var expected = List[Int](length=program.slot_count, fill=-1)
        for word in words:
            for length in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65]:
                for suffix in suffixes:
                    var text = word * length + suffix
                    var span = _span(text)
                    var found = matcher.pike.search(
                        program, span, 0, len(span), False, expected
                    )
                    assert_equal(matcher.captures(span), found, pattern)
                    if found:
                        assert_equal(matcher.slots, expected, pattern)
                    _ = text
