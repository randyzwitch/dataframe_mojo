"""Vector capture runs preserve scalar membership, range bounds and slots."""
# contract-group: kernel.regex
from std.testing import assert_equal
from dataframe.regex import Matcher, Regex
from dataframe.regex.ascii_run import scan_ascii_run
from dataframe.regex.byte_lookup import lookup16
from dataframe.regex.program import consumes
from .test_regex_engine import _span


def test_ascii_run_lookup_preserves_every_table_entry() raises:
    var table = SIMD[DType.uint8, 16]()
    for seed in [0, 7, 127, 255]:
        for i in range(16):
            table[i] = UInt8((seed + i * 17) & 255)
        for shift in range(16):
            var indices = SIMD[DType.uint8, 16]()
            for i in range(16):
                indices[i] = UInt8((i + shift) & 15)
            var actual = lookup16(table, indices)
            for i in range(16):
                assert_equal(actual[i], table[Int(indices[i])])


def test_ascii_runs_match_scalar_predicates_at_all_alignments() raises:
    var patterns: List[String] = [
        "([a-z]+)",
        "([a-zA-Z0-9_]+)",
        "([^a]+)",
        "(?s)(.+)",
        "(.+)",
        "(a+)",
        "(中+)",
    ]
    var bytes = List[UInt8](length=160, fill=0)
    var ptr = (
        bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    for pattern in patterns:
        var regex = Regex(pattern)
        ref program = regex._compiled[].forward
        var pc = 0
        while not consumes(program.op[pc]):
            pc += 1
        for byte in range(256):
            for i in range(len(bytes)):
                bytes[i] = UInt8(byte)
            for offset in range(16):
                var span = Span[UInt8, ImmutAnyOrigin](
                    unsafe_ptr=ptr.unsafe_offset(offset), length=129
                )
                var accepts = byte < 128 and program.step(pc, UInt32(byte))
                for size in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129]:
                    var distance = size // 16 * 16 if accepts else 0
                    assert_equal(
                        scan_ascii_run[False](program, pc, span, 0, size),
                        distance,
                    )
                    assert_equal(
                        scan_ascii_run[True](program, pc, span, size, 0),
                        size - distance,
                    )
    # A rejected byte in every vector lane, with a nonzero floor/end.
    var regex = Regex("(a+)")
    ref program = regex._compiled[].forward
    var pc = 0
    while not consumes(program.op[pc]):
        pc += 1
    for bad in range(1, 129):
        for i in range(len(bytes)):
            bytes[i] = 97
        bytes[bad] = 98
        var span = Span[UInt8, ImmutAnyOrigin](unsafe_ptr=ptr, length=130)
        assert_equal(scan_ascii_run[False](program, pc, span, 1, 129), bad)
        assert_equal(scan_ascii_run[True](program, pc, span, 129, 1), bad + 1)
    _ = bytes


def test_vector_capture_runs_match_vm_across_boundaries_and_priorities() raises:
    var patterns: List[String] = [
        "^(a+)(b*)",
        "(a+)(b*)",
        "^(a+?)(b*)",
        "^((a)+)(b*)",
        "^([ab]+)([!中]*)",
        "^(\\w+)(!*)",
        "^(a+)\\b(!*)",
        "^(a+)(\\b)",
        "(?s)^(.+)",
        "(?s)^(.+?)",
        "^([^\\n]+)(\\n*)",
        "^(a+)$",
        "^((a)+)$",
        "([a-z]+) ([0-9]+)",
    ]
    for pattern in patterns:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        var slots = List[Int](length=program.slot_count, fill=-1)
        for length in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]:
            for suffix in [String(""), "b", "!", "\n", "中", "𐐀", " 123"]:
                var text = "a" * length + suffix
                var span = _span(text)
                var expected = matcher.pike.search(
                    program, span, 0, len(span), False, slots
                )
                assert_equal(matcher.captures(span), expected, pattern)
                if expected:
                    assert_equal(matcher.slots, slots, pattern)
                _ = text
