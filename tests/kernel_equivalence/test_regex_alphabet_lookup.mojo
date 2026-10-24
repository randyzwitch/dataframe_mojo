"""Wide alphabet directories preserve canonical classes and fallback behavior."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_true
from dataframe.regex import Matcher, Regex
from dataframe.regex.alphabet_lookup import build_alphabet_lookup
from dataframe.regex.program import Program, compile_program
from dataframe.regex.syntax import Ast, NODE_CLASS


def _reference(points: List[UInt32], classes: List[Int32], cp: UInt32) -> Int:
    var lo = 0
    var hi = len(points)
    while hi - lo > 1:
        var mid = (lo + hi) >> 1
        if points[mid] <= cp:
            lo = mid
        else:
            hi = mid
    return Int(classes[lo])


def _check_program(mut program: Program) raises:
    var original = program.interval_class.copy()
    original.resize(len(program.boundaries), 0)
    var points = List[UInt32]()
    for page in range(0x1100):
        points.append(UInt32(page << 8))
        points.append(UInt32((page << 8) + 127))
        points.append(UInt32((page << 8) + 255))
    for boundary in program.boundaries:
        points.append(boundary)
        if boundary > 0:
            points.append(boundary - 1)
        if boundary < 0x10FFFF:
            points.append(boundary + 1)
    points.append(0x110000)
    points.append(UInt32.MAX)
    for budget in [0, 16, 17, 272, 273, 1024, 4369]:
        build_alphabet_lookup(
            program.boundaries, program.interval_class, budget
        )
        assert_true(
            len(program.interval_class) <= len(program.boundaries) + budget
        )
        for cp in points:
            assert_equal(
                program.class_of(cp),
                _reference(program.boundaries, original, cp),
            )
        for i in range(len(program.boundaries)):
            assert_equal(program.interval_class[i], original[i])


def test_wide_alphabet_lookup_matches_canonical_classes_and_budgets() raises:
    var patterns: List[String] = [
        "literal",
        "(\\w+) (\\w+)",
        "\\b(\\w+)\\b",
        "([一-丂]+)([𐐀-𐐂]*)",
        "([^丁]+)",
        "([\\p{Greek}\\p{Cyrillic}\\p{Han}]+)",
    ]
    for pattern in patterns:
        var regex = Regex(pattern)
        _check_program(regex._compiled[].forward)
        _check_program(regex._compiled[].reverse)


def test_wide_alphabet_lookup_handles_a_maximally_split_page() raises:
    var ast = Ast()
    var ranges = List[UInt32]()
    for i in range(128):
        ranges.append(UInt32(0x1000 + 2 * i))
        ranges.append(UInt32(0x1000 + 2 * i))
    ast.classes.append(ranges^)
    ast.root = ast.add(NODE_CLASS, 0)
    var program = compile_program(ast, False)
    _check_program(program)


def test_wide_alphabet_dfa_captures_match_independent_vm() raises:
    var patterns: List[String] = [
        "(\\w+) (\\w+) (\\w+)",
        "\\b(\\w+)\\b",
        "([\\p{Han}\\p{Deseret}]+)([!?]*)",
    ]
    var texts: List[String] = [
        "!中文 日本 文字!",
        "𐐀𐐁 𐐂𐐃 𐐄𐐅",
        "αβ 中𐐀 aβ",
        "no-match",
        "!一𐐀中𐐂!?",
    ]
    for pattern in patterns:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        var expected = List[Int](length=program.slot_count, fill=-1)
        for text in texts:
            var span = Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=text.unsafe_ptr().unsafe_origin_cast[
                    ImmutAnyOrigin
                ](),
                length=text.byte_length(),
            )
            for start in [0, 1]:
                if start == 1 and span[0] >= 128:
                    continue
                var found = matcher.pike.search(
                    program, span, start, len(span), False, expected
                )
                assert_equal(matcher.captures(span, start), found, pattern)
                if found:
                    for k in range(len(expected)):
                        assert_equal(matcher.slots[k], expected[k], pattern)
            _ = text
