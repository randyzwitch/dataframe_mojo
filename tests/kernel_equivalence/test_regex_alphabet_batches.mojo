"""Batched alphabet directories agree with the canonical interval partition."""
# contract-group: kernel.regex
from std.testing import assert_equal
from dataframe.regex import Regex
from dataframe.regex.alphabet_batch import alphabet_batch
from dataframe.regex.alphabet_lookup import build_alphabet_lookup
from dataframe.regex.program import Program, compile_program
from dataframe.regex.syntax import Ast, NODE_CLASS
from .test_regex_alphabet_lookup import _reference


def _check_alphabet_batches(mut program: Program) raises:
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
    points.extend([0x7F, 0x80, 0x7FF, 0x800, 0xD800, 0x110000, UInt32.MAX])
    for budget in [0, 16, 17, 272, 273, 1024, 4369]:
        build_alphabet_lookup(
            program.boundaries, program.interval_class, budget
        )
        for base in range(0, len(points), 8):
            var cp = SIMD[DType.uint32, 8]()
            for lane in range(8):
                cp[lane] = points[min(base + lane, len(points) - 1)]
            var actual = alphabet_batch[8](program, cp)
            for lane in range(8):
                assert_equal(
                    actual[lane],
                    Int32(
                        _reference(
                            program.boundaries, program.interval_class, cp[lane]
                        )
                    ),
                )
            var short = alphabet_batch[4](program, cp.slice[4]())
            for lane in range(4):
                assert_equal(short[lane], actual[lane])


def test_alphabet_batches_cover_pages_boundaries_trailing_and_budget_fallbacks() raises:
    for pattern in [
        String("literal"),
        "(\\w+) (\\w+)",
        "\\b(\\w+)\\b",
        "([一-丂]+)([𐐀-𐐂]*)",
        "([^丁]+)",
    ]:
        var regex = Regex(pattern)
        _check_alphabet_batches(regex._compiled[].forward)
        _check_alphabet_batches(regex._compiled[].reverse)


def test_alphabet_batches_handle_maximally_split_pages() raises:
    var ast = Ast()
    var ranges = List[UInt32]()
    for k in range(128):
        ranges.append(UInt32(0x1000 + 2 * k))
        ranges.append(UInt32(0x1000 + 2 * k))
    ast.classes.append(ranges^)
    ast.root = ast.add(NODE_CLASS, 0)
    var program = compile_program(ast, False)
    _check_alphabet_batches(program)
