"""Alphabet colors represent exactly the predicates that can distinguish input."""
# contract-group: kernel.regex
from std.testing import assert_equal
from dataframe.regex import Regex
from dataframe.regex.program import Program, consumes
from dataframe.regex.pikevm import _ascii_word
from dataframe.regex.syntax import LOOK_ASCII, LOOK_UNICODE_WORD


def _check_partition(program: Program) raises:
    # A separate reference evaluates every instruction at each cut. It
    # does not use the builder's range visits or intermediate colors.
    var ids = Dict[String, Int]()
    var representatives = List[UInt32]()
    for i in range(len(program.boundaries)):
        var cp = program.boundaries[i]
        var signature = String()
        for pc in range(len(program)):
            if consumes(program.op[pc]):
                signature += "1" if program.step(pc, cp) else "0"
        signature += "1" if cp == 10 else "0"
        if program.looks & LOOK_UNICODE_WORD:
            signature += "1" if program.is_word(cp) else "0"
        if program.looks & LOOK_ASCII:
            signature += "1" if _ascii_word(cp) else "0"
        var expected = ids.get(signature, -1)
        if expected < 0:
            expected = len(representatives)
            ids[signature] = expected
            representatives.append(cp)
        assert_equal(Int(program.interval_class[i]), expected)
        assert_equal(program.class_of(cp), expected)
        if i + 1 < len(program.boundaries):
            assert_equal(
                program.class_of(program.boundaries[i + 1] - 1), expected
            )
    assert_equal(program.class_rep, representatives)
    assert_equal(program.alphabet, len(representatives))


def test_alphabet_partition_refinement_matches_instruction_membership() raises:
    var patterns: List[String] = [
        "",
        "(abacaba)",
        "([a-d]|[b-e]|[ab])",
        "\\b(\\w+)\\b",
        "(?-u:\\b)([\\p{Han}\\p{Greek}]+)",
        "(?s)(.+)",
    ]
    for pattern in patterns:
        var regex = Regex(pattern)
        _check_partition(regex._compiled[].forward)
        _check_partition(regex._compiled[].reverse)


def test_alphabet_partition_refinement_handles_many_overlapping_sets() raises:
    for count in [1, 8, 63, 64, 65, 100]:
        var pattern = String("(")
        for i in range(count):
            # Distinct nested ranges all contain newline, so the final
            # newline set must split their common intersection as well.
            pattern += "[\\u{0}-\\u{" + String(300 + i) + "}]"
        pattern += ")"
        var regex = Regex(pattern)
        assert_equal(len(regex._compiled[].forward.class_ranges), count)
        _check_partition(regex._compiled[].forward)
        _check_partition(regex._compiled[].reverse)
