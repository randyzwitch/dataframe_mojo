"""Capture engine selection and replacement preserve the public behavior."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_false, assert_true
from dataframe.parallel import run_jobs
from dataframe.regex import Matcher, Regex
from .test_regex_engine import _CaptureJob, _replace, _span


def test_reverse_capture_path_leaves_reverse_dfa_and_vm_buffers_unused() raises:
    var regex = Regex("(\\w+) (\\w+) (\\w+)")
    assert_true(regex._compiled[].reverse_captures)
    var matcher = Matcher(regex)
    var text = String("alpha beta gamma")
    assert_true(matcher.captures(_span(text)))
    var expected: List[Int] = [0, 16, 0, 5, 6, 10, 11, 16]
    assert_equal(matcher.slots, expected)
    assert_true(matcher.forward.state_count() > 0)
    assert_equal(matcher.reverse.state_count(), 0)
    assert_equal(len(matcher.reverse.table), 0)
    assert_equal(len(matcher.pike.current.dense), 0)
    _ = text
    for pattern in [
        String("^(a+)"),
        String("(abc)"),
        String("(\\w)"),
        String("(a*)(a*)"),
    ]:
        var other = Regex(pattern)
        assert_false(other._compiled[].reverse_captures, pattern)


def test_reverse_capture_replacement_groups_and_empty_iteration() raises:
    var patterns: List[String] = [
        "(a+)(b*)",
        "((a)+)(b*)",
        "(a*)()",
        "(a?)",
        "(\\w+)",
        "(?m)^(a+)",
        "(?s)(.*)",
    ]
    var texts: List[String] = [
        "xaaabb yaa",
        "aaaab!",
        "ab",
        "b",
        "中 a",
        "aa\naaa",
        "aa\n",
    ]
    var replacements: List[String] = [
        "$2-$1",
        "$1:$2:$3",
        "[$1][$2]",
        "[$1]",
        "<$1>",
        "<$1>",
        "<$1>",
    ]
    var expected: List[String] = [
        "xbb-aaa y-aa",
        "aaaa:a:b!",
        "[a][]b[][]",
        "[]b[]",
        "<中> <a>",
        "<aa>\n<aaa>",
        "<aa\n>",
    ]
    for i in range(len(patterns)):
        var regex = Regex(patterns[i])
        assert_true(regex._compiled[].reverse_captures, patterns[i])
        assert_equal(
            _replace(patterns[i], texts[i], replacements[i]),
            expected[i],
            patterns[i],
        )


def test_reverse_capture_tables_shared_by_workers() raises:
    var regex = Regex("([a-z]+) ([0-9]+)")
    assert_true(regex._compiled[].reverse_captures)
    var jobs = List[_CaptureJob]()
    for k in range(8):
        jobs.append(_CaptureJob(regex, k * 500, (k + 1) * 500))
    run_jobs(jobs)
    var count = 0
    for k in range(len(jobs)):
        count += jobs[k].count
    assert_equal(count, 4000)
