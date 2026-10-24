"""Cached programs remain valid across reuse, eviction and concurrent callers."""
# contract-group: kernel.regex
from std.memory import Pointer
from std.testing import assert_equal, assert_true, assert_raises

from dataframe.parallel import Job, run_jobs
from dataframe.regex import Matcher, Regex
from dataframe.regex_cache import _PatternCache, cached_pattern
from dataframe.regex_kernels import compile_pattern


def _address(regex: Regex) -> Int:
    return Int(Pointer(to=regex._compiled[]))


def _matches(regex: Regex, text: String) -> Bool:
    var matcher = Matcher(regex)
    var span = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=text.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        length=text.byte_length(),
    )
    var matched = matcher.is_match(span)
    _ = text
    return matched


def test_cached_patterns_reuse_and_evict_without_invalidating_callers() raises:
    var cache = _PatternCache()
    assert_true(cache.ready)
    var first = cache.compile("^value-0$")
    var second = cache.compile("^value-1$")
    for i in range(2, 32):
        _ = cache.compile("^value-" + String(i) + "$")
    var reused = cache.compile("^value-0$")
    assert_equal(_address(first), _address(reused))
    _ = cache.compile("^value-32$")
    assert_equal(len(cache.entries), 32)
    var still_cached = cache.compile("^value-0$")
    assert_equal(_address(first), _address(still_cached))
    var rebuilt = cache.compile("^value-1$")
    assert_true(_address(second) != _address(rebuilt))
    assert_true(_matches(second, "value-1"))
    assert_true(_matches(first, "value-0"))
    assert_equal(len(cache.entries), 32)


def test_invalid_cached_patterns_release_the_lock_and_keep_valid_entries() raises:
    var cache = _PatternCache()
    var first = cache.compile("^valid$")
    for _ in range(3):
        with assert_raises():
            _ = cache.compile("(")
        var reused = cache.compile("^valid$")
        assert_equal(_address(first), _address(reused))
    assert_equal(len(cache.entries), 1)


def test_cached_literal_and_regex_patterns_use_the_same_canonical_program() raises:
    var literal = compile_pattern("a.b", True)
    var escaped = compile_pattern("a\\.b", False)
    assert_equal(_address(literal), _address(escaped))
    assert_true(_matches(literal, "a.b"))
    assert_true(not _matches(literal, "axb"))
    var regex = compile_pattern("a.b", False)
    assert_true(_address(literal) != _address(regex))
    assert_true(_matches(regex, "axb"))


@fieldwise_init
struct _CompileJob(Job):
    var id: Int
    var checked: Int

    def run(mut self) raises:
        for i in range(64):
            var shared = cached_pattern("^([a-z]+)-(\\d+)$")
            assert_true(_matches(shared, "word-123"))
            var text = "worker-" + String(self.id) + "-" + String(i)
            var unique = cached_pattern("^" + text + "$")
            assert_true(_matches(unique, text))
            self.checked += 1


def test_process_pattern_cache_handles_concurrent_hits_misses_and_evictions() raises:
    var jobs = List[_CompileJob]()
    for i in range(8):
        jobs.append(_CompileJob(i, 0))
    run_jobs(jobs)
    for i in range(len(jobs)):
        assert_equal(jobs[i].checked, 64)
