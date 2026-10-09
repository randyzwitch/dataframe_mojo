"""Lists on huge pages hold what the plain constructors would."""
from std.testing import TestSuite, assert_equal
from dataframe.huge_pages import advise_huge_pages, huge_list, huge_uninit


def test_huge_list_is_filled_at_every_size() raises:
    # Under one huge page (no advice), across one, and several.
    for n in [1000, 300_000, 1_200_000]:
        var values = huge_list(n, Int32(-1))
        assert_equal(len(values), n)
        var sum = 0
        for i in range(n):
            sum += Int(values[i])
        assert_equal(sum, -n)
        var words = huge_list(n, UInt64(7))
        assert_equal(len(words), n)
        assert_equal(words[0], UInt64(7))
        assert_equal(words[n // 2], UInt64(7))
        assert_equal(words[n - 1], UInt64(7))


def test_huge_uninit_has_the_length_and_takes_writes() raises:
    for n in [0, 5000, 600_000]:
        var values = huge_uninit[Int](n)
        assert_equal(len(values), n)
        for i in range(n):
            values[i] = i
        if n > 0:
            assert_equal(values[n - 1], n - 1)


def test_advice_on_short_or_unaligned_ranges_is_harmless() raises:
    var values = List[UInt8](length=5 << 20, fill=1)
    var address = Int(values.unsafe_ptr())
    advise_huge_pages(address + 1, 0)
    advise_huge_pages(address + 1, 100)
    advise_huge_pages(address + 1, (5 << 20) - 2)
    advise_huge_pages(address + 4096 * 3 + 7, 3 << 20)
    var sum = 0
    for i in range(0, len(values), 4093):
        sum += Int(values[i])
    assert_equal(sum, (len(values) + 4092) // 4093)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
