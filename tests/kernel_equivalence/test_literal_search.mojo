"""The literal search behind LIKE, contains, replace and split finds every
first occurrence at every haystack length, needle length and start, across
the 64-, 32- and 16-byte vector steps, the overlapped final vector and the
scalar tail."""
# contract-group: kernel.regex
from std.testing import assert_equal

from dataframe.string_bytes import _find_bytes


def _naive(hay: List[UInt8], needle: List[UInt8], start: Int) -> Int:
    var n = len(hay)
    var m = len(needle)
    if m == 0:
        return start if start <= n else -1
    var at = start
    while at + m <= n:
        var same = True
        for k in range(m):
            if hay[at + k] != needle[k]:
                same = False
                break
        if same:
            return at
        at += 1
    return -1


def _span(values: List[UInt8]) -> Span[UInt8, ImmutAnyOrigin]:
    return Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=values.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(values),
    )


def test_find_bytes_matches_a_naive_search_at_every_length_and_start() raises:
    # Three symbols make partial matches (first and last byte equal, middle
    # different) common, so candidate verification is exercised too.
    var state = UInt64(88172645463325252)
    var needle_lengths: List[Int] = [0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 40]
    var starts: List[Int] = [0, 1, 5, 31, 64]
    for length in range(0, 200):
        var hay = List[UInt8]()
        for _ in range(length):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            hay.append(UInt8(97 + Int(state % 3)))
        for m in needle_lengths:
            # A needle cut from the haystack, so it occurs somewhere; and
            # one that does not (a symbol outside the alphabet at its end).
            var cut = List[UInt8]()
            var from_row = (length * 7 + m) % max(length - m + 1, 1)
            for k in range(m):
                cut.append(hay[from_row + k] if from_row + k < length else 97)
            var missing = cut.copy()
            if m > 0:
                missing[m - 1] = 122
            for start in starts:
                assert_equal(
                    _find_bytes(_span(hay), _span(cut), start),
                    _naive(hay, cut, start),
                )
                assert_equal(
                    _find_bytes(_span(hay), _span(missing), start),
                    _naive(hay, missing, start),
                )
