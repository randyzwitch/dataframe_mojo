"""Record boundaries for parallel CSV decoding.

Splitting a buffer for parallel decoding is only safe at a record start, and
a newline inside a quoted field is data, not a boundary. Getting this wrong
mis-splits a frame silently rather than raising, so these tests check the
boundary finder directly, including quotes placed at every offset relative
to a split target.
"""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe.csv import record_splits


def splits(text: String, parts: Int, quoting: Bool = True) raises -> List[Int]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    var span = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(bytes),
    )
    var found = record_splits(span, UInt8(34), quoting, parts)
    var offsets = List[Int]()
    for s in found.splits:
        offsets.append(s.offset)
    _ = bytes^
    return offsets^


def records(text: String, parts: Int) raises -> List[Int]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    var span = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(bytes),
    )
    var found = record_splits(span, UInt8(34), True, parts)
    var counts = List[Int]()
    for s in found.splits:
        counts.append(s.record)
    _ = bytes^
    return counts^


def test_first_is_zero_and_last_is_the_length() raises:
    var text = String("a\nb\nc\nd\n")
    var got = splits(text, 4)
    assert_equal(got[0], 0)
    assert_equal(got[len(got) - 1], text.byte_length())
    # Strictly increasing, so no range is empty.
    for i in range(1, len(got)):
        assert_true(got[i] > got[i - 1], "splits must strictly increase")


def test_every_split_starts_a_record() raises:
    var text = String("aa\nbb\ncc\ndd\nee\nff\n")
    var got = splits(text, 3)
    var bytes = text.as_bytes()
    for i in range(1, len(got) - 1):
        assert_true(
            bytes[got[i] - 1] == 10,
            "a split must fall just after a newline",
        )


def test_newline_inside_quotes_is_not_a_boundary() raises:
    # The only unquoted newlines are the two record terminators.
    var text = String('a,"x\ny"\nb,"p\nq"\n')
    var got = splits(text, 8)
    for i in range(1, len(got) - 1):
        assert_true(
            got[i] == 8, "the only interior boundary is after the first record"
        )


def test_doubled_quotes_do_not_break_parity() raises:
    # "" is an escaped quote, so the newline after it is still inside.
    var text = String('a,"x""\ny"\nb,2\n')
    var got = splits(text, 6)
    var bytes = text.as_bytes()
    for i in range(1, len(got) - 1):
        assert_true(bytes[got[i] - 1] == 10)
        # Offset 10 is the record terminator; nothing before it is a boundary.
        assert_true(got[i] >= 10)


def test_quotes_at_every_offset_relative_to_the_target() raises:
    # A quoted field containing a newline, slid across the buffer one byte at
    # a time, so it straddles the split target in every possible way.
    for pad in range(0, 24):
        var head = String()
        for _ in range(pad):
            head += "z"
        var text = head + ',"a\nb"\n' + "tail,1\n"
        var got = splits(text, 2)
        var bytes = text.as_bytes()
        for i in range(1, len(got) - 1):
            assert_true(
                bytes[got[i] - 1] == 10,
                "split not after a newline at pad " + String(pad),
            )
            # The newline inside the quotes sits at pad + 3.
            assert_true(
                got[i] != pad + 4,
                "split landed inside a quoted field at pad " + String(pad),
            )


def test_crlf_splits_after_the_newline() raises:
    var text = String("a,1\r\nb,2\r\nc,3\r\n")
    var got = splits(text, 3)
    var bytes = text.as_bytes()
    for i in range(1, len(got) - 1):
        assert_equal(bytes[got[i] - 1], UInt8(10))
        assert_equal(bytes[got[i] - 2], UInt8(13))


def test_record_counts_are_cumulative() raises:
    var counts = records("a\nb\nc\nd\ne\nf\n", 3)
    assert_equal(counts[0], 0)
    for i in range(1, len(counts)):
        assert_true(counts[i] >= counts[i - 1])
    # Six terminated records in total.
    assert_equal(counts[len(counts) - 1], 6)


def test_unterminated_quote_yields_one_range() raises:
    # No even-parity newline exists, so the whole buffer is one range rather
    # than a hang or a bad split.
    var got = splits('a,"x\ny\nz\n', 4)
    assert_equal(len(got), 2)
    assert_equal(got[0], 0)


def test_quoting_disabled_treats_quotes_as_data() raises:
    var text = String('a,"x\ny"\nb,2\n')
    var enabled = splits(text, 8)
    var disabled = splits(text, 8, quoting=False)
    # With quoting off the embedded newline becomes a boundary, so there are
    # strictly more places to split.
    assert_true(len(disabled) > len(enabled))


def test_empty_and_single_part() raises:
    assert_equal(len(splits("", 4)), 2)
    var one = splits("a\nb\n", 1)
    assert_equal(len(one), 2)
    assert_equal(one[1], 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
