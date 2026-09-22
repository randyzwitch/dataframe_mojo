"""Focused Polars binary-view string layout and ownership checks."""
from std.sys import size_of
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from dataframe.string_view import (
    STRING_VIEW_INLINE_BYTES,
    StringView,
    StringViewBuilder,
)


def append(mut builder: StringViewBuilder, text: String):
    builder.append(StringSlice(text))


def test_inline_views_cover_all_lengths_and_16_byte_layout() raises:
    assert_equal(size_of[StringView](), 16)
    var builder = StringViewBuilder(13)
    for length in range(STRING_VIEW_INLINE_BYTES + 1):
        append(builder, String("x") * length)
    var storage = builder^.finish()
    assert_equal(len(storage), 13)
    assert_equal(storage.buffer_count(), 0)
    for length in range(STRING_VIEW_INLINE_BYTES + 1):
        var view = storage.view(length)
        assert_true(view.is_inline())
        assert_equal(view.length, UInt32(length))
        assert_equal(String(storage.get(length)), String("x") * length)

    var encoded = StringViewBuilder()
    append(encoded, "abcdefgh")
    var one = encoded^.finish().view(0)
    # Arrow View stores inline bytes at physical offsets 4..15, little-endian
    # within each of the three UInt32 words.
    assert_equal(one.prefix, UInt32(0x64636261))
    assert_equal(one.buffer_index, UInt32(0x68676665))
    assert_equal(one.offset, UInt32(0))


def test_long_views_borrow_blocks_and_null_validity_is_lazy() raises:
    var builder = StringViewBuilder(4)
    append(builder, "short")
    builder.append_null()
    append(builder, "thirteen-bytes")
    append(builder, 'quoted, "field"')
    var storage = builder^.finish()
    assert_equal(storage.null_count(), 1)
    assert_true(storage.valid(0))
    assert_false(storage.valid(1))
    assert_true(storage.is_valid(2))
    assert_equal(String(storage.get(0)), "short")
    assert_equal(String(storage.get(1)), "")
    assert_equal(String(storage.get(2)), "thirteen-bytes")
    assert_equal(String(storage.get(3)), 'quoted, "field"')
    var view = storage.view(2)
    assert_false(view.is_inline())
    assert_equal(view.length, UInt32(14))
    assert_equal(view.prefix, UInt32(0x72696874))  # "thir"
    assert_equal(view.buffer_index, UInt32(0))
    assert_equal(view.offset, UInt32(0))
    var bits = storage.validity_arc()
    assert_equal(bits[][0], UInt8(0b00001101))


def test_block_flush_preserves_prior_views_without_payload_copy() raises:
    var first = String("a") * 5000
    var second = String("b") * 5000
    var third = String("c") * 5000
    var builder = StringViewBuilder()
    append(builder, first)
    append(builder, second)  # 10 KiB exceeds initial 8 KiB block.
    append(builder, third)
    var storage = builder^.finish()
    assert_equal(storage.buffer_count(), 2)
    var buffers = storage.buffers_arc()
    # Polars starts at 8 KiB and doubles to 16 KiB before applying its cap.
    assert_equal(buffers[][0][].capacity(), 8 * 1024)
    assert_equal(buffers[][1][].capacity(), 16 * 1024)
    assert_equal(String(storage.get(0)), first)
    assert_equal(String(storage.get(1)), second)
    assert_equal(String(storage.get(2)), third)
    assert_equal(storage.view(0).buffer_index, UInt32(0))
    assert_equal(storage.view(1).buffer_index, UInt32(1))
    assert_equal(storage.view(2).buffer_index, UInt32(1))
    assert_equal(storage.view(2).offset, UInt32(5000))
    # A copied storage value retains the exact same Arc-owned byte blocks.
    var copied = storage.copy()
    var left = storage.buffers_arc()
    var right = copied.buffers_arc()
    assert_equal(Int(left[][0][].unsafe_ptr()), Int(right[][0][].unsafe_ptr()))
    assert_equal(Int(left[][1][].unsafe_ptr()), Int(right[][1][].unsafe_ptr()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
