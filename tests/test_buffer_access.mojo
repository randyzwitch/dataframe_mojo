"""Public Arrow buffer access: a consumer reads values and validity directly."""
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from dataframe import BoolColumn, Column, DataFrame, Series, StringColumn


def _bit_at(bits: Pointer[UInt8, MutAnyOrigin], i: Int) -> Bool:
    """Read an Arrow validity bit the way a consumer would."""
    return (bits.unsafe_offset(i >> 3)[] >> UInt8(i & 7)) & 1 == 1


def test_values_pointer_reads_the_window() raises:
    var c = Column[Float64]([1.5, 2.5, 3.5, 4.5])
    var values = c.unsafe_values()
    assert_equal(values[], Float64(1.5))
    assert_equal(values.unsafe_offset(3)[], Float64(4.5))
    # A slice shifts the values pointer to its own row 0.
    var window = c.slice(2, 2)
    assert_equal(window.unsafe_values()[], Float64(3.5))
    assert_equal(len(window), 2)


def test_validity_is_indexed_from_the_window_offset() raises:
    var c = Column[Int64]([1, 2, 3, 4, 5], [True, False, True, True, False])
    assert_equal(c.null_count(), 2)
    var bits = c.unsafe_validity()
    assert_equal(c.validity_offset(), 0)
    assert_true(_bit_at(bits, 0))
    assert_false(_bit_at(bits, 1))
    # The bitmap is NOT shifted to the window; the offset says where row 0 is.
    var window = c.slice(1, 3)
    assert_equal(window.validity_offset(), 1)
    var wbits = window.unsafe_validity()
    var base = window.validity_offset()
    assert_false(_bit_at(wbits, base + 0))  # original row 1
    assert_true(_bit_at(wbits, base + 1))  # original row 2
    assert_equal(window.null_count(), 1)


def test_is_valid_matches_the_bitmap() raises:
    var c = Column[Int64]([1, 2, 3], [True, False, True])
    var bits = c.unsafe_validity()
    for i in range(len(c)):
        assert_equal(c.is_valid(i), _bit_at(bits, c.validity_offset() + i))
    assert_true(c.is_valid(0))
    assert_false(c.is_valid(1))


def test_buffers_are_shared_not_copied() raises:
    var df = DataFrame([Series("a", Column[Int64]([0, 1, 2, 3, 4, 5]))])
    var whole = df.column("a").numeric[DType.int64]()
    var window = df.slice(2, 3).column("a").numeric[DType.int64]()
    # Same underlying allocation: the window points into the original buffer.
    var base = Int(whole.unsafe_values().unsafe_offset(2))
    assert_equal(Int(window.unsafe_values()), base)


def test_to_list_keeps_null_slots_as_stored() raises:
    var c = Column[Int64]([10, 20, 30], [True, False, True])
    var values = c.to_list()
    assert_equal(len(values), 3)
    assert_equal(values[0], Int64(10))
    assert_equal(values[2], Int64(30))
    # The null slot carries its stored payload, as Arrow leaves it undefined.
    assert_equal(values[1], Int64(20))
    assert_false(c.is_valid(1))


def test_bool_values_are_a_bitmap() raises:
    var b = BoolColumn([True, False, True, True], [True, True, False, True])
    var values = b.unsafe_values()
    assert_true(_bit_at(values, b.validity_offset() + 0))
    assert_false(_bit_at(values, b.validity_offset() + 1))
    assert_true(_bit_at(values, b.validity_offset() + 3))
    assert_false(b.is_valid(2))
    assert_equal(b.true_count(), 3)


def test_string_bytes_and_offsets() raises:
    var s = StringColumn(Column[String](["ab", "", "xyz"]))
    var offsets = s.unsafe_offsets()
    var bytes = s.unsafe_bytes()
    var base = s.validity_offset()
    assert_equal(offsets.unsafe_offset(base)[], Int64(0))
    assert_equal(offsets.unsafe_offset(base + 1)[], Int64(2))
    assert_equal(offsets.unsafe_offset(base + 3)[], Int64(5))
    # Row 0 is "ab".
    assert_equal(bytes[], UInt8(ord("a")))
    assert_equal(bytes.unsafe_offset(1)[], UInt8(ord("b")))
    assert_equal(s.to_list(), ["ab", "", "xyz"])


def test_empty_column_has_a_length_of_zero() raises:
    var c = Column[Int64]([])
    assert_equal(len(c), 0)
    assert_equal(c.null_count(), 0)
    assert_equal(len(c.to_list()), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
