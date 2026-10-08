"""Shape record dimensions, ring grouping and checked malformed inputs."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Series, from_wkb
from dataframe.shapefile import _shape
from dataframe.shapefile_binary import put_uint, put_number, text, encoding_name


def point(kind: Int, z: Bool = False, m: Bool = False) -> List[UInt8]:
    var bytes = List[UInt8]()
    put_uint(bytes, kind)
    put_number(bytes, 1)
    put_number(bytes, 2)
    if z:
        put_number(bytes, 3)
    if m:
        put_number(bytes, 4)
    return bytes^


def typename(bytes: List[UInt8], declared: Int) raises -> String:
    var wkb = _shape(bytes, declared)
    return from_wkb(Series.binary("g", [wkb^])).geometry_type().get(0).string()


def test_optional_measure_dimensions() raises:
    assert_equal(typename(point(1), 1), "Point")
    assert_equal(typename(point(11, True), 11), "Point Z")
    assert_equal(typename(point(11, True, True), 11), "Point ZM")
    assert_equal(typename(point(21, m=True), 21), "Point M")
    assert_equal(typename(point(21), 21), "Point M")
    var no_measure = point(11, True)
    put_number(no_measure, -1e39)
    assert_equal(typename(no_measure, 11), "Point Z")


def test_malformed_record_lengths_and_types() raises:
    with assert_raises(contains="truncated"):
        _ = _shape(List[UInt8]([UInt8(1), 0]), 1)
    with assert_raises(contains="truncated"):
        _ = _shape(point(11), 11)
    var trailing = point(1)
    trailing.append(0)
    with assert_raises(contains="trailing"):
        _ = _shape(trailing, 1)
    with assert_raises(contains="MultiPatch"):
        _ = _shape(point(31), 31)
    with assert_raises(contains="Unsupported"):
        _ = _shape(point(99), 99)
    with assert_raises(contains="differs"):
        _ = _shape(point(1), 11)
    var null = List[UInt8]([UInt8(0), 0, 0, 0])
    assert_equal(len(_shape(null, 1)), 0)
    null.append(0)
    with assert_raises(contains="trailing"):
        _ = _shape(null, 1)


def test_encoding_overrides_and_invalid_bytes() raises:
    assert_equal(encoding_name("UTF-8"), "utf8")
    assert_equal(
        text(List[UInt8]([UInt8(99), 97, 102, 233]), 0, 4, "latin1"), "café"
    )
    assert_equal(text(List[UInt8]([UInt8(128)]), 0, 1, "cp1252"), "€")
    with assert_raises():
        _ = text(List[UInt8]([UInt8(255)]), 0, 1, "utf8")
    with assert_raises(contains="Undefined"):
        _ = text(List[UInt8]([UInt8(129)]), 0, 1, "cp1252")
    with assert_raises(contains="ASCII"):
        _ = text(List[UInt8]([UInt8(233)]), 0, 1, "ascii")
    with assert_raises(contains="Unsupported DBF encoding"):
        _ = encoding_name("not-an-encoding")


def test_part_offsets_checked_before_coordinates() raises:
    var record = List[UInt8]()
    put_uint(record, 3)
    for _ in range(4):
        put_number(record, 0)
    put_uint(record, 1)
    put_uint(record, 2)
    put_uint(record, 3)
    with assert_raises(contains="part offsets"):
        _ = _shape(record, 3)
    record = List[UInt8]()
    put_uint(record, 8)
    for _ in range(4):
        put_number(record, 0)
    put_uint(record, 2147483647)
    with assert_raises(contains="truncated"):
        _ = _shape(record, 8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
