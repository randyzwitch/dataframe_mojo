"""Public logical type identity, parsing, physical storage and sum result rules."""
# contract-group: types.dtype
from std.sys import size_of
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import AnyValue, DataType


def test_numeric_dtype_names_predicates_storage_and_sum_rules_are_consistent() raises:
    comptime numeric = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float32,
        DType.float64,
    )
    var names: List[String] = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ]
    var short: List[String] = [
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "f32",
        "f64",
    ]
    comptime for i in range(len(numeric)):
        comptime D = numeric[i]
        var dtype = DataType.of(D)
        assert_equal(dtype.name(), names[i])
        assert_equal(String(dtype), names[i])
        assert_equal(dtype.short_name(), short[i])
        assert_true(DataType.parse(names[i]) == dtype)
        assert_true(DataType.is_known(names[i]))
        assert_true(dtype.storage().value() == D)
        assert_equal(dtype.bit_width(), size_of[Scalar[D]]() * 8)
        assert_true(dtype.is_numeric())
        assert_equal(dtype.is_float(), D.is_floating_point())
        assert_equal(dtype.is_integer(), not D.is_floating_point())
        assert_equal(dtype.is_unsigned(), D.is_unsigned())
        assert_equal(dtype.is_signed(), i < 4 or i >= 8)
        var sum_type = (
            DataType.INT64 if i == 0
            or i == 1
            or i == 4
            or i == 5 else dtype.copy()
        )
        assert_true(dtype.sum_type() == sum_type)
        comptime for j in range(len(numeric)):
            assert_equal(dtype == DataType.of(numeric[j]), i == j)
            assert_equal(dtype != DataType.of(numeric[j]), i != j)


def test_logical_types_do_not_equal_their_physical_storage_and_nulls_keep_type() raises:
    assert_true(DataType.DATE != DataType.INT64)
    assert_true(DataType.BINARY != DataType.STRING)
    assert_true(not DataType.DATE.is_numeric())
    assert_true(not DataType.BOOL.is_numeric())
    assert_true(not DataType.STRING.is_numeric())
    assert_equal(DataType.BOOL.bit_width(), 1)
    assert_equal(DataType.STRING.bit_width(), 0)
    assert_true(not Bool(DataType.BOOL.storage()))
    assert_true(not Bool(DataType.STRING.storage()))
    for dtype in [
        DataType.DATE,
        DataType.datetime("ms"),
        DataType.duration("us"),
    ]:
        assert_true(dtype.storage().value() == DType.int64)
        assert_equal(dtype.bit_width(), 64)
    for width in [32, 64, 128]:
        var limit = 9 if width == 32 else 18 if width == 64 else 38
        var dtype = DataType.decimal(limit, 2, width)
        assert_true(DataType.parse(dtype.name()) == dtype)
        assert_equal(dtype.decimal_width(), width)
        assert_equal(dtype.precision(), limit)
        assert_equal(dtype.scale(), 2)
        assert_true(AnyValue.null(dtype).dtype() == dtype)
        assert_true(AnyValue.null(dtype).is_null())
        with assert_raises():
            _ = DataType.decimal(limit + 1, 2, width)
    assert_true(DataType.is_known("int128"))
    assert_true(DataType.parse("int128") == DataType.of(DType.int128))
    assert_equal(DataType.parse("int128").bit_width(), 128)
    with assert_raises(contains="Unknown dtype"):
        _ = DataType.parse("decimal")
