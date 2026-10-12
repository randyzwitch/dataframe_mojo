"""Ordered schemas and name validation, independently of storage implementation."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series


def test_schema_names_heights_and_simultaneous_renames() raises:
    var frame = DataFrame(
        [
            Series("", Column[Int64]([1, 2])),
            Series("A", Column[String](["x", "y"])),
            Series("a", Column[Bool]([True, False])),
        ],
        height=2,
    )
    assert_equal(frame.columns(), ["", "A", "a"])
    assert_equal(frame.shape(), (2, 3))
    assert_equal(frame.select(List[String]()).shape(), (2, 0))
    assert_equal(DataFrame(List[Series]()).shape(), (0, 0))
    assert_equal(DataFrame(List[Series](), height=7).shape(), (7, 0))
    var swapped = frame.rename({"A": "a", "a": "A"})
    assert_equal(swapped.columns(), ["", "a", "A"])
    assert_equal(swapped.item(0, "a").string(), "x")
    assert_equal(swapped.item(1, "A").bool(), False)
    assert_equal(frame.drop(["A", "a"]).columns(), [""])
    var invalid_names: List[List[String]] = [["missing"], ["A", "A"]]
    for names in invalid_names:
        with assert_raises():
            _ = frame.drop(names)
    for mapping in [{"missing": "b"}, {"a": "A"}]:
        with assert_raises():
            _ = frame.rename(mapping)
    with assert_raises():
        _ = frame.select(["missing"])
    with assert_raises():
        _ = DataFrame(
            [Series("x", Column[Int64]([1])), Series("x", Column[Int64]([2]))]
        )
    with assert_raises():
        _ = DataFrame(
            [
                Series("x", Column[Int64]([1])),
                Series("y", Column[Int64]([2, 3])),
            ]
        )
    with assert_raises():
        _ = DataFrame([Series("x", Column[Int64]([1]))], height=2)
    with assert_raises():
        _ = DataFrame(List[Series](), height=-2)
    with assert_raises():
        _ = frame.with_column(Series("bad", Column[Int64]([1])))
    var indexed = frame.with_row_index("row", 11)
    assert_equal(indexed.columns(), ["row", "", "A", "a"])
    assert_equal(indexed.item(0, "row").int64(), Int64(11))
    assert_equal(indexed.item(1, "row").int64(), Int64(12))
    with assert_raises():
        _ = frame.with_row_index("A")


def test_all_logical_type_schemas_survive_empty_projection_and_clear() raises:
    var types: List[DataType] = [
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.INT64,
        DataType.UINT8,
        DataType.UINT16,
        DataType.UINT32,
        DataType.UINT64,
        DataType.FLOAT32,
        DataType.FLOAT64,
        DataType.BOOL,
        DataType.STRING,
        DataType.BINARY,
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("ns", "UTC"),
        DataType.datetime("us"),
        DataType.datetime("ms"),
        DataType.duration("ns"),
        DataType.duration("us"),
        DataType.duration("ms"),
        DataType.decimal(9, 2, 32),
        DataType.decimal(18, 4, 64),
        DataType.decimal(38, 8),
        DataType.list(DataType.INT64),
        DataType.list(DataType.list(DataType.STRING)),
        DataType.struct(["x", "y"], [DataType.INT64, DataType.STRING]),
        DataType.geometry(),
    ]
    var columns = List[Series]()
    var names = List[String]()
    for index in range(len(types)):
        names.append("column_" + String(index))
        columns.append(Series.full_null(names[index], types[index], 3))
    columns.append(
        Series("category", Column[String](["z", "a", "z"])).cast(
            DataType.CATEGORICAL
        )
    )
    names.append("category")
    var categorical_type = columns[len(columns) - 1].dtype()
    assert_true(categorical_type.is_categorical())
    assert_equal(String(categorical_type.dictionary()[].get(0)), "z")
    assert_equal(String(categorical_type.dictionary()[].get(1)), "a")
    types.append(categorical_type)
    var frame = DataFrame(columns^)
    assert_equal(frame.columns(), names)
    assert_equal(frame.dtypes(), types)
    var schema = frame.schema()
    for index in range(len(types)):
        assert_equal(schema[index].name, names[index])
        assert_true(schema[index].dtype == types[index])
        assert_true(frame.column(names[index]).dtype() == types[index])
        assert_true(
            frame.slice(1, 1).column(names[index]).dtype() == types[index]
        )
        assert_true(frame.clear().column(names[index]).dtype() == types[index])
        if index < len(types) - 1:
            for row in range(3):
                assert_true(frame.item(row, names[index]).is_null())
                assert_true(
                    frame.item(row, names[index]).dtype() == types[index]
                )
    assert_equal(frame.select(List[String]()).height(), 3)
    assert_equal(frame.clear().shape(), (0, len(types)))
