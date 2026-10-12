"""Nested values have separate parent and child validity and ordered contents."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    AnyValue,
    Column,
    DataFrame,
    DataType,
    Expr,
    ListColumn,
    Series,
    StructColumn,
    to_csv_string,
    as_struct,
    col,
    lit,
    when,
)
from tests.support.coverage_generators import chunk_layout, offset_layout

comptime NESTED_NUMERIC_DTYPES = (
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


def layout_for_nested(frame: DataFrame, layout: Int) raises -> DataFrame:
    return (
        frame.copy() if layout
        == 0 else offset_layout(frame) if layout
        == 1 else chunk_layout(frame)
    )


def integer_lists() raises -> DataFrame:
    # Nonzero initial offset and poisoned child slots deliberately make
    # neither physical offset nor parent validity equivalent to child validity.
    return DataFrame(
        [
            Series("id", Column[Int64]([0, 1, 2, 3, 4])),
            Series(
                "xs",
                ListColumn(
                    [1, 4, 4, 6, 8, 9],
                    Series(
                        "child",
                        Column[Int64](
                            [999, 3, 900, 1, 901, 902, 903, 904, 7],
                            [
                                True,
                                True,
                                False,
                                True,
                                False,
                                False,
                                True,
                                True,
                                True,
                            ],
                        ),
                    ),
                    [UInt8(0b10111)],
                ),
            ),
        ]
    )


def test_list_access_combines_parent_child_validity_and_bounds() raises:
    var frame = integer_lists()
    var lengths: List[Int64] = [3, 0, 2, 0, 1]
    var parent_valid: List[Bool] = [True, True, True, False, True]
    for layout in range(3):
        var source = layout_for_nested(frame, layout)
        for batch in [1, 3, 64]:
            var actual = source.select_exprs(
                [
                    col("xs").list().len().alias("length"),
                    col("xs").list().first().alias("first"),
                    col("xs").list().last().alias("last"),
                    col("xs").list().contains(Int64(900)).alias("poison"),
                    col("xs").list().contains(Int64(1)).alias("contains"),
                ],
                batch_size=batch,
            )
            assert_equal(actual.height(), 5)
            assert_equal(
                actual.dtypes(),
                [
                    DataType.INT64,
                    DataType.INT64,
                    DataType.INT64,
                    DataType.BOOL,
                    DataType.BOOL,
                ],
            )
            for row in range(5):
                assert_equal(
                    actual.item(row, "length").is_null(), not parent_valid[row]
                )
                if parent_valid[row]:
                    assert_equal(
                        actual.item(row, "length").int64(), lengths[row]
                    )
                    assert_equal(actual.item(row, "poison").bool(), False)
                    assert_equal(actual.item(row, "contains").bool(), row == 0)
                else:
                    assert_true(actual.item(row, "poison").is_null())
                    assert_true(actual.item(row, "contains").is_null())
                for name in [String("first"), String("last")]:
                    assert_equal(
                        actual.item(row, name).is_null(), row != 0 and row != 4
                    )
                    if row == 0:
                        assert_equal(
                            actual.item(row, name).int64(),
                            Int64(3 if name == "first" else 1),
                        )
                    if row == 4:
                        assert_equal(actual.item(row, name).int64(), Int64(7))
            for at in [-8, -3, -2, -1, 0, 1, 2, 3, 8]:
                var got = source.select(col("xs").list().get(at))
                for row in range(5):
                    var pos = at + Int(lengths[row]) if at < 0 else at
                    var valid = (
                        parent_valid[row]
                        and pos >= 0
                        and pos < Int(lengths[row])
                        and (row == 0 and pos != 1 or row == 4)
                    )
                    assert_equal(got.item(row, "xs").is_null(), not valid)
                    if valid:
                        assert_equal(
                            got.item(row, "xs").int64(),
                            Int64(7 if row == 4 else 3 if pos == 0 else 1),
                        )
        assert_equal(
            source.clear().select(col("xs").list().get(0)).shape(), (0, 1)
        )


def numeric_lists[D: DType]() raises:
    var frame = DataFrame(
        [
            Series(
                "xs",
                ListColumn(
                    [0, 3, 3, 5, 7, 8],
                    Series(
                        "child",
                        Column[Scalar[D]](
                            [
                                Scalar[D](3),
                                Scalar[D](90),
                                Scalar[D](1),
                                Scalar[D](91),
                                Scalar[D](92),
                                Scalar[D](93),
                                Scalar[D](94),
                                Scalar[D](7),
                            ],
                            [True, False, True, False, False, True, True, True],
                        ),
                    ),
                    [UInt8(0b10111)],
                ),
            )
        ]
    )
    for layout in range(3):
        var source = layout_for_nested(frame, layout)
        for batch in [1, 3, 64]:
            var actual = source.select_exprs(
                [
                    col("xs").list().sum().alias("sum"),
                    col("xs").list().min().alias("min"),
                    col("xs").list().max().alias("max"),
                    col("xs").list().mean().alias("mean"),
                ],
                batch_size=batch,
            )
            var expected_sum_type = (
                DataType.INT64 if D == DType.int8
                or D == DType.int16
                or D == DType.uint8
                or D == DType.uint16 else DataType.of(D)
            )
            assert_equal(
                actual.dtypes(),
                [
                    expected_sum_type,
                    DataType.of(D),
                    DataType.of(D),
                    DataType.FLOAT64,
                ],
            )
            for row in range(5):
                assert_equal(actual.item(row, "sum").is_null(), row == 3)
                if row != 3:
                    var expected = 4 if row == 0 else 7 if row == 4 else 0
                    comptime if D.is_floating_point():
                        assert_equal(
                            Float64(actual.item(row, "sum").numeric[D]()),
                            Float64(expected),
                        )
                    elif D == DType.int8 or D == DType.int16 or D == DType.uint8 or D == DType.uint16:
                        assert_equal(
                            actual.item(row, "sum").int64(), Int64(expected)
                        )
                    else:
                        assert_equal(
                            Int64(actual.item(row, "sum").numeric[D]()),
                            Int64(expected),
                        )
                for name in [String("min"), String("max"), String("mean")]:
                    assert_equal(
                        actual.item(row, name).is_null(), row != 0 and row != 4
                    )
                    if row == 0 or row == 4:
                        var expected = (
                            7 if row
                            == 4 else 1 if name
                            == "min" else 3 if name
                            == "max" else 2
                        )
                        if name == "mean":
                            assert_equal(
                                actual.item(row, name).float64(),
                                Float64(expected),
                            )
                        else:
                            assert_equal(
                                Float64(actual.item(row, name).numeric[D]()),
                                Float64(expected),
                            )
        var contains: Expr
        comptime if D.is_floating_point():
            contains = col("xs").list().contains(Float64(1))
        else:
            contains = col("xs").list().contains(Int64(1))
        var flags = source.select(contains)
        for row in range(5):
            assert_equal(flags.item(row, "xs").is_null(), row == 3)
            if row != 3:
                assert_equal(flags.item(row, "xs").bool(), row == 0)


def test_list_numeric_reductions_preserve_element_types_and_empty_policy() raises:
    comptime for k in range(len(NESTED_NUMERIC_DTYPES)):
        numeric_lists[NESTED_NUMERIC_DTYPES[k]]()


def test_string_list_join_skips_null_elements_and_split_keeps_empty_parts() raises:
    var frame = DataFrame(
        [
            Series(
                "xs",
                ListColumn(
                    [0, 4, 4, 6, 7],
                    Series(
                        "child",
                        Column[String](
                            ["a", "poison", "", "β", "x", "y", "hidden"],
                            [True, False, True, True, False, False, True],
                        ),
                    ),
                    [UInt8(0b0111)],
                ),
            )
        ]
    )
    for layout in range(3):
        var source = layout_for_nested(frame, layout)
        var actual = source.select_exprs(
            [
                col("xs").list().join("|").alias("joined"),
                col("xs").list().contains("poison").alias("poison"),
                col("xs").list().contains("").alias("empty"),
            ]
        )
        for row in range(4):
            assert_equal(actual.item(row, "joined").is_null(), row == 3)
            if row != 3:
                assert_equal(
                    actual.item(row, "joined").string(),
                    "a||β" if row == 0 else "",
                )
                assert_equal(actual.item(row, "poison").bool(), False)
                assert_equal(actual.item(row, "empty").bool(), row == 0)
    var strings = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ["", "::a::::b::", "α::β", "hidden"],
                    [True, True, True, False],
                ),
            )
        ]
    )
    var plain: List[List[String]] = [[""], ["", "a", "", "b", ""], ["α", "β"]]
    var inclusive_parts: List[List[String]] = [
        [""],
        ["::", "a::", "::", "b::", ""],
        ["α::", "β"],
    ]
    for layout in range(3):
        for inclusive in [False, True]:
            var actual = layout_for_nested(strings, layout).select(
                col("s").str().split("::", inclusive=inclusive)
            )
            assert_equal(actual.dtypes(), [DataType.list(DataType.STRING)])
            assert_true(actual.item(3, "s").is_null())
            for row in range(3):
                var children = actual.item(row, "s").list()
                var expected = (
                    inclusive_parts[row]
                    .copy() if inclusive else plain[row]
                    .copy()
                )
                assert_equal(len(children), len(expected))
                for i in range(len(expected)):
                    assert_equal(children.get(i).string(), expected[i])
    with assert_raises():
        _ = strings.select(col("s").str().split(""))


def struct_frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("before", Column[Int64]([10, 11, 12, 13])),
            Series(
                "record",
                StructColumn(
                    [
                        Series(
                            "number",
                            Column[Int64](
                                [4, 900, 901, 8], [True, False, True, True]
                            ),
                        ),
                        Series(
                            "text",
                            Column[String](
                                ["a", "poison", "hidden", "d"],
                                [True, False, True, False],
                            ),
                        ),
                    ],
                    [UInt8(0b1011)],
                ),
            ),
            Series("after", Column[String](["w", "x", "y", "z"])),
        ]
    )


def test_struct_fields_unnest_and_pack_preserve_schema_and_parent_validity() raises:
    var frame = struct_frame()
    for layout in range(3):
        var source = layout_for_nested(frame, layout)
        var unpacked = source.unnest("record")
        assert_equal(unpacked.columns(), ["before", "number", "text", "after"])
        assert_equal(
            unpacked.dtypes(),
            [DataType.INT64, DataType.INT64, DataType.STRING, DataType.STRING],
        )
        var extracted = source.select_exprs(
            [
                col("record").field("number").alias("number"),
                col("record").field("text").alias("text"),
            ]
        )
        assert_equal(extracted.columns(), ["number", "text"])
        for row in range(4):
            assert_equal(unpacked.item(row, "before").int64(), Int64(10 + row))
            assert_equal(
                unpacked.item(row, "after").string(),
                [String("w"), String("x"), String("y"), String("z")][row],
            )
            for output in [extracted.copy(), unpacked.copy()]:
                assert_equal(
                    output.item(row, "number").is_null(), row == 1 or row == 2
                )
                assert_equal(output.item(row, "text").is_null(), row != 0)
                if row == 0 or row == 3:
                    assert_equal(
                        output.item(row, "number").int64(),
                        Int64(4 if row == 0 else 8),
                    )
                if row == 0:
                    assert_equal(output.item(row, "text").string(), "a")
            assert_equal(source.item(row, "record").is_null(), row == 2)
        var packed = unpacked.pack_struct("packed", ["text", "number"])
        assert_equal(
            packed.columns(), ["before", "number", "text", "after", "packed"]
        )
        assert_equal(
            packed.column("packed").dtype().field_names(), ["text", "number"]
        )
        for row in range(4):
            assert_true(not packed.item(row, "packed").is_null())
            for field in [String("text"), String("number")]:
                var cell = packed.item(row, "packed").struct_field(field)
                var valid = row == 0 or field == "number" and row == 3
                assert_equal(cell.is_null(), not valid)
                if valid:
                    if field == "number":
                        assert_equal(cell.int64(), Int64(4 if row == 0 else 8))
                    else:
                        assert_equal(cell.string(), "a")
        assert_equal(source.clear().unnest("record").shape(), (0, 4))
        with assert_raises():
            _ = source.with_column(
                Series("number", Column[Int64]([1, 2, 3, 4]))
            ).unnest("record")


def test_as_struct_names_expression_fields_and_scalar_broadcast() raises:
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([2, 90, 5], [True, False, True])),
            Series("s", Column[String](["a", "b", "c"])),
        ]
    )
    for layout in range(3):
        var source = layout_for_nested(frame, layout)
        var expression = as_struct(
            [
                (col("x") + lit(Int64(1))).alias("successor"),
                lit("λ").alias("constant"),
                col("s").alias("text"),
            ],
            name="record",
        )
        var actual = source.select(expression)
        assert_equal(actual.columns(), ["record"])
        assert_equal(actual.height(), 3)
        assert_equal(
            actual.column("record").dtype().field_names(),
            ["successor", "constant", "text"],
        )
        assert_equal(
            actual.column("record").dtype().field_dtypes(),
            [DataType.INT64, DataType.STRING, DataType.STRING],
        )
        for row in range(3):
            var value = actual.item(row, "record")
            assert_true(not value.is_null())
            assert_equal(value.struct_field("successor").is_null(), row == 1)
            if row != 1:
                assert_equal(
                    value.struct_field("successor").int64(),
                    Int64(3 if row == 0 else 6),
                )
            assert_equal(value.struct_field("constant").string(), "λ")
            assert_equal(
                value.struct_field("text").string(),
                [String("a"), String("b"), String("c")][row],
            )
        assert_equal(source.clear().select(expression).shape(), (0, 1))
        var scalar = source.select(
            as_struct([lit(Int64(9)).alias("n"), lit("z").alias("s")])
        )
        assert_equal(scalar.shape(), (1, 1))
        assert_equal(scalar.columns(), ["n"])
        assert_equal(scalar.item(0, "n").struct_field("n").int64(), Int64(9))
        assert_equal(scalar.item(0, "n").struct_field("s").string(), "z")


def assert_imploded_literal(cell: AnyValue, row: Int, name: String) raises:
    var valid = row != 1 and (name == "text" or row != 4)
    assert_true(
        cell.dtype() == (DataType.INT64 if name == "value" else DataType.STRING)
    )
    assert_equal(cell.is_null(), not valid)
    if not valid:
        return
    if name == "value":
        var values: List[Int64] = [7, 0, 2, 5, 0]
        assert_equal(cell.int64(), values[row])
    else:
        var texts: List[String] = ["α", "", "β", "", "last"]
        assert_equal(cell.string(), texts[row])


def assert_nested_integer_literal(cell: AnyValue, row: Int) raises:
    assert_true(cell.dtype() == DataType.list(DataType.INT64))
    assert_equal(cell.is_null(), row == 3)
    if row == 3:
        return
    var lengths: List[Int] = [3, 0, 2, 0, 1]
    var children = cell.list()
    assert_equal(len(children), lengths[row])
    assert_true(children.dtype() == DataType.INT64)
    for index in range(lengths[row]):
        var valid = row == 4 or row == 0 and index != 1
        assert_equal(children.get(index).is_null(), not valid)
        if valid:
            assert_equal(
                children.get(index).int64(),
                Int64(7 if row == 4 else 3 if index == 0 else 1),
            )


def test_implode_gathers_ordered_typed_values_and_nulls_per_group() raises:
    var frame = DataFrame(
        [
            Series("key", Column[String](["b", "a", "b", "a", "b"])),
            Series(
                "value",
                Column[Int64](
                    [7, 90, 2, 5, 91], [True, False, True, True, False]
                ),
            ),
            Series(
                "text",
                Column[String](
                    ["α", "hidden", "β", "", "last"],
                    [True, False, True, True, True],
                ),
            ),
        ]
    )
    for layout in range(3):
        var source = layout_for_nested(frame, layout)
        for batch in [1, 3, 64]:
            var all = source.select_exprs(
                [col("value").implode(), col("text").implode()],
                batch_size=batch,
            )
            assert_equal(all.shape(), (1, 2))
            assert_equal(
                all.dtypes(),
                [DataType.list(DataType.INT64), DataType.list(DataType.STRING)],
            )
            for name in [String("value"), String("text")]:
                var children = all.item(0, name).list()
                assert_equal(len(children), 5)
                for row in range(5):
                    assert_imploded_literal(children.get(row), row, name)
            var grouped = source.group_by("key", maintain_order=True).agg(
                [col("value").implode(), col("text").implode()],
                batch_size=batch,
            )
            assert_equal(grouped.columns(), ["key", "value", "text"])
            assert_equal(grouped.height(), 2)
            var indices: List[List[Int]] = [[0, 2, 4], [1, 3]]
            for group in range(2):
                assert_equal(
                    grouped.item(group, "key").string(),
                    "b" if group == 0 else "a",
                )
                for name in [String("value"), String("text")]:
                    var children = grouped.item(group, name).list()
                    assert_equal(len(children), len(indices[group]))
                    for row in range(len(indices[group])):
                        assert_imploded_literal(
                            children.get(row), indices[group][row], name
                        )
        var empty = source.clear().select(col("value").implode())
        assert_equal(empty.shape(), (1, 1))
        assert_equal(len(empty.item(0, "value").list()), 0)
        assert_equal(
            source.clear().group_by("key").agg(col("value").implode()).shape(),
            (0, 2),
        )
    var nested = integer_lists()
    var gathered = nested.select(col("xs").implode())
    assert_equal(
        gathered.dtypes(), [DataType.list(DataType.list(DataType.INT64))]
    )
    var children = gathered.item(0, "xs").list()
    for row in range(5):
        assert_nested_integer_literal(children.get(row), row)


def struct_key_frame(
    hidden: Bool = False, null_fields: Bool = False
) raises -> DataFrame:
    var valid: List[Bool] = [
        True,
        not null_fields,
        hidden,
        True,
        not null_fields,
        hidden,
    ]
    return DataFrame(
        [
            Series("id", Column[Int64]([0, 1, 2, 3, 4, 5])),
            Series(
                "key",
                StructColumn(
                    [
                        Series(
                            "n",
                            Column[Int64]([7, 8, 11, 7, 8, 12], valid.copy()),
                        ),
                        Series(
                            "s",
                            Column[String](
                                ["a", "b", "hidden1", "a", "b", "hidden2"],
                                valid.copy(),
                            ),
                        ),
                    ],
                    [UInt8(0b011011)],
                ),
            ),
        ]
    )


def test_struct_key_equality_and_supported_join_kinds() raises:
    var frame = struct_key_frame()
    for layout in range(3):
        var source = layout_for_nested(frame, layout)
        var groups = source.group_by("key", maintain_order=True).agg(
            col("id").implode()
        )
        assert_equal(groups.height(), 3)
        for group in range(3):
            assert_equal(groups.item(group, "key").is_null(), group == 2)
            var members = groups.item(group, "id").list()
            assert_equal(len(members), 2)
            assert_equal(members.get(0).int64(), Int64(group))
            assert_equal(members.get(1).int64(), Int64(group + 3))
        assert_equal(source.n_unique(["key"]), 3)
        var unique = source.unique(["key"], keep="first", maintain_order=True)
        assert_equal(unique.height(), 3)
        var duplicated = source.is_duplicated(["key"])
        for row in range(6):
            assert_true(duplicated.get(row).bool())
        for row in range(3):
            assert_equal(unique.item(row, "id").int64(), Int64(row))
        var right = (
            source.take([3, 4, 5, 0])
            .select(["key", "id"])
            .rename({"id": "rid"})
        )
        var right_tags: List[Int] = [0, 1, 2, 0]
        var right_ids: List[Int64] = [3, 4, 5, 0]
        for how in [
            String("inner"),
            String("left"),
            String("semi"),
            String("anti"),
        ]:
            var actual = source.join(right, on="key", how=how)
            var expected_left = List[Int]()
            var expected_right = List[Int]()
            for left_row in range(6):
                var found = False
                for right_row in range(4):
                    if (
                        left_row % 3 != 2
                        and left_row % 3 == right_tags[right_row]
                    ):
                        found = True
                        if how == "inner" or how == "left":
                            expected_left.append(left_row)
                            expected_right.append(right_row)
                if (
                    how == "semi"
                    and found
                    or how == "anti"
                    and not found
                    or how == "left"
                    and not found
                ):
                    expected_left.append(left_row)
                    expected_right.append(-1)
            assert_equal(actual.height(), len(expected_left))
            var expected_names: List[String] = ["id", "key"]
            if how == "inner" or how == "left":
                expected_names.append("rid")
            assert_equal(actual.columns(), expected_names)
            for row in range(len(expected_left)):
                assert_equal(
                    actual.item(row, "id").int64(), Int64(expected_left[row])
                )
                var key = actual.item(row, "key")
                var tag = expected_left[row] % 3
                assert_equal(key.is_null(), tag == 2)
                if tag != 2:
                    assert_equal(
                        key.struct_field("n").int64(),
                        Int64(7 if tag == 0 else 8),
                    )
                    assert_equal(
                        key.struct_field("s").string(), "a" if tag == 0 else "b"
                    )
                if how == "inner" or how == "left":
                    assert_equal(
                        actual.item(row, "rid").is_null(),
                        expected_right[row] < 0,
                    )
                    if expected_right[row] >= 0:
                        assert_equal(
                            actual.item(row, "rid").int64(),
                            right_ids[expected_right[row]],
                        )
        for how in [String("right"), String("full")]:
            with assert_raises():
                _ = source.join(right, on="key", how=how)


def test_nested_unsupported_operations_fail_for_nonempty_and_empty_inputs() raises:
    var lists = integer_lists().select(["xs"])
    var records = struct_frame().select(["record"]).rename({"record": "xs"})
    for frame in [lists.copy(), records.copy()]:
        for source in [frame.copy(), frame.clear()]:
            with assert_raises():
                _ = source.sort("xs")
            with assert_raises():
                _ = to_csv_string(source)
            with assert_raises():
                _ = source.select(col("xs").sum())
            with assert_raises():
                _ = source.select(col("xs").min())
            with assert_raises():
                _ = source.select(
                    when(lit(True)).then(col("xs")).otherwise(col("xs"))
                )
    for source in [lists.copy(), lists.clear()]:
        with assert_raises():
            _ = source.group_by("xs").agg(col("xs").implode())
        with assert_raises():
            _ = source.unique(["xs"])
        with assert_raises():
            _ = source.join(source, on="xs")
    var inner = integer_lists().column("xs")
    var nested_keys = DataFrame([Series("key", StructColumn([inner.copy()]))])
    for source in [nested_keys.copy(), nested_keys.clear()]:
        with assert_raises():
            _ = source.group_by("key").agg(List[Expr]())
        with assert_raises():
            _ = source.join(source, on="key")


def test_nested_constructors_and_expression_validation_reject_malformed_inputs() raises:
    var child = Series("child", Column[Int64]([1, 2]))
    var invalid_offsets: List[List[Int64]] = [[], [-1, 1], [0, 2, 1], [0, 3]]
    for offsets in invalid_offsets:
        with assert_raises():
            _ = ListColumn(offsets.copy(), child.copy())
    with assert_raises():
        _ = ListColumn(
            List[Int64](length=10, fill=0), child.copy(), [UInt8(255)]
        )
    with assert_raises():
        _ = ListColumn.from_lists([child.copy()], DataType.STRING)
    with assert_raises():
        _ = ListColumn.from_lists([child.copy()], DataType.INT64, [True, False])
    with assert_raises():
        _ = StructColumn(List[Series]())
    with assert_raises():
        _ = StructColumn([child.copy(), child.copy()])
    with assert_raises():
        _ = StructColumn([child.copy(), child.slice(0, 1).renamed("short")])
    with assert_raises():
        _ = StructColumn(
            [Series("n", Column[Int64](List[Int64](length=9, fill=0)))],
            [UInt8(255)],
        )
    with assert_raises():
        _ = as_struct(List[Expr]())
    var lists = integer_lists()
    var records = struct_frame()
    for source in [lists.copy(), lists.clear()]:
        with assert_raises():
            _ = source.select(col("id").list().len())
        with assert_raises():
            _ = source.select(col("xs").list().contains("wrong"))
        with assert_raises():
            _ = source.select(col("xs").list().contains(Float64(1)))
        with assert_raises():
            _ = source.select(col("xs").list().join(","))
        with assert_raises():
            _ = source.select(col("id").field("missing"))
        with assert_raises():
            _ = source.unnest("id")
        with assert_raises():
            _ = source.pack_struct("record", ["missing"])
        with assert_raises():
            _ = source.select(
                as_struct([col("id").alias("same"), col("id").alias("same")])
            )
    for source in [records.copy(), records.clear()]:
        with assert_raises():
            _ = source.select(col("record").field("missing"))


def test_nested_parent_child_bitmaps_cross_window_and_batch_boundaries() raises:
    for n in [0, 1, 7, 8, 9, 63, 64, 65]:
        for offset in [0, 1, 7]:
            var rows = List[Series]()
            var parents = List[Bool]()
            var values = List[Int64]()
            var children = List[Bool]()
            var bits = List[UInt8](length=(n + offset + 7) // 8, fill=0)
            for row in range(n + offset):
                rows.append(
                    Series("child", Column[Int64]([Int64(row)], [row % 5 != 2]))
                )
                parents.append(row % 3 != 1)
                values.append(Int64(row))
                children.append(row % 5 != 2)
                if row % 3 != 1:
                    bits[row // 8] |= UInt8(1) << UInt8(row % 8)
            var frame = DataFrame(
                [
                    Series(
                        "xs",
                        ListColumn.from_lists(rows, DataType.INT64, parents),
                    ),
                    Series(
                        "record",
                        StructColumn(
                            [
                                Series(
                                    "value", Column[Int64](values^, children^)
                                )
                            ],
                            bits^,
                        ),
                    ),
                ]
            ).slice(offset, n)
            for layout in [0, 2]:
                var source = layout_for_nested(frame, layout)
                for batch in [1, 7, 8, 64]:
                    var actual = source.select_exprs(
                        [
                            col("xs").list().first().alias("list_value"),
                            col("record").field("value").alias("struct_value"),
                            col("xs").list().len().alias("length"),
                        ],
                        batch_size=batch,
                    )
                    assert_equal(actual.shape(), (n, 3))
                    for row in range(n):
                        var original = row + offset
                        var parent = original % 3 != 1
                        var valid = parent and original % 5 != 2
                        assert_equal(
                            actual.item(row, "length").is_null(), not parent
                        )
                        if parent:
                            assert_equal(
                                actual.item(row, "length").int64(), Int64(1)
                            )
                        for name in [
                            String("list_value"),
                            String("struct_value"),
                        ]:
                            assert_equal(
                                actual.item(row, name).is_null(), not valid
                            )
                            if valid:
                                assert_equal(
                                    actual.item(row, name).int64(),
                                    Int64(original),
                                )


def test_struct_null_parent_key_equality_ignores_hidden_field_payloads() raises:
    var source = struct_key_frame(hidden=True)
    var groups = source.group_by("key", maintain_order=True).agg(
        col("id").implode()
    )
    assert_true(
        groups.height() == 3,
        "null struct keys must ignore hidden field payloads",
    )
    assert_equal(source.n_unique(["key"]), 3)
    var members = groups.item(2, "id").list()
    assert_equal(len(members), 2)
    assert_equal(members.get(0).int64(), Int64(2))
    assert_equal(members.get(1).int64(), Int64(5))


def test_struct_join_matches_valid_records_with_null_fields() raises:
    var source = struct_key_frame(null_fields=True)
    var groups = source.group_by("key", maintain_order=True).agg(
        col("id").implode()
    )
    assert_equal(groups.height(), 3)
    assert_true(not groups.item(1, "key").is_null())
    assert_true(groups.item(1, "key").struct_field("n").is_null())
    assert_true(groups.item(2, "key").is_null())
    var right = source.take([1]).rename({"id": "rid"})
    var actual = source.join(right, on="key", how="inner")
    assert_true(
        actual.height() == 2,
        "valid struct join keys must match equal null fields",
    )
    assert_equal(actual.item(0, "id").int64(), Int64(1))
    assert_equal(actual.item(1, "id").int64(), Int64(4))
    for row in range(2):
        assert_equal(actual.item(row, "rid").int64(), Int64(1))
