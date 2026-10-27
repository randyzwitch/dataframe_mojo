"""Set equality and shape transforms across complete logical typed cells."""
# contract-group: table.sets_reshape
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    ListColumn,
    Series,
    concat,
)
from dataframe.dtype import CategoricalDictionary
from tests.support.coverage_generators import chunk_layout
from tests.support.selection_types_reference import (
    SELECTION_TYPES,
    assert_selected_frame,
    assert_selection_cell,
    selection_column,
    selection_dtype,
    selection_frame,
    selection_valid,
)
from tests.support.set_reshape_reference import (
    explode_child_count,
    explode_parent_valid,
    exploded_positions,
    set_counts,
    set_survivors,
)


def literal_rows(n: Int, offset: Int = 7) -> List[Int]:
    var rows = List[Int]()
    for i in range(n):
        rows.append(offset + i)
    return rows^


def assert_set_unordered(
    actual: DataFrame, expected: List[Int], mode: Int
) raises:
    assert_equal(actual.height(), len(expected))
    var observed = List[Int]()
    var seen = List[Bool](length=len(expected), fill=False)
    for row in range(actual.height()):
        var identity = Int(actual.item(row, "id").int64())
        var found = -1
        for i in range(len(expected)):
            if identity == expected[i]:
                found = i
        assert_true(found >= 0, "set result contains an unexpected source row")
        assert_true(not seen[found], "set result repeats a source row")
        seen[found] = True
        observed.append(identity)
    assert_selected_frame(actual, observed, mode)


def assert_shape_dtype(actual: Series, kind: Int) raises:
    if kind == 13:
        # Re-encoding dictionaries may change codes while preserving the
        # categorical logical type and every decoded value.
        assert_true(actual.dtype().is_categorical())
    else:
        assert_true(actual.dtype() == selection_dtype(kind))


def test_set_equality_survivors_and_explode_positions_have_literal_anchors() raises:
    var rows = literal_rows(9, 0)
    assert_equal(
        set_survivors(12, rows, 0, False, "first"), List[Int]([0, 1, 3, 4, 7])
    )
    assert_equal(
        set_survivors(12, rows, 0, False, "last"), List[Int]([4, 5, 6, 7, 8])
    )
    assert_equal(set_survivors(12, rows, 0, False, "none"), List[Int]([4, 7]))
    assert_equal(
        set_counts(12, rows, 0, False), List[Int]([3, 2, 3, 2, 1, 2, 2, 1, 3])
    )
    assert_equal(
        set_survivors(27, rows, 0, False, "none"), List[Int]([0, 2, 3, 5, 6, 8])
    )
    var positions = exploded_positions(3, 0, 0)
    assert_equal(positions[0], List[Int]([0, 1, 2, 2]))
    assert_equal(positions[1], List[Int]([-1, 7, 8, 9]))


def test_unique_extended_keys_preserve_complete_payloads_keep_policies_and_boolean_flags() raises:
    for kind in range(SELECTION_TYPES):
        if kind == 26:
            continue
        for n in [0, 1, 9, 65]:
            for mode in range(3):
                if kind == 27 and mode != 0:
                    # Hidden null-parent equality has a dedicated narrow
                    # regression below; do not let it abort unrelated types.
                    continue
                var source = selection_frame(n, 7, mode, n == 65)
                var rows = literal_rows(n)
                for compound in [False, True]:
                    var subset: List[String] = ["v" + String(kind)]
                    if compound:
                        subset.append("predicate")
                    var counts = set_counts(kind, rows, mode, compound)
                    var first = set_survivors(
                        kind, rows, mode, compound, "first"
                    )
                    assert_equal(source.n_unique(subset), len(first))
                    var duplicated = source.is_duplicated(subset)
                    var unique = source.is_unique(subset)
                    assert_true(duplicated.dtype() == DataType.BOOL)
                    assert_true(unique.dtype() == DataType.BOOL)
                    for i in range(n):
                        assert_equal(duplicated.get(i).bool(), counts[i] > 1)
                        assert_equal(unique.get(i).bool(), counts[i] == 1)
                    for keep in ["first", "last", "none", "any"]:
                        var expected = set_survivors(
                            kind, rows, mode, compound, keep
                        )
                        assert_selected_frame(
                            source.unique(
                                subset, keep=keep, maintain_order=True
                            ),
                            expected,
                            mode,
                        )
                        assert_set_unordered(
                            source.unique(
                                subset, keep=keep, maintain_order=False
                            ),
                            expected,
                            mode,
                        )


# expected-failure: Struct key equality reads hidden fields of null parents,
# expected-failure: retaining extra equivalent null keys.
# expected-diagnostic: struct unique must ignore hidden null-parent fields
def test_struct_unique_null_parents_ignore_every_hidden_field_payload() raises:
    var ignored_parent_failures = 0
    for n in [1, 9, 65]:
        for mode in [1, 2]:
            var source = selection_frame(n, 7, mode, n == 65)
            var expected = set_survivors(
                27, literal_rows(n), mode, False, "first"
            )
            var actual = source.unique(
                ["v27"], keep="first", maintain_order=True
            )
            if actual.height() > len(expected):
                # Narrow this known defect to retained extra null-parent
                # keys. Wrong payloads or dropped rows are separate failures.
                var returned = List[Int]()
                for row in range(actual.height()):
                    var id = Int(actual.item(row, "id").int64())
                    assert_true(id >= 7 and id < 7 + n)
                    if id not in expected:
                        assert_true(
                            actual.item(row, "v27").is_null(),
                            "struct unique split equal valid records",
                        )
                    returned.append(id)
                assert_selected_frame(actual, returned, mode)
                ignored_parent_failures += 1
            else:
                assert_selected_frame(actual, expected, mode)
    assert_equal(
        ignored_parent_failures,
        0,
        "struct unique must ignore hidden null-parent fields",
    )


def test_unique_rejects_list_keys_and_preserves_nested_values_as_payloads() raises:
    for n in [0, 9]:
        var source = selection_frame(n, 7, 1)
        for keep in ["any", "first", "last", "none"]:
            with assert_raises():
                _ = source.unique(["v26"], keep=keep)
        with assert_raises():
            _ = source.is_duplicated(["v26"])
        with assert_raises():
            _ = source.is_unique(["v26"])
        if n > 0:
            with assert_raises():
                _ = source.n_unique(["v26"])


def test_unpivot_preserves_all_logical_value_types_nested_cells_and_column_major_order() raises:
    for kind in range(SELECTION_TYPES):
        for n in [0, 1, 9, 65]:
            for mode in range(3):
                var ids = List[Int64]()
                for i in range(n):
                    ids.append(Int64(i))
                var frame = DataFrame(
                    [
                        Series("id", Column[Int64](ids^)),
                        selection_column(28, n, 7, mode).renamed(
                            "index_payload"
                        ),
                        selection_column(kind, n, 7, mode).renamed("a"),
                        selection_column(kind, n, 101, mode).renamed("b"),
                    ]
                )
                if n == 65:
                    frame = chunk_layout(frame)
                var actual = frame.unpivot(
                    ["b", "a"],
                    ["id", "index_payload"],
                    variable_name="source",
                    value_name="value",
                )
                var names: List[String] = [
                    "id",
                    "index_payload",
                    "source",
                    "value",
                ]
                assert_equal(actual.columns(), names)
                assert_equal(actual.height(), n * 2)
                assert_shape_dtype(actual.column("value"), kind)
                assert_shape_dtype(actual.column("index_payload"), 28)
                for part in range(2):
                    for row in range(n):
                        var destination = part * n + row
                        assert_equal(
                            actual.item(destination, "id").int64(), Int64(row)
                        )
                        assert_equal(
                            actual.item(destination, "source").string(),
                            "b" if part == 0 else "a",
                        )
                        assert_selection_cell(
                            actual.item(destination, "index_payload"),
                            28,
                            row + 7,
                            mode,
                        )
                        assert_selection_cell(
                            actual.item(destination, "value"),
                            kind,
                            row + (101 if part == 0 else 7),
                            mode,
                        )


def test_pivot_first_last_missing_cells_and_counts_preserve_all_logical_value_types() raises:
    var indexes: List[Int64] = [2, 1, 2, 3, 1, 3, 2, 1, 4]
    var index_valid: List[Bool] = [
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    ]
    var labels: List[String] = ["b", "a", "a", "b", "b", "a", "b", "a", "b"]
    var expected_keys: List[Int64] = [2, 1, 3, 0, 4]
    var first_a: List[Int] = [2, 1, -1, 5, -1]
    var first_b: List[Int] = [0, -1, 3, 4, 8]
    var last_a: List[Int] = [2, 7, -1, 5, -1]
    var last_b: List[Int] = [6, -1, 3, 4, 8]
    for kind in range(SELECTION_TYPES):
        for mode in range(3):
            var source = DataFrame(
                [
                    Series(
                        "index",
                        Column[Int64](indexes.copy(), index_valid.copy()),
                    ),
                    Series("label", Column[String](labels.copy())),
                    selection_column(kind, 9, 7, mode).renamed("value"),
                ]
            )
            for sorted_columns in [False, True]:
                for function in ["first", "last", "count", "len"]:
                    if kind in [26, 27]:
                        continue
                    if function == "count" and kind == 16 and mode != 0:
                        # Each nullable unsupported physical count has its own
                        # process-isolated witness: the current kernel aborts.
                        continue
                    var actual = source.pivot(
                        "label",
                        index=["index"],
                        values="value",
                        aggregate_function=function,
                        sort_columns=sorted_columns,
                        batch_size=1,
                    )
                    var names: List[String] = [
                        "index",
                        "a",
                        "b",
                    ] if sorted_columns else ["index", "b", "a"]
                    assert_equal(actual.columns(), names)
                    assert_equal(actual.height(), 5)
                    for name in ["a", "b"]:
                        if function == "first" or function == "last":
                            if kind == 13:
                                # Group reductions read categorical expressions
                                # as their decoded strings.
                                assert_true(
                                    actual.column(name).dtype()
                                    == DataType.STRING
                                )
                            else:
                                assert_shape_dtype(actual.column(name), kind)
                        else:
                            assert_true(
                                actual.column(name).dtype() == DataType.INT64
                            )
                        for row in range(5):
                            var index = actual.item(row, "index")
                            assert_equal(index.is_null(), row == 3)
                            if row != 3:
                                assert_equal(index.int64(), expected_keys[row])
                            var position = (
                                first_a[row] if name == "a" else first_b[row]
                            ) if function != "last" else (
                                last_a[row] if name == "a" else last_b[row]
                            )
                            var cell = actual.item(row, name)
                            if position < 0:
                                assert_true(cell.is_null())
                            elif function == "first" or function == "last":
                                assert_selection_cell(
                                    cell, kind, position + 7, mode
                                )
                            else:
                                var count = 0
                                for original in range(9):
                                    var matches = labels[
                                        original
                                    ] == name and index_valid[original] == (
                                        row != 3
                                    )
                                    if row != 3:
                                        matches = (
                                            matches
                                            and indexes[original]
                                            == expected_keys[row]
                                        )
                                    if matches and (
                                        function == "len"
                                        or selection_valid(original + 7, mode)
                                    ):
                                        count += 1
                                assert_equal(cell.int64(), Int64(count))
                with assert_raises():
                    _ = source.pivot(
                        "label",
                        index=["index"],
                        values="value",
                        sort_columns=sorted_columns,
                    )


def test_concat_extended_payloads_preserve_types_null_padding_names_and_row_order() raises:
    for kind in range(SELECTION_TYPES):
        for n in [0, 1, 9, 65]:
            for mode in range(3):
                var first = DataFrame([selection_column(kind, n, 7, mode)])
                var second = DataFrame([selection_column(kind, n, 101, mode)])
                if kind == 13 and n == 0:
                    # Different-dictionary empty recoding has its own narrow
                    # regression, so it cannot stop the other logical types.
                    second = first.copy()
                var vertical = concat(
                    [first.copy(), first.clear(), second.copy()]
                )
                assert_equal(vertical.height(), 2 * n)
                assert_shape_dtype(vertical.column("x"), kind)
                for row in range(2 * n):
                    assert_selection_cell(
                        vertical.item(row, "x"),
                        kind,
                        row + 7 if row < n else row - n + 101,
                        mode,
                    )
                var horizontal = concat(
                    [first.copy(), second.rename({"x": "right"})], "horizontal"
                )
                var names: List[String] = ["x", "right"]
                assert_equal(horizontal.columns(), names)
                assert_equal(horizontal.height(), n)
                for row in range(n):
                    assert_selection_cell(
                        horizontal.item(row, "x"), kind, row + 7, mode
                    )
                    assert_selection_cell(
                        horizontal.item(row, "right"), kind, row + 101, mode
                    )
                if kind == 13:
                    # Missing categorical columns have a dedicated strict
                    # regression; do not stop the other typed payloads.
                    continue
                var empty_schema = DataFrame(List[Series](), height=n)
                var diagonal = concat(
                    [first.copy(), empty_schema.copy(), second.copy()],
                    "diagonal",
                )
                assert_equal(diagonal.height(), 3 * n)
                assert_shape_dtype(diagonal.column("x"), kind)
                for row in range(3 * n):
                    if row >= n and row < 2 * n:
                        assert_true(diagonal.item(row, "x").is_null())
                    else:
                        assert_selection_cell(
                            diagonal.item(row, "x"),
                            kind,
                            row + 7 if row < n else row - 2 * n + 101,
                            mode,
                        )


def nullable_pivot_count(kind: Int) raises:
    var source = DataFrame(
        [
            Series(
                "label",
                Column[String](["a", "a", "a", "b", "b", "b", "a", "b", "a"]),
            ),
            selection_column(kind, 9, 7, 1).renamed("value"),
        ]
    )
    var actual = source.pivot(
        "label",
        index=List[String](),
        values="value",
        aggregate_function="count",
        batch_size=1,
    )
    assert_equal(actual.height(), 1)
    assert_equal(actual.columns(), List[String](["a", "b"]))
    assert_true(actual.column("a").dtype() == DataType.INT64)
    assert_true(actual.column("b").dtype() == DataType.INT64)
    # Global source rows 7 and 12 are null; a has 7,8,9,13,15 and b 10,11,12,14.
    assert_equal(actual.item(0, "a").int64(), Int64(4))
    assert_equal(actual.item(0, "b").int64(), Int64(3))


def test_pivot_count_nullable_decimal128_uses_parent_validity() raises:
    nullable_pivot_count(16)


def test_pivot_nested_reductions_reject_explicitly_before_execution() raises:
    for kind in [26, 27]:
        for n in [0, 9]:
            var source = DataFrame(
                [
                    Series(
                        "label",
                        Column[String](List[String](length=n, fill="a")),
                    ),
                    selection_column(kind, n, 7, 1).renamed("value"),
                ]
            )
            for function in ["first", "last"]:
                with assert_raises(
                    contains="does not support list or struct expressions yet"
                ):
                    _ = source.pivot(
                        "label",
                        index=List[String](),
                        values="value",
                        aggregate_function=function,
                    )


def test_pivot_nested_counts_use_parent_validity_and_lengths() raises:
    for kind in [26, 27]:
        for n in [0, 9]:
            var source = DataFrame(
                [
                    Series(
                        "label",
                        Column[String](List[String](length=n, fill="a")),
                    ),
                    selection_column(kind, n, 7, 1).renamed("value"),
                ]
            )
            for function in [String("count"), "len"]:
                var result = source.pivot(
                    "label",
                    index=List[String](),
                    values="value",
                    aggregate_function=function,
                    batch_size=1,
                )
                if n == 0:
                    assert_equal(result.height(), 0)
                    assert_equal(result.width(), 0)
                else:
                    var expected = 0
                    for row in range(n):
                        if function == "len" or selection_valid(row + 7, 1):
                            expected += 1
                    assert_equal(result.columns(), [String("a")])
                    assert_equal(result.height(), 1)
                    assert_equal(result.column("a").dtype(), DataType.INT64)
                    assert_equal(result.item(0, "a").int64(), Int64(expected))


def dictionary_series(
    var labels: List[String], var codes: List[UInt32], var valid: List[Bool]
) raises -> Series:
    var dictionary = CategoricalDictionary()
    for label in labels:
        dictionary.append(label)
    return Series("x", Column[UInt32](codes^, valid)).with_dtype(
        DataType.categorical(dictionary^)
    )


def test_categorical_concat_unifies_different_code_orders_and_new_labels_without_changing_sources() raises:
    var left = dictionary_series(
        ["b", "a"], [0, 1, UInt32.MAX, 0], [True, True, False, True]
    )
    var right = dictionary_series(
        ["a", "c", "b"], [0, 1, 2, UInt32.MAX], [True, True, True, False]
    )
    var actual = concat([DataFrame([left.copy()]), DataFrame([right.copy()])])
    var expected: List[String] = ["b", "a", "", "b", "a", "c", "b", ""]
    var validity: List[Bool] = [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert_true(actual.column("x").dtype().is_categorical())
    assert_equal(len(actual.column("x").dtype().dictionary()[]), 3)
    for row in range(8):
        assert_equal(actual.item(row, "x").is_null(), not validity[row])
        if validity[row]:
            assert_equal(actual.item(row, "x").string(), expected[row])
    assert_equal(len(left.dtype().dictionary()[]), 2)
    assert_equal(len(right.dtype().dictionary()[]), 3)
    assert_equal(left._data[Column[UInt32]].value(0), UInt32(0))
    assert_equal(right._data[Column[UInt32]].value(0), UInt32(0))
    assert_equal(left.get(0).string(), "b")
    assert_equal(right.get(0).string(), "a")


def test_categorical_concat_empty_inputs_preserve_logical_type_during_dictionary_union() raises:
    var empty = dictionary_series(["b", "a"], [], [])
    var present = dictionary_series(["a", "c", "b"], [0, 1], [True, True])
    var rejected = 0
    for reverse in [False, True]:
        for how in ["vertical", "diagonal"]:
            var frames: List[DataFrame] = [
                DataFrame([present.copy()]),
                DataFrame([empty.copy()]),
            ] if reverse else [
                DataFrame([empty.copy()]),
                DataFrame([present.copy()]),
            ]
            var actual = DataFrame(List[Series]())
            try:
                actual = concat(frames, how)
            except error:
                var message = String(error)
                if (
                    "concat " + how + ":" in message
                    and "string" in message
                    and "categorical" in message
                ):
                    rejected += 1
                    continue
                raise
            assert_equal(actual.height(), 2)
            assert_true(actual.column("x").dtype().is_categorical())
            assert_equal(actual.item(0, "x").string(), "a")
            assert_equal(actual.item(1, "x").string(), "c")
    assert_equal(
        rejected, 0, "empty categorical concat must preserve logical dtype"
    )


def test_diagonal_concat_missing_categorical_columns_preserve_dictionary_and_null_padding() raises:
    var rejected = 0
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var source = DataFrame([selection_column(13, n, 7, mode)])
            var actual = DataFrame(List[Series]())
            try:
                actual = concat(
                    [source.copy(), DataFrame(List[Series](), height=n)],
                    "diagonal",
                )
            except error:
                if (
                    String(error)
                    == "concat vertical: frame 1 column 0 is x:string, expected x:categorical"
                ):
                    rejected += 1
                    continue
                raise
            assert_equal(actual.height(), 2 * n)
            assert_true(actual.column("x").dtype().is_categorical())
            for row in range(n):
                assert_selection_cell(actual.item(row, "x"), 13, row + 7, mode)
                assert_true(actual.item(row + n, "x").is_null())
    assert_equal(
        rejected,
        0,
        "diagonal missing categorical columns must keep logical dtype",
    )


def test_explode_all_logical_child_types_keeps_parent_association_and_independent_child_validity() raises:
    for kind in range(SELECTION_TYPES):
        for n in [0, 1, 9, 65]:
            for parent_mode in range(3):
                for child_mode in range(3):
                    var offsets: List[Int64] = [0]
                    var bits = List[UInt8](length=(n + 7 + 7) // 8, fill=0)
                    var total = 0
                    for parent in range(n + 7):
                        total += explode_child_count(parent, parent_mode)
                        offsets.append(Int64(total))
                        if explode_parent_valid(parent, parent_mode):
                            bits[parent // 8] |= UInt8(1 << (parent % 8))
                    var child = selection_column(kind, total, 7, child_mode)
                    var lists = Series(
                        "items", ListColumn(offsets^, child^, bits^)
                    ).slice(7, n)
                    var source = selection_frame(n, 7, 1, n == 65).with_column(
                        lists^
                    )
                    var actual = source.explode("items")
                    var expected = exploded_positions(n, 7, parent_mode)
                    assert_selected_frame(
                        actual.drop(["items"]), expected[0], 1
                    )
                    assert_shape_dtype(actual.column("items"), kind)
                    for row in range(len(expected[1])):
                        var original = expected[1][row]
                        if original < 0:
                            assert_true(actual.item(row, "items").is_null())
                        else:
                            assert_selection_cell(
                                actual.item(row, "items"),
                                kind,
                                original,
                                child_mode,
                            )
