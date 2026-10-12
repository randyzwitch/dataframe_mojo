"""Row selection preserves logical cells, metadata, and stable associations.

The model describes input cells and stable scalar order before any dataframe
exists. Logical types are the coverage axis; physical layouts are variations.
"""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, col, lit
from tests.support.key_kernel_reference import KeyCell, expected_order
from tests.support.selection_types_reference import (
    SELECTION_TYPES,
    assert_selected_frame,
    assert_selection_cell,
    bytes_less,
    decimal_text,
    selection_bytes,
    selection_decimal,
    selection_column,
    selection_dtype,
    selection_frame,
    selection_key,
    selection_rank,
    selection_text,
    selection_valid,
    selection_wkb,
)


def test_selection_reference_literal_cells_and_byte_order_anchors() raises:
    assert_equal(decimal_text(Int128(-1234567)), "-12345.67")
    assert_equal(decimal_text(Int128(5)), "0.05")
    assert_equal(decimal_text(Int128(9007199254740993)), "90071992547409.93")
    assert_equal(selection_text(1), "a\x00")
    assert_equal(selection_bytes(4), List[UInt8]())
    assert_true(bytes_less([0], [0, 0]))
    assert_true(bytes_less([128, 255], [255]))
    assert_true(not bytes_less([255], [128, 255]))
    var point: List[UInt8] = [
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        192,
        0,
        0,
        0,
        0,
        0,
        0,
        28,
        192,
    ]
    assert_equal(selection_wkb(4), point)
    # Code-order deliberately differs from lexical value order.
    var dtype = selection_dtype(13)
    assert_equal(String(dtype.dictionary()[].get(0)), "abcdefghijklm\x00é")
    assert_equal(String(dtype.dictionary()[].get(1)), "")


def filter_rows(n: Int, offset: Int, predicate: Int) -> List[Int]:
    var result = List[Int]()
    for row in range(offset, offset + n):
        var known = row % 5 != 2
        var value = row % 3 == 1
        var keep = known and value
        if predicate == 1:
            keep = known and not value
        if predicate == 2:
            keep = not known or known and value and row >= offset + n // 2
        if predicate == 3:
            keep = False
        if predicate >= 4:
            keep = True
        if keep:
            result.append(row)
    return result^


def test_filter_preserves_every_logical_payload_across_offsets_chunks_and_nulls() raises:
    for n in [0, 1, 7, 8, 9, 63, 64, 65]:
        for mode in range(3):
            for layout in range(3):
                var offset = 7 if layout == 1 else 0
                var source = selection_frame(n, offset, mode, layout == 2)
                var predicates: List[Expr] = [
                    col("predicate"),
                    ~col("predicate"),
                    (
                        col("predicate")
                        & (col("id") >= lit(Int64(offset + n // 2)))
                    )
                    | col("predicate").is_null(),
                    col("predicate") & lit(False),
                    col("predicate") | lit(True),
                    lit(True),
                ]
                for batch in [1, 7, 64]:
                    for predicate in range(len(predicates)):
                        assert_selected_frame(
                            source.filter(
                                predicates[predicate], batch_size=batch
                            ),
                            filter_rows(n, offset, predicate),
                            mode,
                        )
                for dense in [False, True]:
                    var bits = List[Bool]()
                    var valid = List[Bool]()
                    var rows = List[Int]()
                    for row in range(offset, offset + n):
                        bits.append(dense or row % 3 == 1)
                        valid.append(dense or row % 5 != 2)
                        if bits[len(bits) - 1] and valid[len(valid) - 1]:
                            rows.append(row)
                    assert_selected_frame(
                        source.filter(Column[Bool](bits^, valid^)), rows, mode
                    )
                assert_selected_frame(
                    source.filter(lit(False)), List[Int](), mode
                )


def test_typed_boolean_predicates_select_rows_and_preserve_unrelated_payloads() raises:
    for mode in range(3):
        for layout in range(3):
            var offset = 1 if layout == 1 else 0
            var source = selection_frame(65, offset, mode, layout == 2)
            var predicates: List[Expr] = [
                col("v10"),
                col("v12") == "",
                col("v13") == "a\x00",
                col("v14") > lit("0.00").cast(selection_dtype(14)),
                col("v15") > lit("0.00").cast(selection_dtype(15)),
                col("v16") > lit("0.00").cast(selection_dtype(16)),
                col("v26").list().len() == 0,
                col("v27").field("number").is_null(),
                ((col("v12") == "") | (col("v13") == "a\x00")) & col("v10"),
            ]
            for predicate in range(len(predicates)):
                var rows = List[Int]()
                for row in range(offset, offset + 65):
                    var valid = selection_valid(row, mode)
                    var keep: Bool
                    if predicate == 0:
                        keep = valid and row % 3 != 0
                    elif predicate == 1:
                        keep = valid and selection_rank(row) == 0
                    elif predicate == 2:
                        keep = valid and selection_rank(row) == 1
                    elif predicate >= 3 and predicate <= 5:
                        keep = valid and row % 3 != 0
                    elif predicate == 6:
                        keep = valid and row % 4 == 0
                    elif predicate == 7:
                        keep = not valid or row % 3 == 1
                    else:
                        keep = (
                            valid
                            and (
                                selection_rank(row) == 0
                                or selection_rank(row) == 1
                            )
                            and row % 3 != 0
                        )
                    if keep:
                        rows.append(row)
                assert_selected_frame(
                    source.filter(predicates[predicate], batch_size=7),
                    rows,
                    mode,
                )


def public_order_rows(
    kind: Int,
    n: Int,
    offset: Int,
    mode: Int,
    descending: List[Bool],
    nulls_last: List[Bool],
) raises -> List[Int]:
    var cells = List[List[KeyCell]]()
    for row in range(offset, offset + n):
        var keys: List[KeyCell] = [selection_key(kind, row, mode)]
        if len(descending) == 2:
            keys.append(KeyCell(0, row % 5 != 2, integer=Int128(row % 3 == 1)))
        cells.append(keys^)
    var order = expected_order(cells, descending, nulls_last)
    for i in range(len(order)):
        order[i] += offset
    return order^


def assert_public_order(
    actual: DataFrame, rows: List[Int], kind: Int, mode: Int
) raises:
    assert_equal(actual.height(), len(rows))
    assert_equal(
        actual.columns(),
        [
            String("id"),
            String("predicate"),
            "v" + String(kind),
            String("v26"),
            String("v27"),
        ],
    )
    assert_true(
        actual.column("v" + String(kind)).dtype() == selection_dtype(kind),
        "sorting must preserve the key column logical dtype",
    )
    assert_true(actual.column("v26").dtype() == selection_dtype(26))
    assert_true(actual.column("v27").dtype() == selection_dtype(27))
    for i in range(len(rows)):
        var row = rows[i]
        assert_equal(actual.item(i, "id").int64(), Int64(row))
        assert_selection_cell(
            actual.item(i, "v" + String(kind)), kind, row, mode
        )
        assert_selection_cell(actual.item(i, "v26"), 26, row, mode)
        assert_selection_cell(actual.item(i, "v27"), 27, row, mode)
        assert_equal(actual.item(i, "predicate").is_null(), row % 5 == 2)
        if row % 5 != 2:
            assert_equal(actual.item(i, "predicate").bool(), row % 3 == 1)


def check_public_sort_type(kind: Int) raises:
    var key = "v" + String(kind)
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            for layout in range(3):
                var offset = 7 if layout == 1 else 0
                var source = selection_frame(
                    n, offset, mode, layout == 2
                ).select(["id", "predicate", key, "v26", "v27"])
                for descending in [False, True]:
                    for nulls_last in [False, True]:
                        var rows = public_order_rows(
                            kind, n, offset, mode, [descending], [nulls_last]
                        )
                        assert_public_order(
                            source.sort(key, descending, nulls_last),
                            rows,
                            kind,
                            mode,
                        )
                        var keys: List[String] = [key, "predicate"]
                        var directions: List[Bool] = [
                            descending,
                            not descending,
                        ]
                        var placements: List[Bool] = [
                            nulls_last,
                            not nulls_last,
                        ]
                        rows = public_order_rows(
                            kind, n, offset, mode, directions, placements
                        )
                        assert_public_order(
                            source.sort(
                                keys,
                                descending=directions,
                                nulls_last=placements,
                            ),
                            rows,
                            kind,
                            mode,
                        )
                    for composite in [False, True]:
                        var keys: List[String] = [key]
                        if composite:
                            keys.append("predicate")
                        var rows = public_order_rows(
                            kind,
                            n,
                            offset,
                            mode,
                            List[Bool](length=len(keys), fill=descending),
                            List[Bool](length=len(keys), fill=True),
                        )
                        for k in [0, 1, 3, 7, 8, 9, 64, 65, 66]:
                            var first = List[Int]()
                            for i in range(min(k, len(rows))):
                                first.append(rows[i])
                            var actual = source.top_k(
                                k, keys
                            ) if descending else source.bottom_k(k, keys)
                            assert_public_order(actual, first, kind, mode)


def test_public_numeric_and_boolean_sort_keep_exact_cells_stability_and_top_k() raises:
    for kind in range(11):
        check_public_sort_type(kind)


def test_public_decimal_and_temporal_sort_preserve_storage_metadata_and_stable_rows() raises:
    for kind in range(14, 26):
        check_public_sort_type(kind)


def test_public_string_binary_and_geometry_sort_use_value_byte_order_and_keep_metadata() raises:
    for kind in [11, 12, 28]:
        check_public_sort_type(kind)


def test_public_categorical_sort_uses_dictionary_values_and_preserves_categorical_dtype() raises:
    check_public_sort_type(13)


def test_sorting_supported_keys_preserves_every_unrelated_logical_payload() raises:
    for mode in range(3):
        for layout in range(3):
            var offset = 1 if layout == 1 else 0
            var source = selection_frame(65, offset, mode, layout == 2)
            for descending in [False, True]:
                var rows = public_order_rows(
                    16,
                    65,
                    offset,
                    mode,
                    [descending, not descending],
                    [True, False],
                )
                assert_selected_frame(
                    source.sort(
                        ["v16", "predicate"],
                        descending=[descending, not descending],
                        nulls_last=[True, False],
                    ),
                    rows,
                    mode,
                )


def test_nested_sort_keys_are_rejected_for_empty_nonempty_and_selection_boundaries() raises:
    for n in [0, 1, 65]:
        for kind in [26, 27]:
            var source = selection_frame(n, 1, 1)
            var key = "v" + String(kind)
            with assert_raises():
                _ = source.sort(key)
            with assert_raises():
                _ = source.sort(
                    ["id", key],
                    descending=[False, True],
                    nulls_last=[True, False],
                )
            for k in [0, 1, n, n + 1]:
                with assert_raises():
                    _ = source.top_k(k, key)
                with assert_raises():
                    _ = source.bottom_k(k, key)


def test_filter_requires_boolean_predicates_for_all_logical_types_even_when_empty() raises:
    for n in [0, 65]:
        var source = selection_frame(n, 7, 1, True)
        for kind in range(SELECTION_TYPES):
            if kind != 10:
                with assert_raises():
                    _ = source.filter(col("v" + String(kind)))


def test_filter_typed_column_comparisons_follow_values_and_independent_null_masks() raises:
    # Two independently generated columns avoid testing a predicate against its
    # own already-computed result. Filtering keeps only the original payloads.
    for kind in range(SELECTION_TYPES):
        if kind == 26 or kind == 27:
            continue
        for mode in range(3):
            for layout in range(3):
                var offset = 7 if layout == 1 else 0
                var original = selection_frame(65, offset, mode, layout == 2)
                var source = original.with_column(
                    selection_column(kind, 65, offset + 1, mode).renamed("rhs")
                )
                var key = col("v" + String(kind))
                var rhs = col("rhs")
                var predicates: List[Expr] = [key < rhs, key >= rhs, key == rhs]
                for operation in range(3):
                    var rows = List[Int]()
                    for row in range(offset, offset + 65):
                        var a = selection_key(kind, row, mode)
                        var b = selection_key(kind, row + 1, mode)
                        if not a.valid or not b.valid:
                            continue
                        var less: Bool
                        var at_least: Bool
                        var equal: Bool
                        if a.kind == 1:
                            # IEEE predicates differ from total ordering:
                            # all three comparisons below are false for NaN.
                            less = a.floating < b.floating
                            at_least = a.floating >= b.floating
                            equal = a.floating == b.floating
                        elif a.kind == 2:
                            less = a.text < b.text
                            at_least = a.text >= b.text
                            equal = a.text == b.text
                        else:
                            less = a.integer < b.integer
                            at_least = a.integer >= b.integer
                            equal = a.integer == b.integer
                        var keep = (
                            less if operation
                            == 0 else at_least if operation
                            == 1 else equal
                        )
                        if keep:
                            rows.append(row)
                    assert_selected_frame(
                        source.filter(
                            predicates[operation],
                            batch_size=7,
                            keep=original.columns(),
                        ),
                        rows,
                        mode,
                    )


def test_decimal_bare_filter_literals_adopt_numeric_column_type() raises:
    # Bare numeric literals adopt the operand dtype per docs/expressions.md.
    # Check every storage width before reporting the shared execution defect.
    var unsupported = 0
    for kind in [14, 15, 16]:
        var source = selection_frame(65, 7, 1)
        var rows = List[Int]()
        for row in range(7, 72):
            if selection_valid(row, 1) and selection_decimal(kind, row) > 0:
                rows.append(row)
        try:
            var actual = source.filter(col("v" + String(kind)) > 0)
            assert_selected_frame(actual, rows, 1)
        except error:
            var expected_error = "Unsupported literal dtype " + String(
                selection_dtype(kind)
            )
            if expected_error not in String(error):
                raise error
            unsupported += 1
    assert_equal(
        unsupported, 0, "decimal bare filter literals must adopt operand dtype"
    )
