"""Complete typed grouped results independent of unspecified output order."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.reduction_types_reference import (
    fixture_key_matches,
    typed_reduction_columns,
    reduction_validity,
    fixture_codes,
    typed_scalar_reductions,
    assert_typed_reductions,
)
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_value


def test_grouped_type_aggregation_matrix_compares_complete_unordered_rows() raises:
    var assignments: List[Int64] = [2, 1, 2, 1, 0, 0]
    var members: List[List[Int]] = [[4, 5], [1, 3], [0, 2]]
    for mode in range(3):
        var valid = reduction_validity(mode)
        for source in typed_reduction_columns(valid):
            if source.dtype().is_categorical():
                continue
            # Nullable decimal count's fast path has its own isolated fatal
            # contract below; other decimal reductions remain in this matrix.
            var include_count = not source.dtype().is_decimal()
            var raw = DataFrame(
                [Series("g", Column[Int64](assignments.copy())), source.copy()]
            )
            for layout in range(3):
                var frame = (
                    raw.copy() if layout
                    == 0 else offset_layout(raw) if layout
                    == 1 else chunk_layout(raw)
                )
                for ordered in [False, True]:
                    var actual = frame.group_by(
                        "g", maintain_order=ordered
                    ).agg(
                        typed_scalar_reductions(source.dtype(), include_count)
                    )
                    var names: List[String] = ["g"]
                    for expression in typed_scalar_reductions(
                        source.dtype(), include_count
                    ):
                        names.append(expression._name)
                    assert_equal(actual.columns(), names)
                    assert_equal(actual.height(), 3)
                    assert_true(actual.column("g").dtype() == DataType.INT64)
                    var seen = List[Bool](length=3, fill=False)
                    for output in range(actual.height()):
                        var group = Int(actual.item(output, "g").int64())
                        assert_true(group >= 0 and group < 3)
                        assert_true(
                            not seen[group],
                            "unordered multiset cannot contain duplicate aggregate rows",
                        )
                        seen[group] = True
                        if ordered:
                            assert_equal(group, 2 - output)
                        assert_typed_reductions(
                            actual,
                            output,
                            source,
                            valid,
                            members[group],
                            include_count,
                        )
                    for present in seen:
                        assert_true(present)
                    var empty = (
                        frame.clear()
                        .group_by("g", maintain_order=ordered)
                        .agg(
                            typed_scalar_reductions(
                                source.dtype(), include_count
                            )
                        )
                    )
                    assert_equal(empty.height(), 0)
                    assert_equal(empty.columns(), names)
                    for name in names:
                        assert_true(
                            empty.column(name).dtype()
                            == actual.column(name).dtype()
                        )


def test_extended_group_keys_preserve_metadata_and_complete_null_groups() raises:
    var assignments: List[Int64] = [7, 11, 13, 17, 19, 23]
    for mode in [0, 1, 2]:
        var valid = reduction_validity(mode)
        for original in typed_reduction_columns(valid):
            var codes = fixture_codes(original.dtype())
            var leaders = List[Int]()
            var members = List[List[Int]]()
            for row in range(6):
                var found = -1
                for group in range(len(leaders)):
                    var leader = leaders[group]
                    if valid[row] == valid[leader] and (
                        not valid[row] or codes[row] == codes[leader]
                    ):
                        found = group
                if found < 0:
                    leaders.append(row)
                    members.append(List[Int]())
                    found = len(leaders) - 1
                members[found].append(row)
            var source = original.renamed("g")
            var raw = DataFrame(
                [source.copy(), Series("v", Column[Int64](assignments.copy()))]
            )
            for layout in range(3):
                var frame = (
                    raw.copy() if layout
                    == 0 else offset_layout(raw) if layout
                    == 1 else chunk_layout(raw)
                )
                for ordered in [False, True]:
                    var actual = frame.group_by(
                        "g", maintain_order=ordered
                    ).agg(
                        [
                            col("v").sum().alias("sum"),
                            col("v").len().alias("len"),
                        ]
                    )
                    assert_equal(actual.columns(), ["g", "sum", "len"])
                    assert_equal(actual.height(), len(leaders))
                    assert_true(actual.column("g").dtype() == source.dtype())
                    assert_true(actual.column("sum").dtype() == DataType.INT64)
                    assert_true(actual.column("len").dtype() == DataType.INT64)
                    var seen = List[Bool](length=len(leaders), fill=False)
                    for output in range(actual.height()):
                        var group = -1
                        var cell = actual.item(output, "g")
                        for candidate in range(len(leaders)):
                            var expected = source.get(leaders[candidate])
                            # Key values are literal fixture ranks; AnyValue
                            # structural equality is not used as aggregate oracle.
                            if cell.is_null() == expected.is_null():
                                if cell.is_null() or fixture_key_matches(
                                    cell,
                                    source.dtype(),
                                    codes[leaders[candidate]],
                                ):
                                    group = candidate
                        assert_true(group >= 0)
                        assert_true(not seen[group])
                        seen[group] = True
                        if ordered:
                            assert_equal(group, output)
                        assert_value(cell, source.get(leaders[group]))
                        var total = Int64(0)
                        for row in members[group]:
                            total += assignments[row]
                        assert_equal(actual.item(output, "sum").int64(), total)
                        assert_equal(
                            actual.item(output, "len").int64(),
                            Int64(len(members[group])),
                        )
                    for present in seen:
                        assert_true(present)


def test_grouped_modes_and_value_counts_return_complete_typed_child_lists() raises:
    from tests.support.reduction_types_reference import assert_distribution

    var assignments: List[Int64] = [2, 1, 2, 1, 0, 0]
    var members: List[List[Int]] = [[4, 5], [1, 3], [0, 2]]
    for mode in range(3):
        var valid = reduction_validity(mode)
        for source in typed_reduction_columns(valid):
            if source.dtype().is_categorical() or source.dtype().is_decimal():
                continue
            var raw = DataFrame(
                [Series("g", Column[Int64](assignments.copy())), source.copy()]
            )
            for layout in range(3):
                var frame = (
                    raw.copy() if layout
                    == 0 else offset_layout(raw) if layout
                    == 1 else chunk_layout(raw)
                )
                for sort in [False, True]:
                    for normalize in [False, True]:
                        var actual = frame.group_by("g").agg(
                            [
                                col("x").mode().alias("mode"),
                                col("x")
                                .value_counts(sort, normalize=normalize)
                                .alias("counts"),
                            ]
                        )
                        assert_equal(actual.columns(), ["g", "mode", "counts"])
                        assert_equal(actual.height(), 3)
                        assert_true(
                            actual.column("mode").dtype()
                            == DataType.list(source.dtype())
                        )
                        var seen = List[Bool](length=3, fill=False)
                        for row in range(3):
                            var group = Int(actual.item(row, "g").int64())
                            assert_true(group >= 0 and group < 3)
                            assert_true(not seen[group])
                            seen[group] = True
                            assert_distribution(
                                actual.item(row, "mode").list(),
                                actual.item(row, "counts").list(),
                                source,
                                valid,
                                members[group],
                                sort,
                                normalize,
                            )


def test_grouped_decimal_count_skips_null_rows_without_storage_confusion() raises:
    for width in [128, 64, 32]:
        var source = (
            Series(
                "x",
                Column[Int128](
                    [100, 999, 300, 999], [True, False, True, False]
                ),
            )
            .with_dtype(DataType.decimal(8, 2))
            .cast(DataType.decimal(8, 2, width))
        )
        var raw = DataFrame(
            [source.copy(), Series("g", Column[Int64]([0, 0, 1, 1]))]
        )
        for layout in range(3):
            var frame = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            var actual = frame.group_by("g", maintain_order=True).agg(
                [col("x").count().alias("count")]
            )
            assert_equal(actual.columns(), ["g", "count"])
            assert_equal(actual.height(), 2)
            assert_true(actual.column("count").dtype() == DataType.INT64)
            for row in range(2):
                assert_equal(actual.item(row, "g").int64(), Int64(row))
                assert_equal(actual.item(row, "count").int64(), Int64(1))


def test_categorical_count_distinct_and_arg_extrema_follow_decoded_value_semantics() raises:
    var groups: List[Int64] = [2, 1, 2, 1, 0, 0]
    var members: List[List[Int]] = [[4, 5], [1, 3], [0, 2]]
    for mode in range(3):
        var valid = reduction_validity(mode)
        var source = Series(
            "x", Column[String](["c", "a", "c", "b", "a", "z"], valid.copy())
        ).cast(DataType.CATEGORICAL)
        var raw = DataFrame(
            [source.copy(), Series("g", Column[Int64](groups.copy()))]
        )
        var expressions: List[Expr] = [
            col("x").count().alias("count"),
            col("x").null_count().alias("nulls"),
            col("x").len().alias("len"),
            col("x").n_unique().alias("unique"),
            col("x").arg_min().alias("arg_min"),
            col("x").arg_max().alias("arg_max"),
        ]
        var codes: List[Int64] = [3, 1, 3, 2, 1, 9]
        for layout in range(3):
            var frame = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            for grouped in [False, True]:
                var actual = frame.group_by("g", maintain_order=True).agg(
                    expressions
                ) if grouped else frame.select_exprs(expressions)
                assert_equal(actual.height(), 3 if grouped else 1)
                for output in range(actual.height()):
                    var rows: List[Int] = [0, 1, 2, 3, 4, 5]
                    if grouped:
                        var group = 2 - output
                        assert_equal(
                            actual.item(output, "g").int64(), Int64(group)
                        )
                        rows = members[group].copy()
                    var count = 0
                    var seen_codes = List[Int64]()
                    var null = False
                    var low = -1
                    var high = -1
                    for local in range(len(rows)):
                        var row = rows[local]
                        if not valid[row]:
                            null = True
                            continue
                        count += 1
                        var duplicate = False
                        for code in seen_codes:
                            duplicate = duplicate or code == codes[row]
                        if not duplicate:
                            seen_codes.append(codes[row])
                        if low < 0 or codes[row] < codes[rows[low]]:
                            low = local
                        if high < 0 or codes[row] > codes[rows[high]]:
                            high = local
                    for name in ["count", "nulls", "len", "unique"]:
                        assert_true(
                            actual.column(name).dtype() == DataType.INT64
                        )
                    assert_equal(
                        actual.item(output, "count").int64(), Int64(count)
                    )
                    assert_equal(
                        actual.item(output, "nulls").int64(),
                        Int64(len(rows) - count),
                    )
                    assert_equal(
                        actual.item(output, "len").int64(), Int64(len(rows))
                    )
                    assert_equal(
                        actual.item(output, "unique").int64(),
                        Int64(len(seen_codes) + Int(null)),
                    )
                    for high_arg in [False, True]:
                        var name = "arg_max" if high_arg else "arg_min"
                        var chosen = high if high_arg else low
                        assert_true(
                            actual.column(name).dtype() == DataType.UINT32
                        )
                        assert_equal(
                            actual.item(output, name).is_null(), chosen < 0
                        )
                        if chosen >= 0:
                            assert_equal(
                                actual.item(output, name).numeric[
                                    DType.uint32
                                ](),
                                UInt32(chosen),
                            )
