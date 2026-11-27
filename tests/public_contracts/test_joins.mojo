"""Full join rows and schema against a nested-loop oracle; duplicates are pairs."""
# contract-group: table.join
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, DataType, Series, col
from tests.support.reference_model import (
    generated_model,
    ReferenceRow,
    ReferenceTable,
)
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import assert_join_model
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def test_all_equality_join_kinds_match_nested_loops() raises:
    var shapes: List[Tuple[Int, Int]] = [
        (0, 0),
        (0, 3),
        (3, 0),
        (1, 1),
        (9, 7),
        (35, 33),
    ]
    var key_sets: List[List[String]] = [["i"], ["s"], ["f"], ["b"], ["s", "i"]]
    for shape in shapes:
        for validity in range(3):
            var left_model = generated_model(shape[0], 3, validity)
            var right_model = generated_model(shape[1], 5, validity)
            for layout in range(3):
                var left = frame_from_model(left_model, 7 if layout == 1 else 0)
                var right = frame_from_model(
                    right_model, 1 if layout == 1 else 0
                )
                if layout == 2:
                    left = chunk_layout(left)
                    right = chunk_layout(right)
                assert_join_model(
                    left.join(right, how="cross"),
                    left_model,
                    right_model,
                    [],
                    "cross",
                )
                for keys in key_sets:
                    for how in [
                        "inner",
                        "left",
                        "right",
                        "full",
                        "semi",
                        "anti",
                    ]:
                        var result = left.join(right, keys, how=how)
                        assert_join_model(
                            result, left_model, right_model, keys, how
                        )
                        if how == "full":
                            assert_join_model(
                                left.join(right, keys, how=how, coalesce=False),
                                left_model,
                                right_model,
                                keys,
                                how,
                                coalesce=False,
                            )


def test_join_float_specials_and_validation() raises:
    var rows = List[ReferenceRow]()
    var values: List[Float64] = [
        0.0,
        -0.0,
        Float64(0) / Float64(0),
        Float64(0) / Float64(0),
        9.0,
    ]
    for index in range(len(values)):
        rows.append(
            ReferenceRow(
                Int64(index),
                values[index],
                String(index),
                False,
                True,
                index != 4,
                True,
                True,
            )
        )
    var model = ReferenceTable(["i", "f", "s", "b"], rows^)
    var frame = frame_from_model(model, 7)
    for how in ["inner", "left", "right", "full", "semi", "anti"]:
        assert_join_model(
            frame.join(frame, "f", how=how), model, model, ["f"], how
        )
    var invalid_keys: List[List[String]] = [[], ["missing"], ["i", "i"]]
    for keys in invalid_keys:
        with assert_raises():
            _ = frame.join(frame, keys)
    with assert_raises():
        _ = frame.join(frame, "i", how="unknown")
    with assert_raises():
        _ = frame.join(frame, left_on=["i"], right_on=["s"])
    with assert_raises():
        _ = frame.join(frame, left_on=["i", "s"], right_on=["i"])
    with assert_raises():
        _ = frame.join(frame, "i", suffix="")
    var right = frame.rename({"i": "key"})
    var different = frame.join(right, left_on=["i"], right_on=["key"])
    var same = frame.join(frame, "i")
    assert_equal(different.columns(), same.columns())
    # Full expected data comes from the oracle, rather than the sibling join.
    assert_join_model(different, model, model, ["i"], "inner")


def test_a_coalesced_right_join_keeps_the_right_key_names() raises:
    # As Polars (a left join with the sides swapped): the left's non-key
    # columns, then every right column, keys included under their own
    # names; a right name the left's other columns hold takes the suffix.
    var left = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 3])),
            Series("bk", Column[Int64]([7, 8, 9])),
            Series("x", Column[Int64]([10, 20, 30])),
        ]
    )
    var right = DataFrame(
        [
            Series("x", Column[String](["a", "b", "c"])),
            Series("bk", Column[Int64]([4, 3, 2])),
            Series("k", Column[Int64]([0, 0, 0])),
        ]
    )
    var names: List[String] = ["bk", "x", "x_right", "bk_right", "k"]
    for lazy in [False, True]:
        var out = (
            left.lazy()
            .join(right.lazy(), left_on=["k"], right_on=["bk"], how="right")
            .collect() if lazy else left.join(
                right, left_on=["k"], right_on=["bk"], how="right"
            )
        )
        assert_equal(out.columns(), names)
        out = out.sort("bk_right")
        assert_equal(out.item(0, "bk_right").int64(), 2)
        assert_equal(out.item(0, "x").int64(), 20)
        assert_equal(out.item(0, "x_right").string(), "c")
        assert_true(out.item(2, "x").is_null())
        assert_equal(out.item(2, "bk_right").int64(), 4)
        # Selecting the right key reads it from the right input.
        var picked = (
            left.lazy()
            .join(right.lazy(), left_on=["k"], right_on=["bk"], how="right")
            .select(["bk_right", "k"])
            .collect()
        )
        assert_equal(picked.columns(), List[String](["bk_right", "k"]))
        assert_equal(picked.height(), 3)
        # The lazy join runs as the right join keeping both keys, which
        # streams, with the left keys dropped.
        var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
        var observed = StrategyTrace()
        var traced = (
            left.lazy()
            .join(right.lazy(), left_on=["k"], right_on=["bk"], how="right")
            .collect()
        )
        assert_true(
            "dataframe-path: lazy.lower_right_join\n" in observed.text(),
            "the right join was not lowered",
        )
        _ = observed^
        _ = trace^
        assert_equal(traced.columns(), names)
        # Keys named alike come out once, after the left's other columns.
        var same = left.drop(["bk"]).join(
            right.drop(["k"]).rename({"bk": "k"}), on="k", how="right"
        )
        assert_equal(same.columns(), List[String](["x", "x_right", "k"]))


def test_a_coalesced_right_join_streams_in_every_mode() raises:
    # The eager join, and a plan run without the optimizer, also run as the
    # right join keeping both keys. A left column named as the kept-keys
    # join would name a right key ("k_right") does not keep it from
    # streaming: that join's own suffix keeps its names apart.
    var left = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 3, 3])),
            Series("k_right", Column[Int64]([7, 8, 9, 9])),
            Series("x", Column[Int64]([10, 20, 30, 31])),
        ]
    )
    var right = DataFrame(
        [
            Series("x", Column[String](["a", "b", "c", "d"])),
            Series("bk", Column[Int64]([4, 3, 2, 3])),
            Series("k", Column[Int64]([5, 6, 7, 8])),
        ]
    )
    var names: List[String] = ["k_right", "x", "x_right", "bk", "k"]
    var expected: List[String] = [
        "8|20|c|2|7",
        "9|30|b|3|6",
        "9|30|d|3|8",
        "9|31|b|3|6",
        "9|31|d|3|8",
        "null|null|a|4|5",
    ]
    for empty_left in [False, True]:
        var probe = left.clear() if empty_left else left.copy()
        for mode in range(3):
            var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
            var observed = StrategyTrace()
            var out: DataFrame
            if mode == 0:
                out = probe.join(
                    right, left_on=["k"], right_on=["bk"], how="right"
                )
            else:
                out = (
                    probe.lazy()
                    .join(
                        right.lazy(),
                        left_on=["k"],
                        right_on=["bk"],
                        how="right",
                    )
                    .collect(optimize=mode == 1)
                )
            var text = observed.text()
            _ = observed^
            _ = trace^
            var label = "mode=" + String(mode) + " empty=" + String(empty_left)
            assert_true(
                "dataframe-path: lazy.lower_right_join\n" in text, label
            )
            assert_equal(out.columns(), names, label)
            var rows = _rows(out, names)
            if empty_left:
                assert_equal(len(rows), 4, label)
                for row in rows:
                    assert_true(row.startswith("null|null|"), label)
            else:
                assert_equal(rows, expected, label)


def _decimal_key_frame(
    keys: List[Int], valid: List[Bool], width: Int, value: String, start: Int
) raises -> DataFrame:
    """Key `d` as decimal(10, 2) stored in `width` bits, a value column
    named `value` counting from `start`, and a float `f` to filter on."""
    var values = List[Int64]()
    var floats = List[Float64]()
    for i in range(len(keys)):
        values.append(Int64(start + i))
        floats.append(Float64(i))
    var key: Series
    if width == 64:
        var stored = List[Int64]()
        for k in keys:
            stored.append(Int64(k))
        key = Series("d", Column[Int64](stored^, valid.copy())).with_dtype(
            DataType.decimal(10, 2, 64)
        )
    else:
        var stored = List[Int128]()
        for k in keys:
            stored.append(Int128(k))
        key = Series("d", Column[Int128](stored^, valid.copy())).with_dtype(
            DataType.decimal(10, 2)
        )
    return DataFrame(
        [
            key^,
            Series(value, Column[Int64](values^)),
            Series("f", Column[Float64](floats^)),
        ]
    )


def _rows(frame: DataFrame, names: List[String]) raises -> List[String]:
    """Each row's cells in `names` order joined by "|", sorted, so results
    compare without depending on row order."""
    var out = List[String]()
    for row in range(frame.height()):
        var cells = List[String]()
        for name in names:
            cells.append(String(frame.item(row, name)))
        out.append(String("|").join(cells))
    sort(out)
    return out^


def _sorted(var names: List[String]) -> List[String]:
    sort(names)
    return names^


def test_decimal_join_keys_of_one_precision_and_scale_join_across_widths() raises:
    """decimal(10, 2) held as 64 and as 128 bits (each file declares its
    own width) is one key type: the keys join on value at the wider width,
    for every join kind, with and without coalescing, eagerly and lazily
    (optimized, unoptimized, streaming or not), with either side narrow
    and a filter below the wide side. Semi and anti joins keep the left
    rows' own type. Found by the differential fuzzer (tests/fuzz)."""
    var wide_type = DataType.decimal(10, 2)
    # Narrow side keys 1.00, 2.50, null; wide side 2.50, 1.00, 1.00,
    # 7.00, null. Nulls never match.
    var narrow = _decimal_key_frame(
        [100, 250, 0], [True, True, False], 64, "l", 1
    )
    var wide = _decimal_key_frame(
        [250, 100, 100, 700, 0], [True, True, True, True, False], 128, "r", 10
    )
    var matched: List[String] = ["1.00|1|11", "1.00|1|12", "2.50|2|10"]
    var left_only: List[String] = ["null|3|null"]
    var right_only: List[String] = ["7.00|null|13", "null|null|14"]
    for narrow_left in [True, False]:
        var left = narrow.copy() if narrow_left else wide.copy()
        var right = wide.drop(["f"]) if narrow_left else narrow.drop(["f"])
        var right_lazy = (
            wide.lazy()
            .filter(col("f") > -1.0)
            .drop(["f"]) if narrow_left else narrow.lazy()
            .filter(col("f") > -1.0)
            .drop(["f"])
        )
        left = left.drop(["f"])
        for how in ["inner", "left", "right", "full"]:
            var wanted = matched.copy()
            var keeps_left = how == "left" or how == "full"
            var keeps_right = how == "right" or how == "full"
            if (keeps_left and narrow_left) or (
                keeps_right and not narrow_left
            ):
                wanted.extend(left_only.copy())
            if (keeps_right and narrow_left) or (
                keeps_left and not narrow_left
            ):
                wanted.extend(right_only.copy())
            for coalesce in [True, False]:
                for mode in range(4):
                    var label = (
                        String(how)
                        + " coalesce="
                        + String(coalesce)
                        + " narrow_left="
                        + String(narrow_left)
                        + " mode="
                        + String(mode)
                    )
                    var out: DataFrame
                    if mode == 0:
                        out = left.join(
                            right, on="d", how=how, coalesce=coalesce
                        )
                    else:
                        out = (
                            left.lazy()
                            .join(
                                right_lazy, on="d", how=how, coalesce=coalesce
                            )
                            .collect(optimize=mode != 2, streaming=mode != 3)
                        )
                    assert_equal(out.column("d").dtype(), wide_type, label)
                    var names: List[String] = ["d", "l", "r"]
                    if not coalesce:
                        assert_equal(
                            out.column("d_right").dtype(), wide_type, label
                        )
                        assert_equal(
                            _sorted(out.columns()),
                            List[String](["d", "d_right", "l", "r"]),
                            label,
                        )
                        # The coalesced key is the non-null of the two.
                        out = out.with_columns(
                            [col("d").fill_null(col("d_right")).alias("d")]
                        )
                    else:
                        assert_equal(_sorted(out.columns()), names, label)
                    assert_equal(
                        _rows(out, names), _sorted(wanted.copy()), label
                    )
        for how in ["semi", "anti"]:
            for lazy in [False, True]:
                var out = (
                    left.lazy()
                    .join(right_lazy, on="d", how=how)
                    .collect() if lazy else left.join(right, on="d", how=how)
                )
                assert_equal(
                    out.column("d").dtype(), left.column("d").dtype(), how
                )
                var count = 2 if narrow_left else 3
                if how == "anti":
                    count = 1 if narrow_left else 2
                assert_equal(out.height(), count, how)
    # Decimals of other scales match by value at a common type, and the
    # output key keeps the left side's declared type.
    var other_scale = wide.with_column(
        wide.column("d").cast(DataType.decimal(10, 3))
    )
    var scaled = narrow.join(other_scale, on="d")
    assert_equal(scaled.column("d").dtype(), narrow.column("d").dtype())
    assert_equal(scaled.height(), narrow.join(wide, on="d").height())
    var text = wide.with_column(wide.column("d").cast(DataType.STRING))
    with assert_raises(contains="Join key dtypes differ"):
        _ = narrow.join(text, on="d")


def test_decimal_join_keys_across_widths_with_an_empty_side() raises:
    """Shrunk fuzzer cases: an empty 128-bit side against a 64-bit side,
    for a full join (the one null key row survives) and a right join
    keeping both keys under different names; and an inner join whose
    64-bit side is filtered. Each runs eagerly and lazily, optimized or
    not."""
    var empty = DataFrame(
        [
            Series("d", Column[Int128](List[Int128]())).with_dtype(
                DataType.decimal(10, 2)
            )
        ]
    )
    var one_null = DataFrame(
        [
            Series("d", Column[Int64]([0], [False])).with_dtype(
                DataType.decimal(10, 2, 64)
            )
        ]
    )
    var named = DataFrame(
        [
            Series("e", Column[Int64]([-197])).with_dtype(
                DataType.decimal(10, 2, 64)
            ),
            Series("x", Column[Int64]([5])),
        ]
    )
    var single = DataFrame(
        [Series("d", Column[Int128]([123])).with_dtype(DataType.decimal(10, 2))]
    )
    var filtered = DataFrame(
        [
            Series(
                "d",
                Column[Int64]([123, 0, -167, -196], [True, False, True, True]),
            ).with_dtype(DataType.decimal(10, 2, 64)),
            Series("f", Column[Float64]([0.0, 63.5, 0.0, -0.5])),
        ]
    )
    for mode in range(3):
        var full: DataFrame
        var right: DataFrame
        var inner: DataFrame
        if mode == 0:
            full = empty.join(one_null, on="d", how="full", coalesce=True)
            right = empty.join(
                named,
                left_on=["d"],
                right_on=["e"],
                how="right",
                coalesce=False,
            )
            inner = single.join(
                filtered.filter(col("f") > -1.5), on="d", how="inner"
            )
        else:
            var optimize = mode == 1
            full = (
                empty.lazy()
                .join(one_null.lazy(), on="d", how="full", coalesce=True)
                .collect(optimize=optimize)
            )
            right = (
                empty.lazy()
                .join(
                    named.lazy(),
                    left_on=["d"],
                    right_on=["e"],
                    how="right",
                    coalesce=False,
                )
                .collect(optimize=optimize)
            )
            inner = (
                single.lazy()
                .join(
                    filtered.lazy().filter(col("f") > -1.5), on="d", how="inner"
                )
                .collect(optimize=optimize)
            )
        var label = "mode=" + String(mode)
        assert_equal(full.height(), 1, label)
        assert_true(full.item(0, "d").is_null(), label)
        assert_equal(full.column("d").dtype(), DataType.decimal(10, 2), label)
        assert_equal(right.height(), 1, label)
        assert_true(right.item(0, "d").is_null(), label)
        assert_equal(String(right.item(0, "e")), "-1.97", label)
        assert_equal(right.item(0, "x").int64(), 5, label)
        assert_equal(right.column("e").dtype(), DataType.decimal(10, 2), label)
        assert_equal(inner.height(), 1, label)
        assert_equal(String(inner.item(0, "d")), "1.23", label)
