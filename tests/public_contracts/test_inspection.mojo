"""Inspection preserves typed rows, bounded text and specified summary layouts."""
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    ListColumn,
    Series,
    StructColumn,
    col,
)
from dataframe import (
    all as all_columns,
    first as first_column,
    last as last_column,
)
from dataframe import exclude, by_dtype, nth
from tests.support.reference_model import generated_model
from tests.support.coverage_generators import (
    frame_from_model,
    offset_layout,
    chunk_layout,
)
from tests.support.coverage_compare import (
    assert_model,
    assert_reference_cell,
    assert_float,
    assert_frame,
)


def test_row_and_item_access_preserve_schema_types_validity_and_bounds() raises:
    for n in [0, 1, 9]:
        var model = generated_model(n, 7, 1)
        for layout in range(3):
            var frame = frame_from_model(model, 7 if layout == 1 else 0)
            if layout == 2:
                frame = chunk_layout(frame)
            var rows = frame.rows()
            assert_equal(len(rows), n)
            for row in range(n):
                var one = frame.row(row)
                assert_equal(len(one), len(model.names))
                assert_equal(len(rows[row]), len(model.names))
                for column in range(len(model.names)):
                    var name = model.names[column]
                    var dtype = (
                        DataType.INT64 if name
                        == "i" else DataType.FLOAT64 if name
                        == "f" else DataType.STRING if name
                        == "s" else DataType.BOOL
                    )
                    assert_true(one[column].dtype() == dtype)
                    assert_true(rows[row][column].dtype() == dtype)
                    assert_true(frame.item(row, name).dtype() == dtype)
                    assert_reference_cell(one[column], model.rows[row], name)
                    assert_reference_cell(
                        rows[row][column], model.rows[row], name
                    )
                    assert_reference_cell(
                        frame.item(row, name), model.rows[row], name
                    )
            for index in [-1, n, n + 1]:
                with assert_raises():
                    _ = frame.row(index)
                with assert_raises():
                    _ = frame.item(index, "i")
            with assert_raises():
                _ = frame.item()
            with assert_raises():
                _ = frame.item(0, "missing")
    var zero = DataFrame(List[Series](), height=3)
    assert_equal(len(zero.rows()), 3)
    for row in range(3):
        assert_equal(len(zero.row(row)), 0)
    var singleton = DataFrame([Series("i", Column[Int64]([42]))])
    assert_equal(singleton.item().int64(), Int64(42))
    with assert_raises():
        _ = singleton.item().float64()
    with assert_raises():
        _ = Series.full_null("i", DataType.INT64, 1).get(0).int64()


def test_selectors_expand_names_in_schema_order_and_validate_empty_frames() raises:
    var model = generated_model(9, 1, 1)
    for layout in range(3):
        var frame = frame_from_model(model, 1 if layout == 1 else 0)
        if layout == 2:
            frame = chunk_layout(frame)
        for empty in [False, True]:
            var input = frame.clear() if empty else frame.copy()
            var expected = model.take(List[Int]()) if empty else model.copy()
            assert_model(input.select(all_columns()), expected)
            assert_model(
                input.select(col(["s", "i"])), expected.project(["s", "i"])
            )
            assert_model(
                input.select(exclude(["f", "b"])), expected.project(["i", "s"])
            )
            assert_model(
                input.select(by_dtype(["bool", "int64"])),
                expected.project(["i", "b"]),
            )
            assert_model(input.select(first_column()), expected.project(["i"]))
            assert_model(input.select(last_column()), expected.project(["b"]))
            for position in range(4):
                assert_model(
                    input.select(nth(position)),
                    expected.project([model.names[position]]),
                )
                assert_model(
                    input.select(nth(position - 4)),
                    expected.project([model.names[position]]),
                )
            var renamed = input.select(
                col(["s", "i"]).name_prefix("pre_").name_suffix("_post")
            )
            assert_equal(renamed.columns(), ["pre_s_post", "pre_i_post"])
            assert_equal(renamed.height(), input.height())
            assert_true(renamed.column("pre_s_post").dtype() == DataType.STRING)
            assert_true(renamed.column("pre_i_post").dtype() == DataType.INT64)
            for row in range(input.height()):
                assert_reference_cell(
                    renamed.item(row, "pre_s_post"), model.rows[row], "s"
                )
                assert_reference_cell(
                    renamed.item(row, "pre_i_post"), model.rows[row], "i"
                )
            var changed = input.with_columns(by_dtype(["int64"]) + 1)
            var changed_model = expected.copy()
            for row in range(len(changed_model.rows)):
                if changed_model.rows[row].iv:
                    changed_model.rows[row].i += 1
            assert_model(changed, changed_model)
            var bad: List[Expr] = [
                nth(4),
                nth(-5),
                col(["missing"]),
                by_dtype(["invalid dtype"]),
                all_columns().alias("duplicate"),
                col(["i", "i"]),
            ]
            for expression in bad:
                with assert_raises():
                    _ = input.select(expression)
            with assert_raises():
                _ = input.filter(by_dtype(["int64", "float64"]) > 0)
    var zero = DataFrame(List[Series](), height=3)
    for expression in [first_column(), last_column(), nth(0)]:
        with assert_raises():
            _ = zero.select(expression)


def test_empty_selector_expansion_preserves_input_height() raises:
    var frame = frame_from_model(generated_model(9))
    var mismatches = List[String]()
    var expressions: List[Expr] = [
        by_dtype(["uint8"]),
        exclude(["i", "f", "s", "b"]),
        col(List[String]()),
    ]
    for index in range(len(expressions)):
        var result = frame.select(expressions[index])
        assert_equal(result.width(), 0)
        if result.height() != frame.height():
            mismatches.append(String(index))
    assert_true(
        len(mismatches) == 0,
        "empty selector expansion must preserve input height: "
        + String(mismatches),
    )


def test_all_selector_preserves_zero_column_frame_height() raises:
    var result = DataFrame(List[Series](), height=3).select(all_columns())
    assert_equal(result.width(), 0)
    assert_true(
        result.height() == 3,
        "all selector must preserve zero-column input height",
    )


def test_display_has_literal_shape_dtype_escaping_and_codepoint_limits() raises:
    var simple = DataFrame([Series("x", Column[Int64]([1, 2]))])
    var expected = "shape: (2, 1)\n┌─────┐\n│ x   │\n│ --- │\n│ i64 │\n╞═════╡\n│ 1   │\n│ 2   │\n└─────┘"
    assert_equal(simple.to_string(), expected)
    assert_equal(String(simple), expected)
    assert_equal(
        DataFrame(List[Series](), height=3).to_string(), "shape: (3, 0)\n┌┐\n└┘"
    )
    var original = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    [
                        "",
                        "null",
                        " a",
                        "a ",
                        "a\tb",
                        "\n",
                        "é🙂z",
                        "hidden payload",
                    ],
                    [True, True, True, True, True, True, True, False],
                ),
            )
        ]
    )
    for layout in range(3):
        var frame = (
            original.copy() if layout
            == 0 else offset_layout(original) if layout
            == 1 else chunk_layout(original)
        )
        var text = frame.to_string(
            max_rows=-1, max_columns=-1, max_string_length=-1
        )
        for token in [
            String('""'),
            String('"null"'),
            String('" a"'),
            String('"a "'),
            String('"a\\tb"'),
            String('"\\n"'),
            String("é🙂z"),
        ]:
            assert_true(text.find(token) >= 0)
        assert_equal(text.find("hidden payload"), -1)
        var clipped = frame.to_string(max_rows=-1, max_string_length=2)
        assert_true(clipped.find("é…") >= 0)
        assert_equal(clipped.find("é🙂z"), -1)
        var series = frame.column("s").to_string(
            max_rows=3, max_string_length=-1
        )
        assert_true(series.find("shape: (8,)\nSeries: s [str]\n[") == 0)
        assert_true(series.find('""') >= 0)
        assert_true(series.find('"null"') >= 0)
        assert_true(series.find("…") >= 0)
        assert_equal(series.find("é🙂z"), -1)
    assert_equal(
        simple.glimpse(max_width=-1), "Rows: 2\nColumns: 1\n$ x <i64> 1, 2"
    )


def test_display_odd_limits_show_extra_front_row_and_column() raises:
    var columns = List[Series]()
    for column in range(18):
        var values = List[String]()
        for row in range(13):
            values.append("c" + String(column) + "r" + String(row) + ".")
        columns.append(
            Series("column_" + String(column), Column[String](values^))
        )
    var frame = DataFrame(columns^)
    var bounded = frame.to_string(
        max_rows=3, max_columns=3, max_string_length=-1
    )
    for token in [
        "column_0",
        "column_1",
        "column_17",
        "c0r0.",
        "c0r1.",
        "c0r12.",
        "c17r12.",
        "…",
    ]:
        assert_true(bounded.find(token) >= 0)
    for token in ["column_2", "column_16", "c0r2.", "c0r11."]:
        assert_equal(bounded.find(token), -1)
    var none = frame.to_string(max_rows=0, max_columns=0)
    assert_true(none.find("shape: (13, 18)") >= 0)
    assert_equal(none.find("column_0"), -1)
    assert_equal(none.find("c0r0."), -1)
    var all = frame.to_string(max_rows=-1, max_columns=-1, max_string_length=-1)
    assert_true(all.find("c8r6.") >= 0)
    assert_equal(all.find("…"), -1)


def test_describe_orders_percentiles_and_preserves_typed_statistic_policies() raises:
    var original = DataFrame(
        [
            Series(
                "i", Column[Int64]([1, 2, 3, 99], [True, True, True, False])
            ),
            Series(
                "b",
                Column[Bool](
                    [False, True, True, False], [True, True, True, False]
                ),
            ),
            Series(
                "s",
                Column[String](
                    ["z", "", "é", "poison"], [True, True, True, False]
                ),
            ),
        ]
    )
    var labels: List[String] = [
        "count",
        "null_count",
        "mean",
        "std",
        "min",
        "0%",
        "25%",
        "50%",
        "75%",
        "100%",
        "max",
    ]
    var numbers: List[Float64] = [3, 1, 2, 1, 1, 1, 1.5, 2, 2.5, 3, 3]
    for layout in range(3):
        var frame = (
            original.copy() if layout
            == 0 else offset_layout(original) if layout
            == 1 else chunk_layout(original)
        )
        var result = frame.describe([1, 0.5, 0, 0.75, 0.25], "linear")
        assert_equal(result.columns(), ["statistic", "i", "b", "s"])
        assert_equal(result.height(), 11)
        assert_true(result.column("i").dtype() == DataType.FLOAT64)
        assert_true(result.column("b").dtype() == DataType.FLOAT64)
        assert_true(result.column("s").dtype() == DataType.STRING)
        for row in range(11):
            assert_equal(result.item(row, "statistic").string(), labels[row])
            assert_float(result.item(row, "i").float64(), numbers[row])
            var bool_valid = row <= 2 or row == 4 or row == 10
            assert_equal(result.item(row, "b").is_null(), not bool_valid)
            if bool_valid:
                var boolean: List[Float64] = [
                    3,
                    1,
                    Float64(2) / 3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ]
                assert_float(result.item(row, "b").float64(), boolean[row])
            var string_valid = row < 2 or row == 4 or row == 10
            assert_equal(result.item(row, "s").is_null(), not string_valid)
            if string_valid:
                assert_equal(
                    result.item(row, "s").string(),
                    "3" if row
                    == 0 else "1" if row
                    == 1 else "" if row
                    == 4 else "é",
                )
        var empty = frame.clear().describe()
        for name in ["i", "b"]:
            assert_equal(empty.item(0, name).float64(), Float64(0))
            assert_equal(empty.item(1, name).float64(), Float64(0))
            for row in range(2, empty.height()):
                assert_true(empty.item(row, name).is_null())
        assert_equal(frame.describe(List[Float64]()).height(), 6)
        assert_equal(
            DataFrame(List[Series](), height=4).describe().columns(),
            ["statistic"],
        )
        for q in [-0.1, 1.1, Float64(0) / Float64(0)]:
            with assert_raises():
                _ = frame.describe([q])


def test_sample_preserves_complete_rows_counts_uniqueness_order_and_seed() raises:
    var model = generated_model(13, 2, 1)
    var original = frame_from_model(model).with_row_index()
    var seen_seeds = List[Int64]()
    for seed in [0, 1, 2, 17]:
        var index = original.sample(1, seed=seed).item(0, "index").int64()
        var seen = False
        for prior in seen_seeds:
            seen = seen or prior == index
        if not seen:
            seen_seeds.append(index)
    assert_true(len(seen_seeds) > 1)
    for layout in range(3):
        var frame = (
            original.copy() if layout
            == 0 else offset_layout(original) if layout
            == 1 else chunk_layout(original)
        )
        for replacement in [False, True]:
            for shuffle in [False, True]:
                for n in [0, 1, 7, 13]:
                    var actual = frame.sample(
                        n,
                        with_replacement=replacement,
                        shuffle=shuffle,
                        seed=17,
                    )
                    assert_equal(actual.height(), n)
                    assert_frame(
                        actual,
                        frame.sample(
                            n,
                            with_replacement=replacement,
                            shuffle=shuffle,
                            seed=17,
                        ),
                    )
                    var indices = List[Int]()
                    var prior = -1
                    for row in range(n):
                        var index = Int(actual.item(row, "index").int64())
                        assert_true(index >= 0 and index < 13)
                        if not shuffle:
                            assert_true(index >= prior)
                        if not replacement:
                            for seen in indices:
                                assert_true(index != seen)
                        indices.append(index)
                        prior = index
                    assert_model(actual.drop(["index"]), model.take(indices))
                    var sampled_series = frame.column("index").sample(
                        n,
                        with_replacement=replacement,
                        shuffle=shuffle,
                        seed=17,
                    )
                    assert_equal(len(sampled_series), n)
                    for row in range(n):
                        assert_equal(
                            sampled_series.get(row).int64(),
                            actual.item(row, "index").int64(),
                        )
        assert_equal(frame.sample(fraction=0.5, seed=1).height(), 6)
        assert_equal(frame.sample(seed=1).height(), 1)
        assert_equal(
            frame.sample(100, with_replacement=True, seed=1).height(), 100
        )
        assert_equal(frame.clear().sample(0).height(), 0)
        for count in [-1, 14]:
            with assert_raises():
                _ = frame.sample(count)
        with assert_raises():
            _ = frame.sample(1, fraction=0.5)
        with assert_raises():
            _ = frame.sample(fraction=-0.1)
        with assert_raises():
            _ = frame.clear().sample(1, with_replacement=True)
    assert_equal(
        DataFrame(List[Series](), height=13).sample(7, seed=0).shape(), (7, 0)
    )


def test_describe_temporal_columns_follow_literal_summary_policies() raises:
    var types: List[DataType] = [
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("ms"),
        DataType.datetime("us", "UTC"),
        DataType.duration("ns"),
    ]
    var lows: List[String] = [
        "1969-12-31",
        "00:00:00",
        "1969-12-31 23:59:59",
        "1969-12-31 23:59:59+00:00",
        "-1s",
    ]
    var means: List[String] = [
        "1970-01-01 00:00:00",
        "00:00:01",
        "1970-01-01 00:00:00",
        "1970-01-01 00:00:00+00:00",
        "0s",
    ]
    var highs: List[String] = [
        "1970-01-02",
        "00:00:02",
        "1970-01-01 00:00:01",
        "1970-01-01 00:00:01+00:00",
        "1s",
    ]
    for i in range(len(types)):
        var scale = Int64(
            1 if i
            == 0 else 1000 if i
            == 2 else 1000000 if i
            == 3 else 1000000000
        )
        var low = Int64(0) if i == 1 else -scale
        var high = 2 * scale if i == 1 else scale
        for all_null in [False, True]:
            var source = DataFrame(
                [
                    Series(
                        "t",
                        Column[Int64](
                            [low, high, 999],
                            [not all_null, not all_null, False],
                        ),
                    ).with_dtype(types[i])
                ]
            )
            for layout in range(3):
                var frame = (
                    source.copy() if layout
                    == 0 else offset_layout(source) if layout
                    == 1 else chunk_layout(source)
                )
                var actual = frame.describe([0, 0.5, 1], "linear")
                assert_equal(actual.columns(), ["statistic", "t"])
                assert_true(actual.column("t").dtype() == DataType.STRING)
                assert_equal(
                    actual.item(0, "t").string(), "0" if all_null else "2"
                )
                assert_equal(
                    actual.item(1, "t").string(), "3" if all_null else "1"
                )
                var expected: List[String] = [
                    means[i],
                    "",
                    lows[i],
                    lows[i],
                    means[i],
                    highs[i],
                    highs[i],
                ]
                # A Date mean is a datetime; its median retains the Date type.
                if i == 0:
                    expected[4] = "1970-01-01"
                for row in range(2, 9):
                    var present = not all_null and row != 3
                    assert_equal(actual.item(row, "t").is_null(), not present)
                    if present:
                        assert_equal(
                            actual.item(row, "t").string(), expected[row - 2]
                        )


def _assert_nested_description(nested: DataFrame) raises:
    for layout in range(3):
        var frame = (
            nested.copy() if layout
            == 0 else offset_layout(nested) if layout
            == 1 else chunk_layout(nested)
        )
        for empty in [False, True]:
            var actual = (frame.clear() if empty else frame.copy()).describe(
                [0, 0.5, 1]
            )
            assert_equal(actual.columns(), ["statistic", "nested"])
            assert_equal(actual.height(), 9)
            assert_true(actual.column("nested").dtype() == DataType.FLOAT64)
            assert_equal(
                actual.item(0, "nested").float64(), Float64(0 if empty else 2)
            )
            assert_equal(
                actual.item(1, "nested").float64(), Float64(0 if empty else 1)
            )
            for row in range(2, 9):
                assert_true(actual.item(row, "nested").is_null())


def test_describe_lists_reports_only_parent_counts() raises:
    _assert_nested_description(
        DataFrame(
            [
                Series(
                    "nested",
                    ListColumn(
                        [0, 1, 1, 3],
                        Series("child", Column[Int64]([1, 2, 3])),
                        [UInt8(0b101)],
                    ),
                )
            ]
        )
    )


def test_describe_structs_reports_only_parent_counts() raises:
    _assert_nested_description(
        DataFrame(
            [
                Series(
                    "nested",
                    StructColumn(
                        [Series("x", Column[Int64]([1, 999, 3]))],
                        [UInt8(0b101)],
                    ),
                )
            ]
        )
    )


def test_describe_numeric_nonfinite_values_distinguish_nan_from_null() raises:
    var nan = Float64(0) / Float64(0)
    var infinity = Float64(1) / Float64(0)
    for mode in range(4):
        var values: List[Float64] = [nan, 1, 3, 999]
        if mode == 1:
            values = [nan, nan, nan, 999]
        elif mode == 2:
            values = [-infinity, 1, infinity, 999]
        elif mode == 3:
            values = [-0.0, 0.0, nan, 999]
        var valid: List[Bool] = [True, True, mode != 3, False]
        var source = DataFrame([Series("x", Column[Float64](values^, valid^))])
        for layout in range(3):
            var frame = (
                source.copy() if layout
                == 0 else offset_layout(source) if layout
                == 1 else chunk_layout(source)
            )
            var actual = frame.describe(List[Float64]())
            assert_true(actual.column("x").dtype() == DataType.FLOAT64)
            assert_equal(
                actual.item(0, "x").float64(), Float64(2 if mode == 3 else 3)
            )
            assert_equal(
                actual.item(1, "x").float64(), Float64(2 if mode == 3 else 1)
            )
            for row in [2, 3, 4, 5]:
                assert_true(not actual.item(row, "x").is_null())
            if mode < 3:
                assert_true(isnan(actual.item(2, "x").float64()))
                assert_true(isnan(actual.item(3, "x").float64()))
            else:
                assert_equal(actual.item(2, "x").float64(), Float64(0))
                assert_equal(actual.item(3, "x").float64(), Float64(0))
            var lo = (
                nan if mode
                == 1 else -infinity if mode
                == 2 else Float64(0) if mode
                == 3 else Float64(1)
            )
            var hi = (
                nan if mode
                == 1 else infinity if mode
                == 2 else Float64(0) if mode
                == 3 else Float64(3)
            )
            # describe promises numerical zero, not a particular zero sign.
            if mode == 3:
                assert_equal(actual.item(4, "x").float64(), Float64(0))
                assert_equal(actual.item(5, "x").float64(), Float64(0))
            else:
                assert_float(actual.item(4, "x").float64(), lo)
                assert_float(actual.item(5, "x").float64(), hi)
