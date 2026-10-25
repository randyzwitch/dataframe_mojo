"""LazyFrame table operations match their eager DataFrame methods."""
# contract-group: lazy.table_ops
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    Expr,
    LazyFrame,
    Series,
    StringColumn,
    col,
    lit,
)
from tests.support.reference_model import generated_model, reference_groups
from tests.support.reference_relations import nonnull_indices
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import assert_frame, assert_model


def collected(plan: LazyFrame) raises -> List[DataFrame]:
    """The plan collected with optimization and streaming each on and off;
    streaming uses small batches so several of them reach each operator."""
    var out = List[DataFrame]()
    for optimize in [False, True]:
        for streaming in [False, True]:
            out.append(
                plan.collect(
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=4 if streaming else 65536,
                    engine="cpu",
                )
            )
    return out^


def assert_lazy(plan: LazyFrame, expected: DataFrame) raises:
    for actual in collected(plan):
        assert_frame(actual, expected)
    # The schema computed without rows is the result's.
    var schema = List[String]()
    for field in expected.schema():
        schema.append(field.name + ": " + field.dtype.name())
    assert_equal(plan.collect_schema(), schema)


def model_frames(n: Int, mode: Int) raises -> List[DataFrame]:
    """One logical table in two storage layouts: offset into its
    buffers, and split into chunks at different rows per column."""
    var model = generated_model(n, 3, mode)
    var frame = frame_from_model(model, 7)
    return [frame.copy(), chunk_layout(frame_from_model(model))]


def case_rows(shape: Int) -> Int:
    """Rows of each generated table: empty, small and several batches."""
    var rows: List[Int] = [0, 65, 9]
    return rows[shape]


def case_nulls(shape: Int) -> Int:
    """Null pattern of each generated table (see generated_model)."""
    var modes: List[Int] = [0, 1, 2]
    return modes[shape]


def nullable_frame() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "i",
                Column[Int64](
                    [1, 0, 3, 0, 5], [True, False, True, False, True]
                ),
            ),
            Series(
                "f",
                Column[Float64](
                    [0.5, 1.5, 0.0, 3.5, 0.0], [True, True, False, True, False]
                ),
            ),
            Series(
                "s",
                Column[String](
                    ["a", "", "c", "d", ""], [True, False, True, True, False]
                ),
            ),
            Series(
                "b",
                Column[Bool](
                    [True, False, False, True, False],
                    [True, True, True, True, False],
                ),
            ),
        ]
    )


def test_lazy_unpivot_matches_eager_and_orders_by_column_then_row() raises:
    for shape in range(3):
        for mode in [case_nulls(shape)]:
            var n = case_rows(shape)
            for frame in model_frames(n, mode):
                var j = frame.with_columns([(col("i") * 2).alias("j")])
                var eager = j.unpivot(
                    ["j", "i"],
                    ["s", "b"],
                    variable_name="source",
                    value_name="value",
                )
                assert_equal(eager.columns(), ["s", "b", "source", "value"])
                var lazy = j.lazy().unpivot(
                    ["j", "i"],
                    ["s", "b"],
                    variable_name="source",
                    value_name="value",
                )
                assert_lazy(lazy, eager)
                # Default `on`: every non-index column, as eager.
                assert_lazy(
                    j.lazy().select(["i", "j", "s"]).unpivot(index=["s"]),
                    j.select(["i", "j", "s"]).unpivot(index=["s"]),
                )
                # Composed: steps below feed it, and a filter, projection
                # and slice above see its rows, not its input's.
                var below = (
                    j.lazy()
                    .filter(col("b").is_not_null())
                    .with_columns([(col("j") + 1).alias("j")])
                )
                var expected_below = j.filter(
                    col("b").is_not_null()
                ).with_columns([(col("j") + 1).alias("j")])
                assert_lazy(
                    below.unpivot(["i", "j"], ["s"]),
                    expected_below.unpivot(["i", "j"], ["s"]),
                )
                assert_lazy(
                    lazy.filter(
                        (col("source") == "i") & (col("value") > 0)
                    ).select(["value", "s"]),
                    eager.filter(
                        (col("source") == "i") & (col("value") > 0)
                    ).select(["value", "s"]),
                )
                assert_lazy(lazy.head(3), eager.head(3))
                assert_lazy(lazy.slice(n, 2), eager.slice(n, 2))
                assert_lazy(
                    lazy.group_by("source")
                    .agg(
                        [
                            col("value").sum(),
                            col("value").null_count().alias("nulls"),
                        ]
                    )
                    .sort("source"),
                    eager.group_by("source")
                    .agg(
                        [
                            col("value").sum(),
                            col("value").null_count().alias("nulls"),
                        ]
                    )
                    .sort("source"),
                )


def test_lazy_unpivot_literal_values_and_errors() raises:
    var frame = DataFrame(
        [
            Series("id", StringColumn(["x", "y"])),
            Series("a", Column[Int64]([1, 2], [True, False])),
            Series("c", Column[Int64]([3, 4])),
            Series("t", StringColumn(["p", "q"])),
        ]
    )
    var plan = frame.lazy().unpivot(["c", "a"], ["id"])
    var expected = DataFrame(
        [
            Series("id", StringColumn(["x", "y", "x", "y"])),
            Series("variable", StringColumn(["c", "c", "a", "a"])),
            Series(
                "value",
                Column[Int64]([3, 4, 1, 0], [True, True, True, False]),
            ),
        ]
    )
    assert_lazy(plan, expected)
    assert_true("UNPIVOT on c, a index id" in plan.explain())
    assert_true("[materialize]" in plan.explain(streaming=True))
    # A filter on its output stays above it.
    var filtered = plan.filter(col("variable") == "a").explain()
    assert_true(
        filtered.find("FILTER") < filtered.find("UNPIVOT"), "filter moved"
    )
    # Eager's errors, raised when the plan runs.
    var invalid: List[LazyFrame] = [
        frame.lazy().unpivot(["a", "t"]),
        frame.lazy().unpivot(["a"], ["a"]),
        frame.lazy().unpivot(["missing"]),
        frame.lazy().unpivot(["a"], ["id"], variable_name="id"),
        frame.lazy().unpivot(["a"], variable_name="v", value_name="v"),
        frame.lazy().unpivot(index=frame.columns()),
    ]
    for bad in invalid:
        for optimize in [False, True]:
            for streaming in [False, True]:
                with assert_raises():
                    _ = bad.collect(
                        optimize=optimize, streaming=streaming, engine="cpu"
                    )
        with assert_raises():
            _ = bad.collect_schema()


def test_lazy_fill_null_chooses_columns_as_eager() raises:
    var frame = nullable_frame()
    var values: List[Expr] = [
        Expr(0),
        Expr(7),
        Expr(Float64(2.5)),
        lit(Int64(9)),
        lit(Float64(9.25)),
        lit(True),
        lit("z"),
    ]
    for value in values:
        assert_lazy(frame.lazy().fill_null(value), frame.fill_null(value))
    assert_lazy(frame.lazy().fill_null("z"), frame.fill_null("z"))
    var subsets: List[List[String]] = [["i"], ["f"], ["i", "f"]]
    for subset in subsets:
        assert_lazy(
            frame.lazy().fill_null(Expr(4), subset),
            frame.fill_null(Expr(4), subset),
        )
    # Independent literals: an untyped integer fills the numeric columns
    # and keeps their types; strings and booleans keep their nulls.
    var expected = DataFrame(
        [
            Series("i", Column[Int64]([1, 0, 3, 0, 5])),
            Series("f", Column[Float64]([0.5, 1.5, 0.0, 3.5, 0.0])),
            frame.column("s"),
            frame.column("b"),
        ]
    )
    assert_lazy(frame.lazy().fill_null(0), expected)
    var strings = DataFrame(
        [
            frame.column("i"),
            frame.column("f"),
            Series("s", StringColumn(["a", "-", "c", "d", "-"])),
            frame.column("b"),
        ]
    )
    assert_lazy(frame.lazy().fill_null("-"), strings)
    # Mismatched subsets and unknown names raise as eager does.
    with assert_raises():
        _ = frame.fill_null("-", ["i"])
    with assert_raises():
        _ = frame.lazy().fill_null("-", ["i"])
    with assert_raises():
        _ = frame.lazy().fill_null(0, ["missing"])
    # Composed: a filter on a filled column stays above the fill, and the
    # fill sees the rows a filter below it keeps.
    var plan = (
        frame.lazy()
        .filter(col("b").is_not_null())
        .fill_null(0)
        .filter(col("i") == 0)
    )
    assert_lazy(
        plan,
        frame.filter(col("b").is_not_null()).fill_null(0).filter(col("i") == 0),
    )
    # Over generated tables in every layout, and over empty tables.
    for shape in range(3):
        for mode in [case_nulls(shape)]:
            var n = case_rows(shape)
            for layout in model_frames(n, mode):
                assert_lazy(
                    layout.lazy().fill_null(1).fill_null("x"),
                    layout.fill_null(1).fill_null("x"),
                )
                assert_lazy(
                    layout.lazy().fill_null(lit(False)),
                    layout.fill_null(lit(False)),
                )


def test_lazy_drop_nulls_matches_scalar_rows() raises:
    var subsets: List[List[String]] = [[], ["i"], ["s", "b"]]
    for shape in range(3):
        for mode in [case_nulls(shape)]:
            var n = case_rows(shape)
            var model = generated_model(n, 3, mode)
            for frame in model_frames(n, mode):
                for subset in subsets:
                    var plan = frame.lazy().drop_nulls(subset)
                    for actual in collected(plan):
                        assert_model(
                            actual, model.take(nonnull_indices(model, subset))
                        )
                    assert_lazy(plan, frame.drop_nulls(subset))
                    # Composed with steps on either side.
                    assert_lazy(
                        frame.lazy()
                        .with_columns([(col("i") + 1).alias("k")])
                        .drop_nulls(subset)
                        .select(["k", "s"])
                        .head(5),
                        frame.with_columns([(col("i") + 1).alias("k")])
                        .drop_nulls(subset)
                        .select(["k", "s"])
                        .head(5),
                    )
    # One filter, which the optimizer may split and push like any other.
    var frame = nullable_frame()
    assert_true("FILTER" in frame.lazy().drop_nulls().explain(optimize=False))
    var expected = DataFrame(
        [
            Series("i", Column[Int64]([1])),
            Series("f", Column[Float64]([0.5])),
            Series("s", StringColumn(["a"])),
            Series("b", Column[Bool]([True])),
        ]
    )
    assert_lazy(frame.lazy().drop_nulls(), expected)
    with assert_raises():
        _ = frame.lazy().drop_nulls(["missing"]).collect(engine="cpu")


def test_lazy_null_count_matches_eager() raises:
    var frame = nullable_frame()
    var expected = DataFrame(
        [
            Series("i", Column[Int64]([2])),
            Series("f", Column[Int64]([2])),
            Series("s", Column[Int64]([2])),
            Series("b", Column[Int64]([1])),
        ]
    )
    assert_frame(frame.null_count(), expected)
    assert_lazy(frame.lazy().null_count(), expected)
    assert_lazy(frame.clear().lazy().null_count(), frame.clear().null_count())
    for shape in range(3):
        for mode in [case_nulls(shape)]:
            var n = case_rows(shape)
            for layout in model_frames(n, mode):
                assert_lazy(layout.lazy().null_count(), layout.null_count())
                assert_lazy(
                    layout.lazy().filter(col("b").is_not_null()).null_count(),
                    layout.filter(col("b").is_not_null()).null_count(),
                )
                assert_lazy(
                    layout.lazy().fill_null(0).null_count(),
                    layout.fill_null(0).null_count(),
                )


def assert_distinct(frame: DataFrame, subset: List[String]) raises:
    var count = DataFrame(
        [Series("n_unique", Column[Int64]([Int64(frame.n_unique(subset))]))]
    )
    assert_lazy(frame.lazy().n_unique(subset), count)
    assert_lazy(
        frame.lazy().is_duplicated(subset),
        DataFrame([frame.is_duplicated(subset)]),
    )
    assert_lazy(
        frame.lazy().is_unique(subset), DataFrame([frame.is_unique(subset)])
    )


def test_lazy_distinct_counts_and_flags_match_eager_and_scalar_groups() raises:
    var subsets: List[List[String]] = [[], ["s"], ["f"], ["s", "b"]]
    for shape in range(3):
        for mode in [case_nulls(shape)]:
            var n = case_rows(shape)
            var model = generated_model(n, 3, mode)
            if n > 1:
                model.rows.append(model.rows[0].copy())
                model.rows.append(model.rows[0].copy())
            var frame = frame_from_model(model, 7)
            for layout in [frame.copy(), chunk_layout(frame)]:
                for subset in subsets:
                    # Scalar groups give the expected count and flags.
                    var keys = (
                        model.names.copy() if not subset else subset.copy()
                    )
                    var groups = reference_groups(model, keys)
                    var repeated = List[Bool](
                        length=len(model.rows), fill=False
                    )
                    var single = List[Bool](length=len(model.rows), fill=False)
                    for group in groups:
                        for row in group:
                            repeated[row] = len(group) > 1
                            single[row] = len(group) == 1
                    assert_equal(layout.n_unique(subset), len(groups))
                    assert_lazy(
                        layout.lazy().n_unique(subset),
                        DataFrame(
                            [
                                Series(
                                    "n_unique",
                                    Column[Int64]([Int64(len(groups))]),
                                )
                            ]
                        ),
                    )
                    var duplicated = DataFrame(
                        [Series("is_duplicated", Column[Bool](repeated^))]
                    )
                    var unique = DataFrame(
                        [Series("is_unique", Column[Bool](single^))]
                    )
                    assert_frame(
                        DataFrame([layout.is_duplicated(subset)]), duplicated
                    )
                    assert_frame(DataFrame([layout.is_unique(subset)]), unique)
                    assert_lazy(layout.lazy().is_duplicated(subset), duplicated)
                    assert_lazy(layout.lazy().is_unique(subset), unique)
    # Float keys: -0.0 equals 0.0 and NaN equals NaN, as eager.
    var nan = Float64(0) / Float64(0)
    var values: List[Float64] = [-0.0, 0.0, nan, nan, 8, 9, 9]
    var valid: List[Bool] = [True, True, True, True, True, False, False]
    var floats = DataFrame([Series("f", Column[Float64](values^, valid))])
    assert_distinct(floats, [])
    var expected = DataFrame(
        [
            Series(
                "is_duplicated",
                Column[Bool]([True, True, True, True, False, True, True]),
            )
        ]
    )
    assert_lazy(floats.lazy().is_duplicated(), expected)
    assert_lazy(
        floats.lazy().n_unique(),
        DataFrame([Series("n_unique", Column[Int64]([4]))]),
    )
    # Composed: counted after a filter, and the flags filtered after.
    var frame = nullable_frame()
    assert_lazy(
        frame.lazy().filter(col("i").is_not_null()).n_unique(["s"]),
        DataFrame(
            [
                Series(
                    "n_unique",
                    Column[Int64](
                        [
                            Int64(
                                frame.filter(col("i").is_not_null()).n_unique(
                                    ["s"]
                                )
                            )
                        ]
                    ),
                )
            ]
        ),
    )
    assert_lazy(
        frame.lazy().is_unique(["b"]).filter(col("is_unique")),
        DataFrame([frame.is_unique(["b"])]).filter(col("is_unique")),
    )
    with assert_raises():
        _ = frame.lazy().n_unique(["missing"]).collect(engine="cpu")
    with assert_raises():
        _ = frame.lazy().is_duplicated(["missing"]).collect(engine="cpu")
