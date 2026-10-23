"""Lazy rename, cast, group_by on expressions and the group_by reductions
give the eager DataFrame methods' results under every executor setting.
rename is a select of every column and cast a with_columns, so filters and
column selections above them move below them:
explain shows where they land."""
# contract-group: lazy.rename_cast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    lit,
)
from dataframe_metal.accelerator.metadata import prune_projections, schema_frame
from tests.support.coverage_compare import assert_frame


def source() raises -> DataFrame:
    """Nine rows: an Int64 key with nulls, a String key, a Float64 payload
    and a String column of numbers."""
    return DataFrame(
        [
            Series(
                "k",
                Column[Int64](
                    [1, 2, 1, 3, 0, 2, 1, 3, 2],
                    [True, True, True, True, False, True, True, True, True],
                ),
            ),
            Series(
                "s",
                Column[String](["a", "b", "a", "c", "b", "b", "a", "c", "a"]),
            ),
            Series(
                "f",
                Column[Float64](
                    [0.5, 1.5, 2.5, 3.0, 4.5, 5.5, 6.5, 7.0, 8.5],
                    [True, True, True, False, True, True, True, True, True],
                ),
            ),
            Series(
                "t",
                Column[String](["1", "2", "x", "4", "5", "6", "7", "8", "9"]),
            ),
        ]
    )


def check(plan: LazyFrame, expected: DataFrame) raises:
    """Every executor setting gives `expected`, and the schema matches."""
    for optimize in [True, False]:
        for streaming in [True, False]:
            for batch_size in [65536, 2]:
                assert_frame(
                    plan.collect(
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=batch_size,
                    ),
                    expected,
                )
    var schema = List[String]()
    for field in expected.schema():
        schema.append(field.name + ": " + field.dtype.name())
    assert_equal(plan.collect_schema(), schema)
    var metadata = schema_frame(plan, len(plan._nodes) - 1)
    assert_equal(metadata.columns(), expected.columns())
    var pruned = plan.copy()
    prune_projections(pruned)
    assert_frame(pruned.collect(optimize=False, streaming=False), expected)


def check_fails(plan: LazyFrame) raises:
    for optimize in [True, False]:
        for streaming in [True, False]:
            with assert_raises():
                _ = plan.collect(optimize=optimize, streaming=streaming)


def mapping(pairs: List[String]) -> Dict[String, String]:
    """A Dict from alternating keys and values."""
    var out = Dict[String, String]()
    for k in range(0, len(pairs), 2):
        out[pairs[k]] = pairs[k + 1]
    return out^


def line_of(text: String, operator: String) -> Int:
    """The line of `text` (from 0) that starts with `operator` once
    indented, or -1."""
    var lines = text.split("\n")
    for k in range(len(lines)):
        if String(lines[k].strip()).startswith(operator):
            return k
    return -1


def test_rename_matches_eager() raises:
    var frame = source()
    var cases: List[List[String]] = [
        ["k", "key"],
        ["k", "s", "s", "k"],
        ["f", "g", "t", "u", "s", "v"],
        ["s", "s"],
    ]
    for pairs in cases:
        var names = mapping(pairs)
        check(frame.lazy().rename(names), frame.rename(names))
        check(
            frame.lazy().filter(col("k") > 1).rename(names).head(3),
            frame.filter(col("k") > 1).rename(names).head(3),
        )
    # Over a computed input, and with a name the input lacks when not
    # strict.
    var loose = mapping(["missing", "x", "f", "g"])
    check(
        frame.lazy()
        .with_columns((col("f") * 2.0).alias("f2"))
        .rename(loose, strict=False),
        frame.with_columns((col("f") * 2.0).alias("f2")).rename(
            loose, strict=False
        ),
    )


def test_rename_errors() raises:
    var frame = source()
    with assert_raises(contains="Unknown column: missing"):
        _ = frame.lazy().rename(mapping(["missing", "x"]))
    with assert_raises(contains="Unknown column: missing"):
        _ = frame.rename(mapping(["missing", "x"]))
    with assert_raises(contains="duplicate column name: s"):
        _ = frame.lazy().rename(mapping(["k", "s"]))
    with assert_raises(contains="duplicate column name: s"):
        _ = frame.rename(mapping(["k", "s"]))


def test_rename_lets_filters_and_projections_through() raises:
    var frame = source()
    var plan = (
        frame.lazy()
        .rename(mapping(["k", "key", "s", "label"]))
        .filter(col("key") > 1)
        .select(["key"])
    )
    check(
        plan,
        frame.rename(mapping(["k", "key", "s", "label"]))
        .filter(col("key") > 1)
        .select(["key"]),
    )
    var text = plan.explain(streaming=False)
    # The filter runs on the input's `k`, below the rename's select, which
    # keeps only `key`; the scan reads only `k`.
    assert_equal(
        text,
        "SELECT key\n  SELECT key\n    FILTER\n      SCAN frame [project k]\n",
    )
    # Unoptimized, the filter stays where it was written.
    var plain = plan.explain(optimize=False, streaming=False)
    assert_true(
        line_of(plain, "FILTER") < line_of(plain, "SELECT key, label"), plain
    )


def test_cast_matches_eager() raises:
    var frame = source()
    var casts = mapping(["k", "float64", "f", "int64"])
    check(frame.lazy().cast(casts), frame.cast(casts))
    var text = mapping(["t", "int64"])
    check(frame.lazy().cast(text, strict=False), frame.cast(text, strict=False))
    # Strict casting raises on the "x" in t, eagerly and when the plan runs.
    with assert_raises():
        _ = frame.cast(text)
    check_fails(frame.lazy().cast(text))
    # An unknown name raises when the plan runs.
    check_fails(frame.lazy().cast(mapping(["missing", "int64"])))
    check(frame.lazy().cast(Dict[String, String]()), frame)


def test_cast_lets_filters_and_projections_through() raises:
    var frame = source()
    var casts = mapping(["t", "int64", "f", "int64"])
    # A filter on a column the cast leaves alone moves below it.
    var other = frame.lazy().cast(casts, strict=False).filter(col("k") > 1)
    check(other, frame.cast(casts, strict=False).filter(col("k") > 1))
    var text = other.explain(streaming=False)
    assert_true(
        line_of(text, "FILTER") > line_of(text, "WITH_COLUMNS t, f"), text
    )
    # A filter on a cast column stays above it.
    var cast_column = (
        frame.lazy().cast(casts, strict=False).filter(col("t") > 4)
    )
    check(cast_column, frame.cast(casts, strict=False).filter(col("t") > 4))
    text = cast_column.explain(streaming=False)
    assert_true(
        line_of(text, "FILTER") < line_of(text, "WITH_COLUMNS t, f"), text
    )
    # A cast column nothing above reads is not computed or scanned.
    var selected = frame.lazy().cast(casts, strict=False).select(["k", "t"])
    check(selected, frame.cast(casts, strict=False).select(["k", "t"]))
    text = selected.explain(streaming=False)
    assert_true(text.find("WITH_COLUMNS t\n") >= 0, text)
    assert_true(text.find("SCAN frame [project k, t]") >= 0, text)


def test_group_by_expressions_match_eager() raises:
    var frame = source()
    var key_sets: List[List[Expr]] = [
        [(col("k") * 2).alias("double")],
        [col("s"), (col("k") % 2).alias("odd")],
        # A computed key under an input column's name.
        [col("k") + 10],
        [col("s").alias("label")],
        [col("k"), col("s")],
    ]
    var aggregations: List[List[Expr]] = [
        [col("f").sum().alias("total")],
        # Reads `k`, which the third key set computes under that name: the
        # aggregation must still see the input's `k`.
        [col("k").max().alias("top"), col("f").mean().alias("avg")],
    ]
    for keys in key_sets:
        for aggs in aggregations:
            check(
                frame.lazy().group_by(keys, maintain_order=True).agg(aggs),
                frame.group_by(keys, maintain_order=True).agg(aggs),
            )
    # Filters on a computed key move below the group_by, not below the
    # key's computation.
    var filtered = (
        frame.lazy()
        .group_by([(col("k") * 2).alias("double")], maintain_order=True)
        .agg(col("f").sum().alias("total"))
        .filter(col("double") > 2)
    )
    check(
        filtered,
        frame.group_by([(col("k") * 2).alias("double")], maintain_order=True)
        .agg(col("f").sum().alias("total"))
        .filter(col("double") > 2),
    )
    var text = filtered.explain(streaming=False)
    var group = line_of(text, "GROUP_BY double")
    var filter = line_of(text, "FILTER")
    var keys = line_of(text, "WITH_COLUMNS double")
    assert_true(group >= 0 and group < filter and filter < keys, text)


def test_group_by_len_and_reductions_match_eager() raises:
    var frame = source()
    var name_keys: List[String] = ["s"]
    check(
        frame.lazy().group_by(name_keys, maintain_order=True).len(),
        frame.group_by(name_keys, maintain_order=True).len(),
    )
    check(
        frame.lazy().group_by("k", maintain_order=True).len("rows"),
        frame.group_by("k", maintain_order=True).len("rows"),
    )
    var keys: List[Expr] = [(col("k") * 2).alias("double")]
    check(
        frame.lazy().group_by(keys, maintain_order=True).len(),
        frame.group_by(keys, maintain_order=True).len(),
    )
    # Reductions of every other column; the numeric ones on a frame
    # without the string columns.
    var numbers = frame.select(["k", "f"]).with_columns(
        (col("k") % 2).alias("odd")
    )
    for keyed in [False, True]:
        var lazy = (
            numbers.lazy()
            .group_by(
                "odd", maintain_order=True
            ) if not keyed else numbers.lazy()
            .group_by([col("k") % 2], maintain_order=True)
        )
        var eager = numbers.group_by(
            "odd", maintain_order=True
        ) if not keyed else numbers.group_by(
            [col("k") % 2], maintain_order=True
        )
        check(lazy.sum(), eager.sum())
        check(lazy.mean(), eager.mean())
        check(lazy.median(), eager.median())
        check(lazy.quantile(0.25), eager.quantile(0.25))
        check(lazy.quantile(0.5, "linear"), eager.quantile(0.5, "linear"))
        check(lazy.min(), eager.min())
        check(lazy.max(), eager.max())
        check(lazy.first(), eager.first())
        check(lazy.last(), eager.last())
        check(lazy.n_unique(), eager.n_unique())
    # Order-free reductions also hold strings.
    var by_s = frame.lazy().group_by("s", maintain_order=True)
    var eager_s = frame.group_by("s", maintain_order=True)
    check(by_s.min(), eager_s.min())
    check(by_s.max(), eager_s.max())
    check(by_s.first(), eager_s.first())
    check(by_s.n_unique(), eager_s.n_unique())
    # Literal expectation: the key `k % 2` keeps its name, so `k` is not
    # reduced again (as in Polars).
    var reduced = (
        numbers.select(["k", "f"])
        .lazy()
        .group_by([col("k") % 2], maintain_order=True)
        .sum()
        .collect()
    )
    assert_equal(reduced.columns(), ["k", "f"])
    assert_equal(reduced.height(), 3)
