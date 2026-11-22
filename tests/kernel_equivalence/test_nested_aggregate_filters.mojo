"""Aggregate filters over list and struct values run in the aggregate
table: `xs.filter(p).first()`, `rec.drop_nulls().len()` and
predicates that read nested values (`xs.list().len() > 1`) keep only their
rows for that reduction, nested validity included. The lazy pipeline and
the eager group-by both take the table; each answer equals grouping the
frame the filter keeps."""
# contract-group: reduce.grouped
# contract-group: expr.nested
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    Expr,
    ListColumn,
    Series,
    StructColumn,
    col,
    lit,
)
from dataframe.storage.column import _pack_bits
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 20_000


def frame(rows: Int = ROWS) raises -> DataFrame:
    """String key k, a list column xs (null lists, null elements, empty
    lists), a struct column rec (null structs, null fields) and a
    predicate p with nulls."""
    var k = List[String]()
    var offsets = List[Int64]()
    offsets.append(0)
    var items = List[Int64]()
    var item_valid = List[Bool]()
    var list_valid = List[Bool]()
    var a = List[Int64]()
    var a_valid = List[Bool]()
    var s = List[String]()
    var rec_valid = List[Bool]()
    var p = List[Bool]()
    var p_valid = List[Bool]()
    for row in range(rows):
        k.append("k" + String(row % 7))
        var length = row % 4
        for i in range(length):
            items.append(Int64((row // 3 + i) % 5))
            item_valid.append((row + i) % 9 != 4)
        offsets.append(Int64(len(items)))
        list_valid.append(row % 11 != 6)
        a.append(Int64(row % 6))
        a_valid.append(row % 13 != 2)
        s.append("s" + String(row % 3))
        rec_valid.append(row % 10 != 7)
        p.append(row % 5 < 3)
        p_valid.append(row % 19 != 8)
    var xs = Series(
        "xs",
        ListColumn(
            offsets^,
            Series("item", Column[Int64](items^, item_valid)),
            _pack_bits(list_valid),
        ),
    )
    var rec = Series(
        "rec",
        StructColumn(
            [
                Series("a", Column[Int64](a^, a_valid)),
                Series("s", Column[String](s^)),
            ],
            _pack_bits(rec_valid),
        ),
    )
    return DataFrame(
        [
            Series("k", Column[String](k^)),
            xs^,
            rec^,
            Series("p", Column[Bool](p^, p_valid)),
        ]
    )


def reductions(value: Expr, prefix: String) -> List[Expr]:
    return [
        value.first().alias(prefix + "f"),
        value.last().alias(prefix + "l"),
        value.len().alias(prefix + "n"),
        value.count().alias(prefix + "c"),
        value.null_count().alias(prefix + "nc"),
    ]


def check_case(
    source: DataFrame,
    filtered: List[Expr],
    kept: Expr,
    plain: List[Expr],
    context: String,
) raises:
    """The filtered reductions, lazy and eager, both through the table,
    against `plain` over the rows `kept` selects."""
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var piped = (
        source.lazy().group_by(["k"]).agg(filtered.copy()).collect().sort(["k"])
    )
    var text = observed.text()
    _ = observed^
    assert_true(
        "dataframe-path: pipeline.group_table\n" in text,
        context + ": the lazy group-by left the table",
    )
    var eager_observed = StrategyTrace()
    var eager = source.group_by(["k"]).agg(filtered).sort(["k"])
    var eager_text = eager_observed.text()
    _ = eager_observed^
    _ = trace^
    assert_true(
        "dataframe-path: group_by.aggregate_table\n" in eager_text,
        context + ": the eager group-by left the table",
    )
    var expected = source.filter(kept).group_by(["k"]).agg(plain).sort(["k"])
    assert_equal(piped.height(), expected.height(), context)
    assert_true(piped.equals(expected), context + ": lazy answer")
    assert_true(eager.equals(expected), context + ": eager answer")


def test_filtered_nested_values() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var source = frame()
    for name in ["xs", "rec"]:
        check_case(
            source,
            reductions(col(name).filter(col("p")), "v"),
            col("p"),
            reductions(col(name), "v"),
            name + " filtered by p",
        )
        check_case(
            source,
            reductions(col(name).drop_nulls(), "v"),
            col(name).is_not_null(),
            reductions(col(name), "v"),
            name + " without nulls",
        )
    _ = workers^


def test_predicates_over_nested_values() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var source = frame()
    var long = col("xs").list().len() > lit(Int64(1))
    var field = col("rec").field("a") >= lit(Int64(3))
    check_case(
        source,
        reductions(col("rec").filter(long.copy()), "r"),
        long.copy(),
        reductions(col("rec"), "r"),
        "rec filtered by list length",
    )
    check_case(
        source,
        reductions(col("xs").filter(field.copy()), "x"),
        field.copy(),
        reductions(col("xs"), "x"),
        "xs filtered by a struct field",
    )
    _ = workers^


def test_one_worker_and_mixed_filters() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "1")
    var source = frame(3_000)
    var exprs = reductions(col("xs").drop_nulls(), "x")
    exprs.extend(reductions(col("rec").filter(col("p")), "r"))
    var piped = (
        source.lazy().group_by(["k"]).agg(exprs.copy()).collect().sort(["k"])
    )
    var eager = source.group_by(["k"]).agg(exprs).sort(["k"])
    var lists = (
        source.filter(col("xs").is_not_null())
        .group_by(["k"])
        .agg(reductions(col("xs"), "x"))
        .sort(["k"])
    )
    var records = (
        source.filter(col("p"))
        .group_by(["k"])
        .agg(reductions(col("rec"), "r"))
        .sort(["k"])
    )
    var names: List[String] = ["k", "xf", "xl", "xn", "xc", "xnc"]
    var rnames: List[String] = ["k", "rf", "rl", "rn", "rc", "rnc"]
    assert_true(piped.select(names).equals(lists))
    assert_true(piped.select(rnames).equals(records))
    assert_true(eager.equals(piped))
    _ = workers^
