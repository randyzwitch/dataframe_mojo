"""List and struct join keys in the join hash table, checked against the
same join on an integer column that numbers each key's class of equal
values (null for a null key): lists equal elementwise with nulls equal,
an empty list apart from a null one, a null struct apart from a struct of
nulls, -0.0 equal to 0.0 and every NaN one value, and two categorical codes
for the same label one value. Every join type, with and without coalesced
keys, alone and next to a flat key, eager and lazy (the probe side a
pipeline over a filter; the build side a frame or a pipeline of its
own), with the probe side larger and smaller than the build."""
# contract-group: table.join
# contract-group: expr.nested
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    Series,
    col,
    lit,
)
from dataframe.kernels.nested_keys import encode_nested_keys
from dataframe.storage.string import StringColumn
from tests.kernel_equivalence.test_nested_group_keys import (
    Case,
    cases,
    other_of,
    pattern_of,
)
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def side(kase: Case, rows: Int, right: Bool) raises -> DataFrame:
    """`rows` rows of the case's patterns as `key`, their class as `c`
    (null for a null key), the row as `id` (`rid` on the right) and a flat
    key `f`."""
    var count = len(kase.classes)
    var picks = List[Int](capacity=rows)
    var classes = List[Int64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var ids = List[Int64](capacity=rows)
    var flat = List[Int64](capacity=rows)
    for row in range(rows):
        var p = other_of(row, count) if right else pattern_of(row, count)
        picks.append(p)
        classes.append(Int64(kase.classes[p]))
        valid.append(not kase.patterns.get(p).is_null())
        ids.append(Int64(row))
        flat.append(Int64(row % 2))
    return DataFrame(
        [
            kase.patterns.take(picks).renamed("key"),
            Series("c", Column[Int64](classes^, valid)),
            Series("rid" if right else "id", Column[Int64](ids^)),
            Series("f", Column[Int64](flat^)),
        ]
    )


def joined(
    left: DataFrame,
    right: DataFrame,
    on: List[String],
    how: String,
    coalesce: Bool,
    mode: Int,
) raises -> DataFrame:
    """Mode 0: the eager join. Mode 1: lazy, the left side a pipeline over
    a filter, the right side a frame. Mode 2: both sides pipelines."""
    if mode == 0:
        return left.join(right, on=on, how=how, coalesce=coalesce)
    var probe = left.lazy().filter(col("id") >= lit(Int64(0)))
    var build = right.lazy()
    if mode == 2:
        build = build.filter(col("rid") >= lit(Int64(0)))
    return probe.join(build, on=on, how=how, coalesce=coalesce).collect()


def pairs(frame: DataFrame, semi: Bool) raises -> DataFrame:
    """The result's left and right rows, -1 for none, in order."""
    var rows = frame.select(["id"]) if semi else frame.select(["id", "rid"])
    var names: List[String] = ["id"] if semi else ["id", "rid"]
    return rows.fill_null(lit(Int64(-1))).sort(names)


def check(
    kase: Case,
    left: DataFrame,
    right: DataFrame,
    how: String,
    coalesce: Bool,
    multi: Bool,
    mode: Int,
) raises:
    var context = (
        kase.name + " " + how + " coalesce=" + String(coalesce) + " mode="
    ) + String(mode)
    var on: List[String] = ["key"]
    var by: List[String] = ["c"]
    if multi:
        on = ["key", "f"]
        by = ["c", "f"]
    var result = joined(left, right, on, how, coalesce, mode)
    var expected = left.drop("key").join(
        right.drop("key"), on=by, how=how, coalesce=coalesce
    )
    var semi = how == "semi" or how == "anti"
    var got = pairs(result, semi)
    var want = pairs(expected, semi)
    assert_equal(got.height(), want.height(), context + ": row count")
    assert_true(got.equals(want), context + ": matched rows differ")
    # Each row's keys: its side's own key value, and a coalesced key the
    # left's where there is a left row, else the right's.
    var split = "key_right" in result.columns()
    var ordered = result.with_columns(
        [
            col("id").fill_null(lit(Int64(-1))).alias("id"),
            col("rid").fill_null(lit(Int64(-1))).alias("rid"),
        ]
    ).sort(["id", "rid"]) if not semi else result.sort("id")
    var n = ordered.height()
    var lrows = values(ordered.column("id"))
    var rrows = List[Int64](length=n, fill=-1)
    if not semi:
        rrows = values(ordered.column("rid"))
    # The expected key of each row as a row of `pool`: the left keys, then
    # the right keys; -1 for null.
    var pool = encode_nested_keys(
        left.column("key").append(right.column("key"))
    )
    var lefts = List[Int](capacity=n)
    var rights = List[Int](capacity=n)
    var coalesced = List[Int](capacity=n)
    for i in range(n):
        var l = Int(lrows[i])
        var r = Int(rrows[i])
        var lk = l if l >= 0 else -1
        var rk = left.height() + r if r >= 0 else -1
        lefts.append(lk)
        rights.append(rk)
        if how == "right":
            coalesced.append(rk)
        else:
            coalesced.append(lk if lk >= 0 else rk)
    if split:
        same_keys(ordered.column("key"), pool, lefts, context + ": left key")
        same_keys(
            ordered.column("key_right"), pool, rights, context + ": right key"
        )
    else:
        same_keys(ordered.column("key"), pool, coalesced, context + ": key")


def values(column: Series) raises -> List[Int64]:
    var whole = column.rechunk() if column.is_chunked() else column.copy()
    ref stored = whole._data[Column[Int64]]
    var out = List[Int64](capacity=len(stored))
    for i in range(len(stored)):
        out.append(stored._get(i))
    return out^


def same_keys(
    keys: Series, pool: Series, rows: List[Int], context: String
) raises:
    """Whether each key is the value of row `rows[i]` of the encoded
    `pool` (null for -1), compared as key bytes: equal values, whatever
    their storage."""
    var encoded = encode_nested_keys(keys)
    ref got = encoded._data[StringColumn]
    ref want = pool._data[StringColumn]
    assert_equal(len(got), len(rows), context)
    for i in range(len(rows)):
        var j = rows[i]
        if j < 0:
            assert_true(not got._valid(i), context + ": expected null")
            continue
        assert_equal(got._valid(i), want._valid(j), context + ": validity")
        if want._valid(j):
            assert_true(got._row_bytes(i) == want._row_bytes(j), context)


def run(mode: Int, threads: Int) raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var hows: List[String] = ["inner", "left", "right", "full", "semi", "anti"]
    var sizes: List[Tuple[Int, Int]] = [(120, 40), (40, 120)]
    for kase in cases():
        for size in sizes:
            var left = side(kase, size[0], False)
            var right = side(kase, size[1], True)
            for how in hows:
                for coalesce in [True, False]:
                    if not coalesce and (how == "semi" or how == "anti"):
                        continue
                    # The flat key joins beside the nested one in one of the
                    # two shapes.
                    for multi in [size[0] < size[1]]:
                        var observed = StrategyTrace()
                        check(kase, left, right, how, coalesce, multi, mode)
                        var text = observed.text()
                        _ = observed^
                        if mode > 0:
                            assert_true(
                                "dataframe-path: pipeline.join_dictionary\n"
                                in text,
                                kase.name
                                + " "
                                + how
                                + ": the join did not probe the hash table"
                                " in the pipeline",
                            )
                        if mode == 2:
                            assert_true(
                                "dataframe-path: pipeline.join_build_sink\n"
                                in text,
                                kase.name
                                + " "
                                + how
                                + ": the build did not sink into the table",
                            )
    _ = trace^
    _ = workers^


def test_eager_nested_join_keys() raises:
    run(0, 4)


def test_pipeline_nested_join_keys_over_a_frame_build() raises:
    run(1, 4)


def test_pipeline_nested_join_keys_over_a_pipeline_build() raises:
    run(2, 4)


def test_empty_sides_join_on_nested_keys() raises:
    for kase in cases():
        var left = side(kase, 40, False)
        var right = side(kase, 40, True)
        for how in ["inner", "left", "right", "full", "semi", "anti"]:
            for mode in range(3):
                var none_right = joined(
                    left, right.clear(), ["key"], how, True, mode
                )
                var none_left = joined(
                    left.clear(), right, ["key"], how, True, mode
                )
                var keeps_left = (
                    how == "left" or how == "full" or (how == "anti")
                )
                var keeps_right = how == "right" or how == "full"
                assert_equal(
                    none_right.height(), 40 if keeps_left else 0, kase.name
                )
                assert_equal(
                    none_left.height(), 40 if keeps_right else 0, kase.name
                )
                assert_true(
                    none_right.column("key").dtype() == kase.patterns.dtype()
                )
