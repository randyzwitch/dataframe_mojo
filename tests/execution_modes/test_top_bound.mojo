"""Streamed Top-N bounds preserve later ties and never bound replaced keys."""
# contract-group: execution.modes
from std.math import isnan
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.pipeline import _TopBound, _BOUND_INT, _BOUND_FLOAT, _BOUND_NONE
from tests.support.lifecycle_jobs import WorkerSetting


def bound_key(row: Int, descending: Bool) -> Int64:
    var value = Int64(0 if row % 16384 == 0 else row // 16)
    return -value if descending else value


def bound_valid(row: Int) -> Bool:
    return row % 20011 != 17


def test_top_bound_literal_thresholds_keep_null_nan_equal_keys_and_decline_unsupported_keys() raises:
    for floating in [False, True]:
        for descending in [False, True]:
            var candidate_keys: List[Int64] = [2, 3, 4]
            if descending:
                candidate_keys = [4, 3, 2]
            var candidates = DataFrame(
                [
                    Series(
                        "key",
                        Column[Int64](candidate_keys^),
                    )
                ]
            )
            if floating:
                var candidate_floats: List[Float64] = [2, 3, 4]
                if descending:
                    candidate_floats = [4, 3, 2]
                candidates = DataFrame(
                    [
                        Series(
                            "key",
                            Column[Float64](candidate_floats^),
                        )
                    ]
                )
            var bound = _TopBound(candidates, "key", descending, 2)
            assert_equal(bound.kind, _BOUND_FLOAT if floating else _BOUND_INT)
            var values: List[Int64] = [1, 2, 3, 3, 4, 5, Int64.MAX]
            var valid: List[Bool] = [True, True, True, True, True, True, False]
            var frame = DataFrame(
                [Series("key", Column[Int64](values^, valid.copy()))]
            )
            var expected: List[Int] = [0, 1, 2, 3, 6]
            if descending:
                expected = [2, 3, 4, 5, 6]
            if floating:
                var raw: List[Float64] = [
                    1,
                    2,
                    3,
                    3,
                    4,
                    Float64(0) / Float64(0),
                    Float64.MAX,
                ]
                frame = DataFrame([Series("key", Column[Float64](raw^, valid))])
                expected = [0, 1, 2, 3, 5, 6] if not descending else [
                    2,
                    3,
                    4,
                    5,
                    6,
                ]
            var rows = bound.rows(frame)
            assert_true(Bool(rows))
            assert_equal(rows.take(), expected)
            assert_equal(
                _TopBound(candidates, "key", descending, 0).kind, _BOUND_NONE
            )
            assert_equal(
                _TopBound(candidates, "key", descending, 4).kind, _BOUND_NONE
            )
    for series in [
        Series("key", Column[Int64]([1], [False])),
        Series("key", Column[Float64]([Float64(0) / Float64(0)])),
        Series("key", Column[UInt64]([UInt64.MAX])),
        Series("key", Column[String](["a"])),
    ]:
        assert_equal(
            _TopBound(DataFrame([series.copy()]), "key", False, 1).kind,
            _BOUND_NONE,
        )


def test_streamed_top_n_observably_prunes_input_and_replaced_sort_keys_disable_source_bounds() raises:
    var setting = WorkerSetting(4)
    var n = 196609
    for floating in [False, True]:
        for descending in [False, True]:
            var integers = List[Int64]()
            var floats = List[Float64]()
            var valid = List[Bool]()
            var seconds = List[Int64]()
            var keep = List[Int64]()
            var ids = List[Int64]()
            for row in range(n + 7):
                var value = bound_key(row, descending)
                integers.append(value if bound_valid(row) else Int64.MAX)
                floats.append(
                    Float64(0) / Float64(0) if row % 20003
                    == 19 else Float64(value) / 4
                )
                valid.append(bound_valid(row))
                seconds.append(-Int64(row % 97))
                keep.append(Int64(row % 3))
                ids.append(Int64(row))
            var key = Series(
                "key", Column[Int64](integers^, valid.copy())
            ).slice(7, n)
            if floating:
                key = Series("key", Column[Float64](floats^, valid)).slice(7, n)
            var frame = DataFrame(
                [
                    key^,
                    Series("second", Column[Int64](seconds^)).slice(7, n),
                    Series("keep", Column[Int64](keep^)).slice(7, n),
                    Series("id", Column[Int64](ids^)).slice(7, n),
                ]
            )
            for replaced in [False, True]:
                for nulls_last in [False, True]:
                    var expected = List[Int]()
                    for row in range(7, n + 7):
                        if row % 3 != 1:
                            expected.append(row)

                    def before(
                        a: Int, b: Int
                    ) {floating, descending, replaced, nulls_last} -> Bool:
                        var av = bound_valid(a)
                        var bv = bound_valid(b)
                        if av != bv:
                            return av if nulls_last else bv
                        if av:
                            var x = Float64(bound_key(a, descending)) / (
                                Float64(4) if floating else Float64(1)
                            )
                            var y = Float64(bound_key(b, descending)) / (
                                Float64(4) if floating else Float64(1)
                            )
                            var an = floating and a % 20003 == 19
                            var bn = floating and b % 20003 == 19
                            if an != bn:
                                # NaN is the largest value.
                                return an if descending else not an
                            if replaced:
                                x = -x
                                y = -y
                            if not an and x != y:
                                return x > y if descending else x < y
                        if a % 97 != b % 97:
                            return a % 97 > b % 97
                        return a < b

                    sort(expected, before)
                    expected.shrink(31)
                    var plan = frame.lazy().filter(col("keep") != 1)
                    if replaced:
                        plan = plan.with_columns((0 - col("key")).alias("key"))
                    plan = plan.sort(
                        ["key", "second"],
                        descending=[descending, False],
                        nulls_last=[nulls_last, nulls_last],
                    ).head(31)
                    for batch in [1024, 4097]:
                        var result = plan.profile(
                            engine="cpu", streaming=True, batch_size=batch
                        )
                        var actual = result[0].copy()
                        assert_equal(
                            actual.columns(),
                            List[String](["key", "second", "keep", "id"]),
                        )
                        assert_equal(actual.height(), 31)
                        for row in range(31):
                            var source = expected[row]
                            assert_equal(
                                actual.item(row, "id").int64(), Int64(source)
                            )
                            assert_equal(
                                actual.item(row, "second").int64(),
                                -Int64(source % 97),
                            )
                            assert_equal(
                                actual.item(row, "keep").int64(),
                                Int64(source % 3),
                            )
                            var cell = actual.item(row, "key")
                            assert_equal(
                                cell.is_null(), not bound_valid(source)
                            )
                            if bound_valid(source):
                                if floating and source % 20003 == 19:
                                    assert_true(isnan(cell.float64()))
                                elif floating:
                                    var value = (
                                        Float64(bound_key(source, descending))
                                        / 4
                                    )
                                    assert_equal(
                                        cell.float64(),
                                        -value if replaced else value,
                                    )
                                else:
                                    var value = bound_key(source, descending)
                                    assert_equal(
                                        cell.int64(),
                                        -value if replaced else value,
                                    )
                        var observed = False
                        for row in range(result[1].height()):
                            if (
                                result[1]
                                .item(row, "operator")
                                .string()
                                .startswith("FILTER")
                            ):
                                observed = True
                                var input = (
                                    result[1].item(row, "input_rows").int64()
                                )
                                var executor = (
                                    result[1].item(row, "executor").string()
                                )
                                assert_true(
                                    executor == "pipeline"
                                    or executor == "streaming"
                                )
                                if replaced:
                                    assert_equal(
                                        input,
                                        Int64(n),
                                        "replaced keys must disable input-value bounds",
                                    )
                                else:
                                    assert_true(
                                        input < Int64(n),
                                        "running Top-N bound must observably prune rows before filter evaluation",
                                    )
                        assert_true(observed)
    _ = setting^
