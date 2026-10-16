"""Planning measurements and profiles observe one execution of each source."""
# contract-group: planner.relational
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit
from dataframe.lazy import SCAN_FRAME


def _counters(mut plan: LazyFrame) -> List[ArcPointer[Int]]:
    var counters = List[ArcPointer[Int]]()
    for index in range(len(plan._nodes)):
        if plan._nodes[index].kind == SCAN_FRAME:
            var count = ArcPointer(Int(0))
            plan._nodes[index]._executions = count.copy()
            counters.append(count^)
    return counters^


def _dimension(key: String, value: String, n: Int) raises -> DataFrame:
    var keys = List[Int64]()
    for row in range(n):
        keys.append(Int64(row))
    return DataFrame(
        [
            Series(key, Column[Int64](keys.copy())),
            Series(value, Column[Int64](keys^)),
        ]
    )


def _facts(n: Int, a_modulus: Int, b_modulus: Int) raises -> DataFrame:
    var ids = List[Int64]()
    var a = List[Int64]()
    var b = List[Int64]()
    for row in range(n):
        ids.append(Int64(row))
        a.append(Int64(row % a_modulus))
        b.append(Int64(row % b_modulus))
    return DataFrame(
        [
            Series("id", Column[Int64](ids^)),
            Series("a", Column[Int64](a^)),
            Series("b", Column[Int64](b^)),
        ]
    )


def _assert_members(
    actual: DataFrame,
    n: Int,
    a_modulus: Int,
    b_modulus: Int,
    a_limit: Int,
    b_limit: Int,
) raises:
    var ordered = actual.sort("id")
    var position = 0
    for row in range(n):
        if row % a_modulus < a_limit and row % b_modulus < b_limit:
            assert_equal(ordered.item(position, "id").int64(), Int64(row))
            assert_equal(
                ordered.item(position, "av").int64(), Int64(row % a_modulus)
            )
            assert_equal(
                ordered.item(position, "bv").int64(), Int64(row % b_modulus)
            )
            position += 1
    assert_equal(actual.height(), position)


def test_join_measurements_are_reused_when_dimension_order_changes_or_stays_fixed() raises:
    for limits in [(3, 6), (12, 2)]:
        for streaming in [False, True]:
            var query = (
                _facts(61, 20, 7)
                .lazy()
                .join(
                    _dimension("a", "av", 20)
                    .lazy()
                    .filter(col("av") < lit(Int64(limits[0]))),
                    "a",
                )
                .join(
                    _dimension("b", "bv", 7)
                    .lazy()
                    .filter(col("bv") < lit(Int64(limits[1]))),
                    "b",
                )
            )
            var counts = _counters(query)
            var planned = query._optimized()
            planned._order_joins(streaming, 7)
            assert_equal(counts[0][], 0)
            assert_equal(counts[1][], 1)
            assert_equal(counts[2][], 1)
            assert_equal(len(planned._frames), 3)
            assert_equal(len(planned._schemas), 3)
            _assert_members(
                planned._execute(len(planned._nodes) - 1, False, streaming, 7),
                61,
                20,
                7,
                limits[0],
                limits[1],
            )
            planned._order_joins(streaming, 7)
            assert_equal(len(planned._frames), 3)
            for count in counts:
                assert_equal(count[], 1)


def test_rejected_measured_push_candidate_still_executes_its_source_once() raises:
    for streaming in [False, True]:
        var host = _facts(140, 20, 7)
        var query = (
            host.select(["id"])
            .lazy()
            .join(host.lazy().select(["id", "a"]), "id")
            .join(
                _dimension("a", "av", 20)
                .lazy()
                .filter(col("av") < lit(Int64(14))),
                "a",
            )
        )
        var counts = _counters(query)
        var planned = query._optimized()
        planned._order_joins(streaming, 7)
        assert_equal(counts[0][], 0)
        assert_equal(counts[1][], 0)
        assert_equal(counts[2][], 1)
        var actual = planned._execute(
            len(planned._nodes) - 1, False, streaming, 7
        ).sort("id")
        var position = 0
        for row in range(140):
            if row % 20 < 14:
                assert_equal(actual.item(position, "id").int64(), Int64(row))
                assert_equal(
                    actual.item(position, "av").int64(), Int64(row % 20)
                )
                position += 1
        assert_equal(actual.height(), position)
        for count in counts:
            assert_equal(count[], 1)


def test_profiles_count_rows_and_single_join_builds_across_batches_without_retaining_report_state() raises:
    var query = (
        _facts(1031, 17, 13)
        .lazy()
        .filter(col("b") < lit(Int64(7)))
        .join(_dimension("a", "av", 17).lazy(), "a")
        .join(_dimension("b", "bv", 13).lazy(), "b")
        .select(["id", "av", "bv"])
    )
    var kept = 0
    for row in range(1031):
        kept += Int(row % 13 < 7)
    for streaming in [False, True]:
        for batch in [1, 7, 64, 257]:
            assert_true(not query._report)
            var profiled = query.profile(
                optimize=False,
                streaming=streaming,
                batch_size=batch,
                engine="cpu",
            )
            _assert_members(profiled[0], 1031, 17, 13, 17, 7)
            ref report = profiled[1]
            var joins = 0
            var filters = 0
            var scan_rows = 0
            for row in range(report.height()):
                if row:
                    assert_true(
                        report.item(row, "node").int64()
                        > report.item(row - 1, "node").int64()
                    )
                var operator = report.item(row, "operator").string()
                assert_equal(report.item(row, "executions").int64(), Int64(1))
                assert_true(report.item(row, "busy_ms").float64() >= 0)
                assert_true(report.item(row, "wall_ms").float64() >= 0)
                if operator.startswith("SCAN frame"):
                    scan_rows += Int(report.item(row, "output_rows").int64())
                elif operator.startswith("FILTER"):
                    filters += 1
                    assert_equal(
                        report.item(row, "input_rows").int64(), Int64(1031)
                    )
                    assert_equal(
                        report.item(row, "output_rows").int64(), Int64(kept)
                    )
                elif operator.startswith("JOIN"):
                    joins += 1
                    assert_equal(report.item(row, "builds").int64(), Int64(1))
                    assert_equal(
                        report.item(row, "input_rows").int64(), Int64(kept)
                    )
                    assert_equal(
                        report.item(row, "output_rows").int64(), Int64(kept)
                    )
            assert_equal(joins, 2)
            assert_equal(filters, 1)
            assert_equal(scan_rows, 1031 + 17 + 13)
            assert_true(not query._report)
            _assert_members(
                query.collect(
                    optimize=False,
                    streaming=streaming,
                    batch_size=batch,
                    engine="cpu",
                ),
                1031,
                17,
                13,
                17,
                7,
            )
            assert_true(not query._report)
