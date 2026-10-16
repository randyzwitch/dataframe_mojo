"""CPU witnesses for rank's default-stream materialization boundary."""
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, Series, Expr, col, lit, when
from tests.support.lifecycle_jobs import WorkerSetting
from tests.tooling.checked_rank_cpu_proofs import ordered_source


def test_cpu_checked_rank_default_stream_fault_order() raises:
    var setting = WorkerSetting(1)
    var completed = 0
    for fixture_index in range(2):
        var frame = ordered_source(
            2050 if fixture_index == 0 else 10000,
            1500 if fixture_index == 0 else 9000,
        )
        var expected = (
            "overflow" if fixture_index == 0 else "nonnegative exponent"
        )
        for partitioned in [False, True]:
            var operand = (col("x") + col("y")).pow(col("p"))
            var ranked = operand.rank("min")
            if partitioned:
                ranked = ranked.over("k")
            for form in range(4):
                var expression = ranked.copy()
                if form == 2:
                    expression = (
                        when(lit(False)).then(ranked).otherwise(lit(Int64(7)))
                    )
                var query = frame.lazy().select(expression.alias("r"))
                if form == 1:
                    query = frame.lazy().with_columns(expression.alias("r"))
                elif form == 3:
                    query = frame.lazy().filter(ranked > 0)
                for head0 in [False, True]:
                    var consumer = query.head(0) if head0 else query.copy()
                    for mode in range(3):
                        print(
                            "CPU_RANK_STREAM",
                            fixture_index,
                            partitioned,
                            form,
                            head0,
                            mode,
                            expected,
                            flush=True,
                        )
                        with assert_raises(contains=expected):
                            if mode == 0:
                                # Omit streaming and batch_size: actual public defaults.
                                _ = consumer.collect(
                                    engine="cpu", optimize=False
                                )
                            else:
                                _ = consumer.collect(
                                    engine="cpu",
                                    optimize=False,
                                    streaming=True,
                                    batch_size=1 if mode == 1 else 65536,
                                )
                        completed += 1
    assert_equal(completed, 96)
    _ = setting^


def test_cpu_checked_rank_default_stream_mean_sibling_literals() raises:
    var setting = WorkerSetting(1)
    var columns = List[Series]()
    columns.append(
        Series("year", Column[Int64]([1999, 1999, 2000, 1999, 1999]))
    )
    columns.append(Series("month", Column[Int64]([2, 1, 1, 1, 3])))
    columns.append(Series("sales", Column[Float64]([4, 2, 10, 8, 6])))
    columns.append(Series("k", Column[Int32]([0, 0, 0, 0, 1])))
    var frame = DataFrame(columns^)
    var completed = 0
    for partitioned in [False, True]:
        var ranked = (col("year") * lit(Int64(100)) + col("month")).rank("min")
        var average = col("sales").mean()
        if partitioned:
            ranked = ranked.over("k")
            average = average.over("k")
        var expressions = List[Expr]()
        expressions.append(average.alias("average"))
        expressions.append(ranked.alias("r"))
        var expected: List[Int64] = [3, 1, 4, 1, 1] if partitioned else [
            3,
            1,
            5,
            1,
            4,
        ]
        for optimize in [False, True]:
            var query = frame.lazy().with_columns(expressions)
            var actual = query.collect(engine="cpu", optimize=optimize)
            for row in range(5):
                assert_equal(actual.item(row, "r").int64(), expected[row])
                assert_equal(actual.item(row, "average").float64(), Float64(6))
            assert_equal(
                query.filter(col("r") == 1)
                .select(col("r"))
                .collect(engine="cpu", optimize=optimize)
                .height(),
                3 if partitioned else 2,
            )
            print("CPU_RANK_STREAM_LITERAL", partitioned, optimize, flush=True)
            completed += 1
    assert_equal(completed, 4)
    _ = setting^
