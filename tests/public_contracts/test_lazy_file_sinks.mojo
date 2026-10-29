"""Lazy sinks publish batch output atomically with public format options."""
# contract-group: io.sinks
from std.os import getenv, mkdir, listdir
from std.os.path import exists
from std.testing import assert_equal, assert_raises, assert_false
from dataframe import (
    Column,
    DataFrame,
    Series,
    col,
    read_csv,
    read_parquet,
    scan_csv,
    write_parquet,
    parquet_row_group_statistics,
)
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import ScopedEnvironment


def _path(name: String) raises -> String:
    var root = getenv("COVERAGE_TEST_TMP") + "/public-lazy-sinks"
    if not exists(root):
        mkdir(root)
    return root + "/" + name


def test_public_lazy_sink_batches_aggregate_and_empty_outputs() raises:
    var source = _path("source.csv")
    with open(source, "w") as file:
        file.write("id,x\n1,10\n2,20\n3,30\n4,40\n5,50\n")
    var plan = (
        scan_csv(source)
        .with_columns((col("x") + 1).alias("plus"))
        .select_exprs([col("id"), col("plus")])
    )
    var expected = DataFrame(
        [
            Series("id", Column[Int64]([1, 2, 3, 4, 5])),
            Series("plus", Column[Int64]([11, 21, 31, 41, 51])),
        ]
    )
    for optimize in [False, True]:
        var csv = _path("rows-" + String(optimize) + ".csv")
        plan.sink_csv(csv, batch_size=1, buffer_size=1, optimize=optimize)
        assert_frame(read_csv(csv), expected)
        var parquet = _path("rows-" + String(optimize) + ".parquet")
        plan.sink_parquet(
            parquet, batch_size=1, row_group_size=2, optimize=optimize
        )
        assert_frame(read_parquet(parquet), expected)
        var statistics = parquet_row_group_statistics(parquet)
        assert_equal(statistics.height(), 3)
        assert_equal(statistics.item(0, "rows").int64(), Int64(2))
        assert_equal(statistics.item(1, "rows").int64(), Int64(2))
        assert_equal(statistics.item(2, "rows").int64(), Int64(1))
    var sum_path = _path("sum.parquet")
    scan_csv(source).select(col("x").sum().alias("total")).sink_parquet(
        sum_path, batch_size=1
    )
    assert_frame(
        read_parquet(sum_path),
        DataFrame([Series("total", Column[Int64]([150]))]),
    )
    var empty_path = _path("empty.parquet")
    plan.filter(col("id") < 0).sink_parquet(empty_path, batch_size=1)
    assert_frame(read_parquet(empty_path), expected.clear())
    var empty_csv = _path("empty.csv")
    plan.filter(col("id") < 0).sink_csv(empty_csv, batch_size=1)
    with open(empty_csv, "r") as file:
        assert_equal(file.read(), String("id,plus\n"))


def test_public_lazy_sink_can_replace_input_and_preserves_destination_on_late_error() raises:
    var source = _path("replace.csv")
    with open(source, "w") as file:
        file.write("id,x\n1,10\n2,20\n3,30\n")
    scan_csv(source).filter(col("id") > 1).sink_csv(source, batch_size=1)
    assert_frame(
        read_csv(source),
        DataFrame(
            [
                Series("id", Column[Int64]([2, 3])),
                Series("x", Column[Int64]([20, 30])),
            ]
        ),
    )
    var bad_source = _path("late-error.csv")
    with open(bad_source, "w") as file:
        file.write("id,text\n")
        for row in range(16):
            file.write(String(row) + ",12\n")
        file.write("16,bad\n")
    var destination = _path("preserved.csv")
    with open(destination, "w") as file:
        file.write("original\n")
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "1")
    with assert_raises():
        scan_csv(bad_source).select_exprs(
            [col("id"), col("text").cast("int64").alias("n")]
        ).sink_csv(destination, batch_size=1, buffer_size=1)
    with open(destination, "r") as file:
        assert_equal(file.read(), String("original\n"))
    var root = getenv("COVERAGE_TEST_TMP") + "/public-lazy-sinks"
    for name in listdir(root):
        assert_false(name.startswith("preserved.csv.dataframe-"))
    _ = threads^


def test_public_partition_keyword_roundtrips_sorted_hive_files() raises:
    var frame = DataFrame(
        [
            Series("year", Column[Int64]([2024, 2025, 2024])),
            Series("x", Column[Int64]([1, 2, 3])),
        ]
    )
    var root = _path("partitioned")
    write_parquet(
        frame,
        root,
        partition_by=[String("year")],
        compression="snappy",
        row_group_size=1,
    )
    var actual = read_parquet(root + "/**/*.parquet", hive_partitioning=True)
    assert_frame(
        actual,
        DataFrame(
            [
                Series("year", Column[Int64]([2024, 2024, 2025])),
                Series("x", Column[Int64]([1, 3, 2])),
            ]
        ),
    )
