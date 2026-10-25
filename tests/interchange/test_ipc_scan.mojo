"""Lazy IPC scans against independent files and deliberately unreadable bodies."""
# contract-group: io.ipc
from std.os import getenv
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
    read_ipc,
    scan_ipc,
    write_ipc,
)
from dataframe.io.ipc import _IpcBatches
from tests.interchange.test_ipc import fixture, check_metadata
from tests.interchange.test_parquet import assert_core_rows


def test_ipc_scan_construction_is_deferred_schema_reads_only_metadata_and_plans_reopen() raises:
    var path = getenv("COVERAGE_TEST_TMP") + "/ipc-scan-deferred.arrow"
    var plan = scan_ipc(path)
    var full = read_ipc(fixture("core-uncompressed.arrow"))
    write_ipc(full, path)
    for mapped in [False, True]:
        var reusable = scan_ipc(path, memory_map=mapped).select(["s", "x"])
        for _ in range(2):
            assert_true(
                reusable.collect(engine="cpu", batch_size=2).equals(
                    full.select(["s", "x"])
                )
            )
    assert_true(plan.collect(engine="cpu").equals(full))
    assert_equal(
        scan_ipc(fixture("early-corrupt.arrow")).collect_schema(),
        List[String](["x: int64", "s: string", "score: float64"]),
    )
    assert_equal(
        scan_ipc(fixture("late-corrupt.arrow")).select(["s"]).collect_schema(),
        List[String](["s: string"]),
    )
    assert_true("SCAN IPC" in plan.explain())


def test_ipc_scan_projection_and_slices_avoid_unselected_or_skipped_corrupt_bodies() raises:
    for streaming in [False, True]:
        for mapped in [False, True]:
            var early = scan_ipc(
                fixture("early-corrupt.arrow"), memory_map=mapped
            )
            assert_core_rows(
                early.select(["s"]).collect(
                    engine="cpu", streaming=streaming, batch_size=1
                ),
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                ["s"],
            )
            var constant = early.select(
                lit(Int64(7)).alias("constant")
            ).collect(engine="cpu", streaming=streaming, batch_size=1)
            assert_equal(constant.height(), 1)
            assert_equal(constant.item(0, "constant").int64(), Int64(7))
            var window = early.select(["s", "x"]).slice(3, 3)
            assert_core_rows(
                window.collect(engine="cpu", streaming=streaming, batch_size=1),
                [3, 4, 5],
                ["s", "x"],
            )
            var explanation = window.explain()
            assert_true("offset 3" in explanation)
            assert_true("n_rows 3" in explanation)
            assert_true("project" in explanation)
            var late = scan_ipc(
                fixture("late-corrupt.arrow"), memory_map=mapped
            )
            assert_core_rows(
                late.head(3).collect(
                    engine="cpu", streaming=streaming, batch_size=1
                ),
                [0, 1, 2],
                ["x", "s", "score"],
            )
            assert_core_rows(
                late.slice(6, 2).collect(
                    engine="cpu", streaming=streaming, batch_size=1
                ),
                [6, 7],
                ["x", "s", "score"],
            )
            assert_equal(
                early.head(0)
                .collect(engine="cpu", streaming=streaming)
                .height(),
                0,
            )
    for path in ["early-corrupt.arrow", "late-corrupt.arrow"]:
        var failed = False
        try:
            _ = scan_ipc(fixture(path)).collect(engine="cpu", batch_size=1)
        except error:
            assert_true("ZSTD" in String(error))
            failed = True
        assert_true(failed)


def test_ipc_scan_offsets_nested_slices_and_codecs_match_unoptimized_execution() raises:
    for codec in ["uncompressed", "lz4", "zstd"]:
        var path = fixture("core-" + codec + ".arrow")
        var full = read_ipc(path)
        for streaming in [False, True]:
            for optimize in [False, True]:
                var base = scan_ipc(path)
                assert_true(
                    base.slice(2, 5)
                    .slice(1, 2)
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=2,
                    )
                    .equals(full.slice(3, 2))
                )
                assert_true(
                    base.slice(3)
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=2,
                    )
                    .equals(full.slice(3))
                )
                assert_true(
                    base.slice(-2, 2)
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=2,
                    )
                    .equals(full.slice(-2, 2))
                )
                assert_true(
                    base.slice(99, 2)
                    .collect(
                        engine="cpu", optimize=optimize, streaming=streaming
                    )
                    .equals(full.clear())
                )
        assert_equal(
            scan_ipc(path)
            .slice(Int.MAX)
            .slice(1)
            .collect(engine="cpu")
            .height(),
            0,
        )


def test_ipc_scan_limits_respect_filters_sort_and_row_index_boundaries() raises:
    var path = fixture("core-zstd.arrow")
    for streaming in [False, True]:
        var base = scan_ipc(path)
        assert_core_rows(
            base.filter(col("x") >= 100)
            .select(["s", "x"])
            .head(2)
            .collect(engine="cpu", streaming=streaming, batch_size=1),
            [4, 5],
            ["s", "x"],
        )
        assert_equal(
            base.head(3)
            .filter(col("x") >= 100)
            .collect(engine="cpu", streaming=streaming, batch_size=1)
            .height(),
            0,
        )
        assert_core_rows(
            base.sort("x", descending=True)
            .head(2)
            .collect(engine="cpu", streaming=streaming, batch_size=1),
            [8, 7],
            ["x", "s", "score"],
        )
        var indexed = (
            base.with_row_index("row")
            .slice(2, 2)
            .collect(engine="cpu", streaming=streaming, batch_size=1)
        )
        assert_equal(indexed.item(0, "row").int64(), Int64(2))
        assert_equal(indexed.item(1, "row").int64(), Int64(3))


def test_ipc_scan_dictionaries_remain_consistent_for_streamed_groups_and_joins() raises:
    var path = fixture("dictionary-delta.arrow")
    var dimension = DataFrame(
        [
            Series("category", StringColumn(["green", "red", "blue"])).cast(
                DataType.CATEGORICAL
            ),
            Series("value", Column[Int64]([40, 10, 20])),
        ]
    )
    for streaming in [False, True]:
        for optimize in [False, True]:
            var grouped = (
                scan_ipc(path)
                .group_by("category", maintain_order=True)
                .agg(col("x").sum())
                .collect(
                    engine="cpu",
                    streaming=streaming,
                    optimize=optimize,
                    batch_size=1,
                )
            )
            var sums: List[Int64] = [3, 6, 2, 4]
            var labels: List[String] = ["red", "blue", "", "green"]
            assert_equal(grouped.height(), 4)
            for row in range(4):
                assert_equal(grouped.item(row, "x").int64(), sums[row])
                assert_equal(grouped.item(row, "category").is_null(), row == 2)
                if row != 2:
                    assert_equal(
                        grouped.item(row, "category").string(), labels[row]
                    )
            var matched = (
                scan_ipc(path)
                .filter(col("category") == "green")
                .select(["x"])
                .collect(
                    engine="cpu",
                    streaming=streaming,
                    optimize=optimize,
                    batch_size=1,
                )
            )
            assert_equal(matched.height(), 1)
            assert_equal(matched.item(0, "x").int64(), Int64(4))
            var joined = (
                scan_ipc(path)
                .join(dimension.lazy(), "category", how="left")
                .sort("x")
                .collect(
                    engine="cpu",
                    streaming=streaming,
                    optimize=optimize,
                    batch_size=1,
                )
            )
            var values: List[Int64] = [10, 20, 0, 10, 40, 20]
            assert_equal(joined.height(), 6)
            for row in range(6):
                assert_equal(joined.item(row, "value").is_null(), row == 2)
                if row != 2:
                    assert_equal(joined.item(row, "value").int64(), values[row])
    var nested = fixture("nested-dictionary.arrow")
    assert_true(
        scan_ipc(nested)
        .collect(engine="cpu", batch_size=1)
        .equals(read_ipc(nested))
    )
    assert_equal(
        scan_ipc(nested)
        .slice(2, 1)
        .collect(engine="cpu", batch_size=1)
        .item(0, "items")
        .list()
        .get(0)
        .string(),
        "red",
    )


def test_ipc_scan_typed_metadata_and_zero_column_or_empty_inputs_keep_shape() raises:
    var typed_path = fixture("typed.arrow")
    var full = read_ipc(typed_path)
    for streaming in [False, True]:
        var typed = scan_ipc(typed_path).collect(
            engine="cpu", streaming=streaming, batch_size=1
        )
        assert_true(typed.equals(full))
        check_metadata(typed.column("i"))
        assert_equal(
            typed.column("narrow_precision_wide_storage").dtype(),
            DataType.decimal(9, 2, 128),
        )
        var zero = (
            scan_ipc(fixture("zero-columns.arrow"))
            .slice(4, 3)
            .collect(engine="cpu", streaming=streaming, batch_size=1)
        )
        assert_equal(zero.height(), 3)
        assert_equal(zero.width(), 0)
        var empty = scan_ipc(fixture("empty.arrow")).collect(
            engine="cpu", streaming=streaming, batch_size=1
        )
        assert_true(
            empty.equals(read_ipc(fixture("core-uncompressed.arrow")).clear())
        )
        var dropped = (
            scan_ipc(fixture("core-uncompressed.arrow"))
            .drop(["x", "s", "score"])
            .collect(engine="cpu", streaming=streaming, batch_size=1)
        )
        assert_equal(dropped.height(), 9)
        assert_equal(dropped.width(), 0)


def test_ipc_scan_repeated_sources_keep_distinct_slices_and_projection_needs() raises:
    var path = fixture("core-uncompressed.arrow")
    var base = scan_ipc(path)
    var query = (
        base.slice(0, 4).select(["x"]).concat(base.slice(4, 5).select(["x"]))
    )
    for streaming in [False, True]:
        assert_core_rows(
            query.collect(engine="cpu", streaming=streaming, batch_size=1),
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            ["x"],
        )
        var sums = base.select(
            col("x").cast(DataType.FLOAT64).sum().alias("total")
        ).concat(base.select(col("score").sum().alias("total")))
        var actual = sums.collect(
            engine="cpu", streaming=streaming, batch_size=1
        )
        assert_equal(actual.height(), 2)
        assert_equal(actual.item(0, "total").float64(), Float64(710))
        assert_equal(actual.item(1, "total").float64(), Float64(42))


def test_ipc_scan_batch_reader_handles_partial_batches_early_stop_and_errors() raises:
    var path = fixture("core-zstd.arrow")
    var source = _IpcBatches(path, ["x"], True, False, 2, 5, 2)
    var rows = List[Int64]()
    var sizes = List[Int]()
    while True:
        var next = source.next()
        if not next:
            break
        var batch = next.take()
        sizes.append(batch.height())
        for row in range(batch.height()):
            rows.append(batch.item(row, "x").int64())
    var expected: List[Int64] = [2, 3, 100, 101, 102]
    assert_equal(rows, expected)
    assert_equal(sizes, List[Int]([1, 2, 1, 1]))
    assert_true(not source.next())
    # Destruction closes a stream before the unread corrupted second body.
    var early = _IpcBatches(
        fixture("late-corrupt.arrow"), ["x"], True, True, 0, -1, 1
    )
    var first = early.next()
    assert_equal(first.value().item(0, "x").int64(), Int64(0))
    var failed = False
    try:
        _ = scan_ipc(fixture("invalid.arrow")).collect_schema()
    except:
        failed = True
    assert_true(failed)
    assert_core_rows(
        scan_ipc(path).head(1).collect(engine="cpu"), [0], ["x", "s", "score"]
    )
