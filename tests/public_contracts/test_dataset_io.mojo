"""Public dataset options preserve schema, rows, and lazy execution choices."""
# contract-group: io.sources.dataset
from std.os import getenv, mkdir
from std.os.path import exists
from std.testing import assert_raises
from dataframe import (
    Column,
    DataFrame,
    LazyFrame,
    Series,
    StringColumn,
    col,
    len as row_len,
    read_csv,
    read_parquet,
    scan_csv,
    scan_parquet,
    write_parquet,
)
from tests.support.coverage_compare import assert_frame


def _path(name: String) raises -> String:
    var root = getenv("COVERAGE_TEST_TMP") + "/public-dataset-io"
    if not exists(root):
        mkdir(root)
    var parts = name.split("/")
    var directory = root.copy()
    for i in range(len(parts) - 1):
        directory += "/" + String(parts[i])
        if not exists(directory):
            mkdir(directory)
    return root + "/" + name


def _csv(name: String, text: String) raises -> String:
    var path = _path(name)
    with open(path, "w") as output:
        output.write(text)
    return path


def _modes(plan: LazyFrame, expected: DataFrame) raises:
    for optimize in [False, True]:
        for streaming in [False, True]:
            assert_frame(
                plan.collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=1,
                ),
                expected,
            )


def test_public_hive_paths_filter_projection_and_limit_barrier() raises:
    for parquet in [False, True]:
        var suffix = String("parquet") if parquet else String("csv")
        var a = _path(suffix + "/year=2024/part." + suffix)
        var b = _path(suffix + "/year=2025/part." + suffix)
        if parquet:
            write_parquet(DataFrame([Series("x", Column[Int64]([1]))]), a)
            write_parquet(DataFrame([Series("x", Column[Int64]([2]))]), b)
        else:
            with open(a, "w") as file:
                file.write("x\n1\n")
            with open(b, "w") as file:
                file.write("x\n2\n")
        var paths: List[String] = [b, a]
        var plan = scan_parquet(
            paths, hive_partitioning=True, include_file_paths="source"
        ) if parquet else scan_csv(
            paths, hive_partitioning=True, include_file_paths="source"
        )
        var expected = DataFrame(
            [
                Series("x", Column[Int64]([1, 2])),
                Series("year", Column[Int64]([2024, 2025])),
                Series("source", StringColumn([a, b])),
            ]
        )
        var eager = read_parquet(
            paths, hive_partitioning=True, include_file_paths="source"
        ) if parquet else read_csv(
            paths, hive_partitioning=True, include_file_paths="source"
        )
        assert_frame(eager, expected)
        _modes(plan, expected)
        _modes(
            plan.filter(col("year") == 2025).select(col("x")),
            DataFrame([Series("x", Column[Int64]([2]))]),
        )
        _modes(plan.head(1).filter(col("year") == 2025), expected.clear())
        _modes(
            plan.select(row_len().alias("n")),
            DataFrame([Series("n", Column[Int64]([2]))]),
        )


def test_public_union_and_missing_extra_rules() raises:
    var a = _csv("union/a.csv", "id,left\n1,10\n")
    var b = _csv("union/b.csv", "id,right\n2,20\n")
    var paths: List[String] = [b, a]
    var union = DataFrame(
        [
            Series("id", Column[Int64]([1, 2])),
            Series("left", Column[Int64]([10, 0], [True, False])),
            Series("right", Column[Int64]([0, 20], [False, True])),
        ]
    )
    assert_frame(read_csv(paths, union_by_name=True), union)
    _modes(scan_csv(paths, union_by_name=True), union)
    var reference = union.select([String("id"), "left"])
    assert_frame(
        read_csv(paths, missing_columns="insert", extra_columns="ignore"),
        reference,
    )
    _modes(
        scan_csv(paths, missing_columns="insert", extra_columns="ignore"),
        reference,
    )
    with assert_raises():
        _ = read_csv(paths)
    with assert_raises():
        _ = (
            scan_csv(paths, missing_columns="insert")
            .select(col("id"))
            .collect(engine="cpu")
        )


def test_public_projection_does_not_validate_unobserved_dtype_and_hive_overrides_physical() raises:
    var a = _path("types/a.parquet")
    var b = _path("types/b.parquet")
    write_parquet(
        DataFrame(
            [
                Series("id", Column[Int64]([1])),
                Series("unused", Column[Int64]([10])),
            ]
        ),
        a,
    )
    write_parquet(
        DataFrame(
            [
                Series("id", Column[Int64]([2])),
                Series("unused", StringColumn(["text"])),
            ]
        ),
        b,
    )
    var paths: List[String] = [a, b]
    var expected = DataFrame([Series("id", Column[Int64]([1, 2]))])
    assert_frame(
        read_parquet(paths, columns=[String("id")], union_by_name=True),
        expected,
    )
    _modes(scan_parquet(paths, union_by_name=True).select(col("id")), expected)
    with assert_raises():
        _ = read_parquet(paths, union_by_name=True)
    var retained = _csv("override/year=2024/data.csv", "year,x\n999,3\n")
    var overridden = DataFrame(
        [Series("year", Column[Int64]([2024])), Series("x", Column[Int64]([3]))]
    )
    assert_frame(read_csv(retained, hive_partitioning=True), overridden)
    _modes(scan_csv(retained, hive_partitioning=True), overridden)
    var collision = _csv("collision/data.csv", "source\nvalue\n")
    with assert_raises(contains="include_file_paths"):
        _ = read_csv(collision, include_file_paths="source")


def test_binary_dataset_plans_ignore_unused_dtype_mismatches_in_all_modes() raises:
    for parquet in [False, True]:
        var suffix = String("parquet") if parquet else String("csv")
        var a = _path("binary/" + suffix + "/a." + suffix)
        var b = _path("binary/" + suffix + "/b." + suffix)
        if parquet:
            write_parquet(
                DataFrame(
                    [
                        Series("id", Column[Int64]([1])),
                        Series("unused", Column[Int64]([10])),
                    ]
                ),
                a,
            )
            write_parquet(
                DataFrame(
                    [
                        Series("id", Column[Int64]([2])),
                        Series("unused", StringColumn(["text"])),
                    ]
                ),
                b,
            )
        else:
            with open(a, "w") as file:
                file.write("id,unused\n1,10\n")
            with open(b, "w") as file:
                file.write("id,unused\n2,text\n")
        var paths: List[String] = [a, b]
        var source = scan_parquet(
            paths, union_by_name=True
        ) if parquet else scan_csv(paths, union_by_name=True)
        var keys = DataFrame([Series("id", Column[Int64]([2]))])
        _modes(
            source.join(keys.lazy(), ["id"]).select(col("id")),
            DataFrame([Series("id", Column[Int64]([2]))]),
        )
        _modes(
            source.concat(source).select(col("id")),
            DataFrame([Series("id", Column[Int64]([1, 2, 1, 2]))]),
        )
        _modes(
            source.concat(source).concat(source).select(col("id")),
            DataFrame([Series("id", Column[Int64]([1, 2, 1, 2, 1, 2]))]),
        )


def test_unoptimized_concat_preserves_intermediate_expression_dependencies() raises:
    var path = _csv("binary/dependencies.csv", "id,unused\n1,10\n")
    var source = scan_csv([path], union_by_name=True)
    var transformed = source.select_exprs(
        [
            col("id"),
            (col("unused") + 1).alias("unused"),
        ]
    )
    _modes(
        source.concat(transformed).select(col("id")),
        DataFrame([Series("id", Column[Int64]([1, 1]))]),
    )
    var frame = DataFrame(
        [
            Series("id", Column[Int64]([2])),
            Series("unused", Column[Int64]([20])),
        ]
    )
    _modes(
        source.concat(frame.lazy()).select(col("id")),
        DataFrame([Series("id", Column[Int64]([1, 2]))]),
    )
