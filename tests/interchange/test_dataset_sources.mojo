"""File lists and recursive globs preserve complete cells in sorted path order."""
# contract-group: io.dataset_sources
from std.testing import assert_equal, assert_raises
from dataframe import (
    Column,
    CsvSchema,
    DataFrame,
    DataType,
    Series,
    col,
    read_csv,
    read_parquet,
    scan_csv,
    scan_parquet,
)
from dataframe.io.file_sources import resolve_paths
from tests.support.coverage_compare import assert_frame

comptime SOURCES = "tests/fixtures/dataset_sources/"


def csv_expected() raises -> DataFrame:
    return DataFrame(
        [
            Series("id", Column[Int64]([1, 2, 5, 6, 3, 4])),
            Series(
                "val",
                Column[String](
                    ["alpha", "", "line\nbreak", "", "quoted, value", "café"],
                    [True, True, True, False, True, True],
                ),
            ),
        ]
    )


def test_path_sources_sort_preserve_repetitions_and_expand_component_globs() raises:
    var expected: List[String] = [
        SOURCES + "a.csv",
        SOURCES + "nested/b.csv",
        SOURCES + "nested/deep/c.csv",
        SOURCES + "odd[1].csv",
        SOURCES + "é.csv",
    ]
    assert_equal(resolve_paths([SOURCES + "**/*.csv"]), expected)
    assert_equal(
        resolve_paths([SOURCES + "nested/../a.csv", SOURCES + "a.csv"]),
        List[String]([SOURCES + "a.csv", SOURCES + "nested/../a.csv"]),
    )
    assert_equal(
        resolve_paths([SOURCES + "[ao]*.csv"]),
        List[String]([SOURCES + "a.csv", SOURCES + "odd[1].csv"]),
    )
    assert_equal(
        resolve_paths([SOURCES + "nested/?.csv"]),
        List[String]([SOURCES + "nested/b.csv"]),
    )
    assert_equal(
        resolve_paths([SOURCES + ".*.csv"]),
        List[String]([SOURCES + ".hidden.csv"]),
    )
    assert_equal(
        resolve_paths([SOURCES + "odd[1].csv"], glob=False),
        List[String]([SOURCES + "odd[1].csv"]),
    )
    with assert_raises(contains="No files matched"):
        _ = resolve_paths([SOURCES + "missing*.csv"])
    with assert_raises(contains="at least one path"):
        _ = resolve_paths(List[String]())


def test_multi_csv_inferred_explicit_projection_and_global_row_limits() raises:
    var expected = csv_expected()
    var pattern = SOURCES + "**/*.csv"
    assert_frame(read_csv(pattern), expected)
    assert_frame(
        read_csv(
            [
                SOURCES + "é.csv",
                SOURCES + "a.csv",
                SOURCES + "nested/**/*.csv",
                SOURCES + "odd[[]1].csv",
            ]
        ),
        expected,
    )
    var paths = resolve_paths([pattern])
    assert_frame(read_csv(paths, glob=False), expected)
    assert_frame(read_csv(paths, CsvSchema.of(expected), glob=False), expected)
    var schema: List[Tuple[String, DataType]] = [
        ("id", DataType.INT64),
        ("val", DataType.STRING),
    ]
    assert_frame(read_csv(paths, schema, glob=False), expected)
    for n in [0, 1, 2, 3, 5, 6, 20]:
        assert_frame(read_csv(pattern, n_rows=n), expected.head(n))
    assert_frame(
        read_csv(pattern, columns=["val"]), expected.select([String("val")])
    )
    assert_frame(
        read_csv([SOURCES + "a.csv", SOURCES + "a.csv"]),
        expected.head(2).vstack(expected.head(2)),
    )
    with assert_raises(contains="concat vertical"):
        _ = read_csv([SOURCES + "a.csv", SOURCES + ".mismatch/wrong.csv"])


def test_multi_csv_lazy_modes_filters_head_and_literal_metacharacters() raises:
    var expected = csv_expected()
    var pattern = SOURCES + "**/*.csv"
    for optimize in [False, True]:
        for streaming in [False, True]:
            assert_frame(
                scan_csv(pattern).collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=1,
                ),
                expected,
            )
            assert_frame(
                scan_csv(
                    resolve_paths([pattern]), CsvSchema.of(expected), glob=False
                )
                .filter(col("id") > 2)
                .select_exprs([col("val"), col("id")])
                .head(3)
                .collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=2,
                ),
                expected.filter(col("id") > 2)
                .select([String("val"), "id"])
                .head(3),
            )
    var literal = SOURCES + "odd[1].csv"
    var one = expected.slice(4, 1)
    assert_frame(scan_csv(literal, glob=False).collect(engine="cpu"), one)
    assert_frame(scan_csv([literal], glob=False).collect(engine="cpu"), one)


def test_multi_parquet_paths_globs_and_literal_brackets_preserve_row_groups() raises:
    var a = SOURCES + "p-a.parquet"
    var b = SOURCES + "p-b[1].parquet"
    var expected = DataFrame(
        [
            Series("id", Column[Int64]([7, 8, 9, 10, 11, 12])),
            Series(
                "val",
                Column[String](
                    ["seven", "", "nine", "ten", "", "twelve"],
                    [True, False, True, True, True, True],
                ),
            ),
        ]
    )
    assert_frame(read_parquet([b, a], glob=False), expected)
    assert_frame(read_parquet(SOURCES + "p-*.parquet"), expected)
    assert_frame(read_parquet(b, glob=False), expected.tail(3))
    assert_frame(
        read_parquet([b, a], glob=False, row_groups=List[Int]([1])),
        expected.take([2, 5]),
    )
    assert_frame(
        read_parquet([b, a], glob=False, columns=["val"]),
        expected.select([String("val")]),
    )
    for optimize in [False, True]:
        for streaming in [False, True]:
            assert_frame(
                scan_parquet(SOURCES + "p-*.parquet").collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=1,
                ),
                expected,
            )
            assert_frame(
                scan_parquet([b, a], glob=False)
                .filter(col("id") >= 8)
                .select(col("val"))
                .head(3)
                .collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=2,
                ),
                expected.filter(col("id") >= 8).select([String("val")]).head(3),
            )
            assert_frame(
                scan_parquet(b, glob=False)
                .filter(col("id") > 0)
                .collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=1,
                ),
                expected.tail(3),
            )
