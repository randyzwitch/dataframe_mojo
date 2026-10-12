"""File-backed children remain complete across an as-of execution barrier."""
from std.os import getenv
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    scan_parquet,
    write_parquet,
)
from tests.support.asof_reference import assert_asof_rows


def test_asof_two_parquet_sources_preserve_literal_pairs_through_projection_and_head() raises:
    var directory = getenv("COVERAGE_TEST_TMP")
    assert_true(Bool(directory))
    var left_path = directory + "/asof-source-left.parquet"
    var right_path = directory + "/asof-source-right.parquet"
    var left = DataFrame(
        [
            Series(
                "k",
                Column[Int64](
                    [-1, 0, 2, 2, 5, 7, 8],
                    [True, True, True, False, True, True, True],
                ),
            ),
            Series(
                "g",
                Column[String](
                    ["A", "B", "A", "B", "A", "B", "A"],
                    [True, True, True, True, True, True, False],
                ),
            ),
            Series("left_id", Column[Int32]([0, 1, 2, 3, 4, 5, 6])),
            Series(
                "left_text",
                Column[String](
                    [
                        "π",
                        "",
                        "duplicate",
                        "null key",
                        "after",
                        "tail",
                        "null group",
                    ]
                ),
            ),
        ]
    )
    var payloads: List[String] = [
        "before A",
        "before B",
        "first A",
        "equal B",
        "last A",
        "",
        "near B",
        "after B",
    ]
    var payload_valid: List[Bool] = [
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    var right = DataFrame(
        [
            Series("k", Column[Int64]([-2, -1, 2, 2, 2, 4, 6, 9])),
            Series(
                "g", Column[String](["A", "B", "A", "B", "A", "A", "B", "B"])
            ),
            Series("rid", Column[UInt16]([10, 11, 12, 13, 14, 15, 16, 17])),
            Series(
                "payload", Column[String](payloads.copy(), payload_valid.copy())
            ),
        ]
    )
    # The writer supplies storage only. These pairs are authored independently:
    # backward takes the last equal row; forward the first; nearest breaks an
    # equal-distance tie toward the greater key and takes its last duplicate.
    var pairs: List[List[Int]] = [
        [0, 1, 0, -1, 5, 6, -1],
        [0, 1, 4, -1, 5, 6, -1],
        [2, 3, 5, -1, -1, 7, -1],
        [2, 3, 2, -1, -1, 7, -1],
        [0, 1, 5, -1, 5, 6, -1],
        [0, 1, 4, -1, 5, 6, -1],
    ]
    # Different group boundaries on the two sides and batch size three make
    # matches and duplicate runs span both file and execution batch boundaries.
    write_parquet(left, left_path, row_group_size=2)
    write_parquet(right, right_path, row_group_size=3)
    var option = 0
    for strategy in ["backward", "forward", "nearest"]:
        for exact in [False, True]:
            var selected = pairs[option].copy()
            option += 1
            var joined = scan_parquet(left_path).join_asof(
                scan_parquet(right_path),
                on="k",
                by="g",
                strategy=strategy,
                allow_exact_matches=exact,
            )
            assert_true("ASOF" in joined.explain())
            assert_true("PARQUET" in joined.explain())
            for optimize in [False, True]:
                for streaming in [False, True]:
                    assert_asof_rows(
                        joined.collect(
                            optimize=optimize,
                            streaming=streaming,
                            batch_size=3,
                            engine="cpu",
                        ),
                        left,
                        right,
                        selected,
                        ["k", "g", "left_id", "left_text", "rid", "payload"],
                        ["rid", "payload"],
                    )
                    for limit in [0, 3, 6, 9]:
                        var actual = (
                            joined.select(["payload", "left_id", "g", "rid"])
                            .head(limit)
                            .collect(
                                optimize=optimize,
                                streaming=streaming,
                                batch_size=3,
                                engine="cpu",
                            )
                        )
                        assert_equal(
                            actual.columns(), ["payload", "left_id", "g", "rid"]
                        )
                        assert_equal(
                            actual.dtypes(),
                            [
                                DataType.STRING,
                                DataType.INT32,
                                DataType.STRING,
                                DataType.UINT16,
                            ],
                        )
                        assert_equal(actual.height(), min(limit, 7))
                        for row in range(actual.height()):
                            assert_equal(
                                actual.item(row, "left_id").numeric[
                                    DType.int32
                                ](),
                                Int32(row),
                            )
                            if row == 6:
                                assert_true(actual.item(row, "g").is_null())
                            else:
                                assert_equal(
                                    actual.item(row, "g").string(),
                                    "A" if row % 2 == 0 else "B",
                                )
                            var chosen = selected[row]
                            if chosen < 0:
                                assert_true(actual.item(row, "rid").is_null())
                            else:
                                assert_equal(
                                    actual.item(row, "rid").numeric[
                                        DType.uint16
                                    ](),
                                    UInt16(10 + chosen),
                                )
                            if chosen < 0 or not payload_valid[chosen]:
                                assert_true(
                                    actual.item(row, "payload").is_null()
                                )
                            else:
                                assert_equal(
                                    actual.item(row, "payload").string(),
                                    payloads[chosen],
                                )
