"""Join aggregate cardinality and physical scan projections use scalar row pairs."""
# contract-group: planner.row_rewrites
# contract-group: planner.relational
from std.os import getenv
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    LazyFrame,
    Series,
    col,
    lit,
    scan_csv,
)
from dataframe.lazy import SCAN_FRAME, SCAN_CSV
from tests.support.join_typed_reference import nested_loop_pairs
from tests.support.coverage_compare import assert_frame


def _string_side(
    n: Int, left: Bool
) raises -> Tuple[DataFrame, List[Int], List[Bool]]:
    var keys = List[String]()
    var valid = List[Bool]()
    var ranks = List[Int]()
    var values = List[Int64]()
    var value_valid = List[Bool]()
    var unused = List[String]()
    for index in range(n):
        var key = index % (5 if left else 6)
        var present = index % (13 if left else 17) != (2 if left else 4)
        keys.append("independent repeated key " + String(key))
        valid.append(present)
        ranks.append(key if present else -1)
        values.append(Int64(index))
        value_valid.append(index % (7 if left else 9) != (1 if left else 3))
        unused.append("unused,quoted UTF8: π")
    return (
        DataFrame(
            [
                Series("k", Column[String](keys^, valid)),
                Series(
                    "lp" if left else "rp",
                    Column[Int64](values^, value_valid.copy()),
                ),
                Series(
                    "unused_left" if left else "unused_right",
                    Column[String](unused^),
                ),
            ]
        ),
        ranks^,
        value_valid^,
    )


def _compound_side(
    left: Bool,
) raises -> Tuple[DataFrame, List[Int], List[Bool]]:
    var nan = Float64(0) / Float64(0)
    var values: List[Float64] = [nan, -0.0, 0.0, nan, 9.0, 0.0]
    var tags: List[String] = ["x", "x", "x", "y", "z", "x"]
    var ranks: List[Int] = [0, 1, 1, 2, 3, -1]
    var payload_valid: List[Bool] = [False, True, True, False, True, False]
    if not left:
        values = [-0.0, nan, 0.0, nan, 8.0, 0.0]
        tags = ["x", "x", "y", "y", "z", "x"]
        ranks = [1, 0, 4, 2, 5, -1]
        payload_valid = [True, True, False, True, False, False]
    var valid: List[Bool] = [True, True, True, True, True, False]
    return (
        DataFrame(
            [
                Series("f", Column[Float64](values^, valid)),
                Series("tag", Column[String](tags^)),
                Series(
                    "lp" if left else "rp",
                    Column[Int64](
                        [10, 20, 30, 40, 50, -999], payload_valid.copy()
                    ),
                ),
                Series(
                    "unused_left" if left else "unused_right",
                    Column[String](["a", "b", "c", "d", "e", "f"]),
                ),
            ]
        ),
        ranks^,
        payload_valid^,
    )


def _count_query(
    left: DataFrame, right: DataFrame, keys: List[String], how: String
) raises -> LazyFrame:
    var joined = left.lazy().join(right.lazy(), keys, how=how)
    if how == "semi" or how == "anti":
        return joined.select_exprs(
            [
                col("lp").len().alias("left_rows"),
                col("lp").count().alias("left_valid"),
            ]
        )
    return joined.select_exprs(
        [
            col("lp").len().alias("left_rows"),
            col("lp").count().alias("left_valid"),
            col("rp").len().alias("right_rows"),
            col("rp").count().alias("right_valid"),
        ]
    )


def _assert_counts(
    result: DataFrame,
    pairs: List[Tuple[Int, Int]],
    left_valid: List[Bool],
    right_valid: List[Bool],
    semi: Bool,
) raises:
    var lcount = Int64(0)
    var rcount = Int64(0)
    for pair in pairs:
        if pair[0] >= 0 and left_valid[pair[0]]:
            lcount += 1
        if pair[1] >= 0 and right_valid[pair[1]]:
            rcount += 1
    var names: List[String] = ["left_rows", "left_valid"]
    var values: List[Int64] = [Int64(len(pairs)), lcount]
    if not semi:
        names.extend(["right_rows", "right_valid"])
        values.extend([Int64(len(pairs)), rcount])
    assert_equal(result.columns(), names)
    assert_equal(result.height(), 1)
    for index in range(len(names)):
        assert_equal(result.column(names[index]).dtype(), DataType.INT64)
        assert_true(not result.item(0, names[index]).is_null())
        assert_equal(result.item(0, names[index]).int64(), values[index])


def _assert_input_projection(
    query: LazyFrame, keys: List[String], payloads: Bool, semi: Bool = False
) raises:
    var optimized = query._optimized()
    var scans = 0
    for node in optimized._plan._nodes:
        if node.kind != SCAN_FRAME:
            continue
        scans += 1
        for key in keys:
            assert_true(key in node.names)
        assert_true(
            "unused_left" not in node.names and "unused_right" not in node.names
        )
        var needs_payload = payloads and (
            not semi or "lp" in optimized._plan._frames[node.offset].columns()
        )
        assert_equal(len(node.names), len(keys) + Int(needs_payload))
    assert_equal(scans, 2)


def test_projected_join_len_count_follow_duplicate_products_payload_validity_and_join_kind() raises:
    var literal_left: List[Int] = [0, 0, 1, -1]
    var literal_right: List[Int] = [0, 0, 0, 2, -1]
    assert_equal(
        len(nested_loop_pairs(literal_left, literal_right, "inner")), 6
    )
    assert_equal(
        len(nested_loop_pairs(literal_left, literal_right, "full")), 10
    )
    for shape in range(3):
        for reverse in [False, True]:
            var left = _string_side(173, True)
            var right = _string_side(211, False)
            var keys: List[String] = ["k"]
            if shape == 1:
                left = _compound_side(True)
                right = _compound_side(False)
                keys = ["f", "tag"]
            elif shape == 2:
                right = _string_side(0, False)
            if reverse:
                var old_left = left^
                left = right^
                right = old_left^
                left[0] = left[0].rename({"rp": "lp"})
                right[0] = right[0].rename({"lp": "rp"})
            for how in ["inner", "left", "right", "full", "semi", "anti"]:
                var pairs = nested_loop_pairs(left[1], right[1], how)
                if how == "inner":
                    # Independent duplicate-frequency product identity.
                    var products = Int64(0)
                    for key in range(6):
                        var a = Int64(0)
                        var b = Int64(0)
                        for rank in left[1]:
                            a += Int64(rank == key)
                        for rank in right[1]:
                            b += Int64(rank == key)
                        products += a * b
                    assert_equal(products, Int64(len(pairs)))
                var query = _count_query(left[0], right[0], keys, how)
                # Current code materializes join rows. This is a projection
                # contract: no count-only join shortcut is presently defined.
                _assert_input_projection(
                    query, keys, True, how == "semi" or how == "anti"
                )
                for optimize in [False, True]:
                    _assert_counts(
                        query.collect(
                            optimize=optimize, streaming=False, engine="cpu"
                        ),
                        pairs,
                        left[2],
                        right[2],
                        how == "semi" or how == "anti",
                    )
                var length = (
                    left[0]
                    .lazy()
                    .join(right[0].lazy(), keys, how=how)
                    .select(col(keys[0]).len().alias("rows"))
                )
                _assert_input_projection(length, keys, False)
                for optimize in [False, True]:
                    var result = length.collect(
                        optimize=optimize, streaming=False, engine="cpu"
                    )
                    assert_equal(result.columns(), ["rows"])
                    assert_equal(result.column("rows").dtype(), DataType.INT64)
                    assert_equal(result.height(), 1)
                    assert_equal(result.item().int64(), Int64(len(pairs)))

    var collision_left = DataFrame(
        [
            Series("k", Column[Int64]([1])),
            Series("v", Column[Int64]([2])),
            Series("v_right", Column[Int64]([3])),
        ]
    )
    var collision_right = DataFrame(
        [
            Series("k", Column[Int64]([1])),
            Series("v", Column[Int64]([4])),
        ]
    )
    var valid_left = _string_side(1, True)
    var valid_right = _string_side(1, False)
    var wrong_key = DataFrame(
        [
            Series("k", Column[Int64]([1])),
            Series("rp", Column[Int64]([1])),
        ]
    )
    for optimize in [False, True]:
        with assert_raises(contains="collision"):
            _ = (
                collision_left.lazy()
                .join(collision_right.lazy(), "k")
                .select(["k"])
                .collect(optimize=optimize, streaming=False, engine="cpu")
            )
        with assert_raises(contains="Join how must be"):
            _ = (
                valid_left[0]
                .lazy()
                .join(valid_right[0].lazy(), "k", how="unrecognized")
                .select(col("k").len())
                .collect(optimize=optimize, streaming=False, engine="cpu")
            )
        with assert_raises(contains="missing"):
            _ = (
                valid_left[0]
                .lazy()
                .join(valid_right[0].lazy(), "k")
                .select(col("missing").len())
                .collect(optimize=optimize, streaming=False, engine="cpu")
            )
        with assert_raises(contains="Join key dtypes differ"):
            _ = (
                valid_left[0]
                .lazy()
                .join(wrong_key.lazy(), "k")
                .select(col("k").len())
                .collect(optimize=optimize, streaming=False, engine="cpu")
            )


def _csv_projection(query: LazyFrame, count_only: Bool) raises:
    var optimized = query._optimized()
    var left_seen = False
    var right_seen = False
    for node in optimized._plan._nodes:
        if node.kind != SCAN_CSV:
            continue
        assert_true("unused" not in node.names and "other" not in node.names)
        if "gate" in node.names:
            left_seen = True
            var wanted: List[String] = ["k", "gate"]
            if not count_only:
                wanted.append("p")
            assert_equal(node.names, wanted)
        else:
            right_seen = True
            var wanted: List[String] = ["k"]
            if not count_only:
                wanted.extend(["p", "r"])
            assert_equal(node.names, wanted)
    assert_true(left_seen and right_seen)


def test_csv_join_projection_keeps_filter_and_collision_dependencies_and_exact_cells() raises:
    var directory = getenv("COVERAGE_TEST_TMP")
    assert_true(Bool(directory))
    var left_path = directory + "/planner-projection-left.csv"
    var right_path = directory + "/planner-projection-right.csv"
    # Independently authored text, including quoted nonselected commas/UTF8.
    with open(left_path, "w") as file:
        file.write(
            'k,gate,p,unused\n2,1,10,"x,π"\n2,0,,ignored\n3,-1,30,drop\n4,2,40,text\n,3,50,null-key\n8,1,80,absent\n'
        )
    with open(right_path, "w") as file:
        file.write(
            'k,p,r,other\n2,101,0,α\n2,,1,"comma,value"\n4,104,2,text\n5,105,3,unmatched\n,999,4,null-key\n'
        )
    var left = scan_csv(
        left_path,
        CsvSchema(
            [
                CsvField.int64("k"),
                CsvField.int64("gate"),
                CsvField.int64("p"),
                CsvField.string("unused"),
            ]
        ),
    ).filter(col("gate") >= lit(Int64(0)))
    var right = scan_csv(
        right_path,
        CsvSchema(
            [
                CsvField.int64("k"),
                CsvField.int64("p"),
                CsvField.int64("r"),
                CsvField.string("other"),
            ]
        ),
    )
    var left_ranks: List[Int] = [2, 2, 4, -1, 8]
    var right_ranks: List[Int] = [2, 2, 4, 5, -1]
    var lp: List[Int64] = [10, -999, 40, 50, 80]
    var lv: List[Bool] = [True, False, True, True, True]
    var rp: List[Int64] = [101, -999, 104, 105, 999]
    var rv: List[Bool] = [True, False, True, True, True]
    var pairs = nested_loop_pairs(left_ranks, right_ranks, "inner")
    assert_equal(pairs, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)])
    var expected_lp = List[Int64]()
    var expected_rp = List[Int64]()
    var expected_rid = List[Int64]()
    var expected_lv = List[Bool]()
    var expected_rv = List[Bool]()
    for pair in pairs:
        expected_lp.append(lp[pair[0]])
        expected_rp.append(rp[pair[1]])
        expected_rid.append(Int64(pair[1]))
        expected_lv.append(lv[pair[0]])
        expected_rv.append(rv[pair[1]])
    var expected = DataFrame(
        [
            Series("left_p", Column[Int64](expected_lp^, expected_lv)),
            Series("right_p", Column[Int64](expected_rp^, expected_rv)),
            Series("r", Column[Int64](expected_rid^)),
        ]
    )
    var joined = left.join(right, "k")
    var cells = joined.select_exprs(
        [
            col("p").alias("left_p"),
            col("p_right").alias("right_p"),
            col("r"),
        ]
    )
    _csv_projection(cells, False)
    var right_cells = joined.select_exprs(
        [col("p_right").alias("right_p"), col("r")]
    )
    _csv_projection(right_cells, False)
    var right_expected = DataFrame(
        [expected.column("right_p").copy(), expected.column("r").copy()]
    )
    var count = joined.select(col("k").len().alias("rows"))
    _csv_projection(count, True)
    for optimize in [False, True]:
        assert_frame(
            cells.collect(optimize=optimize, streaming=False, engine="cpu"),
            expected,
        )
        assert_frame(
            right_cells.collect(
                optimize=optimize, streaming=False, engine="cpu"
            ),
            right_expected,
        )
        var got = count.collect(
            optimize=optimize, streaming=False, engine="cpu"
        )
        assert_equal(got.columns(), ["rows"])
        assert_equal(got.column("rows").dtype(), DataType.INT64)
        assert_equal(got.height(), 1)
        assert_equal(got.item().int64(), Int64(len(pairs)))
