"""Composite row keys and multi-column, typed, and expression group_by keys."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import (
    DataType,
    AnyValue,
    Column,
    DataFrame,
    Expr,
    Series,
    StringColumn,
    col,
    lit,
)
from dataframe.hashing import encode_rows
from dataframe.string_view import StringViewBuilder


def nan() -> Float64:
    return Float64(0) / Float64(0)


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def random_frame(rows: Int, seed: UInt64) raises -> DataFrame:
    var rng = Lcg(seed)
    var i = List[Int64]()
    var iv = List[Bool]()
    var f = List[Float64]()
    var fv = List[Bool]()
    var s = List[String]()
    var sv = List[Bool]()
    var b = List[Bool]()
    var bv = List[Bool]()
    var v = List[Int64]()
    var words: List[String] = ["", "a", "b", "é"]
    for _ in range(rows):
        i.append(Int64(rng.next(3)))
        iv.append(rng.next(5) != 0)
        var pick = rng.next(4)
        f.append(
            nan() if pick
            == 0 else (-0.0 if pick == 1 else (0.0 if pick == 2 else 1.5))
        )
        fv.append(rng.next(6) != 0)
        s.append(words[rng.next(len(words))])
        sv.append(rng.next(5) != 0)
        b.append(rng.next(2) == 0)
        bv.append(rng.next(4) != 0)
        v.append(Int64(rng.next(100)))
    return DataFrame(
        [
            Series("i", Column[Int64](i^, iv^)),
            Series("f", Column[Float64](f^, fv^)),
            Series("s", Column[String](s^, sv^)),
            Series("b", Column[Bool](b^, bv^)),
            Series("v", Column[Int64](v^)),
        ]
    )


def reference_groups(
    frame: DataFrame, keys: List[String]
) raises -> Tuple[List[List[AnyValue]], List[Int64], List[Int64]]:
    """Naive first-occurrence grouping by structural cell equality."""
    var groups = List[List[AnyValue]]()
    var sums = List[Int64]()
    var counts = List[Int64]()
    for row in range(frame.height()):
        var key = List[AnyValue]()
        for name in keys:
            key.append(frame.item(row, name))
        var found = -1
        for g in range(len(groups)):
            var same = True
            for k in range(len(key)):
                if not (groups[g][k] == key[k]):
                    same = False
                    break
            if same:
                found = g
                break
        if found < 0:
            found = len(groups)
            groups.append(key^)
            sums.append(0)
            counts.append(0)
        sums[found] += frame.item(row, "v").int64()
        counts[found] += 1
    return (groups^, sums^, counts^)


def test_encode_rows_policies() raises:
    var keys: List[Series] = [
        Series(
            "f",
            Column[Float64](
                [nan(), -nan(), 0.0, -0.0, 1, 7],
                [True, True, True, True, True, False],
            ),
        ),
        Series("s", Column[String](["x", "x", "y", "y", "x", "z"])),
    ]
    var equal = encode_rows(keys, nulls_equal=True)
    assert_equal(equal.ids, [0, 0, 1, 1, 2, 3])
    assert_equal(equal.representatives, [0, 2, 4, 5])
    var strict = encode_rows(keys, nulls_equal=False)
    assert_equal(strict.ids, [0, 0, 1, 1, 2, -1])
    assert_equal(strict.count(), 3)
    # Nulls in different columns stay distinct from each other and from values.
    var pairs: List[Series] = [
        Series("a", Column[Int64]([1, 1, 0, 0], [True, False, False, True])),
        Series(
            "b",
            Column[Bool]([True, False, True, False], [False, True, True, True]),
        ),
    ]
    assert_equal(encode_rows(pairs, nulls_equal=True).ids, [0, 1, 2, 3])
    with assert_raises(contains="at least one column"):
        _ = encode_rows(List[Series](), True)
    with assert_raises(contains="equal lengths"):
        _ = encode_rows(
            [
                Series("a", Column[Int64]([1])),
                Series("b", Column[Int64]([1, 2])),
            ],
            True,
        )
    assert_equal(
        encode_rows([Series("e", Column[String]([]))], True).count(), 0
    )


def test_multi_key_groups_match_reference() raises:
    var key_sets: List[List[String]] = [
        ["i"],
        ["f"],
        ["b"],
        ["s", "i"],
        ["b", "f"],
        ["i", "s", "b"],
        ["f", "s", "b", "i"],
    ]
    for seed in range(1, 4):
        var frame = random_frame(120, UInt64(seed))
        for keys in key_sets:
            var reference = reference_groups(frame, keys)
            var result = frame.group_by(keys, maintain_order=True).agg(
                [col("v").sum().alias("total"), col("v").count().alias("n")]
            )
            assert_equal(result.columns()[0], keys[0])
            assert_equal(result.height(), len(reference[0]))
            for g in range(result.height()):
                for k in range(len(keys)):
                    assert_true(result.item(g, keys[k]) == reference[0][g][k])
                assert_equal(result.item(g, "total").int64(), reference[1][g])
                assert_equal(result.item(g, "n").int64(), reference[2][g])
            var counted = frame.group_by(keys, maintain_order=True).len()
            for g in range(counted.height()):
                assert_equal(counted.item(g, "len").int64(), reference[2][g])


def test_unordered_output_has_the_same_groups() raises:
    var frame = random_frame(80, 9)
    var ordered = frame.group_by(["s", "b"], maintain_order=True).agg(
        col("v").sum()
    )
    var unordered = frame.group_by(["s", "b"]).agg(col("v").sum())
    assert_equal(ordered.height(), unordered.height())
    var sorted_a = ordered.sort(["s", "b", "v"])
    var sorted_b = unordered.sort(["s", "b", "v"])
    assert_true(sorted_a.equals(sorted_b))


def test_expression_keys() raises:
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([1, 12, 15, 3, 27, 22])),
            Series("k", Column[String](["a", "a", "b", "a", "b", "b"])),
        ]
    )
    var result = frame.group_by(
        [(col("x") // lit(Int64(10))).alias("tens"), col("k")],
        maintain_order=True,
    ).agg([col("x").sum().alias("total"), col("x").max().alias("top")])
    assert_equal(result.columns(), [String("tens"), "k", "total", "top"])
    assert_equal(result.height(), 4)
    assert_equal(result.item(0, "tens").int64(), Int64(0))
    assert_equal(result.item(0, "total").int64(), Int64(4))
    assert_equal(result.item(1, "k").string(), "a")
    assert_equal(result.item(1, "total").int64(), Int64(12))
    # Aggregations see original columns even when a key reuses a column name.
    var parity = frame.group_by(
        [(col("x") % lit(Int64(2))).alias("x")], maintain_order=True
    ).agg(col("k").count().alias("rows"))
    assert_equal(parity.item(0, "x").int64(), Int64(1))
    assert_equal(parity.item(0, "rows").int64(), Int64(4))
    var constant = frame.group_by([lit(Int64(1)).alias("one")]).agg(
        col("x").sum()
    )
    assert_equal(constant.height(), 1)
    assert_equal(constant.item(0, "x").int64(), Int64(80))
    with assert_raises(contains="must not be aggregates"):
        _ = frame.group_by([col("x").sum()])
    with assert_raises(contains="Duplicate expression output name"):
        _ = frame.group_by([col("x"), col("k").alias("x")])


def test_validation_and_empty_input() raises:
    var frame = random_frame(10, 3)
    with assert_raises(contains="at least one key"):
        _ = frame.group_by(List[String]())
    with assert_raises(contains="Duplicate group_by key"):
        _ = frame.group_by(["i", "i"])
    with assert_raises(contains="collides with grouping key: s"):
        _ = frame.group_by(["i", "s"]).agg(col("v").sum().alias("s"))
    with assert_raises(contains="collides with grouping key: i"):
        _ = frame.group_by(["i"]).len("i")
    var empty = (
        frame.clear()
        .group_by(["f", "s"])
        .agg([col("v").sum(), col("b").any().alias("any")])
    )
    assert_equal(empty.height(), 0)
    assert_equal(empty.columns(), [String("f"), "s", "v", "any"])
    assert_equal(
        empty.dtypes(),
        [DataType.FLOAT64, DataType.STRING, DataType.INT64, DataType.BOOL],
    )
    var keys_only = frame.group_by(["b"], maintain_order=True).agg(List[Expr]())
    assert_equal(keys_only.width(), 1)


def test_many_groups() raises:
    var values = List[Int64]()
    for i in range(20000):
        values.append(Int64(i % 7919))
    var frame = DataFrame([Series("k", Column[Int64](values^))])
    var result = frame.group_by("k", maintain_order=True).len()
    assert_equal(result.height(), 7919)
    assert_equal(result.item(0, "len").int64(), Int64(3))
    assert_equal(result.item(7918, "len").int64(), Int64(2))
    assert_equal(result.item(7918, "k").int64(), Int64(7918))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_single_string_key_across_view_and_legacy_chunks() raises:
    var first = StringViewBuilder()
    first.append_null()
    first.append(StringSlice("short"))
    first.append(StringSlice("thirteen-bytes"))
    first.append(StringSlice("short"))
    var second = StringViewBuilder()
    for i in range(300):
        var value = String(i)
        second.append(StringSlice(value))
    second.append(StringSlice("short"))
    second.append(StringSlice("thirteen-bytes"))
    second.append_null()
    var key = Series._from_chunks(
        [
            Series("s", StringColumn(first^.finish())),
            Series("s", StringColumn(second^.finish())),
            Series(
                "s",
                StringColumn(
                    ["short", "thirteen-bytes", "new", ""],
                    [True, True, True, False],
                ),
            ),
        ]
    )
    var equal = encode_rows([key.copy()], nulls_equal=True)
    assert_equal(equal.count(), 304)
    assert_equal(equal.ids[0], 0)
    assert_equal(equal.ids[1], 1)
    assert_equal(equal.ids[2], 2)
    assert_equal(equal.ids[3], 1)
    assert_equal(equal.ids[4], 3)
    assert_equal(equal.ids[303], 302)
    assert_equal(equal.ids[304], 1)
    assert_equal(equal.ids[305], 2)
    assert_equal(equal.ids[306], 0)
    assert_equal(equal.ids[307], 1)
    assert_equal(equal.ids[308], 2)
    assert_equal(equal.ids[309], 303)
    assert_equal(equal.ids[310], 0)
    assert_equal(equal.representatives[0], 0)
    assert_equal(equal.representatives[303], 309)
    var strict = encode_rows([key.copy()], nulls_equal=False)
    assert_equal(strict.count(), 303)
    assert_equal(strict.ids[0], -1)
    assert_equal(strict.ids[1], 0)
    assert_equal(strict.ids[2], 1)
    assert_equal(strict.ids[306], -1)
    assert_equal(strict.ids[309], 302)
    assert_equal(strict.ids[310], -1)
