"""Cross-batch equivalence and bounded-executor lifecycle regression tests."""
from std.ffi import external_call
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
)
from dataframe import (
    Expr,
    Column,
    DataFrame,
    DataType,
    Series,
    CsvSchema,
    CsvField,
    col,
    lit,
    scan_csv,
    scan_parquet,
    read_parquet,
    write_csv,
)

comptime PATH = "/tmp/dataframe_lazy_streaming_test.csv"


def sample() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "k",
                Column[String](
                    ["b", "a", "b", "c", "a", "d", "b"],
                    [True, True, True, False, True, True, True],
                ),
            ),
            Series(
                "v",
                Column[Int64](
                    [1, 2, 3, 4, 5, 6, 7],
                    [True, False, True, True, True, True, True],
                ),
            ),
            Series("f", Column[Float64]([1, 2, 3, 4, 5, 6, 7])),
        ]
    )


def test_ordered_transforms_head_and_breakers() raises:
    var source = sample()
    for size in [1, 2, 3, 8]:
        var plan = (
            source.lazy()
            .filter(col("v") > 1)
            .with_columns((col("v") * 2).alias("twice"))
            .select(["k", "twice"])
        )
        assert_true(
            plan.collect(batch_size=size).equals(plan.collect(streaming=False))
        )
        assert_true(
            plan.slice(1, 3)
            .collect(batch_size=size)
            .equals(plan.slice(1, 3).collect(streaming=False))
        )
        assert_true(
            plan.sort("twice", descending=True)
            .collect(batch_size=size)
            .equals(
                plan.sort("twice", descending=True).collect(streaming=False)
            )
        )
        assert_true(
            source.lazy()
            .select(lit(Int64(42)).alias("constant"))
            .collect(batch_size=size)
            .equals(source.select(lit(Int64(42)).alias("constant")))
        )
    assert_true("[stream]" in source.lazy().filter(col("v") > 1).explain())
    assert_true("[materialize]" in source.lazy().sort("v").explain())
    with assert_raises(contains="positive"):
        _ = source.lazy().collect(batch_size=0)


def test_state_merges_nulls_counts_moments_and_order() raises:
    var source = sample()
    var expressions: List[Expr] = [
        col("v").sum().alias("sum"),
        col("v").count().alias("count"),
        col("v").mean().alias("mean"),
        col("v").first().alias("first"),
        col("v").last().alias("last"),
        col("v").min().alias("min"),
        col("v").max().alias("max"),
        col("v").n_unique().alias("distinct"),
        col("v").len().alias("rows"),
        (col("v").sum() + col("v").count()).alias("combined"),
    ]
    var plan = source.lazy().group_by("k", maintain_order=True).agg(expressions)
    for size in [1, 2, 3, 8]:
        assert_true(
            plan.collect(batch_size=size).equals(plan.collect(streaming=False))
        )
    var global_plan = source.lazy().select_exprs(
        [col("v").sum(), col("v").count().alias("count")]
    )
    assert_true(
        global_plan.collect(batch_size=2).equals(
            global_plan.collect(streaming=False)
        )
    )
    var empty = source.clear().lazy().group_by("k").agg(expressions)
    assert_true(
        empty.collect(batch_size=2).equals(empty.collect(streaming=False))
    )


def test_partial_integer_overflow_is_not_finalized() raises:
    var source = DataFrame(
        [Series("v", Column[Int64]([9223372036854775807, 1, -1]))]
    )
    assert_equal(
        source.lazy()
        .select(col("v").sum())
        .collect(batch_size=2)
        .item()
        .int64(),
        9223372036854775807,
    )
    var bad = DataFrame([Series("v", Column[Int64]([9223372036854775807, 1]))])
    with assert_raises(contains="overflow"):
        _ = bad.lazy().select(col("v").sum()).collect(batch_size=1)


def test_csv_quotes_eof_and_late_error() raises:
    with open(PATH, "w") as file:
        file.write('k,v\n"a\nb",1\na,2\n"a\nb",3\nc,4')
    var plan = (
        scan_csv(PATH)
        .filter(col("v") > 1)
        .group_by("k", maintain_order=True)
        .agg(col("v").sum())
    )
    for size in [1, 2, 3]:
        assert_true(
            plan.collect(batch_size=size).equals(plan.collect(streaming=False))
        )
    with open(PATH, "w") as file:
        file.write("v\n1\n2\n3\ninvalid\n")
    var schema = CsvSchema([CsvField.int64("v")])
    with assert_raises():
        _ = scan_csv(PATH, schema).select(col("v").sum()).collect(batch_size=1)
    with open(PATH, "w") as file:
        file.write("v\n")
    assert_equal(
        scan_csv(PATH, schema)
        .select(col("v").sum())
        .collect(batch_size=1)
        .item()
        .int64(),
        0,
    )


def test_randomized_state_and_join_differential() raises:
    for seed in range(6):
        var state = UInt64(seed + 1)
        var keys = List[Int64]()
        var values = List[Int64]()
        var floats = List[Float64]()
        var flags = List[Bool]()
        var valid = List[Bool]()
        for i in range(137 + seed):
            state = state * 6364136223846793005 + 1
            var value = Int64((state >> 32) % 101) - 50
            keys.append(Int64(i % (seed + 3)))
            values.append(value)
            floats.append(Float64(value) / 4)
            flags.append(value > 0)
            valid.append(i % 7 != 0)
        var source = DataFrame(
            [
                Series("k", Column[Int64](keys^, valid)),
                Series("v", Column[Int64](values^, valid)),
                Series("f", Column[Float64](floats^, valid)),
                Series("b", Column[Bool](flags^, valid)),
            ]
        )
        var query = (
            source.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("v").sum(min_count=3).alias("sum"),
                    col("f").var().alias("var"),
                    col("f").std().alias("std"),
                    col("f").mean().alias("mean"),
                    col("v").null_count().alias("nulls"),
                    col("b").any(ignore_nulls=False).alias("any"),
                    col("b").all(ignore_nulls=False).alias("all"),
                    (col("v").sum() + col("v").count()).alias("combined"),
                ]
            )
        )
        var expected = query.collect(streaming=False)
        for size in [1, 17, 64]:
            var actual = query.collect(batch_size=size)
            assert_equal(actual.columns(), expected.columns())
            assert_equal(actual.height(), expected.height())
            for c in range(actual.width()):
                if actual._columns[c].dtype() == DataType.FLOAT64:
                    for r in range(actual.height()):
                        var left = actual._columns[c].get(r)
                        var right = expected._columns[c].get(r)
                        assert_equal(left.is_null(), right.is_null())
                        if not left.is_null():
                            assert_almost_equal(
                                left.float64(),
                                right.float64(),
                                rtol=1e-10,
                                atol=1e-10,
                            )
                else:
                    assert_true(actual._columns[c].equals(expected._columns[c]))
        var lookup = DataFrame(
            [
                Series("k", Column[Int64]([0, 1, 1, 9])),
                Series("label", Column[String](["a", "b", "c", "d"])),
            ]
        )
        for how in ["inner", "left", "semi", "anti", "right", "full"]:
            var joined = source.lazy().join(lookup.lazy(), "k", how=how)
            assert_true(
                joined.collect(batch_size=17).equals(
                    joined.collect(streaming=False)
                )
            )


def set_env(name: String, value: String):
    var key = List[UInt8](name.as_bytes())
    key.append(0)
    var text = List[UInt8](value.as_bytes())
    text.append(0)
    _ = external_call["setenv", Int32](
        Int(key.unsafe_ptr()), Int(text.unsafe_ptr()), Int32(1)
    )
    _ = key^
    _ = text^


def unset_env(name: String):
    var key = List[UInt8](name.as_bytes())
    key.append(0)
    _ = external_call["unsetenv", Int32](Int(key.unsafe_ptr()))
    _ = key^


def test_deferred_merges_keep_order_and_state() raises:
    """Streaming group-by state (#326): batch states merged in groups, split
    into hash parts (forced on with DATAFRAME_STREAM_SPLIT_GROUPS=1) with a
    persistent key index, and the index's fallback when string keys grow
    past the word encoding mid-stream. Every case must equal the
    non-streaming executor, including first-occurrence group order."""
    var nan = Float64(0) / Float64(0)
    # Four workers: four hash parts when the split is forced on.
    set_env("DATAFRAME_THREADS", "4")
    for shape in range(2):
        var state = UInt64(shape + 11)
        var ints = List[Int64]()
        var words = List[String]()
        var floats = List[Float64]()
        var values = List[Int64]()
        var valid = List[Bool]()
        var rows = 120
        for i in range(rows):
            state = state * 6364136223846793005 + 1
            var r = Int64((state >> 33) % 1_000_000)
            # 0: every row a new group; 1: groups that keep arriving late.
            var key = Int64(i) if shape == 0 else r % (Int64(i) // 10 + 1)
            ints.append(key)
            # Past row 60 some strings exceed the 24-byte word encoding.
            var word = "w" + String(key % 7)
            if i > 60 and key % 3 == 0:
                word += "-a-string-longer-than-24-bytes"
            words.append(word)
            var f = Float64(key % 5) - 2
            if key % 5 == 4:
                f = nan
            elif key % 5 == 2:
                f = -0.0
            floats.append(f)
            values.append(r % 1000 - 500)
            valid.append(i % 11 != 0)
        var source = DataFrame(
            [
                Series("k", Column[Int64](ints^, valid)),
                Series("s", Column[String](words^)),
                Series("x", Column[Float64](floats^, valid)),
                Series("v", Column[Int64](values^, valid)),
            ]
        )
        for keyset in range(3):
            var keys: List[String] = ["k"]
            if keyset == 1:
                keys = ["s", "k"]
            elif keyset == 2:
                keys = ["x"]
            var query = (
                source.lazy()
                .group_by(keys, maintain_order=True)
                .agg(
                    [
                        col("v").sum().alias("sum"),
                        col("v").first().alias("first"),
                        col("v").last().alias("last"),
                        col("v").arg_min().alias("arg_min"),
                        col("v").n_unique().alias("distinct"),
                        col("v").len().alias("rows"),
                    ]
                )
            )
            var expected = query.collect(streaming=False)
            for split in ["4096", "1"]:
                set_env("DATAFRAME_STREAM_SPLIT_GROUPS", split)
                assert_true(query.collect(batch_size=8).equals(expected))
    unset_env("DATAFRAME_STREAM_SPLIT_GROUPS")
    unset_env("DATAFRAME_THREADS")


def test_decimal_states_grow_across_batches() raises:
    """Decimal sum/min/max state per group survives streaming merges that add
    groups, and decimal keys fall back from the key index (#326)."""
    var dtype = DataType.decimal(8, 2)
    var raw = List[Int128]()
    var groups = List[Int64]()
    for i in range(60):
        raw.append(Int128((i * 37) % 1000 - 300))
        groups.append(Int64(i // 3))
    var source = DataFrame(
        [
            Series("g", Column[Int64](groups^)),
            Series("d", Column[Int128](raw^)).with_dtype(dtype),
        ]
    )
    for by_decimal in [False, True]:
        var key_names: List[String] = ["d"] if by_decimal else ["g"]
        var query = (
            source.lazy()
            .group_by(key_names, maintain_order=True)
            .agg(
                [
                    col("d").sum().alias("sum"),
                    col("d").min().alias("min"),
                    col("d").max().alias("max"),
                ]
            )
        )
        assert_true(
            query.collect(batch_size=4).equals(query.collect(streaming=False))
        )


def test_schema_probe_does_not_decode_data() raises:
    with open(PATH, "w") as file:
        file.write("v\ninvalid\n")
    var schema = CsvSchema([CsvField.int64("v")])
    var query = scan_csv(PATH, schema).select(col("v").sum())
    assert_equal(query.collect_schema(), [String("v: int64")])
    with assert_raises():
        _ = query.collect(batch_size=1)


def test_owned_strings_survive_consumed_page_unmapping() raises:
    var texts = List[String]()
    var values = List[Int64]()
    for i in range(1200):
        texts.append(
            "é,"
            + String(i)
            + (String("x") * 5000 if i % 17 == 0 else "line\nvalue")
        )
        values.append(Int64(i))
    var source = DataFrame(
        [
            Series("k", Column[String](texts^)),
            Series("v", Column[Int64](values^)),
        ]
    )
    write_csv(source, PATH)
    var actual = scan_csv(PATH).select(["k", "v"]).collect(batch_size=17)
    assert_true(actual.equals(source))
    assert_equal(scan_csv(PATH).head(19).collect(batch_size=7).height(), 19)


def test_prepared_compound_joins_and_chained_batches() raises:
    var left = DataFrame(
        [
            Series("k", Column[String](["a", "b", "a", "c", "a", "b"])),
            Series(
                "n",
                Column[Int64](
                    [1, 2, 1, 3, 0, 2], [True, True, True, True, False, True]
                ),
            ),
            Series("v", Column[Int64]([0, 1, 2, 3, 4, 5])),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[String](["a", "b", "a", "a"])),
            Series("n", Column[Int64]([1, 2, 1, 0], [True, True, True, False])),
            Series("w", Column[Int64]([10, 20, 30, 40])),
        ]
    )
    for how in ["inner", "left", "semi", "anti"]:
        var plan = left.lazy().join(right.lazy(), ["k", "n"], how=how)
        var expected = left.join(right, ["k", "n"], how=how)
        for size in [1, 2, 4, 20]:
            assert_true(plan.collect(batch_size=size).equals(expected))
            assert_true(
                plan.head(2).collect(batch_size=size).equals(expected.head(2))
            )
            var twice = plan.join(right.lazy(), ["k", "n"], how="semi")
            assert_true(
                twice.collect(batch_size=size).equals(
                    expected.join(right, ["k", "n"], how="semi")
                )
            )
    var empty = right.head(0)
    for how in ["inner", "left", "semi", "anti"]:
        assert_true(
            left.lazy()
            .join(empty.lazy(), ["k", "n"], how=how)
            .collect(batch_size=2)
            .equals(left.join(empty, ["k", "n"], how=how))
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
