"""A join build no larger than the rows it filters hands the pipeline a
bloom filter over its keys;
a probe row whose key the filter lacks is dropped before any step reads it. Joins against a scalar model: inner and semi joins,
Int64 and Int32 keys, nullable probe keys, a second join after the first so
the filter works on rows an earlier filter kept, a probe filter before the
joins (on another column, and on the filtered key itself, whose nulls leave
the filter's mask with a validity bitmap), and builds too big for a filter,
on one and eight workers. The filter itself never loses a key. A perfect
hash build hands the probe its exact bitmap of keys instead, which also
rules out whole vectors by their value ranges."""
# contract-group: table.join
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    col,
)
from dataframe.kernels.runtime_filter import (
    KeyBloom,
    bloom_blocks_for,
)
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 400_000
comptime SPAN = 10_000_000


def spread(row: Int, salt: Int) -> Int:
    """A key scattered over the whole span."""
    return Int((UInt64(row) * 2654435761 + UInt64(salt) * 40503) % UInt64(SPAN))


def key_valid(row: Int) -> Bool:
    return row % 97 != 5


def probe(dtype: DataType) raises -> DataFrame:
    var ids = List[Int64]()
    var a = List[Int64]()
    var b = List[Int64]()
    var valid = List[Bool]()
    for row in range(ROWS):
        ids.append(Int64(row))
        # Few distinct keys, so a small build covers a share of them.
        a.append(Int64(spread(row % 5000, 1)))
        b.append(Int64(spread(row % 7000, 2)))
        valid.append(key_valid(row))
    var frame = DataFrame(
        [
            Series("id", Column[Int64](ids^)),
            Series("ak", Column[Int64](a^, valid.copy())),
            Series("bk", Column[Int64](b^, valid)),
        ]
    )
    if dtype != DataType.INT64:
        frame = frame.with_columns(
            [
                col("ak").cast(dtype).alias("ak"),
                col("bk").cast(dtype).alias("bk"),
            ]
        )
    return frame^


def build(
    name: String, modulus: Int, salt: Int, every: Int, dtype: DataType
) raises -> DataFrame:
    """The keys of probe values `spread(m, salt)` for each `every`-th m
    below `modulus`, with a payload."""
    var keys = List[Int64]()
    var payload = List[Int64]()
    for m in range(0, modulus, every):
        keys.append(Int64(spread(m, salt)))
        payload.append(Int64(m))
    var frame = DataFrame(
        [
            Series(name, Column[Int64](keys^)),
            Series(name + "_v", Column[Int64](payload^)),
        ]
    )
    if dtype != DataType.INT64:
        frame = frame.with_columns([col(name).cast(dtype).alias(name)])
    return frame^


def ids(frame: DataFrame) raises -> List[Int]:
    var sorted = frame.sort(["id"]).column("id")
    var out = List[Int](capacity=len(sorted))
    for r in range(len(sorted)):
        out.append(Int(sorted.get(r).int64()))
    return out^


def check(
    dtype: DataType,
    a_every: Int,
    b_every: Int,
    semi: Bool,
    expect_bloom: Bool,
    prefilter: Int = 0,
) raises:
    """`prefilter` 1 keeps rows whose id is not a multiple of 3 before the
    joins; 2 keeps rows whose `bk` is even (null where `bk` is)."""
    var source = probe(dtype).lazy()
    if prefilter == 1:
        source = source.filter(col("id") % 3 != 0)
    elif prefilter == 2:
        source = source.filter(col("bk") % 2 == 0)
    var plan = source.join(
        build("a", 5000, 1, a_every, dtype).lazy(),
        left_on=["ak"],
        right_on=["a"],
    ).join(
        build("b", 7000, 2, b_every, dtype).lazy(),
        left_on=["bk"],
        right_on=["b"],
        how="semi" if semi else "inner",
    )
    var expected = List[Int]()
    for row in range(ROWS):
        if not key_valid(row):
            continue
        if (row % 5000) % a_every != 0:
            continue
        if (row % 7000) % b_every != 0:
            continue
        if prefilter == 1 and row % 3 == 0:
            continue
        if prefilter == 2 and spread(row % 7000, 2) % 2 != 0:
            continue
        expected.append(row)
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var result = plan.collect()
    var took = "dataframe-path: pipeline.join_bloom\n" in observed.text()
    _ = observed^
    _ = trace^
    if expect_bloom:
        assert_true(took, "no bloom filter was built")
    assert_equal(ids(result), expected)


def test_bloom_filters_keep_exactly_the_joining_rows() raises:
    for threads in [1, 8]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        for dtype in [DataType.INT64, DataType.INT32]:
            # Builds of 50 and 70 keys against 400,000 rows.
            check(dtype, 100, 100, False, True)
        check(DataType.INT64, 100, 100, True, True)
        # Joins after a probe filter: one that reads another column, and
        # one that reads the filtered key itself.
        for prefilter in [1, 2]:
            check(DataType.INT64, 100, 100, False, True, prefilter)
            check(DataType.INT32, 100, 100, True, True, prefilter)
        # A build whose keys are most of the probe's: no filter pays.
        check(DataType.INT64, 1, 1, False, False)
        _ = workers^


def test_a_bloom_filter_never_loses_a_key() raises:
    var blocks = bloom_blocks_for(10_000)
    assert_true(blocks > 0)
    var bloom = KeyBloom(blocks)
    var keys = List[Int64]()
    for k in range(10_000):
        keys.append(Int64(spread(k, 3)))
        bloom.insert(keys[k])
    for k in keys:
        assert_true(bloom.contains(k))
    # Few keys outside it pass: 12 bits a key, four of them set in one
    # word, leaves about 1% false hits.
    var false_hits = 0
    for k in range(10_000):
        if bloom.contains(Int64(SPAN + 1 + k)):
            false_hits += 1
    assert_true(false_hits < 500)


def perfect_probe_key(row: Int) -> Int:
    """Probe keys in [0, 20,000): every third 2,048-row vector holds only
    values between two build keys (none of which joins), the rest are
    spread over the range."""
    var vector = row // 2048
    if vector % 3 == 1:
        return (vector * 50) % 20_000 + 1 + row % 49
    return (row * 7) % 20_000


def test_a_perfect_hash_build_hands_the_probe_its_key_bitmap() raises:
    # Unique build keys over a narrow range join through a perfect hash
    # table, whose bitmap of keys becomes the probe's exact filter: rows in
    # the range whose key no build row has are dropped before any step
    # reads them, and vectors whose value range holds no key are not read.
    for threads in [1, 8]:
        for dtype in [DataType.INT64, DataType.INT32]:
            var workers = ScopedEnvironment(
                "DATAFRAME_THREADS", String(threads)
            )
            var probe_ids = List[Int64]()
            var probe_keys = List[Int64]()
            var valid = List[Bool]()
            for row in range(ROWS):
                probe_ids.append(Int64(row))
                probe_keys.append(Int64(perfect_probe_key(row)))
                valid.append(key_valid(row))
            var source = DataFrame(
                [
                    Series("id", Column[Int64](probe_ids^)),
                    Series("k", Column[Int64](probe_keys^, valid)),
                ]
            ).with_columns([col("k").cast(dtype).alias("k")])
            var keys = List[Int64]()
            var payload = List[Int64]()
            for m in range(0, 20_000, 50):
                keys.append(Int64(m))
                payload.append(Int64(m // 50))
            var dimension = DataFrame(
                [
                    Series("d", Column[Int64](keys^)),
                    Series("d_v", Column[Int64](payload^)),
                ]
            ).with_columns([col("d").cast(dtype).alias("d")])
            var expected = List[Int]()
            for row in range(ROWS):
                if key_valid(row) and perfect_probe_key(row) % 50 == 0:
                    expected.append(row)
            var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
            var observed = StrategyTrace()
            var result = (
                source.lazy()
                .join(dimension.lazy(), left_on=["k"], right_on=["d"])
                .collect()
            )
            var text = observed.text()
            _ = observed^
            _ = trace^
            assert_true(
                "dataframe-path: join.perfect_hash\n" in text,
                "the build did not take a perfect hash table",
            )
            assert_true(
                "dataframe-path: pipeline.join_key_bitmap\n" in text,
                "the probe did not get the build's key bitmap",
            )
            assert_true(
                "dataframe-path: pipeline.join_bloom\n" not in text,
                "a bloom filter was built for a perfect hash table",
            )
            assert_true(
                "dataframe-path: pipeline.join_filter_vector_skip\n" in text,
                "no vector was ruled out by the key bitmap",
            )
            assert_equal(ids(result), expected)
            var sorted = result.sort(["id"]).with_columns(
                [col("k").cast(DataType.INT64).alias("k")]
            )
            var got_keys = sorted.column("k")
            var got_payload = sorted.column("d_v")
            for r in range(len(expected)):
                var key = perfect_probe_key(expected[r])
                assert_equal(Int(got_keys.get(r).int64()), key)
                assert_equal(Int(got_payload.get(r).int64()), key // 50)
            _ = workers^


def test_a_source_filter_runs_among_the_join_filters() raises:
    # The probe's own filter runs in the scan beside the join's key bitmap,
    # cheapest first: whichever runs first, the rows kept are those both
    # keep. A filter keeping few rows (a null test, before the bitmap), one
    # keeping most, and one that can raise (arithmetic, after the bitmap)
    # each give the scalar model's rows.
    for threads in [1, 8]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        var probe_ids = List[Int64]()
        var probe_keys = List[Int64]()
        var marks = List[Int64]()
        var marked = List[Bool]()
        for row in range(ROWS):
            probe_ids.append(Int64(row))
            probe_keys.append(Int64(perfect_probe_key(row)))
            marks.append(Int64(row % 11))
            marked.append(row % 23 != 0)
        var source = DataFrame(
            [
                Series("id", Column[Int64](probe_ids^)),
                Series("k", Column[Int64](probe_keys^)),
                Series("m", Column[Int64](marks^, marked^)),
            ]
        )
        var keys = List[Int64]()
        var payload = List[Int64]()
        for m in range(0, 20_000, 50):
            keys.append(Int64(m))
            payload.append(Int64(m // 50))
        var dimension = DataFrame(
            [
                Series("d", Column[Int64](keys^)),
                Series("d_v", Column[Int64](payload^)),
            ]
        )
        for shape in range(3):
            var filtered = source.lazy()
            if shape == 0:
                filtered = filtered.filter(col("m").is_null())
            elif shape == 1:
                filtered = filtered.filter(col("m") != 4)
            else:
                filtered = filtered.filter((col("id") + 1) % 3 != 0)
            var expected = List[Int]()
            for row in range(ROWS):
                if perfect_probe_key(row) % 50 != 0:
                    continue
                var mark_valid = row % 23 != 0
                if shape == 0 and mark_valid:
                    continue
                if shape == 1 and (not mark_valid or row % 11 == 4):
                    continue
                if shape == 2 and (row + 1) % 3 == 0:
                    continue
                expected.append(row)
            var result = filtered.join(
                dimension.lazy(), left_on=["k"], right_on=["d"]
            ).collect()
            assert_equal(ids(result), expected)
        _ = workers^
