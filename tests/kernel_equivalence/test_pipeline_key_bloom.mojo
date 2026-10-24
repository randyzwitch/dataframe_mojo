"""A join build much smaller than the rows it filters, with keys spread too
wide for its range to tell, hands the pipeline a bloom filter over its keys
(`runtime_filter.mojo`); a probe row whose key the filter lacks is dropped
before any step reads it. Joins against a scalar model: inner and semi joins,
Int64 and Int32 keys, nullable probe keys, a second join after the first so
the filter works on rows an earlier filter kept, and builds too big for a
filter, on one and eight workers. The filter itself never loses a key."""
# contract-group: table.join
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    col,
)
from dataframe.runtime_filter import KeyBloom, bloom_blocks_for, bloom_mask
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
    dtype: DataType, a_every: Int, b_every: Int, semi: Bool, expect_bloom: Bool
) raises:
    var source = probe(dtype)
    var plan = (
        source.lazy()
        .join(
            build("a", 5000, 1, a_every, dtype).lazy(),
            left_on=["ak"],
            right_on=["a"],
        )
        .join(
            build("b", 7000, 2, b_every, dtype).lazy(),
            left_on=["bk"],
            right_on=["b"],
            how="semi" if semi else "inner",
        )
    )
    var expected = List[Int]()
    for row in range(ROWS):
        if not key_valid(row):
            continue
        if (row % 5000) % a_every != 0:
            continue
        if (row % 7000) % b_every != 0:
            continue
        expected.append(row)
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var result = plan.collect()
    var took = "dataframe-path: pipeline.key_bloom\n" in observed.text()
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
    # Few keys outside it pass: 8 bits a key leaves about 2% false hits.
    var false_hits = 0
    for k in range(10_000):
        if bloom.contains(Int64(SPAN + 1 + k)):
            false_hits += 1
    assert_true(false_hits < 500)
    # The mask holds the rows the filter passes: every third row's key is
    # in it.
    var values = List[Int64]()
    for r in range(1000):
        values.append(keys[r] if r % 3 == 0 else Int64(SPAN + 7 + r))
    var series = Series("k", Column[Int64](values^))
    var found = bloom_mask(series, bloom)
    assert_true(found.kept >= 334)
    assert_true(found.kept < 334 + 60)
    for r in range(0, 1000, 3):
        assert_true(found.mask.value(r))
