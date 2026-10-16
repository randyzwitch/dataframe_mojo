"""Metal-target host metadata for one combined per-partition String domain."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_false, assert_raises
from dataframe.accelerator.gpu.residency import Scope
from dataframe.accelerator.gpu.partition_labels import build


def _check_literal_labels(
    scope: Scope, words: List[UInt64], used: List[UInt32]
) raises:
    var labels = build(scope, words, used, 1, False)
    var expected: List[String] = [
        "(-inf, 1]",
        "(1, inf]",
        "(-inf, 12]",
        "(12, inf]",
        "(-inf, inf]",
    ]
    var indices: List[Int] = [0, 1, 2, 3, 4]
    for at in range(len(indices)):
        assert_equal(
            labels.domain[]
            .dtype.dictionary()[]
            .get(Int(labels.codes[indices[at]])),
            expected[at],
        )
    assert_equal(labels.codes[0], UInt32(1))
    assert_equal(labels.codes[1], UInt32(3))
    assert_equal(labels.codes[2], UInt32(0))
    assert_equal(labels.codes[3], UInt32(4))
    assert_equal(labels.codes[4], UInt32(2))
    assert_equal(labels.codes[5], UInt32(0))
    # 47 UTF8 bytes +6 Int64 offsets,24bytes fixed-stride bin map.
    assert_equal(labels.domain[].retained, 95)
    assert_equal(labels.retained, 24)
    assert_equal(scope.info()[0], 119)
    # 2*95 pools +5*12 sort/code indices +6*4 map +8 edgevector
    # +16*(11+32) one-label formatting workspace.
    assert_equal(labels.peak, 970)
    assert_false(scope.initialized())
    _ = labels^


# requires: mojo_gpu
def test_partition_qcut_label_domains_literal_mapping_owner_budget_and_guards() raises:
    var words: List[UInt64] = [0x3FF0000000000000, 0x4028000000000000, 0]
    var used: List[UInt32] = [1, 1, 0]
    var scope = Scope(99999)
    _check_literal_labels(scope, words, used)
    assert_equal(scope.info()[0], 0)
    var exact = Scope(99999, 970)
    var allowed = build(exact, words, used, 1, False)
    assert_equal(exact.info()[0], 119)
    var denied = Scope(99999, 969)
    with assert_raises(contains="budget"):
        _ = build(denied, words, used, 1, False)
    assert_equal(denied.info()[0], 0)
    assert_false(denied.initialized())
    var short_words: List[UInt64] = [0]
    with assert_raises(contains="control geometry"):
        _ = build(scope, short_words, used, 1, False)
    var bad_used: List[UInt32] = [2]
    with assert_raises(contains="used-edge count"):
        _ = build(scope, words, bad_used, 1, False)
    var empty_used = List[UInt32]()
    with assert_raises(contains="control geometry"):
        _ = build(scope, words, empty_used, 1, False)
    var zero_words: List[UInt64] = [0x8000000000000000, 0x7FF8000000000000]
    var two: List[UInt32] = [1, 1]
    var nonfinite = build(scope, zero_words, two, 1, True)
    assert_equal(
        nonfinite.domain[].dtype.dictionary()[].get(Int(nonfinite.codes[0])),
        "[-inf, 0)",
    )
    assert_equal(
        nonfinite.domain[].dtype.dictionary()[].get(Int(nonfinite.codes[1])),
        "[0, inf)",
    )
    assert_equal(
        nonfinite.domain[].dtype.dictionary()[].get(Int(nonfinite.codes[2])),
        "[-inf, nan)",
    )
    assert_equal(
        nonfinite.domain[].dtype.dictionary()[].get(Int(nonfinite.codes[3])),
        "[nan, inf)",
    )
    assert_false(scope.initialized())
