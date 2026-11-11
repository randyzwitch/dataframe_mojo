"""Fixed hash bits and RNG outputs survive shared-finalizer refactoring."""
# contract-group: kernel.order_keys
from std.testing import assert_equal
from dataframe.hash_bits import _splitmix64
from dataframe.kernels.partition import _mix
from dataframe.aggregation.state import _word_hash
from dataframe.kernels.sampling import SplitMix64, sample_indices


def test_splitmix_finalizer_and_public_rng_match_fixed_bit_vectors() raises:
    var inputs: List[UInt64] = [
        0x0000000000000000,
        0x0000000000000001,
        0xFFFFFFFFFFFFFFFF,
        0x8000000000000000,
        0x9E3779B97F4A7C15,
        0x0123456789ABCDEF,
    ]
    var expected: List[UInt64] = [
        0x0000000000000000,
        0x5692161D100B05E5,
        0xB4D055FCF2CBBD7B,
        0x25C26EA579CEA98A,
        0xE220A8397B1DCDAF,
        0xB2C058E4EBB5112C,
    ]
    for i in range(len(inputs)):
        assert_equal(_splitmix64(inputs[i]), expected[i])
        assert_equal(_mix(inputs[i]), expected[i])
    var generator = SplitMix64(0)
    var sequence: List[UInt64] = [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
        0xF88BB8A8724C81EC,
        0x1B39896A51A8749B,
        0x53CB9F0C747EA2EA,
        0x2C829ABE1F4532E1,
        0xC584133AC916AB3C,
    ]
    for value in sequence:
        assert_equal(generator.next(), value)


def test_group_word_hash_keeps_seed_and_signed_word_encoding() raises:
    var words: List[List[Int]] = [[0, 1, -1, Int.MIN], [42, -42, Int.MAX, 1]]
    var expected: List[UInt64] = [
        0x4579B960BB007F46,
        0x8D78E1FBF0BD6706,
        0xA0AB19D60A7CE41A,
        0xC9BF4C894869BA89,
    ]
    for row in range(4):
        assert_equal(_word_hash(words, row), expected[row])


def test_seeded_sampling_keeps_sparse_dense_replacement_and_shuffle_sequences() raises:
    assert_equal(sample_indices(20, 0, False, False, 42), List[Int]())
    assert_equal(sample_indices(20, 0, False, True, 42), List[Int]())
    assert_equal(sample_indices(20, 0, True, False, 42), List[Int]())
    assert_equal(sample_indices(20, 0, True, True, 42), List[Int]())
    assert_equal(
        sample_indices(20, 4, False, False, 42), List[Int]([2, 5, 6, 12])
    )
    assert_equal(
        sample_indices(20, 4, False, True, 42), List[Int]([2, 6, 5, 12])
    )
    assert_equal(
        sample_indices(20, 4, True, False, 42), List[Int]([3, 5, 6, 14])
    )
    assert_equal(
        sample_indices(20, 4, True, True, 42), List[Int]([3, 6, 5, 14])
    )
    assert_equal(
        sample_indices(20, 12, False, False, 42),
        List[Int]([1, 3, 4, 6, 7, 8, 9, 12, 14, 15, 17, 18]),
    )
    assert_equal(
        sample_indices(20, 12, False, True, 42),
        List[Int]([14, 4, 7, 8, 1, 18, 9, 17, 12, 15, 3, 6]),
    )
    assert_equal(
        sample_indices(20, 12, True, False, 42),
        List[Int]([0, 3, 4, 4, 5, 6, 6, 9, 12, 14, 16, 17]),
    )
    assert_equal(
        sample_indices(20, 12, True, True, 42),
        List[Int]([5, 6, 4, 0, 12, 16, 6, 14, 3, 9, 17, 4]),
    )
