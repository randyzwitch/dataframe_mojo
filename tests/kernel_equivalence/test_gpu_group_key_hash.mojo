"""Key hashing is congruent with the existing exact grouping equivalence."""
from std.testing import assert_equal
from dataframe.accelerator.gpu.grouping import key_equal
from dataframe.accelerator.gpu.hash_grouping import _canonical


def test_gpu_group_key_hash_respects_float_encodings_and_integer_identity() raises:
    var words: List[UInt64] = [
        0,
        0x80000000,
        0x8000000000000000,
        1,
        0x80000001,
        0x8000000000000001,
        0x3F800000,
        0x123456783F800000,
        0x3FF0000000000000,
        0xBFF0000000000000,
        0x7FC00031,
        0xFFC00042,
        0x123456787FC00099,
        0x7FF8000000000031,
        0xFFF8000000000042,
        0x7F800000,
        0xFF800000,
        0x7FF0000000000000,
        0xFFF0000000000000,
    ]
    for dtype in [Int64(1), Int64(11)]:
        for a in words:
            for b in words:
                if key_equal(a, b, dtype):
                    assert_equal(_canonical(a, dtype), _canonical(b, dtype))
    for word in words:
        assert_equal(_canonical(word, 3), word)


def main() raises:
    test_gpu_group_key_hash_respects_float_encodings_and_integer_identity()
    print("PASS key equality/hash congruence and integer identity")
