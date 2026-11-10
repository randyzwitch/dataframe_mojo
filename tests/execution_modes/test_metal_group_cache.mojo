"""Thread-local contributions retain the existing exact grouped state."""
# contract-group: execution.backends
from std.atomic import Atomic
from std.memory import bitcast, Pointer
from std.testing import assert_equal
from dataframe_metal.accelerator.gpu.group_local_state import GroupLocalState
from dataframe_metal.accelerator.gpu.group_state import (
    initial_group_word,
    add_integer_sum,
    add_float_sum,
    add_ordered_extrema,
)


def _check_local_state(want_sum: Bool, want_extrema: Bool, flush: Bool) raises:
    var expected = List[UInt32](length=128, fill=0)
    var actual = List[UInt32](length=128, fill=0)
    for i in range(128):
        expected[i] = initial_group_word(i % 32)
        actual[i] = initial_group_word(i % 32)
    var ep = Pointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=Int(expected.unsafe_ptr())
    )
    var ap = Pointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=Int(actual.unsafe_ptr())
    )
    var cache = GroupLocalState()
    var random = UInt64(70031)
    for row in range(10000):
        for field in range(4):
            random = random * 6364136223846793005 + 1442695040888963407
            if row % 13 == field:
                continue
            var bits = random
            var kind = 0 if field in [0, 3] else field
            if kind == 1:
                bits = bitcast[DType.uint64](
                    Int64(bitcast[DType.int32](UInt32(bits)))
                )
            elif kind == 0:
                bits = UInt64(UInt32(bits))
            var state = ep.unsafe_offset(field * 32)
            expected[field * 32] += 1
            if kind == 0:
                if want_sum:
                    add_float_sum(state, UInt32(bits))
                if want_extrema:
                    add_ordered_extrema(state, UInt32(bits), True)
            else:
                if want_sum:
                    add_integer_sum(state, bitcast[DType.int64](bits))
                if kind == 1 and want_extrema:
                    add_ordered_extrema(state, UInt32(bits), False)
            cache.add(
                field,
                kind,
                bits,
                UInt32(row),
                want_sum,
                want_extrema,
                ap.unsafe_offset(field * 32),
            )
        if flush and row % 37 == 36:
            for field in range(4):
                var kind = 0 if field in [0, 3] else field
                cache.flush_field(
                    field,
                    kind,
                    want_sum,
                    want_extrema,
                    ap.unsafe_offset(field * 32),
                )
            cache = GroupLocalState()
    for field in range(4):
        var kind = 0 if field in [0, 3] else field
        cache.flush_field(
            field, kind, want_sum, want_extrema, ap.unsafe_offset(field * 32)
        )
    for i in range(128):
        assert_equal(actual[i], expected[i], String(i))
    print(
        "PASS exact cached Float32 windows, Int32/Int64 sums, counts and extrema:40000contributions"
    )


def test_metal_group_local_cache_preserves_exact_partial_states() raises:
    for mask in range(4):
        _check_local_state(Bool(mask & 1), Bool(mask & 2), True)
    _check_local_state(True, True, False)
