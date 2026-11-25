"""Chunk-independent identity, bounded capacity, and concurrent key claims."""
# contract-group: execution.backends
from max.gpu import global_idx
from std.memory import Pointer
from std.testing import assert_equal, assert_true
from dataframe_metal.accelerator.gpu.group_hash import NO_GROUP, _key_hash
from dataframe_metal.accelerator.gpu.target import METAL_TARGET, MetalContext
from dataframe_metal.accelerator.gpu.word_group import claim_word_group
from dataframe_metal.group_layout import GROUP_CAPACITY_FAULT


def _colliding_keys() -> List[UInt32]:
    # Force wraparound across the reserved null and zero slots.
    var keys = List[UInt32]()
    var raw = UInt32(2)
    while len(keys) < 2:
        if _key_hash(Int64(raw)) & 511 == 511:
            keys.append(raw)
        raw += 1
    return keys^


def test_metal_word_group_key_identity_and_capacity() raises:
    var table = List[UInt32](length=512, fill=0)
    var control: List[UInt32] = [0, 0, 17, 31]
    var t = Pointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=Int(table.unsafe_ptr())
    )
    var c = Pointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=Int(control.unsafe_ptr())
    )
    var colliding = _colliding_keys()
    assert_equal(claim_word_group(1, False, t, 512, c), UInt32(0))
    assert_equal(claim_word_group(UInt32.MAX, False, t, 512, c), UInt32(0))
    assert_equal(claim_word_group(0, True, t, 512, c), UInt32(1))
    assert_equal(claim_word_group(colliding[0], True, t, 512, c), UInt32(511))
    assert_equal(claim_word_group(colliding[1], True, t, 512, c), UInt32(2))
    var keys: List[UInt32] = [1, 0x8000_0000, UInt32.MAX]
    var slots = List[UInt32]()
    for key in keys:
        var slot = claim_word_group(key, True, t, 512, c)
        assert_true(slot >= 2 and slot < 512)
        assert_equal(table[Int(slot)], key)
        slots.append(slot)
    var next = UInt32(2)
    while control[0] < 256:
        _ = claim_word_group(next, True, t, 512, c)
        next += 1
    for i in range(len(keys)):
        assert_equal(claim_word_group(keys[i], True, t, 512, c), slots[i])
    assert_equal(control[0], UInt32(256))
    assert_equal(control[1], UInt32(0))
    assert_equal(claim_word_group(0xDEAD_BEEF, True, t, 512, c), NO_GROUP)
    assert_equal(control[1], GROUP_CAPACITY_FAULT)
    assert_equal(claim_word_group(0, True, t, 512, c), NO_GROUP)
    assert_equal(control[2], UInt32(17))
    assert_equal(control[3], UInt32(31))
    _ = (table^, control^)


def _claim_rows(
    a: UInt32,
    b: UInt32,
    shift: UInt32,
    table: Pointer[UInt32, MutUntrackedOrigin],
    control: Pointer[UInt32, MutUntrackedOrigin],
    slots: Pointer[UInt32, MutUntrackedOrigin],
):
    var row = UInt32(global_idx.x)
    var kind = (row + shift) % 7
    var key = row
    if kind == 1:
        key = 0
    elif kind == 2:
        key = UInt32.MAX
    elif kind == 3:
        key = 0x8000_0000
    elif kind == 4:
        key = 1
    elif kind == 5:
        key = a
    elif kind == 6:
        key = b
    slots[unsafe_offset=Int(shift) * 4096 + Int(row)] = claim_word_group(
        key, kind != 0, table, 512, control
    )


# requires: metal
def test_metal_word_group_concurrent_claims_across_dispatches() raises:
    comptime if METAL_TARGET:
        var ctx = MetalContext(api="metal")
        var table = ctx.enqueue_create_buffer[DType.uint32](512)
        var control = ctx.enqueue_create_buffer[DType.uint32](2)
        var slots = ctx.enqueue_create_buffer[DType.uint32](8192)
        var counters = List[UInt32](length=2, fill=0)
        var received = List[UInt32](length=8192, fill=NO_GROUP)
        var colliding = _colliding_keys()
        ctx.enqueue_memset(table, UInt32(0))
        ctx.enqueue_memset(control, UInt32(0))
        for phase in range(2):
            ctx.enqueue_function[_claim_rows](
                colliding[0],
                colliding[1],
                UInt32(phase),
                table,
                control,
                slots,
                grid_dim=16,
                block_dim=256,
            )
        ctx.enqueue_copy(received.unsafe_ptr(), slots)
        ctx.enqueue_copy(counters.unsafe_ptr(), control)
        ctx.synchronize()
        assert_equal(counters[0], UInt32(7))
        assert_equal(counters[1], UInt32(0))
        assert_equal(received[0], UInt32(0))
        assert_equal(received[1], UInt32(1))
        for kind in range(2, 7):
            assert_true(received[kind] >= 2 and received[kind] < 512)
            for other in range(kind):
                assert_true(received[kind] != received[other])
        for phase in range(2):
            for row in range(4096):
                assert_equal(
                    received[phase * 4096 + row], received[(row + phase) % 7]
                )
