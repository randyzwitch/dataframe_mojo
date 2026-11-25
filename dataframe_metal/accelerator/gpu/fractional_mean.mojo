"""Stable hash bucketing and input-order grouped means over one native window."""
from max.gpu import global_idx
from std.atomic import Atomic
from std.memory import Pointer, bitcast
from .filter_sum import _valid
from .group_hash import _key_hash
from .multi_reduce import Bytes, read_integer
from .soft_binary64 import add_binary64, widen_binary32

comptime MEAN_BUCKETS = 128
comptime MEAN_TILE = 128
comptime MEAN_WINDOW = 65536
comptime MEAN_KEY_WORDS = 5
comptime MEAN_FIELD_WORDS = 4
comptime Words = Pointer[UInt32, MutUntrackedOrigin]


@always_inline
def _present(bits: Bytes, desc: UInt64, row: Int) -> Bool:
    return not (desc & 8) or _valid(bits, UInt32(desc >> 32), row)


@always_inline
def _hash_key(raw: Int64, present: Bool) -> UInt32:
    return _key_hash(raw) if present else UInt32(0xD1B54A35)


def bucket_mean_rows(
    keys: Bytes,
    bits: Bytes,
    desc: UInt64,
    length: Int,
    heads: Words,
    links: Words,
    control: Words,
):
    var tile = global_idx.x
    var start = tile * MEAN_TILE
    if start >= length:
        return
    if tile == 0:
        _ = Atomic[UInt32].fetch_add(control.unsafe_offset(2), UInt32(1))
    var local = heads.unsafe_offset(tile * MEAN_BUCKETS)
    for bucket in range(MEAN_BUCKETS):
        local[unsafe_offset=bucket] = 0
    # One writer per tile. Prepending backwards links every bucket in source order.
    for row in reversed(range(start, min(start + MEAN_TILE, length))):
        var present = _present(bits, desc, row)
        var raw = read_integer(
            keys, UInt32(desc & 1), row
        ) if present else Int64(0)
        var bucket = Int(_hash_key(raw, present) & (MEAN_BUCKETS - 1))
        links[unsafe_offset=row] = local[unsafe_offset=bucket]
        local[unsafe_offset=bucket] = UInt32(row + 1)


@always_inline
def _update_mean(
    values: Bytes, bits: Bytes, desc: UInt64, row: Int, state: Words
):
    if not _present(bits, desc, row):
        return
    state[unsafe_offset=0] += 1
    var total = UInt64(state[unsafe_offset=1]) | (
        UInt64(state[unsafe_offset=2]) << 32
    )
    var tag = UInt32(desc & 7)
    if tag == 2:
        total = add_binary64(
            total,
            widen_binary32(values.unsafe_bitcast[UInt32]()[unsafe_offset=row]),
        )
    elif tag == 4:
        total = add_binary64(
            total, values.unsafe_bitcast[UInt64]()[unsafe_offset=row]
        )
    else:
        var value = read_integer(values, tag, row)
        var previous = total
        total += bitcast[DType.uint64](value)
        state[unsafe_offset=3] += UInt32(total < previous) - UInt32(value < 0)
    state[unsafe_offset=1] = UInt32(total)
    state[unsafe_offset=2] = UInt32(total >> 32)


def reduce_mean_buckets(
    keys: Bytes,
    key_bits: Bytes,
    key_desc: UInt64,
    v0: Bytes,
    b0: Bytes,
    d0: UInt64,
    v1: Bytes,
    b1: Bytes,
    d1: UInt64,
    v2: Bytes,
    b2: Bytes,
    d2: UInt64,
    v3: Bytes,
    b3: Bytes,
    d3: UInt64,
    base: UInt32,
    length: Int,
    fields: Int,
    capacity: UInt32,
    heads: Words,
    links: Words,
    table: Words,
    control: Words,
):
    var bucket = global_idx.x
    if bucket >= MEAN_BUCKETS:
        return
    if bucket == 0:
        _ = Atomic[UInt32].fetch_add(control.unsafe_offset(3), UInt32(1))
    var owner = UInt32(bucket + 1)
    var stride = MEAN_KEY_WORDS + fields * MEAN_FIELD_WORDS
    for tile in range((length + MEAN_TILE - 1) // MEAN_TILE):
        var next_row = heads[unsafe_offset=tile * MEAN_BUCKETS + bucket]
        while next_row:
            var row = Int(next_row) - 1
            next_row = links[unsafe_offset=row]
            if Atomic[UInt32].load(control.unsafe_offset(1)):
                return
            var present = _present(key_bits, key_desc, row)
            var raw = read_integer(
                keys, UInt32(key_desc & 1), row
            ) if present else Int64(0)
            var hashed = _hash_key(raw, present)
            var slot = hashed & (capacity - 1)
            var low = UInt32(bitcast[DType.uint64](raw))
            var high = UInt32(bitcast[DType.uint64](raw) >> 32)
            while True:
                var record = table.unsafe_offset(Int(slot) * stride)
                var found = Atomic[UInt32].load(record)
                if found == 0:
                    if Atomic[UInt32].compare_exchange[weak=True](
                        record, found, owner
                    ):
                        if (
                            Atomic[UInt32].fetch_add(control, UInt32(1))
                            >= capacity // 2
                        ):
                            Atomic[UInt32].store(
                                control.unsafe_offset(1), UInt32(1)
                            )
                            return
                        record[unsafe_offset=1] = base + UInt32(row) + 1
                        record[unsafe_offset=2] = low
                        record[unsafe_offset=3] = high
                        record[unsafe_offset=4] = UInt32(present)
                        found = owner
                    elif found == 0:
                        continue
                # Only this bucket's lane may read or update a published key.
                # Other lanes inspect the atomic owner word alone.
                if (
                    found == owner
                    and record[unsafe_offset=2] == low
                    and record[unsafe_offset=3] == high
                    and record[unsafe_offset=4] == UInt32(present)
                ):
                    var field = record.unsafe_offset(MEAN_KEY_WORDS)
                    _update_mean(v0, b0, d0, row, field)
                    if fields > 1:
                        _update_mean(
                            v1,
                            b1,
                            d1,
                            row,
                            field.unsafe_offset(MEAN_FIELD_WORDS),
                        )
                    if fields > 2:
                        _update_mean(
                            v2,
                            b2,
                            d2,
                            row,
                            field.unsafe_offset(2 * MEAN_FIELD_WORDS),
                        )
                    if fields > 3:
                        _update_mean(
                            v3,
                            b3,
                            d3,
                            row,
                            field.unsafe_offset(3 * MEAN_FIELD_WORDS),
                        )
                    break
                slot = (slot + 1) & (capacity - 1)
