"""Group ids for a perfect hash aggregate.

When every grouping key is an integer whose values span a small range, or a
categorical (a dictionary-coded string's codes) with a small dictionary, a
row's group is its position in a table holding every combination: each key
contributes `value - min + 1` (0 for a null) shifted into its own bits, so
no table is probed and no group is numbered. the bounds come from the key buffers' zone maps
(`pipeline._ensure_zones`), kept with the buffers. The keys together may
need at most `PERFECT_HASH_BITS` bits.
"""
from std.memory import Pointer

from dataframe.storage.column import Column
from dataframe.dtype import DataType
from dataframe.execution import _valid_lanes
from dataframe.pipeline import _ensure_zones, _range_key, _window_bounds
from dataframe.series import Series

# DuckDB's default `perfect_ht_threshold`: at most 2^12 group slots.
comptime PERFECT_HASH_BITS = 12


@fieldwise_init
struct PerfectLayout(Copyable, Movable):
    """Where each key's part of a group slot lies: its least value and the
    shift and width of its bits. The first key takes the highest bits."""

    var minima: List[Int64]
    var shifts: List[Int]
    var bits: Int

    def slots(self) -> Int:
        return 1 << self.bits


def _required_bits(n: UInt64) -> Int:
    """The number of bits needed to represent n."""
    var bits = 0
    var rest = n
    while rest > 0:
        rest >>= 1
        bits += 1
    return bits


def perfect_layout(keys: List[Series]) raises -> Optional[PerfectLayout]:
    """The perfect hash layout of these keys, or None when one is not an
    integer stored as Int64 or Int32 or they need more than
    `PERFECT_HASH_BITS` bits."""
    if len(keys) == 0:
        return None
    for key in keys:
        if key.dtype().is_categorical():
            if not key.dtype().has_dictionary():
                return None
        elif not _range_key(key.dtype()):
            return None
    var minima = List[Int64](capacity=len(keys))
    var widths = List[Int](capacity=len(keys))
    var total = 0
    for key in keys:
        var low = Int64(0)
        var span = UInt64(0)
        if key.dtype().is_categorical():
            # A categorical's codes are 0 to its dictionary's size less
            # one: known without a scan.
            span = UInt64(max(len(key.dtype().dictionary()[]) - 1, 0))
        else:
            _ensure_zones(key)
            var bounds = _window_bounds(key)
            if bounds:
                low = bounds.value()[0]
                var high = bounds.value()[1]
                # As DuckDB, ranges past 2^31 never qualify; checked in 128
                # bits so that a range across all of Int64 cannot wrap.
                var range = Int128(high) - Int128(low)
                if range >= Int128(Int32.MAX):
                    return None
                span = UInt64(range)
        # One slot for null and one more for the one-based values.
        var bits = _required_bits(span + 2)
        total += bits
        if total > PERFECT_HASH_BITS:
            return None
        minima.append(low)
        widths.append(bits)
    var shifts = List[Int](capacity=len(keys))
    var shift = total
    for k in range(len(keys)):
        shift -= widths[k]
        shifts.append(shift)
    return PerfectLayout(minima^, shifts^, total)


def perfect_slots(
    keys: List[Series],
    layout: PerfectLayout,
    start: Int,
    length: Int,
    mut slots: List[Int],
) raises:
    """The group slot of rows [start, start + length) of the keys, into
    slots[0, length). A null adds nothing, so it lands in its key's slot
    0."""
    var all = slots.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for k in range(len(keys)):
        var window = keys[k].slice(start, length)
        var at = 0
        for part in window.chunks():
            var target = all.unsafe_offset(at)
            if part.dtype().physical() == DataType.INT64:
                _add_slots(
                    part._data[Column[Int64]],
                    layout.minima[k],
                    layout.shifts[k],
                    k == 0,
                    target,
                )
            elif part._data.isa[Column[UInt32]]():
                _add_slots(
                    part._data[Column[UInt32]],
                    layout.minima[k],
                    layout.shifts[k],
                    k == 0,
                    target,
                )
            else:
                _add_slots(
                    part._data[Column[Int32]],
                    layout.minima[k],
                    layout.shifts[k],
                    k == 0,
                    target,
                )
            at += len(part)


def _add_slots[
    D: DType
](
    column: Column[Scalar[D]],
    low: Int64,
    shift: Int,
    first: Bool,
    slots: Pointer[Int, MutAnyOrigin],
):
    """Each row's part of its slot, `(value - low + 1) << shift` or 0 for a
    null, stored (the first key) or added to the slots so far, eight rows
    at a time under the validity bitmap."""
    var n = len(column)
    var values = column._ptr()
    var nulls = column.null_count() > 0
    var bits = column.unsafe_validity()
    var base = SIMD[DType.int64, 8](1 - low)
    var by = SIMD[DType.int64, 8](Int64(shift))
    var target = slots.unsafe_bitcast[Int64]()
    var i = 0
    while i + 8 <= n:
        var part = (
            values.unsafe_load[width=8](i).cast[DType.int64]() + base
        ) << by
        if nulls:
            part = _valid_lanes(bits, column._offset + i).select(
                part, SIMD[DType.int64, 8](0)
            )
        if not first:
            part += target.unsafe_load[width=8](i)
        target.unsafe_store(i, part)
        i += 8
    while i < n:
        var part = Int64(0)
        if not nulls or column._valid(i):
            part = (
                values[unsafe_offset=i].cast[DType.int64]() - low + 1
            ) << Int64(shift)
        if not first:
            part += target[unsafe_offset=i]
        target[unsafe_offset=i] = part
        i += 1
