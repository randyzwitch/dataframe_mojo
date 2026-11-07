"""DuckDB's perfect hash aggregate (`PhysicalPerfectHashAggregate` over a
`PerfectAggregateHashTable`, execution/perfect_aggregate_hashtable.cpp),
on the aggregate table's group rows and states (`hash_table.mojo`).

When every grouping key is an integer whose statistics bound it to a small
range, a row's group is its position in a table holding every combination
of key values: each key contributes `value - min + 1` (0 for a null)
shifted into its own bits (`ComputeGroupLocation`), so no key is hashed or
compared and no group is numbered. Each worker holds the whole table, every
group's row allocated and empty up front; the workers' tables are combined
slot by slot (`Combine`), and the occupied slots are scanned out in slot
order, each key rebuilt from its bits (`Scan`, `ReconstructGroupVector`).

`perfect_hash` is the planner's rule (`CanUsePerfectHashAggregate`,
physical_plan/plan_aggregate.cpp): integer keys only (a categorical's codes
count as UINT32, bounded by its dictionary), each key's minimum and maximum
from its column statistics (`segment_statistics.mojo`; an 8- or 16-bit key
with none takes its type's bounds), a range below 2^31, and the keys'
bits together at most DuckDB's `perfect_ht_threshold`.
"""
from std.memory import Pointer, bitcast
from std.sys import size_of

from dataframe.dtype import NUMERIC_DTYPES
from dataframe.kernels.segment_statistics import (
    ensure_statistics,
    series_statistics,
)
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.storage.statistics import NUMERIC_STATS
from dataframe.vector.types import UnifiedVectorFormat, physical_of

# DuckDB's `perfect_ht_threshold` default (settings.hpp): at most 2^12
# group slots.
comptime PERFECT_HT_THRESHOLD = 12


@fieldwise_init
struct PerfectHash(Copyable, Movable):
    """The perfect hash table's shape: each key's least value (as its key
    word, so subtraction wraps as the key's own type does) and the bits it
    takes, the first key the highest. No keys means no perfect hash."""

    var minima: List[UInt64]
    var bits: List[Int]
    var total_bits: Int

    @staticmethod
    def none() -> PerfectHash:
        return PerfectHash(List[UInt64](), List[Int](), 0)

    def used(self) -> Bool:
        return len(self.bits) > 0

    def total_groups(self) -> Int:
        """`total_groups`: 2^total_required_bits."""
        return 1 << self.total_bits


def _required_bits(n: UInt64) -> Int:
    """`RequiredBitsForValue`: the bits needed to represent n."""
    var bits = 0
    var rest = n
    while rest > 0:
        rest >>= 1
        bits += 1
    return bits


def _integer_storage(part: Series) -> Int:
    """The bit width of an integer column's storage, or 0 for any other."""
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        comptime if D.is_integral():
            if part._data.isa[Column[Scalar[D]]]():
                return 8 * size_of[Scalar[D]]()
    return 0


def _type_bounds(part: Series) -> Tuple[Int128, Int128]:
    """An 8- or 16-bit integer storage's least and greatest value."""
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        comptime if D.is_integral() and size_of[Scalar[D]]() <= 2:
            if part._data.isa[Column[Scalar[D]]]():
                return (Int128(Scalar[D].MIN), Int128(Scalar[D].MAX))
    return (Int128(0), Int128(-1))


def perfect_hash(keys: List[Series]) raises -> PerfectHash:
    """`CanUsePerfectHashAggregate` for grouping keys like `keys`, which
    carry (or are given) their statistics: the table's shape, or none
    when some key does not qualify."""
    if len(keys) == 0:
        return PerfectHash.none()
    var minima = List[UInt64](capacity=len(keys))
    var bits = List[Int](capacity=len(keys))
    var total = 0
    for key in keys:
        var low: Int128
        var high: Int128
        var dtype = key.dtype()
        if dtype.is_nested():
            return PerfectHash.none()
        if dtype.is_categorical():
            # The codes of a categorical (DuckDB's ENUM, a UINT32 inside)
            # lie in its dictionary.
            if not dtype.has_dictionary():
                return PerfectHash.none()
            low = 0
            high = Int128(len(dtype.dictionary()[])) - 1
        else:
            if len(key) == 0:
                return PerfectHash.none()
            var part = key.chunks()[0].copy() if key.is_chunked() else key.copy()
            var width = _integer_storage(part)
            if width == 0:
                return PerfectHash.none()
            try:
                ensure_statistics(key)
            except:
                pass
            var stats = series_statistics(key)
            if stats:
                ref found = stats.value()
                if found.stats_type != NUMERIC_STATS or not found.has_min_max:
                    return PerfectHash.none()
                low = found.int_min
                high = found.int_max
            elif width <= 16:
                # No statistics: a small type's own bounds.
                var bounds = _type_bounds(part)
                low = bounds[0]
                high = bounds[1]
            else:
                return PerfectHash.none()
        if high < low:
            return PerfectHash.none()
        var range = high - low
        # Any range of 2^31 or more never qualifies.
        if range >= Int128(Int32.MAX):
            return PerfectHash.none()
        # One slot for null and one more for the one-based values.
        var needed = _required_bits(UInt64(range + 2))
        total += needed
        if total > PERFECT_HT_THRESHOLD:
            return PerfectHash.none()
        minima.append(UInt64(low & Int128(0xFFFFFFFFFFFFFFFF)))
        bits.append(needed)
    return PerfectHash(minima^, bits^, total)


def group_locations(
    format: UnifiedVectorFormat,
    map: Int,
    map_length: Int,
    low: UInt64,
    shift: Int,
    count: Int,
    locations: Pointer[UInt64, MutAnyOrigin],
) raises:
    """`ComputeGroupLocation`: each of a key vector's `count` rows adds
    `(value - low + 1) << shift` to its location, a null adding nothing.
    `map` (with `map_length` entries) replaces a categorical's codes by
    their canonical codes first, as the table's key words do."""
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        comptime if D.is_integral():
            if format.physical == physical_of[D]():
                if format.sel.is_set():
                    _locations[D, True](
                        format, map, map_length, low, shift, count, locations
                    )
                else:
                    _locations[D, False](
                        format, map, map_length, low, shift, count, locations
                    )
                return
    raise Error("perfect hash aggregate: a key vector that is no integer")


def _locations[
    D: DType, selected: Bool
](
    format: UnifiedVectorFormat,
    map: Int,
    map_length: Int,
    low: UInt64,
    shift: Int,
    count: Int,
    locations: Pointer[UInt64, MutAnyOrigin],
) raises:
    """`ComputeGroupLocationTemplated` over values of `D`."""
    var data = format.values[D]()
    var sel = format.sel
    var validity = format.validity
    var nulls = not validity.all_valid()
    var codes = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=map)
    var by = UInt64(shift)
    var base = UInt64(1) - low
    for k in range(count):
        var i: Int
        comptime if selected:
            i = sel.get_index_unsafe(k)
        else:
            i = k
        if nulls and not validity.row_is_valid_unsafe(i):
            continue
        var word: UInt64
        comptime if D.is_unsigned():
            word = UInt64(data[unsafe_offset=i])
            comptime if D == DType.uint32:
                if map != 0:
                    if Int(word) >= map_length:
                        raise Error("Categorical code exceeds its dictionary")
                    word = UInt64(codes[unsafe_offset=Int(word)])
        else:
            word = bitcast[DType.uint64](Int64(data[unsafe_offset=i]))
        locations[unsafe_offset=k] += (word + base) << by


def key_word(perfect: PerfectHash, c: Int, slot: Int) -> Tuple[Bool, UInt64]:
    """`ReconstructGroupVector` for key c of group `slot`: whether it is
    valid, and its key word (the least value plus the key's bits less one,
    wrapping as the key's type does)."""
    var shift = perfect.total_bits
    for k in range(c + 1):
        shift -= perfect.bits[k]
    var mask = (UInt64(1) << UInt64(perfect.bits[c])) - 1
    var index = (UInt64(slot) >> UInt64(shift)) & mask
    if index == 0:
        return (False, UInt64(0))
    return (True, perfect.minima[c] + (index - 1))
