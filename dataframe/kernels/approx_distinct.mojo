"""Full-input, mergeable HyperLogLog sketches for expression reductions.

Polars' CardinalitySketch uses 256 one-byte registers, a harmonic estimate
and linear counting for a small domain. This does the same; planner
statistics retain their separate sampled 64-register sketch.
"""
from std.bit import count_leading_zeros
from std.math import log
from std.memory import bitcast
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.kernels.distinct_statistics import _mix, _hash_bytes, _hash_value
from dataframe.series import Series
from dataframe.storage.string import StringColumn


struct CardinalitySketch(Copyable, Movable):
    var registers: List[UInt8]

    def __init__(out self):
        self.registers = List[UInt8](length=256, fill=0)

    @always_inline
    def insert(mut self, hash: UInt64):
        var index = Int(hash >> 56)
        var rank = UInt8(1 + count_leading_zeros((hash << 8) | UInt64(128)))
        self.registers[index] = max(self.registers[index], rank)

    def merge(mut self, other: Self):
        for k in range(256):
            self.registers[k] = max(self.registers[k], other.registers[k])

    def estimate(self) -> Int:
        var zeros = 0
        var sum = Float64(0)
        for rank in self.registers:
            zeros += Int(rank == 0)
            sum += bitcast[DType.float64]((UInt64(1023) - UInt64(rank)) << 52)
        if zeros == 256:
            return 0
        var m = Float64(256)
        var estimate = (0.7213 / (1.0 + 1.079 / m)) * m * m / sum
        if estimate <= 2.5 * m and zeros > 0:
            estimate = m * log(m / Float64(zeros))
        return max(1, Int(min(Float64(UInt32.MAX), estimate + 0.5)))


@always_inline
def _group(grouped: Bool, groups: List[Int], row: Int) -> Int:
    return groups[row] if grouped else 0


def _wide_hash(value: Int128) -> UInt64:
    var low = UInt64(UInt128(value) & UInt128(0xFFFFFFFFFFFFFFFF))
    var high = UInt64(UInt128(value) >> 64)
    return _mix(low ^ _mix(high + UInt64(0x9E3779B97F4A7C15)))


def update_sketches(
    mut sketches: List[CardinalitySketch],
    input: Series,
    offset: Int,
    grouped: Bool,
    groups: List[Int],
) raises:
    if input.dtype().is_categorical():
        update_sketches(
            sketches, input.cast(DataType.STRING), offset, grouped, groups
        )
        return
    comptime NULL_HASH = UInt64(0xA2B8452D38EAF631)
    comptime VALUE_SALT = UInt64(0xCB9F145237A051ED)
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if input._data.isa[Column[Scalar[D]]]():
            ref column = input._data[Column[Scalar[D]]]
            for row in range(len(input)):
                var hash = _mix(
                    _hash_value[D](column._get(row)) ^ VALUE_SALT
                ) if column._valid(row) else NULL_HASH
                sketches[_group(grouped, groups, offset + row)].insert(hash)
            return
    if input._data.isa[Column[Int128]]():
        ref column = input._data[Column[Int128]]
        for row in range(len(input)):
            var hash = _mix(
                _wide_hash(column._get(row)) ^ VALUE_SALT
            ) if column._valid(row) else NULL_HASH
            sketches[_group(grouped, groups, offset + row)].insert(hash)
        return
    if input._data.isa[StringColumn]():
        ref column = input._data[StringColumn]
        for row in range(len(input)):
            var hash = _mix(
                _hash_bytes(column._row_bytes(row)) ^ VALUE_SALT
            ) if column._valid(row) else NULL_HASH
            sketches[_group(grouped, groups, offset + row)].insert(hash)
        return
    if input._data.isa[BoolColumn]():
        ref column = input._data[BoolColumn]
        for row in range(len(input)):
            var hash = _mix(
                UInt64(column._get(row)) ^ VALUE_SALT
            ) if column._valid(row) else NULL_HASH
            sketches[_group(grouped, groups, offset + row)].insert(hash)
        return
    raise Error("approx_n_unique does not support " + input.dtype().name())
