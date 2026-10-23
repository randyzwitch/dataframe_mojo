"""Approximate distinct-count statistics for columns in loaded tables.

As rows are appended, each column hashes a sample: the first 614 rows of
each 2,048-row chunk for integer types and the first 204 rows for other
types. A 64-register HyperLogLog sketch summarizes those hashes. A
Good-Turing correction scales the sample estimate to the full column.
Statistics are computed on first planner request and shared with copies
of the loaded table. Frames built by other routes do not carry these
statistics.
"""
from std.atomic import Atomic
from std.bit import count_trailing_zeros
from std.math import sqrt
from std.memory import bitcast
from std.sys import size_of

from .bool_column import BoolColumn
from .column import Column
from .dtype import NUMERIC_DTYPES
from .series import Series
from .string_column import StringColumn

# Registers of the sketch, 2^P (`HyperLogLog::P`, `M`, `Q`).
comptime _P = 6
comptime _M = 64
comptime _Q = 58
comptime _ALPHA = 0.721347520444481703680
# DuckDB's vector size and its sampled share of each vector
# (`STANDARD_VECTOR_SIZE`, `INTEGRAL_SAMPLE_RATE`, `BASE_SAMPLE_RATE`).
comptime _VECTOR = 2048
comptime _INTEGRAL_SAMPLE = 614
comptime _BASE_SAMPLE = 204
comptime _MULTIPLIER = UInt64(0xD6E8FEB86659FD93)


@always_inline
def _mix(value: UInt64) -> UInt64:
    """Mix a 64-bit value with xor-shifts and odd multiplications."""
    var x = value
    x ^= x >> 32
    x *= _MULTIPLIER
    x ^= x >> 32
    x *= _MULTIPLIER
    x ^= x >> 32
    return x


def _hash_bytes(bytes: Span[UInt8, _]) -> UInt64:
    """Hash bytes as little-endian 8-byte words, zero-padding the last
    partial word."""
    var n = len(bytes)
    var h = UInt64(0xE17A1465) ^ (UInt64(n) * UInt64(0xC6A4A7935BD1E995))
    var at = 0
    while at + 8 <= n:
        var word = UInt64(0)
        for b in range(8):
            word |= UInt64(bytes[at + b]) << UInt64(8 * b)
        h ^= word
        h *= _MULTIPLIER
        at += 8
    if at < n:
        var word = UInt64(0)
        for b in range(n - at):
            word |= UInt64(bytes[at + b]) << UInt64(8 * b)
        h ^= word
        h *= _MULTIPLIER
    return _mix(h)


@always_inline
def _hash_value[D: DType](value: Scalar[D]) -> UInt64:
    """Hash one fixed-width value: 32-bit and narrower values through
    `uint32`, 64-bit values directly, and 128-bit values by mixing each
    half. Treat -0.0 as 0.0 and all NaNs as one value."""
    comptime if D == DType.int128:
        var wide = rebind[Int128](value)
        var low = UInt64(UInt128(wide) & UInt128(0xFFFFFFFFFFFFFFFF))
        var high = UInt64(UInt128(wide) >> 64)
        return _mix(low) ^ _mix(high)
    elif D.is_floating_point():
        # -0.0 hashes as 0.0 and every NaN as the quiet NaN.
        var x = value
        if x == 0:
            x = 0
        comptime if D == DType.float64:
            if x != x:
                return _mix(UInt64(0x7FF8000000000000))
            return _mix(bitcast[DType.uint64](rebind[Float64](x)))
        else:
            var single = x.cast[DType.float32]()
            if single != single:
                return _mix(UInt64(0x7FC00000))
            return _mix(bitcast[DType.uint32](single).cast[DType.uint64]())
    elif size_of[Scalar[D]]() == 8:
        return _mix(value.cast[DType.uint64]())
    else:
        return _mix(
            value.cast[DType.int32]().cast[DType.uint32]().cast[DType.uint64]()
        )


struct HyperLogLog(Copyable, Movable):
    """A 64-register HyperLogLog cardinality sketch."""

    var registers: List[UInt8]

    def __init__(out self):
        self.registers = List[UInt8](length=_M, fill=0)

    @always_inline
    def insert(mut self, hash: UInt64):
        var index = Int(hash & UInt64(_M - 1))
        var rest = (hash >> UInt64(_P)) | (UInt64(1) << UInt64(_Q))
        var rank = UInt8(count_trailing_zeros(rest) + 1)
        if rank > self.registers[index]:
            self.registers[index] = rank

    def count(self) -> Int:
        """Estimate cardinality from the harmonic register sum, with the
        high-range correction using register value Q."""
        var c = List[Float64](length=_Q + 2, fill=0)
        for r in self.registers:
            c[Int(r)] += 1
        var m = Float64(_M)
        var z = m * _tau((m - c[_Q]) / m)
        for k in range(_Q, 0, -1):
            z += c[k]
            z *= 0.5
        z += m * _sigma(c[0] / m)
        return Int(round(_ALPHA * m * m / z))


def _sigma(value: Float64) -> Float64:
    if value == 1.0:
        return Float64.MAX
    var x = value
    var y = 1.0
    var z = x
    while True:
        x *= x
        var previous = z
        z += x * y
        y += y
        if previous == z:
            return z


def _tau(value: Float64) -> Float64:
    if value == 0.0 or value == 1.0:
        return 0.0
    var x = value
    var y = 1.0
    var z = 1.0 - x
    while True:
        x = sqrt(x)
        var previous = z
        y *= 0.5
        z -= (1.0 - x) * (1.0 - x) * y
        if previous == z:
            return z / 3.0


def _integral(series: Series) -> Bool:
    """Select the higher sampling rate for integer logical types;
    decimals, dates, and timestamps use the base rate."""
    var dtype = series.dtype()
    return dtype.is_integer()


def distinct_estimate(series: Series) raises -> Int:
    """The column's distinct-count estimate after appending its rows,
    or -1 for a type without statistics (Boolean,
    nested). Every 2,048 rows of each chunk give their first 614 rows
    (integers) or 204 (anything else) to the sketch; nulls count as
    sampled rows but are not hashed. The estimate is `u + u^3 (n - s) /
    s^3` for u distinct among s sampled of n rows (`GetCount`)."""
    if series._data.isa[BoolColumn]() or series.dtype().is_nested():
        return -1
    var take = _INTEGRAL_SAMPLE if _integral(series) else _BASE_SAMPLE
    var sketch = HyperLogLog()
    var sampled = 0
    var total = 0
    var parts = series.chunks() if series.is_chunked() else [series.copy()]
    for part in parts:
        var rows = len(part)
        var start = 0
        while start < rows:
            var window = min(_VECTOR, rows - start)
            var count = min(take, window)
            _sample(part, start, count, sketch)
            sampled += count
            total += window
            start += window
    if sampled == 0 or total == 0:
        return 0
    var u = Float64(min(sketch.count(), sampled))
    var s = Float64(sampled)
    var n = Float64(total)
    var unique_once = (u / s) * (u / s) * u
    var estimate = Int(u + unique_once / s * (n - s))
    return min(estimate, total)


def _sample(
    part: Series, start: Int, count: Int, mut sketch: HyperLogLog
) raises:
    """Hash rows [start, start + count) of one chunk into the sketch."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if part._data.isa[Column[Scalar[D]]]():
            ref column = part._data[Column[Scalar[D]]]
            for i in range(start, start + count):
                if column._valid(i):
                    sketch.insert(_hash_value[D](column._get(i)))
            return
    if part._data.isa[Column[Int128]]():
        ref column = part._data[Column[Int128]]
        for i in range(start, start + count):
            if column._valid(i):
                sketch.insert(_hash_value[DType.int128](column._get(i)))
        return
    if part._data.isa[StringColumn]():
        ref column = part._data[StringColumn]
        for i in range(start, start + count):
            if column._valid(i):
                sketch.insert(_hash_bytes(column._row_bytes(i)))
        return


@fieldwise_init
struct _ColumnCount(Copyable, Movable):
    var name: String
    # The column's storage when counted: a count belongs to these rows only.
    var identity: Int
    var rows: Int
    var count: Int


struct TableStatistics(Movable):
    """Distinct counts of one loaded table's columns, shared by every copy
    of the table and filled on first request (`TableStatistics`). Each
    count remembers the storage it was taken over, so a column replaced in
    a copy is counted again rather than read from the original."""

    var lock: Atomic[Int64]
    var counts: List[_ColumnCount]

    def __init__(out self):
        self.lock = Atomic[Int64](0)
        self.counts = List[_ColumnCount]()

    def _acquire(mut self):
        while True:
            var expected = Int64(0)
            if self.lock.compare_exchange(expected, 1):
                return

    def _release(mut self):
        self.lock.store(0)

    def distinct(mut self, series: Series) raises -> Int:
        var identity = _identity(series)
        var rows = len(series)
        self._acquire()
        for entry in self.counts:
            if (
                entry.name == series.name()
                and entry.identity == identity
                and entry.rows == rows
            ):
                var known = entry.count
                self._release()
                return known
        self._release()
        var count = distinct_estimate(series)
        self._acquire()
        self.counts.append(_ColumnCount(series.name(), identity, rows, count))
        self._release()
        return count


def _identity(series: Series) -> Int:
    """The address of the series' storage: its chunk list when chunked,
    else its payload buffer."""
    if series.is_chunked():
        return Int(series._chunked.value().ptr())
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            return Int(series._data[Column[Scalar[D]]]._data.ptr())
    if series._data.isa[Column[Int128]]():
        return Int(series._data[Column[Int128]]._data.ptr())
    if series._data.isa[StringColumn]():
        return Int(series._data[StringColumn]._offsets.ptr())
    return 0
