"""Sort rows by packing every key and the row index into one integer (#331).

The general sort compares row indices through per-key word lists, so each
comparison chases an index into every list, and a string key is ranked by
a sort of its own first. Here each row instead becomes one 64- or 128-bit
integer: its keys' order-preserving values, most significant key first,
above its row index. Integers sort natively, and since each holds its row,
they are distinct and their order is the stable sort's order -- ties in
every key fall back to row order in either direction.

The packing stays small by storing each key relative to its own range:
`value - min` (or `max - value` descending) takes only as many bits as the
column spans, so an Int64 key of 1..100,000 costs 17 bits, not 64. A key
with nulls or NaN adds a two-bit tier above its value (nulls first, value,
NaN, nulls last), which keeps both placements independent of direction as
the general path documents. A string key drops the prefix every value
shares and keeps its next 7 bytes and its remaining length, capped at 8:
exact for strings that end within those 7 bytes, and for longer strings a
prefix key whose ties are settled by comparing the strings themselves. That
settling needs the equal-key rows to be contiguous with nothing below the
string to reorder them, so a long string key must come last; any other
case, or a packing wider than 128 bits, returns None for the general path.

Sorting is a most-significant-digit bucket pass and then independent sorts.
The top bits of the packed value choose one of up to 4,096 buckets, rows
are scattered to their buckets in row order, and every bucket is then
sorted on its own, buckets being claimed dynamically since their sizes
vary. Because keys are stored relative to their range, the top bits are
well spread, and a bucket whose rows all share a key -- the whole column,
for a key with fewer values than there are buckets -- comes out of the
scatter already sorted and is only checked. A bucket too large to share a
worker is sorted in parallel runs that are then merged.
"""
from std.bit import count_leading_zeros
from std.memory import bitcast
from std.sys import bit_width_of

from .bool_column import BoolColumn
from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .parallel import Job, Pool, configured_workers
from .series import Series
from .string_column import StringColumn

# Tier codes as stored: a key with nulls or NaN orders tiers first.
comptime _NULLS_FIRST = 0
comptime _VALUE = 1
comptime _NAN = 2
comptime _NULLS_LAST = 3
# Per-row tier while extracting: 0 value, 1 NaN, 2 null.
comptime _ROW_VALUE = UInt8(0)
comptime _ROW_NAN = UInt8(1)
comptime _ROW_NULL = UInt8(2)

comptime _SIGN = UInt64(1) << 63
# At most this many bytes of a string are kept in its key after the shared
# prefix; the length below them is capped one past the bytes kept.
comptime _STRING_BYTES = 7
comptime _MAX_BUCKET_BITS = 12
# Below this many rows, one range and one bucket.
comptime _MIN_PARALLEL_ROWS = 1 << 14


@always_inline
def _bit_length(x: UInt64) -> Int:
    if x == 0:
        return 0
    return 64 - Int(count_leading_zeros(x))


@always_inline
def _order[D: DType](value: Scalar[D]) -> UInt64:
    """An unsigned integer ordered as the value is; -0.0 equals 0.0."""
    comptime if D.is_floating_point():
        var x = value.cast[DType.float64]()
        if x == 0:
            return _SIGN
        var bits = bitcast[DType.uint64](x)
        return ~bits if (bits & _SIGN) != 0 else bits ^ _SIGN
    elif D.is_unsigned():
        return value.cast[DType.uint64]()
    else:
        return value.cast[DType.int64]().cast[DType.uint64]() ^ _SIGN


@always_inline
def _string_order(
    bytes: Span[UInt8, ImmutAnyOrigin], prefix: Int, width: Int
) -> UInt64:
    """`width` bytes after `prefix`, big-endian and zero-padded, above the
    remaining length capped at width + 1. Zero padding with the length
    compared last is exact for strings that end within `width` bytes,
    including one that is a prefix of another and embedded NUL bytes."""
    var rest = len(bytes) - prefix
    var key = UInt64(0)
    var ptr = bytes.unsafe_ptr()
    for b in range(width):
        key <<= 8
        if b < rest:
            key |= UInt64(ptr[unsafe_offset=prefix + b])
    return (key << UInt64(_length_bits(width))) | UInt64(min(rest, width + 1))


@always_inline
def _length_bits(width: Int) -> Int:
    return _bit_length(UInt64(width + 1))


struct _Key(ImplicitlyCopyable, Movable):
    """One key's place in the packed integer."""

    var string: Bool
    var descending: Bool
    var nulls_last: Bool
    var tiers: Bool
    var low: UInt64
    var high: UInt64
    var bits: Int
    var shift: Int
    # Strings: the shared prefix length and whether any value is longer
    # than the key holds.
    var prefix: Int
    var width: Int
    var long: Bool

    def __init__(out self, string: Bool, descending: Bool, nulls_last: Bool):
        self.string = string
        self.descending = descending
        self.nulls_last = nulls_last
        self.tiers = False
        self.low = UInt64.MAX
        self.high = 0
        self.bits = 0
        self.shift = 0
        self.prefix = 0
        self.width = 0
        self.long = False

    @always_inline
    def pack[T: DType](self, order: UInt64, tier: UInt8) -> Scalar[T]:
        """This key's bits, tier above value, shifted into place. The tier
        is placed in the wide type: above a 63-bit value it would overflow
        a UInt64."""
        if tier == _ROW_VALUE:
            var value = (self.high - order) if self.descending else (
                order - self.low
            )
            var bits = Scalar[T](value) << Scalar[T](self.shift)
            if self.tiers:
                bits |= Scalar[T](_VALUE) << Scalar[T](self.shift + self.bits)
            return bits
        var code = _NAN
        if tier != _ROW_NAN:
            code = _NULLS_LAST if self.nulls_last else _NULLS_FIRST
        return Scalar[T](code) << Scalar[T](self.shift + self.bits)

    @always_inline
    def packed_bits(self) -> Int:
        return self.bits + (2 if self.tiers else 0)


struct _PrefixJob(Job):
    """Shared prefix with `reference` and the longest length over rows
    [first, last) of a string key."""

    var column: StringColumn
    var reference: List[UInt8]
    var first: Int
    var last: Int
    var nulls: Bool
    var prefix: Int
    var longest: Int

    def __init__(
        out self,
        column: StringColumn,
        var reference: List[UInt8],
        first: Int,
        last: Int,
        nulls: Bool,
    ):
        self.column = column.copy()
        self.prefix = len(reference)
        self.reference = reference^
        self.first = first
        self.last = last
        self.nulls = nulls
        self.longest = 0

    def run(mut self) raises:
        var ref_ptr = self.reference.unsafe_ptr()
        for i in range(self.first, self.last):
            if self.nulls and not self.column._valid(i):
                continue
            var bytes = self.column._row_bytes(i)
            var length = len(bytes)
            self.longest = max(self.longest, length)
            var limit = min(self.prefix, length)
            var ptr = bytes.unsafe_ptr()
            var p = 0
            while (
                p < limit and ptr[unsafe_offset=p] == ref_ptr[unsafe_offset=p]
            ):
                p += 1
            self.prefix = p


struct _OrderJob(Job):
    """Order values and tiers of one key over rows [first, last), with the
    range of its ordinary values."""

    var column: Series
    var orders: Int
    var tiers: Int
    var first: Int
    var last: Int
    var nulls: Bool
    var prefix: Int
    var width: Int
    var low: UInt64
    var high: UInt64
    var nan: Bool

    def __init__(
        out self,
        column: Series,
        orders: Int,
        tiers: Int,
        first: Int,
        last: Int,
        nulls: Bool,
        prefix: Int,
        width: Int,
    ):
        self.column = column.copy()
        self.orders = orders
        self.tiers = tiers
        self.first = first
        self.last = last
        self.nulls = nulls
        self.prefix = prefix
        self.width = width
        self.low = UInt64.MAX
        self.high = 0
        self.nan = False

    def run(mut self) raises:
        var order = Pointer[List[UInt64], MutAnyOrigin](
            unsafe_from_address=self.orders
        )[].unsafe_ptr()
        var tier = Pointer[List[UInt8], MutAnyOrigin](
            unsafe_from_address=self.tiers
        )[].unsafe_ptr()
        var low = self.low
        var high = self.high
        comptime for t in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[t]
            if self.column._data.isa[Column[Scalar[D]]]():
                ref typed = self.column._data[Column[Scalar[D]]]
                for i in range(self.first, self.last):
                    if self.nulls and not typed._valid(i):
                        tier[unsafe_offset=i] = _ROW_NULL
                        continue
                    var value = typed._get(i)
                    comptime if D.is_floating_point():
                        if value != value:
                            tier[unsafe_offset=i] = _ROW_NAN
                            self.nan = True
                            continue
                    var o = _order[D](value)
                    order[unsafe_offset=i] = o
                    tier[unsafe_offset=i] = _ROW_VALUE
                    low = min(low, o)
                    high = max(high, o)
                self.low = low
                self.high = high
                return
        if self.column._data.isa[BoolColumn]():
            ref typed = self.column._data[BoolColumn]
            for i in range(self.first, self.last):
                if self.nulls and not typed._valid(i):
                    tier[unsafe_offset=i] = _ROW_NULL
                    continue
                var o = UInt64(Int(typed._get(i)))
                order[unsafe_offset=i] = o
                tier[unsafe_offset=i] = _ROW_VALUE
                low = min(low, o)
                high = max(high, o)
        else:
            ref typed = self.column._data[StringColumn]
            for i in range(self.first, self.last):
                if self.nulls and not typed._valid(i):
                    tier[unsafe_offset=i] = _ROW_NULL
                    continue
                var o = _string_order(
                    typed._row_bytes(i), self.prefix, self.width
                )
                order[unsafe_offset=i] = o
                tier[unsafe_offset=i] = _ROW_VALUE
                low = min(low, o)
                high = max(high, o)
        self.low = low
        self.high = high


@always_inline
def _bucket[T: DType](packed: Scalar[T], shift: Int) -> Int:
    """The bucket of a packed value: its bits from `shift` up. A shift of
    the full width means one bucket; shifting by it would be undefined."""
    if shift >= bit_width_of[T]():
        return 0
    return Int(packed >> Scalar[T](shift))


struct _PackJob[T: DType](Job):
    """Pack rows [first, last) and count them per bucket."""

    var keys: List[_Key]
    var orders: List[Int]
    var tiers: List[Int]
    var packed: Int
    var first: Int
    var last: Int
    var bucket_shift: Int
    var counts: List[Int]

    def __init__(
        out self,
        keys: List[_Key],
        orders: List[Int],
        tiers: List[Int],
        packed: Int,
        first: Int,
        last: Int,
        bucket_shift: Int,
        buckets: Int,
    ):
        self.keys = keys.copy()
        self.orders = orders.copy()
        self.tiers = tiers.copy()
        self.packed = packed
        self.first = first
        self.last = last
        self.bucket_shift = bucket_shift
        self.counts = List[Int](length=buckets, fill=0)

    def run(mut self) raises:
        var out = Pointer[List[Scalar[Self.T]], MutAnyOrigin](
            unsafe_from_address=self.packed
        )[].unsafe_ptr()
        var counts = self.counts.unsafe_ptr()
        # One key is the common case; keep its loop free of the key list.
        if len(self.keys) == 1:
            var key = self.keys[0]
            var order = Pointer[List[UInt64], MutAnyOrigin](
                unsafe_from_address=self.orders[0]
            )[].unsafe_ptr()
            var tier = Pointer[List[UInt8], MutAnyOrigin](
                unsafe_from_address=self.tiers[0]
            )[].unsafe_ptr()
            for i in range(self.first, self.last):
                var p = key.pack[Self.T](
                    order[unsafe_offset=i], tier[unsafe_offset=i]
                ) | Scalar[Self.T](i)
                out[unsafe_offset=i] = p
                counts[unsafe_offset=_bucket(p, self.bucket_shift)] += 1
            return
        var order_ptrs = List[Pointer[UInt64, MutAnyOrigin]]()
        var tier_ptrs = List[Pointer[UInt8, MutAnyOrigin]]()
        for k in range(len(self.keys)):
            order_ptrs.append(
                Pointer[List[UInt64], MutAnyOrigin](
                    unsafe_from_address=self.orders[k]
                )[].unsafe_ptr()
            )
            tier_ptrs.append(
                Pointer[List[UInt8], MutAnyOrigin](
                    unsafe_from_address=self.tiers[k]
                )[].unsafe_ptr()
            )
        for i in range(self.first, self.last):
            var p = Scalar[Self.T](i)
            for k in range(len(self.keys)):
                p |= self.keys[k].pack[Self.T](
                    order_ptrs[k][unsafe_offset=i],
                    tier_ptrs[k][unsafe_offset=i],
                )
            out[unsafe_offset=i] = p
            counts[unsafe_offset=_bucket(p, self.bucket_shift)] += 1


struct _ScatterJob[T: DType](Job):
    """Move rows [first, last) to their buckets, in row order."""

    var source: Int
    var target: Int
    var first: Int
    var last: Int
    var bucket_shift: Int
    var next: List[Int]

    def __init__(
        out self,
        source: Int,
        target: Int,
        first: Int,
        last: Int,
        bucket_shift: Int,
        var next: List[Int],
    ):
        self.source = source
        self.target = target
        self.first = first
        self.last = last
        self.bucket_shift = bucket_shift
        self.next = next^

    def run(mut self) raises:
        var src = Pointer[List[Scalar[Self.T]], MutAnyOrigin](
            unsafe_from_address=self.source
        )[].unsafe_ptr()
        var dst = Pointer[List[Scalar[Self.T]], MutAnyOrigin](
            unsafe_from_address=self.target
        )[].unsafe_ptr()
        var next = self.next.unsafe_ptr()
        for i in range(self.first, self.last):
            var p = src[unsafe_offset=i]
            var b = _bucket(p, self.bucket_shift)
            dst[unsafe_offset=next[unsafe_offset=b]] = p
            next[unsafe_offset=b] += 1


def _is_sorted[
    T: DType
](data: Pointer[Scalar[T], MutAnyOrigin], start: Int, end: Int) -> Bool:
    for i in range(start + 1, end):
        if data[unsafe_offset=i] < data[unsafe_offset=i - 1]:
            return False
    return True


struct _Settle(ImplicitlyCopyable, Movable):
    """How to settle ties left by a long string key, which is the last key:
    a run whose packed bits above the row agree, and whose length byte says
    its strings continue past the key, is re-sorted by the strings."""

    var active: Bool
    var column: Int
    var descending: Bool
    var low: UInt64
    var high: UInt64
    var shift: Int
    var bits: Int
    var tiers: Bool
    var row_bits: Int
    var width: Int

    def __init__(out self):
        self.active = False
        self.column = 0
        self.descending = False
        self.low = 0
        self.high = 0
        self.shift = 0
        self.bits = 0
        self.tiers = False
        self.row_bits = 0
        self.width = 0

    def is_long[T: DType](self, packed: Scalar[T]) -> Bool:
        """Whether this row's string continues past its key."""
        var value_mask = (Scalar[T](1) << Scalar[T](self.bits)) - 1
        if self.tiers:
            var code = Int(
                (packed >> Scalar[T](self.shift + self.bits)) & Scalar[T](3)
            )
            if code != _VALUE:
                return False
        var field = UInt64((packed >> Scalar[T](self.shift)) & value_mask)
        var order = (self.high - field) if self.descending else (
            self.low + field
        )
        var length_mask = (UInt64(1) << UInt64(_length_bits(self.width))) - 1
        return (order & length_mask) == UInt64(self.width + 1)


def _settle_runs[
    T: DType
](
    data: Pointer[Scalar[T], MutAnyOrigin],
    start: Int,
    end: Int,
    settle: _Settle,
) raises:
    ref column = Pointer[StringColumn, MutAnyOrigin](
        unsafe_from_address=settle.column
    )[]
    var row_bits = Scalar[T](settle.row_bits)
    var row_mask = (Scalar[T](1) << row_bits) - 1
    var descending = settle.descending

    def less(a: Int, b: Int) {imm column, imm descending} -> Bool:
        var x = column._get(a)
        var y = column._get(b)
        if x != y:
            return (y < x) if descending else (x < y)
        return a < b

    var i = start
    while i < end:
        var head = data[unsafe_offset=i] >> row_bits
        var j = i + 1
        while j < end and (data[unsafe_offset=j] >> row_bits) == head:
            j += 1
        if j - i > 1 and settle.is_long[T](data[unsafe_offset=i]):
            var rows = List[Int](capacity=j - i)
            for k in range(i, j):
                rows.append(Int(data[unsafe_offset=k] & row_mask))
            sort(rows, less)
            var high = data[unsafe_offset=i] & ~row_mask
            for k in range(len(rows)):
                data[unsafe_offset=i + k] = high | Scalar[T](rows[k])
        i = j


struct _BucketJob[T: DType](Job):
    """Sort buckets [first, last) in place."""

    var data: Int
    var offsets: Int
    var first: Int
    var last: Int
    var settle: _Settle

    def __init__(
        out self,
        data: Int,
        offsets: Int,
        first: Int,
        last: Int,
        settle: _Settle,
    ):
        self.data = data
        self.offsets = offsets
        self.first = first
        self.last = last
        self.settle = settle

    def run(mut self) raises:
        ref data = Pointer[List[Scalar[Self.T]], MutAnyOrigin](
            unsafe_from_address=self.data
        )[]
        var ptr = data.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        ref offsets = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.offsets
        )[]
        for b in range(self.first, self.last):
            var start = offsets[b]
            var end = offsets[b + 1]
            if end - start > 1 and not _is_sorted[Self.T](ptr, start, end):
                sort(Span(data)[start:end])
            if self.settle.active:
                _settle_runs[Self.T](ptr, start, end, self.settle)


struct _RunJob[T: DType](Job):
    """Sort data[start, end) in place: one run of a heavy bucket."""

    var data: Int
    var start: Int
    var end: Int

    def __init__(out self, data: Int, start: Int, end: Int):
        self.data = data
        self.start = start
        self.end = end

    def run(mut self) raises:
        ref data = Pointer[List[Scalar[Self.T]], MutAnyOrigin](
            unsafe_from_address=self.data
        )[]
        sort(Span(data)[self.start : self.end])


struct _MergeJob[T: DType](Job):
    """Outputs [first, last) of merging sorted runs [start, mid) and
    [mid, end). Values are distinct, so co-ranking needs no tie rule and one
    merge splits across every worker."""

    var source: Int
    var target: Int
    var start: Int
    var mid: Int
    var end: Int
    var first: Int
    var last: Int

    def __init__(
        out self,
        source: Int,
        target: Int,
        start: Int,
        mid: Int,
        end: Int,
        first: Int,
        last: Int,
    ):
        self.source = source
        self.target = target
        self.start = start
        self.mid = mid
        self.end = end
        self.first = first
        self.last = last

    def run(mut self) raises:
        var src = Pointer[List[Scalar[Self.T]], MutAnyOrigin](
            unsafe_from_address=self.source
        )[].unsafe_ptr()
        var dst = Pointer[List[Scalar[Self.T]], MutAnyOrigin](
            unsafe_from_address=self.target
        )[].unsafe_ptr()
        # How many of the first k outputs come from the left run.
        var k = self.first - self.start
        var lo = max(0, k - (self.end - self.mid))
        var hi = min(k, self.mid - self.start)
        while lo < hi:
            var h = (lo + hi) // 2
            if (
                src[unsafe_offset=self.start + h]
                < src[unsafe_offset=self.mid + k - h - 1]
            ):
                lo = h + 1
            else:
                hi = h
        var i = self.start + lo
        var j = self.mid + (k - lo)
        for at in range(self.first, self.last):
            if j >= self.end or (
                i < self.mid and src[unsafe_offset=i] < src[unsafe_offset=j]
            ):
                dst[unsafe_offset=at] = src[unsafe_offset=i]
                i += 1
            else:
                dst[unsafe_offset=at] = src[unsafe_offset=j]
                j += 1


def _sort_heavy[
    T: DType
](
    mut pool: Pool,
    mut data: List[Scalar[T]],
    mut scratch: List[Scalar[T]],
    start: Int,
    end: Int,
    workers: Int,
) raises:
    """Sort data[start, end) on every worker: runs, then merge rounds whose
    merges are split so each round keeps all workers busy."""
    var n = end - start
    var bounds = List[Int]()
    for w in range(workers + 1):
        bounds.append(start + n * w // workers)
    var runs = List[_RunJob[T]]()
    for w in range(workers):
        runs.append(_RunJob[T](Int(Pointer(to=data)), bounds[w], bounds[w + 1]))
    pool.run(runs)
    var in_data = True
    while len(bounds) > 2:
        var source = Int(Pointer(to=data)) if in_data else Int(
            Pointer(to=scratch)
        )
        var target = Int(Pointer(to=scratch)) if in_data else Int(
            Pointer(to=data)
        )
        var merges = List[_MergeJob[T]]()
        var next = List[Int]()
        var pairs = (len(bounds) - 1 + 1) // 2
        var pieces = max(1, workers // pairs)
        var k = 0
        while k + 1 < len(bounds):
            var lo = bounds[k]
            var mid = bounds[k + 1]
            var hi = bounds[k + 2] if k + 2 < len(bounds) else mid
            for p in range(pieces):
                merges.append(
                    _MergeJob[T](
                        source,
                        target,
                        lo,
                        mid,
                        hi,
                        lo + (hi - lo) * p // pieces,
                        lo + (hi - lo) * (p + 1) // pieces,
                    )
                )
            next.append(lo)
            k += 2
        next.append(end)
        pool.run(merges)
        bounds = next^
        in_data = not in_data
    if not in_data:
        var dst = data.unsafe_ptr()
        var src = scratch.unsafe_ptr()
        for i in range(start, end):
            dst[unsafe_offset=i] = src[unsafe_offset=i]


struct _RowsJob[T: DType](Job):
    """Unpack the row index of positions [first, last)."""

    var packed: Int
    var rows: Int
    var first: Int
    var last: Int
    var mask: Scalar[Self.T]

    def __init__(
        out self,
        packed: Int,
        rows: Int,
        first: Int,
        last: Int,
        mask: Scalar[Self.T],
    ):
        self.packed = packed
        self.rows = rows
        self.first = first
        self.last = last
        self.mask = mask

    def run(mut self) raises:
        var src = Pointer[List[Scalar[Self.T]], MutAnyOrigin](
            unsafe_from_address=self.packed
        )[].unsafe_ptr()
        var dst = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.rows
        )[].unsafe_ptr()
        for i in range(self.first, self.last):
            dst[unsafe_offset=i] = Int(src[unsafe_offset=i] & self.mask)


def _sort_packed[
    T: DType
](
    mut pool: Pool,
    workers: Int,
    bounds: List[Int],
    keys: List[_Key],
    orders: List[Int],
    tiers: List[Int],
    n: Int,
    total_bits: Int,
    row_bits: Int,
    settle: _Settle,
) raises -> List[Int]:
    var ranges = len(bounds) - 1
    var bucket_bits = 0
    if n >= _MIN_PARALLEL_ROWS:
        bucket_bits = min(_MAX_BUCKET_BITS, _bit_length(UInt64(n)) - 6)
    bucket_bits = min(bucket_bits, total_bits)
    var buckets = 1 << bucket_bits
    var bucket_shift = total_bits - bucket_bits

    var packed = List[Scalar[T]](length=n, fill=0)
    var packs = List[_PackJob[T]](capacity=ranges)
    for r in range(ranges):
        packs.append(
            _PackJob[T](
                keys,
                orders,
                tiers,
                Int(Pointer(to=packed)),
                bounds[r],
                bounds[r + 1],
                bucket_shift,
                buckets,
            )
        )
    pool.run(packs)

    # Bucket b starts at offsets[b]; range r writes its share of b after
    # the earlier ranges' shares, which keeps each bucket in row order.
    var offsets = List[Int](length=buckets + 1, fill=0)
    for b in range(buckets):
        var size = 0
        for r in range(ranges):
            size += packs[r].counts[b]
        offsets[b + 1] = offsets[b] + size
    var sorted = List[Scalar[T]](length=n, fill=0)
    var scatters = List[_ScatterJob[T]](capacity=ranges)
    var running = offsets.copy()
    for r in range(ranges):
        var next = List[Int](capacity=buckets)
        for b in range(buckets):
            next.append(running[b])
            running[b] += packs[r].counts[b]
        scatters.append(
            _ScatterJob[T](
                Int(Pointer(to=packed)),
                Int(Pointer(to=sorted)),
                bounds[r],
                bounds[r + 1],
                bucket_shift,
                next^,
            )
        )
    pool.run(scatters)

    # Buckets small enough to share a worker are grouped into tasks of
    # about equal rows and claimed dynamically; the rest are heavy.
    var heavy_rows = max(n // max(workers, 1), 1 << 16)
    var task_rows = max(n // max(4 * workers, 1), 4096)
    var tasks = List[_BucketJob[T]]()
    var heavy = List[Int]()
    var b = 0
    while b < buckets:
        var size = offsets[b + 1] - offsets[b]
        if workers > 1 and size > heavy_rows:
            heavy.append(b)
            b += 1
            continue
        var first = b
        var rows = 0
        while b < buckets and rows < task_rows:
            var s = offsets[b + 1] - offsets[b]
            if workers > 1 and s > heavy_rows:
                break
            rows += s
            b += 1
        tasks.append(
            _BucketJob[T](
                Int(Pointer(to=sorted)),
                Int(Pointer(to=offsets)),
                first,
                b,
                settle,
            )
        )
    pool.run(tasks, claim=True)
    for h in heavy:
        var start = offsets[h]
        var end = offsets[h + 1]
        if not _is_sorted[T](
            sorted.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), start, end
        ):
            _sort_heavy[T](pool, sorted, packed, start, end, workers)
        if settle.active:
            _settle_runs[T](
                sorted.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                start,
                end,
                settle,
            )

    var rows = List[Int](length=n, fill=0)
    var mask = (Scalar[T](1) << Scalar[T](row_bits)) - 1
    var unpack = List[_RowsJob[T]](capacity=ranges)
    for r in range(ranges):
        unpack.append(
            _RowsJob[T](
                Int(Pointer(to=sorted)),
                Int(Pointer(to=rows)),
                bounds[r],
                bounds[r + 1],
                mask,
            )
        )
    pool.run(unpack)
    _ = packed^
    _ = sorted^
    _ = offsets^
    return rows^


@always_inline
def _pack_row[
    T: DType
](
    keys: List[_Key],
    order_ptrs: List[Pointer[UInt64, MutAnyOrigin]],
    tier_ptrs: List[Pointer[UInt8, MutAnyOrigin]],
    i: Int,
) -> Scalar[T]:
    var p = Scalar[T](i)
    for k in range(len(keys)):
        p |= keys[k].pack[T](
            order_ptrs[k][unsafe_offset=i], tier_ptrs[k][unsafe_offset=i]
        )
    return p


def _key_pointers(
    orders: List[Int], tiers: List[Int]
) -> Tuple[
    List[Pointer[UInt64, MutAnyOrigin]],
    List[Pointer[UInt8, MutAnyOrigin]],
]:
    var order_ptrs = List[Pointer[UInt64, MutAnyOrigin]]()
    var tier_ptrs = List[Pointer[UInt8, MutAnyOrigin]]()
    for k in range(len(orders)):
        order_ptrs.append(
            Pointer[List[UInt64], MutAnyOrigin](
                unsafe_from_address=orders[k]
            )[].unsafe_ptr()
        )
        tier_ptrs.append(
            Pointer[List[UInt8], MutAnyOrigin](
                unsafe_from_address=tiers[k]
            )[].unsafe_ptr()
        )
    return (order_ptrs^, tier_ptrs^)


struct _SelectJob[T: DType](Job):
    """The `limit` smallest packed rows of [first, last), ascending.

    Accepted values collect in a buffer of twice the limit; when it fills,
    it is sorted and cut back to the limit, whose largest value becomes the
    bar a row must beat. Most rows of a large range fail that one
    comparison, so selection costs about one pass plus O(k log k) per k
    accepted rows.
    """

    var keys: List[_Key]
    var orders: List[Int]
    var tiers: List[Int]
    var first: Int
    var last: Int
    var limit: Int
    var best: List[Scalar[Self.T]]

    def __init__(
        out self,
        keys: List[_Key],
        orders: List[Int],
        tiers: List[Int],
        first: Int,
        last: Int,
        limit: Int,
    ):
        self.keys = keys.copy()
        self.orders = orders.copy()
        self.tiers = tiers.copy()
        self.first = first
        self.last = last
        self.limit = limit
        self.best = List[Scalar[Self.T]]()

    def run(mut self) raises:
        var pointers = _key_pointers(self.orders, self.tiers)
        var buffer = List[Scalar[Self.T]](capacity=2 * self.limit + 1)
        var bar = Scalar[Self.T].MAX
        var full = False
        for i in range(self.first, self.last):
            var p = _pack_row[Self.T](self.keys, pointers[0], pointers[1], i)
            if full and p >= bar:
                continue
            buffer.append(p)
            if len(buffer) >= 2 * self.limit:
                sort(buffer)
                buffer.shrink(self.limit)
                bar = buffer[self.limit - 1]
                full = True
        sort(buffer)
        if len(buffer) > self.limit:
            buffer.shrink(self.limit)
        self.best = buffer^


struct _TiedJob[T: DType](Job):
    """Rows of [first, last) whose packed bits above the row equal `head`:
    the rows tied with the boundary of a selection whose ties still need
    settling by the strings."""

    var keys: List[_Key]
    var orders: List[Int]
    var tiers: List[Int]
    var first: Int
    var last: Int
    var head: Scalar[Self.T]
    var row_bits: Int
    var tied: List[Scalar[Self.T]]

    def __init__(
        out self,
        keys: List[_Key],
        orders: List[Int],
        tiers: List[Int],
        first: Int,
        last: Int,
        head: Scalar[Self.T],
        row_bits: Int,
    ):
        self.keys = keys.copy()
        self.orders = orders.copy()
        self.tiers = tiers.copy()
        self.first = first
        self.last = last
        self.head = head
        self.row_bits = row_bits
        self.tied = List[Scalar[Self.T]]()

    def run(mut self) raises:
        var pointers = _key_pointers(self.orders, self.tiers)
        var shift = Scalar[Self.T](self.row_bits)
        for i in range(self.first, self.last):
            var p = _pack_row[Self.T](self.keys, pointers[0], pointers[1], i)
            if (p >> shift) == self.head:
                self.tied.append(p)


def _select_packed[
    T: DType
](
    mut pool: Pool,
    bounds: List[Int],
    keys: List[_Key],
    orders: List[Int],
    tiers: List[Int],
    row_bits: Int,
    settle: _Settle,
    limit: Int,
) raises -> List[Int]:
    """The first `limit` rows of the packed order.

    Each range keeps its `limit` smallest packed values; the answer is
    among their union, since a row among the global first `limit` is among
    its own range's. When a long string key leaves ties to settle, rows
    whose keys equal the boundary row's may be missing from the union, so
    they are collected in a second pass and settled with the rest.
    """
    if limit == 0:
        return List[Int]()
    var ranges = len(bounds) - 1
    var jobs = List[_SelectJob[T]](capacity=ranges)
    for r in range(ranges):
        jobs.append(
            _SelectJob[T](keys, orders, tiers, bounds[r], bounds[r + 1], limit)
        )
    pool.run(jobs)
    var candidates = List[Scalar[T]]()
    for r in range(ranges):
        candidates.extend(Span(jobs[r].best))
    sort(candidates)
    var row_mask = (Scalar[T](1) << Scalar[T](row_bits)) - 1
    if settle.active and len(candidates) > 0:
        var take = min(limit, len(candidates))
        var shift = Scalar[T](row_bits)
        var head = candidates[take - 1] >> shift
        if settle.is_long[T](candidates[take - 1]):
            var tied_jobs = List[_TiedJob[T]](capacity=ranges)
            for r in range(ranges):
                tied_jobs.append(
                    _TiedJob[T](
                        keys,
                        orders,
                        tiers,
                        bounds[r],
                        bounds[r + 1],
                        head,
                        row_bits,
                    )
                )
            pool.run(tied_jobs)
            var merged = List[Scalar[T]]()
            for c in candidates:
                if (c >> shift) < head:
                    merged.append(c)
            for r in range(ranges):
                merged.extend(Span(tied_jobs[r].tied))
            sort(merged)
            candidates = merged^
        _settle_runs[T](
            candidates.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            0,
            len(candidates),
            settle,
        )
    var rows = List[Int](capacity=min(limit, len(candidates)))
    for i in range(min(limit, len(candidates))):
        rows.append(Int(candidates[i] & row_mask))
    return rows^


def packed_arg_sort(
    columns: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
) raises -> Optional[List[Int]]:
    """Row order of the stable sort by `columns`, or None when the keys do
    not pack (see the module docstring); the caller then sorts generally.
    """
    return _packed_order(columns, descending, nulls_last, -1)


def packed_top_rows(
    columns: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    k: Int,
    threads: Int = 0,
) raises -> Optional[List[Int]]:
    """The first `k` rows of `packed_arg_sort`'s order, selected in about
    one pass instead of sorting every row; None when the keys do not pack.
    `threads` caps the workers (0: the configured count), for callers that
    already run on a worker.
    """
    if k < 0:
        raise Error("k must be nonnegative")
    return _packed_order(columns, descending, nulls_last, k, threads)


def _packed_order(
    columns: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    limit: Int,
    threads: Int = 0,
) raises -> Optional[List[Int]]:
    """The sorted row order, or its first `limit` rows when `limit` is not
    negative and smaller than the row count."""
    var n = len(columns[0])
    for column in columns:
        var dtype = column.dtype()
        if dtype.is_decimal() or dtype.is_nested():
            return None
        var supported = (
            column._data.isa[BoolColumn]() or column._data.isa[StringColumn]()
        )
        comptime for t in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[t]
            if column._data.isa[Column[Scalar[D]]]():
                supported = True
        if not supported and not column.is_chunked():
            return None
    var workers = configured_workers() if n >= _MIN_PARALLEL_ROWS else 1
    if threads > 0:
        workers = min(workers, threads)
    var ranges = max(1, min(workers, n // 8192))
    var bounds = List[Int](capacity=ranges + 1)
    for r in range(ranges + 1):
        bounds.append(n * r // ranges)
    var row_bits = max(1, _bit_length(UInt64(max(n - 1, 0))))

    var pool = Pool(workers)
    var keys = List[_Key]()
    var owned = List[Series]()
    var order_lists = List[List[UInt64]]()
    var tier_lists = List[List[UInt8]]()
    for k in range(len(columns)):
        var column = (
            columns[k]
            .rechunk() if columns[k]
            .is_chunked() else columns[k]
            .copy()
        )
        if not (
            column._data.isa[BoolColumn]() or column._data.isa[StringColumn]()
        ):
            var numeric = False
            comptime for t in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[t]
                if column._data.isa[Column[Scalar[D]]]():
                    numeric = True
            if not numeric:
                pool.release()
                return None
        owned.append(column^)
        order_lists.append(List[UInt64](length=n, fill=0))
        tier_lists.append(List[UInt8](length=n, fill=0))
    # Addresses are taken only once the lists stop moving.
    var orders = List[Int]()
    var tiers = List[Int]()
    for k in range(len(owned)):
        orders.append(Int(Pointer(to=order_lists[k])))
        tiers.append(Int(Pointer(to=tier_lists[k])))

    for k in range(len(owned)):
        ref column = owned[k]
        var nulls = column.null_count() > 0
        var string = column._data.isa[StringColumn]()
        var key = _Key(string, descending[k], nulls_last[k])
        if string:
            ref typed = column._data[StringColumn]
            var reference = List[UInt8]()
            for i in range(n):
                if not nulls or typed._valid(i):
                    reference.extend(typed._row_bytes(i))
                    break
            var jobs = List[_PrefixJob](capacity=ranges)
            for r in range(ranges):
                jobs.append(
                    _PrefixJob(
                        typed, reference.copy(), bounds[r], bounds[r + 1], nulls
                    )
                )
            pool.run(jobs)
            var prefix = len(reference)
            var longest = 0
            for r in range(len(jobs)):
                prefix = min(prefix, jobs[r].prefix)
                longest = max(longest, jobs[r].longest)
            key.prefix = prefix
            key.width = min(longest - prefix, _STRING_BYTES)
            key.long = longest - prefix > _STRING_BYTES
            if key.long and k != len(owned) - 1:
                pool.release()
                return None
        var jobs = List[_OrderJob](capacity=ranges)
        for r in range(ranges):
            jobs.append(
                _OrderJob(
                    column,
                    orders[k],
                    tiers[k],
                    bounds[r],
                    bounds[r + 1],
                    nulls,
                    key.prefix,
                    key.width,
                )
            )
        pool.run(jobs)
        var nan = False
        for r in range(len(jobs)):
            key.low = min(key.low, jobs[r].low)
            key.high = max(key.high, jobs[r].high)
            nan = nan or jobs[r].nan
        if key.low > key.high:
            key.low = 0
            key.high = 0
        key.bits = _bit_length(key.high - key.low)
        key.tiers = nulls or nan
        keys.append(key)

    var shift = row_bits
    for k in reversed(range(len(keys))):
        keys[k].shift = shift
        shift += keys[k].packed_bits()
    var total_bits = shift

    var settle = _Settle()
    ref last = keys[len(keys) - 1]
    if last.string and last.long:
        settle.active = True
        settle.column = Int(
            Pointer(to=owned[len(owned) - 1]._data[StringColumn])
        )
        settle.descending = last.descending
        settle.low = last.low
        settle.high = last.high
        settle.shift = last.shift
        settle.bits = last.bits
        settle.tiers = last.tiers
        settle.row_bits = row_bits
        settle.width = last.width

    var result: Optional[List[Int]] = None
    var select = limit >= 0 and limit < n
    if total_bits <= 64:
        if select:
            result = _select_packed[DType.uint64](
                pool, bounds, keys, orders, tiers, row_bits, settle, limit
            )
        else:
            result = _sort_packed[DType.uint64](
                pool,
                workers,
                bounds,
                keys,
                orders,
                tiers,
                n,
                total_bits,
                row_bits,
                settle,
            )
    elif total_bits <= 128:
        if select:
            result = _select_packed[DType.uint128](
                pool, bounds, keys, orders, tiers, row_bits, settle, limit
            )
        else:
            result = _sort_packed[DType.uint128](
                pool,
                workers,
                bounds,
                keys,
                orders,
                tiers,
                n,
                total_bits,
                row_bits,
                settle,
            )
    pool.release()
    _ = order_lists^
    _ = tier_lists^
    _ = owned^
    return result^


# Rows per chunk when a large selection is split across workers.
comptime _SELECT_CHUNK = 1 << 16


struct _ChunkTopJob(Job):
    """The first `limit` rows of one chunk of rows, as whole-column rows."""

    var columns: List[Series]
    var descending: List[Bool]
    var nulls_last: List[Bool]
    var start: Int
    var length: Int
    var limit: Int
    var rows: List[Int]
    var packed: Bool

    def __init__(
        out self,
        columns: List[Series],
        descending: List[Bool],
        nulls_last: List[Bool],
        start: Int,
        length: Int,
        limit: Int,
    ):
        self.columns = columns.copy()
        self.descending = descending.copy()
        self.nulls_last = nulls_last.copy()
        self.start = start
        self.length = length
        self.limit = limit
        self.rows = List[Int]()
        self.packed = False

    def run(mut self) raises:
        var chunk = List[Series](capacity=len(self.columns))
        for column in self.columns:
            chunk.append(column.slice(self.start, self.length))
        var rows = packed_top_rows(
            chunk, self.descending, self.nulls_last, self.limit, threads=1
        )
        if not rows:
            return
        self.packed = True
        for row in rows.value():
            self.rows.append(self.start + row)


def chunked_top_rows(
    columns: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    k: Int,
) raises -> Optional[List[Int]]:
    """`packed_top_rows` for a large input: each 65,536-row chunk selects
    its own first `k` rows on one worker, then the first `k` of those
    candidates are selected again. A row among the overall first `k` is
    among its chunk's, and candidates are taken in row order, so ties keep
    the stable order. Chunks fit in cache, where packing the whole column
    first would stream it through memory twice; streaming execution does
    the same per batch. None when the keys do not pack."""
    if k < 0:
        raise Error("k must be nonnegative")
    var n = len(columns[0])
    var chunks = (n + _SELECT_CHUNK - 1) // _SELECT_CHUNK
    var workers = configured_workers()
    if chunks < 4 or workers <= 1:
        return packed_top_rows(columns, descending, nulls_last, k)
    var jobs = List[_ChunkTopJob](capacity=chunks)
    for c in range(chunks):
        var start = c * _SELECT_CHUNK
        jobs.append(
            _ChunkTopJob(
                columns,
                descending,
                nulls_last,
                start,
                min(_SELECT_CHUNK, n - start),
                k,
            )
        )
    var pool = Pool(min(workers, chunks))
    pool.run(jobs, claim=True)
    pool.release()
    var candidates = List[Int]()
    for c in range(chunks):
        if not jobs[c].packed:
            return None
        candidates.extend(Span(jobs[c].rows))
    sort(candidates)
    var gathered = List[Series](capacity=len(columns))
    for column in columns:
        gathered.append(column.take(candidates))
    var picked = packed_top_rows(gathered, descending, nulls_last, k, threads=1)
    if not picked:
        return None
    var rows = List[Int](capacity=len(picked.value()))
    for i in picked.value():
        rows.append(candidates[i])
    return rows^
