"""CSV record scanning primitives.

Identify LF terminators outside quote parity. Callers retain CR trimming and
CSV field validation. Chunk sizing uses a thread and width allocation budget.
"""
from std.bit import count_leading_zeros, pop_count
from .csv_bits import _mask64, _prefix_xor_inclusive


comptime _SIMD_WIDTH = 64
comptime _MAX_CHUNK = 16 * 1024 * 1024
comptime _MIN_CHUNK = 4 * 1024
comptime _ALLOCATION_BUDGET = 500_000


@fieldwise_init
struct CsvCount(Copyable):
    var rows: Int
    # Byte offset of the last valid LF, zero when no row is found.
    var last_newline: Int


@fieldwise_init
struct CsvFindNext(Copyable):
    var rows: Int
    var last_newline: Int
    # The doubled chunk-size hint for the next caller invocation.
    var chunk_size: Int


struct CountLines:
    var quote: UInt8
    var eol: UInt8
    var quoting: Bool

    def __init__(
        out self,
        quote: UInt8 = UInt8(34),
        quoting: Bool = True,
        eol: UInt8 = UInt8(10),
    ):
        self.quote = quote
        self.eol = eol
        self.quoting = quoting

    def count(self, bytes: Span[UInt8, ImmutAnyOrigin]) -> CsvCount:
        """Count complete records and return the last valid LF offset."""
        var total = 0
        var rows = 0
        var last = 0
        # True when the next SIMD block begins outside a quoted field.
        var outside_previous = True
        var n = len(bytes)
        # Polars leaves exactly one 64-byte block for its scalar tail.
        while total + _SIMD_WIDTH < n:
            var block = bytes.unsafe_ptr().unsafe_load[width=_SIMD_WIDTH](total)
            var eols = _mask64(
                block.eq(SIMD[DType.uint8, _SIMD_WIDTH](self.eol))
            )
            var valid = eols
            if self.quoting:
                var quotes = _mask64(
                    block.eq(SIMD[DType.uint8, _SIMD_WIDTH](self.quote))
                )
                var outside = _prefix_xor_inclusive(quotes)
                if outside_previous:
                    outside = ~outside
                outside_previous = (
                    outside & (UInt64(1) << UInt64(_SIMD_WIDTH - 1))
                ) != 0
                valid &= outside
            if valid != 0:
                rows += Int(pop_count(valid))
                last = total + _SIMD_WIDTH - 1 - Int(count_leading_zeros(valid))
            total += _SIMD_WIDTH
        var in_field = not outside_previous
        while total < n:
            var byte = bytes[total]
            if self.quoting and byte == self.quote:
                in_field = not in_field
            elif byte == self.eol and not in_field:
                rows += 1
                last = total
            total += 1
        return CsvCount(rows, last)

    def find_next(
        self, bytes: Span[UInt8, ImmutAnyOrigin], chunk_size: Int
    ) -> CsvFindNext:
        """Scan a chunk, doubling its hint until it contains an LF or EOF."""
        var hint = max(1, chunk_size)
        var n = len(bytes)
        while True:
            var width = min(hint, n)
            var found = self.count(bytes[0:width])
            if found.rows > 0 or width == n:
                return CsvFindNext(found.rows, found.last_newline, hint)
            if hint > Int.MAX // 2:
                hint = Int.MAX
            else:
                hint *= 2


def chunk_size(bytes: Int, threads: Int, projected_width: Int) -> Int:
    """Decode chunk hint: 4–16x threads, width cap, 4 KiB minimum."""
    var workers = max(1, threads)
    var width = max(1, projected_width)
    var allocation_limit = _ALLOCATION_BUDGET // width
    # Keep small decode ranges from becoming tiny physical Arrow chunks.
    # Grow from four to sixteen ranges per worker as the file grows, while
    # respecting the projected-width allocation budget.
    var parts_hint = min(workers * 16, max(allocation_limit, workers))
    if bytes < workers * 16 * 512 * 1024:
        var target_parts = max(workers * 4, max(0, bytes) // (512 * 1024))
        parts_hint = min(parts_hint, target_parts)
    var initial = min(max(0, bytes) // max(1, parts_hint), _MAX_CHUNK)
    return max(initial, _MIN_CHUNK)
