"""Borrowed CSV field splitting.

* ``CsvSplitFields.next``: a plain field scans
  for separator/LF, while a field beginning with the quote character runs a
  quote-parity scan and returns the raw field with ``needs_escaping=True``.
* ``_prefix_xor_inclusive`` computes inclusive quote-prefix parity.
  ``cached_ends`` is its relative end-position cache, shifted after each field.
* ``CsvFieldSpan.bytes`` is the Mojo borrowing handoff. It yields a view into
  caller-owned bytes and never copies the field.

This primitive returns offsets, keeps a CR before LF for the decoder to trim,
and leaves quote validation and unescaping to the typed CSV buffers. The
splitter only recognizes
quote parity for fields that start with a quote.
"""
from std.bit import count_trailing_zeros
from .csv_bits import _mask64, _prefix_xor_inclusive


comptime _SIMD_WIDTH = 64


@fieldwise_init
struct CsvFieldSpan(Copyable):
    """A zero-copy field range in the caller-owned input buffer."""

    var start: Int
    var end: Int
    var needs_escaping: Bool
    var ends_record: Bool
    # Separator, LF, or zero when this field ends at EOF.
    var terminator: UInt8

    def bytes(
        self, input: Span[UInt8, ImmutAnyOrigin]
    ) -> Span[UInt8, ImmutAnyOrigin]:
        return input[self.start : self.end]

    @always_inline
    def _unsafe_bytes(
        self, input: Span[UInt8, ImmutAnyOrigin]
    ) -> Span[UInt8, ImmutAnyOrigin]:
        """Borrow an iterator-produced field without rechecking its range.

        `CsvSplitFields.next` establishes `0 <= start <= end <= len(input)`
        before it returns a span. Keep `bytes` above as the checked API for
        independently constructed spans.
        """
        return Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=input.unsafe_ptr().unsafe_offset(self.start),
            length=self.end - self.start,
        )


struct CsvSplitFields:
    """Iterate fields in one complete CSV record or byte range.

    This is intentionally a tokenizer primitive. It accepts Polars'
    quote-parity splitting behavior and reports the raw quoted field; callers
    enforcing stricter CSV grammar validate the returned span before decoding.
    """

    var separator: UInt8
    var quote: UInt8
    var quoting: Bool
    # Polars stores these SIMD splats in `SplitFields`, rather than rebuilding
    # them in every 64-byte block scan.
    var simd_separator: SIMD[DType.uint8, _SIMD_WIDTH]
    var simd_eol: SIMD[DType.uint8, _SIMD_WIDTH]
    var simd_quote: SIMD[DType.uint8, _SIMD_WIDTH]
    var position: Int
    var finished: Bool
    # `previous_valid_ends` from Polars. Bit zero is relative to `position`;
    # consuming an end shifts the remaining mask into the next field's frame.
    var cached_ends: UInt64

    def __init__(
        out self,
        separator: UInt8,
        quote: UInt8 = UInt8(34),
        quoting: Bool = True,
    ):
        self.separator = separator
        self.quote = quote
        self.quoting = quoting
        self.simd_separator = SIMD[DType.uint8, _SIMD_WIDTH](separator)
        self.simd_eol = SIMD[DType.uint8, _SIMD_WIDTH](10)
        self.simd_quote = SIMD[DType.uint8, _SIMD_WIDTH](quote)
        self.position = 0
        self.finished = False
        self.cached_ends = 0

    @always_inline
    def _cached_end(mut self) -> Int:
        if self.cached_ends == 0:
            return -1
        var offset = Int(count_trailing_zeros(self.cached_ends))
        # Match `previous_valid_ends >>= pos + 1` in SplitFields::next.
        self.cached_ends >>= UInt64(offset + 1)
        return self.position + offset

    def consumed(self) -> Int:
        """Bytes consumed from the input after the most recent field."""
        return self.position

    @always_inline
    def _plain_end(
        mut self, input: Span[UInt8, ImmutAnyOrigin]
    ) -> CsvFieldSpan:
        var n = len(input)
        var i = self.position
        while i + _SIMD_WIDTH < n:
            var block = input.unsafe_ptr().unsafe_load[width=_SIMD_WIDTH](i)
            var mask = _mask64(
                block.eq(self.simd_separator) | block.eq(self.simd_eol)
            )
            if mask != 0:
                var end = i + Int(count_trailing_zeros(mask))
                var record = input[end] == 10
                var result = CsvFieldSpan(
                    self.position, end, False, record, input[end]
                )
                self.position = end + 1
                if record:
                    self.finished = True
                return result^
            i += _SIMD_WIDTH
        while i < n:
            if input[i] == self.separator or input[i] == 10:
                var record = input[i] == 10
                var result = CsvFieldSpan(
                    self.position, i, False, record, input[i]
                )
                self.position = i + 1
                if record:
                    self.finished = True
                return result^
            i += 1
        self.finished = True
        var result = CsvFieldSpan(self.position, n, False, True, 0)
        self.position = n
        return result^

    def _quoted_end(
        mut self, input: Span[UInt8, ImmutAnyOrigin]
    ) -> CsvFieldSpan:
        var n = len(input)
        var i = self.position
        var inside = False
        while i + _SIMD_WIDTH < n:
            var block = input.unsafe_ptr().unsafe_load[width=_SIMD_WIDTH](i)
            var quote_mask = _mask64(block.eq(self.simd_quote))
            var structural = _mask64(
                block.eq(self.simd_separator) | block.eq(self.simd_eol)
            )
            # Equivalent to Polars' `prefix_xorsum_inclusive`: one bit for
            # every byte that is inside a quoted field after that byte.
            var inside_mask = _prefix_xor_inclusive(quote_mask)
            if inside:
                inside_mask = ~inside_mask
            var outside = ~inside_mask
            var ends = structural & outside
            if ends != 0:
                var bit = Int(count_trailing_zeros(ends))
                # Match Rust's cache relative to the slice advanced past the
                # selected end. Shifting by 64 is invalid, hence its explicit
                # final-lane branch in the source.
                self.cached_ends = UInt64(
                    0
                ) if bit == _SIMD_WIDTH - 1 else ends >> UInt64(bit + 1)
                var end = i + bit
                var record = input[end] == 10
                var result = CsvFieldSpan(
                    self.position, end, True, record, input[end]
                )
                self.position = end + 1
                if record:
                    self.finished = True
                    self.cached_ends = 0
                return result^
            inside = (inside_mask & (UInt64(1) << UInt64(_SIMD_WIDTH - 1))) != 0
            i += _SIMD_WIDTH
        while i < n:
            var byte = input[i]
            if byte == self.quote:
                inside = not inside
            elif not inside and (byte == self.separator or byte == 10):
                var record = byte == 10
                var result = CsvFieldSpan(
                    self.position, i, True, record, input[i]
                )
                self.position = i + 1
                if record:
                    self.finished = True
                return result^
            i += 1
        self.finished = True
        var result = CsvFieldSpan(self.position, n, True, True, 0)
        self.position = n
        return result^

    @always_inline
    def next(
        mut self, input: Span[UInt8, ImmutAnyOrigin]
    ) -> Optional[CsvFieldSpan]:
        if self.finished:
            return None
        if self.position > len(input):
            self.finished = True
            return None
        if self.position == len(input):
            self.finished = True
            return CsvFieldSpan(self.position, self.position, False, True, 0)
        var cached = self._cached_end()
        if cached >= 0:
            var record = input[cached] == 10
            var escaped = self.quoting and input[self.position] == self.quote
            var result = CsvFieldSpan(
                self.position, cached, escaped, record, input[cached]
            )
            self.position = cached + 1
            if record:
                self.finished = True
                self.cached_ends = 0
            return result^
        if self.quoting and input[self.position] == self.quote:
            return self._quoted_end(input)
        return self._plain_end(input)
