"""Binary view storage for CSV strings.

Each view is four
32-bit words: length; then inline bytes for strings through 12 bytes, or the
first-four-byte prefix, buffer index, and byte offset for longer strings.
Long values append to growing blocks.  A full block is moved into an
Arc-owned buffer so a finished storage object never copies string payloads.

The block policy uses: 8 KiB initial growth and a
16 MiB exponential-growth ceiling.  Validity stays absent until the first
null, as in validity bitmaps.
"""
from std.memory import ArcPointer, Pointer, bitcast, unsafe_memcpy
from .column import _append_validity_bit, _count_valid, _validity_bit


comptime STRING_VIEW_INLINE_BYTES = 12
comptime STRING_VIEW_DEFAULT_BLOCK_SIZE = 8 * 1024
comptime STRING_VIEW_MAX_EXP_BLOCK_SIZE = 16 * 1024 * 1024
comptime _STRING_VIEW_BUFFER_LIMIT = Int(Int32.MAX)


@fieldwise_init
struct StringView(Copyable):
    """The Arrow binary-view 16-byte descriptor layout."""

    var length: UInt32
    var prefix: UInt32
    var buffer_index: UInt32
    var offset: UInt32

    def is_inline(self) -> Bool:
        return self.length <= UInt32(STRING_VIEW_INLINE_BYTES)


@always_inline
def _load_prefix_little_endian(bytes: Span[UInt8, ImmutAnyOrigin]) -> UInt32:
    """The direct four-byte prefix load in `View::new_noninline_unchecked`."""
    return bitcast[DType.uint32, 1](bytes.unsafe_ptr().unsafe_load[width=4]())


@always_inline
def _inline_view(bytes: Span[UInt8, ImmutAnyOrigin]) -> StringView:
    """Exact `View::new_inline_unchecked` byte-range copy."""
    var length = len(bytes)
    assert length <= STRING_VIEW_INLINE_BYTES
    var view = StringView(UInt32(length), 0, 0, 0)
    # Polars copies directly into the bytes after the length field. This
    # preserves the exact 16-byte View layout, including zeroed tail bytes.
    unsafe_memcpy(
        dest=Pointer(to=view).unsafe_bitcast[UInt8]().unsafe_offset(4),
        src=bytes.unsafe_ptr(),
        count=length,
    )
    return view^


@always_inline
def _external_view(
    bytes: Span[UInt8, ImmutAnyOrigin], buffer_index: UInt32, offset: UInt32
) -> StringView:
    """Port ``View::new_noninline_unchecked`` for a non-inline value."""
    assert len(bytes) > STRING_VIEW_INLINE_BYTES
    return StringView(
        UInt32(len(bytes)),
        _load_prefix_little_endian(bytes),
        buffer_index,
        offset,
    )


struct StringViewStorage(Copyable, Sized):
    """Finished zero-copy view descriptors, byte blocks, and validity.

    The Arc accessors intentionally expose immutable metadata ownership to
    StringColumn's later slice/gather integration.  No mutable builder state
    is retained after construction.
    """

    var _views: ArcPointer[List[StringView]]
    var _buffers: ArcPointer[List[ArcPointer[List[UInt8]]]]
    var _bits: ArcPointer[List[UInt8]]
    var _length: Int
    var _total_bytes: Int
    var _total_buffer_bytes: Int

    def __init__(
        out self,
        var views: List[StringView],
        var buffers: List[ArcPointer[List[UInt8]]],
        var bits: List[UInt8],
        length: Int,
        total_bytes: Int,
        total_buffer_bytes: Int,
    ):
        self._views = ArcPointer(views^)
        self._buffers = ArcPointer(buffers^)
        self._bits = ArcPointer(bits^)
        self._length = length
        self._total_bytes = total_bytes
        self._total_buffer_bytes = total_buffer_bytes

    def __len__(self) -> Int:
        return self._length

    def views_arc(self) -> ArcPointer[List[StringView]]:
        return self._views.copy()

    def buffers_arc(self) -> ArcPointer[List[ArcPointer[List[UInt8]]]]:
        return self._buffers.copy()

    def validity_arc(self) -> ArcPointer[List[UInt8]]:
        return self._bits.copy()

    def total_bytes(self) -> Int:
        return self._total_bytes

    def total_buffer_bytes(self) -> Int:
        return self._total_buffer_bytes

    def buffer_count(self) -> Int:
        return len(self._buffers[])

    @always_inline
    def _view_unchecked(self, row: Int) -> StringView:
        """Unchecked descriptor copy for StringColumn's borrowed row path."""
        return self._views[][row].copy()

    def view(self, row: Int) raises -> StringView:
        if row < 0 or row >= self._length:
            raise Error("string view row index out of bounds")
        return self._view_unchecked(row)

    def valid(self, row: Int) -> Bool:
        return _validity_bit(self._bits[], row)

    def is_valid(self, row: Int) raises -> Bool:
        if row < 0 or row >= self._length:
            raise Error("string view row index out of bounds")
        return self.valid(row)

    def null_count(self) -> Int:
        return self._length - _count_valid(self._bits[], 0, self._length)

    @always_inline
    def _get_unchecked(self, row: Int) -> StringSlice[ImmutAnyOrigin]:
        """Borrow row bytes without a bounds check or payload copy."""
        var view = self._view_unchecked(row)
        var length = Int(view.length)
        if view.is_inline():
            # `StringView` is four adjacent UInt32 values. Inline bytes occupy
            # the last three words exactly as Arrow's repr(C) View does.
            var start = self._views[].unsafe_ptr().unsafe_offset(row)
            var bytes = start.unsafe_bitcast[UInt8]().unsafe_offset(4)
            return StringSlice[ImmutAnyOrigin](
                unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                    unsafe_ptr=bytes.unsafe_mut_cast[
                        False
                    ]().unsafe_origin_cast[ImmutAnyOrigin](),
                    length=length,
                )
            )
        var buffer = self._buffers[][Int(view.buffer_index)]
        return StringSlice[ImmutAnyOrigin](
            unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=buffer[]
                .unsafe_ptr()
                .unsafe_offset(Int(view.offset))
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                length=length,
            )
        )

    def get(self, row: Int) raises -> StringSlice[ImmutAnyOrigin]:
        if row < 0 or row >= self._length:
            raise Error("string view row index out of bounds")
        return self._get_unchecked(row)

    def _gather(
        self, indices: List[Int], offset: Int, allow_missing: Bool = False
    ) raises -> Self:
        return self._gather_range(
            indices, 0, len(indices), offset, allow_missing
        )

    def _gather_range(
        self,
        indices: List[Int],
        first: Int,
        last: Int,
        offset: Int,
        allow_missing: Bool = False,
    ) raises -> Self:
        """Copy descriptors/validity while retaining every referenced block.

        Arrow view descriptors use storage-local buffer indexes, so retaining
        the complete Arc buffer list leaves external descriptors unchanged.
        This is the zero-payload-copy gather route used by StringColumn.
        """
        var count = last - first
        var views = List[StringView](capacity=count)
        var buffers = List[ArcPointer[List[UInt8]]](
            capacity=len(self._buffers[])
        )
        for buffer in self._buffers[]:
            buffers.append(buffer.copy())
        var bits = List[UInt8]()
        var total_bytes = 0
        var total_buffer_bytes = 0
        for at in range(first, last):
            var index = indices[at]
            if index == -1 and allow_missing:
                _append_validity_bit(bits, len(views), False, count)
                views.append(StringView(0, 0, 0, 0))
                continue
            if index < 0 or index >= self._length - offset:
                raise Error("string view gather index out of bounds")
            var source = offset + index
            var view = self._view_unchecked(source)
            _append_validity_bit(
                bits,
                len(views),
                _validity_bit(self._bits[], source),
                count,
            )
            total_bytes += Int(view.length)
            if not view.is_inline():
                total_buffer_bytes += Int(view.length)
            views.append(view^)
        return Self(
            views^,
            buffers^,
            bits^,
            count,
            total_bytes,
            total_buffer_bytes,
        )

    def _concat(
        self,
        self_offset: Int,
        self_length: Int,
        other: Self,
        other_offset: Int,
        other_length: Int,
    ) -> Self:
        """Concatenate metadata and retain both arrays' Arc byte blocks."""
        var views = List[StringView](capacity=self_length + other_length)
        var buffers = List[ArcPointer[List[UInt8]]](
            capacity=len(self._buffers[]) + len(other._buffers[])
        )
        for buffer in self._buffers[]:
            buffers.append(buffer.copy())
        var right_buffer_base = len(buffers)
        for buffer in other._buffers[]:
            buffers.append(buffer.copy())
        var bits = List[UInt8]()
        var total_bytes = 0
        var total_buffer_bytes = 0
        for i in range(self_length):
            var source = self_offset + i
            var view = self._view_unchecked(source)
            _append_validity_bit(
                bits,
                len(views),
                _validity_bit(self._bits[], source),
                self_length + other_length,
            )
            total_bytes += Int(view.length)
            if not view.is_inline():
                total_buffer_bytes += Int(view.length)
            views.append(view^)
        for i in range(other_length):
            var source = other_offset + i
            var view = other._view_unchecked(source)
            if not view.is_inline():
                view.buffer_index += UInt32(right_buffer_base)
            _append_validity_bit(
                bits,
                len(views),
                _validity_bit(other._bits[], source),
                self_length + other_length,
            )
            total_bytes += Int(view.length)
            if not view.is_inline():
                total_buffer_bytes += Int(view.length)
            views.append(view^)
        return Self(
            views^,
            buffers^,
            bits^,
            self_length + other_length,
            total_bytes,
            total_buffer_bytes,
        )

    @staticmethod
    def _concat_many(
        parts: List[Self], offsets: List[Int], lengths: List[Int]
    ) raises -> Self:
        """Batch descriptor concatenation for Series.rechunk.

        Every input byte block remains Arc-owned; this copies each 16-byte
        descriptor once and adjusts only non-inline buffer indexes.
        """
        if (
            len(parts) == 0
            or len(parts) != len(offsets)
            or len(parts) != len(lengths)
        ):
            raise Error("String view concat metadata length mismatch")
        var rows = 0
        var buffers_count = 0
        for i in range(len(parts)):
            if (
                offsets[i] < 0
                or lengths[i] < 0
                or offsets[i] > len(parts[i])
                or lengths[i] > len(parts[i]) - offsets[i]
            ):
                raise Error("String view concat window out of bounds")
            rows += lengths[i]
            buffers_count += parts[i].buffer_count()
        var views = List[StringView](capacity=rows)
        var buffers = List[ArcPointer[List[UInt8]]](capacity=buffers_count)
        var bits = List[UInt8]()
        var total_bytes = 0
        var total_buffer_bytes = 0
        var buffer_base = 0
        for p in range(len(parts)):
            var storage = parts[p].copy()
            for buffer in storage._buffers[]:
                buffers.append(buffer.copy())
            for i in range(lengths[p]):
                var source = offsets[p] + i
                var view = storage._view_unchecked(source)
                if not view.is_inline():
                    view.buffer_index += UInt32(buffer_base)
                    total_buffer_bytes += Int(view.length)
                _append_validity_bit(
                    bits,
                    len(views),
                    _validity_bit(storage._bits[], source),
                    rows,
                )
                total_bytes += Int(view.length)
                views.append(view^)
            buffer_base += storage.buffer_count()
        return Self(
            views^, buffers^, bits^, rows, total_bytes, total_buffer_bytes
        )


struct StringViewBuilder(Movable):
    """Port of ``MutableBinaryViewArray::push_value_into_buffer``."""

    var _views: List[StringView]
    var _completed_buffers: List[ArcPointer[List[UInt8]]]
    var _in_progress: List[UInt8]
    var _bits: List[UInt8]
    var _total_bytes: Int
    var _total_buffer_bytes: Int

    def __init__(out self, capacity: Int = 0):
        self._views = List[StringView](capacity=max(0, capacity))
        self._completed_buffers = List[ArcPointer[List[UInt8]]]()
        self._in_progress = List[UInt8]()
        self._bits = List[UInt8]()
        self._total_bytes = 0
        self._total_buffer_bytes = 0

    def __len__(self) -> Int:
        return len(self._views)

    def _flush_in_progress(mut self):
        if len(self._in_progress) != 0:
            self._completed_buffers.append(ArcPointer(self._in_progress^))
        self._in_progress = List[UInt8]()

    def _new_buffer(mut self, bytes: Int):
        """Exact Polars block replacement and exponential-capacity policy."""
        var doubled = self._in_progress.capacity() * 2
        var grown = min(
            STRING_VIEW_MAX_EXP_BLOCK_SIZE,
            max(STRING_VIEW_DEFAULT_BLOCK_SIZE, doubled),
        )
        var capacity = max(bytes, grown)
        self._flush_in_progress()
        self._in_progress = List[UInt8](capacity=capacity)

    def _push_value_into_buffer(
        mut self, bytes: Span[UInt8, ImmutAnyOrigin]
    ) -> StringView:
        assert len(bytes) <= Int(UInt32.MAX)
        if len(bytes) <= STRING_VIEW_INLINE_BYTES:
            return _inline_view(bytes)

        self._total_buffer_bytes += len(bytes)
        # Polars flushes before a growth realloc so a prior view always points
        # at immutable completed storage; it never copies between blocks.
        if len(self._in_progress) + len(bytes) > min(
            _STRING_VIEW_BUFFER_LIMIT, self._in_progress.capacity()
        ):
            self._new_buffer(len(bytes))
        var offset = UInt32(len(self._in_progress))
        var index = UInt32(len(self._completed_buffers))
        self._in_progress.extend(bytes)
        return _external_view(bytes, index, offset)

    def append(mut self, text: StringSlice):
        var input = text.as_bytes()
        var bytes = Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=input.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            length=len(input),
        )
        _append_validity_bit(
            self._bits, len(self._views), True, self._views.capacity()
        )
        self._total_bytes += len(bytes)
        self._views.append(self._push_value_into_buffer(bytes))

    def append_null(mut self):
        _append_validity_bit(
            self._bits, len(self._views), False, self._views.capacity()
        )
        self._views.append(StringView(0, 0, 0, 0))

    def finish(deinit self) -> StringViewStorage:
        self._flush_in_progress()
        var length = len(self._views)
        return StringViewStorage(
            self._views^,
            self._completed_buffers^,
            self._bits^,
            length,
            self._total_bytes,
            self._total_buffer_bytes,
        )
