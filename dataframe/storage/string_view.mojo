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
from dataframe.storage.buffer import _Buffer, _BufferData
from std.sys.intrinsics import prefetch
from std.memory import ArcPointer, Pointer, bitcast, unsafe_memcpy
from dataframe.storage.allocation import huge_uninit
from dataframe.storage.column import (
    _append_validity_bit,
    _count_valid,
    _validity_at,
    _validity_bit,
)


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


@always_inline
def _mapped(rows: Int, shift: Int, i: Int) -> Int:
    """Row i's descriptor index under a `_row_map` (`rows` 0: i itself)."""
    if rows == 0:
        return i
    return (
        Pointer[Int, ImmutAnyOrigin](unsafe_from_address=rows)[unsafe_offset=i]
        + shift
    )


struct StringViewStorage(Copyable, Sized):
    """Finished zero-copy view descriptors, byte blocks, and validity.

    The Arc accessors intentionally expose immutable metadata ownership to
    StringColumn's later slice/gather integration.  No mutable builder state
    is retained after construction.

    A storage can select its rows from another's descriptors instead of
    holding its own (its descriptor buffer then carries a row list,
    `_Buffer.selecting`), as DuckDB's dictionary vector is a child vector
    and a selection vector: a join's build-side strings are
    then the build's descriptors and the matched rows, and no descriptor
    is copied until a kernel needs them contiguous (`_flat`). Validity is
    always the rows' own.
    """

    var _views: _Buffer[StringView]
    var _buffers: ArcPointer[List[_Buffer[UInt8]]]
    var _bits: _Buffer[UInt8]
    var _length: Int
    var _total_buffer_bytes: Int

    def __init__(
        out self,
        var views: List[StringView],
        var buffers: List[_Buffer[UInt8]],
        var bits: List[UInt8],
        length: Int,
        total_buffer_bytes: Int,
    ):
        self._views = _Buffer(views^)
        self._buffers = ArcPointer(buffers^)
        self._bits = _Buffer(bits^)
        self._length = length
        self._total_buffer_bytes = total_buffer_bytes

    def __init__(
        out self,
        *,
        views: _Buffer[StringView],
        var buffers: List[_Buffer[UInt8]],
        bits: _Buffer[UInt8],
        length: Int,
        total_buffer_bytes: Int,
    ):
        self._views = views.copy()
        self._buffers = ArcPointer(buffers^)
        self._bits = bits.copy()
        self._length = length
        self._total_buffer_bytes = total_buffer_bytes

    def __len__(self) -> Int:
        return self._length

    @staticmethod
    def _selecting(
        source: Self,
        rows: ArcPointer[List[Int]],
        first: Int,
        shift: Int,
        length: Int,
        var bits: List[UInt8],
    ) -> Self:
        """`length` rows that read `source`'s descriptors: row i is its
        descriptor `rows[first + i] + shift`, with validity `bits`. The
        caller has checked every row; `source` must not select."""
        var result = Self(
            views=_Buffer.selecting(source._views, rows, first, shift),
            buffers=List[_Buffer[UInt8]](),
            bits=_Buffer(bits^),
            length=length,
            total_buffer_bytes=source._total_buffer_bytes,
        )
        result._buffers = source._buffers.copy()
        return result^

    @always_inline
    def _selects(self) -> Bool:
        return True if self._views[].selection else False

    @always_inline
    def _descriptor(self, row: Int) -> Int:
        """The descriptor row `row` reads."""
        if self._views[].selection:
            return self._selected(row)
        return row

    @no_inline
    def _selected(self, row: Int) -> Int:
        # Out of line: every string read inlines `_descriptor`, and the
        # selection's reads made the common path's code larger.
        ref selection = self._views[].selection.value()[]
        return (
            selection.rows[].unsafe_ptr()[unsafe_offset=selection.first + row]
            + selection.shift
        )

    def _row_map(self, offset: Int) -> Tuple[Int, Int, Int]:
        """For kernels that read descriptors through pointers taken once,
        the rows of a window from `offset`: the address descriptors are
        indexed from, the address of the window's first selected row (0
        when the storage does not select), and the shift added to each
        selected row. Row i's descriptor is at index i, or at
        `rows[i] + shift`."""
        var base = Int(self._views[].unsafe_ptr())
        if not self._views[].selection:
            return (base + 16 * offset, 0, 0)
        ref selection = self._views[].selection.value()[]
        return (
            base,
            Int(
                selection.rows[]
                .unsafe_ptr()
                .unsafe_offset(selection.first + offset)
            ),
            selection.shift,
        )

    def _flat(self) -> Self:
        """This storage with one descriptor a row (itself when it already
        has them): what kernels reading descriptors through a pointer
        need."""
        if not self._views[].selection:
            return self.copy()
        var views = huge_uninit[StringView](self._length)
        var target = views.unsafe_ptr().unsafe_bitcast[SIMD[DType.uint32, 4]]()
        var source = (
            self._views[].unsafe_ptr().unsafe_bitcast[SIMD[DType.uint32, 4]]()
        )
        ref selection = self._views[].selection.value()[]
        var rows = selection.rows[].unsafe_ptr().unsafe_offset(selection.first)
        var shift = selection.shift
        var total = 0
        for k in range(self._length):
            var word = source[unsafe_offset=rows[unsafe_offset=k] + shift]
            target[unsafe_offset=k] = word
            var length = Int(word[0])
            if length > STRING_VIEW_INLINE_BYTES:
                total += length
        var result = Self(
            views=_Buffer(views^),
            buffers=List[_Buffer[UInt8]](),
            bits=self._bits,
            length=self._length,
            total_buffer_bytes=total,
        )
        result._buffers = self._buffers.copy()
        return result^

    def views_arc(self) -> _Buffer[StringView]:
        """The descriptors, one a row (copied out of a selection)."""
        if self._views[].selection:
            return self._flat()._views.copy()
        return self._views.copy()

    def buffers_arc(self) -> ArcPointer[List[_Buffer[UInt8]]]:
        return self._buffers.copy()

    def validity_arc(self) -> _Buffer[UInt8]:
        return self._bits.copy()

    def total_buffer_bytes(self) -> Int:
        return self._total_buffer_bytes

    def buffer_count(self) -> Int:
        return len(self._buffers[])

    @always_inline
    def _view_unchecked(self, row: Int) -> StringView:
        """Unchecked descriptor copy for StringColumn's borrowed row path."""
        return (
            self._views[]
            .unsafe_ptr()[unsafe_offset=self._descriptor(row)]
            .copy()
        )

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
        var at = self._descriptor(row)
        var view = self._views[].unsafe_ptr()[unsafe_offset=at].copy()
        var length = Int(view.length)
        if view.is_inline():
            # `StringView` is four adjacent UInt32 values. Inline bytes occupy
            # the last three words exactly as Arrow's repr(C) View does.
            var start = self._views[].unsafe_ptr().unsafe_offset(at)
            var bytes = start.unsafe_bitcast[UInt8]().unsafe_offset(4)
            return StringSlice[ImmutAnyOrigin](
                unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                    unsafe_ptr=bytes.unsafe_mut_cast[
                        False
                    ]().unsafe_origin_cast[ImmutAnyOrigin](),
                    length=length,
                )
            )
        # The storage already owns this immutable block. Borrowing avoids
        # an atomic Arc retain/release on every long-string read.
        ref buffer = self._buffers[][Int(view.buffer_index)]
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

    @staticmethod
    def _sharing(
        var views: List[StringView],
        buffers: ArcPointer[List[_Buffer[UInt8]]],
        var bits: List[UInt8],
        length: Int,
        total_buffer_bytes: Int,
    ) -> Self:
        """Storage over `views` that shares an existing buffer list."""
        var result = Self(
            views^,
            List[_Buffer[UInt8]](),
            bits^,
            length,
            total_buffer_bytes,
        )
        result._buffers = buffers.copy()
        return result^

    def _gather_range(
        self,
        indices: List[Int],
        first: Int,
        last: Int,
        offset: Int,
        allow_missing: Bool = False,
    ) raises -> Self:
        """Copy descriptors/validity while retaining every referenced block.

        Arrow view descriptors use storage-local buffer indexes, so sharing
        the source's buffer list leaves external descriptors unchanged. The
        list itself is shared (one reference count), not copied entry by
        entry, and views and validity bits are written directly (#395).
        """
        var count = last - first
        var source_views = self._views[].unsafe_ptr()
        var nulls = len(self._bits[]) > 0
        var bits = List[UInt8](
            length=(count + 7) // 8 if (nulls or allow_missing) else 0, fill=0
        )
        var missing = False
        var total_buffer_bytes = 0
        var rows = indices.unsafe_ptr().unsafe_offset(first)
        var limit = self._length - offset
        if self._views[].selection:
            # Rows of a selection read the descriptors it selects: the
            # gather composes the two row lists (validity is this
            # storage's own, by row).
            ref selection = self._views[].selection.value()[]
            var inner = (
                selection.rows[]
                .unsafe_ptr()
                .unsafe_offset(selection.first + offset)
            )
            var shift = selection.shift
            var through = List[Int](unsafe_uninit_length=count)
            var at = through.unsafe_ptr()
            for k in range(count):
                var index = rows[unsafe_offset=k]
                if index < 0 or index >= limit:
                    if index == -1 and allow_missing:
                        at[unsafe_offset=k] = -1
                        continue
                    raise Error("string view gather index out of bounds")
                at[unsafe_offset=k] = inner[unsafe_offset=index] + shift
            var base = Self(
                views=self._views.unselected(),
                buffers=List[_Buffer[UInt8]](),
                bits=_Buffer(List[UInt8]()),
                length=len(self._views[]),
                total_buffer_bytes=self._total_buffer_bytes,
            )
            base._buffers = self._buffers.copy()
            var gathered = base._gather_range(
                through, 0, count, 0, allow_missing
            )
            if not nulls:
                return gathered^
            # The rows' validity, from this storage's bits by row.
            var bits = List[UInt8](length=(count + 7) // 8, fill=0)
            var out_bits = bits.unsafe_ptr()
            var in_bits = self._bits[].unsafe_ptr()
            for k in range(count):
                var index = rows[unsafe_offset=k]
                if index >= 0 and _validity_at(in_bits, offset + index):
                    out_bits[unsafe_offset=k >> 3] |= UInt8(1) << UInt8(k & 7)
            gathered._bits = _Buffer(bits^)
            return gathered^
        # Each view is 16 bytes copied as four words into a buffer sized
        # once: a join's gather of seven customer strings per fact row
        # (TPC-DS q4) spent 19 ns a view appending them one by one.
        var views = huge_uninit[StringView](count)
        var target = views.unsafe_ptr().unsafe_bitcast[SIMD[DType.uint32, 4]]()
        var source = source_views.unsafe_bitcast[SIMD[DType.uint32, 4]]()
        # Random rows make each view's load a cache miss; the view sixteen
        # rows ahead is fetched while this one is copied, so the misses
        # overlap rather than queue.
        var ahead = 16
        for k in range(count):
            if k + ahead < count:
                var next = rows[unsafe_offset=k + ahead]
                if next >= 0 and next < limit:
                    prefetch(
                        source.unsafe_offset(offset + next).unsafe_bitcast[
                            UInt32
                        ]()
                    )
            var index = rows[unsafe_offset=k]
            if index < 0 or index >= limit:
                if index == -1 and allow_missing:
                    missing = True
                    target[unsafe_offset=k] = SIMD[DType.uint32, 4](0)
                    continue
                raise Error("string view gather index out of bounds")
            var word = source[unsafe_offset=offset + index]
            target[unsafe_offset=k] = word
            var length = Int(word[0])
            if length > STRING_VIEW_INLINE_BYTES:
                total_buffer_bytes += length
        if len(bits) > 0:
            # Through pointers: a bounds check on both bitmaps per row cost
            # as much as reading the bits.
            var out_bits = bits.unsafe_ptr()
            var in_bits = self._bits[].unsafe_ptr()
            for k in range(count):
                var index = rows[unsafe_offset=k]
                if index < 0:
                    continue
                if not nulls or _validity_at(in_bits, offset + index):
                    out_bits[unsafe_offset=k >> 3] |= UInt8(1) << UInt8(k & 7)
        if not nulls and not missing:
            bits = List[UInt8]()
        return Self._sharing(
            views^,
            self._buffers,
            bits^,
            count,
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
        var buffers = List[_Buffer[UInt8]](
            capacity=len(self._buffers[]) + len(other._buffers[])
        )
        for buffer in self._buffers[]:
            buffers.append(buffer.copy())
        var right_buffer_base = len(buffers)
        for buffer in other._buffers[]:
            buffers.append(buffer.copy())
        var bits = List[UInt8]()
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
            if not view.is_inline():
                total_buffer_bytes += Int(view.length)
            views.append(view^)
        return Self(
            views^,
            buffers^,
            bits^,
            self_length + other_length,
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
        # Each distinct byte block is listed once: gathered pieces of one
        # column usually share the same blocks, and listing them again per
        # piece made the lists, and every later gather's copy of them, grow
        # with each merge (#395).
        var views = List[StringView](unsafe_uninit_length=rows)
        var out = views.unsafe_ptr()
        var buffers = List[_Buffer[UInt8]]()
        var position = Dict[Int, Int]()
        var any_nulls = False
        for p in range(len(parts)):
            any_nulls = any_nulls or len(parts[p]._bits[]) > 0
        var bits = List[UInt8](
            length=(rows + 7) // 8 if any_nulls else 0, fill=0
        )
        var total_buffer_bytes = 0
        var row = 0
        for p in range(len(parts)):
            # A selecting part's descriptors, one a row, first.
            var flat = parts[p]._flat()
            ref storage = flat
            var remap = List[UInt32](capacity=storage.buffer_count())
            for buffer in storage._buffers[]:
                var key = Int(buffer.ptr())
                var at = position.get(key, -1)
                if at < 0:
                    at = len(buffers)
                    position[key] = at
                    buffers.append(buffer.copy())
                remap.append(UInt32(at))
            var source_views = storage._views[].unsafe_ptr()
            var nulls = len(storage._bits[]) > 0
            var identity = True
            for k in range(len(remap)):
                identity = identity and Int(remap[k]) == k
            var maps = remap.unsafe_ptr()
            var count = lengths[p]
            var first = source_views.unsafe_offset(offsets[p])
            if identity:
                # Buffer indexes stay as they are: one block copy of the
                # descriptors, then only their lengths are read.
                unsafe_memcpy(
                    dest=out.unsafe_offset(row), src=first, count=count
                )
            for i in range(count):
                ref view = first.unsafe_offset(i)[]
                var length = Int(view.length)
                if not view.is_inline():
                    total_buffer_bytes += length
                    if not identity:
                        var moved = view.copy()
                        moved.buffer_index = maps[
                            unsafe_offset=Int(view.buffer_index)
                        ]
                        out.unsafe_offset(row + i)[] = moved^
                elif not identity:
                    out.unsafe_offset(row + i)[] = view.copy()
                if any_nulls and (
                    not nulls or _validity_bit(storage._bits[], offsets[p] + i)
                ):
                    var at = row + i
                    bits[at >> 3] |= UInt8(1) << UInt8(at & 7)
            row += count
        return Self(views^, buffers^, bits^, rows, total_buffer_bytes)


struct StringViewBuilder(Movable):
    """Port of ``MutableBinaryViewArray::push_value_into_buffer``."""

    var _views: List[StringView]
    var _completed_buffers: List[_Buffer[UInt8]]
    var _in_progress: List[UInt8]
    var _bits: List[UInt8]
    var _total_buffer_bytes: Int

    def __init__(out self, capacity: Int = 0):
        self._views = List[StringView](capacity=max(0, capacity))
        self._completed_buffers = List[_Buffer[UInt8]]()
        self._in_progress = List[UInt8]()
        self._bits = List[UInt8]()
        self._total_buffer_bytes = 0

    def __len__(self) -> Int:
        return len(self._views)

    def _flush_in_progress(mut self):
        if len(self._in_progress) != 0:
            self._completed_buffers.append(_Buffer(self._in_progress^))
        self._in_progress = List[UInt8]()

    def _new_buffer(mut self, bytes: Int):
        """Block replacement and exponential-capacity policy for string views."""
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
            self._total_buffer_bytes,
        )
