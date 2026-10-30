"""Owned physical windows for numeric kernels with multiple inputs."""
from dataframe.series import Series
from dataframe.storage.column import Column


struct _NumericInput[D: DType](Movable):
    """A forward cursor; only a whole one-row operand broadcasts.

    The source and current column retain their buffer owners. Callers bound
    each SIMD block by available(), then advance at physical boundaries.
    Payload pointers already include the column offset; bitmap indices do not.
    """

    var owner: Series
    var column: Column[Scalar[Self.D]]
    var chunk: Int
    var row: Int
    var scalar: Bool

    def __init__(out self, input: Series):
        self.owner = input.copy()
        self.chunk = 0
        self.row = 0
        self.scalar = len(input) == 1
        if input.is_chunked():
            self.column = (
                input._chunked.value()[]
                .arrays[0][Column[Scalar[Self.D]]]
                .copy()
            )
        else:
            self.column = input._data[Column[Scalar[Self.D]]].copy()
        self._skip_empty()

    def _skip_empty(mut self):
        if not self.owner.is_chunked():
            return
        ref chunks = self.owner._chunked.value()[]
        while self.row == len(self.column) and self.chunk + 1 < len(
            chunks.arrays
        ):
            self.chunk += 1
            self.column = chunks.arrays[self.chunk][
                Column[Scalar[Self.D]]
            ].copy()
            self.row = 0

    def available(self, remaining: Int) -> Int:
        return remaining if self.scalar else min(
            remaining, len(self.column) - self.row
        )

    def advance(mut self, count: Int):
        if self.scalar:
            return
        assert count >= 0 and count <= len(self.column) - self.row
        self.row += count
        self._skip_empty()

    @always_inline
    def values[W: Int](self, relative: Int = 0) -> SIMD[Self.D, W]:
        if self.scalar:
            return SIMD[Self.D, W](self.column._get(0))
        return self.column._ptr().unsafe_load[width=W](self.row + relative)

    @always_inline
    def valid[W: Int](self, relative: Int = 0) -> SIMD[DType.bool, W]:
        var bits = self.column._bits[].span()
        if len(bits) == 0:
            return SIMD[DType.bool, W](fill=True)
        if self.scalar:
            return SIMD[DType.bool, W](fill=self.column._valid(0))
        var start = self.column._offset + self.row + relative
        var flags = SIMD[DType.bool, W](fill=False)
        comptime for byte in range((W + 7) // 8):
            var bit = start + 8 * byte
            var index = bit // 8
            var shift = UInt8(bit % 8)
            var value = bits[index] >> shift
            if shift != 0 and index + 1 < len(bits):
                value |= bits[index + 1] << (UInt8(8) - shift)
            comptime for lane in range(min(8, W - 8 * byte)):
                flags[8 * byte + lane] = (value >> UInt8(lane)) & 1 != 0
        return flags
