"""Native byte/packed-bit input windows and their independent validity offsets."""
from max.gpu.host import DeviceBuffer


struct NumericView(Copyable, Movable):
    var values: DeviceBuffer[DType.uint8]
    var bits: DeviceBuffer[DType.uint8]
    var bit_offset: Int
    var rows: Int
    var tag: Int
    var nullable: Bool
    var value_bit_offset: Int

    def __init__(
        out self,
        var values: DeviceBuffer[DType.uint8],
        var bits: DeviceBuffer[DType.uint8],
        bit_offset: Int,
        rows: Int,
        tag: Int,
        nullable: Bool,
        value_bit_offset: Int = 0,
    ):
        self.values = values^
        self.bits = bits^
        self.bit_offset = bit_offset
        self.rows = rows
        self.tag = tag
        self.nullable = nullable
        self.value_bit_offset = value_bit_offset

    def view(self, start: Int, length: Int) raises -> Self:
        if (
            start < 0
            or length < 0
            or start > self.rows
            or length > self.rows - start
        ):
            raise Error("Metal native input window is out of bounds")
        if self.tag == 3:
            var first = self.value_bit_offset + start
            return Self(
                self.values.create_sub_buffer[DType.uint8](
                    first // 8, (first % 8 + length + 7) // 8
                ),
                self.bits.copy(),
                self.bit_offset + start if self.nullable else 0,
                length,
                self.tag,
                self.nullable,
                first % 8,
            )
        var width = 8 if self.tag == 1 else 4
        return Self(
            self.values.create_sub_buffer[DType.uint8](
                start * width, length * width
            ),
            self.bits.copy(),
            self.bit_offset + start if self.nullable else 0,
            length,
            self.tag,
            self.nullable,
        )
