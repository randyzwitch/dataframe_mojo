"""Flat native grouped output columns, independent of input owners."""
from std.memory import bitcast
from dataframe import Column, DataType, Series
from dataframe.expr import COUNT, MEAN, MIN, MAX
from .multi_plan import ReductionOutput
from .extrema_value import float_extrema_bits
from .accelerator.faults import data_fault


struct GroupOutput(Movable):
    var output: ReductionOutput
    var dtype: DataType
    var row_count: Bool
    var narrow: List[Int32]
    var wide: List[Int64]
    var single: List[Float32]
    var double: List[Float64]
    var valid: List[Bool]

    def __init__(
        out self, output: ReductionOutput, dtype: DataType, row_count: Bool
    ):
        self.output = output.copy()
        self.dtype = dtype
        self.row_count = row_count
        self.narrow = List[Int32]()
        self.wide = List[Int64]()
        self.single = List[Float32]()
        self.double = List[Float64]()
        self.valid = List[Bool]()

    def append(
        mut self,
        integer: Int128,
        floating: Float64,
        count: Int,
        minimum: Int64,
        maximum: Int64,
        numeric_count: Int,
        flags: UInt32,
        rows: Int,
    ) raises:
        var op = self.output.op
        if self.row_count or op == COUNT:
            self.wide.append(Int64(rows if self.row_count else count))
            return
        if op == MEAN:
            var total = (
                floating if self.dtype
                == DataType.FLOAT32 else integer.cast[DType.float64]()
            )
            self.double.append(total / Float64(count) if count else Float64(0))
            self.valid.append(count > 0)
            return
        var present = count >= self.output.min_count
        if op == MIN or op == MAX:
            present = numeric_count != 0
            var value = minimum if op == MIN else maximum
            if self.dtype == DataType.FLOAT32:
                self.single.append(
                    bitcast[DType.float32](
                        float_extrema_bits(value, present, flags)
                    )
                )
            elif self.dtype == DataType.INT32:
                self.narrow.append(Int32(value) if present else Int32(0))
            else:
                self.wide.append(value if present else Int64(0))
        elif self.dtype == DataType.FLOAT32:
            self.single.append(Float32(floating))
        elif self.dtype == DataType.INT32:
            if integer > Int128(Int32.MAX) or integer < Int128(Int32.MIN):
                raise data_fault("int32 expression sum overflow")
            self.narrow.append(integer.cast[DType.int32]())
        else:
            if integer > Int128(Int64.MAX) or integer < Int128(Int64.MIN):
                raise data_fault("int64 expression sum overflow")
            self.wide.append(integer.cast[DType.int64]())
        self.valid.append(present)

    def finish(deinit self) raises -> Series:
        if self.row_count or self.output.op == COUNT:
            return Series(self.output.name, Column[Int64](self.wide^))
        if self.output.op == MEAN:
            return Series(
                self.output.name, Column[Float64](self.double^, self.valid)
            )
        if self.dtype == DataType.FLOAT32:
            return Series(
                self.output.name, Column[Float32](self.single^, self.valid)
            )
        if self.dtype == DataType.INT32:
            return Series(
                self.output.name, Column[Int32](self.narrow^, self.valid)
            )
        return Series(self.output.name, Column[Int64](self.wide^, self.valid))


def group_order(first: List[UInt32]) -> List[Int]:
    """Order only discovered slots; never inspect input key values."""
    var order = List[Int]()
    for slot in range(len(first)):
        if first[slot] != UInt32.MAX:
            order.append(slot)

    def before(a: Int, b: Int) {imm first} -> Bool:
        return first[a] < first[b]

    sort[stable=True](order, before)
    return order^
