"""Result-sized native payload gathering and filtered/group-local indices."""
from dataframe import Column, DataFrame, DataType, Series
from dataframe.expr import ARG_MIN, ARG_MAX
from .ordered_plan import OrderedPlan, ordered_side
from .group_output import group_order
from .accelerator.gpu.ordered_value import ordered_ticket_row


struct OrderedColumn(Movable):
    var name: String
    var dtype: DataType
    var narrow: List[Int32]
    var wide: List[Int64]
    var floating: List[Float32]
    var boolean: List[Bool]
    var indices: List[UInt32]
    var valid: List[Bool]

    def __init__(out self, name: String, dtype: DataType):
        self.name = name.copy()
        self.dtype = dtype
        self.narrow = List[Int32]()
        self.wide = List[Int64]()
        self.floating = List[Float32]()
        self.boolean = List[Bool]()
        self.indices = List[UInt32]()
        self.valid = List[Bool]()

    def append(
        mut self,
        frame: DataFrame,
        column: String,
        row: UInt32,
        rank: UInt32 = 0,
    ) raises:
        var present = row != UInt32.MAX
        if present and row >= UInt32(frame.height()):
            raise Error("Metal ordered ticket is outside its native input")
        if self.dtype == DataType.UINT32:
            self.indices.append(rank)
            self.valid.append(present)
            return
        if present:
            present = not frame.item(Int(row), column).is_null()
        self.valid.append(present)
        if self.dtype == DataType.INT32:
            self.narrow.append(
                frame.item(Int(row), column).int32() if present else Int32(0)
            )
        elif self.dtype == DataType.INT64:
            self.wide.append(
                frame.item(Int(row), column).int64() if present else Int64(0)
            )
        elif self.dtype == DataType.FLOAT32:
            self.floating.append(
                Float32(
                    frame.item(Int(row), column).float32()
                ) if present else Float32(0)
            )
        else:
            self.boolean.append(
                frame.item(Int(row), column).bool() if present else False
            )

    def finish(deinit self) raises -> Series:
        if self.dtype == DataType.INT32:
            return Series(self.name, Column[Int32](self.narrow^, self.valid))
        if self.dtype == DataType.INT64:
            return Series(self.name, Column[Int64](self.wide^, self.valid))
        if self.dtype == DataType.FLOAT32:
            return Series(
                self.name, Column[Float32](self.floating^, self.valid)
            )
        if self.dtype == DataType.UINT32:
            return Series(self.name, Column[UInt32](self.indices^, self.valid))
        return Series(self.name, Column[Bool](self.boolean^, self.valid))


def finish_ordered(
    plan: OrderedPlan,
    frame: DataFrame,
    first: List[UInt32],
    final: List[UInt32],
) raises -> DataFrame:
    var slots = group_order(first) if plan.keys else List[Int]()
    if not plan.keys:
        slots.append(0)
    var columns = List[Series]()
    if plan.keys:
        for key in range(len(plan.keys.value().names)):
            var name = plan.keys.value().names[key].copy()
            var output = OrderedColumn(name, plan.keys.value().dtypes[key])
            for slot in slots:
                output.append(frame, name, first[slot])
            columns.append(output^.finish())
    for output in plan.outputs:
        ref field = plan.fields[output.field]
        var indexed = output.op == ARG_MIN or output.op == ARG_MAX
        var column = OrderedColumn(
            output.name, DataType.UINT32 if indexed else field.dtype
        )
        for slot in slots:
            var at = (slot * len(plan.fields) + output.field) * 8
            var ticket = UInt32.MAX
            var rank = UInt32(0)
            if final:
                ticket = ordered_ticket_row(
                    final[at + ordered_side(output.op)], output.op
                )
                if indexed:
                    rank = final[at + 4 + Int(output.op == ARG_MAX)]
            column.append(frame, field.column, ticket, rank)
        columns.append(column^.finish())
    return DataFrame(columns^)
