"""Gather ordered payloads by absolute source row and return group-local ranks."""
from dataframe import Column, DataFrame, DataType, Series
from dataframe.storage.boolean import BoolColumn
from dataframe.expr import ARG_MIN, ARG_MAX
from .ordered_plan import OrderedPlan, ordered_side
from .group_output import group_order
from .group_gather import gather_group_keys
from .accelerator.gpu.ordered_value import ordered_ticket_row


def _payload[
    D: DType
](name: String, source: Series, tickets: List[UInt32]) raises -> Series:
    """Gather arbitrary winning rows without coalescing the source arrays."""
    var parts = source.chunks()
    var ends = List[Int](capacity=len(parts))
    var constants = List[Bool](capacity=len(parts))
    var total = 0
    for part in parts:
        total += len(part)
        ends.append(total)
        constants.append(part.is_constant())
    var n = len(tickets)
    var bits = List[UInt8](length=(n + 7) // 8, fill=0)
    var packed = List[UInt8]()
    var values = List[Scalar[D]]()
    comptime if D == DType.bool:
        packed = List[UInt8](length=len(bits), fill=0)
    else:
        values = List[Scalar[D]](length=n, fill=0)
    var chunk = 0
    var start = 0
    for i in range(n):
        if tickets[i] == UInt32.MAX:
            continue
        var row = Int(tickets[i])
        if row >= total:
            raise Error("Metal ordered ticket is outside its native input")
        if row < start or row >= ends[chunk]:
            var lo = 0
            var hi = len(ends)
            while lo < hi:
                var mid = (lo + hi) // 2
                if row < ends[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            chunk = lo
            start = 0 if chunk == 0 else ends[chunk - 1]
        var local = row - start
        var mask = UInt8(1) << UInt8(i % 8)
        comptime if D == DType.bool:
            ref column = parts[chunk]._data[BoolColumn]
            if constants[chunk]:
                if not column.constant_valid():
                    continue
                if column.constant_value():
                    packed[i // 8] |= mask
            else:
                if not column._valid(local):
                    continue
                if column._get(local):
                    packed[i // 8] |= mask
        else:
            ref column = parts[chunk]._data[Column[Scalar[D]]]
            if constants[chunk]:
                if not column.constant_valid():
                    continue
                values[i] = column.constant_value()
            else:
                if not column._valid(local):
                    continue
                values[i] = column._get(local)
        bits[i // 8] |= mask
    comptime if D == DType.bool:
        return Series(name, BoolColumn(values=packed^, bits=bits^, length=n))
    else:
        return Series(name, Column[Scalar[D]](values=values^, bits=bits^))


def finish_ordered(
    plan: OrderedPlan,
    frame: DataFrame,
    first: List[UInt32],
    final: List[UInt32],
) raises -> DataFrame:
    var slots = group_order(first) if plan.keys else List[Int]()
    var columns = List[Series]()
    if plan.keys:
        columns = gather_group_keys(frame, plan.keys.value(), first, slots)
    else:
        slots.append(0)
    for output in plan.outputs:
        ref field = plan.fields[output.field]
        var indexed = output.op == ARG_MIN or output.op == ARG_MAX
        var tickets = List[UInt32](capacity=len(slots))
        var valid = List[Bool]()
        if indexed:
            valid.reserve(len(slots))
        for slot in slots:
            var at = (slot * len(plan.fields) + output.field) * 8
            var ticket = UInt32.MAX
            if final:
                ticket = ordered_ticket_row(
                    final[at + ordered_side(output.op)], output.op
                )
            if indexed:
                if ticket != UInt32.MAX and Int(ticket) >= frame.height():
                    raise Error(
                        "Metal ordered ticket is outside its native input"
                    )
                valid.append(ticket != UInt32.MAX)
                tickets.append(
                    final[at + 4 + Int(output.op == ARG_MAX)] if ticket
                    != UInt32.MAX else UInt32(0)
                )
            else:
                tickets.append(ticket)
        if indexed:
            columns.append(
                Series(output.name, Column[UInt32](tickets^, valid^))
            )
            continue
        var source = frame.column(field.column)
        if field.dtype == DataType.INT32:
            columns.append(_payload[DType.int32](output.name, source, tickets))
        elif field.dtype == DataType.INT64:
            columns.append(_payload[DType.int64](output.name, source, tickets))
        elif field.dtype == DataType.FLOAT32:
            columns.append(
                _payload[DType.float32](output.name, source, tickets)
            )
        else:
            columns.append(_payload[DType.bool](output.name, source, tickets))
    return DataFrame(columns^)
