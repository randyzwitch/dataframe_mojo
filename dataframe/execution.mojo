"""Bounded eager execution; expression nodes dispatch once per column batch.

Reductions are separate passes, not per-row callbacks or per-group dataframes.
The reference reduction schedule is serial; Float64 results may be reassociated
by future parallel implementations. Int64 sums use exact wide states and check final overflow.
"""
from .expr import (
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_STRING,
    ADD,
    SUB,
    MUL,
    GT,
    EQ,
    SUM,
    COUNT,
)
from .binding import BoundExpr, ROWS
from .column import Column
from .series import Series
from .expr_kernels import binary
from .reductions import IntSumState, FloatSumState


def _empty(dtype: String) raises -> Series:
    if dtype == "int64":
        return Series("", Column[Int64]([]))
    if dtype == "float64":
        return Series("", Column[Float64]([]))
    if dtype == "bool":
        return Series("", Column[Bool]([]))
    if dtype == "string":
        return Series("", Column[String]([]))
    raise Error("Unknown expression dtype")


def _batch[
    width: Int
](
    bound: BoundExpr,
    columns: List[Series],
    aggregates: List[Series],
    target: Int,
    offset: Int,
    length: Int,
    grouped: Bool,
) raises -> Series:
    var needed = List[Bool](length=target + 1, fill=False)
    needed[target] = True
    for reverse in range(target + 1):
        var i = target - reverse
        if needed[i]:
            var node = bound.expr._nodes[i].copy()
            if node.op != SUM and node.op != COUNT:
                if node.left >= 0:
                    needed[node.left] = True
                if node.right >= 0:
                    needed[node.right] = True
    var results = List[Series](capacity=target + 1)
    for i in range(target + 1):
        if not needed[i]:
            results.append(_empty("bool"))
            continue
        var node = bound.expr._nodes[i].copy()
        if node.op == COL:
            results.append(columns[bound.sources[i]].slice(offset, length))
        elif node.op == LIT_INT:
            results.append(Series("", Column[Int64]([node.integer])))
        elif node.op == LIT_FLOAT:
            results.append(Series("", Column[Float64]([node.floating])))
        elif node.op == LIT_BOOL:
            results.append(Series("", Column[Bool]([Bool(node.integer)])))
        elif node.op == LIT_STRING:
            results.append(Series("", Column[String]([node.text])))
        elif node.op == SUM or node.op == COUNT:
            if grouped:
                results.append(aggregates[i].slice(offset, length))
            else:
                results.append(aggregates[i].copy())
        elif node.op == ADD:
            results.append(
                binary[ADD, width](results[node.left], results[node.right])
            )
        elif node.op == SUB:
            results.append(
                binary[SUB, width](results[node.left], results[node.right])
            )
        elif node.op == MUL:
            results.append(
                binary[MUL, width](results[node.left], results[node.right])
            )
        elif node.op == GT:
            results.append(
                binary[GT, width](results[node.left], results[node.right])
            )
        elif node.op == EQ:
            results.append(
                binary[EQ, width](results[node.left], results[node.right])
            )
        else:
            raise Error("Unsupported expression node")
    return results[target].copy()


def _accumulate_int(
    values: Column[Int64],
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    mut states: List[IntSumState],
):
    for i in range(len(values)):
        if values._valid(i):
            var g = groups[offset + i] if grouped else 0
            states[g].add(values._values[i])


def _accumulate_float(
    values: Column[Float64],
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    mut states: List[FloatSumState],
):
    for i in range(len(values)):
        if values._valid(i):
            var g = groups[offset + i] if grouped else 0
            states[g].add(values._values[i])


def _count[
    T: Copyable & Deinitable
](
    values: Column[T],
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    mut counts: List[Int64],
):
    for i in range(len(values)):
        if values._valid(i):
            var g = groups[offset + i] if grouped else 0
            counts[g] += 1


def evaluate[
    width: Int = 4
](
    bound: BoundExpr,
    columns: List[Series],
    height: Int,
    *,
    batch_size: Int = 1024,
    grouped: Bool = False,
    groups: List[Int] = List[Int](),
    group_count: Int = 1,
) raises -> Series:
    """Evaluate a bound expression; temporary vectors are bounded by batch_size.

    Full output and O(groups * aggregates) states are materialized. Each
    aggregate currently scans its own input; shared-subexpression fusion and
    parallel state merging are future execution changes, not API changes.
    """
    if batch_size <= 0:
        raise Error("batch_size must be positive")
    if grouped and (len(groups) != height or group_count < 0):
        raise Error("Invalid group mapping")
    var states = List[Series]()
    for i in range(len(bound.expr._nodes)):
        states.append(_empty(bound.dtypes[i]))
    for node_index in range(len(bound.expr._nodes)):
        var node = bound.expr._nodes[node_index].copy()
        if node.op != SUM and node.op != COUNT:
            continue
        var counts = List[Int64](length=group_count, fill=0)
        var integers = List[IntSumState](length=group_count, fill=IntSumState())
        var floats = List[FloatSumState](
            length=group_count, fill=FloatSumState()
        )
        for offset in range(0, height, batch_size):
            var chunk = _batch[width](
                bound,
                columns,
                states,
                node.left,
                offset,
                min(batch_size, height - offset),
                False,
            )
            if node.op == COUNT:
                if chunk._data.isa[Column[Int64]]():
                    _count(
                        chunk._data[Column[Int64]],
                        offset,
                        grouped,
                        groups,
                        counts,
                    )
                elif chunk._data.isa[Column[Float64]]():
                    _count(
                        chunk._data[Column[Float64]],
                        offset,
                        grouped,
                        groups,
                        counts,
                    )
                elif chunk._data.isa[Column[Bool]]():
                    _count(
                        chunk._data[Column[Bool]],
                        offset,
                        grouped,
                        groups,
                        counts,
                    )
                else:
                    _count(
                        chunk._data[Column[String]],
                        offset,
                        grouped,
                        groups,
                        counts,
                    )
            elif bound.dtypes[node_index] == "int64":
                _accumulate_int(
                    chunk._data[Column[Int64]],
                    offset,
                    grouped,
                    groups,
                    integers,
                )
            else:
                _accumulate_float(
                    chunk._data[Column[Float64]],
                    offset,
                    grouped,
                    groups,
                    floats,
                )
        var valid = List[Bool](length=group_count, fill=True)
        if node.op == COUNT:
            states[node_index] = Series("", Column[Int64](counts^))
        elif bound.dtypes[node_index] == "int64":
            var output = List[Int64](length=group_count, fill=0)
            for g in range(group_count):
                valid[g] = integers[g].count >= Int64(node.min_count)
                if valid[g]:
                    output[g] = integers[g].value()
            states[node_index] = Series("", Column[Int64](output^, valid))
        else:
            var output = List[Float64](length=group_count, fill=0)
            for g in range(group_count):
                valid[g] = floats[g].count >= Int64(node.min_count)
                if valid[g]:
                    output[g] = floats[g].total
            states[node_index] = Series("", Column[Float64](output^, valid))
    var root = len(bound.expr._nodes) - 1
    var result = _empty(bound.dtypes[root])
    var size = height if bound.shape() == ROWS else 1
    if grouped:
        size = group_count
    for offset in range(0, size, batch_size):
        var chunk = _batch[width](
            bound,
            columns,
            states,
            root,
            offset,
            min(batch_size, size - offset),
            grouped,
        )
        result._append_series(chunk)
    result._name = bound.expr._name
    return result^
