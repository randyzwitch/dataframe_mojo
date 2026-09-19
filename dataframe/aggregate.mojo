"""Per-group reduction state, updated one evaluated batch at a time.

A Reducer owns one state per group for one reduction node. Batches arrive in
row order with their offset, so group ids come from the shared mapping.
"""
from .column import Column
from .series import Series
from .expr import SUM, COUNT, ANY, ALL, NULL_COUNT
from .reductions import IntSumState, FloatSumState, LogicState


def _group(grouped: Bool, groups: List[Int], row: Int) -> Int:
    return groups[row] if grouped else 0


def _count_valid[
    T: Copyable & Deinitable
](
    column: Column[T],
    offset: Int,
    grouped: Bool,
    groups: List[Int],
    nulls: Bool,
    mut counts: List[Int64],
):
    for i in range(len(column)):
        if column._valid(i) != nulls:
            counts[_group(grouped, groups, offset + i)] += 1


struct Reducer(Movable):
    var op: Int
    var dtype: String
    var group_count: Int
    var min_count: Int
    var integer: Int64
    var counts: List[Int64]
    var int_sums: List[IntSumState]
    var float_sums: List[FloatSumState]
    var logic: List[LogicState]

    def __init__(
        out self,
        op: Int,
        input_dtype: String,
        group_count: Int,
        min_count: Int,
        integer: Int64,
    ):
        self.op = op
        self.dtype = input_dtype
        self.group_count = group_count
        self.min_count = min_count
        self.integer = integer
        var n = group_count
        self.counts = List[Int64](length=n, fill=0)
        self.int_sums = List[IntSumState](
            length=n if op == SUM and input_dtype == "int64" else 0,
            fill=IntSumState(),
        )
        self.float_sums = List[FloatSumState](
            length=n if op == SUM and input_dtype == "float64" else 0,
            fill=FloatSumState(),
        )
        self.logic = List[LogicState](
            length=n if op == ANY or op == ALL else 0, fill=LogicState()
        )

    def update(
        mut self, chunk: Series, offset: Int, grouped: Bool, groups: List[Int]
    ) raises:
        if self.op == COUNT or self.op == NULL_COUNT:
            var nulls = self.op == NULL_COUNT
            if chunk._data.isa[Column[Int64]]():
                _count_valid(
                    chunk._data[Column[Int64]],
                    offset,
                    grouped,
                    groups,
                    nulls,
                    self.counts,
                )
            elif chunk._data.isa[Column[Float64]]():
                _count_valid(
                    chunk._data[Column[Float64]],
                    offset,
                    grouped,
                    groups,
                    nulls,
                    self.counts,
                )
            elif chunk._data.isa[Column[Bool]]():
                _count_valid(
                    chunk._data[Column[Bool]],
                    offset,
                    grouped,
                    groups,
                    nulls,
                    self.counts,
                )
            else:
                _count_valid(
                    chunk._data[Column[String]],
                    offset,
                    grouped,
                    groups,
                    nulls,
                    self.counts,
                )
        elif self.op == SUM and self.dtype == "int64":
            ref column = chunk._data[Column[Int64]]
            for i in range(len(column)):
                if column._valid(i):
                    self.int_sums[_group(grouped, groups, offset + i)].add(
                        column._values[i]
                    )
        elif self.op == SUM:
            ref column = chunk._data[Column[Float64]]
            for i in range(len(column)):
                if column._valid(i):
                    self.float_sums[_group(grouped, groups, offset + i)].add(
                        column._values[i]
                    )
        elif self.op == ANY or self.op == ALL:
            ref column = chunk._data[Column[Bool]]
            for i in range(len(column)):
                self.logic[_group(grouped, groups, offset + i)].add(
                    column._valid(i), column._values[i]
                )
        else:
            raise Error("Unsupported reduction")

    def finish(self) raises -> Series:
        var n = self.group_count
        var valid = List[Bool](length=n, fill=True)
        if self.op == COUNT or self.op == NULL_COUNT:
            return Series("", Column[Int64](self.counts.copy()))
        if self.op == SUM and self.dtype == "int64":
            var output = List[Int64](length=n, fill=0)
            for g in range(n):
                valid[g] = self.int_sums[g].count >= Int64(self.min_count)
                if valid[g]:
                    output[g] = self.int_sums[g].value()
            return Series("", Column[Int64](output^, valid))
        if self.op == SUM:
            var output = List[Float64](length=n, fill=0)
            for g in range(n):
                valid[g] = self.float_sums[g].count >= Int64(self.min_count)
                if valid[g]:
                    output[g] = self.float_sums[g].total
            return Series("", Column[Float64](output^, valid))
        var output = List[Bool](length=n, fill=False)
        var ignore_nulls = self.integer != 0
        for g in range(n):
            var result = self.logic[g].any(
                ignore_nulls
            ) if self.op == ANY else self.logic[g].all(ignore_nulls)
            valid[g] = Bool(result)
            if result:
                output[g] = result.value()
        return Series("", Column[Bool](output^, valid))
