"""Host preparation of the exact programs consumed by Mojo GPU kernels.

Only dtype/plan metadata and literal payloads are read. The same descriptors
are inspected by metadata contracts and uploaded by the executor.
"""
from std.memory import ArcPointer, bitcast
from dataframe import DataType
from dataframe.expr import (
    LIT_BOOL,
    LIT_INT,
    LIT_FLOAT,
    SUM,
    MEAN,
    DT_YEAR,
    DT_TIME,
)
from dataframe.accelerator.rows import RowPlan, ROW_IANA_ZONE_FIELDS
from . import decimal_descriptors, statistics, quantiles
from dataframe.accelerator.iana_mutations import (
    ROW_IANA_MUTATION,
    validate_descriptor,
)


def kind(dtype: DataType) raises -> Int64:
    if dtype.is_decimal() and dtype.decimal_width() == 128:
        return 12
    if dtype == DataType.BOOL:
        return 4
    var physical = dtype.physical()
    var types: List[DataType] = [
        DataType.FLOAT32,
        DataType.INT32,
        DataType.INT64,
        DataType.BOOL,
        DataType.INT8,
        DataType.INT16,
        DataType.UINT8,
        DataType.UINT16,
        DataType.UINT32,
        DataType.UINT64,
        DataType.FLOAT64,
    ]
    for i in range(len(types)):
        if physical == types[i]:
            return Int64(i + 1)
    raise Error("Mojo GPU storage is not implemented for " + dtype.name())


struct Program(Movable):
    var code: List[Int64]
    var literals: List[UInt64]
    var node_types: List[Int64]
    var outputs: List[Int64]
    var ordered_ranks: ArcPointer[List[UInt32]]
    var iana_mask_bases: List[Int]
    var has_statistics: Bool
    var has_quantiles: Bool

    def __init__(out self, plan: RowPlan) raises:
        self.code = plan.code.copy()
        self.iana_mask_bases = plan.iana_mask_bases.copy()
        self.ordered_ranks = plan.ordered_ranks.copy()
        self.literals = List[UInt64](
            capacity=len(plan.literals) + len(plan.zone_words)
        )
        self.node_types = List[Int64](capacity=len(plan.node_dtypes))
        self.outputs = List[Int64]()
        self.has_statistics = False
        self.has_quantiles = False
        for node in range(len(plan.literals)):
            var op = Int(self.code[4 * node])
            var dtype = plan.node_dtypes[node]
            var value = bitcast[DType.uint64](plan.literals[node])
            if op == LIT_BOOL:
                value = UInt64(plan.literals[node] != 0)
            elif op == LIT_INT and dtype.is_float():
                var integer = bitcast[DType.int64](plan.literals[node])
                if dtype == DataType.FLOAT32:
                    value = UInt64(bitcast[DType.uint32](Float32(integer)))
                else:
                    value = bitcast[DType.uint64](Float64(integer))
            elif op == LIT_FLOAT and dtype == DataType.FLOAT32:
                value = UInt64(
                    bitcast[DType.uint32](Float32(plan.literals[node]))
                )
            self.literals.append(value)
            self.node_types.append(kind(dtype))
        decimal_descriptors.prepare(plan, self.code, self.literals)
        var ordinary_literals = len(self.literals)
        for node in range(len(plan.node_dtypes)):
            var op = self.code[4 * node]
            if op == ROW_IANA_MUTATION:
                if plan.ordered_mode != 5:
                    raise Error(
                        "IANA mutation cannot enter a legacy row program"
                    )
                validate_descriptor(
                    plan.zone_words, Int(plan.code[4 * node + 2])
                )
                self.code[4 * node + 2] += Int64(ordinary_literals)
            if (
                op >= DT_YEAR
                and op <= DT_TIME
                and (self.code[4 * node + 3] & ROW_IANA_ZONE_FIELDS)
            ):
                self.code[4 * node + 2] += Int64(ordinary_literals)
        for word in plan.zone_words:
            self.literals.append(word)
        for base in plan.iana_descriptors:
            for field in [2, 3, 4, 5]:
                self.literals[ordinary_literals + base + field] += UInt64(
                    ordinary_literals
                )
        for step in range(len(self.iana_mask_bases)):
            if self.iana_mask_bases[step] >= 0:
                self.iana_mask_bases[step] += ordinary_literals

        if plan.reductions:
            for output in plan.outputs:
                self.has_statistics = (
                    self.has_statistics
                    or statistics.supported(Int64(output.reduction))
                )
                self.has_quantiles = self.has_quantiles or quantiles.supported(
                    Int64(output.reduction)
                )
                self.outputs.extend(
                    [
                        Int64(output.slot),
                        kind(plan.slot_dtypes[output.slot]),
                        kind(output.dtype),
                        decimal_descriptors.reduction_op(
                            Int64(output.reduction),
                            plan.slot_dtypes[output.slot],
                        ),
                        decimal_descriptors.reduction_parameter(
                            plan.slot_dtypes[output.slot], output.min_count
                        ) if plan.slot_dtypes[output.slot].is_decimal()
                        and (
                            output.reduction == SUM or output.reduction == MEAN
                        ) else Int64(output.min_count),
                    ]
                )
