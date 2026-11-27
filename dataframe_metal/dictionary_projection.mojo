"""Compact, shared state for direct grouped dictionary counts and sums."""
from dataframe import DataFrame, Expr
from dataframe.expr import COUNT, LEN, SUM
from .multi_plan import ReductionOutput


@fieldwise_init
struct DictionaryProjection(Copyable, Movable):
    var outputs: List[ReductionOutput]
    var offsets: List[Int]
    var needs_sum: List[Bool]
    var needs_count: List[Bool]
    var needs_rows: Bool
    var stride: Int


def project_dictionary(
    keys: List[String], frame: DataFrame, expressions: List[Expr]
) raises -> DictionaryProjection:
    var names = frame.columns()
    var sums = List[Bool](length=frame.width() - len(keys), fill=False)
    var counts = List[Bool](length=len(sums), fill=False)
    var outputs = List[ReductionOutput]()
    var needs_rows = False
    for expr in expressions:
        ref aggregate = expr._nodes[1]
        ref name = expr._nodes[0].text
        var field = -1
        if aggregate.op != LEN:
            if name in keys:
                # Within a group a key is either valid for every row or none.
                field = -2 - keys.index(name)
            else:
                field = names.index(name) - len(keys)
                if aggregate.op == COUNT or aggregate.min_count > 0:
                    counts[field] = True
                if aggregate.op == SUM:
                    sums[field] = True
        needs_rows = needs_rows or field < 0
        outputs.append(
            ReductionOutput(
                field, expr._name.copy(), aggregate.op, aggregate.min_count
            )
        )
    var offsets = List[Int]()
    # Each occupied slot has a first-row ticket and an optional row count.
    var stride = 1 + Int(needs_rows)
    for i in range(len(sums)):
        offsets.append(stride)
        # COUNT and positive min_count share one word. Exact integer sums use three.
        stride += Int(counts[i]) + 3 * Int(sums[i])
    return DictionaryProjection(
        outputs^, offsets^, sums^, counts^, needs_rows, stride
    )
