"""Finish exact counts and both null policies from bounded GPU result state."""
from dataframe import Column, DataFrame, Series
from .distinct_plan import DistinctPlan
from .ordered_finish import OrderedColumn
from .group_output import group_order


def finish_distinct(
    plan: DistinctPlan,
    frame: DataFrame,
    first: List[UInt32],
    counts: List[UInt32],
    nulls: List[UInt32],
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
        var values = List[Int64]()
        for slot in slots:
            var at = slot * len(plan.fields) + output.field
            var count = Int64(counts[at]) if counts else Int64(0)
            var missing = Int64(nulls[at]) if nulls else Int64(0)
            if missing > 1 or count < missing:
                raise Error("Metal distinct count state is inconsistent")
            values.append(count - missing if output.ignore_nulls else count)
        columns.append(Series(output.name.copy(), Column[Int64](values^)))
    return DataFrame(columns^)
