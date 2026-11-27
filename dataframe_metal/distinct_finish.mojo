"""Finish exact counts and both null policies from bounded GPU result state."""
from dataframe import Column, DataFrame, Series
from .distinct_plan import DistinctPlan
from .group_gather import gather_group_keys
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
        columns = gather_group_keys(frame, plan.keys.value(), first, slots)
    for output in plan.outputs:
        var values = List[Int64](capacity=len(slots))
        for slot in slots:
            var at = slot * len(plan.fields) + output.field
            var count = Int64(counts[at]) if counts else Int64(0)
            var missing = Int64(nulls[at]) if nulls else Int64(0)
            if missing > 1 or count < missing:
                raise Error("Metal distinct count state is inconsistent")
            values.append(count - missing if output.ignore_nulls else count)
        columns.append(Series(output.name.copy(), Column[Int64](values^)))
    return DataFrame(columns^)
