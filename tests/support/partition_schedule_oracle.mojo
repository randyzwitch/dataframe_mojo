"""Actual CPU OVER dispatcher/layout oracle for isolated GPU metadata probes."""
from dataframe import Series
from dataframe.execution import _dense_partitions, _partition_ids
from dataframe.partition import low_cardinality
from dataframe.window_layout import (
    PartitionLayout,
    partition_layout,
    keyed_layout,
)


struct Expected(Movable):
    var phase: List[UInt32]
    var rank: List[UInt32]
    var offset: List[UInt32]
    var row_phase: List[UInt32]
    var ordinal: List[UInt32]
    var facts: List[UInt32]

    def __init__(out self, rows: Int, groups: Int):
        self.phase = List[UInt32](length=groups, fill=0)
        self.rank = List[UInt32](length=groups, fill=0)
        self.offset = List[UInt32](length=groups, fill=0)
        self.row_phase = List[UInt32](length=rows, fill=0)
        self.ordinal = List[UInt32](length=rows, fill=0)
        self.facts = List[UInt32](length=4, fill=0)


def _cpu_layout(keys: List[Series], lanes: Int) raises -> PartitionLayout:
    var dense = _dense_partitions(keys)
    if dense:
        var groups = dense.take()
        return partition_layout(groups.ids.copy(), groups.count(), lanes)
    return keyed_layout(keys, lanes)


def expected(
    keys: List[Series],
    ids: List[Int],
    representatives: List[Int],
    configured: Int,
    allow_contiguous: Bool,
    cached_layout: Bool = False,
) raises -> Expected:
    var n = len(ids)
    var groups = len(representatives)
    var result = Expected(n, groups)
    var dense = _dense_partitions(keys)
    result.facts[0] = UInt32(Bool(dense))
    var workers = max(1, min(configured, n // 65536))
    result.facts[1] = UInt32(
        True if dense or workers <= 1 else low_cardinality(keys)
    )
    var lanes = min(configured, n // 2048)
    if lanes > 1 and allow_contiguous:
        var layout = _cpu_layout(keys, lanes)
        # This is the production _over_contiguous boundary loop, deliberately
        # independent from the GPU's accumulate-until-target implementation.
        var target = max(2048, (n + 4 * lanes - 1) // (4 * lanes))
        var cuts: List[Int] = [0]
        var since = 0
        for group in range(layout.count):
            if layout.starts[group] - since >= target:
                cuts.append(group)
                since = layout.starts[group]
        cuts.append(layout.count)
        if len(cuts) > 2:
            result.facts[2] = UInt32(len(cuts) - 1)
            result.facts[3] = 1
            for run in range(len(cuts) - 1):
                var lo = layout.starts[cuts[run]]
                for group in range(cuts[run], cuts[run + 1]):
                    var first = layout.starts[group]
                    var original = ids[layout.order[first]]
                    result.phase[original] = UInt32(run)
                    result.rank[original] = UInt32(group - cuts[run])
                    result.offset[original] = UInt32(first - lo)
                    for at in range(first, layout.starts[group + 1]):
                        var row = layout.order[at]
                        result.row_phase[row] = UInt32(run)
                        result.ordinal[row] = UInt32(at - lo)
            return result^
    # Use the actual configured CPU fallback, including its dense, whole-frame
    # and partitioned dispatch. Neither its hash order nor its sample decision
    # is reconstructed from the GPU code in this oracle.
    var fallback_ids = List[Int]()
    if cached_layout:
        var layout = _cpu_layout(keys, max(1, lanes))
        fallback_ids = layout.ids.copy()
    else:
        var fallback = _partition_ids(keys, n)
        fallback_ids = fallback.ids.copy()
    result.facts[2] = 1
    for group in range(groups):
        result.rank[group] = UInt32(fallback_ids[representatives[group]])
    for row in range(n):
        result.ordinal[row] = UInt32(row)
    return result^
