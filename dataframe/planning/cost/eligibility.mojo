"""Semantic eligibility for morsel pipelines."""
from dataframe.aggregation.expressions import (
    _row_local,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.expr import (
    ARG_MAX,
    ARG_MIN,
    COL,
    Expr,
    FIRST,
    LAST,
    MAX,
    MEDIAN,
    MIN,
    N_UNIQUE,
    QUANTILE,
    SUM,
    col,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.join_hash import (
    PreparedHashIndex,
)
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_SEMI,
)
from dataframe.planning.cost.model import (
    PipelineEligibility,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from dataframe.planning.model import (
    DROP,
    FILTER,
    JOIN,
    PlanNode,
    SCAN_FRAME,
    SELECT,
    WITH_COLUMNS,
)
from std.collections import (
    Optional,
)


def _pipeline_steps(
    operations: List[PlanNode], indexes: List[Optional[PreparedHashIndex]]
) -> Bool:
    """Whether every operation is a step a pipeline runs: a row-local
    step, or an inner, left, semi, anti or (uncoalesced) right join whose
    build index was prepared (its morsels probe the shared index on their
    own worker)."""
    for node in operations:
        if (
            node.kind == FILTER
            or node.kind == SELECT
            or node.kind == WITH_COLUMNS
        ):
            if not _row_local(node.exprs):
                return False
        elif node.kind == JOIN:
            if node.how == JOIN_RIGHT:
                # Only without coalescing: its unmatched build rows come
                # out as a final morsel with the probe columns null.
                if node.coalesce:
                    return False
            elif node.how not in [JOIN_INNER, JOIN_LEFT, JOIN_SEMI, JOIN_ANTI]:
                return False
            if node.offset >= len(indexes) or not indexes[node.offset]:
                return False
        elif node.kind != DROP:
            return False
    return True


def pipeline_eligibility(
    source_kind: Int,
    top: Int,
    limit: Int,
    skip: Int,
    operations: List[PlanNode],
    indexes: List[Optional[PreparedHashIndex]],
    distinct: Bool,
    key_count: Int,
    join_count: Int,
    ordered: Bool,
    expressions: List[Expr],
) -> PipelineEligibility:
    """Apply the original pipeline guards in their original short-circuit order."""
    if source_kind != SCAN_FRAME:
        return PipelineEligibility(False, "source_requires_batches")
    if not (top < 0 and limit < 0 and skip == 0 or top > 0):
        return PipelineEligibility(False, "ordered_slice_requires_batches")
    if not _pipeline_steps(operations, indexes):
        return PipelineEligibility(False, "operation_requires_batches")
    if not (not distinct or key_count > 0 and join_count > 0):
        return PipelineEligibility(False, "distinct_without_probe_join")
    if not (join_count == 0 or key_count == 0 or not ordered):
        return PipelineEligibility(False, "ordered_grouped_join")
    if not (key_count > 0 or not _order_sensitive(expressions)):
        return PipelineEligibility(False, "positional_global_reduction")
    return PipelineEligibility(True, "row_local_frame_pipeline")


def consider_inline_table(
    eligibility: PipelineEligibility,
    key_count: Int,
    expression_count: Int,
    ordered: Bool,
    distinct: Bool,
) -> Bool:
    """Whether the existing empty-frame state-eligibility probe is needed."""
    return (
        eligibility.allowed
        and key_count > 0
        and expression_count > 0
        and not ordered
        and not distinct
    )


def _counts_distinct(expressions: List[Expr]) -> Bool:
    for expression in expressions:
        for node in expression._nodes:
            if node.op == N_UNIQUE:
                return True
    return False


def _order_sensitive(expressions: List[Expr]) -> Bool:
    """Whether a reduction's answer depends on the order rows arrive
    (first, last, the position of a minimum or maximum): its states
    merge by row order, which workers taking morsels from a shared
    cursor do not keep."""
    for expression in expressions:
        for node in expression._nodes:
            if node.op in [FIRST, LAST, ARG_MIN, ARG_MAX]:
                return True
    return False


def _has_quantile(expressions: List[Expr]) -> Bool:
    """Whether some expression reduces to a median or quantile."""
    for expression in expressions:
        for node in expression._nodes:
            if node.op == MEDIAN or node.op == QUANTILE:
                return True
    return False


def _row_steps_only(operations: List[PlanNode]) -> Bool:
    for operation in operations:
        if operation.kind not in [FILTER, SELECT, WITH_COLUMNS, DROP]:
            return False
    return True


def _filters(operations: List[PlanNode]) -> Bool:
    for operation in operations:
        if operation.kind == FILTER:
            return True
    return False


def _key_read(name: String, expressions: List[Expr]) -> Bool:
    """Whether a reduction reads the key `name` (or any column)."""
    for expression in expressions:
        var reads = _references(expression)
        if not reads or name in reads.value():
            return True
    return False


def _row_merge_reductions(expressions: List[Expr]) -> Optional[List[Expr]]:
    """The reductions that merge groups' partial results, when each of
    `expressions` is a sum (skipping nulls, or null when none is valid), a
    minimum or a maximum: the same reduction over the partial results.
    None when one is anything else."""
    var merge = List[Expr]()
    for e in expressions:
        var root = e._nodes[len(e._nodes) - 1].copy()
        var again = col(e._name)
        if root.op == SUM and root.min_count <= 1:
            again = again.sum(min_count=root.min_count)
        elif root.op == MIN:
            again = again.min()
        elif root.op == MAX:
            again = again.max()
        else:
            return None
        merge.append(again.alias(e._name))
    return Optional(merge^)


def _late_key_build(
    name: String,
    source_columns: List[String],
    builds: List[DataFrame],
    operations: List[PlanNode],
    expressions: List[Expr],
) raises -> Int:
    """The build frame a group key `name` comes from when the key can be
    grouped by its codes instead of its values: a String column of exactly
    one inner or left join's build side, not in the source, not a join key,
    read by no step or reduction (a plain pass-through select aside) and
    made by none. -1 otherwise."""
    if name in source_columns:
        return -1
    var holder = -1
    for b in range(len(builds)):
        if name in builds[b].columns():
            if holder >= 0:
                return -1
            holder = b
    if holder < 0:
        return -1
    if builds[holder].column(name).dtype() != DataType.STRING:
        return -1
    for expression in expressions:
        var reads = _references(expression)
        if not reads or name in reads.value():
            return -1
    for operation in operations:
        if operation.kind == JOIN:
            if name in operation.names:
                return -1
            if operation.offset == holder:
                if operation.how != JOIN_INNER and operation.how != JOIN_LEFT:
                    return -1
                if name in operation.right_keys:
                    return -1
            continue
        if operation.kind == DROP:
            if name in operation.names:
                return -1
            continue
        for e in operation.exprs:
            # A select passing the column through unchanged is the one
            # step that may read or name it.
            var passes = (
                operation.kind == SELECT
                and len(e._nodes) == 1
                and e._nodes[0].op == COL
                and e._nodes[0].text == name
                and e._name == name
            )
            if passes:
                continue
            var reads = _references(e)
            if not reads or name in reads.value() or e._name == name:
                return -1
    return holder
