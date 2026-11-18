"""Semantic eligibility for morsel pipelines."""
from dataframe.aggregation.expressions import (
    _row_local,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.expr import (
    COL,
    Expr,
    IMPLODE,
    MAX,
    MEDIAN,
    MIN,
    MODE,
    N_UNIQUE,
    QUANTILE,
    SUM,
    col,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.joins.hash_table import (
    JoinHashTable,
)
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
    join_emits_build_rows,
)
from dataframe.planning.cost.model import (
    PipelineEligibility,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from dataframe.planning.model import (
    DROP,
    EXPLODE,
    FILTER,
    JOIN,
    PlanNode,
    SELECT,
    UNNEST,
    WITH_COLUMNS,
)
from std.collections import (
    Optional,
)


def _pipeline_steps(
    operations: List[PlanNode], indexes: List[Optional[JoinHashTable]]
) -> Bool:
    """Whether every operation is a step a pipeline runs: a row-local step,
    an explode or unnest (each morsel's rows through the operator), a cross
    product with its collected build, or a join probing its build. A join's
    morsels probe the shared hash index on their own worker; with no index
    (keys the table does not hold, such as nested values) each morsel
    joins the build's rows on its own, which serves every join that emits
    only probe-side matches. A right, full, right-semi or right-anti join
    also emits build rows by whether any morsel matched them, flags only
    the shared index keeps."""
    for node in operations:
        if (
            node.kind == FILTER
            or node.kind == SELECT
            or node.kind == WITH_COLUMNS
        ):
            if not _row_local(node.exprs):
                return False
        elif node.kind == JOIN:
            if node.how == JOIN_CROSS:
                continue
            if node.how == JOIN_RIGHT and node.coalesce:
                # Its keys take the build rows' values, which a probe
                # step does not write.
                return False
            if node.how not in [
                JOIN_INNER,
                JOIN_LEFT,
                JOIN_SEMI,
                JOIN_ANTI,
                JOIN_RIGHT,
                JOIN_FULL,
                JOIN_RIGHT_SEMI,
                JOIN_RIGHT_ANTI,
            ]:
                return False
            var prepared = node.join_build_slot() < len(indexes) and Bool(
                indexes[node.join_build_slot()]
            )
            if not prepared and join_emits_build_rows(node.how):
                return False
        elif node.kind not in [DROP, EXPLODE, UNNEST]:
            return False
    return True


def pipeline_eligibility(
    top: Int,
    operations: List[PlanNode],
    indexes: List[Optional[JoinHashTable]],
    distinct: Bool,
    key_count: Int,
    ordered: Bool,
    expressions: List[Expr],
) -> PipelineEligibility:
    """Apply the original pipeline guards in their original short-circuit
    order. Any scan is a pipeline's source: an in-memory frame's morsels,
    or a file's batches (`ScanSource`). A slice runs in the pipeline's
    limited sink, a sort's first rows in its top-N sink; a top-N of no rows
    keeps no heap for the pipeline's sink to fill."""
    if top == 0:
        return PipelineEligibility(False, "empty_top_requires_batches")
    if not _pipeline_steps(operations, indexes):
        return PipelineEligibility(False, "build_rows_need_table")
    if ordered and key_count > 0 and _emits_build_rows(operations):
        return PipelineEligibility(False, "ordered_groups_need_join_order")
    return PipelineEligibility(True, "row_local_frame_pipeline")


def _emits_build_rows(operations: List[PlanNode]) -> Bool:
    """Whether some join emits build rows: a right join's rows come in its
    build side's order and a full join's unmatched build rows come after
    its probe rows, while the pipeline emits a join's build rows after
    every probe morsel, in the hash table's order. Groups kept in order of
    first appearance need the join's own order."""
    for node in operations:
        if node.kind == JOIN and join_emits_build_rows(node.how):
            return True
    return False


def consider_inline_table(
    eligibility: PipelineEligibility,
    key_count: Int,
    expression_count: Int,
    ordered: Bool,
    distinct: Bool,
) -> Bool:
    """Whether the existing empty-frame state-eligibility probe is needed.
    `ordered` groupings take the table too: its group rows keep their first
    row's sequence number, and the finish orders by it."""
    return eligibility.allowed and key_count > 0 and expression_count > 0


def _counts_distinct(expressions: List[Expr]) -> Bool:
    for expression in expressions:
        for node in expression._nodes:
            if node.op == N_UNIQUE:
                return True
    return False


def _has_quantile(expressions: List[Expr]) -> Bool:
    """Whether some expression reduces to a median, quantile, mode or
    implode: a state only the aggregate table holds for groups."""
    for expression in expressions:
        for node in expression._nodes:
            if node.op in [MEDIAN, QUANTILE, MODE, IMPLODE]:
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
