"""Central streaming-reduction strategy selection and cost preferences."""
from dataframe.expr import (
    Expr,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.join_type import (
    JOIN_FULL,
    JOIN_RIGHT,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.cost.model import (
    AggregationDecision,
    BATCH_STREAM,
    EAGER_FRAME,
    MORSEL_PIPELINE,
    PipelineEligibility,
    TABLE_PIPELINE,
)
from dataframe.planning.model import (
    JOIN,
    PlanNode,
    SCAN_FRAME,
)
from dataframe.planning.cost.eligibility import (
    _counts_distinct,
    _has_quantile,
    _row_steps_only,
)


def select_aggregation_strategy(
    self: QueryContext,
    cursor: Int,
    batch_size: Int,
    keys: List[String],
    expressions: List[Expr],
    operations: List[PlanNode],
    joins: List[DataFrame],
    distinct: Bool,
    ordered: Bool,
    top: Int,
    grouping: Bool,
    eligibility: PipelineEligibility,
    inline_table_eligible: Bool,
) raises -> AggregationDecision:
    """Choose one route, retaining the original preference and sampling order.

    Semantic pipeline/inline-state eligibility is supplied by the caller;
    group keys are grouped as their values, as DuckDB's aggregate does.
    """
    var choice = AggregationDecision(
        MORSEL_PIPELINE if eligibility.allowed else BATCH_STREAM,
        eligibility.reason,
    )
    if inline_table_eligible:
        choice.choose(TABLE_PIPELINE, "inline_aggregate_states")
    if not choice.uses_aggregate_table() and (
        grouping or _has_quantile(expressions)
    ):
        choice.choose(EAGER_FRAME, "inline_table_required")
        return choice^
    # Right and full joins require their unmatched build rows to be
    # emitted by a morsel pipeline. A row-only in-memory fallback may prefer eager work.
    if not choice.uses_pipeline():
        for operation in operations:
            if operation.kind == JOIN and (
                operation.how == JOIN_RIGHT
                or operation.how == JOIN_FULL
                or operation.how == JOIN_RIGHT_SEMI
                or operation.how == JOIN_RIGHT_ANTI
            ):
                choice.choose(EAGER_FRAME, "right_join_requires_pipeline")
                return choice^
    if (
        not choice.uses_pipeline()
        and len(expressions)
        and self._nodes[cursor].kind == SCAN_FRAME
        and _row_steps_only(operations)
    ):
        if _counts_distinct(expressions):
            choice.choose(EAGER_FRAME, "in_memory_distinct_reduction")
            return choice^
        elif len(keys) > 0:
            # The eager group-by is the aggregate table over the frame.
            choice.choose(EAGER_FRAME, "in_memory_grouping")
            return choice^
    if (
        top >= 0
        and len(operations) == 0
        and self._nodes[cursor].kind == SCAN_FRAME
    ):
        choice.choose(EAGER_FRAME, "in_memory_topk_without_steps")
    return choice^
