"""Central streaming-reduction strategy selection and cost preferences."""
from dataframe.aggregation.expressions import has_aggregate_filter
from dataframe.expr import (
    Expr,
)
from dataframe.planning.cost.model import (
    AggregationDecision,
    EAGER_FRAME,
    MORSEL_PIPELINE,
    PipelineEligibility,
    TABLE_PIPELINE,
)
from dataframe.planning.cost.eligibility import (
    _has_quantile,
)


def select_aggregation_strategy(
    expressions: List[Expr],
    grouping: Bool,
    eligibility: PipelineEligibility,
    inline_table_eligible: Bool,
) raises -> AggregationDecision:
    """The route of a stream's terminal: the aggregate table pipeline when
    it holds every state, else the morsel pipeline with its other sinks
    (ungrouped reductions, top-N, slice, materialize), whatever the source
    (a file's batches, a frame in memory, a computed input). Only work no
    pipeline takes runs on the eager frame: steps a pipeline does not run
    (`pipeline_eligibility`), and grouping sets, grouped quantiles or
    aggregate filters whose states the table does not hold."""
    var choice = AggregationDecision(
        MORSEL_PIPELINE if eligibility.allowed else EAGER_FRAME,
        eligibility.reason,
    )
    if inline_table_eligible:
        choice.choose(TABLE_PIPELINE, "inline_aggregate_states")
    if not choice.uses_aggregate_table() and (
        grouping
        or _has_quantile(expressions)
        or has_aggregate_filter(expressions)
    ):
        choice.choose(EAGER_FRAME, "inline_table_required")
    return choice^
