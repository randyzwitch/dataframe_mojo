"""Explicit streaming-reduction strategies and their observed decision facts."""
from std.io import stderr
from std.os import getenv

comptime BATCH_STREAM = 0
comptime MORSEL_PIPELINE = 1
comptime TABLE_PIPELINE = 3
comptime EAGER_FRAME = 4


@fieldwise_init
struct PipelineEligibility(Copyable):
    """Semantic permission to use morsels, independent of cost preferences."""

    var allowed: Bool
    var reason: String


struct AggregationDecision(Copyable, Movable):
    """One strategy, a reason, and the facts that selected it.

    Pipeline/table/insertion/eager flags are derived from one route, so
    contradictory sink combinations cannot survive selection.
    """

    var route: Int
    var reason: String
    var eligibility_reason: String

    def __init__(out self, route: Int, reason: String):
        self.route = route
        self.reason = reason.copy()
        self.eligibility_reason = reason.copy()

    def uses_pipeline(self) -> Bool:
        return self.route == MORSEL_PIPELINE or self.route == TABLE_PIPELINE

    def uses_aggregate_table(self) -> Bool:
        return self.route == TABLE_PIPELINE

    def is_eager(self) -> Bool:
        return self.route == EAGER_FRAME

    def choose(mut self, route: Int, reason: String):
        self.route = route
        self.reason = reason.copy()

    def name(self) -> String:
        if self.route == BATCH_STREAM:
            return "batch_stream"
        if self.route == MORSEL_PIPELINE:
            return "morsel_pipeline"
        if self.route == TABLE_PIPELINE:
            return "table_pipeline"
        return "eager_frame"

    def explain(self) -> String:
        var out = self.name() + ": " + self.reason
        out += "; eligibility=" + self.eligibility_reason
        return out


def trace_decision(decision: AggregationDecision):
    """Report the selected policy under the existing opt-in trace setting.

    A separate prefix leaves specialized-path coverage events unchanged.
    """
    if getenv("DATAFRAME_TRACE_PATHS"):
        print("dataframe-strategy:", decision.explain(), file=stderr)
