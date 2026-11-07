"""Explicit streaming-reduction strategies and their observed decision facts."""
from dataframe.expr import Expr
from std.io import RawFileDescriptor
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
    """One strategy, a reason, and the facts/key plan that selected it.

    Pipeline/table/insertion/eager flags are derived from one route, so
    contradictory sink combinations cannot survive selection.
    """

    var route: Int
    var reason: String
    var eligibility_reason: String
    var code_late_keys: Bool
    var row_holder: Int
    var row_names: List[String]
    var row_unique: Bool
    var row_merge: List[Expr]

    def __init__(out self, route: Int, reason: String):
        self.route = route
        self.reason = reason.copy()
        self.eligibility_reason = reason.copy()
        self.code_late_keys = False
        self.row_holder = -1
        self.row_names = List[String]()
        self.row_unique = False
        self.row_merge = List[Expr]()

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
        print(
            "dataframe-strategy:", decision.explain(), file=RawFileDescriptor(2)
        )
