"""Explicit streaming-reduction strategies and their observed decision facts."""
from dataframe.expr import Expr
from std.io import RawFileDescriptor
from std.os import getenv

comptime BATCH_STREAM = 0
comptime MORSEL_PIPELINE = 1
comptime INSERT_PIPELINE = 2
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
    var sample_rows: Int
    var sample_groups: Int
    var group_bound: Int
    var pending_string_sample: Bool
    var code_late_keys: Bool
    var row_holder: Int
    var row_names: List[String]
    var row_unique: Bool
    var row_merge: List[Expr]
    var key_input_rows: Float64
    var key_finish_groups: Float64
    var key_saved_work: Float64
    var key_paid_work: Float64
    var key_cost_reason: String

    def __init__(out self, route: Int, reason: String):
        self.route = route
        self.reason = reason.copy()
        self.eligibility_reason = reason.copy()
        self.sample_rows = -1
        self.sample_groups = -1
        self.group_bound = -1
        self.pending_string_sample = False
        self.code_late_keys = False
        self.row_holder = -1
        self.row_names = List[String]()
        self.row_unique = False
        self.row_merge = List[Expr]()
        self.key_input_rows = -1.0
        self.key_finish_groups = -1.0
        self.key_saved_work = -1.0
        self.key_paid_work = -1.0
        self.key_cost_reason = String()

    def uses_pipeline(self) -> Bool:
        return (
            self.route == MORSEL_PIPELINE
            or self.route == INSERT_PIPELINE
            or self.route == TABLE_PIPELINE
        )

    def inserts_groups(self) -> Bool:
        return self.route == INSERT_PIPELINE

    def uses_aggregate_table(self) -> Bool:
        return self.route == TABLE_PIPELINE

    def is_eager(self) -> Bool:
        return self.route == EAGER_FRAME

    def choose(mut self, route: Int, reason: String):
        self.route = route
        self.reason = reason.copy()

    def choose_insertion(mut self, preferred: Bool, reason: String):
        self.choose(INSERT_PIPELINE if preferred else MORSEL_PIPELINE, reason)

    def name(self) -> String:
        if self.route == BATCH_STREAM:
            return "batch_stream"
        if self.route == MORSEL_PIPELINE:
            return "morsel_pipeline"
        if self.route == INSERT_PIPELINE:
            return "insert_pipeline"
        if self.route == TABLE_PIPELINE:
            return "table_pipeline"
        return "eager_frame"

    def explain(self) -> String:
        var out = self.name() + ": " + self.reason
        out += "; eligibility=" + self.eligibility_reason
        if self.sample_rows >= 0:
            out += "; sampled_groups=" + String(self.sample_groups)
            out += "; sampled_rows=" + String(self.sample_rows)
        if self.group_bound >= 0:
            out += "; group_bound=" + String(self.group_bound)
        if self.key_cost_reason != "":
            out += "; key_cost_reason=" + self.key_cost_reason
            out += "; key_input_rows=" + String(self.key_input_rows)
            out += "; key_finish_groups=" + String(self.key_finish_groups)
            out += "; key_saved_work=" + String(self.key_saved_work)
            out += "; key_paid_work=" + String(self.key_paid_work)
            out += "; key_strategy=" + (
                "row_ids" if self.row_holder >= 0 else "materialized_keys"
            )
        return out


def trace_decision(decision: AggregationDecision):
    """Report the selected policy under the existing opt-in trace setting.

    A separate prefix leaves specialized-path coverage events unchanged.
    """
    if getenv("DATAFRAME_TRACE_PATHS"):
        print(
            "dataframe-strategy:", decision.explain(), file=RawFileDescriptor(2)
        )
