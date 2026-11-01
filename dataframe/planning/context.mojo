"""Shared query state and runtime ports supplied by the public API."""
from dataframe.planning.node_builders import (
    _frame_scan_node,
)
from dataframe.execution_report import (
    ExecutionReport,
)
from dataframe.execution_settings import (
    ExecutionSettings,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.io.csv import (
    CsvSchema,
)
from dataframe.kernels.join_hash import (
    PreparedHashIndex,
)
from dataframe.planning.model import (
    HANDOFF_OFF,
    PlanNode,
    SCAN_FRAME,
)
from std.collections import (
    Dict,
    Optional,
)
from std.memory import (
    ArcPointer,
)


struct _ReadyIndex(Copyable, Movable):
    """A prefetched build side's hash index: the keys it hashes and the
    probe rows bound it was made for (`finish_build`)."""

    var keys: List[String]
    var probe_rows: Int
    var index: PreparedHashIndex

    def __init__(
        out self,
        keys: List[String],
        probe_rows: Int,
        var index: PreparedHashIndex,
    ):
        self.keys = keys.copy()
        self.probe_rows = probe_rows
        self.index = index^


struct QueryContext(Copyable, Deinitable, Movable):
    """Plan storage and per-run state, with explicit runtime ports."""

    var _nodes: List[PlanNode]
    var _frames: List[DataFrame]
    var _schemas: List[Optional[CsvSchema]]
    # Observed execution counters (#439), attached by `profile` and shared
    # by every copy of the plan made while it runs; None otherwise.
    var _report: Optional[ArcPointer[ExecutionReport]]
    # For a frame slot holding a shared subplan's result: the rows of the
    # table that subplan narrowed, so the join order still sees how
    # selective it is (`_spine_rows`, `_selectivity`).
    var _origin_rows: Dict[Int, Int]
    # Provider-neutral envelope on dispatch copies; user plans stay unchanged.
    var _accelerator_settings: Optional[ExecutionSettings]
    # Each node's empty result (its output names and types) while an
    # optimizer pass reads them without rewiring the plan, as Polars' IR
    # and DuckDB's logical operators keep their schemas; cleared by every
    # rewrite (`_reorder`), None outside those passes.
    var _schema_memo: Optional[ArcPointer[Dict[Int, DataFrame]]]
    # Results computed ahead of the node that reads them, by node: a
    # stream's breaker inputs run at once before its joins build
    # (`_prefetch_breakers`), and `_execute` takes each result out once.
    # Shared by the copies a run makes; None outside `collect`.
    var _prefetched: Optional[ArcPointer[Dict[Int, DataFrame]]]
    # The hash indexes made with prefetched build sides, by node, beside
    # the frames in `_prefetched`; the stream takes one with its frame.
    var _prefetched_indexes: Optional[ArcPointer[Dict[Int, _ReadyIndex]]]
    # How `_execute` offers nodes to the accelerator provider, which may run
    # them on the columns the CPU computed below (`_handoff_region`):
    # HANDOFF_OFF, HANDOFF_FORCED (every node the provider supports, for
    # engine="accel"), or HANDOFF_AUTO (where measured costs favor it).
    var _handoff: Int

    var _executor: def(
        QueryContext, Int, Bool, Bool, Int
    ) raises thin -> DataFrame
    var _collector: def(QueryContext, Bool, Bool, Int) raises thin -> DataFrame
    var _supports: def(QueryContext) thin -> Bool
    var _estimate: def(QueryContext, Int) thin -> Tuple[String, Float64]
    var _handoff_execute: def(QueryContext) raises thin -> DataFrame

    def __init__(
        out self,
        var nodes: List[PlanNode],
        var frames: List[DataFrame],
        var schemas: List[Optional[CsvSchema]],
        executor: def(
            QueryContext, Int, Bool, Bool, Int
        ) raises thin -> DataFrame,
        collector: def(QueryContext, Bool, Bool, Int) raises thin -> DataFrame,
        supports: def(QueryContext) thin -> Bool,
        estimate: def(QueryContext, Int) thin -> Tuple[String, Float64],
        handoff_execute: def(QueryContext) raises thin -> DataFrame,
    ):
        self._nodes = nodes^
        self._frames = frames^
        self._schemas = schemas^
        self._report = None
        self._origin_rows = Dict[Int, Int]()
        self._accelerator_settings = None
        self._schema_memo = None
        self._prefetched = None
        self._prefetched_indexes = None
        self._handoff = HANDOFF_OFF
        self._executor = executor
        self._collector = collector
        self._supports = supports
        self._estimate = estimate
        self._handoff_execute = handoff_execute

    def with_input(self, frame: DataFrame) -> Self:
        return Self(
            [_frame_scan_node(slot=0)],
            [frame.copy()],
            [Optional[CsvSchema]()],
            self._executor,
            self._collector,
            self._supports,
            self._estimate,
            self._handoff_execute,
        )

    def push(self, var node: PlanNode) -> Self:
        var result = self.copy()
        node.left = len(result._nodes) - 1
        result._nodes.append(node^)
        return result^

    def _execute(
        self,
        index: Int,
        empty: Bool,
        streaming: Bool = False,
        batch_size: Int = 65536,
    ) raises -> DataFrame:
        return self._executor(self, index, empty, streaming, batch_size)

    def _collect(
        self,
        *,
        optimize: Bool,
        streaming: Bool,
        batch_size: Int,
    ) raises -> DataFrame:
        return self._collector(self, optimize, streaming, batch_size)


def _context_supports(plan: QueryContext) -> Bool:
    return plan._supports(plan)


def _context_estimate(plan: QueryContext, rows: Int) -> Tuple[String, Float64]:
    return plan._estimate(plan, rows)


def _context_handoff(plan: QueryContext) raises -> DataFrame:
    return plan._handoff_execute(plan)
