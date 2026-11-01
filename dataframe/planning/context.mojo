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


struct FrameSlots(Copyable, Movable, Sized):
    """A plan's in-memory tables, by slot, shared by every copy of the plan.

    A table never changes once it holds a slot; plans only add slots. So a
    copy of the plan shares the list, and the copy that adds a slot while
    another shares it copies the list first. Building a plan copies it at
    every step (`push`, a join's two sides), and execution copies it for
    each job: a copy that copied every table's columns made TPC-DS q14's
    57 tables (1,555 columns) cost 214 us of a 281 us plan copy. Polars'
    plans and DuckDB's relations hold their tables by shared pointer.
    """

    var _shared: ArcPointer[List[DataFrame]]

    def __init__(out self, var frames: List[DataFrame]):
        self._shared = ArcPointer(frames^)

    def __len__(self) -> Int:
        return len(self._shared[])

    def __getitem__(self, slot: Int) -> ref[MutAnyOrigin] DataFrame:
        """The table in `slot`, to read: no plan changes a table in place."""
        debug_assert(0 <= slot < len(self._shared[]), "frame slot out of range")
        return Pointer[DataFrame, MutAnyOrigin](
            unsafe_from_address=Int(
                self._shared[].unsafe_ptr().unsafe_offset(slot)
            )
        )[]

    def append(mut self, var frame: DataFrame):
        """Add a slot, copying the list first if another plan shares it."""
        if self._shared.count() > 1:
            self._shared = ArcPointer(self._shared[].copy())
        self._shared[].append(frame^)


struct QueryContext(Copyable, Deinitable, Movable):
    """Plan storage and per-run state, with explicit runtime ports."""

    var _nodes: List[PlanNode]
    var _frames: FrameSlots
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
    # How `_execute` offers nodes to the accelerator provider, which may run
    # them on the columns the CPU computed below (`_handoff_region`):
    # HANDOFF_OFF, HANDOFF_FORCED (every node the provider supports, for
    # engine="accel"), or HANDOFF_AUTO (where measured costs favor it).
    var _handoff: Int
    # Whether the logical optimizer passes already ran on these nodes:
    # a shared subplan cut from an optimized plan is collected without
    # optimizing it again, as Polars optimizes a cached subplan with the
    # plan around it and DuckDB plans a materialized CTE once.
    var _optimized: Bool

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
        self._frames = FrameSlots(frames^)
        self._schemas = schemas^
        self._report = None
        self._origin_rows = Dict[Int, Int]()
        self._accelerator_settings = None
        self._schema_memo = None
        self._prefetched = None
        self._handoff = HANDOFF_OFF
        self._optimized = False
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
