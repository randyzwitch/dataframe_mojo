"""Grouped-key query staged for a backend: the plan, its schema and key domains."""
from .frame import DataFrame
from .lazy import LazyFrame
from .key_domains import KeyDomain


@fieldwise_init
struct KeyQuery(Deinitable, Movable):
    var query: LazyFrame
    var schema: DataFrame
    var domains: List[KeyDomain]
    var names: List[String]
    var rows: Int
    var staging_bytes: Int
    var key_outputs: List[Int]

    def __deinit__(deinit self):
        # Concrete witnesses across PlanNode/KeyQuery/LazyFrame avoid eager
        # documentation trait inference cycles. Owned fields clean up
        # automatically after this body; no owner is released manually.
        pass

    def __init__(out self, *, deinit move: Self):
        # Spell the move explicitly to break LazyFrame/PlanNode/KeyQuery's
        # generated trait inference cycle. The immutable source buffers and
        # Arc owners remain shared; only query metadata is copied.
        self.query = move.query.copy()
        self.schema = move.schema^
        self.domains = move.domains^
        self.names = move.names^
        self.rows = move.rows
        self.staging_bytes = move.staging_bytes
        self.key_outputs = move.key_outputs^
