"""Measured costs for placing handed-off nodes under `engine="auto"`.

When the executor reaches a node a provider can run (`LazyFrame.
_handoff_region`), the provider names the operator (a key such as
`filter_sum:int32:filtered`) and estimates its own time for the input's
rows, data movement included. The executor estimates the CPU's time for
the same key from this machine's calibration and runs the node wherever
the estimate is lower. Both estimates are linear in rows, `fixed_ns +
row_ns * rows`, fitted from measurements on this machine; no row count is
chosen by hand, so faster hardware moves the decision without code
changes.

The calibration is a JSON file named by DATAFRAME_HANDOFF_CALIBRATION:

    {"operators": {"filter_sum:int32:filtered":
        {"cpu": {"fixed_ns": 2.0e4, "row_ns": 0.34},
         "accel": {"fixed_ns": 2.1e6, "row_ns": 0.23}}, ...}}

`cpu` is read by the executor and `accel` by the installed provider, which
measures and names its own operators.

Without it, nothing is handed off automatically and `engine="auto"` runs
on the CPU as it always has.
"""
from std.atomic import Atomic
from std.collections import Dict
from std.ffi import _get_global
from std.memory import Pointer
from std.memory.alloc import Layout, alloc
from std.os import getenv

from dataframe.json.value import JsonValue


struct LinearCost(Copyable, Movable):
    """`fixed_ns + row_ns * rows` nanoseconds."""

    var fixed_ns: Float64
    var row_ns: Float64

    def __init__(out self, fixed_ns: Float64, row_ns: Float64):
        self.fixed_ns = fixed_ns
        self.row_ns = row_ns

    def ns(self, rows: Int) -> Float64:
        return self.fixed_ns + self.row_ns * Float64(rows)


def calibration_path() -> String:
    return getenv("DATAFRAME_HANDOFF_CALIBRATION")


def _number(value: JsonValue) raises -> Float64:
    return Float64(value.canonical())


struct _CostCache(Movable):
    """The calibration last read, by path, so a query reads the file once
    however many nodes it offers."""

    var lock: Atomic[Int64]
    var path: String
    var costs: Dict[String, Dict[String, LinearCost]]

    def __init__(out self):
        self.lock = Atomic[Int64](0)
        self.path = String()
        self.costs = Dict[String, Dict[String, LinearCost]]()


def _cache_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    # Leaked on purpose: the cache lives for the whole process.
    var slot = alloc(Layout[_CostCache].single()).unsafe_leak()
    slot.unsafe_write(_CostCache())
    return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(slot))


def _cache_keep(address: Optional[Pointer[NoneType, MutUntrackedOrigin]]):
    pass


def load_costs(engine: String) raises -> Dict[String, LinearCost]:
    """Each operator's measured cost on `engine` ("cpu" or "accel"); empty
    when no calibration is configured. The file is read once per path;
    nodes offered at once, on several workers, share it under a lock."""
    var path = calibration_path()
    if not path:
        return Dict[String, LinearCost]()
    var address = _get_global[
        "DATAFRAME_HANDOFF_COSTS", _cache_init, _cache_keep
    ]()
    ref cache = Pointer[_CostCache, MutAnyOrigin](
        unsafe_from_address=Int(address.value())
    )[]
    var unlocked = Int64(0)
    while not cache.lock.compare_exchange(unlocked, 1):
        unlocked = 0
    try:
        if cache.path != path:
            cache.costs = Dict[String, Dict[String, LinearCost]]()
            cache.costs["cpu"] = _read_costs(path, "cpu")
            cache.costs["accel"] = _read_costs(path, "accel")
            cache.path = path
        var found = cache.costs.get(engine)
        cache.lock.store(0)
        return found.value().copy() if found else Dict[String, LinearCost]()
    except error:
        cache.lock.store(0)
        raise error^


def _read_costs(
    path: String, engine: String
) raises -> Dict[String, LinearCost]:
    var costs = Dict[String, LinearCost]()
    var text: String
    with open(path, "r") as source:
        text = source.read()
    var operators = JsonValue(text).get("operators")
    for key in operators.keys():
        var entry = operators.get(key)
        if not entry.has(engine):
            continue
        var model = entry.get(engine)
        var cost = LinearCost(
            _number(model.get("fixed_ns")), _number(model.get("row_ns"))
        )
        if cost.fixed_ns < 0 or cost.row_ns < 0:
            raise Error(
                "handoff calibration " + key + ": costs must be nonnegative"
            )
        costs[key] = cost^
    return costs^


def crossover_rows(cpu: LinearCost, other: LinearCost) -> Int:
    """The row count from which `other` is estimated faster than the CPU
    for every larger input, or -1 when no such count exists."""
    if other.row_ns >= cpu.row_ns:
        return -1
    var rows = (other.fixed_ns - cpu.fixed_ns) / (cpu.row_ns - other.row_ns)
    return max(0, Int(rows) + 1)
