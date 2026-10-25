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
from std.collections import Dict
from std.os import getenv

from .json_value import JsonValue


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


def load_costs(engine: String) raises -> Dict[String, LinearCost]:
    """Each operator's measured cost on `engine` ("cpu" or "accel"); empty
    when no calibration is configured."""
    var costs = Dict[String, LinearCost]()
    var path = calibration_path()
    if not path:
        return costs^
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
