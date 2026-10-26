"""Value types the planner owns and a backend only reads.

PlanEstimate is what a backend reports for automatic placement. Backends
import it from here; nothing in this module knows a backend exists.
"""


@fieldwise_init
struct PlanEstimate(Copyable):
    var cpu: List[Float64]
    var accelerator: List[Float64]
    var peak_bytes: Int
    var family: String
    var placement: String
