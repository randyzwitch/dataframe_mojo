"""Backend selection before executor-specific planning or resource setup.

This module has no accelerator imports. Availability is the first gate;
operation/type/semantic and memory checks belong to a backend's lowering
once that backend exists. CPU scheduling remains owned by the CPU executor.
"""

from ._accel_provider import installed


struct _BackendDecision(Movable):
    var engine: String
    var available: Bool
    var reason: String

    def __init__(out self, engine: String, available: Bool, reason: String):
        self.engine = engine
        self.available = available
        self.reason = reason

    def require_available(self) raises:
        if not self.available:
            raise Error(self.reason)

    def describe(self) -> String:
        return "ENGINE " + self.engine + ": " + self.reason + "\n"


def _select_backend(engine: String) raises -> _BackendDecision:
    if engine == "cpu":
        return _BackendDecision("cpu", True, "explicit CPU execution")
    if engine == "auto":
        return _BackendDecision(
            "cpu",
            True,
            "auto uses CPU; automatic accelerator selection is not enabled",
        )
    if engine == "accel":
        return _BackendDecision(
            "accel",
            installed(),
            "registered accelerator provider" if installed() else "Accelerator provider is not installed; use the optional GPU package or accelerator=runtime",
        )
    raise Error(
        "Unknown engine '" + engine + "'; expected 'cpu', 'auto', or 'accel'"
    )
