"""Overflow-checked allocation estimates for the bounded float GPU region.

Bytes describe requested device payloads, not SDK allocator reservations or
other processes. A free-memory check is a preflight estimate, not a guarantee.
"""
from .accel_plan import AccelPlan
from .dtype import DataType

comptime ACCEL_THREADS = 256
comptime ACCEL_MAX_BLOCKS = 1024


@fieldwise_init
struct AccelMemory(Copyable):
    var input_bytes: Int
    var workspace_bytes: Int
    var output_bytes: Int
    var peak_bytes: Int
    var download_bytes: Int
    var blocks: Int
    var launches: Int

    def require_budget(self, budget: Int) raises:
        if budget < 0:
            raise Error("Accelerator memory budget must be nonnegative")
        if self.peak_bytes > budget:
            raise Error(
                "NVIDIA memory preflight rejected: estimated device payload "
                + String(self.peak_bytes)
                + " bytes exceeds budget "
                + String(budget)
                + " bytes (input="
                + String(self.input_bytes)
                + ", workspace="
                + String(self.workspace_bytes)
                + ", output="
                + String(self.output_bytes)
                + ")"
            )

    def describe(self) -> String:
        return (
            "input="
            + String(self.input_bytes)
            + " B; workspace="
            + String(self.workspace_bytes)
            + " B; output="
            + String(self.output_bytes)
            + " B; peak_requested="
            + String(self.peak_bytes)
            + " B; download="
            + String(self.download_bytes)
            + " B; kernels="
            + String(self.launches)
        )


def estimate_memory(
    rows: Int,
    item_bytes: Int,
    has_validity: Bool,
    bit_offset: Int,
    projections: Int,
) raises -> AccelMemory:
    """Pure shape estimate, also usable without a GPU or materialized input."""
    if (
        rows < 0
        or (item_bytes != 4 and item_bytes != 8)
        or bit_offset < 0
        or bit_offset > 7
        or projections < 1
    ):
        raise Error("Invalid accelerator memory shape")
    var bitmap = (
        rows // 8 + (rows % 8 + bit_offset + 7) // 8
    ) if has_validity and rows else 0
    var blocks = max(
        1,
        min(
            ACCEL_MAX_BLOCKS,
            rows // ACCEL_THREADS + Int(rows % ACCEL_THREADS != 0),
        ),
    )
    var workspace = blocks * 16
    var output = 16
    if (
        rows > (Int.MAX - bitmap - workspace - output) // item_bytes
        or projections > Int.MAX // 16
    ):
        raise Error("Accelerator memory estimate overflows Int")
    var input = rows * item_bytes + bitmap
    return AccelMemory(
        input,
        workspace,
        output,
        input + workspace + output,
        projections * 16,
        blocks,
        projections * 2,
    )


def _estimate[D: DType](plan: AccelPlan) raises -> AccelMemory:
    var source = plan.source.numeric[D]()
    return estimate_memory(
        len(source),
        4 if D == DType.float32 else 8,
        len(source._bits[]) != 0,
        source.validity_offset() % 8,
        len(plan.reductions),
    )


def plan_memory(plan: AccelPlan) raises -> AccelMemory:
    if plan.source.dtype() == DataType.FLOAT32:
        return _estimate[DType.float32](plan)
    return _estimate[DType.float64](plan)
