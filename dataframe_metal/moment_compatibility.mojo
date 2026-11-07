"""Route ill-conditioned statistical finishing through the CPU executor.

Exact GPU moments remain useful for well-conditioned values. A large shared
offset can make CPU centering sensitive to Float64 rounding; those values
must use the CPU implementation rather than a different numerical answer.
"""
from std.math import sqrt
from .accelerator.faults import data_fault
from .moment_finish import MomentMagnitude

comptime MOMENT_CPU_TOLERANCE = Float64(1e-10)


def require_cpu_moment_compatibility(
    sums: List[Int128],
    count: Int,
    ddof: Int,
    unit_exponent: Int,
    variance: Float64,
) raises:
    """Adjudicate centering-sensitive values without rereading input rows.

    This is a conservative conditioning guard, not a universal error bound
    for the CPU's accumulation order. Cross-engine contracts also check
    ordinary completions against the CPU within the public tolerance.
    """
    if count <= 1 or count <= ddof:
        return
    var magnitude = MomentMagnitude.from_bins(sums)
    var mean = magnitude.scaled(unit_exponent) / Float64(count)
    if variance == 0:
        # Constant Float32 values have at most 24 significant bits. Integer
        # values within 2**46 also leave room for an exact 128-value block
        # sum in Float64. Larger constant integers need CPU centering.
        if unit_exponent == -149 or mean <= Float64(Int64(1) << 46):
            return
        raise data_fault("variance centering needs CPU adjudication")
    var population = variance * Float64(count - ddof) / Float64(count)
    var spread = sqrt(population)
    # Allow a 128-operation centering margin, matching the CPU block size.
    # Keep this deliberately conservative for grouped running moments too.
    var resolution_margin = mean * Float64(2.220446049250313e-16) * 128
    if resolution_margin > spread * MOMENT_CPU_TOLERANCE:
        raise data_fault("variance centering needs CPU adjudication")
