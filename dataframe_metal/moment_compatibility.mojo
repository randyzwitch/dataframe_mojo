"""Route ill-conditioned statistical finishing through the CPU executor.

Exact GPU moments remain useful for well-conditioned values. A large shared
offset can make CPU centering sensitive to Float64 rounding; those values
must use the CPU implementation rather than a different numerical answer.
"""
from std.math import sqrt
from std.bit import count_leading_zeros
from .accelerator.faults import data_fault
from .moment_finish import MomentMagnitude

comptime MOMENT_CPU_TOLERANCE = Float64(1e-10)
comptime MOMENT_SMALL_ACCUMULATION_ROWS = 65536


def require_cpu_moment_precision(squares: List[Int128], count: Int) raises:
    """Adjudicate large, high-precision second-moment accumulations.

    Raw square bins are nonnegative sums of individual 32-bit words, before
    carrying between bins. Zero lower bins therefore give a common binary
    quantum for every contribution. A span of at most 53 bits proves that
    all partial sums of these squared values are exactly representable in
    Float64. Larger spans can expose CPU running-M2 accumulation drift even
    when the mean is zero; centering sensitivity alone does not detect it.
    This proof concerns square summation, not the entire Welford recurrence.
    """
    if count <= MOMENT_SMALL_ACCUMULATION_ROWS:
        return
    var magnitude = MomentMagnitude.from_bins(squares)
    var used = magnitude.used()
    if not used:
        return
    var first = 0
    while first < len(squares) and squares[first] == 0:
        first += 1
    var width = (
        (used - 1) * 32
        + 32
        - Int(count_leading_zeros(magnitude.digits[used - 1]))
    )
    if width - first * 32 > 53:
        raise data_fault("variance accumulation needs CPU adjudication")


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
