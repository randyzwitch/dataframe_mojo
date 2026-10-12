"""Exact centered raw moments over integer-valued samples, no dataframe imports.

Accumulate powers around the first valid sample in one scalar pass. Converting
only the final exact rational moments avoids the production online recurrence
and keeps large common offsets out of Int128 powers.
"""
from std.math import abs


struct ExactMoments(Copyable):
    var count: Int
    var origin: Int128
    var s1: Int128
    var s2: Int128
    var s3: Int128
    var s4: Int128
    var magnitude: Float64
    var span: Float64

    def __init__(out self, values: List[Int64], valid: List[Bool]):
        self.count = 0
        self.origin = 0
        self.s1 = 0
        self.s2 = 0
        self.s3 = 0
        self.s4 = 0
        self.magnitude = 0
        self.span = 0
        for row in range(len(values)):
            if not valid[row]:
                continue
            var x = Int128(values[row])
            if self.count == 0:
                self.origin = x
            var d = x - self.origin
            self.s1 += d
            self.s2 += d * d
            self.s3 += d * d * d
            self.s4 += d * d * d * d
            self.count += 1
            self.magnitude += abs(Float64(x))
            self.span = max(self.span, abs(Float64(d)))

    def total(self) -> Int128:
        return self.origin * Int128(self.count) + self.s1

    def mean(self) -> Float64:
        if self.count == 0:
            return 0
        return Float64(self.origin) + Float64(self.s1) / Float64(self.count)

    def central(self, power: Int) -> Float64:
        if self.count == 0:
            return 0
        var n = Int128(self.count)
        if power == 2:
            return Float64(n * self.s2 - self.s1 * self.s1) / Float64(n)
        if power == 3:
            return Float64(
                n * n * self.s3
                - 3 * n * self.s1 * self.s2
                + 2 * self.s1 * self.s1 * self.s1
            ) / Float64(n * n)
        return Float64(
            n * n * n * self.s4
            - 4 * n * n * self.s1 * self.s3
            + 6 * n * self.s1 * self.s1 * self.s2
            - 3 * self.s1 * self.s1 * self.s1 * self.s1
        ) / Float64(n * n * n)

    def moment_budget(self, power: Int) -> Float64:
        # Account for online mean rounding at the sample's common offset,
        # then bound its effect on centered powers by the derivative p*x^(p-1).
        var epsilon = Float64(2.220446049250313e-16)
        var mean_error = (
            Float64(32 * max(1, self.count))
            * epsilon
            * (abs(Float64(self.origin)) + self.span + 1)
        )
        var slope = Float64(power)
        for _ in range(power - 1):
            slope *= max(Float64(1), self.span)
        return Float64(max(1, self.count)) * slope * mean_error + Float64(
            64 * max(1, self.count)
        ) * epsilon * max(Float64(1), abs(self.central(power)))

    def sum_budget(self) -> Float64:
        var steps = Float64(2 * self.count + 1)
        var epsilon = Float64(2.220446049250313e-16)
        return steps * epsilon / (1 - steps * epsilon) * self.magnitude


def exact_comoment(
    xs: List[Int64], ys: List[Int64], valid: List[Bool]
) -> Float64:
    var count = Int128(0)
    var ox = Int128(0)
    var oy = Int128(0)
    var sx = Int128(0)
    var sy = Int128(0)
    var sxy = Int128(0)
    for row in range(len(xs)):
        if not valid[row]:
            continue
        if count == 0:
            ox = Int128(xs[row])
            oy = Int128(ys[row])
        var x = Int128(xs[row]) - ox
        var y = Int128(ys[row]) - oy
        count += 1
        sx += x
        sy += y
        sxy += x * y
    return Float64(count * sxy - sx * sy) / Float64(
        count
    ) if count else Float64(0)
