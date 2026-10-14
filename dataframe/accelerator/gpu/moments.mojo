"""Mergeable statistical states using integer-only binary64 arithmetic.

Values and results are IEEE binary64 bit patterns. Each expression retains
the operation order of reductions.mojo and the Metal implementation.
Callers apply input validity before add and result validity before finish.
"""

from . import floating as f


@fieldwise_init
struct VarianceState(Copyable):
    var count: UInt64
    var mean: UInt64
    var m2: UInt64

    def __init__(out self):
        self.count = 0
        self.mean = 0
        self.m2 = 0

    def add(mut self, value: UInt64):
        self.count += 1
        var delta = f.sub(value, self.mean)
        self.mean = f.add(self.mean, f.div(delta, f.from_unsigned(self.count)))
        self.m2 = f.add(self.m2, f.mul(delta, f.sub(value, self.mean)))

    def merge(mut self, other: Self):
        if other.count == 0:
            return
        if self.count == 0:
            self = other.copy()
            return
        var total = self.count + other.count
        var delta = f.sub(other.mean, self.mean)
        var weight = f.div(f.from_unsigned(other.count), f.from_unsigned(total))
        self.mean = f.add(self.mean, f.mul(delta, weight))
        var correction = f.div(
            f.mul(
                f.mul(f.mul(delta, delta), f.from_unsigned(self.count)),
                f.from_unsigned(other.count),
            ),
            f.from_unsigned(total),
        )
        self.m2 = f.add(self.m2, f.add(other.m2, correction))
        self.count = total

    def valid(self, ddof: UInt64) -> Bool:
        return self.count > ddof

    def finish(self, ddof: UInt64, standard: Bool = False) -> UInt64:
        var value = f.div(self.m2, f.from_unsigned(self.count - ddof))
        return f.sqrt(value) if standard else value


@fieldwise_init
struct CentralMomentState(Copyable):
    var count: UInt64
    var mean: UInt64
    var m2: UInt64
    var m3: UInt64
    var m4: UInt64

    def __init__(out self):
        self.count = 0
        self.mean = 0
        self.m2 = 0
        self.m3 = 0
        self.m4 = 0

    def add(mut self, value: UInt64):
        var before = f.from_unsigned(self.count)
        self.count += 1
        var n = f.from_unsigned(self.count)
        var two = f.from_unsigned(2)
        var three = f.from_unsigned(3)
        var four = f.from_unsigned(4)
        var six = f.from_unsigned(6)
        var delta = f.sub(value, self.mean)
        var dn = f.div(delta, n)
        var dn2 = f.mul(dn, dn)
        var term = f.mul(f.mul(delta, dn), before)
        self.mean = f.add(self.mean, dn)
        var polynomial = f.add(f.sub(f.mul(n, n), f.mul(three, n)), three)
        self.m4 = f.add(
            self.m4,
            f.sub(
                f.add(
                    f.mul(f.mul(term, dn2), polynomial),
                    f.mul(f.mul(six, dn2), self.m2),
                ),
                f.mul(f.mul(four, dn), self.m3),
            ),
        )
        self.m3 = f.add(
            self.m3,
            f.sub(
                f.mul(f.mul(term, dn), f.sub(n, two)),
                f.mul(f.mul(three, dn), self.m2),
            ),
        )
        self.m2 = f.add(self.m2, term)

    def merge(mut self, other: Self):
        if other.count == 0:
            return
        if self.count == 0:
            self = other.copy()
            return
        var a = f.from_unsigned(self.count)
        var b = f.from_unsigned(other.count)
        var n = f.add(a, b)
        var d = f.sub(other.mean, self.mean)
        var d2 = f.mul(d, d)
        var n2 = f.mul(n, n)
        var three = f.from_unsigned(3)
        var four = f.from_unsigned(4)
        var six = f.from_unsigned(6)
        var m2 = f.add(
            f.add(self.m2, other.m2), f.div(f.mul(f.mul(d2, a), b), n)
        )
        var m3 = f.add(
            f.add(
                f.add(self.m3, other.m3),
                f.div(f.mul(f.mul(f.mul(f.mul(d2, d), a), b), f.sub(a, b)), n2),
            ),
            f.div(
                f.mul(
                    f.mul(three, d),
                    f.sub(f.mul(a, other.m2), f.mul(b, self.m2)),
                ),
                n,
            ),
        )
        var polynomial = f.add(f.sub(f.mul(a, a), f.mul(a, b)), f.mul(b, b))
        var m4 = f.add(
            f.add(
                f.add(
                    f.add(self.m4, other.m4),
                    f.div(
                        f.mul(f.mul(f.mul(f.mul(d2, d2), a), b), polynomial),
                        f.mul(n2, n),
                    ),
                ),
                f.div(
                    f.mul(
                        f.mul(six, d2),
                        f.add(
                            f.mul(f.mul(a, a), other.m2),
                            f.mul(f.mul(b, b), self.m2),
                        ),
                    ),
                    n2,
                ),
            ),
            f.div(
                f.mul(
                    f.mul(four, d), f.sub(f.mul(a, other.m3), f.mul(b, self.m3))
                ),
                n,
            ),
        )
        self.mean = f.add(self.mean, f.div(f.mul(d, b), n))
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.count += other.count

    def valid(self, kurtosis: Bool, flags: UInt64) -> Bool:
        """Flags bit 0 enables bias; bit 1 requests Fisher excess kurtosis."""
        return self.count != 0 and (
            flags & 1 != 0
            or self.count >= (UInt64(4) if kurtosis else UInt64(3))
        )

    def finish(self, kurtosis: Bool, flags: UInt64) -> UInt64:
        var n = f.from_unsigned(self.count)
        var one = f.from_unsigned(1)
        var two = f.from_unsigned(2)
        var three = f.from_unsigned(3)
        var six = f.from_unsigned(6)
        if not kurtosis:
            var g = f.div(
                f.mul(f.sqrt(n), self.m3), f.mul(self.m2, f.sqrt(self.m2))
            )
            if flags & 1 != 0:
                return g
            return f.div(
                f.mul(g, f.sqrt(f.mul(n, f.sub(n, one)))), f.sub(n, two)
            )
        var excess = f.sub(
            f.div(f.mul(n, self.m4), f.mul(self.m2, self.m2)), three
        )
        if flags & 1 == 0:
            excess = f.div(
                f.mul(f.add(f.mul(f.add(n, one), excess), six), f.sub(n, one)),
                f.mul(f.sub(n, two), f.sub(n, three)),
            )
        return excess if flags & 2 != 0 else f.add(excess, three)


@fieldwise_init
struct CovarianceState(Copyable):
    var count: UInt64
    var mean_x: UInt64
    var mean_y: UInt64
    var cxy: UInt64
    var m2x: UInt64
    var m2y: UInt64

    def __init__(out self):
        self.count = 0
        self.mean_x = 0
        self.mean_y = 0
        self.cxy = 0
        self.m2x = 0
        self.m2y = 0

    def add(mut self, x: UInt64, y: UInt64):
        self.count += 1
        var n = f.from_unsigned(self.count)
        var dx = f.sub(x, self.mean_x)
        self.mean_x = f.add(self.mean_x, f.div(dx, n))
        var dy = f.sub(y, self.mean_y)
        self.mean_y = f.add(self.mean_y, f.div(dy, n))
        self.cxy = f.add(self.cxy, f.mul(dx, f.sub(y, self.mean_y)))
        self.m2x = f.add(self.m2x, f.mul(dx, f.sub(x, self.mean_x)))
        self.m2y = f.add(self.m2y, f.mul(dy, f.sub(y, self.mean_y)))

    def merge(mut self, other: Self):
        if other.count == 0:
            return
        if self.count == 0:
            self = other.copy()
            return
        var a = f.from_unsigned(self.count)
        var b = f.from_unsigned(other.count)
        var n = f.add(a, b)
        var dx = f.sub(other.mean_x, self.mean_x)
        var dy = f.sub(other.mean_y, self.mean_y)
        self.cxy = f.add(
            self.cxy,
            f.add(other.cxy, f.div(f.mul(f.mul(f.mul(dx, dy), a), b), n)),
        )
        self.m2x = f.add(
            self.m2x,
            f.add(other.m2x, f.div(f.mul(f.mul(f.mul(dx, dx), a), b), n)),
        )
        self.m2y = f.add(
            self.m2y,
            f.add(other.m2y, f.div(f.mul(f.mul(f.mul(dy, dy), a), b), n)),
        )
        self.mean_x = f.add(self.mean_x, f.div(f.mul(dx, b), n))
        self.mean_y = f.add(self.mean_y, f.div(f.mul(dy, b), n))
        self.count += other.count

    def valid(self, correlation: Bool = False) -> Bool:
        # Correlation always produces a scalar, including NaN for no pairs.
        return correlation or self.count != 0

    def finish(self, ddof: UInt64, correlation: Bool = False) -> UInt64:
        if correlation:
            if self.count < 2:
                return f.div(0, 0)
            return f.div(self.cxy, f.sqrt(f.mul(self.m2x, self.m2y)))
        if self.count <= ddof:
            return 0
        return f.div(self.cxy, f.from_unsigned(self.count - ddof))


@fieldwise_init
struct RollingVarianceState(Copyable):
    """Translated sliding Welford state; caller owns bounds and origin.

    A removal returns whether window._moments_bounds would rebuild. Rebuilds
    must reset the state and choose the first finite value as the new origin.
    """

    var count: UInt64
    var bad: UInt64
    var mean: UInt64
    var m2: UInt64

    def __init__(out self):
        self.count = 0
        self.bad = 0
        self.mean = 0
        self.m2 = 0

    def add(mut self, value: UInt64, origin: UInt64):
        if f.absolute(value) >= f.INF:
            self.bad += 1
            return
        var translated = f.sub(value, origin)
        self.count += 1
        var delta = f.sub(translated, self.mean)
        self.mean = f.add(self.mean, f.div(delta, f.from_unsigned(self.count)))
        self.m2 = f.fma(delta, f.sub(translated, self.mean), self.m2)

    def remove(mut self, value: UInt64, origin: UInt64) -> Bool:
        if f.absolute(value) >= f.INF:
            self.bad -= 1
            return False
        var translated = f.sub(value, origin)
        var previous_m2 = self.m2
        self.count -= 1
        if self.count == 0:
            self.mean = 0
            self.m2 = 0
        else:
            var delta = f.sub(translated, self.mean)
            self.mean = f.sub(
                self.mean, f.div(delta, f.from_unsigned(self.count))
            )
            self.m2 = f.fma(
                delta ^ f.SIGN, f.sub(translated, self.mean), self.m2
            )
            # maximumNumber maps NaN, negative values and either zero to +0.
            if not f.less(0, self.m2):
                self.m2 = 0
        return (
            self.count == 0
            or f.absolute(previous_m2) >= f.INF
            or f.less(self.m2, f.mul(previous_m2, 0x3FE0000000000000))
        )

    def valid(self, needed: UInt64, ddof: UInt64) -> Bool:
        var total = self.count + self.bad
        return total >= needed and total > ddof

    def finish(self, ddof: UInt64, standard: Bool = False) -> UInt64:
        if self.bad != 0:
            return f.invalid()
        var value = f.div(
            self.m2 if f.less(0, self.m2) else UInt64(0),
            f.from_unsigned(self.count - ddof),
        )
        return f.sqrt(value) if standard else value
