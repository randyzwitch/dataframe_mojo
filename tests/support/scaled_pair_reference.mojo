"""Exact sufficient statistics of binary-exact quarter-valued fixture pairs."""
from std.math import sqrt


struct ScaledPairs(Copyable):
    var count: Int128
    var sx: Int128
    var sy: Int128
    var sxx: Int128
    var syy: Int128
    var sxy: Int128

    def __init__(out self):
        self.count = 0
        self.sx = 0
        self.sy = 0
        self.sxx = 0
        self.syy = 0
        self.sxy = 0

    def add(mut self, x_quarters: Int, y_quarters: Int):
        var x = Int128(x_quarters)
        var y = Int128(y_quarters)
        self.count += 1
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.syy += y * y
        self.sxy += x * y

    def variance(self, ddof: Int) -> Float64:
        return (
            Float64(self.sxx * self.count - self.sx * self.sx)
            / Float64(self.count * (self.count - Int128(ddof)))
            / 16
        )

    def covariance(self, ddof: Int) -> Float64:
        return (
            Float64(self.sxy * self.count - self.sx * self.sy)
            / Float64(self.count * (self.count - Int128(ddof)))
            / 16
        )

    def correlation(self) -> Float64:
        return Float64(self.sxy * self.count - self.sx * self.sy) / sqrt(
            Float64(self.sxx * self.count - self.sx * self.sx)
            * Float64(self.syy * self.count - self.sy * self.sy)
        )
