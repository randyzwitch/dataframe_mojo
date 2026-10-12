"""Pairwise row filtering, centered sums, and quadratic average-rank oracle."""
from std.math import sqrt, isnan


def average_ranks(values: List[Float64]) -> List[Float64]:
    var result = List[Float64]()
    for value in values:
        var below = 0
        var equal = 0
        for other in values:
            below += Int(other < value)
            equal += Int(other == value)
        result.append(Float64(below) + (Float64(equal) + 1) / 2)
    return result^


def centered_comoment(
    xs: List[Float64], ys: List[Float64]
) -> Tuple[Float64, Float64, Float64]:
    var total_x = 0.0
    var total_y = 0.0
    for i in range(len(xs)):
        total_x += xs[i]
        total_y += ys[i]
    var mean_x = total_x / Float64(max(1, len(xs)))
    var mean_y = total_y / Float64(max(1, len(xs)))
    var xx = 0.0
    var yy = 0.0
    var xy = 0.0
    for i in range(len(xs)):
        var dx = xs[i] - mean_x
        var dy = ys[i] - mean_y
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy
    return (xx, yy, xy)


struct PairReference(Copyable):
    var count: Int
    var pearson: Float64
    var spearman: Float64
    var comoment: Float64

    def __init__(
        out self,
        xs: List[Float64],
        ys: List[Float64],
        xv: List[Bool],
        yv: List[Bool],
        rows: List[Int],
    ):
        var x = List[Float64]()
        var y = List[Float64]()
        var nan = False
        for row in rows:
            if xv[row] and yv[row]:
                x.append(xs[row])
                y.append(ys[row])
                nan = nan or isnan(xs[row]) or isnan(ys[row])
        self.count = len(x)
        var moments = centered_comoment(x, y)
        self.comoment = moments[2]
        self.pearson = moments[2] / sqrt(moments[0] * moments[1])
        if self.count < 2 or nan:
            self.pearson = Float64(0) / Float64(0)
            self.spearman = self.pearson
        else:
            var ranks = centered_comoment(average_ranks(x), average_ranks(y))
            self.spearman = ranks[2] / sqrt(ranks[0] * ranks[1])

    def covariance(self, ddof: Int) -> Float64:
        return 0.0 if self.count <= ddof else self.comoment / Float64(
            self.count - ddof
        )
