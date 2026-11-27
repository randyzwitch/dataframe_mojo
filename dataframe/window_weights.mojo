"""Weighted fixed-size rolling windows.

A weighted window multiplies each value by the weight of its position in
the window: weights[0] for the window's first row, weights[window_size -
1] for its last. A window cut short at a partition's edge keeps the weights
of the positions it still covers: the last ones for a trailing window at
the start, and for a centered window the ones lined up with the rows on
each side of the center. Each row's result is computed from its own
window's values and weights, O(window_size) a row, since every weight
changes as the window moves.

Nulls are skipped with their weights. Fewer than `needed` valid values, or
none, give null, and so do a mean, a variance and a quantile whose valid
values carry no weight. Results:

- sum: the weighted sum; min and max: the least and greatest weighted
  value (NaN when any is NaN);
- mean: the weighted sum over the sum of the weights;
- var and std: the weighted mean of squared deviations from the weighted
  mean (the weights act as proportions, so ddof does not apply);
- quantile: the window's values in order, each spanning its weight; the
  quantile falls at q times the total weight less the first value's
  weight, plus that weight, and the interpolation method picks or joins
  the values on either side. The window's values are kept in order as it
  moves (a row entering is inserted where it belongs and a row leaving is
  removed), so no window is sorted from scratch.
"""
from std.math import isnan, sqrt, ceil
from std.memory import Pointer
from dataframe.storage.column import Column
from .dtype import DataType
from .series import Series


comptime W_SUM = 0
comptime W_MEAN = 1
comptime W_MIN = 2
comptime W_MAX = 3
comptime W_VAR = 4
comptime W_STD = 5
comptime W_QUANTILE = 6


@always_inline
def window_bounds(
    j: Int, lo: Int, hi: Int, window: Int, center: Bool
) -> Tuple[Int, Int]:
    """Row j's window [begin, end) in its partition [lo, hi): the window
    rows ending at j, or centered on j with the extra row of an even
    window before it."""
    if center:
        var right = (window + 1) // 2
        return (max(lo, j - (window - right)), min(hi, j + right))
    return (max(lo, j - window + 1), j + 1)


@always_inline
def weights_start(
    center: Bool, window: Int, j: Int, begin: Int, lo: Int, length: Int
) -> Int:
    """The weight of row `begin`, the window's first row, when the window
    is cut short to `length` rows."""
    if center:
        return max(window // 2 - (j - begin), 0)
    if begin == lo:
        return window - length
    return 0


@always_inline
def _before(a: Float64, b: Float64) -> Bool:
    """Total order with NaN above every number."""
    if isnan(a):
        return False
    if isnan(b):
        return True
    return a < b


def weighted_frames(
    input: Series,
    starts: List[Int],
    kind: Int,
    window: Int,
    needed: Int,
    center: Bool,
    weights: List[Float64],
    quantile: Float64,
    method: String,
) raises -> Series:
    """The weighted rolling `kind` of `input` over each row's window in the
    partitions at `starts`, as Float64."""
    var floats = input.cast(DataType.FLOAT64).rechunk()
    ref column = floats._data[Column[Float64]]
    var values = column._ptr()
    var nulls = column.null_count() > 0
    var n = len(column)
    var out = List[Float64](length=n, fill=0)
    var ok = List[Bool](length=n, fill=False)
    # The window's valid rows by value (then row), for a quantile.
    var ordered = List[Int](capacity=window)
    var pairs = List[Tuple[Float64, Float64]](capacity=window)
    for p in range(len(starts) - 1):
        var lo = starts[p]
        var hi = starts[p + 1]
        ordered.clear()
        var first = lo
        var last = lo
        for j in range(lo, hi):
            var bounds = window_bounds(j, lo, hi, window, center)
            var b = bounds[0]
            var e = bounds[1]
            var w0 = weights_start(center, window, j, b, lo, e - b)
            if kind == W_QUANTILE:
                # Slide the ordered rows to [b, e).
                while first < b:
                    if not nulls or column._valid(first):
                        var at = _find(ordered, values, first)
                        _ = ordered.pop(at)
                    first += 1
                while last < e:
                    if not nulls or column._valid(last):
                        ordered.insert(_find(ordered, values, last), last)
                    last += 1
            var count = 0
            var total = Float64(0)
            var weight = Float64(0)
            var low = Float64(0)
            var high = Float64(0)
            var nan = False
            var wp = weights.unsafe_ptr()
            if not nulls and kind != W_MIN and kind != W_MAX:
                # No nulls: the weighted sum is a dot product of the
                # window's values with its weights, taken a vector at a
                # time.
                comptime width = 8
                var length = e - b
                var acc = SIMD[DType.float64, width](0)
                var sums = SIMD[DType.float64, width](0)
                var k = 0
                while k + width <= length:
                    var xs = values.unsafe_load[width=width](b + k)
                    var ws = wp.unsafe_load[width=width](w0 + k)
                    acc += xs * ws
                    sums += ws
                    k += width
                total = acc.reduce_add()
                weight = sums.reduce_add()
                while k < length:
                    total += (
                        values[unsafe_offset=b + k] * wp[unsafe_offset=w0 + k]
                    )
                    weight += wp[unsafe_offset=w0 + k]
                    k += 1
                count = length
            for k in range(b, e):
                if count > 0 and not nulls and kind != W_MIN and kind != W_MAX:
                    break
                if nulls and not column._valid(k):
                    continue
                count += 1
                var w = weights[w0 + k - b]
                var x = values[unsafe_offset=k]
                if kind == W_MIN or kind == W_MAX:
                    var y = x * w
                    if isnan(y):
                        nan = True
                    elif count == 1 or (y < low if kind == W_MIN else y > high):
                        low = y
                        high = y
                else:
                    total += x * w
                    weight += w
            if count < needed or count == 0:
                continue
            if kind == W_SUM:
                out[j] = total
            elif kind == W_MIN or kind == W_MAX:
                out[j] = Float64(FloatLiteral.nan) if nan else (
                    low if kind == W_MIN else high
                )
            elif weight == 0:
                continue
            elif kind == W_MEAN:
                out[j] = total / weight
            elif kind == W_VAR or kind == W_STD:
                var mean = total / weight
                var squares = Float64(0)
                if not nulls:
                    comptime width = 8
                    var length = e - b
                    var acc = SIMD[DType.float64, width](0)
                    var k = 0
                    while k + width <= length:
                        var d = values.unsafe_load[width=width](b + k) - mean
                        acc += wp.unsafe_load[width=width](w0 + k) * d * d
                        k += width
                    squares = acc.reduce_add()
                    while k < length:
                        var d = values[unsafe_offset=b + k] - mean
                        squares += wp[unsafe_offset=w0 + k] * d * d
                        k += 1
                else:
                    for k in range(b, e):
                        if not column._valid(k):
                            continue
                        var d = values[unsafe_offset=k] - mean
                        squares += weights[w0 + k - b] * d * d
                # Rounding can leave a tiny negative; NaN (from an infinite
                # value) stays NaN.
                var variance = squares / weight
                if variance < 0:
                    variance = 0
                out[j] = sqrt(variance) if kind == W_STD else variance
            else:
                pairs.clear()
                var span = Float64(0)
                for row in ordered:
                    var w = weights[w0 + row - b]
                    if w != 0:
                        pairs.append((values[unsafe_offset=row], w))
                        span += w
                if len(pairs) == 0:
                    continue
                out[j] = _weighted_quantile(pairs, quantile, span, method)
            ok[j] = True
    _ = floats^
    return Series("", Column[Float64](out^, ok))


def _find(
    ordered: List[Int], values: Pointer[Float64, MutAnyOrigin], row: Int
) -> Int:
    """Where `row` is, or belongs, among `ordered` (by value, then row)."""
    var x = values[unsafe_offset=row]
    var a = 0
    var z = len(ordered)
    while a < z:
        var mid = (a + z) // 2
        var other = ordered[mid]
        var y = values[unsafe_offset=other]
        if _before(y, x) or (not _before(x, y) and other < row):
            a = mid + 1
        else:
            z = mid
    return a


def _weighted_quantile(
    pairs: List[Tuple[Float64, Float64]],
    q: Float64,
    total: Float64,
    method: String,
) -> Float64:
    """The quantile of values in order, each spanning its weight: walk the
    cumulative weight up to h = q * (total - w0) + w0 (w0 the first
    value's weight), the weighted form of position q * (n - 1) counted
    from 1."""
    var s = Float64(0)
    var s_old = Float64(0)
    # Starting from the first value, a window of one value gives that
    # value whatever the method.
    var vk = pairs[0][0]
    var v_old = pairs[0][0]
    var h = q * (total - pairs[0][1]) + pairs[0][1]
    for pair in pairs:
        if s > h:
            break
        s_old = s
        v_old = vk
        vk = pair[0]
        s += pair[1]
    if h == s_old or method == "lower":
        return v_old
    if method == "higher":
        return vk
    if method == "nearest":
        return v_old if s - h > h - s_old else vk
    if method == "equiprobable":
        return vk if s > ceil(total * q) - 1 else v_old
    if method == "midpoint":
        return (vk + v_old) * 0.5
    return v_old + (h - s_old) / (s - s_old) * (vk - v_old)
