"""Exponentially weighted means, variances, standard deviations and sums.

Each row's result weighs the values up to it by (1 - alpha) for every step
back, so one pass over a partition's rows carries the whole state forward:
a weighted mean and its total weight, and for a variance the weighted
mean squared deviation with the sums of the weights and of their squares
(for the unbiased correction). With `adjust`, the weights are those of an
infinite series cut at the first value (1, 1 - alpha, (1 - alpha)^2, ...,
normalized by their sum); without it, each new value enters with weight
alpha and the earlier ones keep 1 - alpha of theirs. A null row stays null
and, unless nulls are ignored, still ages the earlier values by one step.
A row is null until `min_samples` values have been seen.

With a time index (`ewm_mean_by`, `ewm_sum_by`), the earlier result decays
by 0.5^(elapsed / half_life) between consecutive rows in time order.
"""
from std.math import sqrt
from dataframe.storage.column import Column
from .dtype import DataType
from .series import Series


@no_inline
def _product[D: DType](a: Scalar[D], b: Scalar[D]) -> Scalar[D]:
    """`a * b` rounded on its own. Kept out of line so the compiler cannot
    fuse it with a following addition into one multiply-add, which rounds
    once: in Float32 a recurrence that subtracts nearly equal terms would
    then drift from the same steps rounded one by one."""
    return a * b


comptime EWM_KIND_MEAN = 0
comptime EWM_KIND_VAR = 1
comptime EWM_KIND_STD = 2
comptime EWM_KIND_SUM = 3


def ewm_kind(name: String) raises -> Int:
    if name == "mean":
        return EWM_KIND_MEAN
    if name == "var":
        return EWM_KIND_VAR
    if name == "std":
        return EWM_KIND_STD
    if name == "sum":
        return EWM_KIND_SUM
    raise Error("unknown exponentially weighted function: " + name)


def ewm_frames(
    input: Series,
    starts: List[Int],
    kind: Int,
    alpha: Float64,
    adjust: Bool,
    bias: Bool,
    min_samples: Int,
    ignore_nulls: Bool,
) raises -> Series:
    """The exponentially weighted `kind` of `input` within each partition
    (row ranges at `starts`). Float32 is computed and returned as Float32,
    other numbers as Float64."""
    if input.dtype() == DataType.FLOAT32:
        var narrow = input.rechunk()
        return _ewm[DType.float32](
            narrow._data[Column[Float32]],
            starts,
            kind,
            alpha,
            adjust,
            bias,
            min_samples,
            ignore_nulls,
        )
    var floats = input.cast(DataType.FLOAT64).rechunk()
    return _ewm[DType.float64](
        floats._data[Column[Float64]],
        starts,
        kind,
        alpha,
        adjust,
        bias,
        min_samples,
        ignore_nulls,
    )


def _ewm[
    D: DType
](
    column: Column[Scalar[D]],
    starts: List[Int],
    kind: Int,
    alpha_: Float64,
    adjust: Bool,
    bias: Bool,
    min_samples: Int,
    ignore_nulls: Bool,
) raises -> Series:
    comptime T = Scalar[D]
    var n = len(column)
    var out = List[T](length=n, fill=T(0))
    var ok = List[Bool](length=n, fill=False)
    var alpha = T(alpha_)
    var keep = T(1) - alpha
    # The weight a new value enters with.
    var fresh = T(1) if adjust else alpha
    var needed = max(min_samples, 1)
    var nulls = column.null_count() > 0
    for p in range(len(starts) - 1):
        var seen = 0
        var mean = T(0)
        var weight = T(0)
        var cov = T(0)
        var weights = T(0)
        var squares = T(0)
        var total = T(0)
        for j in range(starts[p], starts[p + 1]):
            var valid = not nulls or column._valid(j)
            if kind == EWM_KIND_SUM:
                if valid:
                    var x = column._get(j)
                    total = x if seen == 0 else _product[D](total, keep) + x
                    seen += 1
                elif seen > 0 and not ignore_nulls:
                    total *= keep
                if valid and seen >= needed:
                    out[j] = total
                    ok[j] = True
                continue
            if seen == 0 and valid:
                var x = column._get(j)
                seen = 1
                mean = x
                weight = T(1)
                cov = T(0)
                weights = T(1)
                squares = T(1)
            else:
                if valid or not ignore_nulls:
                    weights *= keep
                    squares *= keep * keep
                    weight *= keep
                if valid:
                    var x = column._get(j)
                    seen += 1
                    var grown = weight + fresh
                    var share = fresh / grown
                    var next_mean = mean + _product[D](x - mean, share)
                    if kind != EWM_KIND_MEAN:
                        cov = (
                            weight
                            * (cov + (mean - next_mean) * (mean - next_mean))
                            + fresh * (x - next_mean) * (x - next_mean)
                        ) / grown
                        weights += fresh
                        squares += fresh * fresh
                        if not adjust:
                            weights /= grown
                            squares /= grown * grown
                    mean = next_mean
                    weight = grown if adjust else T(1)
            if not valid or seen < needed:
                continue
            if kind == EWM_KIND_MEAN:
                out[j] = mean
                ok[j] = True
                continue
            var variance = cov
            if not bias:
                var numerator = weights * weights
                var denominator = numerator - squares
                if not denominator > 0:
                    continue
                variance = numerator / denominator * cov
            out[j] = sqrt(variance) if kind == EWM_KIND_STD else variance
            ok[j] = True
    return Series("", Column[T](out^, ok))


def ewm_by_frames(
    input: Series,
    times: List[Int64],
    times_valid: List[Bool],
    starts: List[Int],
    half_life: Float64,
    mean: Bool,
) raises -> Series:
    """`ewm_mean_by` (mean) or `ewm_sum_by` of `input`, whose rows lie in
    time order within each partition (row ranges at `starts`); `times`
    and `half_life` share a unit. A row whose value or time is null stays
    null and leaves the state as it is. Float32 is computed and returned
    as Float32, other numbers as Float64."""
    if input.dtype() == DataType.FLOAT32:
        var narrow = input.rechunk()
        return _ewm_by[DType.float32](
            narrow._data[Column[Float32]],
            times,
            times_valid,
            starts,
            half_life,
            mean,
        )
    var floats = input.cast(DataType.FLOAT64).rechunk()
    return _ewm_by[DType.float64](
        floats._data[Column[Float64]],
        times,
        times_valid,
        starts,
        half_life,
        mean,
    )


def _ewm_by[
    D: DType
](
    column: Column[Scalar[D]],
    times: List[Int64],
    times_valid: List[Bool],
    starts: List[Int],
    half_life: Float64,
    mean: Bool,
) raises -> Series:
    comptime T = Scalar[D]
    var n = len(column)
    var out = List[T](length=n, fill=T(0))
    var ok = List[Bool](length=n, fill=False)
    var span = T(half_life)
    for p in range(len(starts) - 1):
        var started = False
        var previous = T(0)
        var last = Int64(0)
        for j in range(starts[p], starts[p + 1]):
            if not column._valid(j) or not times_valid[j]:
                continue
            var x = column._get(j)
            if not started:
                previous = x
                started = True
            elif not (mean and x == previous):
                # The power is taken in Float64 and rounded once, so a
                # Float32 decay is the correctly rounded one.
                var ratio = T(times[j] - last) / span
                var decay = (Float64(0.5) ** ratio.cast[DType.float64]()).cast[
                    D
                ]()
                var gain = T(1) - decay if mean else T(1)
                previous = _product[D](decay, previous) + _product[D](gain, x)
            last = times[j]
            out[j] = previous
            ok[j] = True
    return Series("", Column[T](out^, ok))
