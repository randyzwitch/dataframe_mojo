"""Small scalar order models; no dataframe code supplies expected answers.

Each rolling result is recomputed from its member rows. Ranking counts pairwise
relations rather than sharing the engine's sort. Values here are bounded exact
integers or IEEE fixtures; wide-integer overflow has separate literal cases.
"""
from std.math import isnan, sqrt


struct WindowCells(Copyable, Movable):
    var values: List[Float64]
    var valid: List[Bool]

    def __init__(out self, values: List[Float64], valid: List[Bool]):
        self.values = values.copy()
        self.valid = valid.copy()


def order_less(a: Float64, b: Float64) -> Bool:
    # Ordered expressions put NaN above numbers and treat both zeros as ties.
    return not isnan(a) and (isnan(b) or a < b)


def order_equal(a: Float64, b: Float64) -> Bool:
    return a == b or (isnan(a) and isnan(b))


def cumulative(cells: WindowCells, kind: String, reverse: Bool) -> WindowCells:
    var n = len(cells.values)
    var values = List[Float64](length=n, fill=0)
    var valid = cells.valid.copy()
    var total = Float64(0)
    var count = 0
    var best = Float64(0)
    for step in range(n):
        var row = n - 1 - step if reverse else step
        if cells.valid[row]:
            var value = cells.values[row]
            total += value
            if count == 0 or (
                order_less(value, best) if kind
                == "min" else order_less(best, value)
            ):
                best = value
            count += 1
        if kind == "count":
            values[row] = Float64(count)
            valid[row] = True
        elif cells.valid[row]:
            values[row] = total if kind == "sum" else best
    return WindowCells(values^, valid^)


def shifted(cells: WindowCells, distance: Int) -> WindowCells:
    var values = List[Float64](length=len(cells.values), fill=0)
    var valid = List[Bool](length=len(cells.values), fill=False)
    for row in range(len(values)):
        var source = row - distance
        if source >= 0 and source < len(values):
            values[row] = cells.values[source]
            valid[row] = cells.valid[source]
    return WindowCells(values^, valid^)


def changed(cells: WindowCells, distance: Int, relative: Bool) -> WindowCells:
    var prior = shifted(cells, distance)
    var values = List[Float64](length=len(cells.values), fill=0)
    var valid = List[Bool](length=len(cells.values), fill=False)
    for row in range(len(values)):
        valid[row] = cells.valid[row] and prior.valid[row]
        if valid[row]:
            values[row] = cells.values[row] - prior.values[row]
            if relative:
                values[row] /= prior.values[row]
    return WindowCells(values^, valid^)


def filled(cells: WindowCells, reverse: Bool, limit: Int) -> WindowCells:
    var values = cells.values.copy()
    var valid = cells.valid.copy()
    # Search each row's nearest original valid neighbor. Filled rows are never
    # reused as evidence, so a finite limit cannot propagate through a gap.
    for row in range(len(values)):
        if valid[row]:
            continue
        var distance = 1
        while limit < 0 or distance <= limit:
            var source = row + distance if reverse else row - distance
            if source < 0 or source >= len(values):
                break
            if cells.valid[source]:
                values[row] = cells.values[source]
                valid[row] = True
                break
            distance += 1
    return WindowCells(values^, valid^)


def rolling(
    cells: WindowCells, width: Int, needed: Int, kind: String, ddof: Int = 1
) -> WindowCells:
    var values = List[Float64](length=len(cells.values), fill=0)
    var valid = List[Bool](length=len(cells.values), fill=False)
    var minimum = width if needed < 0 else needed
    for row in range(len(values)):
        var members = List[Float64]()
        for source in range(max(0, row - width + 1), row + 1):
            if cells.valid[source]:
                members.append(cells.values[source])
        var count = len(members)
        if (
            count < minimum
            or count == 0
            or ((kind == "var" or kind == "std") and count <= ddof)
        ):
            continue
        valid[row] = True
        var total = Float64(0)
        var best = members[0]
        for value in members:
            total += value
            if order_less(value, best) if kind == "min" else order_less(
                best, value
            ):
                best = value
        if kind == "sum":
            values[row] = total
        elif kind == "mean":
            values[row] = total / Float64(count)
        elif kind == "min" or kind == "max":
            values[row] = best
        else:
            var mean = total / Float64(count)
            var squares = Float64(0)
            for value in members:
                var difference = value - mean
                squares += difference * difference
            values[row] = squares / Float64(count - ddof)
            if kind == "std":
                values[row] = sqrt(values[row])
    return WindowCells(values^, valid^)


def ranked(cells: WindowCells, method: String, descending: Bool) -> WindowCells:
    var result = List[Float64](length=len(cells.values), fill=0)
    for row in range(len(result)):
        if not cells.valid[row]:
            continue
        var value = cells.values[row]
        var lower = 0
        var equal = 0
        var earlier = 0
        var distinct_lower = List[Float64]()
        for other in range(len(result)):
            if not cells.valid[other]:
                continue
            var candidate = cells.values[other]
            if order_equal(value, candidate):
                equal += 1
                earlier += Int(other < row)
            elif order_less(value, candidate) if descending else order_less(
                candidate, value
            ):
                lower += 1
                var seen = False
                for prior in distinct_lower:
                    seen = seen or order_equal(candidate, prior)
                if not seen:
                    distinct_lower.append(candidate)
        if method == "average":
            result[row] = Float64(lower) + Float64(equal + 1) / 2
        elif method == "min":
            result[row] = Float64(lower + 1)
        elif method == "max":
            result[row] = Float64(lower + equal)
        elif method == "dense":
            result[row] = Float64(len(distinct_lower) + 1)
        else:
            result[row] = Float64(lower + earlier + 1)
    return WindowCells(result^, cells.valid.copy())


def interpolated(
    cells: WindowCells, nearest: Bool, axis: List[Float64]
) -> WindowCells:
    var result = cells.values.copy()
    var valid = cells.valid.copy()
    for row in range(len(result)):
        if valid[row]:
            continue
        var before = row - 1
        var after = row + 1
        while before >= 0 and not cells.valid[before]:
            before -= 1
        while after < len(result) and not cells.valid[after]:
            after += 1
        if before < 0 or after >= len(result):
            continue
        if nearest:
            result[row] = cells.values[
                before if row - before < after - row else after
            ]
        else:
            var span = axis[after] - axis[before]
            if span == 0:
                continue
            var fraction = (axis[row] - axis[before]) / span
            result[row] = (
                cells.values[before] * (1 - fraction)
                + cells.values[after] * fraction
            )
        valid[row] = True
    return WindowCells(result^, valid^)
