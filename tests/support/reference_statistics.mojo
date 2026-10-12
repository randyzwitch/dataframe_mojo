"""Scalar statistics over independent rows; deliberately no dataframe imports."""
from tests.support.reference_model import ReferenceTable


def sorted_integers(model: ReferenceTable) -> List[Int64]:
    var values = List[Int64]()
    for row in model.rows:
        if row.iv:
            values.append(row.i)
    for index in range(1, len(values)):
        var value = values[index]
        var at = index
        while at > 0 and value < values[at - 1]:
            values[at] = values[at - 1]
            at -= 1
        values[at] = value
    return values^


def integer_quantile(
    model: ReferenceTable, q: Float64, interpolation: String = "linear"
) raises -> Float64:
    if q < 0 or q > 1:
        raise Error("Reference quantile outside [0, 1]")
    var values = sorted_integers(model)
    if not values:
        raise Error("Reference quantile needs values")
    var position = q * Float64(len(values) - 1)
    var lower = Int(position)
    var upper = min(lower + 1, len(values) - 1)
    var fraction = position - Float64(lower)
    if interpolation == "lower":
        return Float64(values[lower])
    if interpolation == "higher":
        return Float64(values[lower if fraction == 0 else upper])
    if interpolation == "nearest":
        return Float64(values[lower if fraction < 0.5 else upper])
    if interpolation == "midpoint":
        return (
            Float64(values[lower]) if fraction
            == 0 else (Float64(values[lower]) + Float64(values[upper])) / 2
        )
    if interpolation == "linear":
        return (
            Float64(values[lower]) * (1 - fraction)
            + Float64(values[upper]) * fraction
        )
    raise Error("Unknown reference interpolation")


def integer_variance(model: ReferenceTable, ddof: Int = 1) raises -> Float64:
    var count = model.int_count()
    if count <= Int64(ddof):
        raise Error("Reference variance has insufficient values")
    var mean = Float64(model.int_sum()) / Float64(count)
    var squared = Float64(0)
    for row in model.rows:
        if row.iv:
            var delta = Float64(row.i) - mean
            squared += delta * delta
    return squared / Float64(count - Int64(ddof))
