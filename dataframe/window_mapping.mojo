"""Assemble already evaluated partition values for window mappings."""

from dataframe.series import Series
from dataframe.storage.nested import ListColumn


def assemble_window_mapping(
    parts: List[Series],
    groups: List[Int],
    mapping_strategy: String,
    scalar: Bool,
    prototype: Series,
) raises -> Series:
    """Map partition results back to rows or concatenate them in group order.

    `parts` follows stable first-appearance partition order and `groups` maps
    each original row to its partition. Scalar/aggregate partitions contain
    exactly one value. `prototype` supplies the value schema and result name
    when there are no partitions, including recursive Arrow field metadata.
    """
    if mapping_strategy != "join" and mapping_strategy != "explode":
        raise Error("Window mapping assembly requires join or explode")
    for group in groups:
        if group < 0 or group >= len(parts):
            raise Error("Window partition index is out of bounds")
    if scalar:
        for part in parts:
            if len(part) != 1:
                raise Error("Scalar window partition must produce one value")

    var empty = List[Int]()
    var values = (
        prototype.take(empty) if len(parts)
        == 0 else Series._from_chunks(parts).rechunk()
    )
    if mapping_strategy == "explode":
        return values.renamed(prototype.name())
    if scalar:
        return values.take(groups).renamed(prototype.name())

    var offsets = List[Int64](capacity=len(parts) + 1)
    offsets.append(0)
    var total = Int64(0)
    for part in parts:
        if Int64(len(part)) > Int64.MAX - total:
            raise Error("Window list child length overflows")
        total += Int64(len(part))
        offsets.append(total)
    var lists = Series(
        prototype.name(), ListColumn(offsets^, values.renamed("item"))
    )
    return lists.take(groups)
