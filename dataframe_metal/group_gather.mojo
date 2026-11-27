"""Recover first-seen group keys from the native input chunks."""
from dataframe import Column, DataFrame, DataType, Series
from dataframe.kernels.scalar_gather import gather_scalars
from .group_keys import GroupKeys


def _gather_keys[D: DType](source: Series, indices: List[Int]) raises -> Series:
    """Gather first-seen rows into owned storage, walking each key's own chunks."""
    var parts = List[Series]()
    var first = 0
    var base = 0
    for part in source.chunks():
        var end = base + len(part)
        var last = first
        while last < len(indices) and indices[last] < end:
            last += 1
        if first < last:
            ref column = part._data[Column[Scalar[D]]]
            if column.is_constant():
                parts.append(
                    Series(source.name(), column._resized(last - first))
                )
            else:
                parts.append(
                    Series(
                        source.name(),
                        gather_scalars[D](
                            column, indices, False, first, last, base
                        ),
                    )
                )
        first = last
        base = end
        if first == len(indices):
            break
    if first != len(indices):
        raise Error("Grouped first row exceeds key column length")
    if not parts:
        return Series(source.name(), Column[Scalar[D]](List[Scalar[D]]()))
    if len(parts) == 1:
        return parts.pop()
    return Series._from_chunks(parts^).rechunk()


def gather_group_keys(
    frame: DataFrame, keys: GroupKeys, first: List[UInt32], slots: List[Int]
) raises -> List[Series]:
    var indices = List[Int](capacity=len(slots))
    for slot in slots:
        indices.append(Int(first[slot]))
    var answer = List[Series](capacity=len(keys.names))
    for i in range(len(keys.names)):
        var source = frame.column(keys.names[i])
        if keys.dtypes[i] == DataType.INT32:
            answer.append(_gather_keys[DType.int32](source, indices))
        else:
            answer.append(_gather_keys[DType.int64](source, indices))
    return answer^
