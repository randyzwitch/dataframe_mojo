"""Recover first-seen group keys from the native input chunks."""
from dataframe import DataFrame, DataType, Series
from .group_finish import _gather_keys
from .group_keys import GroupKeys


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
