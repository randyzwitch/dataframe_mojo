"""Schema-only native integer tuple keys for broader grouped reductions."""
from dataframe import DataFrame, DataType

comptime MAX_GROUP_KEYS = 4


@fieldwise_init
struct GroupKeys(Copyable, Movable):
    var names: List[String]
    var dtypes: List[DataType]


def compile_group_keys(
    names: List[String], frame: DataFrame
) raises -> Optional[GroupKeys]:
    if len(names) == 0 or len(names) > MAX_GROUP_KEYS:
        return None
    var dtypes = List[DataType]()
    for i in range(len(names)):
        if names[i] not in frame.columns():
            return None
        for previous in range(i):
            if names[previous] == names[i]:
                return None
        var dtype = frame.column(names[i]).dtype()
        if dtype != DataType.INT32 and dtype != DataType.INT64:
            return None
        dtypes.append(dtype)
    return GroupKeys(names.copy(), dtypes^)
