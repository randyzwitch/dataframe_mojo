"""Schema-derived value dtype masks for grouped kernel specialization."""
from dataframe import DataType

comptime GROUP_FLOAT32 = 1
comptime GROUP_INT32 = 2
comptime GROUP_INT64 = 4


def group_value_types(dtypes: List[DataType]) raises -> Int:
    var mask = 0
    for dtype in dtypes:
        if dtype == DataType.FLOAT32:
            mask |= GROUP_FLOAT32
        elif dtype == DataType.INT32:
            mask |= GROUP_INT32
        elif dtype == DataType.INT64:
            mask |= GROUP_INT64
        else:
            raise Error("Metal grouped value dtype must be native numeric")
    if mask == 0:
        raise Error("Metal grouped value dtypes cannot be empty")
    return mask
