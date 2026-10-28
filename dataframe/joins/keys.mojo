"""Joint key encoding for the dictionary join path."""
from dataframe.frame import (
    DataFrame,
)
from dataframe.kernels.hashing import (
    encode_rows,
)
from dataframe.joins.range import (
    _bounded_int64_join_ids,
)
from dataframe.parallel import (
    worker_count,
)
from dataframe.kernels.partition import (
    encode_partitioned,
    low_cardinality,
)
from dataframe.series import (
    Series,
)


def _joint_key_ids(
    left: DataFrame,
    right: DataFrame,
    left_keys: List[Int],
    right_keys: List[Int],
) raises -> Tuple[List[Int], List[Int], Int]:
    """Encode both sides' keys in one id space; null keys get id -1."""
    if len(left_keys) == 1:
        var direct = _bounded_int64_join_ids(
            left, right, left_keys[0], right_keys[0]
        )
        if direct[0]:
            return (direct[1].copy(), direct[2].copy(), direct[3])
    var stacked = List[Series](capacity=len(left_keys))
    for k in range(len(left_keys)):
        stacked.append(
            left._columns[left_keys[k]].append(right._columns[right_keys[k]])
        )
    # A join's row order comes from iterating rows, not from the id
    # numbering, so ids may be assigned in any consistent order. On a
    # high-cardinality key a single dictionary over both sides is the
    # dominant cost of the whole join, so encode per hash bucket instead.
    var total = left.height() + right.height()
    var workers = worker_count(total)
    var keys = encode_partitioned(
        stacked, workers, nulls_equal=False
    ) if workers > 1 and not low_cardinality(stacked) else encode_rows(
        stacked, nulls_equal=False
    )
    var n = left.height()
    var left_ids = List[Int](capacity=n)
    var right_ids = List[Int](capacity=right.height())
    for i in range(len(keys.ids)):
        if i < n:
            left_ids.append(keys.ids[i])
        else:
            right_ids.append(keys.ids[i])
    return (left_ids^, right_ids^, keys.count())
