"""Compatibility imports; implementation lives in dataframe.kernels.comparison_sort."""
from dataframe.kernels.comparison_sort import (
    KeyOrder,
    _merge_runs,
    _co_rank,
    _merge_slice,
    _sort_range,
    _SortRangeJob,
    _MergeJob,
    _MIN_ROWS_PER_RUN,
    comparison_arg_sort,
    _PrefixSortJob,
    _prefix_buckets,
)
