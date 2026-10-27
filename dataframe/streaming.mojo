"""Compatibility exports for shared streaming reduction state.

The implementation lives in aggregation/state.mojo and also serves eager grouping.
"""
from .aggregation.state import _StreamReduction, _StreamMergeJob, _finish_parts
