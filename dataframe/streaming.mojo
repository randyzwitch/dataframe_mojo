"""Merge aggregate states across ordered batches without finalizing partial sums.

The state type lives in frame.mojo, where eager group_by also uses it to
reduce worker row ranges before merging them.
"""
from .frame import _StreamReduction
