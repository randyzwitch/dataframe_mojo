"""Compatibility imports; implementation lives in dataframe.kernels.join_fuse."""
from dataframe.kernels.join_fuse import (
    MatchJob,
    FusedJoin,
    _PIECE_HUGE_PAGES,
    probe_block_rows,
    _row_bytes,
    probe_blocks,
    _FusedJob,
    _contiguous,
    _assemble,
    fused_gather,
)
