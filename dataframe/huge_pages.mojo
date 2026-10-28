"""Compatibility imports; implementation lives in dataframe.storage.allocation."""
from dataframe.storage.allocation import (
    _MADV_HUGEPAGE,
    HUGE_PAGE_BYTES,
    _HUGE_PAGE,
    _PAGE,
    advise_huge_pages,
    huge_list,
    huge_uninit,
    huge_capacity,
)
