"""Compatibility imports; implementation lives in dataframe.storage.string_view."""
from dataframe.storage.string_view import (
    STRING_VIEW_INLINE_BYTES,
    STRING_VIEW_DEFAULT_BLOCK_SIZE,
    STRING_VIEW_MAX_EXP_BLOCK_SIZE,
    _STRING_VIEW_BUFFER_LIMIT,
    StringView,
    _load_prefix_little_endian,
    _inline_view,
    _external_view,
    StringViewStorage,
    StringViewBuilder,
)
