"""Compatibility imports; implementation lives in dataframe.text.unicode.normalization."""
from dataframe.text.unicode.normalization import (
    _Row,
    _codepoint,
    _append_utf8,
    _quick,
    _decompose,
    _sort_marks,
    _order,
    _compose,
    _recompose,
    normalize_series,
)
