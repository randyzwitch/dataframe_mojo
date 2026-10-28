"""Compatibility imports; implementation lives in dataframe.text.parse."""
from dataframe.text.parse import (
    _short_decimal,
    parse_int64,
    parse_integer,
    edge_ascii_whitespace,
    _is_decimal,
    _is_special_float,
    _EXACT_LIMIT,
    parse_float64,
    _scan_float64,
    _checked_float64,
    parse_bool,
    _float_scan_fallback,
    _parse_float64_borrowed,
)
