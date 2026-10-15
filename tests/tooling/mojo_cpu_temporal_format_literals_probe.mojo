"""Focused CPU regression driver; no accelerator or producer imports."""
from tests.public_contracts.test_temporal import (
    test_temporal_strftime_escaped_percent_formats_one_literal_percent,
)
from tests.public_contracts.test_temporal_text_utf8 import (
    test_temporal_format_utf8_and_percent_literal_roundtrip,
    test_temporal_strftime_utf8_literals_preserve_nulls_and_sliced_rows,
    test_temporal_format_utc_directives_and_unknown_directives_remain_literal,
)


def main() raises:
    test_temporal_strftime_escaped_percent_formats_one_literal_percent()
    test_temporal_format_utf8_and_percent_literal_roundtrip()
    test_temporal_strftime_utf8_literals_preserve_nulls_and_sliced_rows()
    test_temporal_format_utc_directives_and_unknown_directives_remain_literal()
    print(
        "PASS CPU temporal percent escape, UTF8 literals, null/slice and UTC directives"
    )
