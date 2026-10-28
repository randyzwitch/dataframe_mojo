"""Scan-node encoding and CSV options, independent of query execution."""
from dataframe.planning.node_builders import (
    _csv_source_node,
)
from dataframe.io.csv_types import (
    _CsvScanOptions,
)
from dataframe.planning.model import (
    PlanNode,
    SCAN_CSV,
)
from dataframe.planning.properties import (
    _same_flags,
    _same_names,
)


def _csv_scan_node(path: String, options: _CsvScanOptions) raises -> PlanNode:
    """A CSV scan node holding its reader options, checked here: `flags`
    the switches, `names2` the text and numeric settings followed by the
    schema overrides as name, dtype pairs, `right_keys` the null values.
    A scan with read_csv's defaults keeps none of them, so plans built
    without options are unchanged."""
    _ = options.reader()
    var defaults = _CsvScanOptions.defaults()
    var overrides = List[String]()
    for item in options.schema_overrides.items():
        overrides.append(item.key)
        overrides.append(item.value)
    var settings: List[String] = [
        options.separator,
        options.quote_char,
        options.comment_prefix,
        options.encoding,
        String(options.skip_rows),
        String(options.infer_schema_length),
    ]
    var flags: List[Bool] = [
        options.has_header,
        options.ignore_errors,
        options.truncate_ragged_lines,
        options.try_parse_dates,
    ]
    var default_settings: List[String] = [
        defaults.separator,
        defaults.quote_char,
        defaults.comment_prefix,
        defaults.encoding,
        String(defaults.skip_rows),
        String(defaults.infer_schema_length),
    ]
    var default_flags: List[Bool] = [
        defaults.has_header,
        defaults.ignore_errors,
        defaults.truncate_ragged_lines,
        defaults.try_parse_dates,
    ]
    if (
        _same_names(settings, default_settings)
        and _same_flags(flags, default_flags)
        and len(overrides) == 0
        and len(options.null_values) == 0
    ):
        return _csv_source_node(path=path, slot=0)
    return _csv_source_node(
        path=path,
        slot=0,
        settings=settings + overrides^,
        switches=flags,
        null_values=options.null_values,
    )


def _scan_options(node: PlanNode) raises -> _CsvScanOptions:
    """The reader options a CSV scan node holds (see `_csv_scan_node`)."""
    var options = _CsvScanOptions.defaults()
    if not node.csv_has_options():
        return options^
    options.separator = node.csv_separator()
    options.quote_char = node.csv_quote_char()
    options.comment_prefix = node.csv_comment_prefix()
    options.encoding = node.csv_encoding()
    options.skip_rows = Int(node.csv_skip_rows())
    options.infer_schema_length = Int(node.csv_infer_schema_length())
    for k in range(node.csv_override_count()):
        options.schema_overrides[
            node.csv_override_name(k)
        ] = node.csv_override_dtype(k)
    options.has_header = node.csv_has_header()
    options.ignore_errors = node.csv_ignore_errors()
    options.truncate_ragged_lines = node.csv_truncate_ragged_lines()
    options.try_parse_dates = node.csv_try_parse_dates()
    options.null_values = node.csv_null_values().copy()
    return options^
