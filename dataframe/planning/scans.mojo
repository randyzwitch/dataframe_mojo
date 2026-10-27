"""Scan-node encoding and CSV options, independent of query execution."""
from dataframe.io.csv_types import (
    _CsvScanOptions,
)
from dataframe.planning.model import (
    PlanNode,
    SCAN_CSV,
    _plan_node,
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
        return _plan_node(SCAN_CSV, text=path, offset=0)
    return _plan_node(
        SCAN_CSV,
        text=path,
        offset=0,
        names2=settings + overrides^,
        flags=flags,
        right_keys=options.null_values,
    )


def _scan_options(node: PlanNode) raises -> _CsvScanOptions:
    """The reader options a CSV scan node holds (see `_csv_scan_node`)."""
    var options = _CsvScanOptions.defaults()
    if len(node.names2) < 6:
        return options^
    options.separator = node.names2[0]
    options.quote_char = node.names2[1]
    options.comment_prefix = node.names2[2]
    options.encoding = node.names2[3]
    options.skip_rows = Int(node.names2[4])
    options.infer_schema_length = Int(node.names2[5])
    var k = 6
    while k + 1 < len(node.names2):
        options.schema_overrides[node.names2[k]] = node.names2[k + 1]
        k += 2
    options.has_header = node.flags[0]
    options.ignore_errors = node.flags[1]
    options.truncate_ragged_lines = node.flags[2]
    options.try_parse_dates = node.flags[3]
    options.null_values = node.right_keys.copy()
    return options^
