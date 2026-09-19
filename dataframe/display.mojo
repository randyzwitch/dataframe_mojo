"""Bounded text rendering. Only displayed cells are ever formatted."""
from .dtype import DataType
from .temporal import format as format_temporal
from .column import Column
from .series import Series

comptime ELLIPSIS = "…"


def short_dtype(dtype: DataType) -> String:
    return dtype.short_name()


def _codepoints(text: String) -> Int:
    return len(text.codepoints())


def _needs_quotes(text: String) -> Bool:
    if text.byte_length() == 0 or text == "null":
        return True
    var bytes = text.as_bytes()
    var first = bytes[0]
    var last = bytes[len(bytes) - 1]
    if first == 32 or first == 9 or last == 32 or last == 9:
        return True
    for b in bytes:
        if b < 32:
            return True
    return False


def _escape(text: String) -> String:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _truncate(text: String, max_length: Int) -> String:
    if max_length <= 0 or _codepoints(text) <= max_length:
        return text
    var result = String()
    var count = 0
    for codepoint in text.codepoints():
        if count >= max_length - 1:
            break
        result += String(codepoint)
        count += 1
    return result + ELLIPSIS


def format_string(text: String, max_length: Int) -> String:
    """Quote strings that would otherwise be ambiguous with null or blanks."""
    if _needs_quotes(text):
        return _truncate('"' + _escape(text) + '"', max_length)
    return _truncate(text, max_length)


def format_cell(series: Series, row: Int, max_string_length: Int) -> String:
    """Format one cell; assumes the row index is in bounds."""
    if series._data.isa[Column[Int64]]():
        ref column = series._data[Column[Int64]]
        if not column._valid(row):
            return "null"
        if series.dtype().is_temporal():
            return format_temporal(column._values[row], series.dtype())
        return String(column._values[row])
    if series._data.isa[Column[Float64]]():
        ref column = series._data[Column[Float64]]
        if not column._valid(row):
            return "null"
        return String(column._values[row])
    if series._data.isa[Column[Bool]]():
        ref column = series._data[Column[Bool]]
        if not column._valid(row):
            return "null"
        return "true" if column._values[row] else "false"
    ref column = series._data[Column[String]]
    if not column._valid(row):
        return "null"
    return format_string(column._values[row], max_string_length)


def _visible(count: Int, limit: Int) -> List[Int]:
    """Indices to show, with -1 marking an elided run."""
    var indices = List[Int]()
    if limit < 0 or count <= limit:
        for i in range(count):
            indices.append(i)
        return indices^
    var front = (limit + 1) // 2
    var back = limit // 2
    for i in range(front):
        indices.append(i)
    indices.append(-1)
    for i in range(count - back, count):
        indices.append(i)
    return indices^


def _pad(text: String, width: Int) -> String:
    var padding = width - _codepoints(text)
    if padding <= 0:
        return text
    return text + String(" ") * padding


def _rule(
    widths: List[Int], left: String, middle: String, right: String, fill: String
) -> String:
    var line = left
    for i in range(len(widths)):
        if i > 0:
            line += middle
        line += fill * (widths[i] + 2)
    return line + right


def render_frame(
    columns: List[Series],
    height: Int,
    max_rows: Int,
    max_columns: Int,
    max_string_length: Int,
) -> String:
    """Render a box table with shape, names, dtypes, and elided rows/columns."""
    var out = String("shape: (", height, ", ", len(columns), ")\n")
    var column_ids = _visible(len(columns), max_columns)
    var row_ids = _visible(height, max_rows)
    if len(column_ids) == 0:
        return out + "┌┐\n└┘"
    var cells = List[List[String]]()
    var widths = List[Int]()
    for c in column_ids:
        var column_cells = List[String]()
        if c < 0:
            column_cells.append(ELLIPSIS)
            column_cells.append(ELLIPSIS)
            for _ in row_ids:
                column_cells.append(ELLIPSIS)
        else:
            column_cells.append(
                format_string(
                    columns[c].name(), max_string_length
                ) if _needs_quotes(columns[c].name()) else _truncate(
                    columns[c].name(), max_string_length
                )
            )
            column_cells.append(short_dtype(columns[c].dtype()))
            for r in row_ids:
                if r < 0:
                    column_cells.append(ELLIPSIS)
                else:
                    column_cells.append(
                        format_cell(columns[c], r, max_string_length)
                    )
        var width = 3
        for cell in column_cells:
            width = max(width, _codepoints(cell))
        widths.append(width)
        cells.append(column_cells^)
    out += _rule(widths, "┌", "┬", "┐", "─") + "\n"
    for line in range(2):
        out += "│"
        for c in range(len(widths)):
            if c > 0:
                out += "┆"
            out += " " + _pad(cells[c][line], widths[c]) + " "
        out += "│\n"
        if line == 0:
            out += "│"
            for c in range(len(widths)):
                if c > 0:
                    out += "┆"
                out += " " + _pad("---", widths[c]) + " "
            out += "│\n"
    out += _rule(widths, "╞", "╪", "╡", "═")
    for r in range(len(row_ids)):
        out += "\n│"
        for c in range(len(widths)):
            if c > 0:
                out += "┆"
            out += " " + _pad(cells[c][r + 2], widths[c]) + " "
        out += "│"
    out += "\n" + _rule(widths, "└", "┴", "┘", "─")
    return out^


def render_series(
    series: Series, max_rows: Int, max_string_length: Int
) -> String:
    var out = String(
        "shape: (",
        len(series),
        ",)\nSeries: ",
        format_string(series.name(), max_string_length) if _needs_quotes(
            series.name()
        ) else series.name(),
        " [",
        short_dtype(series.dtype()),
        "]\n[",
    )
    for r in _visible(len(series), max_rows):
        out += "\n\t"
        out += ELLIPSIS if r < 0 else format_cell(series, r, max_string_length)
    return out + "\n]"


def render_glimpse(
    columns: List[Series], height: Int, max_width: Int, max_string_length: Int
) -> String:
    """One line per column: name, dtype, and as many leading values as fit."""
    var out = String("Rows: ", height, "\nColumns: ", len(columns))
    var name_width = 0
    for column in columns:
        name_width = max(name_width, _codepoints(column.name()))
    for column in columns:
        var line = (
            "$ "
            + _pad(column.name(), name_width)
            + " <"
            + short_dtype(column.dtype())
            + ">"
        )
        var prefix = _codepoints(line)
        for r in range(len(column)):
            var cell = format_cell(column, r, max_string_length)
            var separator = " " if r == 0 else ", "
            if (
                max_width > 0
                and prefix + _codepoints(separator + cell) + 3 > max_width
            ):
                line += ", " + ELLIPSIS if r > 0 else " " + ELLIPSIS
                break
            line += separator + cell
            prefix += _codepoints(separator + cell)
        out += "\n" + line
    return out^
