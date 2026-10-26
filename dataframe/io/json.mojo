"""JSON tables over the shared parser, schema inference and typed builders."""
from std.collections import Optional
from ..dtype import DataType
from ..frame import DataFrame, concat
from ..series import Series
from ..json_reader import JsonReader, json_bytes
from ..json_decode import JsonDecoder, JsonDecodeOptions, validate_json_dtype
from ..json_number import validate_json_numbers
from .json_schema import JsonSchema
from .json_source import JsonInput, JsonLines, JsonArrayRows
from .json_parallel import json_frame as _frame, read_parallel


def _options(
    schema: Optional[DataType],
    infer_schema_length: Int,
    batch_size: Int,
    n_rows: Int = -1,
) raises:
    if infer_schema_length == 0 or infer_schema_length < -1:
        raise Error("infer_schema_length must be positive or -1 for all rows")
    if batch_size <= 0:
        raise Error("JSON batch_size must be positive")
    if n_rows < -1:
        raise Error("JSON n_rows must be nonnegative or -1")
    if schema:
        if not schema.value().is_struct():
            raise Error("JSON table schema must be a struct")
        validate_json_dtype(schema.value(), io=True)


def _ndjson_schema(input: JsonInput, length: Int) raises -> DataType:
    var lines = JsonLines(input.span())
    var empty = String("")
    var reader = JsonReader(json_bytes(empty))
    var inferred = JsonSchema()
    while length < 0 or inferred.rows < length:
        var record = lines.next()
        if not record:
            break
        try:
            reader.parse(
                input.span()[record.value().start : record.value().end]
            )
            inferred.observe(reader)
        except e:
            raise Error(
                "NDJSON schema at line "
                + String(record.value().line + 1)
                + ": "
                + String(e)
            )
    return inferred.finish()


def _json_schema(input: JsonInput, length: Int) raises -> DataType:
    var rows = JsonArrayRows(input.span())
    var inferred = JsonSchema()
    while length < 0 or inferred.rows < length:
        if not rows.next():
            break
        inferred.observe(rows.reader)
    if inferred.rows == 0:
        return DataType.struct(List[String](), List[DataType]())
    return inferred.finish()


struct NdjsonReader(Movable):
    """A source-owned map and one reusable decoder for bounded row batches."""

    var input: JsonInput
    var lines: JsonLines
    var dtype: DataType
    var decoder: JsonDecoder
    var reader: JsonReader
    var batch_size: Int
    var limit: Int
    var emitted: Bool
    var ended: Bool

    def __init__(
        out self,
        path: String,
        schema: Optional[DataType] = None,
        *,
        infer_schema_length: Int = 100,
        ignore_errors: Bool = False,
        batch_size: Int = 65536,
        n_rows: Int = -1,
        columns: List[String] = List[String](),
    ) raises:
        _options(schema, infer_schema_length, batch_size, n_rows)
        self.input = JsonInput(path)
        self.dtype = schema.value() if schema else _ndjson_schema(
            self.input, infer_schema_length
        )
        self.lines = JsonLines(self.input.span())
        self.reader = JsonReader(self.input.span())
        var capacity = min(batch_size, self.input.size // 2)
        if n_rows >= 0:
            capacity = min(capacity, n_rows)
        self.decoder = JsonDecoder(
            self.dtype,
            capacity,
            JsonDecodeOptions(
                io=True, ignore_errors=ignore_errors, reject_unknown_fields=True
            ),
            columns,
        )
        self.batch_size = batch_size
        self.limit = n_rows
        self.emitted = False
        self.ended = False

    def schema(self) raises -> DataType:
        return self.decoder.output_dtype()

    def next(mut self) raises -> Optional[DataFrame]:
        if self.ended:
            return None
        while self.decoder.rows() < self.batch_size:
            if self.limit >= 0 and self.lines.rows >= self.limit:
                self.ended = True
                break
            var record = self.lines.next()
            if not record:
                self.ended = True
                break
            try:
                self.reader.parse(
                    self.input.span()[record.value().start : record.value().end]
                )
                validate_json_numbers(self.reader)
                self.decoder.append(self.reader)
            except e:
                raise Error(
                    "NDJSON at line "
                    + String(record.value().line + 1)
                    + ": "
                    + String(e)
                )
        if self.decoder.rows() == 0 and self.emitted:
            return None
        self.emitted = True
        return _frame(self.decoder.finish())


def read_ndjson(
    path: String,
    schema: Optional[DataType] = None,
    *,
    infer_schema_length: Int = 100,
    ignore_errors: Bool = False,
    batch_size: Int = 65536,
    n_rows: Int = -1,
    columns: List[String] = List[String](),
) raises -> DataFrame:
    """Read local newline-delimited JSON objects into typed columns.

    Infer from the first infer_schema_length objects (-1 means all).
    Fields outside that schema raise unless ignore_errors=True; conversion
    errors then become null cells. JSON syntax errors always raise.
    """
    if n_rows < 0:
        _options(schema, infer_schema_length, batch_size, n_rows)
        var input = JsonInput(path)
        var dtype = schema.value() if schema else _ndjson_schema(
            input, infer_schema_length
        )
        var result = read_parallel(
            input,
            dtype,
            JsonDecodeOptions(
                io=True, ignore_errors=ignore_errors, reject_unknown_fields=True
            ),
            columns,
            batch_size,
        )
        _ = input^
        return result^
    var reader = NdjsonReader(
        path,
        schema,
        infer_schema_length=infer_schema_length,
        ignore_errors=ignore_errors,
        batch_size=batch_size,
        n_rows=n_rows,
        columns=columns,
    )
    var frames = List[DataFrame]()
    while True:
        var batch = reader.next()
        if not batch:
            break
        frames.append(batch.take())
    return concat(frames) if len(frames) > 1 else frames.pop(0)


def read_json(
    path: String,
    schema: Optional[DataType] = None,
    *,
    infer_schema_length: Int = 100,
    ignore_errors: Bool = False,
    batch_size: Int = 65536,
    columns: List[String] = List[String](),
) raises -> DataFrame:
    """Read a JSON array of objects; retain a tape for one object at a time."""
    _options(schema, infer_schema_length, batch_size)
    var input = JsonInput(path)
    var dtype = schema.value() if schema else _json_schema(
        input, infer_schema_length
    )
    var rows = JsonArrayRows(input.span())
    var decoder = JsonDecoder(
        dtype,
        min(batch_size, input.size // 2),
        JsonDecodeOptions(
            io=True, ignore_errors=ignore_errors, reject_unknown_fields=True
        ),
        columns,
    )
    var frames = List[DataFrame]()
    while rows.next():
        try:
            validate_json_numbers(rows.reader)
            decoder.append(rows.reader)
        except e:
            raise Error(
                "JSON array row " + String(rows.rows) + ": " + String(e)
            )
        if decoder.rows() >= batch_size:
            frames.append(_frame(decoder.finish()))
    if decoder.rows() != 0 or len(frames) == 0:
        frames.append(_frame(decoder.finish()))
    # JsonArrayRows borrows an erased address. Keep the mapping alive until
    # the last parser scan, including when all output fields were projected.
    _ = input^
    return concat(frames) if len(frames) > 1 else frames.pop(0)
