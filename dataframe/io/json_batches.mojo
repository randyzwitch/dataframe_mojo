"""Bounded, ordered NDJSON decode jobs for the lazy streaming executor."""
from std.collections import Optional
from std.memory import ArcPointer
from ..dtype import DataType
from dataframe.json.decode import JsonDecodeOptions
from .json import _options, _ndjson_schema
from .json_source import JsonInput, JsonLines
from .json_parallel import _Context, _DecodeJob


struct _NdjsonBatches(Movable):
    var input: JsonInput
    var lines: JsonLines
    var context: ArcPointer[_Context]
    var batch_size: Int
    var remaining: Int
    var emitted: Bool

    def __init__(
        out self,
        path: String,
        schema: Optional[DataType],
        infer_schema_length: Int,
        ignore_errors: Bool,
        columns: List[String],
        rows: Int,
        batch_size: Int,
    ) raises:
        _options(schema, infer_schema_length, batch_size, rows)
        self.input = JsonInput(path)
        var dtype = schema.value() if schema else _ndjson_schema(
            self.input, infer_schema_length
        )
        self.lines = JsonLines(self.input.span())
        self.context = ArcPointer(
            _Context(
                dtype,
                JsonDecodeOptions(
                    io=True,
                    ignore_errors=ignore_errors,
                    reject_unknown_fields=True,
                ),
                columns,
                batch_size,
            )
        )
        self.remaining = rows
        self.batch_size = batch_size
        self.emitted = False

    def next(mut self) raises -> Optional[_DecodeJob]:
        if self.emitted and (
            self.lines.pos >= self.input.size or self.remaining == 0
        ):
            return None
        var start = self.lines.pos
        var end = start
        var rows = 0
        var size = self.batch_size if self.remaining < 0 else min(
            self.batch_size, self.remaining
        )
        while rows < size:
            var record = self.lines.next()
            if not record:
                break
            end = self.lines.pos
            rows += 1
        if self.remaining >= 0:
            self.remaining -= rows
        if rows == 0 and self.emitted:
            return None
        self.emitted = True
        return _DecodeJob(self.context, self.input.address, start, end - start)
