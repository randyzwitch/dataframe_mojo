"""Incremental CSV output using the established CSV cell formatter."""
from std.ffi import external_call
from dataframe.categorical import decode
from dataframe.frame import DataFrame
from .arrow import _c_string
from .csv import _CsvWriter, _reject_nested
from .parquet_writer import (
    _AtomicOutput,
    _check_schema,
    _validate_path,
    _validate_schema,
)


struct _CsvFileWriter(Movable):
    var output: _AtomicOutput
    var file: Int
    var writer: _CsvWriter
    var schema: DataFrame
    var buffer: String
    var buffer_size: Int

    def __init__(
        out self,
        frame: DataFrame,
        path: String,
        *,
        has_header: Bool = True,
        separator: String = ",",
        quote_style: String = "necessary",
        null_value: String = "",
        line_terminator: String = "\n",
        buffer_size: Int = 65536,
    ) raises:
        _validate_path(path)
        _validate_schema(frame)
        _reject_nested(frame)
        if buffer_size <= 0:
            raise Error("CSV buffer_size must be positive")
        self.schema = frame.clear()
        self.writer = _CsvWriter(
            self.schema,
            separator,
            quote_style,
            null_value,
            line_terminator,
        )
        self.buffer_size = buffer_size
        self.buffer = self.writer.header(
            self.schema
        ) if has_header else String()
        self.file = 0
        self.output = _AtomicOutput(path)
        var filename = _c_string(self.output.temporary)
        var mode = _c_string("wb")
        self.file = external_call["fopen", Int](
            filename.unsafe_ptr(), mode.unsafe_ptr()
        )
        _ = filename^
        _ = mode^
        if self.file == 0:
            raise Error(
                "cannot open CSV temporary output: " + self.output.temporary
            )

    def __deinit__(deinit self):
        if self.file != 0:
            _ = external_call["fclose", Int32](self.file)

    def _flush(mut self) raises:
        if self.buffer.byte_length() == 0:
            return
        var bytes = _c_string(self.buffer)
        var count = self.buffer.byte_length()
        var written = external_call["fwrite", UInt](
            bytes.unsafe_ptr(),
            UInt(1),
            UInt(count),
            self.file,
        )
        _ = bytes^
        if written != UInt(count):
            raise Error("cannot write CSV temporary output")
        self.buffer = String()

    def append(mut self, frame: DataFrame) raises:
        if self.file == 0:
            raise Error("CSV writer is closed")
        _check_schema(self.schema, frame)
        var data = frame.copy()
        for k in range(data.width()):
            if data._columns[k].dtype().is_categorical():
                data._columns[k] = decode(data._columns[k])
        for row in range(data.height()):
            self.buffer += self.writer.row(data, row)
            if self.buffer.byte_length() >= self.buffer_size:
                self._flush()

    def finish(mut self) raises:
        if self.file == 0:
            raise Error("CSV writer is closed")
        self._flush()
        var file = self.file
        self.file = 0
        if external_call["fclose", Int32](file) != 0:
            raise Error("cannot close CSV temporary output")
        self.output.publish()
