"""Incremental CSV output using the established CSV cell formatter."""
from std.collections import Optional
from std.io import FileHandle
from dataframe.categorical import decode
from dataframe.frame import DataFrame
from .csv import _CsvWriter, _reject_nested
from .parquet_writer import (
    _AtomicOutput,
    _check_schema,
    _validate_path,
    _validate_schema,
)


struct _CsvFileWriter(Movable):
    var output: _AtomicOutput
    var file: Optional[FileHandle]
    var writer: _CsvWriter
    var schema: DataFrame
    var buffer: String
    var buffer_size: Int
    var pending: String

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
        self.pending = String()
        self.file = None
        self.output = _AtomicOutput(path)
        try:
            self.file = open(self.output.temporary, "w")
        except:
            raise Error(
                "cannot open CSV temporary output: " + self.output.temporary
            )

    def _flush(mut self) raises:
        if self.buffer.byte_length() == 0:
            return
        # C stdio previously coalesced tiny formatter buffers. Retain a
        # small-write staging buffer with a 4 KiB flush threshold. Normal
        # chunks borrow their existing bytes with no extra allocation.
        if self.buffer.byte_length() < 4096:
            self.pending += self.buffer
            self.buffer = String()
            if self.pending.byte_length() >= 4096:
                self._flush_pending()
            return
        self._flush_pending()
        try:
            self.file.value().write_all(self.buffer.as_bytes())
        except:
            raise Error("cannot write CSV temporary output")
        self.buffer = String()

    def _flush_pending(mut self) raises:
        if self.pending.byte_length() == 0:
            return
        try:
            self.file.value().write_all(self.pending.as_bytes())
        except:
            raise Error("cannot write CSV temporary output")
        self.pending = String()

    def append(mut self, frame: DataFrame) raises:
        if not self.file:
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
        if not self.file:
            raise Error("CSV writer is closed")
        self._flush()
        self._flush_pending()
        var file = self.file.take()
        try:
            file^.close()
        except:
            raise Error("cannot close CSV temporary output")
        self.output.publish()
