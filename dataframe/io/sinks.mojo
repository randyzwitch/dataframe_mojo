"""Ordered file consumers for query execution batches."""
from dataframe.frame import DataFrame
from .csv_writer import _CsvFileWriter
from .parquet_writer import _ParquetWriter, _check_schema


struct FileSink(Movable):
    """Own one atomic file output; retain a typed empty execution sentinel.

    append consumes each batch synchronously. finish alone replaces the final
    path, allowing a scan to safely write back to its own source path.
    """

    var parquet: List[_ParquetWriter]
    var csv: List[_CsvFileWriter]
    var schema: DataFrame
    var emitted_rows: Int
    var finished: Bool

    def __init__(
        out self,
        schema: DataFrame,
        path: String,
        format: String,
        *,
        compression: String = "zstd",
        row_group_size: Int = 1_000_000,
        has_header: Bool = True,
        separator: String = ",",
        quote_style: String = "necessary",
        null_value: String = "",
        line_terminator: String = "\n",
        buffer_size: Int = 65536,
    ) raises:
        self.parquet = List[_ParquetWriter]()
        self.csv = List[_CsvFileWriter]()
        self.schema = schema.clear()
        self.emitted_rows = 0
        self.finished = False
        if format == "parquet":
            self.parquet.append(
                _ParquetWriter(self.schema, path, compression, row_group_size)
            )
        elif format == "csv":
            self.csv.append(
                _CsvFileWriter(
                    self.schema,
                    path,
                    has_header=has_header,
                    separator=separator,
                    quote_style=quote_style,
                    null_value=null_value,
                    line_terminator=line_terminator,
                    buffer_size=buffer_size,
                )
            )
        else:
            raise Error("unsupported output format: " + format)

    def append(mut self, frame: DataFrame) raises:
        if self.finished:
            raise Error("file sink is closed")
        try:
            _check_schema(self.schema, frame)
            if len(self.parquet):
                self.parquet[0].append(frame)
            else:
                self.csv[0].append(frame)
        except caught:
            # A failed append must never allow a partial result to publish.
            self.finished = True
            raise caught^
        self.emitted_rows += frame.height()

    def finish(mut self) raises:
        if self.finished:
            raise Error("file sink is closed")
        self.finished = True
        if len(self.parquet):
            self.parquet[0].finish()
        else:
            self.csv[0].finish()
