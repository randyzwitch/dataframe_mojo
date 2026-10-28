"""Ordered NDJSON range jobs using the existing scoped CSV worker pool."""
from std.memory import ArcPointer, Pointer
from ..dtype import DataType
from ..frame import DataFrame, concat
from ..series import Series
from dataframe.json.decode import JsonDecoder, JsonDecodeOptions
from dataframe.json.reader import JsonReader, JsonBytes
from dataframe.json.number import validate_json_numbers
from ..parallel import Job, Pool, _ProducedJobs, configured_workers
from .csv_scan import CountLines, chunk_size
from .json_source import JsonInput, JsonLines


def json_frame(series: Series) raises -> DataFrame:
    var column = series.struct_column()
    var fields = List[Series](capacity=column.field_count())
    for i in range(column.field_count()):
        fields.append(column.field(i))
    return DataFrame(fields^, height=len(series))


struct _Context(Movable):
    var dtype: DataType
    var options: JsonDecodeOptions
    var columns: List[String]
    var batch_size: Int

    def __init__(
        out self,
        dtype: DataType,
        options: JsonDecodeOptions,
        columns: List[String],
        batch_size: Int,
    ):
        self.dtype = dtype.copy()
        self.options = options
        self.columns = columns.copy()
        self.batch_size = batch_size


struct _DecodeJob(Job):
    var context: ArcPointer[_Context]
    var address: Int
    var offset: Int
    var length: Int
    var result: DataFrame

    def __init__(
        out self,
        context: ArcPointer[_Context],
        address: Int,
        offset: Int,
        length: Int,
    ) raises:
        self.context = context.copy()
        self.address = address
        self.offset = offset
        self.length = length
        self.result = DataFrame(List[Series](), height=0)

    def span(self, offset: Int, length: Int) -> JsonBytes:
        return JsonBytes(
            unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=self.address + offset
            ),
            length=length,
        )

    def run(mut self) raises:
        var bytes = self.span(self.offset, self.length)
        var lines = JsonLines(bytes)
        var reader = JsonReader(bytes)
        var decoder = JsonDecoder(
            self.context[].dtype,
            min(self.context[].batch_size, self.length // 2),
            self.context[].options,
            self.context[].columns,
        )
        var frames = List[DataFrame]()
        while True:
            var record = lines.next()
            if not record:
                break
            try:
                reader.parse(bytes[record.value().start : record.value().end])
                validate_json_numbers(reader)
                decoder.append(reader)
            except e:
                # Count earlier physical lines only on failure. Normal reads
                # need neither a whole-file row index nor a counting pass.
                var preceding = (
                    CountLines(34, False).count(self.span(0, self.offset)).rows
                )
                raise Error(
                    "NDJSON at line "
                    + String(preceding + record.value().line + 1)
                    + ": "
                    + String(e)
                )
            if decoder.rows() >= self.context[].batch_size:
                frames.append(json_frame(decoder.finish()))
        if decoder.rows() != 0 or len(frames) == 0:
            frames.append(json_frame(decoder.finish()))
        self.result = concat(frames) if len(frames) > 1 else frames.pop(0)

    def into_frame(deinit self) -> DataFrame:
        return self.result^


def read_parallel(
    input: JsonInput,
    dtype: DataType,
    options: JsonDecodeOptions,
    columns: List[String],
    batch_size: Int,
) raises -> DataFrame:
    """Decode disjoint LF-aligned ranges and assemble their owned arrays.

    Like Polars' NDJSON reader, ranges borrow one mapped source and retain
    source order. Reuse CSV's byte-budget sizing and produced-job lifetime
    protocol; the caller keeps the input alive through every worker join.
    """
    var workers = min(configured_workers(), max(1, input.size // (256 * 1024)))
    var size = chunk_size(input.size, workers, dtype.field_count())
    var capacity = 2 * ((input.size + size - 1) // size) + 2
    var context = ArcPointer(_Context(dtype, options, columns, batch_size))
    var pool = Pool(workers)
    var produced = _ProducedJobs[_DecodeJob](capacity)
    pool.run_produced(produced)
    var offset = 0
    var bytes = input.span()
    var counter = CountLines(34, False)
    while offset < input.size:
        var boundary = counter.find_next(bytes[offset:], size)
        size = boundary.chunk_size
        var end = (
            offset
            + boundary.last_newline
            + 1 if boundary.rows else (input.size)
        )
        produced.submit(
            _DecodeJob(context, input.address, offset, end - offset)
        )
        offset = end
    if input.size == 0:
        produced.submit(_DecodeJob(context, input.address, 0, 0))
    var jobs = produced.finish()
    pool.release()
    var frames = List[DataFrame](capacity=len(jobs))
    jobs.reverse()
    while len(jobs) > 0:
        frames.append(jobs.pop().into_frame())
    return concat(frames) if len(frames) > 1 else frames.pop(0)
