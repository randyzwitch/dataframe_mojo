"""Ordered file consumers for query execution batches."""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import Allocation, Layout, Pointer, alloc, dealloc
from dataframe.frame import DataFrame
from dataframe.series import Series
from dataframe.parallel import Job, configured_workers, run_jobs
from .csv_writer import _CsvFileWriter
from .parquet_writer import _ParquetWriter, _check_schema


struct _Prepared(Movable):
    """One batch made ready to write on the thread that produced it: CSV
    text already formatted, or the rows as they are for a Parquet writer,
    whose encoding runs as its row groups fill."""

    var text: String
    var frame: DataFrame
    var rows: Int

    def __init__(out self, var text: String, var frame: DataFrame, rows: Int):
        self.text = text^
        self.frame = frame^
        self.rows = rows


struct _Turn(Movable):
    """The number of the next batch to write and whether a writer failed,
    in their own heap cells, which every worker reads and the writing one
    changes: the sink holding them may move, the cells stay put."""

    var address: Int

    def __init__(out self):
        var pointer = alloc(Layout[Atomic[Int64]](length=2)).unsafe_leak()
        pointer.unsafe_write(Atomic[Int64](0))
        (pointer + 1).unsafe_write(Atomic[Int64](0))
        self.address = Int(pointer)

    def __deinit__(deinit self):
        dealloc(
            Allocation[Atomic[Int64]](
                unsafe_owned_ptr=Pointer[Atomic[Int64], MutUntrackedOrigin](
                    unsafe_from_address=self.address
                ),
                layout=Layout[Atomic[Int64]](length=2),
            )
        )

    @always_inline
    def _cell(self, k: Int) -> ref[MutAnyOrigin] Atomic[Int64]:
        return Pointer[Atomic[Int64], MutAnyOrigin](
            unsafe_from_address=self.address + 8 * k
        )[]

    def next(self) -> Int:
        return Int(self._cell(0).load())

    def set_next(self, batch: Int):
        self._cell(0).store(Int64(batch))

    def failed(self) -> Bool:
        return self._cell(1).load() != 0

    def fail(self):
        self._cell(1).store(1)


struct _PrepareJob(Job):
    """One slice of a frame made ready to write (`FileSink.prepare`)."""

    var sink: Int
    var frame: DataFrame
    var prepared: List[_Prepared]

    def __init__(out self, sink: Int, var frame: DataFrame):
        self.sink = sink
        self.frame = frame^
        self.prepared = List[_Prepared]()

    def run(mut self) raises:
        self.prepared.append(
            Pointer[FileSink, MutAnyOrigin](
                unsafe_from_address=self.sink
            )[].prepare(self.frame)
        )


def _write_at(address: Int, var prepared: _Prepared) raises:
    """`FileSink._write` on the sink at `address`."""
    Pointer[FileSink, MutAnyOrigin](unsafe_from_address=address)[]._write(
        prepared^
    )


struct FileSink(Movable):
    """Own one atomic file output; retain a typed empty execution sentinel.

    Rows reach the file in batch order. A pipeline's workers hand their
    batches over as each is complete (`offer_at`), numbered by the morsel
    they came from: each worker formats its own batch in parallel with the
    others, then writes it when every batch before it is written. Workers
    take morsels in increasing order, so a worker waits only for the
    morsels the others still hold. `append` and `write_frame` add rows
    after every batch handed over. finish alone replaces the final path,
    allowing a scan to safely write back to its own source path.
    """

    var parquet: List[_ParquetWriter]
    var csv: List[_CsvFileWriter]
    var schema: DataFrame
    var emitted_rows: Int
    var finished: Bool
    # The next batch number to write, and whether a writer failed.
    var turn: _Turn

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
        self.turn = _Turn()
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

    def prepare(self, frame: DataFrame) raises -> _Prepared:
        """`frame` made ready to write, on the calling thread: it reads
        only the sink's settings."""
        _check_schema(self.schema, frame)
        if len(self.csv):
            return _Prepared(
                self.csv[0].format(frame),
                DataFrame(List[Series](), height=0),
                frame.height(),
            )
        return _Prepared(String(), frame.copy(), frame.height())

    @no_inline
    def _write(mut self, var prepared: _Prepared) raises:
        """One prepared batch into the file, by the one thread writing. Not
        inlined: its changes to the writer are made before it returns, so
        before the next turn is published."""
        try:
            if len(self.parquet):
                if prepared.rows > 0:
                    self.parquet[0].append(prepared.frame)
            else:
                self.csv[0].write_text(prepared.text)
        except caught:
            # A failed write must never allow a partial result to publish.
            self.finished = True
            raise caught^
        self.emitted_rows += prepared.rows

    @staticmethod
    def offer_at(address: Int, batch: Int, frame: DataFrame) raises:
        """Batch `batch` to the sink at `address`, from the worker that
        made it: prepared on this thread at once, then written when its
        turn comes, once every batch numbered before it is written. A
        pipeline numbers its batches from 0 with no gap (a batch with no
        rows is offered empty), and a worker hands its batch over before
        it waits for anything else, so the batch whose turn it is always
        comes. Only the worker whose turn it is touches the writer."""
        ref sink = Pointer[FileSink, MutAnyOrigin](
            unsafe_from_address=address
        )[]
        var prepared = sink.prepare(frame)
        while sink.turn.next() != batch:
            if sink.turn.failed():
                raise Error("file sink is closed")
            _ = external_call["sched_yield", Int32]()
        try:
            _write_at(address, prepared^)
        except caught:
            sink.turn.fail()
            raise caught^
        sink.turn.set_next(batch + 1)

    @staticmethod
    def fail_at(address: Int):
        """A worker of the pipeline writing to the sink at `address` failed:
        the batch it held never comes, so no worker waits for it."""
        Pointer[FileSink, MutAnyOrigin](
            unsafe_from_address=address
        )[].turn.fail()

    def append(mut self, frame: DataFrame) raises:
        """`frame`'s rows after every row before them, on this thread."""
        if self.finished:
            raise Error("file sink is closed")
        self._write(self.prepare(frame))

    def write_frame(mut self, frame: DataFrame, batch_size: Int) raises:
        """`frame`'s rows after every row before them: its slices of
        `batch_size` rows prepared in parallel, up to two per worker at a
        time, and written in order."""
        if self.finished:
            raise Error("file sink is closed")
        var size = max(1, batch_size)
        var wave = 2 * max(1, configured_workers())
        var offset = 0
        while offset < frame.height():
            var jobs = List[_PrepareJob]()
            while offset < frame.height() and len(jobs) < wave:
                jobs.append(
                    _PrepareJob(
                        Int(Pointer(to=self)), frame.slice(offset, size)
                    )
                )
                offset += size
            if len(jobs) == 1:
                jobs[0].run()
            else:
                run_jobs(jobs)
            for k in range(len(jobs)):
                self._write(jobs[k].prepared.pop())

    def finish(mut self) raises:
        if self.finished:
            raise Error("file sink is closed")
        self.finished = True
        if len(self.parquet):
            self.parquet[0].finish()
        else:
            self.csv[0].finish()
