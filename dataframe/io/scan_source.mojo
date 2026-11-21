"""A file source's parallel scan state: the batches of one CSV, Parquet,
IPC or NDJSON file (or a dataset of them) handed out in source order to
whichever thread asks next.

A file scan is the source of a pipeline: one shared state hands each
thread the next piece of the file under a lock, and the thread decodes
that piece itself, so reading is serialized only for the short cut while
decoding runs on every thread at once. `ScanSource` is that shared state
over this library's batch readers: `take_at` cuts the next batch under the
lock (a CSV or NDJSON byte range, or an imported Arrow batch), and
`ScanBatch.into_frame` decodes a byte range on the calling
thread, outside the lock."""
from std.atomic import Atomic
from std.collections import Optional
from std.ffi import external_call
from std.memory import ArcPointer, Allocation, Layout, Pointer, alloc, dealloc

from dataframe.categorical import ScanDictionary, recode_codes
from dataframe.dtype import CategoricalDictionary, DataType
from dataframe.frame import DataFrame
from dataframe.series import Series
from .csv_reader import _CsvBatches, _DecodeJob
from .dataset_scan import _DatasetBatches
from .ipc import _IpcBatches
from .json_batches import _NdjsonBatches
from .json_parallel import _DecodeJob as _JsonDecodeJob
from .parquet import _ParquetBatches


struct ScanBatch(Movable):
    """One batch of a scan, numbered in source order: a decoded frame, or
    the CSV or NDJSON byte range to decode."""

    var number: Int
    var frame: DataFrame
    var csv: List[_DecodeJob]
    var json: List[_JsonDecodeJob]

    def __init__(out self, number: Int) raises:
        self.number = number
        self.frame = DataFrame(List[Series](), height=0)
        self.csv = List[_DecodeJob]()
        self.json = List[_JsonDecodeJob]()

    def into_frame(deinit self) raises -> DataFrame:
        """The batch's rows, decoding its byte range here."""
        if len(self.csv):
            self.csv[0].run()
            return self.csv.pop().into_frame()
        if len(self.json):
            self.json[0].run()
            return self.json.pop().into_frame()
        return self.frame^


struct _ScanLock(Movable):
    """A spin lock word in its own heap cell. Workers reach the source it
    guards by address (`ScanSource.take_at`), never through a `mut`
    argument: a `mut` argument promises the callee sole access to the
    value's memory, so the compiler may move that memory's loads and
    stores across the lock's atomics, and a worker would read state
    another worker is still changing (or spin on a value it never
    reloads). The word sits outside the source, so taking it touches no
    memory any argument claims."""

    var address: Int

    def __init__(out self):
        var pointer = alloc(Layout[Atomic[Int64]](length=1)).unsafe_leak()
        pointer.unsafe_write(Atomic[Int64](0))
        self.address = Int(pointer)

    def __deinit__(deinit self):
        dealloc(
            Allocation[Atomic[Int64]](
                unsafe_owned_ptr=Pointer[Atomic[Int64], MutUntrackedOrigin](
                    unsafe_from_address=self.address
                ),
                layout=Layout[Atomic[Int64]](length=1),
            )
        )

    @always_inline
    def _word(self) -> ref[MutAnyOrigin] Atomic[Int64]:
        return Pointer[Atomic[Int64], MutAnyOrigin](
            unsafe_from_address=self.address
        )[]

    def acquire(self):
        while True:
            var free = Int64(0)
            if self._word().compare_exchange(free, 1):
                return
            _ = external_call["sched_yield", Int32]()

    def release(self):
        self._word().store(0)


def _source_at(address: Int) -> ref[MutAnyOrigin] ScanSource:
    return Pointer[ScanSource, MutAnyOrigin](unsafe_from_address=address)[]


struct ScanSource(Movable):
    """One file source's readers (exactly one of the lists holds one), the
    batches handed out so far, and the lock `take_at` holds while it cuts
    the next. A CSV or NDJSON byte range points into the file's mapping,
    which stays mapped until `finish`.

    Workers share one source and reach it by address (`take_at`,
    `frame_at`); the state the lock guards is changed only inside the
    methods those call while holding it, which are not inlined, so each
    change is complete before the lock is released."""

    var csv: List[_CsvBatches]
    var dataset: List[_DatasetBatches]
    var parquet: List[_ParquetBatches]
    var ipc: List[_IpcBatches]
    var ndjson: List[_NdjsonBatches]
    var lock: _ScanLock
    var taken: Int
    var ended: Bool
    # A batch `schema` read ahead to learn the source's columns, handed
    # out first.
    var ahead: List[ScanBatch]
    # Each categorical column's dictionary over the whole scan, by name,
    # once `share_dictionaries` set them up; added to under `lock`.
    var dictionary_names: List[String]
    var dictionaries: List[ScanDictionary]

    def __init__(out self):
        self.csv = List[_CsvBatches]()
        self.dataset = List[_DatasetBatches]()
        self.parquet = List[_ParquetBatches]()
        self.ipc = List[_IpcBatches]()
        self.ndjson = List[_NdjsonBatches]()
        self.lock = _ScanLock()
        self.taken = 0
        self.ended = False
        self.ahead = List[ScanBatch]()
        self.dictionary_names = List[String]()
        self.dictionaries = List[ScanDictionary]()

    @staticmethod
    def of(var csv: _CsvBatches) -> Self:
        var source = Self()
        source.csv.append(csv^)
        return source^

    @staticmethod
    def of(var dataset: _DatasetBatches) -> Self:
        var source = Self()
        source.dataset.append(dataset^)
        return source^

    @staticmethod
    def of(var parquet: _ParquetBatches) -> Self:
        var source = Self()
        source.parquet.append(parquet^)
        return source^

    @staticmethod
    def of(var ipc: _IpcBatches) -> Self:
        var source = Self()
        source.ipc.append(ipc^)
        return source^

    @staticmethod
    def of(var ndjson: _NdjsonBatches) -> Self:
        var source = Self()
        source.ndjson.append(ndjson^)
        return source^

    def _next(mut self) raises -> Optional[ScanBatch]:
        """The next batch, numbered; called under the lock (or by the one
        thread that owns the source)."""
        if len(self.ahead):
            return self.ahead.pop()
        if self.ended:
            return None
        var batch = ScanBatch(self.taken)
        if len(self.csv):
            var next = self.csv[0].next()
            if not next:
                self.ended = True
                return None
            batch.csv.append(next.take())
        elif len(self.ndjson):
            var next = self.ndjson[0].next()
            if not next:
                self.ended = True
                return None
            batch.json.append(next.take())
        else:
            var next = Optional[DataFrame]()
            if len(self.dataset):
                next = self.dataset[0].next()
            elif len(self.parquet):
                next = self.parquet[0].next()
            elif len(self.ipc):
                next = self.ipc[0].next()
            if not next:
                self.ended = True
                return None
            batch.frame = next.take()
        self.taken += 1
        return batch^

    @staticmethod
    def take_at(
        address: Int, last: Int = Int.MAX
    ) raises -> Optional[ScanBatch]:
        """The next batch in source order from the source at `address`, for
        any thread; None once the source is done, or when the next batch
        would be numbered past `last` (a limit's rows are all in the
        batches before it). Decoding a byte range is left to the caller.
        """
        ref source = _source_at(address)
        source.lock.acquire()
        try:
            var batch = _source_at(address)._take_locked(last)
            source.lock.release()
            return batch^
        except e:
            source.lock.release()
            raise e^

    @no_inline
    def _take_locked(mut self, last: Int) raises -> Optional[ScanBatch]:
        """`take_at`'s work, under the lock."""
        try:
            var number = self.ahead[len(self.ahead) - 1].number if len(
                self.ahead
            ) else self.taken
            if number > last:
                return None
            return self._next()
        except e:
            self.ended = True
            raise e^

    def schema(mut self) raises -> Optional[DataFrame]:
        """The source's columns, empty, before any batch is handed out: a
        CSV or NDJSON byte range of no rows decoded, or the first decoded
        batch read ahead. None when the source has no batch at all."""
        var batch = self._next()
        if not batch:
            return None
        var first = batch.take()
        if len(first.csv):
            ref job = first.csv[0]
            var empty = _DecodeJob(job.context, job.address, 0, 0, job.record)
            empty.run()
            var columns = empty^.into_frame()
            self.ahead.append(first^)
            return columns^
        if len(first.json):
            ref job = first.json[0]
            var empty = _JsonDecodeJob(job.context, job.address, job.offset, 0)
            empty.run()
            var columns = empty^.into_frame()
            self.ahead.append(first^)
            return columns^
        var columns = first.frame.clear()
        self.ahead.append(first^)
        return columns^

    def share_dictionaries(mut self, mut columns: DataFrame) raises -> Bool:
        """Give each categorical column of the source's `columns` (its
        `schema`) one dictionary over the whole scan: `frame` moves every
        batch's codes onto it, and `columns` are typed by it from here on.
        Before any batch is handed out. Whether any column is categorical.
        """
        for c in range(columns.width()):
            ref column = columns._columns[c]
            if not column.dtype().is_categorical():
                continue
            if column.name() in self.dictionary_names:
                continue
            var shared = ScanDictionary()
            # The column's type gains the scan's dictionary but it is the
            # same field, so its Arrow metadata stays, as each batch's does.
            var typed = column.with_dtype(shared.schema_type())
            typed._field_metadata = column._field_metadata
            columns._columns[c] = typed^
            self.dictionary_names.append(column.name())
            self.dictionaries.append(shared^)
        return len(self.dictionaries) > 0

    @staticmethod
    def frame_at(address: Int, var batch: ScanBatch) raises -> DataFrame:
        """The batch's rows (`ScanBatch.into_frame`), decoded on this
        thread, with each categorical column's codes moved onto its
        dictionary over the scan when `share_dictionaries` set one up.
        The source at `address`; any thread."""
        var frame = batch^.into_frame()
        # The names and the list of dictionaries are fixed before any batch
        # is handed out; only a dictionary's values change, under the lock.
        ref source = _source_at(address)
        if len(source.dictionaries) == 0:
            return frame^
        for c in range(frame.width()):
            ref column = frame._columns[c]
            if (
                not column.dtype().is_categorical()
                or not column.dtype().has_dictionary()
            ):
                continue
            var at = -1
            for d in range(len(source.dictionary_names)):
                if source.dictionary_names[d] == column.name():
                    at = d
                    break
            if at < 0:
                continue
            var codes = List[UInt32]()
            var dtype = DataType.NULL
            source.lock.acquire()
            try:
                var added = _source_at(address)._add_locked(
                    at, column.dtype().dictionary()
                )
                codes = added[0].copy()
                dtype = added[1]
            except e:
                source.lock.release()
                raise e^
            source.lock.release()
            var recoded = recode_codes(column, codes, dtype)
            frame._columns[c] = recoded^
        return frame^

    @no_inline
    def _add_locked(
        mut self, at: Int, values: ArcPointer[CategoricalDictionary]
    ) raises -> Tuple[List[UInt32], DataType]:
        """A batch's dictionary added to scan dictionary `at`, under the
        lock: each batch value's code in it, and its type now."""
        return self.dictionaries[at].add(values)

    def settle(mut self):
        """Every value each scan dictionary holds into the schema's, once
        no batch is being read: columns typed from the schema hold codes
        from every batch."""
        for d in range(len(self.dictionaries)):
            self.dictionaries[d].settle()

    def batches(self) -> Int:
        """The batches handed out so far."""
        return self.taken

    def discard(mut self):
        """Unmap the CSV pages every batch handed out has read past; only
        once no byte range handed out is still being decoded."""
        if len(self.csv):
            self.csv[0].discard()

    def finish(mut self) raises:
        """Close the readers; the source hands out no further batch."""
        self.ended = True
        self.ahead.clear()
        if len(self.csv):
            self.csv[0].finish()
        if len(self.dataset):
            self.dataset[0].finish()
