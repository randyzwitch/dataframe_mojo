"""A file source's parallel scan state: the batches of one CSV, Parquet,
IPC or NDJSON file (or a dataset of them) handed out in source order to
whichever thread asks next.

A file scan is the source of a pipeline: one shared state hands each
thread the next piece of the file under a lock, and the thread decodes
that piece itself, so reading is serialized only for the short cut while
decoding runs on every thread at once. `ScanSource` is that shared state
over this library's batch readers: `take` cuts the next batch under the
lock (a CSV or NDJSON byte range, or an imported Arrow batch), and
`ScanBatch.into_frame` decodes a byte range on the calling
thread, outside the lock."""
from std.collections import Optional
from std.memory import Pointer
from std.utils.lock import BlockingSpinLock

from dataframe.categorical import ScanDictionary, recode_codes
from dataframe.dtype import DataType
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


struct ScanSource(Movable):
    """One file source's readers (exactly one of the lists holds one), the
    batches handed out so far, and the lock `take` holds while it cuts the
    next. A CSV or NDJSON byte range points into the file's mapping, which
    stays mapped until `finish`."""

    var csv: List[_CsvBatches]
    var dataset: List[_DatasetBatches]
    var parquet: List[_ParquetBatches]
    var ipc: List[_IpcBatches]
    var ndjson: List[_NdjsonBatches]
    var lock: BlockingSpinLock
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
        self.lock = BlockingSpinLock()
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

    def take(mut self, last: Int = Int.MAX) raises -> Optional[ScanBatch]:
        """The next batch in source order, for any thread; None once the
        source is done, or when the next batch would be numbered past
        `last` (a limit's rows are all in the batches before it). Decoding
        a byte range is left to the caller."""
        var owner = Int(Pointer(to=self))
        self.lock.lock(owner)
        try:
            var number = self.ahead[len(self.ahead) - 1].number if len(
                self.ahead
            ) else self.taken
            if number > last:
                _ = self.lock.unlock(owner)
                return None
            var batch = self._next()
            _ = self.lock.unlock(owner)
            return batch^
        except e:
            self.ended = True
            _ = self.lock.unlock(owner)
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
            var typed = column.with_dtype(shared.schema_type())
            columns._columns[c] = typed^
            self.dictionary_names.append(column.name())
            self.dictionaries.append(shared^)
        return len(self.dictionaries) > 0

    def frame(mut self, var batch: ScanBatch) raises -> DataFrame:
        """The batch's rows (`ScanBatch.into_frame`), with each categorical
        column's codes moved onto its dictionary over the scan when
        `share_dictionaries` set one up. Any thread."""
        var frame = batch^.into_frame()
        if len(self.dictionaries) == 0:
            return frame^
        for c in range(frame.width()):
            ref column = frame._columns[c]
            if (
                not column.dtype().is_categorical()
                or not column.dtype().has_dictionary()
            ):
                continue
            var at = -1
            for d in range(len(self.dictionary_names)):
                if self.dictionary_names[d] == column.name():
                    at = d
                    break
            if at < 0:
                continue
            var codes = List[UInt32]()
            var dtype = DataType.NULL
            var owner = Int(Pointer(to=self))
            self.lock.lock(owner)
            try:
                var added = self.dictionaries[at].add(
                    column.dtype().dictionary()
                )
                codes = added[0].copy()
                dtype = added[1]
            except e:
                _ = self.lock.unlock(owner)
                raise e^
            _ = self.lock.unlock(owner)
            var recoded = recode_codes(column, codes, dtype)
            frame._columns[c] = recoded^
        return frame^

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
