"""Owned source columns and row mappings at a pipeline's output boundary.

Like a dictionary vector, a selected column keeps its source and a row list.
Columns from the same filter share that list; join columns may use different
lists. These batches own both the sources and the selections, so no pointer
into a worker's reusable chunk or arena escapes the worker.
"""
from std.collections import Optional
from dataframe.frame import DataFrame, concat
from dataframe.series import Series


struct SelectionBatch(Movable):
    var morsel: Int
    var columns: List[Series]
    var starts: List[Int]
    var lists: List[Int]
    var rows: List[List[UInt32]]
    var count: Int
    # Values computed into chunk caches need owned storage. Source columns
    # keep their buffers and do not contribute to this count.
    var copied_columns: Int

    def __init__(out self):
        self.morsel = 0
        self.columns = List[Series]()
        self.starts = List[Int]()
        self.lists = List[Int]()
        self.rows = List[List[UInt32]]()
        self.count = 0
        self.copied_columns = 0

    def validate(self) raises:
        """Check every mapping before a consumer submits asynchronous work."""
        var width = len(self.columns)
        if (
            self.count < 0
            or len(self.starts) != width
            or len(self.lists) != width
        ):
            raise Error("selection batch has inconsistent dimensions")
        var largest = List[Int](capacity=len(self.rows))
        for rows in self.rows:
            if len(rows) != self.count:
                raise Error("selection batch row list has the wrong length")
            var last = -1
            for row in rows:
                last = max(last, Int(row))
            largest.append(last)
        for c in range(width):
            var start = self.starts[c]
            var list = self.lists[c]
            if start < 0 or list < -1 or list >= len(self.rows):
                raise Error("selection batch has an invalid column mapping")
            var last = self.count - 1 if list < 0 else largest[list]
            var length = len(self.columns[c])
            if start > length or (last >= 0 and last >= length - start):
                raise Error("selection batch row is outside its source column")

    def materialize(self) raises -> DataFrame:
        """An explicit adapter for a consumer that requires ordinary columns."""
        self.validate()
        var columns = List[Series](capacity=len(self.columns))
        for c in range(len(self.columns)):
            var list = self.lists[c]
            if list < 0:
                columns.append(
                    self.columns[c].slice(self.starts[c], self.count)
                )
            else:
                var picked = List[Int](capacity=self.count)
                for row in self.rows[list]:
                    picked.append(self.starts[c] + Int(row))
                columns.append(self.columns[c].take(picked))
        return DataFrame(columns^, height=self.count)


struct SelectionBatches(Movable):
    var schema: DataFrame
    var batches: List[SelectionBatch]
    var rows: Int
    var ready: Bool
    # A source that could not end in a selection sink was executed once by
    # its usual path. Its frame can go to the ordinary provider port.
    var fallback: Optional[DataFrame]

    def __init__(out self, schema: DataFrame) raises:
        self.schema = schema.clear()
        self.batches = List[SelectionBatch]()
        self.rows = 0
        self.ready = False
        self.fallback = None

    def append(mut self, var batch: SelectionBatch) raises:
        if batch.count < 0 or batch.count > Int.MAX - self.rows:
            raise Error("selection batch cardinality overflow")
        self.rows += batch.count
        self.batches.append(batch^)

    def validate(self) raises:
        if not self.ready or self.fallback:
            raise Error(
                "selection batches are not ready for a selected consumer"
            )
        var rows = 0
        var previous = -1
        for batch_index in range(len(self.batches)):
            ref batch = self.batches[batch_index]
            batch.validate()
            if (
                batch.morsel < previous
                or len(batch.columns) != self.schema.width()
            ):
                raise Error(
                    "selection batches have inconsistent order or schema"
                )
            previous = batch.morsel
            for c in range(len(batch.columns)):
                if batch.columns[c].dtype() != self.schema._columns[c].dtype():
                    raise Error(
                        "selection batch column type differs from its schema"
                    )
            if batch.count > Int.MAX - rows:
                raise Error("selection batch cardinality overflow")
            rows += batch.count
        if rows != self.rows:
            raise Error("selection batch cardinality differs from its batches")

    def materialize(self) raises -> DataFrame:
        if self.fallback:
            return self.fallback.value().copy()
        self.validate()
        if len(self.batches) == 0:
            return self.schema.copy()
        var frames = List[DataFrame](capacity=len(self.batches))
        for batch_index in range(len(self.batches)):
            ref batch = self.batches[batch_index]
            frames.append(batch.materialize())
        return concat(frames)
