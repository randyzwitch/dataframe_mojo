"""Owned incremental frame iterators over shared immutable buffers."""
from std.collections import Optional
from std.iter import Iterator, StopIteration
from std.memory import ArcPointer
from .frame import DataFrame
from .series import Series
from .value import AnyValue


@fieldwise_init
struct RowView(Copyable, Sized):
    """An owned row reference; cells convert to AnyValue when accessed."""

    var _frame: ArcPointer[DataFrame]
    var _row: Int

    def __len__(self) -> Int:
        return self._frame[].width()

    def __getitem__(self, index: Int) raises -> AnyValue:
        """Read a cell in schema order; reject an out-of-range index."""
        if index < 0 or index >= len(self):
            raise Error("Row column index out of bounds")
        return self._frame[]._columns[index].get(self._row)

    def __getitem__(self, name: String) raises -> AnyValue:
        """Read a cell by column name."""
        return self._frame[].column(name).get(self._row)

    def to_list(self) raises -> List[AnyValue]:
        """Materialize this row in schema order, propagating conversion errors."""
        return self._frame[].row(self._row)


struct RowIterator(Iterator, Movable):
    """Yield owned row views without converting unvisited cells."""

    comptime Element = RowView
    var _frame: ArcPointer[DataFrame]
    var _next: Int

    def __init__(out self, frame: DataFrame):
        self._frame = ArcPointer(frame.copy())
        self._next = 0

    def __iter__(var self) -> Self:
        return self^

    def __next__(mut self) raises StopIteration -> RowView:
        if self._next == self._frame[].height():
            raise StopIteration()
        var row = self._next
        self._next += 1
        return RowView(self._frame.copy(), row)

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var remaining = self._frame[].height() - self._next
        return (remaining, Optional(remaining))


struct ColumnIterator(Iterator, Movable):
    """Yield owned Series in schema order, with no value-buffer copies."""

    comptime Element = Series
    var _frame: ArcPointer[DataFrame]
    var _next: Int

    def __init__(out self, frame: DataFrame):
        self._frame = ArcPointer(frame.copy())
        self._next = 0

    def __iter__(var self) -> Self:
        return self^

    def __next__(mut self) raises StopIteration -> Series:
        if self._next == self._frame[].width():
            raise StopIteration()
        var column = self._next
        self._next += 1
        return self._frame[]._columns[column].copy()

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var remaining = self._frame[].width() - self._next
        return (remaining, Optional(remaining))


struct SliceIterator(Iterator, Movable):
    """Yield zero-copy frame slices lazily; retain buffers after iteration."""

    comptime Element = DataFrame
    var _frame: ArcPointer[DataFrame]
    var _next: Int
    var _size: Int

    def __init__(out self, frame: DataFrame, n: Int) raises:
        if n <= 0:
            raise Error("iter_slices needs a positive slice size")
        self._frame = ArcPointer(frame.copy())
        self._next = 0
        self._size = n

    def __iter__(var self) -> Self:
        return self^

    def __next__(mut self) raises StopIteration -> DataFrame:
        var remaining = self._frame[].height() - self._next
        if remaining == 0:
            raise StopIteration()
        var length = min(self._size, remaining)
        var frame = self._frame[].copy()
        var columns = List[Series](capacity=frame.width())
        for column in frame._columns:
            columns.append(column._slice_validated(self._next, length))
        frame._columns = columns^
        frame._height = length
        frame._statistics = None
        self._next += length
        return frame^

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var remaining = self._frame[].height() - self._next
        var count = remaining // self._size + (
            1 if remaining % self._size else 0
        )
        return (count, Optional(count))
