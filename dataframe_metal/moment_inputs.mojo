"""Shared native input spans for four-value moment queries."""
from dataframe import Column, DataFrame, DataType, Series
from .accelerator.gpu.typed_storage import coalesced_native_chunks


def coalesced_moment_frame(
    frame: DataFrame, value_count: Int
) raises -> DataFrame:
    """Remove adjacent chunk wrappers without copying values or bitmaps.

    Four-value moments otherwise select a costly fragmented kernel path.
    Only shared, adjacent native windows are joined. Independent allocations
    and gaps keep their existing submission path; other value counts retain
    their established reduction geometry.
    """
    if value_count != 4:
        return frame.copy()
    var columns = List[Series]()
    var changed = False
    var separate = False
    for name in frame.columns():
        var source = frame.column(name)
        var selected = source.copy()
        if source.is_chunked():
            comptime for kind in range(3):
                comptime D = DType.int32 if kind == 0 else (
                    DType.int64 if kind == 1 else DType.float32
                )
                if source.dtype() == DataType.of(D):
                    var spans = coalesced_native_chunks[D](source)
                    if (
                        len(spans) == 1
                        and spans[0]._data.isa[Column[Scalar[D]]]()
                    ):
                        selected = spans[0].copy()
                        changed = True
                    else:
                        separate = True
        columns.append(selected^)
    if not changed or separate:
        return frame.copy()
    return DataFrame(columns^)
