"""Typed fixtures from the independently validated CPU20/private23c witnesses."""
from std.ffi import external_call
from std.memory import bitcast
from std.os import getenv
from dataframe import DataFrame, Series, Column


struct FlushSetting(Movable):
    var previous: String

    def __init__(out self, threshold: Int):
        self.previous = getenv("DATAFRAME_PIPELINE_FLUSH_GROUPS", "\0").copy()
        var name = String("DATAFRAME_PIPELINE_FLUSH_GROUPS\0")
        var value = String(threshold) + "\0"
        _ = external_call["setenv", Int32](
            Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
        )
        _ = name^
        _ = value^

    def __deinit__(deinit self):
        var name = String("DATAFRAME_PIPELINE_FLUSH_GROUPS\0")
        if self.previous == "\0":
            _ = external_call["unsetenv", Int32](Int(name.unsafe_ptr()))
        else:
            var value = self.previous + "\0"
            _ = external_call["setenv", Int32](
                Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
            )
            _ = value^
        _ = name^


def source[
    D: DType
](mode: Int, reverse: Bool = False, n: Int = 256) raises -> DataFrame:
    var domain: List[Int32] = [7, 10, 14, 17, 40, 45, 52, 73]
    var keys = List[Int32](length=n, fill=0)
    var values = List[Scalar[D]](length=n, fill=0)
    var valid = List[Bool](length=n, fill=True)
    for row in range(n):
        keys[row] = domain[row % 8]
        if mode == 0:
            if row < 8:
                values[row] = Scalar[D](1152921504606846976.0)
            elif (
                row >= n // 4
                and row < n // 4 + 8
                or row >= 3 * n // 4
                and row < 3 * n // 4 + 8
            ):
                values[row] = 1
            elif row >= n // 2 and row < n // 2 + 8:
                values[row] = Scalar[D](-1152921504606846976.0)
        else:
            var negative = (row < 64) != reverse
            comptime if D == DType.float32:
                values[row] = Scalar[D](
                    bitcast[DType.float32](
                        UInt32(0x80000000) if negative else UInt32(0)
                    )
                )
            else:
                values[row] = Scalar[D](
                    bitcast[DType.float64](
                        UInt64(0x8000000000000000) if negative else UInt64(0)
                    )
                )
            if mode == 2:
                if row % 8 == 1 or row % 8 == 2 and row < 64:
                    comptime if D == DType.float32:
                        values[row] = Scalar[D](
                            bitcast[DType.float32](
                                UInt32(0x7F800123) if row
                                < 64 else UInt32(0xFFC00456)
                            )
                        )
                    else:
                        values[row] = Scalar[D](
                            bitcast[DType.float64](
                                UInt64(0x7FF0000000000123) if row
                                < 64 else UInt64(0xFFF8000000000456)
                            )
                        )
                elif row % 8 == 3:
                    valid[row] = False
                elif row % 8 == 4:
                    comptime if D == DType.float32:
                        values[row] = Scalar[D](
                            bitcast[DType.float32](
                                UInt32(0xFF800000) if row
                                < 64 else UInt32(0x7F800000)
                            )
                        )
                    else:
                        values[row] = Scalar[D](
                            bitcast[DType.float64](
                                UInt64(0xFFF0000000000000) if row
                                < 64 else UInt64(0x7FF0000000000000)
                            )
                        )
        if mode == 3 and row % 8 == 0:
            comptime if D == DType.float32:
                values[row] = Scalar[D](
                    bitcast[DType.float32](
                        UInt32(0x7FC00123) if row < 64 else UInt32(0xFF800456)
                    )
                )
            else:
                values[row] = Scalar[D](
                    bitcast[DType.float64](
                        UInt64(0x7FF8000000000123) if row
                        < 64 else UInt64(0xFFF0000000000456)
                    )
                )
        if mode == 3 and row % 8 == 1:
            comptime if D == DType.float32:
                values[row] = Scalar[D](
                    bitcast[DType.float32](
                        UInt32(0x7F800000) if row < 64 else UInt32(0xFF800000)
                    )
                )
            else:
                values[row] = Scalar[D](
                    bitcast[DType.float64](
                        UInt64(0x7FF0000000000000) if row
                        < 64 else UInt64(0xFFF0000000000000)
                    )
                )
    return DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("v", Column[Scalar[D]](values^, valid)),
        ]
    )


def priority_source[
    D: DType
](first: UInt64, second: UInt64) raises -> DataFrame:
    var domain: List[Int32] = [7, 10, 14, 17, 40, 45, 52, 73]
    var keys = List[Int32](length=128, fill=0)
    var values = List[Scalar[D]](length=128, fill=0)
    var valid = List[Bool](length=128, fill=False)
    for row in range(128):
        keys[row] = domain[row % 8]
        if row < 8 or row >= 64 and row < 72:
            var bits = first if row < 64 else second
            comptime if D == DType.float32:
                values[row] = Scalar[D](bitcast[DType.float32](UInt32(bits)))
            else:
                values[row] = Scalar[D](bitcast[DType.float64](bits))
            valid[row] = True
    return DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("v", Column[Scalar[D]](values^, valid)),
        ]
    )


def capture_source(chunked: Bool) raises -> DataFrame:
    var frame = source[DType.float64](0, n=6)
    if not chunked:
        return frame^
    var columns = List[Series]()
    for name in frame.columns():
        var series = frame.column(name)
        columns.append(
            Series._from_chunks([series.slice(0, 3), series.slice(3, 3)])
        )
    return DataFrame(columns^)
