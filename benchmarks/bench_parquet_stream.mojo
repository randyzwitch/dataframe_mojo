"""One eager read; external driver records process peak RSS. Args: path rows."""
from std.sys import argv
from std.time import monotonic
from dataframe import read_parquet


def main() raises:
    var args = argv()
    var start = monotonic()
    var frame = read_parquet(String(args[1]))
    var elapsed = monotonic() - start
    var rows = Int(String(args[2]))
    if frame.height() != rows or frame.width() != 8:
        raise Error("wrong frame shape")
    for column in range(8):
        if frame._columns[column].get(rows - 1).int64() != Int64(
            rows - 1 + column
        ):
            raise Error("wrong last value")
    print(elapsed, frame.column("c0").n_chunks())
