"""Spike: read a Parquet file with marrow and import it through the Arrow
C Data Interface into a DataFrame. Not part of the package.

Usage: spike_parquet FILE.parquet
"""
from std.memory import Pointer
from std.sys import argv

from marrow.c_data import CArrowArray, CArrowSchema
from marrow.parquet import read_table

from dataframe import DataFrame, Series
from dataframe.arrow import import_arrow_series


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: spike_parquet FILE.parquet")
    var table = read_table(String(args[1]))
    var batch = table.combine_chunks()
    var columns = List[Series]()
    for i in range(batch.num_columns()):
        var c_array = CArrowArray.from_array(batch.column(i))
        var c_schema = CArrowSchema.from_field(batch.schema.fields[i])
        columns.append(
            import_arrow_series(
                Int(Pointer(to=c_array)), Int(Pointer(to=c_schema))
            )
        )
        # The address arguments carry no origin, so this use keeps the
        # structs alive until the importer has released them; it also
        # checks that the release handshake happened exactly once.
        if not c_array.is_released() or not c_schema.is_released():
            raise Error("import did not release the exported structs")
    var frame = DataFrame(columns^, height=batch.num_rows())
    print(frame)
    for name in frame.columns():
        print(name, frame.column(name).dtype(), sep="\t")
