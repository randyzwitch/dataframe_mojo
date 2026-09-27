"""Import-only copy timing. Args: rows columns repetitions."""
from std.sys import argv
from std.time import monotonic
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    Series,
    export_arrow,
    import_arrow,
)


def main() raises:
    var args = argv()
    var rows = Int(String(args[1]))
    var width = Int(String(args[2]))
    var reps = Int(String(args[3]))
    var columns = List[Series]()
    for k in range(width):
        var values = List[Int64](capacity=rows)
        for i in range(rows):
            values.append(Int64(i + k))
        columns.append(Series("c" + String(k), Column[Int64](values^)))
    var source = DataFrame(columns^)
    for rep in range(reps + 1):
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow(source, array, schema)
        var start = monotonic()
        var result = import_arrow(array, schema)
        var elapsed = monotonic() - start
        for k in range(width):
            if not result._columns[k].equals(source._columns[k]):
                raise Error("import differs")
        if array.release != 0 or schema.release != 0:
            raise Error("unreleased exports")
        if rep > 0:
            print(elapsed)
