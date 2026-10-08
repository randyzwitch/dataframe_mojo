"""Issue #224 mechanism measurement; input loading and output writing untimed."""
from std.sys import argv
from std.time import monotonic
from dataframe import DataFrame, read_parquet, write_parquet


def main() raises:
    var args = argv()
    var path = String(args[1])
    var strategy = String(args[2])
    var grouped = String(args[3]) == "grouped"
    var reps = Int(String(args[4]))
    var left = read_parquet(path + "/left.parquet")
    var right = read_parquet(path + "/right.parquet")
    var by = List[String](["g"]) if grouped else List[String]()
    var result = left.head(0)
    for i in range(reps + 1):
        var start = monotonic()
        var output = left.join_asof(right, on="k", by=by, strategy=strategy)
        var elapsed = monotonic() - start
        if output.height() != left.height():
            raise Error("as-of benchmark changed left cardinality")
        result = output^
        if i > 0:
            print(elapsed)
    write_parquet(result, path + "/mojo-output.parquet")
