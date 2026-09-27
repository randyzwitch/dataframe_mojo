"""Sort rank-word sweep. Args: rows words distinct skew-percent bits reps.

Construction and stable permutation validation are outside timing. The driver
builds isolated forced-bucket and merge binaries; production is unmodified.
"""
from std.sys import argv
from std.time import monotonic
from dataframe.series import sort_indices, _rank_less


def main() raises:
    var args = argv()
    if len(args) != 7:
        raise Error("rows words distinct skew-percent bits reps")
    var n = Int(String(args[1]))
    var words = Int(String(args[2]))
    var distinct = Int(String(args[3]))
    var skew = Int(String(args[4]))
    var bits = Int(String(args[5]))
    var reps = Int(String(args[6]))
    var ranks = List[List[Int]]()
    var state = UInt64(20260926)
    for word in range(words):
        var values = List[Int](capacity=n)
        for row in range(n):
            state = state * 6364136223846793005 + 1442695040888963407
            var value = Int((state >> 16) & ((UInt64(1) << UInt64(bits)) - 1))
            if word == 0:
                value = Int((state >> 16) % UInt64(distinct))
                if skew > 0:
                    value = 0 if row % 100 < skew else 1 + value % (
                        distinct - 1
                    )
            values.append(value)
        ranks.append(values^)
    for rep in range(reps + 1):
        var start = monotonic()
        var order = sort_indices(ranks)
        var elapsed = monotonic() - start
        if len(order) != n:
            raise Error("wrong output length")
        var seen = List[Bool](length=n, fill=False)
        for i in range(n):
            var row = order[i]
            if row < 0 or row >= n or seen[row]:
                raise Error("not a permutation")
            seen[row] = True
            if i > 0:
                var previous = order[i - 1]
                if _rank_less(ranks, row, previous):
                    raise Error("not sorted")
                if not _rank_less(ranks, previous, row) and previous > row:
                    raise Error("not stable")
        if rep > 0:
            print(elapsed)
