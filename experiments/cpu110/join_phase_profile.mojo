"""Phase profile of the production inner-join path after CSV loading.

Usage: join_phase_profile DATA_DIR ROWS REPETITIONS
The four reported phases are the exact helpers used by DataFrame.join for an
inner join on `jk`; CSV reads are setup and excluded.
"""
from std.sys import argv
from std.time import monotonic

from dataframe import CsvField, CsvSchema, DataFrame, Series, read_csv
from dataframe.frame import _group_index, _group_rows, _joint_key_ids
from dataframe.gather import take_parallel
from dataframe.parallel import worker_count


def left_schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("key_low"),
            CsvField.int64("key_high"),
            CsvField.int64("key_skew"),
            CsvField.string("key_str"),
            CsvField.int64("jk"),
            CsvField.float64("x"),
            CsvField.float64("y"),
            CsvField.int64("n"),
        ]
    )


def right_schema() raises -> CsvSchema:
    return CsvSchema([CsvField.int64("jk"), CsvField.float64("r")])


@fieldwise_init
struct Phases(Copyable):
    var encode: Int
    var csr: Int
    var expand: Int
    var left_gather: Int
    var right_gather: Int
    var rows: Int


def _profile(left: DataFrame, right: DataFrame) raises -> Phases:
    var start = monotonic()
    var ids = _joint_key_ids(left, right, [4], [0])
    var encode = monotonic() - start
    var left_ids = ids[0].copy()
    var right_ids = ids[1].copy()

    start = monotonic()
    var starts = _group_index(right_ids, ids[2])
    var flat = _group_rows(right_ids, starts)
    var csr = monotonic() - start

    start = monotonic()
    var left_rows = List[Int]()
    var right_rows = List[Int]()
    for row in range(len(left_ids)):
        var key = left_ids[row]
        if key >= 0:
            for offset in range(starts[key], starts[key + 1]):
                left_rows.append(row)
                right_rows.append(flat[offset])
    var expand = monotonic() - start

    var workers = worker_count(len(left_rows))
    var left_sources = List[Series](capacity=left.width())
    for column in left._columns:
        left_sources.append(column.copy())
    start = monotonic()
    var left_out = take_parallel(
        left_sources^, left_rows.copy(), workers, or_null=True
    )
    var left_gather = monotonic() - start

    var right_sources = List[Series]()
    right_sources.append(right._columns[1].copy())
    start = monotonic()
    var right_out = take_parallel(
        right_sources^, right_rows.copy(), workers, or_null=True
    )
    var right_gather = monotonic() - start
    if len(left_out) != left.width() or len(right_out) != 1:
        raise Error("join phase gather returned wrong schema")
    return Phases(
        encode,
        csr,
        expand,
        left_gather,
        right_gather,
        len(left_rows),
    )


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error("usage: join_phase_profile DATA_DIR ROWS REPETITIONS")
    var directory = String(args[1])
    var rows = String(args[2])
    var repetitions = Int(String(args[3]))
    var left = read_csv(directory + "/left_" + rows + ".csv", left_schema())
    var right = read_csv(directory + "/right_" + rows + ".csv", right_schema())
    var warm = _profile(left, right)
    var best = Phases(Int.MAX, Int.MAX, Int.MAX, Int.MAX, Int.MAX, 0)
    for _ in range(repetitions):
        var phase = _profile(left, right)
        best.encode = min(best.encode, phase.encode)
        best.csr = min(best.csr, phase.csr)
        best.expand = min(best.expand, phase.expand)
        best.left_gather = min(best.left_gather, phase.left_gather)
        best.right_gather = min(best.right_gather, phase.right_gather)
        best.rows = phase.rows
    if best.rows != warm.rows:
        raise Error("join output row count changed")
    print("phase,best_ns,rows")
    print("joint_key_encode,", best.encode, ",", best.rows, sep="")
    print("right_csr,", best.csr, ",", best.rows, sep="")
    print("ordered_match_expand,", best.expand, ",", best.rows, sep="")
    print("left_gather_8_columns,", best.left_gather, ",", best.rows, sep="")
    print("right_gather_1_column,", best.right_gather, ",", best.rows, sep="")
