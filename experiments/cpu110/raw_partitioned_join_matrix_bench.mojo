"""Full-output phase bench for the isolated raw-key #105 join prototype.

The raw hash path intentionally differs from production key encoding, but it
now retains the same single-key inner-join output frame and validates it
against DataFrame.join after every measured call. Logs from the earlier
row-pair-only gather are provisional and are not comparable full-join timings.

Usage:
  raw_partitioned_join_matrix_bench int64|string ROWS REPS WORKERS CARDINALITY repeat|skew|sparse

`repeat` gives dense Int64 values, `sparse` starts at Int64.MIN so the
production bounded-range path must fall back, and `skew` sends 90% of left
rows to one key. The right side remains a unique-key dimension.
"""
from std.sys import argv
from std.time import monotonic

from dataframe import Column, DataFrame, Series
from raw_partitioned_join_lib import (
    RawPartitionedJoin,
    raw_partitioned_int64_inner,
    raw_partitioned_string_inner,
)


def left_index(row: Int, cardinality: Int, shape: String) -> Int:
    return 0 if shape == "skew" and row % 10 != 0 else row % cardinality


def int_value(index: Int, shape: String) -> Int64:
    # `sparse` intentionally includes Int64.MIN, which rejects dense direct
    # ids before subtraction and exercises the production dictionary fallback.
    return Int64.MIN + Int64(index) * 1_000_003 if shape == "sparse" else Int64(
        index
    )


def int_inputs(
    rows: Int, cardinality: Int, shape: String
) raises -> Tuple[DataFrame, DataFrame]:
    var left_keys = List[Int64](capacity=rows)
    var left_valid = List[Bool](capacity=rows)
    var left_payload = List[Int64](capacity=rows)
    var right_keys = List[Int64](capacity=cardinality)
    var right_payload = List[Int64](capacity=cardinality)
    for key in range(cardinality):
        right_keys.append(int_value(key, shape))
        right_payload.append(Int64(key))
    for row in range(rows):
        left_keys.append(int_value(left_index(row, cardinality, shape), shape))
        left_valid.append(row % 127 != 0)
        left_payload.append(Int64(row))
    return (
        DataFrame(
            [
                Series("k", Column[Int64](left_keys^, left_valid^)),
                Series("left_payload", Column[Int64](left_payload^)),
            ]
        ),
        DataFrame(
            [
                Series("k", Column[Int64](right_keys^)),
                Series("right_payload", Column[Int64](right_payload^)),
            ]
        ),
    )


def string_inputs(
    rows: Int, cardinality: Int, shape: String
) raises -> Tuple[DataFrame, DataFrame]:
    var left_keys = List[String](capacity=rows)
    var left_valid = List[Bool](capacity=rows)
    var left_payload = List[Int64](capacity=rows)
    var right_keys = List[String](capacity=cardinality)
    var right_payload = List[Int64](capacity=cardinality)
    for key in range(cardinality):
        right_keys.append("key_" + String(key))
        right_payload.append(Int64(key))
    for row in range(rows):
        left_keys.append("key_" + String(left_index(row, cardinality, shape)))
        left_valid.append(row % 127 != 0)
        left_payload.append(Int64(row))
    return (
        DataFrame(
            [
                Series("k", Column[String](left_keys^, left_valid^)),
                Series("left_payload", Column[Int64](left_payload^)),
            ]
        ),
        DataFrame(
            [
                Series("k", Column[String](right_keys^)),
                Series("right_payload", Column[Int64](right_payload^)),
            ]
        ),
    )


def _check_output(result: RawPartitionedJoin, expected: DataFrame) raises:
    if not result.frame.equals(expected):
        raise Error("raw join output differs from DataFrame.join")


def phase_total(result: RawPartitionedJoin) -> Int:
    return (
        result.partition_ns
        + result.build_probe_ns
        + result.order_ns
        + result.gather_ns
    )


def report(
    kind: String,
    shape: String,
    rows: Int,
    cardinality: Int,
    baseline_ns: Int,
    raw_wall_ns: Int,
    result: RawPartitionedJoin,
):
    # `raw_wall_total_ns` matches the production benchmark scope: it starts
    # immediately before the raw call and ends once the retained output frame
    # returns. The phase sum is diagnostic only and intentionally excludes
    # small wrapper/setup/cleanup work between phase timestamps.
    print(
        "kind,shape,rows,cardinality,output_rows,baseline_join_ns,raw_partition_ns,raw_build_probe_ns,raw_order_ns,raw_gather_ns,raw_phase_sum_ns,raw_wall_total_ns"
    )
    print(
        kind,
        ",",
        shape,
        ",",
        rows,
        ",",
        cardinality,
        ",",
        len(result.left_rows),
        ",",
        baseline_ns,
        ",",
        result.partition_ns,
        ",",
        result.build_probe_ns,
        ",",
        result.order_ns,
        ",",
        result.gather_ns,
        ",",
        phase_total(result),
        ",",
        raw_wall_ns,
        sep="",
    )


def main() raises:
    var args = argv()
    if len(args) != 7:
        raise Error(
            "usage: raw_partitioned_join_matrix_bench int64|string ROWS REPS WORKERS CARDINALITY repeat|skew|sparse"
        )
    var kind = String(args[1])
    var rows = Int(String(args[2]))
    var repetitions = Int(String(args[3]))
    var workers = Int(String(args[4]))
    var cardinality = Int(String(args[5]))
    var shape = String(args[6])
    if rows <= 0 or repetitions <= 0 or workers <= 0 or cardinality <= 0:
        raise Error(
            "rows, repetitions, workers, and cardinality must be positive"
        )
    if shape != "repeat" and shape != "skew" and shape != "sparse":
        raise Error("shape must be repeat, skew, or sparse")
    if kind == "int64":
        var inputs = int_inputs(rows, cardinality, shape)
        var warm = inputs[0].join(inputs[1], "k")
        var best_baseline = Int.MAX
        var raw_start = monotonic()
        var best_raw = raw_partitioned_int64_inner(
            inputs[0], inputs[1], 0, 0, workers
        )
        var best_raw_wall_ns = monotonic() - raw_start
        _check_output(best_raw, warm)
        for _ in range(repetitions):
            var start = monotonic()
            var joined = inputs[0].join(inputs[1], "k")
            best_baseline = min(best_baseline, monotonic() - start)
            if not joined.equals(warm):
                raise Error("baseline join output changed")
            raw_start = monotonic()
            var raw = raw_partitioned_int64_inner(
                inputs[0], inputs[1], 0, 0, workers
            )
            var raw_wall_ns = monotonic() - raw_start
            _check_output(raw, warm)
            if raw_wall_ns < best_raw_wall_ns:
                best_raw = raw^
                best_raw_wall_ns = raw_wall_ns
        report(
            kind,
            shape,
            rows,
            cardinality,
            best_baseline,
            best_raw_wall_ns,
            best_raw,
        )
        return
    if kind == "string":
        var inputs = string_inputs(rows, cardinality, shape)
        var warm = inputs[0].join(inputs[1], "k")
        var best_baseline = Int.MAX
        var raw_start = monotonic()
        var best_raw = raw_partitioned_string_inner(
            inputs[0], inputs[1], 0, 0, workers
        )
        var best_raw_wall_ns = monotonic() - raw_start
        _check_output(best_raw, warm)
        for _ in range(repetitions):
            var start = monotonic()
            var joined = inputs[0].join(inputs[1], "k")
            best_baseline = min(best_baseline, monotonic() - start)
            if not joined.equals(warm):
                raise Error("baseline join output changed")
            raw_start = monotonic()
            var raw = raw_partitioned_string_inner(
                inputs[0], inputs[1], 0, 0, workers
            )
            var raw_wall_ns = monotonic() - raw_start
            _check_output(raw, warm)
            if raw_wall_ns < best_raw_wall_ns:
                best_raw = raw^
                best_raw_wall_ns = raw_wall_ns
        report(
            kind,
            shape,
            rows,
            cardinality,
            best_baseline,
            best_raw_wall_ns,
            best_raw,
        )
        return
    raise Error("kind must be int64 or string")
