"""Paired join/gather cutoff sweeps; construct inputs and validate untimed.

Build with `mojo build -I . benchmarks/bench_join_cutoffs.mojo -o ...`.
Args: CASE ROWS SPACING PROBE_PERCENT WIDTH ALGORITHM REPS
Cases: csr, progression, strided, bounded, membership, ids, gather,
join, ordered_join. Public joins take inner/left/full/semi/anti as ALGORITHM;
other comparisons use serial/parallel or bounded/hash labels.
For csr, SPACING is rows per distinct key and WIDTH=2 selects ordered IDs.
For gathers WIDTH is the column count; for other cases WIDTH>1 splits the
inputs into that many chunks. Join SPACING controls key density, with an
irregular offset preventing GCD compression except in the progression,
strided and ordered_join cases. The driver builds forced-path copies for
calibration; the production API gains no algorithm-selection arguments.
"""
from std.sys import argv
from std.time import monotonic
from std.memory import ArcPointer
from dataframe import Column, DataFrame, Series
from dataframe.frame import (
    _bounded_int64_join_rows,
    _dense_right_int64_rows,
    _group_index,
    _group_rows,
    _joint_key_ids,
    _parallel_group_rows,
    _range_int64_membership_rows,
)
from dataframe.gather import (
    _SortedChunkTakeJob,
    _take_sorted_chunked_partitioned,
)
from dataframe.join_hash import (
    direct_hash_join_rows,
    direct_hash_semi_anti_rows,
)
from dataframe.parallel import configured_workers, run_jobs, worker_count


def shuffle(mut values: List[Int64]):
    var state = UInt64(91723)
    for i in range(len(values) - 1, 0, -1):
        state = state * 6364136223846793005 + 1442695040888963407
        var j = Int((state >> 16) % UInt64(i + 1))
        var temp = values[i]
        values[i] = values[j]
        values[j] = temp


def gather_columns(
    columns: List[Series], indices: List[Int], parallel: Bool
) raises -> List[Series]:
    if parallel:
        return _take_sorted_chunked_partitioned(
            columns,
            indices.copy(),
            min(4, max(1, configured_workers() // len(columns))),
        )
    var shared = ArcPointer(indices.copy())
    var jobs = List[_SortedChunkTakeJob]()
    for column in columns:
        jobs.append(_SortedChunkTakeJob(column, shared, False))
    run_jobs(jobs)
    var result = List[Series]()
    while len(jobs) > 0:
        result.append(jobs.pop(0).into_result())
    return result^


def main() raises:
    var args = argv()
    if len(args) != 8:
        raise Error("CASE ROWS SPACING PROBE_PERCENT WIDTH ALGORITHM REPS")
    var kind = String(args[1])
    var n = Int(String(args[2]))
    var spacing = Int(String(args[3]))
    var probe = n * Int(String(args[4])) // 100
    var width = Int(String(args[5]))
    var algorithm = String(args[6])
    var reps = Int(String(args[7]))
    if n <= 0 or spacing <= 0 or probe <= 0 or width <= 0 or reps <= 0:
        raise Error("sizes, spacing, width and repetitions must be positive")
    var samples = List[Int]()
    if kind == "csr":
        var keys = List[Int64](capacity=n)
        var count = max(1, n // spacing)
        for i in range(n):
            keys.append(
                Int64(i % count if width == 1 else min(i // spacing, count - 1))
            )
        if width == 1:
            shuffle(keys)
        var ids = List[Int](capacity=n)
        for key in keys:
            ids.append(Int(key))
        var starts = _group_index(ids, count)
        var reference = _group_rows(ids, starts)
        for rep in range(reps + 1):
            var start = monotonic()
            var result = _parallel_group_rows(
                ids, starts, worker_count(n)
            ) if algorithm == "parallel" else _group_rows(ids, starts)
            var elapsed = monotonic() - start
            if len(result) != n:
                raise Error("CSR length")
            if rep == 0:
                for i in range(n):
                    if result[i] != reference[i]:
                        raise Error("CSR order")
            else:
                samples.append(elapsed)
    elif kind == "gather":
        var columns = List[Series]()
        for c in range(width):
            var chunks = List[Series]()
            for chunk in range(32):
                var vals = List[Int64]()
                for i in range(2 * n * chunk // 32, 2 * n * (chunk + 1) // 32):
                    vals.append(Int64(i + c))
                chunks.append(Series(String(c), Column[Int64](vals^)))
            columns.append(Series._from_chunks(chunks^))
        var indices = List[Int](capacity=n)
        for i in range(n):
            indices.append(i * 2)
        for rep in range(reps + 1):
            var start = monotonic()
            var result = gather_columns(
                columns, indices, algorithm == "parallel"
            )
            var elapsed = monotonic() - start
            for c in range(width):
                if len(result[c]) != n or result[c].get(n - 1).int64() != Int64(
                    2 * (n - 1) + c
                ):
                    raise Error("gather values")
            if rep > 0:
                samples.append(elapsed)
    else:
        var build = List[Int64](capacity=n)
        for i in range(n):
            build.append(
                Int64(
                    i * spacing
                    + (
                        1 if spacing > 1
                        and i % 3 == 0
                        and kind != "progression"
                        and kind != "strided"
                        and kind != "ordered_join" else 0
                    )
                )
            )
        if kind != "progression" and kind != "ordered_join":
            shuffle(build)
        var probes = List[Int64](capacity=probe)
        for i in range(probe):
            probes.append(build[n - 1 - i % n])
        var left = Series("k", Column[Int64](probes^))
        var right = Series("k", Column[Int64](build^))
        if width > 1:
            var left_chunks = List[Series]()
            var right_chunks = List[Series]()
            for chunk in range(width):
                var first = probe * chunk // width
                var end = probe * (chunk + 1) // width
                left_chunks.append(left.slice(first, end - first))
                first = n * chunk // width
                end = n * (chunk + 1) // width
                right_chunks.append(right.slice(first, end - first))
            left = Series._from_chunks(left_chunks^)
            right = Series._from_chunks(right_chunks^)
        var left_frame = DataFrame([left.copy()])
        var right_frame = DataFrame([right.copy()])
        if kind == "join" or kind == "ordered_join":
            var lvalues = List[Int64](capacity=probe)
            var rvalues = List[Int64](capacity=n)
            for i in range(probe):
                lvalues.append(Int64(i))
            for i in range(n):
                rvalues.append(Int64(i))
            left_frame = left_frame.with_column(
                Series("l", Column[Int64](lvalues^))
            )
            right_frame = right_frame.with_column(
                Series("r", Column[Int64](rvalues^))
            )
        for rep in range(reps + 1):
            var start = monotonic()
            var out_left: List[Int]
            var out_right = List[Int]()
            if kind == "join" or kind == "ordered_join":
                var result = left_frame.join(right_frame, "k", algorithm)
                var elapsed = monotonic() - start
                var expected = max(probe, n) if algorithm == "full" else probe
                if algorithm == "anti":
                    expected = 0
                if result.height() != expected:
                    raise Error("public join height")
                if rep > 0:
                    samples.append(elapsed)
                continue
            elif kind == "progression":
                var result = _dense_right_int64_rows(left, right)
                out_left = result[1].copy()
                out_right = result[2].copy()
            elif kind == "bounded" or kind == "strided":
                if algorithm == "hash":
                    var result = direct_hash_join_rows(
                        [left.copy()], [right.copy()], False
                    )
                    out_left = result[0].copy()
                    out_right = result[1].copy()
                else:
                    var result = _bounded_int64_join_rows(left, right, False)
                    if not result[0]:
                        raise Error("bounded path rejected fixture")
                    out_left = result[1].copy()
                    out_right = result[2].copy()
            elif kind == "membership":
                if algorithm == "hash":
                    out_left = direct_hash_semi_anti_rows(
                        [left.copy()], [right.copy()], True
                    )
                else:
                    var result = _range_int64_membership_rows(left, right, True)
                    if not result[0]:
                        raise Error("membership path rejected fixture")
                    out_left = result[1].copy()
            elif kind == "ids":
                var result = _joint_key_ids(left_frame, right_frame, [0], [0])
                var starts = _group_index(result[1], result[2])
                out_right = _group_rows(result[1], starts)
                out_left = result[0].copy()
            else:
                raise Error("unknown kind")
            var elapsed = monotonic() - start
            if len(out_left) != probe:
                raise Error("join output length")
            if kind == "bounded" or kind == "strided" or kind == "progression":
                if len(out_right) != probe:
                    raise Error("join pair length")
                if rep == 0:
                    for i in range(probe):
                        if (
                            out_left[i] != i
                            or left.get(i).int64()
                            != right.get(out_right[i]).int64()
                        ):
                            raise Error("join pair/order")
            if rep > 0:
                samples.append(elapsed)
    for sample in samples:
        print(
            kind,
            n,
            spacing,
            probe,
            width,
            algorithm,
            configured_workers(),
            sample,
            sep=",",
        )
