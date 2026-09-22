"""Isolated raw-key hash-partitioned inner join experiment for #105.

This deliberately bypasses `_joint_key_ids`: both inputs are partitioned by
the same raw-key hash, then every bucket builds a private `Dict[key, group]`
over the right rows and probes left values directly.  Output pairs are rebuilt
from per-left match counts, restoring the public left-major/right-row order.
The count and fill passes currently rebuild each private dictionary; that
intentional experimental duplication remains included in the reported timing.
It is an experiment, not a dataframe API.
"""
from std.collections import Dict
from std.memory import ArcPointer, Pointer
from std.time import monotonic

from dataframe import Column, DataFrame, DataType, Series
from dataframe.gather import take_parallel
from dataframe.parallel import Job, run_jobs, worker_count
from dataframe.partition import Partitioner
from dataframe.string_column import StringColumn


@fieldwise_init
struct RawPartitionedJoin(Movable):
    # The retained output has the same single-key inner-join schema as
    # DataFrame.join: all left columns, then right non-key columns, with the
    # default suffix on collisions. Keeping it makes gather timing comparable
    # to the production result rather than a discarded two-frame materialization.
    var frame: DataFrame
    var left_rows: List[Int]
    var right_rows: List[Int]
    var partition_ns: Int
    var build_probe_ns: Int
    var order_ns: Int
    var gather_ns: Int


struct _Int64CountJob(Job):
    var left_key: Series
    var right_key: Series
    var left_order: ArcPointer[List[Int]]
    var right_order: ArcPointer[List[Int]]
    var left_start: Int
    var left_end: Int
    var right_start: Int
    var right_end: Int
    var counts: Int

    def __init__(
        out self,
        left_key: Series,
        right_key: Series,
        left_order: ArcPointer[List[Int]],
        right_order: ArcPointer[List[Int]],
        left_start: Int,
        left_end: Int,
        right_start: Int,
        right_end: Int,
        counts: Int,
    ):
        self.left_key = left_key.copy()
        self.right_key = right_key.copy()
        self.left_order = left_order.copy()
        self.right_order = right_order.copy()
        self.left_start = left_start
        self.left_end = left_end
        self.right_start = right_start
        self.right_end = right_end
        self.counts = counts

    def run(mut self) raises:
        ref left = self.left_key._data[Column[Int64]]
        ref right = self.right_key._data[Column[Int64]]
        ref counts = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.counts
        )[]
        var ids = Dict[Int64, Int]()
        var matches = List[List[Int]]()
        for at in range(self.right_start, self.right_end):
            var row = self.right_order[][at]
            if right._valid(row):
                var value = right._get(row)
                var group = ids.get(value, -1)
                if group < 0:
                    group = len(matches)
                    ids[value] = group
                    matches.append(List[Int]())
                matches[group].append(row)
        for at in range(self.left_start, self.left_end):
            var row = self.left_order[][at]
            if left._valid(row):
                var group = ids.get(left._get(row), -1)
                if group >= 0:
                    counts[row] = len(matches[group])


struct _Int64FillJob(Job):
    var left_key: Series
    var right_key: Series
    var left_order: ArcPointer[List[Int]]
    var right_order: ArcPointer[List[Int]]
    var left_start: Int
    var left_end: Int
    var right_start: Int
    var right_end: Int
    var offsets: Int
    var right_rows: Int

    def __init__(
        out self,
        left_key: Series,
        right_key: Series,
        left_order: ArcPointer[List[Int]],
        right_order: ArcPointer[List[Int]],
        left_start: Int,
        left_end: Int,
        right_start: Int,
        right_end: Int,
        offsets: Int,
        right_rows: Int,
    ):
        self.left_key = left_key.copy()
        self.right_key = right_key.copy()
        self.left_order = left_order.copy()
        self.right_order = right_order.copy()
        self.left_start = left_start
        self.left_end = left_end
        self.right_start = right_start
        self.right_end = right_end
        self.offsets = offsets
        self.right_rows = right_rows

    def run(mut self) raises:
        ref left = self.left_key._data[Column[Int64]]
        ref right = self.right_key._data[Column[Int64]]
        ref offsets = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.offsets
        )[]
        ref output = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.right_rows
        )[]
        var ids = Dict[Int64, Int]()
        var matches = List[List[Int]]()
        for at in range(self.right_start, self.right_end):
            var row = self.right_order[][at]
            if right._valid(row):
                var value = right._get(row)
                var group = ids.get(value, -1)
                if group < 0:
                    group = len(matches)
                    ids[value] = group
                    matches.append(List[Int]())
                matches[group].append(row)
        for at in range(self.left_start, self.left_end):
            var row = self.left_order[][at]
            if left._valid(row):
                var group = ids.get(left._get(row), -1)
                if group >= 0:
                    var out = offsets[row]
                    for right_row in matches[group]:
                        output[out] = right_row
                        out += 1


struct _StringCountJob(Job):
    var left_key: Series
    var right_key: Series
    var left_order: ArcPointer[List[Int]]
    var right_order: ArcPointer[List[Int]]
    var left_start: Int
    var left_end: Int
    var right_start: Int
    var right_end: Int
    var counts: Int

    def __init__(
        out self,
        left_key: Series,
        right_key: Series,
        left_order: ArcPointer[List[Int]],
        right_order: ArcPointer[List[Int]],
        left_start: Int,
        left_end: Int,
        right_start: Int,
        right_end: Int,
        counts: Int,
    ):
        self.left_key = left_key.copy()
        self.right_key = right_key.copy()
        self.left_order = left_order.copy()
        self.right_order = right_order.copy()
        self.left_start = left_start
        self.left_end = left_end
        self.right_start = right_start
        self.right_end = right_end
        self.counts = counts

    def run(mut self) raises:
        ref left = self.left_key._data[StringColumn]
        ref right = self.right_key._data[StringColumn]
        ref counts = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.counts
        )[]
        var ids = Dict[StringSlice[ImmutAnyOrigin], Int]()
        var matches = List[List[Int]]()
        for at in range(self.right_start, self.right_end):
            var row = self.right_order[][at]
            if right._valid(row):
                var value = right._get(row)
                var group = ids.get(value, -1)
                if group < 0:
                    group = len(matches)
                    ids[value] = group
                    matches.append(List[Int]())
                matches[group].append(row)
        for at in range(self.left_start, self.left_end):
            var row = self.left_order[][at]
            if left._valid(row):
                var group = ids.get(left._get(row), -1)
                if group >= 0:
                    counts[row] = len(matches[group])


struct _StringFillJob(Job):
    var left_key: Series
    var right_key: Series
    var left_order: ArcPointer[List[Int]]
    var right_order: ArcPointer[List[Int]]
    var left_start: Int
    var left_end: Int
    var right_start: Int
    var right_end: Int
    var offsets: Int
    var right_rows: Int

    def __init__(
        out self,
        left_key: Series,
        right_key: Series,
        left_order: ArcPointer[List[Int]],
        right_order: ArcPointer[List[Int]],
        left_start: Int,
        left_end: Int,
        right_start: Int,
        right_end: Int,
        offsets: Int,
        right_rows: Int,
    ):
        self.left_key = left_key.copy()
        self.right_key = right_key.copy()
        self.left_order = left_order.copy()
        self.right_order = right_order.copy()
        self.left_start = left_start
        self.left_end = left_end
        self.right_start = right_start
        self.right_end = right_end
        self.offsets = offsets
        self.right_rows = right_rows

    def run(mut self) raises:
        ref left = self.left_key._data[StringColumn]
        ref right = self.right_key._data[StringColumn]
        ref offsets = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.offsets
        )[]
        ref output = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.right_rows
        )[]
        var ids = Dict[StringSlice[ImmutAnyOrigin], Int]()
        var matches = List[List[Int]]()
        for at in range(self.right_start, self.right_end):
            var row = self.right_order[][at]
            if right._valid(row):
                var value = right._get(row)
                var group = ids.get(value, -1)
                if group < 0:
                    group = len(matches)
                    ids[value] = group
                    matches.append(List[Int]())
                matches[group].append(row)
        for at in range(self.left_start, self.left_end):
            var row = self.left_order[][at]
            if left._valid(row):
                var group = ids.get(left._get(row), -1)
                if group >= 0:
                    var out = offsets[row]
                    for right_row in matches[group]:
                        output[out] = right_row
                        out += 1


def _partitions(
    left: DataFrame,
    right: DataFrame,
    left_key: Series,
    right_key: Series,
    workers: Int,
) raises -> Tuple[
    ArcPointer[List[Int]], ArcPointer[List[Int]], List[Int], List[Int]
]:
    var left_keys = List[Series]([left_key.copy()])
    var right_keys = List[Series]([right_key.copy()])
    var left_partitioner = Partitioner(left_keys, workers)
    var left_parts = left_partitioner.scatter(workers)
    var right_partitioner = Partitioner(right_keys, workers)
    var right_parts = right_partitioner.scatter(workers)
    if left_parts.buckets() != right_parts.buckets():
        raise Error("raw partitioners chose incompatible bucket counts")
    return (
        ArcPointer(left_parts.order.copy()),
        ArcPointer(right_parts.order.copy()),
        left_parts.bounds.copy(),
        right_parts.bounds.copy(),
    )


def _finish(
    left: DataFrame,
    right: DataFrame,
    right_column: Int,
    suffix: String,
    counts: List[Int],
    partition_ns: Int,
    build_probe_ns: Int,
    order_start: Int,
    mut fill_int: List[_Int64FillJob],
    mut fill_string: List[_StringFillJob],
) raises -> RawPartitionedJoin:
    var offsets = List[Int](length=len(counts) + 1, fill=0)
    for row in range(len(counts)):
        if counts[row] > Int.MAX - offsets[row]:
            raise Error("Join output row count overflows")
        offsets[row + 1] = offsets[row] + counts[row]
    var right_rows = List[Int](length=offsets[len(counts)], fill=0)
    if len(fill_int) > 0:
        for i in range(len(fill_int)):
            fill_int[i].offsets = Int(Pointer(to=offsets))
            fill_int[i].right_rows = Int(Pointer(to=right_rows))
        run_jobs(fill_int)
    if len(fill_string) > 0:
        for i in range(len(fill_string)):
            fill_string[i].offsets = Int(Pointer(to=offsets))
            fill_string[i].right_rows = Int(Pointer(to=right_rows))
        run_jobs(fill_string)
    var left_rows = List[Int](length=len(right_rows), fill=0)
    for row in range(len(counts)):
        for at in range(offsets[row], offsets[row + 1]):
            left_rows[at] = row
    var order_ns = monotonic() - order_start

    # Match DataFrame.join's single-key inner output: all left columns, then
    # right non-key columns.  Gather every returned column through the same
    # parallel gatherer production uses, and retain the assembled frame.
    var gather_start = monotonic()
    var workers = worker_count(len(left_rows))
    var left_sources = List[Series](capacity=left.width())
    var names = Dict[String, Bool]()
    for column in left._columns:
        left_sources.append(column.copy())
        names[column.name()] = True
    var columns = take_parallel(
        left_sources, left_rows.copy(), workers, or_null=True
    )
    var right_sources = List[Series]()
    var right_names = List[String]()
    for c in range(right.width()):
        if c == right_column:
            continue
        var name = right._columns[c].name()
        if name in names:
            name += suffix
        if name in names:
            raise Error("Join output name collision: " + name)
        names[name] = True
        right_sources.append(right._columns[c].copy())
        right_names.append(name)
    var from_right = take_parallel(
        right_sources, right_rows.copy(), workers, or_null=True
    )
    for c in range(len(from_right)):
        columns.append(from_right[c].renamed(right_names[c]))
    var frame = DataFrame(columns^, height=len(left_rows))
    return RawPartitionedJoin(
        frame^,
        left_rows^,
        right_rows^,
        partition_ns,
        build_probe_ns,
        order_ns,
        monotonic() - gather_start,
    )


def raw_partitioned_int64_inner(
    left: DataFrame,
    right: DataFrame,
    left_column: Int,
    right_column: Int,
    workers: Int,
    suffix: String = "_right",
) raises -> RawPartitionedJoin:
    ref left_key = left._columns[left_column]
    ref right_key = right._columns[right_column]
    if (
        left_key.dtype().physical() != DataType.INT64
        or right_key.dtype().physical() != DataType.INT64
    ):
        raise Error("raw Int64 prototype requires physical Int64 keys")
    var partition_start = monotonic()
    var parts = _partitions(left, right, left_key, right_key, workers)
    var left_order = parts[0].copy()
    var right_order = parts[1].copy()
    var left_bounds = parts[2].copy()
    var right_bounds = parts[3].copy()
    var partition_ns = monotonic() - partition_start
    var count_start = monotonic()
    var counts = List[Int](length=left.height(), fill=0)
    var count_jobs = List[_Int64CountJob](capacity=len(left_bounds) - 1)
    var fill_jobs = List[_Int64FillJob](capacity=len(left_bounds) - 1)
    for bucket in range(len(left_bounds) - 1):
        count_jobs.append(
            _Int64CountJob(
                left_key,
                right_key,
                left_order,
                right_order,
                left_bounds[bucket],
                left_bounds[bucket + 1],
                right_bounds[bucket],
                right_bounds[bucket + 1],
                Int(Pointer(to=counts)),
            )
        )
        fill_jobs.append(
            _Int64FillJob(
                left_key,
                right_key,
                left_order,
                right_order,
                left_bounds[bucket],
                left_bounds[bucket + 1],
                right_bounds[bucket],
                right_bounds[bucket + 1],
                0,
                0,
            )
        )
    run_jobs(count_jobs)
    var build_probe_ns = monotonic() - count_start
    var no_string_fills = List[_StringFillJob]()
    return _finish(
        left,
        right,
        right_column,
        suffix,
        counts,
        partition_ns,
        build_probe_ns,
        monotonic(),
        fill_jobs,
        no_string_fills,
    )


def raw_partitioned_string_inner(
    left: DataFrame,
    right: DataFrame,
    left_column: Int,
    right_column: Int,
    workers: Int,
    suffix: String = "_right",
) raises -> RawPartitionedJoin:
    ref left_key = left._columns[left_column]
    ref right_key = right._columns[right_column]
    if (
        left_key.dtype().physical() != DataType.STRING
        or right_key.dtype().physical() != DataType.STRING
    ):
        raise Error("raw String prototype requires string keys")
    var partition_start = monotonic()
    var parts = _partitions(left, right, left_key, right_key, workers)
    var left_order = parts[0].copy()
    var right_order = parts[1].copy()
    var left_bounds = parts[2].copy()
    var right_bounds = parts[3].copy()
    var partition_ns = monotonic() - partition_start
    var count_start = monotonic()
    var counts = List[Int](length=left.height(), fill=0)
    var count_jobs = List[_StringCountJob](capacity=len(left_bounds) - 1)
    var fill_jobs = List[_StringFillJob](capacity=len(left_bounds) - 1)
    for bucket in range(len(left_bounds) - 1):
        count_jobs.append(
            _StringCountJob(
                left_key,
                right_key,
                left_order,
                right_order,
                left_bounds[bucket],
                left_bounds[bucket + 1],
                right_bounds[bucket],
                right_bounds[bucket + 1],
                Int(Pointer(to=counts)),
            )
        )
        fill_jobs.append(
            _StringFillJob(
                left_key,
                right_key,
                left_order,
                right_order,
                left_bounds[bucket],
                left_bounds[bucket + 1],
                right_bounds[bucket],
                right_bounds[bucket + 1],
                0,
                0,
            )
        )
    run_jobs(count_jobs)
    var build_probe_ns = monotonic() - count_start
    var no_int_fills = List[_Int64FillJob]()
    return _finish(
        left,
        right,
        right_column,
        suffix,
        counts,
        partition_ns,
        build_probe_ns,
        monotonic(),
        no_int_fills,
        fill_jobs,
    )
