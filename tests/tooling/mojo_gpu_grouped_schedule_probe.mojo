"""CPU/literal whole/bucket/heavy ordering and resident List assembly proof."""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import Pointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe.parallel import configured_workers
from dataframe.partition import low_cardinality, _mix
from dataframe.accelerator.gpu.residency import Scope, Table
from dataframe.accelerator.gpu.runtime import (
    _preflight,
    execute_plan,
    materialize,
)
from dataframe.accelerator.gpu.grouped_schedule import (
    build,
    read_phases,
    gather_phase,
)
from dataframe.accelerator.gpu.list_groups import build_groups
from dataframe.accelerator.gpu.grouped_list_targets import (
    build_destinations,
    create_target,
    append_phase,
    finish_target,
)
from dataframe.accelerator.gpu.device_lists import consume
from dataframe.accelerator.gpu.grouped_phase_groups import (
    build_phase_groups,
    PhaseGroups,
)
from tests.support.lifecycle_jobs import WorkerSetting


def _stage(frame: DataFrame, scope: Scope) raises -> Table:
    var expressions = List[Expr]()
    for field in frame._columns:
        expressions.append(col(field.name()))
    var plan = _preflight(frame.lazy().select_exprs(expressions))
    var result = execute_plan(plan, scope, export_values=False)
    return result[0].copy()


def _fault_frame(kind: Int) raises -> DataFrame:
    var n = 262145 if kind == 2 else 65537
    var hot = 2 * n // 5
    var values = List[Int64]()
    for row in range(n):
        var key = Int64(0) if kind == 0 else Int64(row)
        if kind == 2:
            key = (
                Int64(4) if row
                < hot else Int64(0) if row
                == hot else Int64(1000000 + row)
            )
        values.append(key)
    var integers = List[Int64](length=n, fill=0)
    var exponents = List[Int64](length=n, fill=0)
    integers[hot if kind == 2 else 4] = 128
    exponents[0] = -1
    return DataFrame(
        [
            Series("k", Column[Int64](values^)),
            Series("u", Column[Int64](integers^)),
            Series("base", Column[Int64](List[Int64](length=n, fill=2))),
            Series("exp", Column[Int64](exponents^)),
        ]
    )


def _run_faults(frame: DataFrame, scope: Scope, kind: Int) raises:
    var source = _stage(frame, scope)
    var schedule = build(source, ["k"], scope, configured_workers())
    assert_equal(schedule[].facts[0], UInt32(kind == 0))
    assert_equal(schedule[].facts[2], UInt32(16 if kind == 2 else 8))
    assert_equal(schedule[].facts[3], UInt32(1 if kind == 2 else 0))
    var phases = read_phases(schedule, scope)
    if kind == 2:
        assert_equal(phases.order[0], UInt32(11))
        assert_equal(phases.starts[12] - phases.starts[11], UInt32(114707))
        assert_equal(phases.starts[1] - phases.starts[0], UInt32(9904))
    elif kind == 1:
        assert_equal(phases.order[0], UInt32(0))
    var expressions = List[Expr](
        [
            col("u").cast(DataType.INT8).alias("first"),
            col("base").pow(col("exp")).alias("later"),
        ]
    )
    # These are actual direct Mojo row-program producers inside the CPU's
    # selected bucket phase, not a host simulation or error-sort stand-in.
    for phase in range(Int(schedule[].facts[4])):
        var bucket = gather_phase(source, schedule, phases, phase, scope)
        for expression in expressions:
            var plan = _preflight(
                bucket.schema.lazy().select_exprs([expression.copy()]),
                source_rows=bucket.rows,
            )
            var output = execute_plan(
                plan, scope, Optional(bucket.copy()), export_values=False
            )
            output[0].observe_errors()


def _cpu_gpu_fault_priority() raises:
    # Independent literal bucket selectors complement the actual CPU oracle.
    assert_equal(_mix(UInt64(0)), UInt64(0))
    assert_equal(_mix(UInt64(4)), UInt64(0xB7A4712C74562914))
    for kind in range(3):
        var setting = WorkerSetting(8 if kind == 2 else 4)
        assert_equal(configured_workers(), 8 if kind == 2 else 4)
        var frame = _fault_frame(kind)
        assert_equal(low_cardinality([frame.column("k")]), kind == 0)
        var query = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("u").cast(DataType.INT8).implode().alias("first"),
                    col("base").pow(col("exp")).implode().alias("later"),
                ]
            )
        )
        with assert_raises(
            contains="cast from int64 to int8 failed" if kind
            == 0 else "nonnegative exponent"
        ):
            _ = query.collect(engine="cpu", optimize=False, streaming=False)
        var scope = Scope()
        with assert_raises(
            contains="strict cast failed" if kind
            == 0 else "nonnegative exponent"
        ):
            _run_faults(frame, scope, kind)
        # Failed producers do not poison the query context or next region.
        var recovered = _stage(
            DataFrame([Series("x", Column[Int64]([7]))]), scope
        )
        var recovered_host = materialize(recovered)
        assert_equal(recovered_host.column("x").int64().value(0), Int64(7))
        _ = setting^


def _check_group_geometry(
    index: Pointer[UInt32, ImmutAnyOrigin],
    ids: Pointer[UInt32, ImmutAnyOrigin],
    ordinals: Pointer[UInt32, ImmutAnyOrigin],
    sizes: Pointer[UInt32, ImmutAnyOrigin],
    order: Pointer[UInt32, ImmutAnyOrigin],
    failure: Pointer[UInt32, MutAnyOrigin],
    begin: UInt32,
    rows: UInt32,
    active: UInt32,
    total: UInt32,
):
    if global_idx.x != 0:
        return
    var bad = UInt32(0)
    for row in range(Int(rows)):
        var original = index[unsafe_offset=Int(begin) + row]
        if ids[unsafe_offset=row] != original % UInt32(257) or ordinals[
            unsafe_offset=row
        ] != original // UInt32(257):
            bad = 1
    var counted = UInt32(0)
    var at = 0
    for group in range(257):
        var size = sizes[unsafe_offset=group]
        if size:
            if size != (total - 1 - UInt32(group)) // UInt32(257) + 1:
                bad = 1
            if at >= Int(active) or order[unsafe_offset=at] != UInt32(group):
                bad = 1
            at += 1
            counted += size
    if at != Int(active) or counted != rows:
        bad = 1
    failure[unsafe_offset=0] = bad


def _verify_geometry(
    index: DeviceBuffer[DType.uint32],
    geometry: PhaseGroups,
    begin: UInt32,
    scope: Scope,
    total: Int,
) raises:
    _ = scope.check(8)
    var failure = scope.context().enqueue_create_buffer[DType.uint32](1)
    var held_failure = failure.copy()
    var held_index = index.copy()
    var held_ids = geometry.row_to_group.copy()
    var held_ordinals = geometry.group_ordinal.copy()
    var held_sizes = geometry.group_sizes.copy()
    var held_order = geometry.active_order.copy()
    try:
        scope.context().enqueue_function[_check_group_geometry](
            index.unsafe_ptr(),
            geometry.row_to_group.unsafe_ptr(),
            geometry.group_ordinal.unsafe_ptr(),
            geometry.group_sizes.unsafe_ptr(),
            geometry.active_order.unsafe_ptr(),
            failure.unsafe_ptr(),
            begin,
            UInt32(geometry.rows),
            UInt32(geometry.active_count),
            UInt32(total),
            grid_dim=1,
            block_dim=1,
        )
        scope.record_launches()
        var proof = scope.read_control(failure)
        scope.record_download(4)
        assert_equal(proof[0], UInt32(0))
        _ = held_failure^
        _ = held_index^
        _ = held_ids^
        _ = held_ordinals^
        _ = held_sizes^
        _ = held_order^
    except error:
        scope.synchronize()
        _ = held_failure^
        _ = held_index^
        _ = held_ids^
        _ = held_ordinals^
        _ = held_sizes^
        _ = held_order^
        raise error^


def _schedule_budget() raises:
    var keys = List[Int64]()
    for row in range(257):
        keys.append(Int64(row))
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("x", Column[Int64](List[Int64](length=257, fill=7))),
        ]
    )
    var scope = Scope()
    var source = _stage(frame, scope)
    var schedule = build(source, ["k"], scope, 4)
    var phases = read_phases(schedule, scope)
    var peak = scope.info()[1]
    assert_true(peak > 0)
    for limit in List[Int]([peak - 1, peak]):
        var limited = Scope(budget=limit)
        var input = _stage(frame, limited)
        var before = limited.transfers()
        if limit < peak:
            with assert_raises(contains="memory budget before submission"):
                _ = build(input, ["k"], limited, 4)
            assert_equal(limited.transfers(), before)
        else:
            var exact = build(input, ["k"], limited, 4)
            _ = read_phases(exact, limited)
            assert_equal(limited.info()[1], peak)
    var invalid = Scope(999999)
    with assert_raises(contains="another query context"):
        _ = build(source, ["k"], invalid, 4)
    assert_false(invalid.initialized())
    var before = scope.transfers()
    with assert_raises(contains="worker count"):
        _ = build(source, ["k"], scope, 0)
    assert_equal(scope.transfers(), before)


def _resident_bucket_append() raises:
    var setting = WorkerSetting(4)
    assert_equal(configured_workers(), 4)
    var n = 65537
    var keys = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(n):
        keys.append(Int64(row % 257))
        values.append(Int64(row))
        valid.append(row % 11 != 0)
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("x", Column[Int64](values^, valid^)),
        ]
    )
    var scope = Scope()
    var source = _stage(frame, scope)
    var schedule = build(source, ["k"], scope, configured_workers())
    assert_false(Bool(schedule[].facts[0]))
    assert_equal(schedule[].facts[3], UInt32(0))
    var phases = read_phases(schedule, scope)
    var layout = build_groups(source, ["k"], scope)
    assert_true(Bool(layout))
    var destinations = build_destinations(layout.value(), scope)
    var target = create_target(
        destinations, schedule, frame.column("x").clear(), scope, "xs"
    )
    var target2 = create_target(
        destinations, schedule, frame.column("x").clear(), scope, "ys"
    )
    var incomplete = create_target(
        destinations, schedule, frame.column("x").clear(), scope, "incomplete"
    )
    with assert_raises(contains="incomplete children"):
        _ = finish_target(incomplete^, scope)
    var before = scope.transfers()
    for phase in range(Int(schedule[].facts[4])):
        var bucket = gather_phase(source, schedule, phases, phase, scope)
        var geometry = build_phase_groups(
            schedule, destinations, phases, phase, scope
        )
        _verify_geometry(
            schedule[].index,
            geometry,
            phases.starts[Int(phases.order[phase])],
            scope,
            n,
        )
        var plan = _preflight(
            bucket.schema.lazy().select_exprs([col("x")]),
            source_rows=bucket.rows,
        )
        var output = execute_plan(
            plan, scope, Optional(bucket.copy()), export_values=False
        )
        append_phase(target, output[0], phases, phase, scope)
        append_phase(target2, output[0], phases, phase, scope)
    var after = scope.transfers()
    # Only 4-byte deferred-error observations per produced child can download
    # during appending; descriptor offsets and child payloads stay on-device.
    assert_equal(after[1] - before[1], 8 * n + 8 * Int(schedule[].facts[4]))
    var lists = finish_target(target^, scope)
    var lists2 = finish_target(target2^, scope)
    assert_true(lists.descriptors.ptr() == lists2.descriptors.ptr())
    var lengths = materialize(consume[True](lists, scope, "lengths"))
    var last = materialize(consume[False](lists, scope, "last", -1))
    var lengths_column = lengths.column("lengths").int64()
    var last_column = last.column("last").int64()
    assert_equal(lengths.height(), 257)
    for group in range(257):
        var expected_count = (n - 1 - group) // 257 + 1
        var expected_last = group + (expected_count - 1) * 257
        assert_equal(lengths_column.value(group), Int64(expected_count))
        assert_equal(last_column.is_valid(group), expected_last % 11 != 0)
        if last_column.is_valid(group):
            assert_equal(last_column.value(group), Int64(expected_last))
    # Actual CPU grouping agrees with the independent source-order formulas.
    var cpu = frame.group_by("k", maintain_order=True).agg(
        [col("x").implode().alias("xs")]
    )
    var expected = (
        cpu.lazy()
        .select_exprs(
            [
                col("xs").list().len().alias("lengths"),
                col("xs").list().value(-1).alias("last"),
            ]
        )
        .collect(engine="cpu", optimize=False, streaming=False)
    )
    for group in range(257):
        assert_equal(
            expected.column("lengths").int64().value(group),
            lengths_column.value(group),
        )
        assert_equal(
            expected.column("last").int64().is_valid(group),
            last_column.is_valid(group),
        )
        if last_column.is_valid(group):
            assert_equal(
                expected.column("last").int64().value(group),
                last_column.value(group),
            )

    _ = setting^


def main() raises:
    _cpu_gpu_fault_priority()
    _resident_bucket_append()
    _schedule_budget()
    print(
        "PASS actual whole/bucket/heavy GPU producer priority and resident canonical List append"
    )
