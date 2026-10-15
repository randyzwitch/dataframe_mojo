"""Device wall-time helpers compared with the independent CPU TimeZone API."""
from max.gpu import global_idx
from std.memory import Pointer
from std.testing import assert_equal
from dataframe.timezone import TimeZone
from dataframe.accelerator import timezone_resolution_tables as transport
from dataframe.accelerator.gpu import iana_resolution as resolution
from dataframe.accelerator.gpu.residency import Scope


def _evaluate(
    tables: Pointer[UInt64, ImmutAnyOrigin],
    cases: Pointer[UInt64, ImmutAnyOrigin],
    output: Pointer[UInt64, MutAnyOrigin],
    count: UInt32,
):
    var row = global_idx.x
    if row >= Int(count):
        return
    var at = 6 * row
    var base = Int(cases[unsafe_offset=at])
    var saves = Int(cases[unsafe_offset=at + 1])
    var candidates = Int(cases[unsafe_offset=at + 2])
    var candidate_count = Int(cases[unsafe_offset=at + 3])
    var local = cases[unsafe_offset=at + 4].cast[DType.int64]()
    var original = cases[unsafe_offset=at + 5].cast[DType.int64]()
    var found = resolution.resolve(
        tables, base, candidates, candidate_count, local
    )
    var shifted = resolution.relocalize(
        tables, base, saves, candidates, candidate_count, local, original
    )
    var target = 7 * row
    output[unsafe_offset=target] = UInt64(
        resolution.type_at(tables, base, original)
    )
    output[unsafe_offset=target + 1] = resolution.save_at(
        tables, base, saves, original
    ).cast[DType.uint64]()
    output[unsafe_offset=target + 2] = UInt64(found[0])
    output[unsafe_offset=target + 3] = found[1].cast[DType.uint64]()
    output[unsafe_offset=target + 4] = found[2].cast[DType.uint64]()
    output[unsafe_offset=target + 5] = shifted[0].cast[DType.uint64]()
    output[unsafe_offset=target + 6] = UInt64(shifted[1])


def _append(
    zone: TimeZone,
    base: Int,
    saves: Int,
    candidates: Int,
    local: Int64,
    original: Int64,
    mut cases: List[UInt64],
    mut expected: List[UInt64],
) raises:
    cases.extend(
        [
            UInt64(base),
            UInt64(saves),
            UInt64(candidates),
            UInt64(len(zone._candidates)),
            local.cast[DType.uint64](),
            original.cast[DType.uint64](),
        ]
    )
    var found = zone.resolve(local)
    var instant = Int64(0)
    var status = resolution.OK
    try:
        instant = zone.relocalize(local, original)
    except error:
        var message = String(error)
        if "could not localize" in message:
            status = resolution.COULD_NOT_LOCALIZE
        elif "non-existent" in message:
            status = resolution.SHIFTED_NONEXISTENT
        elif "ambiguous" in message:
            status = resolution.SHIFTED_AMBIGUOUS
        else:
            raise error^
    expected.extend(
        [
            UInt64(zone.type_at(original)),
            zone.save_at(original).cast[DType.uint64](),
            UInt64(found[0]),
            found[1].cast[DType.uint64](),
            found[2].cast[DType.uint64](),
            instant.cast[DType.uint64](),
            UInt64(status),
        ]
    )


def main() raises:
    var tables = List[UInt64]()
    var cases = List[UInt64]()
    var expected = List[UInt64]()
    var originals: List[Int64] = [
        -4000000000,
        0,
        1710054000,
        1730613600,
        16725225600,
    ]
    var deltas: List[Int64] = [-7200, -3600, -1, 0, 1, 1800, 3600, 7200]
    var transition_deltas: List[Int64] = [-3600, -1, 0, 1, 3600]
    for name in [
        "UTC",
        "+05:45",
        "America/New_York",
        "Europe/Berlin",
        "Australia/Lord_Howe",
        "Pacific/Apia",
        "Europe/Dublin",
    ]:
        var zone = TimeZone.load(name)
        var table = transport.serialize(zone)
        var base = len(tables)
        var saves = base + table.saves
        var candidates = base + table.candidates
        var candidate_count = table.candidate_count
        tables.extend(table.words.copy())
        transport.validate(tables, base, saves, candidates, candidate_count)
        for original in originals:
            for delta in deltas:
                _append(
                    zone,
                    base,
                    saves,
                    candidates,
                    original + zone.offset_at(original) + delta,
                    original,
                    cases,
                    expected,
                )
        for transition in zone._times:
            for delta in transition_deltas:
                var original = transition + delta
                _append(
                    zone,
                    base,
                    saves,
                    candidates,
                    transition + zone.offset_at(original),
                    original,
                    cases,
                    expected,
                )
    var scope = Scope()
    var context = scope.context()
    var device_tables = context.enqueue_create_buffer[DType.uint64](len(tables))
    var device_cases = context.enqueue_create_buffer[DType.uint64](len(cases))
    var device_output = context.enqueue_create_buffer[DType.uint64](
        len(expected)
    )
    var actual = List[UInt64](length=len(expected), fill=0xBAD)
    var count = len(cases) // 6
    try:
        context.enqueue_copy(device_tables, tables.unsafe_ptr())
        context.enqueue_copy(device_cases, cases.unsafe_ptr())
        context.enqueue_function[_evaluate](
            device_tables.unsafe_ptr(),
            device_cases.unsafe_ptr(),
            device_output.unsafe_ptr(),
            UInt32(count),
            grid_dim=(count + 127) // 128,
            block_dim=128,
        )
        scope.read_control_into(device_output, actual)
    except error:
        scope.synchronize()
        _ = device_tables^
        _ = device_cases^
        _ = device_output^
        _ = tables^
        _ = cases^
        _ = actual^
        raise error^
    _ = device_tables^
    _ = device_cases^
    _ = device_output^
    _ = tables^
    _ = cases^
    for index in range(len(expected)):
        assert_equal(actual[index], expected[index])
    print(
        "PASS actual GPU IANA wall-time resolution cases:",
        count,
        "exact words:",
        len(expected),
        flush=True,
    )
