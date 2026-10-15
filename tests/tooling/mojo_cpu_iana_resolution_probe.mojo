"""Independent CPU TimeZone and literal DST anchors before device admission."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe.timezone import TimeZone
from dataframe.accelerator import timezone_tables
from dataframe.accelerator import timezone_resolution_tables as transport
from dataframe.accelerator.gpu import iana_resolution as device


def _check(
    zone: TimeZone,
    ref table: transport.ResolutionTable,
    local: Int64,
    original: Int64,
) raises:
    var pointer = (
        table.words.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    assert_equal(device.type_at(pointer, 0, original), zone.type_at(original))
    assert_equal(
        device.save_at(pointer, 0, table.saves, original),
        zone.save_at(original),
    )
    var expected = zone.resolve(local)
    var actual = device.resolve(
        pointer, 0, table.candidates, table.candidate_count, local
    )
    assert_equal(actual[0], expected[0])
    assert_equal(actual[1], expected[1])
    assert_equal(actual[2], expected[2])
    var expected_status = device.OK
    var expected_instant = Int64(0)
    try:
        expected_instant = zone.relocalize(local, original)
    except error:
        var diagnostic = String(error)
        if "could not localize" in diagnostic:
            expected_status = device.COULD_NOT_LOCALIZE
        elif "non-existent" in diagnostic:
            expected_status = device.SHIFTED_NONEXISTENT
        elif "ambiguous" in diagnostic:
            expected_status = device.SHIFTED_AMBIGUOUS
        else:
            raise error^
    var relocated = device.relocalize(
        pointer,
        0,
        table.saves,
        table.candidates,
        table.candidate_count,
        local,
        original,
    )
    assert_equal(relocated[1], expected_status)
    if not expected_status:
        assert_equal(relocated[0], expected_instant)


def main() raises:
    var checks = 0
    var originals: List[Int64] = [
        -4000000000,
        0,
        1710054000,
        1730613600,
        16725225600,
    ]
    var wall_deltas: List[Int64] = [-7200, -3600, -1, 0, 1, 1800, 3600, 7200]
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
        transport.validate(
            table.words, 0, table.saves, table.candidates, table.candidate_count
        )
        timezone_tables.validate(table.words, 0)
        assert_equal(table.words.capacity(), len(table.words))
        assert_true(table.construction_peak_bytes >= len(table.words) * 8)
        assert_equal(table.candidate_count, len(zone._candidates))
        for index in range(table.candidate_count):
            assert_equal(
                table.words[table.candidates + index].cast[DType.int64](),
                zone._candidates[index],
            )
        for index in range(len(zone._saves)):
            assert_equal(
                table.words[table.saves + index].cast[DType.int64](),
                zone._saves[index],
            )
        with assert_raises(contains="out of range"):
            transport.validate(
                table.words, 0, 0, table.candidates, table.candidate_count
            )
        with assert_raises(contains="out of range"):
            transport.validate(
                table.words, 0, table.saves, len(table.words) + 1, 0
            )
        for original in originals:
            for delta in wall_deltas:
                _check(
                    zone,
                    table,
                    original + zone.offset_at(original) + delta,
                    original,
                )
                checks += 1
        # Before/at/after each actual CPU transition, including historical types.
        for transition in zone._times:
            for delta in transition_deltas:
                var original = transition + delta
                _check(
                    zone, table, transition + zone.offset_at(original), original
                )
                checks += 1
    var new_york = TimeZone.load("America/New_York")
    var ny = transport.serialize(new_york)
    var pointer = (
        ny.words.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    # 2024-03-10 02:30 local is a gap. 2024-11-03 01:30 has two UTC instants.
    var gap = device.resolve(
        pointer, 0, ny.candidates, ny.candidate_count, 1710037800
    )
    assert_equal(gap[0], 0)
    var fold = device.resolve(
        pointer, 0, ny.candidates, ny.candidate_count, 1730597400
    )
    assert_equal(fold[0], 2)
    assert_equal(fold[1], Int64(1730611800))
    assert_equal(fold[2], Int64(1730615400))
    var invalid = TimeZone.utc()
    invalid._saves.clear()
    with assert_raises(contains="save table size"):
        _ = transport.serialize(invalid)
    print(
        "PASS CPU IANA resolution and relocalization checks:",
        checks,
        flush=True,
    )
