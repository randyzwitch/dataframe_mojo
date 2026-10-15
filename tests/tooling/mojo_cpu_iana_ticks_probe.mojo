"""Run device tick/calendar functions on host against independent CPU methods."""
from std.testing import assert_equal
from tests.tooling.iana_ticks_cases import build_cases, evaluate
from dataframe.timezone import TimeZone
from dataframe.accelerator import timezone_resolution_tables as transport
from dataframe.accelerator.gpu import iana_ticks as device


def main() raises:
    var fixture = build_cases()
    var tables = (
        fixture.tables.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var inputs = (
        fixture.inputs.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var count = len(fixture.inputs) // 11
    for row in range(count):
        var actual = evaluate(tables, inputs, row)
        assert_equal(UInt64(actual[1]), fixture.expected[3 * row + 1])
        assert_equal(UInt64(actual[2]), fixture.expected[3 * row + 2])
        if actual[1] and not actual[2]:
            assert_equal(actual[0], fixture.expected[3 * row])
    _ = fixture^
    var table = transport.serialize(TimeZone.load("America/New_York"))
    var pointer = (
        table.words.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    # 2024-03-10 01:30 New York + one calendar day spans only 23 UTC hours.
    var start = UInt64(1710052200123)
    var day = device.offset_by(
        pointer,
        0,
        table.saves,
        table.candidates,
        table.candidate_count,
        start,
        1,
        0,
        1,
        0,
    )
    var hours = device.offset_by(
        pointer,
        0,
        table.saves,
        table.candidates,
        table.candidate_count,
        start,
        1,
        0,
        0,
        86400000,
    )
    assert_equal(day[1], device.OK)
    assert_equal(hours[1], device.OK)
    assert_equal(day[0], UInt64(1710135000123))
    assert_equal(hours[0], UInt64(1710138600123))
    for policy in [1, 2]:
        var fold = device.localize(
            pointer,
            0,
            table.candidates,
            table.candidate_count,
            UInt64(1730597400123),
            1,
            policy,
            0,
        )
        assert_equal(fold[1], True)
        assert_equal(fold[2], device.OK)
        assert_equal(
            fold[0],
            UInt64(1730611800123) if policy == 1 else UInt64(1730615400123),
        )
    _ = table^
    print(
        "PASS CPU IANA ticks/localize/relocalize/calendar cases:",
        count,
        flush=True,
    )
