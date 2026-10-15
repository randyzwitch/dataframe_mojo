"""Independent CPU calendar/localization oracle and flattened device cases."""
from std.memory import Pointer
from dataframe import DataType
from dataframe import temporal as cpu
from dataframe.temporal_kernels import Truncation, ZonedOffset
from dataframe.timezone import TimeZone
from dataframe.accelerator import timezone_resolution_tables as transport
from dataframe.accelerator.gpu import iana_ticks as device
from dataframe.accelerator.gpu import iana_resolution


@fieldwise_init
struct Cases(Movable):
    var tables: List[UInt64]
    var inputs: List[UInt64]
    var expected: List[UInt64]


def evaluate(
    tables: Pointer[UInt64, ImmutAnyOrigin],
    inputs: Pointer[UInt64, ImmutAnyOrigin],
    row: Int,
) -> Tuple[UInt64, Bool, Int]:
    var at = 11 * row
    var base = Int(inputs[unsafe_offset=at])
    var saves = Int(inputs[unsafe_offset=at + 1])
    var candidates = Int(inputs[unsafe_offset=at + 2])
    var count = Int(inputs[unsafe_offset=at + 3])
    var value = inputs[unsafe_offset=at + 4]
    var original = inputs[unsafe_offset=at + 5]
    var unit = inputs[unsafe_offset=at + 6].cast[DType.int64]()
    var op = inputs[unsafe_offset=at + 7]
    var a = inputs[unsafe_offset=at + 8].cast[DType.int64]()
    var b = inputs[unsafe_offset=at + 9].cast[DType.int64]()
    var c = inputs[unsafe_offset=at + 10].cast[DType.int64]()
    if op == 0:
        return device.localize(
            tables, base, candidates, count, value, unit, Int(a), Int(b)
        )
    var result: Tuple[UInt64, Int]
    if op == 1:
        result = device.relocalize(
            tables, base, saves, candidates, count, value, original, unit
        )
    elif op == 2:
        result = device.truncate(
            tables, base, saves, candidates, count, value, unit, a, b != 0
        )
    else:
        result = device.offset_by(
            tables, base, saves, candidates, count, value, unit, a, b, c
        )
    return (result[0], result[1] == device.OK, result[1])


def _status(message: String, shifted: Bool) raises -> Int:
    if "overflow" in message:
        return device.TICK_OVERFLOW
    if "could not localize" in message:
        return iana_resolution.COULD_NOT_LOCALIZE
    if "non-existent" in message:
        return (
            iana_resolution.SHIFTED_NONEXISTENT if shifted else device.NONEXISTENT
        )
    if "ambiguous" in message:
        return (
            iana_resolution.SHIFTED_AMBIGUOUS if shifted else device.AMBIGUOUS
        )
    raise Error("Unclassified independent CPU temporal diagnostic: " + message)


def _append(
    mut fixture: Cases,
    zone: TimeZone,
    dtype: DataType,
    base: Int,
    saves: Int,
    candidates: Int,
    unit: Int,
    value: Int64,
    original: Int64,
    op: Int,
    a: Int64 = 0,
    b: Int64 = 0,
    c: Int64 = 0,
    interval: String = "",
) raises:
    var result = Int64(0)
    var valid = True
    var status = device.OK
    try:
        if op == 0:
            var localized = cpu.localize(value, dtype, zone, Int(a), Int(b))
            valid = Bool(localized)
            if localized:
                result = localized.value()
        elif op == 1:
            result = cpu.relocalize(value, original, dtype, zone)
        elif op == 2:
            var truncation = Truncation(dtype, interval)
            result = cpu.relocalize(
                truncation.apply(cpu.to_local(value, dtype, zone)),
                value,
                dtype,
                zone,
            )
        else:
            result = ZonedOffset(dtype, interval).apply(value, zone)
    except error:
        status = _status(String(error), op != 0)
        valid = False
    fixture.inputs.extend(
        [
            UInt64(base),
            UInt64(saves),
            UInt64(candidates),
            UInt64(len(zone._candidates)),
            value.cast[DType.uint64](),
            original.cast[DType.uint64](),
            UInt64(unit),
            UInt64(op),
            a.cast[DType.uint64](),
            b.cast[DType.uint64](),
            c.cast[DType.uint64](),
        ]
    )
    fixture.expected.extend(
        [
            result.cast[DType.uint64]() if valid
            and status == device.OK else UInt64(0),
            UInt64(valid),
            UInt64(status),
        ]
    )


def build_cases() raises -> Cases:
    var fixture = Cases(List[UInt64](), List[UInt64](), List[UInt64]())
    var epochs: List[Int64] = [
        -4000000000,
        -1,
        0,
        1710037800,
        1710052200,
        1710054000,
        1730597400,
        1730611800,
        1730615400,
    ]
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
        var base = len(fixture.tables)
        var saves = base + table.saves
        var candidates = base + table.candidates
        var count = table.candidate_count
        fixture.tables.extend(table.words.copy())
        transport.validate(fixture.tables, base, saves, candidates, count)
        for unit in range(1, 4):
            var dtype = DataType.datetime(
                "ms" if unit == 1 else "us" if unit == 2 else "ns", name
            )
            var per = dtype.per_second()
            var values: List[Int64] = [
                Int64.MIN,
                Int64.MIN + 1,
                Int64.MAX - 1,
                Int64.MAX,
            ]
            for seconds in epochs:
                values.append(seconds * per - 1)
                values.append(seconds * per)
                values.append(seconds * per + 1)
            if unit < 3:
                values.append(16725225600 * per + 123)
            for value in values:
                for ambiguous in range(4):
                    for non_existent in range(2):
                        _append(
                            fixture,
                            zone,
                            dtype,
                            base,
                            saves,
                            candidates,
                            unit,
                            value,
                            value,
                            0,
                            Int64(ambiguous),
                            Int64(non_existent),
                        )
                for delta in List[Int64]([Int64(-3600), Int64(0), Int64(3600)]):
                    var original = (
                        value.cast[DType.uint64]()
                        + (delta * per).cast[DType.uint64]()
                    ).cast[DType.int64]()
                    _append(
                        fixture,
                        zone,
                        dtype,
                        base,
                        saves,
                        candidates,
                        unit,
                        value,
                        original,
                        1,
                    )
                for every in ["1h", "1d", "1w", "1mo", "3mo", "1y"]:
                    var truncation = Truncation(dtype, every)
                    _append(
                        fixture,
                        zone,
                        dtype,
                        base,
                        saves,
                        candidates,
                        unit,
                        value,
                        value,
                        2,
                        truncation.months if truncation.months else truncation.step,
                        Int64(truncation.months != 0),
                        interval=every,
                    )
                for by in [
                    "0d",
                    "1d",
                    "-1d",
                    "1mo",
                    "-1mo",
                    "1mo1d1h",
                    "-1mo1d1h",
                    "1h",
                    "24h",
                ]:
                    var offset = ZonedOffset(dtype, by)
                    _append(
                        fixture,
                        zone,
                        dtype,
                        base,
                        saves,
                        candidates,
                        unit,
                        value,
                        value,
                        3,
                        offset.months,
                        offset.days,
                        offset.ticks,
                        by,
                    )
    return fixture^
