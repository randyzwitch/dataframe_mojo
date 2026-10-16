"""Unbuilt integration fixtures: IANA tables preserve exact CPU local fields."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe.metal import MetalRuntime
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows, ROW_IANA_ZONE_FIELDS
from dataframe.accelerator.gpu.runtime import capabilities, estimate_resources
from dataframe.accelerator.gpu.program import Program
from tests.support.coverage_compare import assert_frame
from tests.execution_modes.test_mojo_gpu_timezone_conversion import (
    _check_local_literals,
)
from tests.support.coverage_generators import offset_layout, chunk_layout


def _fields() raises -> List[Expr]:
    return [
        col("t").dt().year().alias("year"),
        col("t").dt().month().alias("month"),
        col("t").dt().day().alias("day"),
        col("t").dt().hour().alias("hour"),
        col("t").dt().minute().alias("minute"),
        col("t").dt().second().alias("second"),
        col("t").dt().nanosecond().alias("nano"),
        col("t").dt().weekday().alias("weekday"),
        col("t").dt().ordinal_day().alias("ordinal"),
        col("t").dt().date().alias("date"),
        col("t").dt().time().alias("time"),
    ]


def _literals() -> List[List[Int64]]:
    # Independent Python zoneinfo/arbitrary-integer Gregorian fixtures.
    # Columns: zone index, unit1/2/3, UTC ticks, eleven exact local fields.
    return [
        [
            0,
            1,
            1710053999999,
            2024,
            3,
            10,
            1,
            59,
            59,
            999000000,
            7,
            70,
            19792,
            7199999000000,
        ],
        [
            0,
            2,
            1710053999999999,
            2024,
            3,
            10,
            1,
            59,
            59,
            999999000,
            7,
            70,
            19792,
            7199999999000,
        ],
        [
            0,
            3,
            1710053999999999999,
            2024,
            3,
            10,
            1,
            59,
            59,
            999999999,
            7,
            70,
            19792,
            7199999999999,
        ],
        [
            0,
            1,
            1710054000000,
            2024,
            3,
            10,
            3,
            0,
            0,
            0,
            7,
            70,
            19792,
            10800000000000,
        ],
        [
            0,
            2,
            1710054000000000,
            2024,
            3,
            10,
            3,
            0,
            0,
            0,
            7,
            70,
            19792,
            10800000000000,
        ],
        [
            0,
            3,
            1710054000000000000,
            2024,
            3,
            10,
            3,
            0,
            0,
            0,
            7,
            70,
            19792,
            10800000000000,
        ],
        [
            0,
            1,
            1730613599999,
            2024,
            11,
            3,
            1,
            59,
            59,
            999000000,
            7,
            308,
            20030,
            7199999000000,
        ],
        [
            0,
            2,
            1730613599999999,
            2024,
            11,
            3,
            1,
            59,
            59,
            999999000,
            7,
            308,
            20030,
            7199999999000,
        ],
        [
            0,
            3,
            1730613599999999999,
            2024,
            11,
            3,
            1,
            59,
            59,
            999999999,
            7,
            308,
            20030,
            7199999999999,
        ],
        [
            0,
            1,
            1730613600000,
            2024,
            11,
            3,
            1,
            0,
            0,
            0,
            7,
            308,
            20030,
            3600000000000,
        ],
        [
            0,
            2,
            1730613600000000,
            2024,
            11,
            3,
            1,
            0,
            0,
            0,
            7,
            308,
            20030,
            3600000000000,
        ],
        [
            0,
            3,
            1730613600000000000,
            2024,
            11,
            3,
            1,
            0,
            0,
            0,
            7,
            308,
            20030,
            3600000000000,
        ],
        [
            0,
            1,
            1730611800000,
            2024,
            11,
            3,
            1,
            30,
            0,
            0,
            7,
            308,
            20030,
            5400000000000,
        ],
        [
            0,
            2,
            1730611800000000,
            2024,
            11,
            3,
            1,
            30,
            0,
            0,
            7,
            308,
            20030,
            5400000000000,
        ],
        [
            0,
            3,
            1730611800000000000,
            2024,
            11,
            3,
            1,
            30,
            0,
            0,
            7,
            308,
            20030,
            5400000000000,
        ],
        [
            0,
            1,
            1730615400000,
            2024,
            11,
            3,
            1,
            30,
            0,
            0,
            7,
            308,
            20030,
            5400000000000,
        ],
        [
            0,
            2,
            1730615400000000,
            2024,
            11,
            3,
            1,
            30,
            0,
            0,
            7,
            308,
            20030,
            5400000000000,
        ],
        [
            0,
            3,
            1730615400000000000,
            2024,
            11,
            3,
            1,
            30,
            0,
            0,
            7,
            308,
            20030,
            5400000000000,
        ],
        [
            1,
            1,
            1712415599999,
            2024,
            4,
            7,
            1,
            59,
            59,
            999000000,
            7,
            98,
            19820,
            7199999000000,
        ],
        [
            1,
            2,
            1712415599999999,
            2024,
            4,
            7,
            1,
            59,
            59,
            999999000,
            7,
            98,
            19820,
            7199999999000,
        ],
        [
            1,
            3,
            1712415599999999999,
            2024,
            4,
            7,
            1,
            59,
            59,
            999999999,
            7,
            98,
            19820,
            7199999999999,
        ],
        [
            1,
            1,
            1712415600000,
            2024,
            4,
            7,
            1,
            30,
            0,
            0,
            7,
            98,
            19820,
            5400000000000,
        ],
        [
            1,
            2,
            1712415600000000,
            2024,
            4,
            7,
            1,
            30,
            0,
            0,
            7,
            98,
            19820,
            5400000000000,
        ],
        [
            1,
            3,
            1712415600000000000,
            2024,
            4,
            7,
            1,
            30,
            0,
            0,
            7,
            98,
            19820,
            5400000000000,
        ],
        [
            1,
            1,
            1728142199999,
            2024,
            10,
            6,
            1,
            59,
            59,
            999000000,
            7,
            280,
            20002,
            7199999000000,
        ],
        [
            1,
            2,
            1728142199999999,
            2024,
            10,
            6,
            1,
            59,
            59,
            999999000,
            7,
            280,
            20002,
            7199999999000,
        ],
        [
            1,
            3,
            1728142199999999999,
            2024,
            10,
            6,
            1,
            59,
            59,
            999999999,
            7,
            280,
            20002,
            7199999999999,
        ],
        [
            1,
            1,
            1728142200000,
            2024,
            10,
            6,
            2,
            30,
            0,
            0,
            7,
            280,
            20002,
            9000000000000,
        ],
        [
            1,
            2,
            1728142200000000,
            2024,
            10,
            6,
            2,
            30,
            0,
            0,
            7,
            280,
            20002,
            9000000000000,
        ],
        [
            1,
            3,
            1728142200000000000,
            2024,
            10,
            6,
            2,
            30,
            0,
            0,
            7,
            280,
            20002,
            9000000000000,
        ],
        [
            2,
            1,
            504901799999,
            1985,
            12,
            31,
            23,
            59,
            59,
            999000000,
            2,
            365,
            5843,
            86399999000000,
        ],
        [
            2,
            2,
            504901799999999,
            1985,
            12,
            31,
            23,
            59,
            59,
            999999000,
            2,
            365,
            5843,
            86399999999000,
        ],
        [
            2,
            3,
            504901799999999999,
            1985,
            12,
            31,
            23,
            59,
            59,
            999999999,
            2,
            365,
            5843,
            86399999999999,
        ],
        [2, 1, 504901800000, 1986, 1, 1, 0, 15, 0, 0, 3, 1, 5844, 900000000000],
        [
            2,
            2,
            504901800000000,
            1986,
            1,
            1,
            0,
            15,
            0,
            0,
            3,
            1,
            5844,
            900000000000,
        ],
        [
            2,
            3,
            504901800000000000,
            1986,
            1,
            1,
            0,
            15,
            0,
            0,
            3,
            1,
            5844,
            900000000000,
        ],
        [
            3,
            1,
            1325239199999,
            2011,
            12,
            29,
            23,
            59,
            59,
            999000000,
            4,
            363,
            15337,
            86399999000000,
        ],
        [
            3,
            2,
            1325239199999999,
            2011,
            12,
            29,
            23,
            59,
            59,
            999999000,
            4,
            363,
            15337,
            86399999999000,
        ],
        [
            3,
            3,
            1325239199999999999,
            2011,
            12,
            29,
            23,
            59,
            59,
            999999999,
            4,
            363,
            15337,
            86399999999999,
        ],
        [3, 1, 1325239200000, 2011, 12, 31, 0, 0, 0, 0, 6, 365, 15339, 0],
        [3, 2, 1325239200000000, 2011, 12, 31, 0, 0, 0, 0, 6, 365, 15339, 0],
        [3, 3, 1325239200000000000, 2011, 12, 31, 0, 0, 0, 0, 6, 365, 15339, 0],
        [
            0,
            1,
            -5364662400000,
            1799,
            12,
            31,
            19,
            3,
            58,
            0,
            2,
            365,
            -62092,
            68638000000000,
        ],
        [
            0,
            2,
            -5364662400000000,
            1799,
            12,
            31,
            19,
            3,
            58,
            0,
            2,
            365,
            -62092,
            68638000000000,
        ],
        [
            0,
            3,
            -5364662400000000000,
            1799,
            12,
            31,
            19,
            3,
            58,
            0,
            2,
            365,
            -62092,
            68638000000000,
        ],
        [
            4,
            1,
            -2208988800000,
            1900,
            1,
            1,
            0,
            9,
            21,
            0,
            1,
            1,
            -25567,
            561000000000,
        ],
        [
            4,
            2,
            -2208988800000000,
            1900,
            1,
            1,
            0,
            9,
            21,
            0,
            1,
            1,
            -25567,
            561000000000,
        ],
        [
            4,
            3,
            -2208988800000000000,
            1900,
            1,
            1,
            0,
            9,
            21,
            0,
            1,
            1,
            -25567,
            561000000000,
        ],
        [
            0,
            1,
            2525817600000,
            2050,
            1,
            14,
            19,
            0,
            0,
            0,
            5,
            14,
            29233,
            68400000000000,
        ],
        [
            0,
            2,
            2525817600000000,
            2050,
            1,
            14,
            19,
            0,
            0,
            0,
            5,
            14,
            29233,
            68400000000000,
        ],
        [
            0,
            3,
            2525817600000000000,
            2050,
            1,
            14,
            19,
            0,
            0,
            0,
            5,
            14,
            29233,
            68400000000000,
        ],
        [
            0,
            1,
            2541456000000,
            2050,
            7,
            14,
            20,
            0,
            0,
            0,
            4,
            195,
            29414,
            72000000000000,
        ],
        [
            0,
            2,
            2541456000000000,
            2050,
            7,
            14,
            20,
            0,
            0,
            0,
            4,
            195,
            29414,
            72000000000000,
        ],
        [
            0,
            3,
            2541456000000000000,
            2050,
            7,
            14,
            20,
            0,
            0,
            0,
            4,
            195,
            29414,
            72000000000000,
        ],
        [
            1,
            1,
            2525817600000,
            2050,
            1,
            15,
            11,
            0,
            0,
            0,
            6,
            15,
            29234,
            39600000000000,
        ],
        [
            1,
            2,
            2525817600000000,
            2050,
            1,
            15,
            11,
            0,
            0,
            0,
            6,
            15,
            29234,
            39600000000000,
        ],
        [
            1,
            3,
            2525817600000000000,
            2050,
            1,
            15,
            11,
            0,
            0,
            0,
            6,
            15,
            29234,
            39600000000000,
        ],
        [
            1,
            1,
            2541456000000,
            2050,
            7,
            15,
            10,
            30,
            0,
            0,
            5,
            196,
            29415,
            37800000000000,
        ],
        [
            1,
            2,
            2541456000000000,
            2050,
            7,
            15,
            10,
            30,
            0,
            0,
            5,
            196,
            29415,
            37800000000000,
        ],
        [
            1,
            3,
            2541456000000000000,
            2050,
            7,
            15,
            10,
            30,
            0,
            0,
            5,
            196,
            29415,
            37800000000000,
        ],
        [5, 1, 2525817600000, 2050, 1, 15, 0, 0, 0, 0, 6, 15, 29234, 0],
        [5, 2, 2525817600000000, 2050, 1, 15, 0, 0, 0, 0, 6, 15, 29234, 0],
        [5, 3, 2525817600000000000, 2050, 1, 15, 0, 0, 0, 0, 6, 15, 29234, 0],
        [
            5,
            1,
            2541456000000,
            2050,
            7,
            15,
            1,
            0,
            0,
            0,
            5,
            196,
            29415,
            3600000000000,
        ],
        [
            5,
            2,
            2541456000000000,
            2050,
            7,
            15,
            1,
            0,
            0,
            0,
            5,
            196,
            29415,
            3600000000000,
        ],
        [
            5,
            3,
            2541456000000000000,
            2050,
            7,
            15,
            1,
            0,
            0,
            0,
            5,
            196,
            29415,
            3600000000000,
        ],
        [
            0,
            1,
            Int64.MIN,
            292278994,
            8,
            17,
            2,
            16,
            53,
            808000000,
            7,
            229,
            106751991167,
            8213808000000,
        ],
        [
            0,
            1,
            -9223372036854775807,
            292278994,
            8,
            17,
            2,
            16,
            53,
            809000000,
            7,
            229,
            106751991167,
            8213809000000,
        ],
        [
            0,
            1,
            -1,
            1969,
            12,
            31,
            18,
            59,
            59,
            999000000,
            3,
            365,
            -1,
            68399999000000,
        ],
        [0, 1, 0, 1969, 12, 31, 19, 0, 0, 0, 3, 365, -1, 68400000000000],
        [0, 1, 1, 1969, 12, 31, 19, 0, 0, 1000000, 3, 365, -1, 68400001000000],
        [
            0,
            1,
            9223372036854775806,
            292278994,
            8,
            17,
            3,
            12,
            55,
            806000000,
            7,
            229,
            106751991167,
            11575806000000,
        ],
        [
            0,
            1,
            Int64.MAX,
            292278994,
            8,
            17,
            3,
            12,
            55,
            807000000,
            7,
            229,
            106751991167,
            11575807000000,
        ],
        [
            0,
            2,
            Int64.MIN,
            294247,
            1,
            9,
            23,
            4,
            52,
            775808000,
            6,
            9,
            106751990,
            83092775808000,
        ],
        [
            0,
            2,
            -9223372036854775807,
            294247,
            1,
            9,
            23,
            4,
            52,
            775809000,
            6,
            9,
            106751990,
            83092775809000,
        ],
        [
            0,
            2,
            -1,
            1969,
            12,
            31,
            18,
            59,
            59,
            999999000,
            3,
            365,
            -1,
            68399999999000,
        ],
        [0, 2, 0, 1969, 12, 31, 19, 0, 0, 0, 3, 365, -1, 68400000000000],
        [0, 2, 1, 1969, 12, 31, 19, 0, 0, 1000, 3, 365, -1, 68400000001000],
        [
            0,
            2,
            9223372036854775806,
            294247,
            1,
            9,
            23,
            0,
            54,
            775806000,
            6,
            9,
            106751990,
            82854775806000,
        ],
        [
            0,
            2,
            Int64.MAX,
            294247,
            1,
            9,
            23,
            0,
            54,
            775807000,
            6,
            9,
            106751990,
            82854775807000,
        ],
        [
            0,
            3,
            Int64.MIN,
            2262,
            4,
            11,
            18,
            51,
            14,
            854775808,
            5,
            101,
            106751,
            67874854775808,
        ],
        [
            0,
            3,
            -9223372036854775807,
            2262,
            4,
            11,
            18,
            51,
            14,
            854775809,
            5,
            101,
            106751,
            67874854775809,
        ],
        [
            0,
            3,
            -1,
            1969,
            12,
            31,
            18,
            59,
            59,
            999999999,
            3,
            365,
            -1,
            68399999999999,
        ],
        [0, 3, 0, 1969, 12, 31, 19, 0, 0, 0, 3, 365, -1, 68400000000000],
        [0, 3, 1, 1969, 12, 31, 19, 0, 0, 1, 3, 365, -1, 68400000000001],
        [
            0,
            3,
            9223372036854775806,
            2262,
            4,
            11,
            19,
            47,
            16,
            854775806,
            5,
            101,
            106751,
            71236854775806,
        ],
        [
            0,
            3,
            Int64.MAX,
            2262,
            4,
            11,
            19,
            47,
            16,
            854775807,
            5,
            101,
            106751,
            71236854775807,
        ],
    ]


# requires: metal
def test_mojo_gpu_iana_fields_literal_transitions_footer_extrema_and_nullable_layouts() raises:
    var zones: List[String] = [
        "America/New_York",
        "Australia/Lord_Howe",
        "Asia/Kathmandu",
        "Pacific/Apia",
        "Europe/Paris",
        "Europe/Dublin",
    ]
    var records = _literals()
    var fields = _fields()
    var names: List[String] = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "nano",
        "weekday",
        "ordinal",
        "date",
        "time",
    ]
    var runtime = MetalRuntime()
    for zone in range(len(zones)):
        for unit in range(1, 4):
            var values = List[Int64]()
            var valid = List[Bool]()
            var expected_rows = List[List[Int64]]()
            for record in records:
                if record[0] == Int64(zone) and record[1] == Int64(unit):
                    values.append(record[2])
                    valid.append(True)
                    expected_rows.append(record.copy())
            # Hidden endpoint payload must stay null even across lookup/wrap.
            values.append(Int64.MIN)
            valid.append(False)
            var dtype = DataType.datetime(
                "ms" if unit == 1 else "us" if unit == 2 else "ns", zones[zone]
            )
            var source = DataFrame(
                [Series("t", Column[Int64](values^, valid^)).with_dtype(dtype)]
            )
            var expected_columns = List[Series]()
            for field in range(len(names)):
                var words = List[Int64]()
                var bits = List[Bool]()
                for record in expected_rows:
                    words.append(record[field + 3])
                    bits.append(True)
                words.append(0)
                bits.append(False)
                var column = Series(names[field], Column[Int64](words^, bits^))
                if field == 9:
                    column = column.with_dtype(DataType.DATE)
                elif field == 10:
                    column = column.with_dtype(DataType.TIME)
                expected_columns.append(column^)
            var expected = DataFrame(expected_columns^)
            for layout in range(3):
                var frame = (
                    source.copy() if layout
                    == 0 else offset_layout(source) if layout
                    == 1 else chunk_layout(source)
                )
                var query = frame.lazy().select_exprs(fields)
                assert_frame(
                    query.collect(engine="cpu", streaming=False), expected
                )
                assert_frame(query.collect(accelerator=runtime), expected)
                var empty = (
                    frame.head(0)
                    .lazy()
                    .select_exprs(fields)
                    .collect(accelerator=runtime)
                )
                assert_equal(empty.height(), 0)
                assert_equal(empty.dtypes(), expected.dtypes())


# requires: metal
def test_mojo_gpu_iana_fields_descriptor_dedup_budgets_conversion_and_refusals() raises:
    _check_distinct_zone_tables_after_decimal_program()
    assert_true(not RowCapabilities().iana_zone_temporal_fields)
    var source = DataFrame(
        [
            Series("t", Column[Int64]([1710054000000, -1])).with_dtype(
                DataType.datetime("ms", "America/New_York")
            )
        ]
    )
    var query = source.lazy().select_exprs(_fields())
    var caps = capabilities()
    caps.iana_zone_temporal_fields = False
    with assert_raises(contains="time zone field extraction"):
        _ = lower_rows(query, caps)
    caps.iana_zone_temporal_fields = True
    var plan = lower_rows(query, caps)
    var single = lower_rows(source.lazy().select(col("t").dt().year()), caps)
    assert_true(len(plan.zone_words) > 0)
    assert_equal(len(plan.zone_words), len(single.zone_words))
    var program = Program(plan)
    assert_equal(len(program.node_types), len(plan.node_dtypes))
    assert_equal(
        len(program.literals), len(plan.literals) + len(plan.zone_words)
    )
    var flagged = 0
    for node in range(len(plan.node_dtypes)):
        var op = plan.code[4 * node]
        if (
            op >= 130
            and op <= 140
            and (plan.code[4 * node + 3] & ROW_IANA_ZONE_FIELDS)
        ):
            flagged += 1
            assert_true(program.code[4 * node + 2] >= Int64(len(plan.literals)))
    assert_equal(flagged, 11)
    var utc = source.column("t").with_dtype(DataType.datetime("ms", "UTC"))
    var plain_query = DataFrame([utc.copy()]).lazy().select_exprs(_fields())
    var plain = lower_rows(plain_query, caps)
    var facts = estimate_resources(plan)
    var plain_facts = estimate_resources(plain)
    var bytes = len(plan.zone_words) * 8
    assert_equal(facts.program_bytes - plain_facts.program_bytes, bytes)
    assert_equal(facts.peak_bytes - plain_facts.peak_bytes, 3 * bytes)
    assert_equal(facts.upload_bytes - plain_facts.upload_bytes, bytes)
    var empty = lower_rows(source.head(0).lazy().select_exprs(_fields()), caps)
    var empty_plain = lower_rows(
        DataFrame([utc.head(0)]).lazy().select_exprs(_fields()), caps
    )
    assert_equal(
        estimate_resources(empty).peak_bytes
        - estimate_resources(empty_plain).peak_bytes,
        2 * bytes,
    )
    var measured = query.profile(accelerator=MetalRuntime())
    var peak = Int(measured[1].item(0, "peak_requested_bytes").int64())
    assert_frame(
        query.collect(accelerator=MetalRuntime(memory_limit_bytes=peak)),
        measured[0],
    )
    var limited = MetalRuntime(memory_limit_bytes=peak - 1)
    with assert_raises(contains="memory budget"):
        _ = query.collect(accelerator=limited)
    assert_equal(
        DataFrame([Series("x", Column[Int64]([7]))])
        .lazy()
        .select(col("x"))
        .collect(accelerator=limited)
        .item()
        .int64(),
        Int64(7),
    )
    var converted = (
        DataFrame([utc.copy()])
        .lazy()
        .select(
            col("t")
            .dt()
            .convert_time_zone("America/New_York")
            .dt()
            .hour()
            .alias("hour")
        )
    )
    assert_frame(
        converted.collect(accelerator=MetalRuntime()),
        converted.collect(engine="cpu"),
    )
    assert_equal(
        converted.collect(accelerator=MetalRuntime())
        .column("hour")
        .int64()
        .to_list(),
        [Int64(3), Int64(18)],
    )
    # 2024-03-10T07:00Z is 03:00 EDT; local midnight is still EST.
    # -1ms is the preceding local day, whose midnight was 05:00Z.
    _check_local_literals(
        source.lazy().select(col("t").dt().truncate("1d")),
        DataType.datetime("ms", "America/New_York"),
        [Int64(1710046800000), Int64(-68400000)],
    )
    _check_local_literals(
        source.lazy().select(col("t").dt().replace_time_zone("UTC")),
        DataType.datetime("ms", "UTC"),
        [Int64(1710039600000), Int64(-18000001)],
    )


def _check_distinct_zone_tables_after_decimal_program() raises:
    # Independent UTC instants straddling New York's 2024 spring transition.
    # Paris is still on UTC+1; -1ms is 00:59:59.999 there and the prior day in NY.
    var ticks: List[Int64] = [1710053999999, 1710054000000, -1]
    var source = DataFrame(
        [
            Series("ny", Column[Int64](ticks.copy())).with_dtype(
                DataType.datetime("ms", "America/New_York")
            ),
            Series("paris", Column[Int64](ticks.copy())).with_dtype(
                DataType.datetime("ms", "Europe/Paris")
            ),
            Series("d", Column[Int64]([125, -250, 0])).with_dtype(
                DataType.decimal(18, 2, 64)
            ),
        ]
    )
    var expected = DataFrame(
        [
            Series("ny_hour", Column[Int64]([1, 3, 18])),
            Series("paris_hour", Column[Int64]([7, 8, 0])),
            Series("ny_minute", Column[Int64]([59, 0, 59])),
            Series("ny_year", Column[Int64]([2024, 2024, 1969])),
            Series("mixed", Column[Float64]([1.25, -2.5, 0.0])),
            Series("literal", Column[Int64]([7, 7, 7])),
        ]
    )
    for variant in range(3):
        var frame = (
            source.copy() if variant
            == 0 else offset_layout(source) if variant
            == 1 else chunk_layout(source)
        )
        var query = (
            frame.lazy()
            .with_columns(col("d").cast(DataType.FLOAT64).alias("mixed"))
            .select_exprs(
                [
                    col("ny").dt().hour().alias("ny_hour"),
                    col("paris").dt().hour().alias("paris_hour"),
                    col("ny").dt().minute().alias("ny_minute"),
                    col("ny").dt().year().alias("ny_year"),
                    col("mixed"),
                    lit(Int64(7)).alias("literal"),
                ]
            )
        )
        var plan = lower_rows(query, capabilities())
        var program = Program(plan)
        var bases = List[Int64]()
        var later_step = False
        for step in plan.steps:
            for node in range(step.start, step.start + step.nodes):
                var op = plan.code[4 * node]
                if (
                    op >= 130
                    and op <= 140
                    and (plan.code[4 * node + 3] & ROW_IANA_ZONE_FIELDS)
                ):
                    var base = plan.code[4 * node + 2]
                    bases.append(base)
                    later_step |= step.start > 0
                    assert_equal(
                        program.code[4 * node + 2],
                        Int64(len(plan.literals)) + base,
                    )
        assert_equal(len(bases), 4)
        assert_equal(bases[0], Int64(0))
        assert_true(bases[1] > 0)
        assert_equal(bases[2], bases[0])
        assert_equal(bases[3], bases[0])
        assert_true(later_step)
        assert_equal(len(program.node_types), len(plan.node_dtypes))
        assert_frame(query.collect(engine="cpu", streaming=False), expected)
        assert_frame(query.collect(accelerator=MetalRuntime()), expected)
