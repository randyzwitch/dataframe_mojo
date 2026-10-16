"""A pipeline's join step against a scalar model. Where each probe row
matches at most once, the step keeps the probe side's columns where they are
under a selection (row indices for sparse selections, a mask for dense
ones) and gathers only the build side's columns; a repeated build key makes
it gather every column at the repeated rows. Inner, semi and anti joins run
on fresh morsels and after dense and sparse filters, with nullable keys,
unmatched keys, a suffixed name and uncoalesced keys, for several worker
counts and batch sizes."""
# contract-group: table.join
from std.testing import assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    LazyFrame,
    Series,
    col,
    lit,
)
from dataframe.value import AnyValue
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 50_000


def integer(value: AnyValue) raises -> Int128:
    """An Int64 or Int32 result cell."""
    if value.dtype() == DataType.INT32:
        return Int128(value.int32())
    return Int128(value.int64())


comptime DIMENSIONS = 600
comptime PROBE_KEYS = 700


def key_valid(row: Int) -> Bool:
    return row % 97 != 3


def key_of(row: Int) -> Int:
    return (row * 7919) % PROBE_KEYS


def has_dimension(k: Int) -> Bool:
    return k < DIMENSIONS and k % 13 != 5


def text_of(row: Int) -> String:
    return "probe text value number " + String(row % 50)


def name_of(k: Int) -> String:
    return "dimension payload " + String(k % 37)


def probe() raises -> DataFrame:
    var keys = List[Int64]()
    var valid = List[Bool]()
    var a = List[Int64]()
    var s = List[String]()
    for row in range(ROWS):
        keys.append(Int64(key_of(row)))
        valid.append(key_valid(row))
        a.append(Int64(row))
        s.append(text_of(row))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^, valid^)),
            Series("a", Column[Int64](a^)),
            Series("s", Column[String](s^)),
        ]
    )


def dimension(repeated: Bool = False) raises -> DataFrame:
    """Unique keys, or each key that is a multiple of three twice."""
    var keys = List[Int64]()
    var names = List[String]()
    var a = List[Int64]()
    var copy = List[Int64]()
    for k in range(DIMENSIONS):
        if not has_dimension(k):
            continue
        var copies = 2 if repeated and k % 3 == 0 else 1
        for c in range(copies):
            keys.append(Int64(k))
            names.append(name_of(k))
            a.append(Int64(2 * k))
            copy.append(Int64(c))
    return DataFrame(
        [
            Series("dk", Column[Int64](keys^)),
            Series("name", Column[String](names^)),
            Series("a", Column[Int64](a^)),
            Series("copy", Column[Int64](copy^)),
        ]
    )


def kept(row: Int, mode: Int) -> Bool:
    """No filter (0), a dense filter (1) or a sparse one (2)."""
    if mode == 1:
        return row >= 1_000
    if mode == 2:
        return row % 16 == 0
    return True


def filtered(mode: Int) raises -> LazyFrame:
    var plan = probe().lazy()
    if mode == 1:
        plan = plan.filter(col("a") >= lit(Int64(1_000)))
    elif mode == 2:
        plan = plan.filter((col("a") % lit(Int64(16))) == lit(Int64(0)))
    return plan^


def matched(row: Int) -> Bool:
    return key_valid(row) and has_dimension(key_of(row))


def check_inner(actual: DataFrame, mode: Int, coalesce: Bool) raises:
    var names: List[String] = ["k", "a", "s", "name", "a_right", "copy"]
    if not coalesce:
        names = ["k", "a", "s", "dk", "name", "a_right", "copy"]
    assert_equal(actual.columns(), names)
    var r = 0
    for row in range(ROWS):
        if not (kept(row, mode) and matched(row)):
            continue
        var k = key_of(row)
        assert_equal(integer(actual.item(r, "a")), Int128(row))
        assert_equal(integer(actual.item(r, "k")), Int128(k))
        assert_equal(actual.item(r, "s").string(), text_of(row))
        assert_equal(actual.item(r, "name").string(), name_of(k))
        assert_equal(integer(actual.item(r, "a_right")), Int128(2 * k))
        if not coalesce:
            assert_equal(integer(actual.item(r, "dk")), Int128(k))
        r += 1
    assert_equal(actual.height(), r)


def check_semi_anti(actual: DataFrame, mode: Int, semi: Bool) raises:
    assert_equal(actual.columns(), ["k", "a", "s"])
    var r = 0
    for row in range(ROWS):
        if not kept(row, mode) or matched(row) != semi:
            continue
        assert_equal(integer(actual.item(r, "a")), Int128(row))
        assert_equal(actual.item(r, "s").string(), text_of(row))
        r += 1
    assert_equal(actual.height(), r)


def test_inner_semi_and_anti_steps_match_scalar_rows_for_every_mode() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    for threads in [1, 8]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        for batch in [997, 65536]:
            for mode in range(3):
                for coalesce in [True, False]:
                    var observed = StrategyTrace()
                    var inner = (
                        filtered(mode)
                        .join(
                            dimension().lazy(),
                            left_on=["k"],
                            right_on=["dk"],
                            coalesce=coalesce,
                        )
                        .collect(batch_size=batch)
                    )
                    assert_true(
                        "dataframe-path: pipeline.join_selection\n"
                        in observed.text(),
                        "the join did not keep a selection",
                    )
                    _ = observed^
                    check_inner(inner, mode, coalesce)
                for semi in [True, False]:
                    check_semi_anti(
                        filtered(mode)
                        .join(
                            dimension().lazy(),
                            left_on=["k"],
                            right_on=["dk"],
                            how="semi" if semi else "anti",
                        )
                        .collect(batch_size=batch),
                        mode,
                        semi,
                    )
        _ = workers^
    _ = trace^


def test_repeated_build_keys_repeat_every_probe_column() raises:
    # Keys that are multiples of three have two build rows: their probe
    # rows come back twice, once per copy, with every probe column.
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    for mode in range(3):
        var actual = (
            filtered(mode)
            .join(dimension(True).lazy(), left_on=["k"], right_on=["dk"])
            .collect(batch_size=4096)
        )
        var expected = Dict[String, Int]()
        var total = 0
        for row in range(ROWS):
            if not (kept(row, mode) and matched(row)):
                continue
            var k = key_of(row)
            for c in range(2 if k % 3 == 0 else 1):
                var key = String(row) + "|" + String(c)
                expected[key] = expected.get(key, 0) + 1
                total += 1
        assert_equal(actual.height(), total)
        for r in range(actual.height()):
            var row = Int(integer(actual.item(r, "a")))
            assert_equal(actual.item(r, "s").string(), text_of(row))
            assert_equal(actual.item(r, "name").string(), name_of(key_of(row)))
            var key = (
                String(row) + "|" + String(Int(integer(actual.item(r, "copy"))))
            )
            assert_true(key in expected, "unexpected joined row " + key)
            expected[key] -= 1
            assert_true(expected[key] >= 0, "joined row twice: " + key)
    _ = workers^
