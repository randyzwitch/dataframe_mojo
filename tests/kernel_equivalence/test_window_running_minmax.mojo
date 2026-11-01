"""Running min and max over partitions take one pass for every integer
storage, decimal32, decimal64 and decimal128 and dates included, and keep
the input's type. Several window expressions over one partition key, in a
select or nested in another over(), share one partition layout. Each
against a scalar model, with nulls, many partitions, forward and reverse."""
# contract-group: expr.order_windows
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, DataType, Series, col, lit, when

comptime ROWS = 20_000
comptime GROUPS = 700


def value_of(row: Int) -> Int64:
    return Int64((row * 7919) % 1000) - 500


def value_valid(row: Int) -> Bool:
    return row % 13 != 4


def group_of(row: Int) -> Int64:
    return Int64((row * 31) % GROUPS)


def frame() raises -> DataFrame:
    var values = List[Int64]()
    var valid = List[Bool]()
    var groups = List[Int64]()
    for row in range(ROWS):
        values.append(value_of(row))
        valid.append(value_valid(row))
        groups.append(group_of(row))
    return DataFrame(
        [
            Series("v", Column[Int64](values^, valid)),
            Series("g", Column[Int64](groups^)),
        ]
    )


def expected(take_max: Bool, reverse: Bool) -> Tuple[List[Int64], List[Bool]]:
    """Per row: the partition's running extreme up to it (in row order, or
    from the last row when reversed), null on a null row."""
    var best = List[Int64](length=GROUPS, fill=0)
    var seen = List[Bool](length=GROUPS, fill=False)
    var out = List[Int64](length=ROWS, fill=0)
    var valid = List[Bool](length=ROWS, fill=False)
    for step in range(ROWS):
        var row = ROWS - 1 - step if reverse else step
        if not value_valid(row):
            continue
        var g = Int(group_of(row))
        var v = value_of(row)
        if not seen[g] or (v > best[g] if take_max else v < best[g]):
            best[g] = v
            seen[g] = True
        out[row] = best[g]
        valid[row] = True
    return (out^, valid^)


def check(
    actual: Series, dtype: DataType, take_max: Bool, reverse: Bool
) raises:
    assert_true(actual.dtype() == dtype, "the input's type is not kept")
    var model = expected(take_max, reverse)
    # Every type here holds the model's integers exactly as Float64.
    var numbers = actual.cast(DataType.FLOAT64)
    for row in range(ROWS):
        var cell = numbers.get(row)
        assert_equal(cell.is_null(), not model[1][row])
        if model[1][row]:
            assert_equal(cell.float64(), Float64(model[0][row]))


def test_running_min_and_max_keep_every_integer_storage_type() raises:
    var source = frame()
    var types: List[DataType] = [
        DataType.INT32,
        DataType.INT64,
        DataType.decimal(9, 0, 32),
        DataType.decimal(18, 0, 64),
        DataType.decimal(38, 0, 128),
    ]
    for dtype in types:
        var typed = source.with_columns([col("v").cast(dtype).alias("v")])
        for take_max in [False, True]:
            for reverse in [False, True]:
                var running = (
                    col("v")
                    .cum_max(reverse=reverse) if take_max else col("v")
                    .cum_min(reverse=reverse)
                ).over("g")
                var result = typed.select_exprs([running.alias("r")])
                check(result.column("r"), dtype, take_max, reverse)
    # Dates: days after the epoch, shifted to stay positive.
    var dated = source.with_columns(
        [(col("v") + lit(Int64(1000))).cast(DataType.DATE).alias("v")]
    )
    var result = dated.select_exprs([col("v").cum_max().over("g").alias("r")])
    assert_true(result.column("r").dtype() == DataType.DATE)


def test_window_expressions_over_one_key_share_its_layout() raises:
    """TPC-DS q51's shape: a running count and a running max over the same
    key in one select, the max over values filled with the partition's
    minimum (a nested over()), as decimal128."""
    var typed = frame().with_columns(
        [col("v").cast(DataType.decimal(38, 0, 128)).alias("v")]
    )
    var result = typed.select_exprs(
        [
            when(col("v").cum_count().over("g") > lit(Int64(0)))
            .then(
                col("v").fill_null(col("v").min().over("g")).cum_max().over("g")
            )
            .end()
            .alias("r"),
            col("v").cum_min().over("g").alias("low"),
        ]
    )
    check(result.column("low"), DataType.decimal(38, 0, 128), False, False)
    # The filled running max: a null row takes the partition's minimum,
    # which never raises it, so it shows the running max so far, and rows
    # before the partition's first value are null.
    var model = expected(True, False)
    var seen = List[Bool](length=GROUPS, fill=False)
    var best = List[Int64](length=GROUPS, fill=0)
    var numbers = result.column("r").cast(DataType.FLOAT64)
    for row in range(ROWS):
        var g = Int(group_of(row))
        if value_valid(row):
            seen[g] = True
            best[g] = model[0][row]
        var cell = numbers.get(row)
        assert_equal(cell.is_null(), not seen[g])
        if seen[g]:
            assert_equal(cell.float64(), Float64(best[g]))


def test_nested_over_on_other_keys_sees_its_whole_partition() raises:
    """A nested over() runs inside the outer one's runs of whole
    partitions only when it partitions by every outer key: by another key
    (`h` cuts across `g`) it reads rows of other runs; by more keys
    (`g`, `h`) each of its partitions lies within one of `g`'s."""
    var source = frame()
    var hs = List[Int64](capacity=ROWS)
    for row in range(ROWS):
        hs.append(Int64(row % 5))
    var typed = DataFrame(
        [
            source.column("v").copy(),
            source.column("g").copy(),
            Series("h", Column[Int64](hs^)),
        ]
    )
    var keys: List[List[String]] = [["h"], ["g", "h"]]
    for k in range(len(keys)):
        var filled = (
            col("v").fill_null(col("v").max().over(keys[k])).cum_sum().over("g")
        )
        var result = typed.select_exprs([filled.alias("r")])
        # The model: each partition's largest value, then running sums of
        # the filled values within `g` in row order.
        var parts = GROUPS * 5
        var top = List[Int64](length=parts, fill=0)
        var has = List[Bool](length=parts, fill=False)
        for row in range(ROWS):
            var p = row % 5 if k == 0 else Int(group_of(row)) * 5 + row % 5
            if value_valid(row) and (not has[p] or value_of(row) > top[p]):
                top[p] = value_of(row)
                has[p] = True
        var total = List[Int64](length=GROUPS, fill=0)
        var numbers = result.column("r")
        for row in range(ROWS):
            var g = Int(group_of(row))
            var p = row % 5 if k == 0 else g * 5 + row % 5
            var cell = numbers.get(row)
            if value_valid(row) or has[p]:
                total[g] += value_of(row) if value_valid(row) else top[p]
                assert_equal(cell.int64(), total[g])
            else:
                assert_true(cell.is_null())
