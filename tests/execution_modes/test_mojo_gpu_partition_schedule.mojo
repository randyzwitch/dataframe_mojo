"""Whole-group CPU OVER runs retain floating and checked decimal prefixes."""
# contract-group: execution.apple_parity
# contract-group: execution.backends

from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.categorical import decode
from dataframe.dtype import CategoricalDictionary
from dataframe.decimal import pow10
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_moments import near
from tests.support.lifecycle_jobs import WorkerSetting


def _skew_keys(hashed: Bool) -> List[UInt64]:
    # First-occurrence groups have sizes180000,10000,72147. Dense key IDs
    # therefore differ from GPU lexical order. For nondense keys, CPU's
    # eight hash buckets are respectively0,4,7, while lexical order is1,0,2.
    var labels: List[UInt64] = [2, 0, 1]
    if hashed:
        labels = [17592186044427, 1099511627777, 281474976710662]
    var keys = List[UInt64](capacity=262147)
    for group in range(3):
        var size = 180000 if group == 0 else 10000 if group == 1 else 72147
        for _ in range(size):
            keys.append(labels[group])
    return keys^


def _float_skew_frame(hashed: Bool) raises -> DataFrame:
    var values = List[Float64](length=262147, fill=0)
    # CPU's first run contains group0 only: two workers, step90112.
    # Former global workers split at73728. Incorrect lexical hash-key runs
    # put a10000-row group first, splitting this group instead at88304.
    var split = 88304 if hashed else 73728
    values[split - 1] = -1e16
    values[split] = 1e16
    values[split + 1] = 1
    return DataFrame(
        [
            Series("x", Column[Float64](values^)),
            Series("k", Column[UInt64](_skew_keys(hashed))),
        ]
    )


# requires: metal
def test_mojo_gpu_partition_schedule_skewed_float_runs() raises:
    var setting = WorkerSetting(4)
    for hashed in [False, True]:
        var frame = _float_skew_frame(hashed)
        var query = (
            frame.lazy()
            .select_exprs(
                [
                    col("x").sum().over("k").alias("sum"),
                    col("x").mean().over("k").alias("mean"),
                    col("x").var().over("k").alias("variance"),
                    col("x").std().over("k").alias("standard"),
                    col("x").skew().over("k").alias("skew"),
                    col("x").kurtosis().over("k").alias("kurtosis"),
                ]
            )
            .head(1)
        )
        var expected = query.collect(engine="cpu", streaming=False)
        # Independent cancellation anchors: serial(-1e16,+1e16,+1)=1.
        # Merging chunks split between the first two values instead gives0.
        assert_equal(expected.item(0, "sum").float64(), Float64(1))
        assert_equal(expected.item(0, "mean").float64(), Float64(1) / 180000)
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        assert_equal(actual.item(0, "sum").float64(), Float64(1))
        assert_equal(actual.item(0, "mean").float64(), Float64(1) / 180000)
        near(actual, expected)
    _ = setting^


# requires: metal
def test_mojo_gpu_partition_schedule_skewed_decimal_prefixes() raises:
    var setting = WorkerSetting(4)
    for hashed in [False, True]:
        var values = List[Int128](length=262147, fill=0)
        var amount = Int128(6) * pow10(37)
        var split = 88304 if hashed else 73728
        values[split - 1] = -amount
        values[split] = amount
        values[split + 1] = amount
        values[split + 2] = -amount
        var frame = DataFrame(
            [
                Series("x", Column[Int128](values^)).with_dtype(
                    DataType.decimal(38, 0)
                ),
                Series("k", Column[UInt64](_skew_keys(hashed))),
            ]
        )
        var query = (
            frame.lazy()
            .select_exprs(
                [
                    col("x").sum().over("k").alias("sum"),
                    col("x").mean().over("k").alias("mean"),
                ]
            )
            .head(1)
        )
        var expected = query.collect(engine="cpu", streaming=False)
        assert_equal(expected.column("sum")._decimal_raw(0), Int128(0))
        assert_equal(expected.item(0, "mean").float64(), Float64(0))
        # Every CPU worker sees safe prefixes[-A,0,A,0]. A worker split at
        # the cancellation pair would see[A,2A,A] and falsely report overflow.
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        assert_equal(actual.column("sum")._decimal_raw(0), Int128(0))
        assert_equal(actual.item(0, "mean").float64(), Float64(0))
        assert_true(actual.equals(expected))
    _ = setting^


# requires: metal
def test_mojo_gpu_partition_schedule_dictionary_provenance() raises:
    var setting = WorkerSetting(4)
    var dictionary = CategoricalDictionary()
    for name in ["m32", "z3", "a0"]:
        dictionary.append(name)
    var keys = _skew_keys(False)
    var codes = List[UInt32](capacity=len(keys))
    for code in keys:
        codes.append(UInt32(code))
    var categorical = Series("k", Column[UInt32](codes^)).with_dtype(
        DataType.categorical(dictionary^)
    )
    var frame = DataFrame(
        [_float_skew_frame(True).column("x"), categorical.copy()]
    )
    var raw = (
        frame.lazy()
        .select_exprs([col("x").sum().over("k").alias("sum")])
        .head(1)
    )
    var raw_cpu = raw.collect(engine="cpu", streaming=False)
    assert_equal(raw_cpu.item(0, "sum").float64(), Float64(1))
    assert_equal(
        raw.collect(accelerator=MetalRuntime()).item(0, "sum").float64(),
        Float64(1),
    )
    # An explicit COL of the categorical key in the ORIGINAL bound expression
    # causes CPU preparation to decode that source before OVER. Its physical
    # codes otherwise stay numeric. Logical string buckets for a0,m32,z3
    # are4,0,7: unlike dense codes they batch the small group before group0,
    # moving group0's worker boundary from90112 to88304, so this sum is0.
    var selected = (
        frame.lazy()
        .select_exprs(
            [
                when(col("k") == lit("a0"))
                .then(col("x").sum().over("k"))
                .otherwise(lit(Float64(0)))
                .alias("sum")
            ]
        )
        .head(1)
    )
    var selected_cpu = selected.collect(engine="cpu", streaming=False)
    assert_equal(selected_cpu.item(0, "sum").float64(), Float64(0))
    assert_equal(
        selected.collect(accelerator=MetalRuntime()).item(0, "sum").float64(),
        Float64(0),
    )
    var strings = DataFrame([frame.column("x"), decode(categorical)])
    var string_query = (
        strings.lazy()
        .select_exprs([col("x").sum().over("k").alias("sum")])
        .head(1)
    )
    var string_cpu = string_query.collect(engine="cpu", streaming=False)
    assert_equal(string_cpu.item(0, "sum").float64(), Float64(0))
    assert_equal(
        string_query.collect(accelerator=MetalRuntime())
        .item(0, "sum")
        .float64(),
        Float64(0),
    )
    _ = setting^
