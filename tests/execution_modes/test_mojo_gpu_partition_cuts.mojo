"""Public resident partition CUT/QCUT, literal outputs and CPU fault order."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, StringColumn, Expr, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.support.lifecycle_jobs import WorkerSetting


def _check(query: LazyFrame) raises:
    var expected = query.collect(engine="cpu", optimize=False, streaming=False)
    var actual = query.collect(accelerator=MetalRuntime(), optimize=False, streaming=False)
    same(actual, expected)


def test_public_partition_cut_qcut_resident_outputs() raises:
    _ = require_metal()
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000042))
    var frame = DataFrame(
        [
            Series("x", Column[Float64]([0, 10, 2, 14, 99, nan, 4, 8], [True, True, True, True, False, True, True, True])),
            Series("k", Column[Int32]([0, 1, 0, 1, 0, 1, 7, 7], [True, True, True, True, True, True, False, False])),
            Series("s", StringColumn(["a", "b", "a", "b", "a", "b", "n", "n"])),
        ]
    )
    var probabilities: List[Float64] = [0.5]
    var breaks: List[Float64] = [2.0]
    var custom: List[String] = ["low", "high"]
    for workers in [1, 4]:
        var setting = WorkerSetting(workers)
        for closed in [False, True]:
            for key in [String("k"), "s"]:
                _check(frame.lazy().select(col("x").cut(breaks, left_closed=closed).over(key)))
                _check(frame.lazy().select(col("x").qcut(probabilities, left_closed=closed).over(key)))
                _check(frame.lazy().select(col("x").cut(breaks, labels=custom, left_closed=closed).over(key)))
                _check(frame.lazy().select(col("x").qcut(probabilities, labels=custom, left_closed=closed).over(key)))
                _check(frame.lazy().select(col("x").qcut(probabilities, include_breaks=True, left_closed=closed).over(key)))
                _check(frame.lazy().select(col("x").cut(breaks, include_breaks=True, left_closed=closed).over(key)))
        _check(frame.lazy().select_exprs([
            col("x").qcut(probabilities).over("k").alias("first"),
            when(lit(False)).then(col("x").cut(breaks).over("s")).otherwise(lit("safe")).alias("second"),
            col("x").qcut(probabilities, labels=custom).over("s").alias("third"),
        ]))
        _check(frame.lazy().select(col("x").qcut(probabilities).over("k")).head(0))
        _check(frame.clear().lazy().select(col("x").qcut(probabilities).over("k")))
        _ = setting^
    var literal = frame.lazy().select(col("x").qcut(probabilities).over("k")).collect(accelerator=MetalRuntime(), optimize=False, streaming=False)
    assert_equal(literal.item(0, "x").string(), "(-inf, 1]")
    assert_equal(literal.item(1, "x").string(), "(-inf, 12]")
    assert_equal(literal.item(6, "x").string(), "(-inf, 6]")
    print("PASS public partition CUT/QCUT resident literal outputs")


def test_public_partition_cut_qcut_eager_fault_order() raises:
    _ = require_metal()
    var values = List[Int64](length=8192, fill=0)
    values[4096] = Int64.MAX
    var keys = List[Int32](length=8192, fill=0)
    for row in range(4096, 8192):
        keys[row] = 1
    var frame = DataFrame([Series("x", Column[Int64](values^)), Series("k", Column[Int32](keys^))])
    var specifications: List[Float64] = [0.0, 0.5]
    for workers in [1, 4]:
        var setting = WorkerSetting(workers)
        var state = (col("x") + lit(Int64(1))).qcut(specifications).over("k")
        var expected = "breaks must be unique" if workers >= 2 else "overflow"
        for query in [
            frame.lazy().select(state.copy()).head(0),
            frame.lazy().select(when(lit(False)).then(state.copy()).otherwise(lit("safe"))),
        ]:
            with assert_raises(contains=expected):
                _ = query.collect(engine="cpu", optimize=False, streaming=False)
            with assert_raises(contains=expected):
                _ = query.collect(accelerator=MetalRuntime(), optimize=False, streaming=False)
        _ = setting^
    print("PASS public partition CUT/QCUT eager state/run fault order")


def test_public_partition_cuts_original_categorical_keys() raises:
    from dataframe.dtype import CategoricalDictionary, DataType

    _ = require_metal()
    var dictionary = CategoricalDictionary()
    dictionary.append("same")
    dictionary.append("same")
    var dtype = DataType.categorical(dictionary^)
    var frame = DataFrame([
        Series("x", Column[Float64]([0, 10, 2, 14])),
        Series("k", Column[UInt32]([0, 1, 0, 1])).with_dtype(dtype),
    ])
    var probabilities: List[Float64] = [0.5]
    var query = frame.lazy().select(col("x").qcut(probabilities).over("k"))
    _check(query)
    var actual = query.collect(accelerator=MetalRuntime(), optimize=False, streaming=False)
    assert_equal(actual.item(0, "x").string(), "(-inf, 1]")
    assert_equal(actual.item(1, "x").string(), "(-inf, 12]")
    # A COL occurrence anywhere in this bound expression makes CPU evaluate
    # prepare the categorical key as decoded String before every OVER state.
    var decoded_query = frame.lazy().select(
        when(col("k").is_not_null()).then(col("x").qcut(probabilities).over("k"))
        .otherwise(lit("missing")).alias("x")
    )
    _check(decoded_query)
    var decoded = decoded_query.collect(accelerator=MetalRuntime(), optimize=False, streaming=False)
    assert_equal(decoded.item(0, "x").string(), "(-inf, 6]")
    assert_equal(decoded.item(1, "x").string(), "(6, inf]")
    print("PASS public partition original categorical codes and decoded key scope")
