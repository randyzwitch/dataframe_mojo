"""Public resident partition CUT/QCUT, literal outputs and CPU fault order."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    DataType,
    Series,
    Column,
    StringColumn,
    Expr,
    col,
    lit,
    when,
)
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.support.lifecycle_jobs import WorkerSetting


def _check(query: LazyFrame) raises:
    var expected = query.collect(engine="cpu", optimize=False, streaming=False)
    var actual = query.collect(
        accelerator=MetalRuntime(), optimize=False, streaming=False
    )
    same(actual, expected)


def test_public_partition_cut_qcut_resident_outputs() raises:
    _ = require_metal()
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000042))
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [0, 10, 2, 14, 99, nan, 4, 8],
                    [True, True, True, True, False, True, True, True],
                ),
            ),
            Series(
                "k",
                Column[Int32](
                    [0, 1, 0, 1, 0, 1, 7, 7],
                    [True, True, True, True, True, True, False, False],
                ),
            ),
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
                _check(
                    frame.lazy().select(
                        col("x").cut(breaks, left_closed=closed).over(key)
                    )
                )
                _check(
                    frame.lazy().select(
                        col("x")
                        .qcut(probabilities, left_closed=closed)
                        .over(key)
                    )
                )
                _check(
                    frame.lazy().select(
                        col("x")
                        .cut(breaks, labels=custom, left_closed=closed)
                        .over(key)
                    )
                )
                _check(
                    frame.lazy().select(
                        col("x")
                        .qcut(probabilities, labels=custom, left_closed=closed)
                        .over(key)
                    )
                )
                _check(
                    frame.lazy().select(
                        col("x")
                        .qcut(
                            probabilities,
                            include_breaks=True,
                            left_closed=closed,
                        )
                        .over(key)
                    )
                )
                _check(
                    frame.lazy().select(
                        col("x")
                        .cut(breaks, include_breaks=True, left_closed=closed)
                        .over(key)
                    )
                )
        _check(
            frame.lazy().select_exprs(
                [
                    col("x").qcut(probabilities).over("k").alias("first"),
                    when(lit(False))
                    .then(col("x").cut(breaks).over("s"))
                    .otherwise(lit("safe"))
                    .alias("second"),
                    col("x")
                    .qcut(probabilities, labels=custom)
                    .over("s")
                    .alias("third"),
                ]
            )
        )
        _check(
            frame.lazy().select(col("x").qcut(probabilities).over("k")).head(0)
        )
        _check(
            frame.clear().lazy().select(col("x").qcut(probabilities).over("k"))
        )
        _ = setting^
    var literal = (
        frame.lazy()
        .select(col("x").qcut(probabilities).over("k"))
        .collect(accelerator=MetalRuntime(), optimize=False, streaming=False)
    )
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
    var frame = DataFrame(
        [Series("x", Column[Int64](values^)), Series("k", Column[Int32](keys^))]
    )
    var specifications: List[Float64] = [0.0, 0.5]
    for workers in [1, 4]:
        var setting = WorkerSetting(workers)
        var state = (col("x") + lit(Int64(1))).qcut(specifications).over("k")
        var expected = "breaks must be unique" if workers >= 2 else "overflow"
        for query in [
            frame.lazy().select(state.copy()).head(0),
            frame.lazy().select(
                when(lit(False)).then(state.copy()).otherwise(lit("safe"))
            ),
        ]:
            with assert_raises(contains=expected):
                _ = query.collect(engine="cpu", optimize=False, streaming=False)
            with assert_raises(contains=expected):
                _ = query.collect(
                    accelerator=MetalRuntime(), optimize=False, streaming=False
                )
        _ = setting^
    print("PASS public partition CUT/QCUT eager state/run fault order")


def test_public_partition_cuts_original_categorical_keys() raises:
    from dataframe.dtype import CategoricalDictionary, DataType

    _ = require_metal()
    var dictionary = CategoricalDictionary()
    dictionary.append("same")
    dictionary.append("same")
    var dtype = DataType.categorical(dictionary^)
    var frame = DataFrame(
        [
            Series("x", Column[Float64]([0, 10, 2, 14])),
            Series("k", Column[UInt32]([0, 1, 0, 1])).with_dtype(dtype),
        ]
    )
    var probabilities: List[Float64] = [0.5]
    var query = frame.lazy().select(col("x").qcut(probabilities).over("k"))
    _check(query)
    var actual = query.collect(
        accelerator=MetalRuntime(), optimize=False, streaming=False
    )
    assert_equal(actual.item(0, "x").string(), "(-inf, 1]")
    assert_equal(actual.item(1, "x").string(), "(-inf, 12]")
    # A COL occurrence anywhere in this bound expression makes CPU evaluate
    # prepare the categorical key as decoded String before every OVER state.
    var decoded_query = frame.lazy().select(
        when(col("k").is_not_null())
        .then(col("x").qcut(probabilities).over("k"))
        .otherwise(lit("missing"))
        .alias("x")
    )
    _check(decoded_query)
    var decoded = decoded_query.collect(
        accelerator=MetalRuntime(), optimize=False, streaming=False
    )
    assert_equal(decoded.item(0, "x").string(), "(-inf, 6]")
    assert_equal(decoded.item(1, "x").string(), "(6, inf]")
    print(
        "PASS public partition original categorical codes and decoded key scope"
    )


def test_public_partition_cuts_nested_numeric_scopes() raises:
    _ = require_metal()
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([0, 10, 2, 14])),
            Series("k", Column[Int32]([0, 1, 0, 1])),
            Series("h", Column[Int32]([1, 1, 0, 0])),
        ]
    )
    var probabilities: List[Float64] = [0.5]
    var inner = (col("x") + lit(Int64(1))).over("h")
    var nested = inner.copy().qcut(probabilities).over("k")
    _check(frame.lazy().select(nested.copy()))
    _check(
        frame.lazy().select_exprs(
            [
                col("x").qcut(probabilities).over("k").alias("prior"),
                nested.copy().alias("nested"),
                (inner.copy() + lit(Int64(2)))
                .over("k")
                .qcut(probabilities)
                .over("h")
                .alias("deep"),
            ]
        )
    )
    _check(frame.lazy().with_columns(nested.alias("nested")))
    var actual = (
        frame.lazy()
        .select(nested)
        .collect(accelerator=MetalRuntime(), optimize=False, streaming=False)
    )
    assert_equal(actual.item(0, "x").string(), "(-inf, 2]")
    assert_equal(actual.item(1, "x").string(), "(-inf, 13]")
    var values = List[Int64](length=8192, fill=0)
    values[4096] = Int64.MAX
    var keys = List[Int32](length=8192, fill=0)
    for row in range(4096, 8192):
        keys[row] = 1
    var faults = DataFrame(
        [Series("x", Column[Int64](values^)), Series("k", Column[Int32](keys^))]
    )
    for workers in [1, 4]:
        var setting = WorkerSetting(workers)
        # The nested numeric state fails before the outer qcut observes its
        # duplicate edges, regardless of the outer fallback's group order.
        var state = (
            (col("x") + lit(Int64(1))).over("k").qcut([0.0, 0.5]).over("k")
        )
        for query in [
            faults.lazy().select(state.copy()).head(0),
            faults.lazy().select(
                when(lit(False)).then(state.copy()).otherwise(lit("safe"))
            ),
        ]:
            with assert_raises(contains="overflow"):
                _ = query.collect(engine="cpu", optimize=False, streaming=False)
            with assert_raises(contains="overflow"):
                _ = query.collect(
                    accelerator=MetalRuntime(), optimize=False, streaming=False
                )
        _ = setting^
    print("PASS public partition nested numeric states and cache snapshots")


def _literal_fault(
    query: LazyFrame, cpu_message: String, gpu_message: String
) raises:
    with assert_raises(contains=cpu_message):
        _ = query.collect(engine="cpu", optimize=False, streaming=False)
    with assert_raises(contains=gpu_message):
        _ = query.collect(
            accelerator=MetalRuntime(), optimize=False, streaming=False
        )


def test_public_partition_when_head_and_sibling_error_contracts() raises:
    _ = require_metal()
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([Int64.MAX, 1])),
            Series("k", Column[Int32]([0, 0])),
        ]
    )
    var probabilities: List[Float64] = [0.5]
    var custom: List[String] = ["low", "high"]
    var safe_state = col("x").qcut(probabilities, labels=custom).over("k")
    var invalid_state = col("x").qcut([2.0]).over("k")
    var overflow = col("x") + lit(Int64(1))
    # Every eager window/OVER state precedes final row masks. The checked row
    # expression in an unselected branch is nevertheless suppressed.
    var suppressed = frame.lazy().select(
        when(lit(False))
        .then(overflow.copy() > lit(Int64(0)))
        .otherwise(safe_state.copy() == lit("low"))
        .alias("wanted")
    )
    _check(suppressed)
    var result = suppressed.collect(
        accelerator=MetalRuntime(), optimize=False, streaming=False
    )
    assert_equal(result.item(0, "wanted").bool(), False)
    assert_equal(result.item(1, "wanted").bool(), True)
    var eager = frame.lazy().select(
        when(lit(True))
        .then(overflow.copy() > lit(Int64(0)))
        .otherwise(invalid_state.copy() == lit("low"))
        .alias("wanted")
    )
    _literal_fault(
        eager,
        "quantiles must be between 0 and 1",
        "quantiles must be between 0 and 1",
    )
    _literal_fault(
        eager.head(0),
        "quantiles must be between 0 and 1",
        "quantiles must be between 0 and 1",
    )
    # Projection siblings bind together, then finish in original sibling order.
    var row_first = frame.lazy().select_exprs(
        [
            overflow.copy().alias("row"),
            invalid_state.copy().alias("state"),
        ]
    )
    var state_first = frame.lazy().select_exprs(
        [
            invalid_state.copy().alias("state"),
            overflow.copy().alias("row"),
        ]
    )
    _literal_fault(row_first.head(0), "overflow", "overflow")
    _literal_fault(
        state_first.head(0),
        "quantiles must be between 0 and 1",
        "quantiles must be between 0 and 1",
    )
    var invalid_binding = frame.lazy().select_exprs(
        [
            overflow.alias("row"),
            col("missing").qcut(probabilities).over("k").alias("state"),
        ]
    )
    _literal_fault(invalid_binding.head(0), "missing", "missing")
    # CUT/QCUT themselves are eager windows in expr.is_window. Their original
    # AST order competes with OVER states even beneath opposite WHEN arms.
    var finite = DataFrame(
        [
            Series("x", Column[Float64]([0, 2])),
            Series("k", Column[Int32]([0, 0])),
        ]
    )
    var bad_breaks: List[Float64] = [2.0, 1.0]
    var bad_probabilities: List[Float64] = [2.0]
    var cut_first = finite.lazy().select(
        when(lit(True))
        .then(col("x").cut(bad_breaks))
        .otherwise(col("x").qcut(bad_probabilities).over("k"))
    )
    var over_first = finite.lazy().select(
        when(lit(False))
        .then(col("x").qcut(bad_probabilities).over("k"))
        .otherwise(col("x").cut(bad_breaks))
    )
    _literal_fault(
        cut_first.head(0), "breaks must be unique", "breaks must be unique"
    )
    _literal_fault(
        over_first.head(0),
        "quantiles must be between 0 and 1",
        "quantiles must be between 0 and 1",
    )
    # Original scalar checked occurrences survive public dictionary/literal
    # normalization and retain their CPU child-first order.
    var cast = lit(Int64(128)).cast(DataType.INT8).cast(DataType.INT64)
    var power = lit(Int64(2)).pow(lit(Int64(-1)))
    var cast_first = finite.lazy().select(
        (col("x").cast(DataType.INT64) + (cast.copy() + power.copy()))
        .qcut(bad_probabilities)
        .over("k")
    )
    var power_first = finite.lazy().select(
        (col("x").cast(DataType.INT64) + (power.copy() + cast.copy()))
        .qcut(bad_probabilities)
        .over("k")
    )
    _literal_fault(
        cast_first.head(0),
        "cast from int64 to int8 failed",
        "strict cast failed",
    )
    _literal_fault(
        power_first.head(0), "nonnegative exponent", "nonnegative exponent"
    )
    # Constructor-owned literal metadata is unaffected by later caller writes.
    var retained_probabilities: List[Float64] = [0.5]
    var retained_labels: List[String] = ["low", "high"]
    var retained = (
        col("x").qcut(retained_probabilities, labels=retained_labels).over("k")
    )
    retained_probabilities[0] = 2.0
    retained_labels[0] = "changed"
    var immutable = (
        finite.lazy()
        .select(retained)
        .collect(accelerator=MetalRuntime(), optimize=False, streaming=False)
    )
    assert_equal(immutable.item(0, "x").string(), "low")
    assert_equal(immutable.item(1, "x").string(), "high")
    print(
        "PASS public WHEN masks, eager/head/sibling order and immutable controls"
    )


def test_public_partition_qcut_signed_zero_breakpoint_literals() raises:
    _ = require_metal()
    var negative_zero = bitcast[DType.float64](UInt64(0x8000000000000000))
    var frame = DataFrame(
        [
            Series("x", Column[Float64]([negative_zero, 0, negative_zero, 0])),
            Series("k", Column[Int32]([0, 1, 0, 1])),
        ]
    )
    var probabilities: List[Float64] = [0.0]
    var query = frame.lazy().select(
        col("x").qcut(probabilities, include_breaks=True).over("k")
    )
    var cpu = query.collect(engine="cpu", optimize=False, streaming=False)
    var gpu = query.collect(
        accelerator=MetalRuntime(), optimize=False, streaming=False
    )
    var expected: List[UInt64] = [0x8000000000000000, 0, 0x8000000000000000, 0]
    for row in range(4):
        assert_equal(
            bitcast[DType.uint64](
                cpu.item(row, "x").struct_field("breakpoint").float64()
            ),
            expected[row],
        )
        assert_equal(
            bitcast[DType.uint64](
                gpu.item(row, "x").struct_field("breakpoint").float64()
            ),
            expected[row],
        )
        assert_equal(
            cpu.item(row, "x").struct_field("category").string(), "(-inf, 0]"
        )
        assert_equal(
            gpu.item(row, "x").struct_field("category").string(), "(-inf, 0]"
        )
    print("PASS public partition qcut literal signed-zero breakpoint words")
