"""Float32 floor division preserves its rounded quotient before flooring."""
# contract-group: execution.backends
from std.memory import bitcast
from tests.execution_modes.test_backend_metal import require_metal
from std.testing import assert_equal, assert_true
from dataframe import DataFrame, Series, Column, col, lit
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_float64_emulation import same


# requires: metal
def test_native_metal_float32_floor_division_bits_nulls_children_and_cost() raises:
    _ = require_metal()
    var words: List[UInt32] = [
        0,
        1,
        2,
        0x007FFFFE,
        0x007FFFFF,
        0x00800000,
        0x00800001,
        0x3DCCCCCD,
        0x3E99999A,
        0x3F000000,
        0x3F7FFFFF,
        0x3F800000,
        0x3F800001,
        0x40000000,
        0x40400000,
        0x7F7FFFFF,
        0x7F800000,
        0x7FC00042,
        0x7F800042,
    ]
    var count = len(words)
    for i in range(count):
        words.append(words[i] | UInt32(0x80000000))
    var left = List[Float32]()
    var right = List[Float32]()
    var left_valid = List[Bool]()
    var right_valid = List[Bool]()
    for a in words:
        for b in words:
            left.append(bitcast[DType.float32](a))
            right.append(bitcast[DType.float32](b))
            left_valid.append(len(left) % 7 != 0)
            right_valid.append(len(left) % 11 != 0)
    var frame = DataFrame(
        [
            Series("a", Column[Float32](left^, left_valid^)),
            Series("b", Column[Float32](right^, right_valid^)),
        ]
    )
    var runtime = MetalRuntime()
    var expression = col("a") // col("b")
    var query = frame.lazy().select(expression.alias("quotient"))
    assert_true(";mojo-metal-v1;ops=" in runtime.estimate_auto(query).family)
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    var child = (
        frame.lazy().with_columns(expression.alias("quotient")).slice(3, 17)
    )
    same(child.collect(accelerator=runtime), child.collect(engine="cpu"))
    var filtered = (
        frame.lazy().filter(expression.is_finite()).select(expression).head(31)
    )
    same(filtered.collect(accelerator=runtime), filtered.collect(engine="cpu"))
    var empty = frame.slice(0, 0).lazy().select(expression)
    same(empty.collect(accelerator=runtime), empty.collect(engine="cpu"))
    var rounded = (
        DataFrame(
            [
                Series("a", Column[Float32]([1, 3])),
                Series("b", Column[Float32]([0.1, 0.3])),
            ]
        )
        .lazy()
        .select((col("a") // col("b")).alias("quotient"))
    )
    var result = rounded.collect(accelerator=runtime)
    same(result, rounded.collect(engine="cpu"))
    assert_equal(result.column("quotient").float32()._get(0), Float32(10))
    assert_equal(result.column("quotient").float32()._get(1), Float32(10))
    same(
        query.collect(engine="auto", accelerator=runtime),
        query.collect(engine="cpu"),
    )
