"""Device order statistics against host words and independent CPU selection."""

from max.gpu.host import DeviceContext
from std.math import isnan, isfinite
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe.aggregate import quantile_of
from dataframe.expr import COUNT
from dataframe.accelerator.rows import ROW_QUANTILE_LINEAR
from dataframe.accelerator.gpu import quantiles as q

comptime CASES = 12
comptime ROWS = 257
comptime STRIDE = ROWS + 5
comptime QUANTILES = 25
comptime OUTPUTS = CASES * QUANTILES + 1


def expected(
    words: List[UInt64],
    mask: List[UInt8],
    slot: Int,
    probability: UInt64,
    op: Int64,
) -> Tuple[UInt64, UInt8]:
    var data = words.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin]()
    return q.reduce[False](
        data,
        mask.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        data.unsafe_bitcast[UInt32](),
        slot * STRIDE,
        0,
        ROWS,
        probability,
        op,
    )


def main() raises:
    var probabilities: List[Float64] = [0, 0.25, 0.37, 0.5, 1]
    var methods: List[String] = [
        "linear",
        "lower",
        "higher",
        "nearest",
        "midpoint",
    ]
    var words = List[UInt64](length=CASES * STRIDE, fill=0xBAD)
    var mask = List[UInt8](length=CASES * STRIDE, fill=0)
    var errors = List[UInt32](length=CASES * STRIDE, fill=0)
    var samples = List[List[Float64]]()
    for dataset in range(CASES):
        var values = List[Float64]()
        var count = dataset if dataset < 5 else ROWS
        for row in range(count):
            var word = bitcast[DType.uint64](Float64((row * 17) % 37 - 18) / 8)
            if dataset == 1:
                word = bitcast[DType.uint64](Float64(7))
            elif dataset == 2:
                word = bitcast[DType.uint64](Float64(row * 8))
            elif dataset == 4:
                word = UInt64(0x7FF8000000000011) + UInt64(row)
            elif dataset == 5:
                word = UInt64(row % 2) << 63
            elif dataset == 6:
                word = UInt64(row + 1)
            elif dataset == 7 and row % 5 == 0:
                word = 0x7FF0000000000000
            elif dataset == 8 and row % 5 == 0:
                word = 0xFFF0000000000000
            elif dataset == 9 and row % 5 == 0:
                word = UInt64(0xFFF8000000000011) + UInt64(row)
            elif dataset == 10:
                word = bitcast[DType.uint64](
                    Float64(1e12) + Float64(row % 7) / 8
                )
            var valid = dataset < 7 or row % 7 != 0
            words[dataset * STRIDE + row] = word
            mask[dataset * STRIDE + row] = UInt8(valid)
            if valid:
                values.append(bitcast[DType.float64](word))
        samples.append(values^)
    errors[7 * STRIDE] = 8  # Invalid input errors are still observed.
    var descriptors = List[Int64]()
    for dataset in range(CASES):
        for p in range(len(probabilities)):
            for method in range(len(methods)):
                descriptors.extend(
                    [
                        Int64(dataset),
                        11,
                        11,
                        Int64(ROW_QUANTILE_LINEAR + method),
                        bitcast[DType.int64](probabilities[p]),
                    ]
                )
    descriptors.extend([0, 11, 3, COUNT, 0])
    var result = List[UInt64](length=OUTPUTS, fill=0xABC)
    var valid = List[UInt8](length=OUTPUTS, fill=90)
    var failed = List[UInt32](length=OUTPUTS, fill=32)
    var partials = List[UInt64](length=OUTPUTS * 3 * 8, fill=0xDEF)
    var ctx = DeviceContext()
    var dw = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var dv = ctx.enqueue_create_buffer[DType.uint8](len(mask))
    var de = ctx.enqueue_create_buffer[DType.uint32](len(errors))
    var dd = ctx.enqueue_create_buffer[DType.int64](len(descriptors))
    var dp = ctx.enqueue_create_buffer[DType.uint64](len(partials))
    var dr = ctx.enqueue_create_buffer[DType.uint64](len(result))
    var dm = ctx.enqueue_create_buffer[DType.uint8](len(valid))
    var df = ctx.enqueue_create_buffer[DType.uint32](len(failed))
    ctx.enqueue_copy(dw, words.unsafe_ptr())
    ctx.enqueue_copy(dv, mask.unsafe_ptr())
    ctx.enqueue_copy(de, errors.unsafe_ptr())
    ctx.enqueue_copy(dd, descriptors.unsafe_ptr())
    ctx.enqueue_copy(dp, partials.unsafe_ptr())
    ctx.enqueue_copy(dr, result.unsafe_ptr())
    ctx.enqueue_copy(dm, valid.unsafe_ptr())
    ctx.enqueue_copy(df, failed.unsafe_ptr())
    ctx.enqueue_function[q.reduce_partials](
        dw.unsafe_ptr(),
        dv.unsafe_ptr(),
        de.unsafe_ptr(),
        dd.unsafe_ptr(),
        dp.unsafe_ptr(),
        UInt32(OUTPUTS),
        UInt32(3),
        UInt32(ROWS),
        UInt32(STRIDE),
        grid_dim=(OUTPUTS * 3 + 127) // 128,
        block_dim=128,
    )
    ctx.enqueue_function[q.reduce_finish](
        dp.unsafe_ptr(),
        dd.unsafe_ptr(),
        dr.unsafe_ptr(),
        dm.unsafe_ptr(),
        df.unsafe_ptr(),
        UInt32(OUTPUTS),
        UInt32(3),
        UInt32(ROWS),
        grid_dim=(OUTPUTS + 127) // 128,
        block_dim=128,
    )
    ctx.enqueue_copy(result.unsafe_ptr(), dr)
    ctx.enqueue_copy(valid.unsafe_ptr(), dm)
    ctx.enqueue_copy(failed.unsafe_ptr(), df)
    ctx.enqueue_copy(partials.unsafe_ptr(), dp)
    ctx.synchronize()
    for dataset in range(CASES):
        for p in range(len(probabilities)):
            for method in range(len(methods)):
                var output = dataset * QUANTILES + p * len(methods) + method
                var host = expected(
                    words,
                    mask,
                    dataset,
                    bitcast[DType.uint64](probabilities[p]),
                    Int64(ROW_QUANTILE_LINEAR + method),
                )
                assert_equal(result[output], host[0])
                assert_equal(valid[output], host[1])
                assert_equal(failed[output], UInt32(8 if dataset == 7 else 0))
                var cpu = quantile_of(
                    samples[dataset].copy(), probabilities[p], methods[method]
                )
                assert_equal(valid[output] != 0, cpu.__bool__())
                if cpu:
                    var actual = bitcast[DType.float64](result[output])
                    var target = cpu.value()
                    if actual == target or (isnan(actual) and isnan(target)):
                        pass
                    else:
                        assert_true(isfinite(actual) and isfinite(target))
                        assert_true(
                            abs(actual - target)
                            <= 1e-12 * max(Float64(1), abs(target))
                        )
                for field in range(8, 24):
                    assert_equal(partials[output * 24 + field], UInt64(0xDEF))
    assert_equal(result[OUTPUTS - 1], UInt64(0xABC))
    assert_equal(valid[OUTPUTS - 1], UInt8(90))
    assert_equal(failed[OUTPUTS - 1], UInt32(32))
    # Independent anchors: two samples [0,8], zero canonicalization, NaNs.
    assert_equal(
        bitcast[DType.float64](result[2 * QUANTILES + 3 * 5]), Float64(4)
    )
    for field in range(QUANTILES):
        assert_equal(result[5 * QUANTILES + field], UInt64(0))
        assert_equal(result[4 * QUANTILES + field], UInt64(0x7FF8000000000000))
    print("PASS Mojo quantile host/device/CPU cases:", CASES * QUANTILES)
