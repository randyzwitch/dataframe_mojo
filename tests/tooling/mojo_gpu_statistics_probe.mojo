"""Exercise statistical kernels with shared buffers, masks, errors and stride."""

from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.testing import assert_equal
from dataframe.expr import COUNT, VAR, STD, SKEW, KURTOSIS, COV, CORR
from dataframe.accelerator.gpu.statistics import (
    State,
    supported,
    empty_result,
    reduce_partials,
    reduce_finish,
)


def check(n: Int, groups: Int = 7, configured: Int = 11) raises:
    var capacity = n + 5
    var ops: List[Int64] = [
        COUNT,
        VAR,
        VAR,
        STD,
        SKEW,
        SKEW,
        KURTOSIS,
        KURTOSIS,
        KURTOSIS,
        KURTOSIS,
        COV,
        COV,
        COV,
        CORR,
        COUNT,
    ]
    var parameters: List[Int64] = [
        0,
        0,
        1,
        1,
        1,
        0,
        3,
        1,
        2,
        0,
        0,
        1,
        Int64(Int32.MAX),
        0,
        0,
    ]
    var descriptors = List[Int64]()
    for column in range(len(ops)):
        descriptors.append(2)
        descriptors.append(11)
        descriptors.append(11)
        descriptors.append(ops[column])
        # Right input slot zero; worker metadata must not alter ddof/flags.
        descriptors.append(parameters[column] | (Int64(configured) << 47))
    var words = List[UInt64](length=capacity * 3, fill=0xDEADBEEF)
    var validity = List[UInt8](length=capacity * 3, fill=0)
    var errors = List[UInt32](length=capacity * 3, fill=0)
    for row in range(n):
        words[2 * capacity + row] = bitcast[DType.uint64](
            Float64(row % 17 + 1) / 4
        )
        words[row] = bitcast[DType.uint64](
            Float64(row % 17 + 1) / 2 + Float64(row % 3) / 8
        )
        validity[2 * capacity + row] = UInt8(row % 7 != 0)
        validity[row] = UInt8(row % 5 != 1)
    if n > 3:
        errors[2 * capacity + 3] = 8
    if n > 5:
        errors[5] = 16
    var result = List[UInt64](length=len(ops), fill=0xABC)
    var masks = List[UInt8](length=len(ops), fill=90)
    var failures = List[UInt32](length=len(ops), fill=32)
    var partials = List[UInt64](length=len(ops) * groups * 8, fill=0xDEF)
    var ctx = DeviceContext()
    var dwords = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var dvalid = ctx.enqueue_create_buffer[DType.uint8](len(validity))
    var derrors = ctx.enqueue_create_buffer[DType.uint32](len(errors))
    var dcode = ctx.enqueue_create_buffer[DType.int64](len(descriptors))
    var dpartials = ctx.enqueue_create_buffer[DType.uint64](len(partials))
    var dresult = ctx.enqueue_create_buffer[DType.uint64](len(result))
    var dmasks = ctx.enqueue_create_buffer[DType.uint8](len(masks))
    var dfailures = ctx.enqueue_create_buffer[DType.uint32](len(failures))
    ctx.enqueue_copy(dwords, words.unsafe_ptr())
    ctx.enqueue_copy(dvalid, validity.unsafe_ptr())
    ctx.enqueue_copy(derrors, errors.unsafe_ptr())
    ctx.enqueue_copy(dcode, descriptors.unsafe_ptr())
    ctx.enqueue_copy(dpartials, partials.unsafe_ptr())
    ctx.enqueue_copy(dresult, result.unsafe_ptr())
    ctx.enqueue_copy(dmasks, masks.unsafe_ptr())
    ctx.enqueue_copy(dfailures, failures.unsafe_ptr())
    ctx.enqueue_function[reduce_partials](
        dwords.unsafe_ptr(),
        dvalid.unsafe_ptr(),
        derrors.unsafe_ptr(),
        dcode.unsafe_ptr(),
        dpartials.unsafe_ptr(),
        UInt32(len(ops)),
        UInt32(groups),
        UInt32(n),
        UInt32(capacity),
        grid_dim=1,
        block_dim=256,
    )
    ctx.enqueue_function[reduce_finish](
        dpartials.unsafe_ptr(),
        dcode.unsafe_ptr(),
        dresult.unsafe_ptr(),
        dmasks.unsafe_ptr(),
        dfailures.unsafe_ptr(),
        UInt32(len(ops)),
        UInt32(groups),
        UInt32(n),
        grid_dim=1,
        block_dim=256,
    )
    ctx.enqueue_copy(result.unsafe_ptr(), dresult)
    ctx.enqueue_copy(masks.unsafe_ptr(), dmasks)
    ctx.enqueue_copy(failures.unsafe_ptr(), dfailures)
    ctx.enqueue_copy(partials.unsafe_ptr(), dpartials)
    ctx.synchronize()
    for column in range(len(ops)):
        var op = ops[column]
        if not supported(op):
            assert_equal(result[column], UInt64(0xABC))
            assert_equal(masks[column], UInt8(90))
            assert_equal(failures[column], UInt32(32))
            for i in range(groups * 8):
                assert_equal(partials[column * groups * 8 + i], UInt64(0xDEF))
            continue
        var state = State()
        var workers = max(1, min(configured, n // 65536))
        var step = ((n + workers - 1) // workers + 8191) // 8192 * 8192
        for worker in range(workers):
            var partial = State()
            for row in range(
                min(n, worker * step), min(n, (worker + 1) * step)
            ):
                var valid = validity[2 * capacity + row] != 0
                var error = errors[2 * capacity + row]
                if op == COV or op == CORR:
                    valid &= validity[row] != 0
                    error |= errors[row]
                partial.add(
                    words[2 * capacity + row], words[row], valid, error, op
                )
            state.merge(partial, op)
        var expected = state.finish(op, descriptors[column * 5 + 4])
        assert_equal(result[column], expected[0])
        assert_equal(masks[column], expected[1])
        assert_equal(failures[column], expected[2])
        if n == 0:
            var empty = empty_result(op, descriptors[column * 5 + 4])
            assert_equal(expected[0], empty[0])
            assert_equal(expected[1], empty[1])
            assert_equal(expected[2], empty[2])


def main() raises:
    check(0)
    check(1)
    check(257)
    # Two statistical workers, including the one-partial fallback path.
    check(131073, 7, 2)
    check(131073, 1, 2)
    print(
        "PASS Mojo statistics kernels: empty/singleton/masked, errors, mixed descriptors, padded stride"
    )
