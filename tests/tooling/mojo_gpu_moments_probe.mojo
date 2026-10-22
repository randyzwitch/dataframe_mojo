"""Validate statistical word states on host and Apple GPU against CPU states.

Run with: pixi run -e gpu mojo -I . tests/tooling/mojo_gpu_moments_probe.mojo
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import abs, isfinite, isnan, sqrt
from std.memory import Pointer, bitcast
from std.testing import assert_equal, assert_true
from dataframe_metal.accelerator.gpu import floating as f
from dataframe_metal.accelerator.gpu.moments import (
    VarianceState,
    CentralMomentState,
    CovarianceState,
    RollingVarianceState,
)
from dataframe.reductions import VarState, MomentState, CoMomentState

comptime CASES = 128
comptime LENGTH = 17
comptime INPUT = 1 + LENGTH * 3
comptime OUTPUT = 72


def evaluate_row(
    inputs: Pointer[UInt64, ImmutAnyOrigin],
    outputs: Pointer[UInt64, MutAnyOrigin],
    row: Int,
):
    var n = Int(inputs[unsafe_offset=row * INPUT])
    for mode in range(2):
        var variance = VarianceState()
        var central = CentralMomentState()
        var covariance = CovarianceState()
        var right_v = VarianceState()
        var right_c = CentralMomentState()
        var right_p = CovarianceState()
        for i in range(n):
            var at = row * INPUT + 1 + i * 3
            var x = inputs[unsafe_offset=at]
            var y = inputs[unsafe_offset=at + 1]
            var mask = inputs[unsafe_offset=at + 2]
            if mode == 1 and i >= n // 2:
                if mask & 1 != 0:
                    right_v.add(x)
                    right_c.add(x)
                if mask == 3:
                    right_p.add(x, y)
            else:
                if mask & 1 != 0:
                    variance.add(x)
                    central.add(x)
                if mask == 3:
                    covariance.add(x, y)
        # Also exercise the empty-state identity in sequential mode.
        variance.merge(right_v)
        central.merge(right_c)
        covariance.merge(right_p)
        var at = row * OUTPUT + mode * 30
        outputs[unsafe_offset=at] = variance.count
        outputs[unsafe_offset=at + 1] = variance.mean
        outputs[unsafe_offset=at + 2] = variance.m2
        outputs[unsafe_offset=at + 3] = UInt64(variance.valid(0))
        outputs[unsafe_offset=at + 4] = variance.finish(0) if variance.valid(
            0
        ) else UInt64(0)
        outputs[unsafe_offset=at + 5] = variance.finish(
            1, True
        ) if variance.valid(1) else UInt64(0)
        outputs[unsafe_offset=at + 6] = central.count
        outputs[unsafe_offset=at + 7] = central.mean
        outputs[unsafe_offset=at + 8] = central.m2
        outputs[unsafe_offset=at + 9] = central.m3
        outputs[unsafe_offset=at + 10] = central.m4
        outputs[unsafe_offset=at + 11] = UInt64(central.valid(False, 1))
        outputs[unsafe_offset=at + 12] = UInt64(central.valid(False, 0))
        outputs[unsafe_offset=at + 13] = UInt64(central.valid(True, 1))
        outputs[unsafe_offset=at + 14] = UInt64(central.valid(True, 0))
        outputs[unsafe_offset=at + 15] = central.finish(
            False, 1
        ) if central.valid(False, 1) else UInt64(0)
        outputs[unsafe_offset=at + 16] = central.finish(
            False, 0
        ) if central.valid(False, 0) else UInt64(0)
        outputs[unsafe_offset=at + 17] = central.finish(
            True, 3
        ) if central.valid(True, 3) else UInt64(0)
        outputs[unsafe_offset=at + 18] = central.finish(
            True, 2
        ) if central.valid(True, 2) else UInt64(0)
        outputs[unsafe_offset=at + 19] = central.finish(
            True, 0
        ) if central.valid(True, 0) else UInt64(0)
        outputs[unsafe_offset=at + 20] = covariance.count
        outputs[unsafe_offset=at + 21] = covariance.mean_x
        outputs[unsafe_offset=at + 22] = covariance.mean_y
        outputs[unsafe_offset=at + 23] = covariance.cxy
        outputs[unsafe_offset=at + 24] = covariance.m2x
        outputs[unsafe_offset=at + 25] = covariance.m2y
        outputs[unsafe_offset=at + 26] = UInt64(covariance.valid())
        outputs[unsafe_offset=at + 27] = covariance.finish(
            0
        ) if covariance.valid() else UInt64(0)
        outputs[unsafe_offset=at + 28] = covariance.finish(
            1
        ) if covariance.valid() else UInt64(0)
        outputs[unsafe_offset=at + 29] = covariance.finish(0, True)

    var rolling = RollingVarianceState()
    var origin = UInt64(0)
    for i in range(n):
        var at = row * INPUT + 1 + i * 3
        var x = inputs[unsafe_offset=at]
        if inputs[unsafe_offset=at + 2] & 1 != 0 and f.absolute(x) < f.INF:
            origin = x
            break
    for i in range(n):
        var at = row * INPUT + 1 + i * 3
        if inputs[unsafe_offset=at + 2] & 1 != 0:
            rolling.add(inputs[unsafe_offset=at], origin)
    var at = row * OUTPUT + 60
    outputs[unsafe_offset=at] = rolling.count
    outputs[unsafe_offset=at + 1] = rolling.bad
    outputs[unsafe_offset=at + 2] = rolling.mean
    outputs[unsafe_offset=at + 3] = rolling.m2
    outputs[unsafe_offset=at + 4] = UInt64(rolling.valid(3, 1))
    outputs[unsafe_offset=at + 5] = rolling.finish(1) if rolling.valid(
        3, 1
    ) else UInt64(0)
    var rebuild = False
    for i in range(n):
        var source = row * INPUT + 1 + i * 3
        if inputs[unsafe_offset=source + 2] & 1 != 0:
            rebuild = rolling.remove(inputs[unsafe_offset=source], origin)
            break
    outputs[unsafe_offset=at + 6] = rolling.count
    outputs[unsafe_offset=at + 7] = rolling.bad
    outputs[unsafe_offset=at + 8] = rolling.mean
    outputs[unsafe_offset=at + 9] = rolling.m2
    outputs[unsafe_offset=at + 10] = UInt64(rebuild)
    outputs[unsafe_offset=at + 11] = rolling.finish(0, True) if rolling.valid(
        1, 0
    ) else UInt64(0)


def evaluate(
    inputs: Pointer[UInt64, ImmutAnyOrigin],
    outputs: Pointer[UInt64, MutAnyOrigin],
):
    var row = global_idx.x
    if row < CASES:
        evaluate_row(inputs, outputs, row)


def close(actual: UInt64, expected: Float64) raises:
    var value = bitcast[DType.float64](actual)
    if isnan(expected):
        assert_true(isnan(value))
    elif not isfinite(expected):
        assert_equal(value, expected)
    else:
        assert_true(
            abs(value - expected) <= 1e-12 * max(abs(expected), Float64(1))
        )


def check_cpu(inputs: List[UInt64], results: List[UInt64]) raises:
    for row in range(CASES):
        var n = Int(inputs[row * INPUT])
        for mode in range(2):
            var v = VarState()
            var c = MomentState()
            var p = CoMomentState()
            var rv = VarState()
            var rc = MomentState()
            var rp = CoMomentState()
            for i in range(n):
                var at = row * INPUT + 1 + i * 3
                var x = bitcast[DType.float64](inputs[at])
                var y = bitcast[DType.float64](inputs[at + 1])
                var mask = inputs[at + 2]
                if mode == 1 and i >= n // 2:
                    if mask & 1 != 0:
                        rv.add(x)
                        rc.add(x)
                    if mask == 3:
                        rp.add(x, y)
                else:
                    if mask & 1 != 0:
                        v.add(x)
                        c.add(x)
                    if mask == 3:
                        p.add(x, y)
            v.merge(rv)
            c.merge(rc)
            p.merge(rp)
            var at = row * OUTPUT + mode * 30
            assert_equal(results[at], UInt64(v.count))
            close(results[at + 1], v.mean)
            close(results[at + 2], v.m2)
            assert_equal(results[at + 3] != 0, v.variance(0).__bool__())
            if v.variance(0):
                close(results[at + 4], v.variance(0).value())
            if v.variance(1):
                close(results[at + 5], sqrt(v.variance(1).value()))
            assert_equal(results[at + 6], UInt64(c.count))
            close(results[at + 7], c.mean)
            close(results[at + 8], c.m2)
            close(results[at + 9], c.m3)
            close(results[at + 10], c.m4)
            assert_equal(results[at + 11] != 0, c.skew(True).__bool__())
            assert_equal(results[at + 12] != 0, c.skew(False).__bool__())
            assert_equal(
                results[at + 13] != 0, c.kurtosis(True, True).__bool__()
            )
            assert_equal(
                results[at + 14] != 0, c.kurtosis(True, False).__bool__()
            )
            if c.skew(True):
                close(results[at + 15], c.skew(True).value())
            if c.skew(False):
                close(results[at + 16], c.skew(False).value())
            if c.kurtosis(True, True):
                close(results[at + 17], c.kurtosis(True, True).value())
            if c.kurtosis(True, False):
                close(results[at + 18], c.kurtosis(True, False).value())
                close(results[at + 19], c.kurtosis(False, False).value())
            assert_equal(results[at + 20], UInt64(p.count))
            close(results[at + 21], p.mean_x)
            close(results[at + 22], p.mean_y)
            close(results[at + 23], p.cxy)
            close(results[at + 24], p.m2x)
            close(results[at + 25], p.m2y)
            assert_equal(results[at + 26] != 0, p.covariance(0).__bool__())
            if p.covariance(0):
                close(results[at + 27], p.covariance(0).value())
                close(results[at + 28], p.covariance(1).value())
            close(results[at + 29], p.correlation())


def evaluate_host(inputs: List[UInt64], mut outputs: List[UInt64]):
    for row in range(CASES):
        evaluate_row(
            inputs.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
            outputs.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            row,
        )


def main() raises:
    var inputs = List[UInt64](length=CASES * INPUT, fill=0)
    var random = UInt64(0xD0F00D)
    for row in range(CASES):
        var n = row % (LENGTH + 1)
        inputs[row * INPUT] = UInt64(n)
        for i in range(n):
            random = random * 6364136223846793005 + 1442695040888963407
            var x = Float64(Int64(random >> 32) % 2001 - 1000) / 32
            var y = x * 3 + Float64(i % 5) / 16
            if row % 8 == 0:
                x = 7
            elif row % 8 == 1:
                x += 1e12
            var xword = bitcast[DType.uint64](x)
            if row % 8 == 2 and i == n // 2:
                xword = f.INF
            elif row % 8 == 3 and i == n // 2:
                xword = 0x7FF8000000000123
            elif row % 8 == 4:
                xword = UInt64(i + 1)
            elif row % 8 == 5:
                xword = UInt64(i % 2) * f.SIGN
            var at = row * INPUT + 1 + i * 3
            inputs[at] = xword
            inputs[at + 1] = bitcast[DType.uint64](y)
            inputs[at + 2] = UInt64(3 if row % 7 != 0 else i % 4)
    var expected = List[UInt64](length=CASES * OUTPUT, fill=0)
    var actual = List[UInt64](length=CASES * OUTPUT, fill=0)
    evaluate_host(inputs, expected)
    check_cpu(inputs, expected)
    var ctx = DeviceContext()
    var device_inputs = ctx.enqueue_create_buffer[DType.uint64](len(inputs))
    var device_outputs = ctx.enqueue_create_buffer[DType.uint64](len(actual))
    ctx.enqueue_copy(device_inputs, inputs.unsafe_ptr())
    ctx.enqueue_function[evaluate](
        device_inputs.unsafe_ptr(),
        device_outputs.unsafe_ptr(),
        grid_dim=1,
        block_dim=256,
    )
    ctx.enqueue_copy(actual.unsafe_ptr(), device_outputs)
    ctx.synchronize()
    for i in range(len(actual)):
        if actual[i] != expected[i]:
            print(
                "STATE MISMATCH case/field/host/device",
                i // OUTPUT,
                i % OUTPUT,
                expected[i],
                actual[i],
            )
        assert_equal(actual[i], expected[i])
    check_cpu(inputs, actual)
    print("PASS Mojo moments host/device fields:", CASES * OUTPUT)
