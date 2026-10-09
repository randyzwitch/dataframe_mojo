"""Optional CUDA storage/lifetime tests; run with pixi run -e gpu test-nvidia-runtime."""
from std.memory import bitcast
from std.sys import size_of
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)
from dataframe.column import Column
from dataframe.dtype import NUMERIC_DTYPES
from dataframe_accel.nvidia import NvidiaColumn, NvidiaRuntime, _HostDownload


def same_bits[D: DType](a: Column[Scalar[D]], b: Column[Scalar[D]]) raises:
    assert_equal(len(a), len(b))
    var pa = a.unsafe_values().unsafe_bitcast[UInt8]()
    var pb = b.unsafe_values().unsafe_bitcast[UInt8]()
    for i in range(len(a) * size_of[Scalar[D]]()):
        assert_equal(pa[unsafe_offset=i], pb[unsafe_offset=i])
    for i in range(len(a)):
        assert_equal(a.is_valid(i), b.is_valid(i))
    assert_equal(b.validity_offset(), 0)


def round_trip[
    D: DType
](runtime: NvidiaRuntime, source: Column[Scalar[D]]) raises:
    var device = runtime.upload[D](source)
    assert_equal(len(device), len(source))
    assert_equal(device.device_id(), runtime.device_id())
    assert_true(device.belongs_to(runtime))
    assert_false(device.host_accessible())
    assert_false(device.is_ready())
    var bitmap = len(source._bits[]) != 0
    assert_equal(device.has_validity(), bitmap)
    var offset = source.validity_offset() % 8 if bitmap else 0
    assert_equal(device.validity_offset(), offset)
    var bitmap_bytes = (offset + len(source) + 7) // 8 if bitmap and len(
        source
    ) else 0
    assert_equal(
        device.storage_bytes(),
        len(source) * size_of[Scalar[D]]() + bitmap_bytes,
    )
    same_bits[D](source, device.download())
    assert_true(device.is_ready())
    assert_false(device._source)
    # Repeated reads retain the allocation and do not upload again.
    same_bits[D](source, device.download())


def test_float_windows_and_validity_boundaries() raises:
    var runtime = NvidiaRuntime()
    comptime for d in range(2):
        comptime D = DType.float32 if d == 0 else DType.float64
        var values = List[Scalar[D]]()
        var valid = List[Bool]()
        for i in range(1100):
            values.append(Scalar[D](i - 500) / 8)
            valid.append(i % 3 != 0)
        var plain = Column[Scalar[D]](values.copy())
        var nullable = Column[Scalar[D]](values.copy(), valid)
        var all_valid = Column[Scalar[D]](
            values.copy(), List[Bool](length=len(values), fill=True)
        )
        var all_null = Column[Scalar[D]](
            values^, List[Bool](length=len(valid), fill=False)
        )
        for offset in [0, 1, 7, 8, 9, 19, 31]:
            for rows in [0, 1, 7, 8, 9, 31, 32, 33, 255, 256, 257, 1025]:
                round_trip[D](runtime, plain.slice(offset, rows))
                round_trip[D](runtime, nullable.slice(offset, rows))
                round_trip[D](runtime, all_valid.slice(offset, rows))
                round_trip[D](runtime, all_null.slice(offset, rows))


def test_special_float_payload_bits_are_unchanged() raises:
    var runtime = NvidiaRuntime()
    var a = Column[Float32](
        [
            bitcast[DType.float32](UInt32(0x7FC12345)),
            -0.0,
            0.0,
            Float32(Float64("inf")),
            Float32(Float64("-inf")),
            1e-30,
        ],
        [False, True, True, True, False, True],
    )
    var b = Column[Float64](
        [
            bitcast[DType.float64](UInt64(0x7FF8000000012345)),
            -0.0,
            0.0,
            Float64("inf"),
            Float64("-inf"),
            1e-300,
        ],
        [True, True, True, False, True, True],
    )
    round_trip[DType.float32](runtime, a)
    round_trip[DType.float64](runtime, b)


def test_fixed_width_integer_storage() raises:
    var runtime = NvidiaRuntime()
    comptime for i in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[i]
        comptime if not D.is_floating_point():
            var source = Column[Scalar[D]](
                [0, 1, Scalar[D].MIN, Scalar[D].MAX],
                [True, False, True, True],
            )
            round_trip[D](runtime, source)
            round_trip[D](runtime, source.slice(1, 3))


def upload_temporary(
    runtime: NvidiaRuntime,
) raises -> NvidiaColumn[DType.float32]:
    var source = Column[Float32]([3, 5, 7], [False, True, True])
    return runtime.upload[DType.float32](source.slice(1, 2))


def temporary_runtime() raises -> NvidiaColumn[DType.float32]:
    var runtime = NvidiaRuntime()
    return upload_temporary(runtime)


def test_context_and_source_ownership() raises:
    var runtime = NvidiaRuntime()
    var copied = runtime.copy()
    assert_true(runtime.shares_context(copied))
    var independent = NvidiaRuntime(runtime.device_id())
    assert_false(runtime.shares_context(independent))
    var device = upload_temporary(runtime)
    assert_true(device.belongs_to(copied))
    assert_false(device.belongs_to(independent))
    assert_true(device._source)
    device.wait()
    device.wait()
    assert_true(device.is_ready())
    assert_false(device._source)
    same_bits[DType.float32](Column[Float32]([5, 7]), device.download())
    # The context and source remain alive after both caller locals disappear.
    var detached = temporary_runtime()
    same_bits[DType.float32](Column[Float32]([5, 7]), detached.download())


def raise_during_upload(runtime: NvidiaRuntime) raises:
    var device = upload_temporary(runtime)
    try:
        raise Error("intentional pending upload failure")
    except error:
        assert_false(device.is_ready())
        raise error^


def raise_during_download(device: NvidiaColumn[DType.float32]) raises:
    var host = _HostDownload[DType.float32](device._owner._ctx, len(device), 0)
    host.pending = True
    device._owner._ctx.enqueue_copy(host.values.unsafe_ptr(), device._values)
    try:
        raise Error("intentional pending download failure")
    except error:
        assert_true(host.pending)
        raise error^


def test_pending_transfers_are_drained_on_unwind() raises:
    var runtime = NvidiaRuntime()
    for _ in range(20):
        with assert_raises(contains="intentional pending upload failure"):
            raise_during_upload(runtime)
        var device = upload_temporary(runtime)
        with assert_raises(contains="intentional pending download failure"):
            raise_during_download(device)
        same_bits[DType.float32](Column[Float32]([5, 7]), device.download())


def test_device_and_dtype_rejection() raises:
    with assert_raises(contains="NVIDIA device is unavailable"):
        _ = NvidiaRuntime(-1)
    with assert_raises(contains="NVIDIA device is unavailable"):
        _ = NvidiaRuntime(NvidiaRuntime.device_count())
    var runtime = NvidiaRuntime()
    with assert_raises(contains="Unsupported NVIDIA column storage dtype"):
        _ = runtime.upload[DType.float16](Column[Float16]([1]))
    if NvidiaRuntime.device_count() > 1:
        var other = NvidiaRuntime(1)
        var source = Column[Float32]([1, 2])
        var a = runtime.upload[DType.float32](source)
        var b = other.upload[DType.float32](source)
        assert_equal(b.device_id(), 1)
        assert_false(a.belongs_to(other))
        assert_false(b.belongs_to(runtime))
        same_bits[DType.float32](source, a.download())
        same_bits[DType.float32](source, b.download())
    else:
        print("Second-device hardware check skipped: only one CUDA device")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
