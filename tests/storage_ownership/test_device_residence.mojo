"""Device-resident buffers: length without copying, one host copy under
concurrent readers, failures that raise, detaching writes, unified memory
read in place, and one release after the last reference."""
# contract-group: storage.device_residence
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import ArcPointer, Pointer, unsafe_memcpy
from std.sys import size_of
from std.time import perf_counter_ns
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column
from dataframe.storage.buffer import _Buffer
from dataframe.storage.residence import (
    _CopyToHost,
    _DeviceResidence,
    _ReleaseDevice,
)
from dataframe.parallel import Job, Pool
from tests.support.lifecycle_jobs import WorkerSetting


struct FakeDevice(Movable):
    """Device memory kept in host memory, counting what the core asks of it."""

    var data: List[Int64]
    var copies: Atomic[Int64]
    var releases: Atomic[Int64]
    var fail: Bool

    def __init__(out self, var data: List[Int64], fail: Bool = False):
        self.data = data^
        self.copies = Atomic[Int64](0)
        self.releases = Atomic[Int64](0)
        self.fail = fail


def fake_to_host(context: Int, address: Int, bytes: Int) abi("C") -> Int:
    var device = Pointer[FakeDevice, MutUntrackedOrigin](
        unsafe_from_address=context
    )
    _ = device[].copies.fetch_add(1)
    if device[].fail:
        return 1
    # Hold the copy open briefly so concurrent first readers really overlap.
    var until = perf_counter_ns() + 2_000_000
    while perf_counter_ns() < until:
        _ = external_call["sched_yield", Int32]()
    unsafe_memcpy(
        dest=Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address),
        src=Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(device[].data.unsafe_ptr())
        ),
        count=bytes,
    )
    return 0


def fake_release(context: Int) abi("C"):
    var device = Pointer[FakeDevice, MutUntrackedOrigin](
        unsafe_from_address=context
    )
    _ = device[].releases.fetch_add(1)


def residence(
    device: ArcPointer[FakeDevice], host_address: Int = 0
) -> ArcPointer[_DeviceResidence]:
    var copy: _CopyToHost = fake_to_host
    var release: _ReleaseDevice = fake_release
    return ArcPointer(
        _DeviceResidence(
            backend="test",
            device=0,
            context=Int(Pointer(to=device[])),
            bytes=len(device[].data) * size_of[Int64](),
            to_host=Pointer(to=copy).unsafe_bitcast[Int]()[],
            release=Pointer(to=release).unsafe_bitcast[Int]()[],
            host_address=host_address,
        )
    )


def device_column(device: ArcPointer[FakeDevice]) raises -> Column[Int64]:
    var length = len(device[].data)
    return Column[Int64]._from_buffers(
        _Buffer[Int64].on_device(length, residence(device)),
        _Buffer(List[UInt8]()),
        0,
        length,
    )


def values(count: Int) -> List[Int64]:
    var result = List[Int64]()
    for i in range(count):
        result.append(Int64(i * 3 - 7))
    return result^


def test_length_needs_no_copy_and_reads_copy_once() raises:
    var device = ArcPointer(FakeDevice(values(1000)))
    var column = device_column(device)
    assert_equal(len(column), 1000)
    assert_false(column._data.on_host())
    assert_equal(device[].copies.load(), Int64(0))
    var total = Int64(0)
    for i in range(len(column)):
        total += column._get(i)
    assert_equal(total, Int64(3 * 999 * 1000 // 2 - 7 * 1000))
    assert_equal(column._get(999), Int64(999 * 3 - 7))
    assert_true(column._data.on_host())
    assert_equal(device[].copies.load(), Int64(1))
    # The device copy stays for device consumers after the host copy.
    assert_true(column._data.residence())
    assert_equal(device[].releases.load(), Int64(0))
    var window = column.slice(10, 5)
    assert_equal(len(window), 5)
    assert_equal(window._get(0), Int64(10 * 3 - 7))
    _ = column^
    assert_equal(device[].releases.load(), Int64(0))
    _ = window^
    assert_equal(device[].releases.load(), Int64(1))
    assert_equal(device[].copies.load(), Int64(1))


struct Meeting(Movable):
    var arrived: Atomic[Int64]

    def __init__(out self):
        self.arrived = Atomic[Int64](0)


struct ReadJob(Job):
    var column: Column[Int64]
    var meeting: ArcPointer[Meeting]
    var workers: Int
    var total: Int64

    def __init__(
        out self,
        column: Column[Int64],
        meeting: ArcPointer[Meeting],
        workers: Int,
    ):
        self.column = column.copy()
        self.meeting = meeting.copy()
        self.workers = workers
        self.total = 0

    def run(mut self) raises:
        _ = self.meeting[].arrived.fetch_add(1)
        var deadline = perf_counter_ns() + 2_000_000_000
        while self.meeting[].arrived.load() < Int64(self.workers):
            if perf_counter_ns() > deadline:
                raise Error("device-residence readers did not overlap")
            _ = external_call["sched_yield", Int32]()
        for i in range(len(self.column)):
            self.total += self.column._get(i)


def test_concurrent_first_reads_share_one_copy() raises:
    var setting = WorkerSetting(4)
    var pool = Pool(4)
    for _ in range(3):
        var device = ArcPointer(FakeDevice(values(4096)))
        var column = device_column(device)
        var meeting = ArcPointer(Meeting())
        var jobs = List[ReadJob]()
        for _ in range(4):
            jobs.append(ReadJob(column, meeting, 4))
        pool.run(jobs)
        assert_equal(device[].copies.load(), Int64(1))
        for i in range(len(jobs)):
            assert_equal(jobs[i].total, Int64(3 * 4095 * 4096 // 2 - 7 * 4096))
        _ = jobs^
        _ = column^
        assert_equal(device[].releases.load(), Int64(1))
    _ = pool^
    _ = setting^


def test_failed_copy_raises_every_time_and_releases_once() raises:
    var device = ArcPointer(FakeDevice(values(16), fail=True))
    var buffer = _Buffer[Int64].on_device(16, residence(device))
    with assert_raises(contains="test device buffer to the host failed"):
        buffer.copy_to_host()
    with assert_raises(contains="test device buffer to the host failed"):
        buffer.copy_to_host()
    assert_equal(device[].copies.load(), Int64(1))
    assert_false(buffer.on_host())
    _ = buffer^
    assert_equal(device[].releases.load(), Int64(1))


def test_writable_detaches_from_the_device_copy() raises:
    var device = ArcPointer(FakeDevice(values(8)))
    var original = _Buffer[Int64].on_device(8, residence(device))
    var changed = original.copy()
    changed.writable()[0] = 100
    assert_false(changed.residence())
    assert_equal(changed[0], Int64(100))
    assert_equal(original[0], Int64(-7))
    assert_true(original.residence())
    assert_equal(device[].copies.load(), Int64(1))
    _ = original^
    assert_equal(device[].releases.load(), Int64(1))
    _ = changed^


def test_unified_memory_is_read_in_place() raises:
    var device = ArcPointer(FakeDevice(values(32)))
    var buffer = _Buffer[Int64].on_device(
        32, residence(device, host_address=Int(device[].data.unsafe_ptr()))
    )
    assert_true(buffer.on_host())
    buffer.copy_to_host()
    assert_equal(buffer[31], Int64(31 * 3 - 7))
    assert_equal(Int(buffer.unsafe_ptr()), Int(device[].data.unsafe_ptr()))
    assert_equal(device[].copies.load(), Int64(0))
    _ = buffer^
    assert_equal(device[].releases.load(), Int64(1))


def test_size_must_match_length() raises:
    var device = ArcPointer(FakeDevice(values(4)))
    with assert_raises(contains="does not match its length"):
        _ = _Buffer[Int64].on_device(5, residence(device))
    with assert_raises(contains="does not match its length"):
        _ = _Buffer[Int32].on_device(4, residence(device))
