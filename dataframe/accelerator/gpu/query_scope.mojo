"""Shared query scope, context ownership and checked byte accounting."""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.atomic import Atomic
from std.memory import ArcPointer, Pointer
from std.time import perf_counter_ns
from .target import METAL_TARGET


def checked_add(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or a > Int.MAX - b:
        raise Error("Mojo GPU resident allocation size overflow")
    return a + b


def checked_mul(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or (b and a > Int.MAX // b):
        raise Error("Mojo GPU resident allocation size overflow")
    return a * b


# Metadata-only targets retain accounting and lease ownership without
# instantiating an unavailable DeviceContext destructor. Real Metal keeps
# the exact context owner type, including its queue lifetime and teardown.
comptime _ContextCell = Optional[DeviceContext if METAL_TARGET else Int]


struct _ScopeState(Movable):
    var context: _ContextCell
    var device: Int
    var budget: Int
    var live: Atomic[Int64]
    var peak: Atomic[Int64]
    var uploads: Atomic[Int64]
    var downloads: Atomic[Int64]
    var launches: Atomic[Int64]
    var waits: Atomic[Int64]
    var context_lock: Atomic[Int64]
    var context_ready: Atomic[Int64]
    var profiling: Bool
    var context_ns: Atomic[Int64]
    var synchronization_ns: Atomic[Int64]

    def __init__(out self, device: Int, budget: Int, profiling: Bool):
        self.context = None
        self.device = device
        self.budget = budget
        self.live = Atomic[Int64](0)
        self.peak = Atomic[Int64](0)
        self.uploads = Atomic[Int64](0)
        self.downloads = Atomic[Int64](0)
        self.launches = Atomic[Int64](0)
        self.waits = Atomic[Int64](0)
        self.context_lock = Atomic[Int64](0)
        self.context_ready = Atomic[Int64](0)
        self.profiling = profiling
        self.context_ns = Atomic[Int64](0)
        self.synchronization_ns = Atomic[Int64](0)


struct Scope(Copyable):
    var _state: ArcPointer[_ScopeState]

    def __init__(
        out self,
        device_id: Int = 0,
        budget: Int = -1,
        *,
        profiling: Bool = False,
    ) raises:
        if device_id < 0 or budget < -1:
            raise Error("Mojo GPU resident device or budget is invalid")
        self._state = ArcPointer(_ScopeState(device_id, budget, profiling))

    @staticmethod
    def device_count() raises -> Int:
        return DeviceContext.number_of_devices(api="metal")

    def initialized(self) -> Bool:
        return self._state[].context_ready.load() != 0

    def context(self) raises -> DeviceContext:
        comptime if not METAL_TARGET:
            raise Error(
                "Mojo GPU resident context requires a compiled Metal target"
            )
        else:
            # Arc ownership is immutable externally. This one protected cell is
            # interior mutable so copied scopes initialize exactly the same queue.
            var unlocked = Int64(0)
            while not self._state[].context_lock.compare_exchange(unlocked, 1):
                unlocked = 0
            var state = Pointer(to=self._state[]).unsafe_origin_cast[
                MutAnyOrigin
            ]()
            try:
                if not state[].context:
                    var started = Int(
                        perf_counter_ns()
                    ) if state[].profiling else 0
                    state[].context = _ContextCell(
                        rebind[DeviceContext if METAL_TARGET else Int](
                            DeviceContext(state[].device, api="metal")
                        )
                    )
                    if state[].profiling:
                        _ = state[].context_ns.fetch_add(
                            Int64(Int(perf_counter_ns()) - started)
                        )
                    state[].context_ready.store(1)
                var result = rebind[DeviceContext](
                    state[].context.value()
                ).copy()
                state[].context_lock.store(0)
                return result^
            except error:
                state[].context_lock.store(0)
                raise error^

    def same_query(self, other: Self) -> Bool:
        return Int(Pointer(to=self._state[])) == Int(Pointer(to=other._state[]))

    def info(self) -> Tuple[Int, Int, Int]:
        return (
            Int(self._state[].live.load()),
            Int(self._state[].peak.load()),
            self._state[].budget,
        )

    def transfers(self) -> Tuple[Int, Int, Int, Int]:
        return (
            Int(self._state[].uploads.load()),
            Int(self._state[].downloads.load()),
            Int(self._state[].launches.load()),
            Int(self._state[].waits.load()),
        )

    def _record_peak(self, total: Int):
        var previous = self._state[].peak.load()
        while Int64(total) > previous:
            if self._state[].peak.compare_exchange(previous, Int64(total)):
                break

    def check(self, additional: Int) raises -> Int:
        var total = checked_add(Int(self._state[].live.load()), additional)
        if self._state[].budget >= 0 and total > self._state[].budget:
            raise Error(
                "Mojo GPU resident query exceeds memory budget before submission"
            )
        self._record_peak(total)
        return total

    def _retain(self, bytes: Int) raises:
        var previous = self._state[].live.load()
        while True:
            var total = checked_add(Int(previous), bytes)
            if self._state[].budget >= 0 and total > self._state[].budget:
                raise Error(
                    "Mojo GPU resident query exceeds memory budget before submission"
                )
            if self._state[].live.compare_exchange(previous, Int64(total)):
                self._record_peak(total)
                return

    def _release(self, bytes: Int):
        _ = self._state[].live.fetch_sub(Int64(bytes))

    def record_upload(self, bytes: Int):
        _ = self._state[].uploads.fetch_add(Int64(bytes))

    def record_download(self, bytes: Int):
        _ = self._state[].downloads.fetch_add(Int64(bytes))

    def record_launches(self, count: Int = 1):
        _ = self._state[].launches.fetch_add(Int64(count))

    def synchronize(self) raises:
        if self.initialized():
            var started = Int(
                perf_counter_ns()
            ) if self._state[].profiling else 0
            self.context().synchronize()
            if self._state[].profiling:
                _ = self._state[].synchronization_ns.fetch_add(
                    Int64(Int(perf_counter_ns()) - started)
                )
            _ = self._state[].waits.fetch_add(1)

    def timing_ns(self) -> Tuple[Int, Int]:
        """Host context construction and waits; neither is GPU-only time."""
        return (
            Int(self._state[].context_ns.load()),
            Int(self._state[].synchronization_ns.load()),
        )

    def read_control_into[
        dtype: DType
    ](
        self, buffer: DeviceBuffer[dtype], mut values: List[Scalar[dtype]]
    ) raises:
        """Complete a control copy before its caller's host owner can unwind."""
        if len(values) != len(buffer):
            raise Error("Mojo GPU control readback shape mismatch")
        var ctx = self.context()
        try:
            ctx.enqueue_copy(values.unsafe_ptr(), buffer)
            self.synchronize()
        except error:
            self.synchronize()
            raise error^

    def read_control[
        dtype: DType
    ](self, buffer: DeviceBuffer[dtype]) raises -> List[Scalar[dtype]]:
        """Read shape/error metadata while keeping its host owner alive.

        Preserve the caller's existing synchronization point. Byte accounting
        remains with the executor, which knows this control record's meaning.
        """
        var values = List[Scalar[dtype]](length=len(buffer), fill=0)
        self.read_control_into(buffer, values)
        return values^

    def device_info(self) raises -> Tuple[Int, String, String]:
        var ctx = self.context()
        return (self._state[].device, ctx.name(), ctx.api())
