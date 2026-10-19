"""Shared query scope, context ownership and checked byte accounting."""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.atomic import Atomic
from std.memory import ArcPointer, Pointer
from std.time import perf_counter_ns
from .target import METAL_TARGET
from .scratch_pool import ScratchPool


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


struct _DeviceServiceState(Movable):
    var context: _ContextCell
    var device: Int
    var cached_lease: Atomic[Int64]
    var generation: Atomic[Int64]
    var pool: ScratchPool
    var pool_lock: Atomic[Int64]

    def __init__(out self, device: Int, idle_bytes: Int, idle_entries: Int):
        self.context = None
        self.device = device
        self.cached_lease = Atomic[Int64](0)
        self.generation = Atomic[Int64](0)
        self.pool = ScratchPool(idle_bytes, idle_entries)
        self.pool_lock = Atomic[Int64](0)

    def __deinit__(deinit self):
        # Buffers keep their context alive; release idle owners before the
        # service's final context handle, without issuing new GPU work.
        _ = self.pool^
        _ = self.context^


struct DeviceService(Copyable):
    """Lazy, runtime-owned cache for one reusable Metal context.

    Only one Scope may lease the cached context at once. Overlapping scopes
    receive a private context, so copies of MetalRuntime remain concurrent.
    A separately bounded idle pool owns only private reduction scratch.
    All active allocations remain charged to their per-query reservations.
    """

    var _state: ArcPointer[_DeviceServiceState]

    def __init__(
        out self,
        device_id: Int = 0,
        *,
        idle_scratch_bytes: Int = 8 * 1024 * 1024,
        idle_scratch_entries: Int = 32,
    ) raises:
        if device_id < 0 or idle_scratch_bytes < 0 or idle_scratch_entries < 0:
            raise Error(
                "Mojo GPU device service device or idle limit is invalid"
            )
        self._state = ArcPointer(
            _DeviceServiceState(
                device_id, idle_scratch_bytes, idle_scratch_entries
            )
        )

    def same_service(self, other: Self) -> Bool:
        return Int(Pointer(to=self._state[])) == Int(Pointer(to=other._state[]))

    def cached_generations(self) -> Int:
        return Int(self._state[].generation.load())

    def device_id(self) -> Int:
        return self._state[].device

    def _lock_pool(self):
        var unlocked = Int64(0)
        while not self._state[].pool_lock.compare_exchange(unlocked, 1):
            unlocked = 0

    def scratch_info(self) -> Tuple[Int, Int, Int, Int]:
        """Idle bytes, idle entries, hits and misses; separate from query peak.
        """
        self._lock_pool()
        var result = (
            self._state[].pool.idle_bytes,
            self._state[].pool.idle_entries,
            self._state[].pool.hits,
            self._state[].pool.misses,
        )
        self._state[].pool_lock.store(0)
        return result

    def _scratch[
        dtype: DType
    ](self, context: DeviceContext, length: Int) raises -> DeviceBuffer[dtype]:
        self._lock_pool()
        var state = Pointer(to=self._state[]).unsafe_origin_cast[MutAnyOrigin]()
        var found = Optional[DeviceBuffer[dtype]]()
        try:
            found = state[].pool.take[dtype](context, length)
            state[].pool_lock.store(0)
        except error:
            state[].pool_lock.store(0)
            raise error^
        if found:
            return found.take()
        return context.enqueue_create_buffer[dtype](length)

    def _recycle_scratch[
        dtype: DType
    ](self, context: DeviceContext, var buffer: DeviceBuffer[dtype]) raises:
        self._lock_pool()
        var state = Pointer(to=self._state[]).unsafe_origin_cast[MutAnyOrigin]()
        try:
            state[].pool.put[dtype](context, buffer^)
            state[].pool_lock.store(0)
        except error:
            state[].pool_lock.store(0)
            raise error^

    def _acquire(self) raises -> Tuple[DeviceContext, Bool]:
        var unlocked = Int64(0)
        if self._state[].cached_lease.compare_exchange(unlocked, 1):
            var state = Pointer(to=self._state[]).unsafe_origin_cast[
                MutAnyOrigin
            ]()
            try:
                if not state[].context:
                    comptime if not METAL_TARGET:
                        raise Error(
                            "Mojo GPU device service requires a compiled Metal"
                            " target"
                        )
                    else:
                        state[].context = _ContextCell(
                            rebind[DeviceContext if METAL_TARGET else Int](
                                DeviceContext(state[].device, api="metal")
                            )
                        )
                        _ = state[].generation.fetch_add(1)
                var result = rebind[DeviceContext](
                    state[].context.value()
                ).copy()
                return (result^, True)
            except error:
                self._lock_pool()
                state[].pool.clear()
                state[].pool_lock.store(0)
                state[].context = None
                state[].cached_lease.store(0)
                raise error^
        comptime if not METAL_TARGET:
            raise Error(
                "Mojo GPU device service requires a compiled Metal target"
            )
        else:
            return (DeviceContext(self._state[].device, api="metal"), False)

    def _release(self, reusable: Bool):
        var state = Pointer(to=self._state[]).unsafe_origin_cast[MutAnyOrigin]()
        if not reusable:
            # The Scope still owns its context copy until after this method.
            # Never make a context involved in a failed query available again.
            self._lock_pool()
            state[].pool.clear()
            state[].pool_lock.store(0)
            state[].context = None
        state[].cached_lease.store(0)


struct _ScopeState(Movable):
    var context: _ContextCell
    var service: Optional[DeviceService]
    var cached_context_lease: Bool
    var reusable: Atomic[Int64]
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

    def __init__(
        out self,
        device: Int,
        budget: Int,
        profiling: Bool,
        service: Optional[DeviceService],
    ):
        self.context = None
        self.service = service.copy()
        self.cached_context_lease = False
        self.reusable = Atomic[Int64](0)
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

    def __deinit__(deinit self):
        if self.cached_context_lease and self.service:
            self.service.value()._release(self.reusable.load() == 1)


struct Scope(Copyable):
    var _state: ArcPointer[_ScopeState]

    def __init__(
        out self,
        device_id: Int = 0,
        budget: Int = -1,
        *,
        profiling: Bool = False,
        service: Optional[DeviceService] = None,
    ) raises:
        if device_id < 0 or budget < -1:
            raise Error("Mojo GPU resident device or budget is invalid")
        if service and service.value().device_id() != device_id:
            raise Error("Mojo GPU device service is for a different device")
        self._state = ArcPointer(
            _ScopeState(device_id, budget, profiling, service)
        )

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
                    if state[].service:
                        var acquired = state[].service.value()._acquire()
                        var cached_lease = acquired[1]
                        var acquired_context = acquired[0].copy()
                        state[].context = _ContextCell(
                            rebind[DeviceContext if METAL_TARGET else Int](
                                acquired_context^
                            )
                        )
                        state[].cached_context_lease = cached_lease
                    else:
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
                if state[].cached_context_lease and state[].service:
                    state[].service.value()._release(False)
                    state[].cached_context_lease = False
                state[].context_lock.store(0)
                raise error^

    def mark_success(self):
        # The service lease is returned with _ScopeState destruction, after the
        # wrapper has completed terminal materialization and its final drain.
        var pending = Int64(0)
        _ = self._state[].reusable.compare_exchange(pending, 1)

    def mark_failed(self):
        # Failure is terminal for one Scope: a later success mark cannot
        # return the context from a failed invocation to the service.
        _ = self._state[].reusable.store(2)

    def uses_cached_context(self) -> Bool:
        return self._state[].cached_context_lease

    def scratch[dtype: DType](self, length: Int) raises -> DeviceBuffer[dtype]:
        """Obtain private scratch after its caller reserves the exact capacity.
        """
        if length <= 0:
            raise Error("Mojo GPU reusable scratch length must be positive")
        var context = self.context()
        if self._state[].cached_context_lease and self._state[].service:
            return (
                self._state[].service.value()._scratch[dtype](context, length)
            )
        return context.enqueue_create_buffer[dtype](length)

    def recycle_scratch[
        dtype: DType
    ](self, var buffer: DeviceBuffer[dtype]) raises:
        """Return an unaliased root after successful synchronization of its last use.

        Only lexical operator-private scratch may enter this API. Native-column
        values and views keep their existing ownership and are never pooled.
        The caller retains its reservation until all these returns complete.
        """
        if (
            self._state[].cached_context_lease
            and self._state[].service
            and self._state[].reusable.load() != 2
        ):
            self._state[].service.value()._recycle_scratch[dtype](
                self.context(), buffer^
            )

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
                "Mojo GPU resident query exceeds memory budget before"
                " submission"
            )
        self._record_peak(total)
        return total

    def _retain(self, bytes: Int) raises:
        var previous = self._state[].live.load()
        while True:
            var total = checked_add(Int(previous), bytes)
            if self._state[].budget >= 0 and total > self._state[].budget:
                raise Error(
                    "Mojo GPU resident query exceeds memory budget before"
                    " submission"
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
            try:
                self.synchronize()
            except:
                pass
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
