"""Worker threads for partitioned execution (POSIX threads via libc).

Mojo 1.1's standard library has no thread pool, so jobs run on threads
created with `pthread_create` through the C FFI. There is no new dependency:
libc provides pthreads on Linux and macOS.

A job is a self-contained value implementing `Job`: it owns (reference-counted)
copies of its inputs and writes only its own outputs, so no locking is
needed. `run_jobs` runs every job, the last one on the calling thread, joins
the others, and re-raises the first worker error on the caller (errors cannot
unwind across threads). Shared column buffers are safe to read concurrently:
their reference counts are atomic and they are never mutated while shared.

`worker_count` sizes the pool: `DATAFRAME_THREADS` (1 disables parallelism),
else the physical core count, capped so each worker gets at least
`MIN_ROWS_PER_WORKER` rows (small inputs stay single-threaded).

An algorithm that runs several rounds of jobs -- a sort's run pass and then
its merge rounds -- would pay that creation cost once per round. `Pool`
creates the threads once and wakes them for each round instead, which is
about twenty times cheaper per round. Its workers are **joined when the
pool is released**, which is the whole design: a pool that outlives the
work, holding threads parked until the process exits, crashes on the way
out. Under `mojo run` the JIT frees the compiled code while those threads
are still parked in it -- reproducibly, about one run in three -- and there
is no point at which Mojo code can run at process exit to join them first
(`atexit` is not a dynamic symbol under glibc, and a `__cxa_atexit` handler
crashes under the JIT even with no threads involved). So a pool is scoped
to the operation that creates it and never becomes process-wide. See #103.
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import Pointer
from std.os import getenv
from std.sys import num_physical_cores, size_of

comptime MIN_ROWS_PER_WORKER = 65536

# pthread_mutex_t and pthread_cond_t are opaque and platform-sized (up to 64
# bytes on the platforms built for); over-allocating is harmless.
comptime _SYNC_BYTES = 128

# How long a worker spins watching for the next round before parking on the
# condition variable. Parking is what a pool is supposed to do between
# rounds, and it turned out to be the whole problem: waking 31 workers from
# `pthread_cond_wait` costs enough that a sort's run round took 21 ms
# against the 12 ms it takes when each round creates its own threads. The
# rounds of one operation arrive within microseconds of each other, so a
# worker that spins first catches the next round without a park/wake pair
# at all, and the same round costs 11.3 ms. Measured knee: 200,000 is too
# short (21 ms), 1,000,000 is enough, and beyond that nothing improves.
# A pool is scoped to one operation, so this only ever burns a core during
# that operation's own serial gaps, and never while the process is idle.
comptime _SPIN_LIMIT = 2_000_000


trait Job(Deinitable, Movable):
    """Work for one thread; raise to report failure to `run_jobs`."""

    def run(mut self) raises:
        ...


comptime _Entry = def(Int) thin abi("C") -> Int


struct _Slot[J: Job](Movable):
    var job: Self.J
    var failed: Bool
    var message: String

    def __init__(out self, var job: Self.J):
        self.job = job^
        self.failed = False
        self.message = ""

    def run(mut self):
        try:
            self.job.run()
        except e:
            self.failed = True
            self.message = String(e)

    def into_job(deinit self) -> Self.J:
        return self.job^


def _entry[J: Job](address: Int) abi("C") -> Int:
    Pointer[_Slot[J], MutAnyOrigin](unsafe_from_address=address)[].run()
    return 0


struct _Task(Copyable, Movable):
    """One slot to run, as a function pointer and the slot's address, so a
    worker can run it without knowing the job type."""

    var entry: Int
    var argument: Int

    def __init__(out self, entry: Int, argument: Int):
        self.entry = entry
        self.argument = argument

    def run(self):
        var entry = Pointer(to=self.entry).unsafe_bitcast[_Entry]()[]
        _ = entry(self.argument)


struct _Shared(Movable):
    """Pool state the workers read. Lives at a fixed heap address for as
    long as any worker might touch it."""

    var mutex: Array[UInt8, _SYNC_BYTES]
    var cond: Array[UInt8, _SYNC_BYTES]
    # Bumped under the mutex to wake sleeping workers for the next round.
    var generation: Atomic[Int64]
    # The current round: the address of its List[_Task], its length, and how
    # many tasks have finished. Both are published before `generation` is
    # bumped, and a worker reads them only after observing that bump, so it
    # cannot see a half-published round or a length left over from the last
    # one. (PR #114 read a plain `count` after claiming from a counter and
    # dropped exactly one task per hang.)
    var tasks: Atomic[Int64]
    var count: Atomic[Int64]
    # The next task to claim for a dynamically scheduled round. Static rounds
    # deliberately avoid touching this counter: their equally sized work is
    # faster with fixed striding (see `_run_share`).
    var next: Atomic[Int64]
    var done: Atomic[Int64]
    # Workers that have left the current round. `done` counts only tasks;
    # this barrier keeps a late waking worker from reading a cleared list.
    var arrived: Atomic[Int64]
    # 0 is a fixed share, 1 claims a completed list, and 2 receives tasks
    # while the caller is still producing them.
    var mode: Atomic[Int64]
    # A produced round has no more tasks once this is set. It is separate
    # from `stopping`, which tears the pool down for good.
    var closed: Atomic[Int64]
    # Set once, at release, to let parked workers return instead of waiting.
    var stopping: Atomic[Int64]
    var threads: Int

    def __init__(out self):
        self.mutex = Array[UInt8, _SYNC_BYTES](fill=0)
        self.cond = Array[UInt8, _SYNC_BYTES](fill=0)
        self.generation = Atomic[Int64](0)
        self.tasks = Atomic[Int64](0)
        self.count = Atomic[Int64](0)
        self.next = Atomic[Int64](0)
        self.done = Atomic[Int64](0)
        self.arrived = Atomic[Int64](0)
        self.mode = Atomic[Int64](0)
        self.closed = Atomic[Int64](0)
        self.stopping = Atomic[Int64](0)
        self.threads = 0

    def _mutex(self) -> Int:
        return Int(self.mutex.unsafe_ptr())

    def _cond(self) -> Int:
        return Int(self.cond.unsafe_ptr())

    def _run_share(mut self, index: Int, participants: Int):
        """Run this participant's tasks: index, index + P, index + 2P, ...

        Statically, not by claiming from a shared counter. A counter shares
        work dynamically, which sounds better and measured much worse: the
        calling thread never parks, so it starts claiming while the workers
        are still waking, and with one long task per participant it takes a
        second task before a worker takes its first. A sort's run round --
        32 tasks of about 12 ms -- went from 12 ms to 23 ms that way, which
        is one task's time against two. Rounds here are cut into equal
        pieces by `partitions`, so a fixed share per participant is both
        balanced and free of any claiming at all.

        Both fields are read after the caller published them and bumped the
        generation this worker just observed, so neither can be stale.
        """
        var count = Int(self.count.load())
        ref tasks = Pointer[List[_Task], MutAnyOrigin](
            unsafe_from_address=Int(self.tasks.load())
        )[]
        var i = index
        while i < count:
            tasks[i].run()
            _ = self.done.fetch_add(1)
            i += participants

    def _run_claim(mut self):
        """Claim coarse, uneven tasks until this round has none left.

        This is intentionally separate from `_run_share`: a claim counter
        gives a stalled worker's remaining work to another participant, but
        costs more than static striding for the small, even sort tasks that
        motivated the pool.
        """
        var count = Int(self.count.load())
        ref tasks = Pointer[List[_Task], MutAnyOrigin](
            unsafe_from_address=Int(self.tasks.load())
        )[]
        while True:
            var i = Int(self.next.fetch_add(1))
            if i >= count:
                return
            tasks[i].run()
            _ = self.done.fetch_add(1)

    def _run_produced(mut self):
        """Claim tasks published by the producer, parking between submits."""
        while True:
            _ = external_call["pthread_mutex_lock", Int32](self._mutex())
            while (
                self.next.load() >= self.count.load()
                and self.closed.load() == 0
            ):
                _ = external_call["pthread_cond_wait", Int32](
                    self._cond(), self._mutex()
                )
            if self.next.load() >= self.count.load():
                _ = external_call["pthread_mutex_unlock", Int32](self._mutex())
                return
            var i = Int(self.next.load())
            self.next.store(Int64(i + 1))
            var tasks = Pointer[_Task, MutAnyOrigin](
                unsafe_from_address=Int(self.tasks.load())
            )
            var task = tasks.unsafe_offset(i)[].copy()
            _ = external_call["pthread_mutex_unlock", Int32](self._mutex())
            task.run()
            _ = self.done.fetch_add(1)


def _worker(argument: Int) abi("C") -> Int:
    # `argument` points at a (shared address, participant index) pair.
    var pair = Pointer[Int, MutAnyOrigin](unsafe_from_address=argument)
    var address = pair[]
    var index = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=argument + size_of[Int]()
    )[]
    ref shared = Pointer[_Shared, MutAnyOrigin](unsafe_from_address=address)[]
    var seen = Int64(0)
    while True:
        # Spin before parking. Rounds arrive back to back -- a sort's merge
        # rounds are one after another -- and a parked worker takes long
        # enough to wake that the calling thread, which never parked, claims
        # several tasks and runs them one after another in the meantime. That
        # is what makes a round cost more than the longest task in it.
        var spins = 0
        while (
            spins < _SPIN_LIMIT
            and shared.generation.load() == seen
            and shared.stopping.load() == 0
        ):
            spins += 1
        if shared.generation.load() == seen and shared.stopping.load() == 0:
            _ = external_call["pthread_mutex_lock", Int32](shared._mutex())
            while (
                shared.generation.load() == seen and shared.stopping.load() == 0
            ):
                _ = external_call["pthread_cond_wait", Int32](
                    shared._cond(), shared._mutex()
                )
            _ = external_call["pthread_mutex_unlock", Int32](shared._mutex())
        if shared.stopping.load() != 0:
            return 0
        seen = shared.generation.load()
        if shared.mode.load() == 1:
            shared._run_claim()
        elif shared.mode.load() == 2:
            shared._run_produced()
        else:
            shared._run_share(index, shared.threads + 1)
        _ = shared.arrived.fetch_add(1)


struct _ProducedJobs[J: Job](Movable):
    """A fixed-capacity job list a caller fills while a pool consumes it.

    `submit` publishes only after both the slot and its type-erased task are
    in their preallocated lists. Holding the pool mutex during that short
    append makes the list headers safe to inspect from a worker and provides
    the release point for the published count.
    """

    var address: Int
    var slots: List[_Slot[Self.J]]
    var tasks: List[_Task]
    var capacity: Int
    var started: Bool
    var finished: Bool

    def __init__(out self, capacity: Int):
        self.address = 0
        self.slots = List[_Slot[Self.J]](capacity=capacity)
        self.tasks = List[_Task](capacity=capacity)
        self.capacity = capacity
        self.started = False
        self.finished = False

    def _shared(mut self) -> ref[MutAnyOrigin] _Shared:
        return Pointer[_Shared, MutAnyOrigin](
            unsafe_from_address=self.address
        )[]

    def _begin(mut self, address: Int):
        self.address = address
        self.started = address != 0
        if not self.started:
            return
        ref shared = self._shared()
        _ = external_call["pthread_mutex_lock", Int32](shared._mutex())
        shared.tasks.store(Int64(Int(self.tasks.unsafe_ptr())))
        shared.count.store(0)
        shared.next.store(0)
        shared.done.store(0)
        shared.arrived.store(0)
        shared.closed.store(0)
        shared.mode.store(2)
        _ = shared.generation.fetch_add(1)
        _ = external_call["pthread_cond_broadcast", Int32](shared._cond())
        _ = external_call["pthread_mutex_unlock", Int32](shared._mutex())

    def submit(mut self, var job: Self.J) raises:
        """Publish one job. Jobs are returned in submit order after finish."""
        if self.finished:
            raise Error("cannot submit to a finished producer")
        if len(self.slots) >= self.capacity:
            raise Error("produced job capacity exceeded")
        if not self.started:
            self.slots.append(_Slot[Self.J](job^))
            self.slots[len(self.slots) - 1].run()
            return
        ref shared = self._shared()
        _ = external_call["pthread_mutex_lock", Int32](shared._mutex())
        self.slots.append(_Slot[Self.J](job^))
        var entry: _Entry = _entry[Self.J]
        var entry_address = Pointer(to=entry).unsafe_bitcast[Int]()[]
        self.tasks.append(
            _Task(
                entry_address,
                Int(Pointer(to=self.slots[len(self.slots) - 1])),
            )
        )
        shared.count.store(Int64(len(self.tasks)))
        _ = external_call["pthread_cond_broadcast", Int32](shared._cond())
        _ = external_call["pthread_mutex_unlock", Int32](shared._mutex())

    def _close(mut self):
        if self.finished:
            return
        if not self.started:
            self.finished = True
            return
        ref shared = self._shared()
        _ = external_call["pthread_mutex_lock", Int32](shared._mutex())
        shared.closed.store(1)
        _ = external_call["pthread_cond_broadcast", Int32](shared._cond())
        _ = external_call["pthread_mutex_unlock", Int32](shared._mutex())
        # After publishing EOF, the producer can decode unclaimed work.
        shared._run_produced()
        while shared.done.load() < shared.count.load():
            _ = external_call["sched_yield", Int32]()
        while shared.arrived.load() < Int64(shared.threads):
            _ = external_call["sched_yield", Int32]()
        shared.count.store(0)
        shared.tasks.store(0)
        self.finished = True

    def finish(mut self) raises -> List[Self.J]:
        """Drain workers, re-raise the first error, and return submitted jobs."""
        self._close()
        for t in range(len(self.slots)):
            if self.slots[t].failed:
                raise Error(self.slots[t].message)
        var jobs = List[Self.J](capacity=len(self.slots))
        self.slots.reverse()
        while len(self.slots) > 0:
            jobs.append(self.slots.pop().into_job())
        return jobs^

    def __deinit__(deinit self):
        # A scanner may raise before EOF. Close and drain before destroying
        # slots so no worker can retain a pointer into this producer.
        self._close()


struct Pool(Movable):
    """Worker threads reused across rounds, for the life of one operation.

    Create one where several rounds of jobs are about to run, call `run` per
    round, and `release` when done -- which joins every worker. Nothing may
    outlive `release`, so a pool must not be stored anywhere that survives
    the call that made it; see the module docstring for why.
    """

    var address: Int
    # Thread ids, kept beside the pool so `release` can join them.
    var _ids: List[UInt64]
    # (shared address, participant index) pairs, one per worker, on the C
    # heap so a worker's argument stays valid however the pool is moved.
    var _args: Int

    def __init__(out self, workers: Int):
        """Start `workers` - 1 threads; the caller is the remaining worker."""
        self.address = 0
        self._ids = List[UInt64]()
        self._args = 0
        if workers <= 1:
            return
        var address = external_call["malloc", Int](size_of[_Shared]())
        if address == 0:
            return
        var pair_bytes = 2 * size_of[Int]()
        var args = external_call["malloc", Int]((workers - 1) * pair_bytes)
        if args == 0:
            _ = external_call["free", NoneType](address)
            return
        var pointer = Pointer[_Shared, MutAnyOrigin](
            unsafe_from_address=address
        )
        pointer.unsafe_write(_Shared())
        ref shared = pointer[]
        _ = external_call["pthread_mutex_init", Int32](shared._mutex(), 0)
        _ = external_call["pthread_cond_init", Int32](shared._cond(), 0)
        var entry: _Entry = _worker
        var entry_address = Pointer(to=entry).unsafe_bitcast[Int]()[]
        var threads = List[UInt64](length=workers - 1, fill=0)
        var started = 0
        for t in range(workers - 1):
            var slot = args + t * pair_bytes
            Pointer[Int, MutAnyOrigin](unsafe_from_address=slot)[] = address
            Pointer[Int, MutAnyOrigin](
                unsafe_from_address=slot + size_of[Int]()
            )[] = t
            var rc = external_call["pthread_create", Int32](
                Int(threads.unsafe_ptr()) + 8 * t,
                0,
                entry_address,
                slot,
            )
            if rc != 0:
                break
            started += 1
        # Written after the last create, so a worker's participant count is
        # the number that actually started. No worker reads it before the
        # first round, which cannot begin until this returns.
        shared.threads = started
        self.address = address
        self._args = args
        self._ids = threads^

    def workers(self) -> Int:
        """Threads started, not counting the caller."""
        if self.address == 0:
            return 0
        return Pointer[_Shared, MutAnyOrigin](
            unsafe_from_address=self.address
        )[].threads

    def run_produced[J: Job](mut self, mut jobs: _ProducedJobs[J]):
        """Start a produced round; call `submit` while discovering work."""
        jobs._begin(self.address if self.workers() > 0 else 0)

    def run[J: Job](mut self, mut jobs: List[J], *, claim: Bool = False) raises:
        """Run one round of jobs, returning them with their results.

        Same contract as `run_jobs`: jobs come back in the order given, and
        the first error in job order is re-raised once the round has
        finished. With no workers, every job runs on the caller. `claim`
        dynamically schedules coarse, uneven jobs; the default preserves the
        static shares used by sort's fine, balanced rounds.
        """
        if len(jobs) == 0:
            return
        var slots = List[_Slot[J]](capacity=len(jobs))
        while len(jobs) > 0:
            slots.append(_Slot[J](jobs.pop(0)))

        if self.workers() == 0 or len(slots) == 1:
            for t in range(len(slots)):
                slots[t].run()
        else:
            ref shared = Pointer[_Shared, MutAnyOrigin](
                unsafe_from_address=self.address
            )[]
            var entry: _Entry = _entry[J]
            var entry_address = Pointer(to=entry).unsafe_bitcast[Int]()[]
            var tasks = List[_Task](capacity=len(slots))
            for t in range(len(slots)):
                tasks.append(_Task(entry_address, Int(Pointer(to=slots[t]))))
            shared.tasks.store(Int64(Int(Pointer(to=tasks))))
            shared.count.store(Int64(len(tasks)))
            shared.next.store(0)
            shared.done.store(0)
            shared.arrived.store(0)
            shared.closed.store(0)
            shared.mode.store(Int64(1 if claim else 0))
            _ = external_call["pthread_mutex_lock", Int32](shared._mutex())
            _ = shared.generation.fetch_add(1)
            _ = external_call["pthread_cond_broadcast", Int32](shared._cond())
            _ = external_call["pthread_mutex_unlock", Int32](shared._mutex())
            if claim:
                shared._run_claim()
            else:
                # The caller takes the last share, as run_jobs has it run the
                # last job itself.
                shared._run_share(shared.threads, shared.threads + 1)
            while shared.done.load() < Int64(len(tasks)):
                _ = external_call["sched_yield", Int32]()
            while shared.arrived.load() < Int64(shared.threads):
                _ = external_call["sched_yield", Int32]()
            # Close the round before opening the next: a late claimant must
            # see an empty round, never a half-published one.
            shared.count.store(0)
            shared.tasks.store(0)
            shared.mode.store(0)
            # `tasks` and `slots` must outlive every worker's use of them.
            _ = tasks^

        for t in range(len(slots)):
            if slots[t].failed:
                raise Error(slots[t].message)
        while len(slots) > 0:
            jobs.append(slots.pop(0).into_job())

    def release(mut self):
        """Wake every worker to return, join them, and free the state.

        Joining is what keeps a pool safe: no thread of this pool is alive
        once this returns, so nothing of ours is parked in JIT-compiled code
        when the process exits. Idempotent.
        """
        if self.address == 0:
            return
        ref shared = Pointer[_Shared, MutAnyOrigin](
            unsafe_from_address=self.address
        )[]
        _ = external_call["pthread_mutex_lock", Int32](shared._mutex())
        shared.stopping.store(1)
        _ = external_call["pthread_cond_broadcast", Int32](shared._cond())
        _ = external_call["pthread_mutex_unlock", Int32](shared._mutex())
        for t in range(shared.threads):
            _ = external_call["pthread_join", Int32](self._ids[t], 0)
        _ = external_call["pthread_mutex_destroy", Int32](shared._mutex())
        _ = external_call["pthread_cond_destroy", Int32](shared._cond())
        _ = external_call["free", NoneType](self.address)
        if self._args != 0:
            _ = external_call["free", NoneType](self._args)
            self._args = 0
        self.address = 0

    def __deinit__(deinit self):
        """A pool that goes out of scope still joins its workers, so a raise
        on the way out cannot leak threads into process exit."""
        self.release()


def run_jobs[J: Job](mut jobs: List[J]) raises:
    """Run each job on its own thread and return them with their results."""
    if len(jobs) == 0:
        return
    var slots = List[_Slot[J]](capacity=len(jobs))
    while len(jobs) > 0:
        slots.append(_Slot[J](jobs.pop(0)))
    var entry: _Entry = _entry[J]
    var entry_address = Pointer(to=entry).unsafe_bitcast[Int]()[]
    var spawned = len(slots) - 1
    var threads = List[UInt64](length=spawned, fill=0)
    var started = 0
    var spawn_error = String()
    for t in range(spawned):
        var rc = external_call["pthread_create", Int32](
            Int(threads.unsafe_ptr()) + 8 * t,
            0,
            entry_address,
            Int(Pointer(to=slots[t])),
        )
        if rc != 0:
            spawn_error = "pthread_create failed with code " + String(rc)
            break
        started += 1
    # The caller does the last job itself (or every unstarted job on error).
    for t in range(started, len(slots)):
        slots[t].run()
    for t in range(started):
        _ = external_call["pthread_join", Int32](threads[t], 0)
    if spawn_error:
        raise Error(spawn_error)
    for t in range(len(slots)):
        if slots[t].failed:
            raise Error(slots[t].message)
    while len(slots) > 0:
        jobs.append(slots.pop(0).into_job())


def configured_workers() -> Int:
    """The thread count itself, before any per-stage minimum is applied:
    `DATAFRAME_THREADS` if set, else the physical core count.

    A stage whose work per row is more than a scan -- sorting, which is
    n log n -- can divide further than `MIN_ROWS_PER_WORKER` allows and
    still keep every thread busy, so it sizes itself from this directly.
    """
    var configured = num_physical_cores()
    var setting = getenv("DATAFRAME_THREADS")
    if setting:
        try:
            configured = Int(setting)
        except:
            pass
    return max(1, configured)


def worker_count(rows: Int) -> Int:
    """Threads to use for `rows` rows (see the module docstring)."""
    return max(1, min(configured_workers(), rows // MIN_ROWS_PER_WORKER))


def partitions(rows: Int, workers: Int, align: Int) -> List[Int]:
    """Boundaries of `workers` contiguous ranges covering [0, rows), each a
    multiple of `align` rows except the last. Returns workers + 1 offsets."""
    var bounds = List[Int](capacity=workers + 1)
    var step = (rows + workers - 1) // max(workers, 1)
    step = ((step + align - 1) // align) * align
    for w in range(workers):
        bounds.append(min(rows, w * step))
    bounds.append(rows)
    return bounds^
