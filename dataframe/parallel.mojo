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

Threads come from one process-wide budget of `DATAFRAME_THREADS` - 1
helpers, shared by every `run_jobs` call and `Pool` round, nested or
concurrent, so the setting is a limit on threads running jobs (#390).

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
from std.ffi import _get_global, external_call
from std.memory import Pointer
from std.os import getenv
from std.time import perf_counter_ns
from std.sys import CompilationTarget, num_physical_cores, size_of

# Threadripper 3970X and M1, 4/8/16-worker sweeps: 16k helps 100k-row
# expressions on M1 but makes cheap counts 5-7x slower and oversubscribed
# grouping 4x slower. Keep the shared 64k floor; per-operation tuning is
# separate. See docs/worker-calibration.md.
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


def _read_performance_core_count(
    name: List[UInt8], mut count: Int32, mut size: Int
) -> Int32:
    comptime if CompilationTarget.is_macos():
        return external_call["sysctlbyname", Int32](
            name.unsafe_ptr(), Pointer(to=count), Pointer(to=size), 0, 0
        )
    else:
        return -1


def _performance_core_count() -> Int:
    """Physical cores, using the performance cluster on heterogeneous Macs."""
    var cores = num_physical_cores()
    comptime if CompilationTarget.is_macos():
        # On heterogeneous Apple Silicon, count performance cores rather
        # than all cores. Fall back to the physical count on older systems.
        var name = List[UInt8]()
        name.extend(String("hw.perflevel0.physicalcpu").as_bytes())
        name.append(0)
        var count = Int32(0)
        var size = size_of[Int32]()
        if _read_performance_core_count(name, count, size) == 0 and count > 0:
            cores = min(cores, Int(count))
    return max(1, cores)


def _pool_spin_limit(participants: Int) -> Int:
    """Park when spinning would compete with workers that need CPU time."""
    var cores = _performance_core_count()
    # M1 (4P+4E), Mojo 1.2: at 8 workers, 1M-row merge sort improves
    # 49 -> 43 ms with no spin; at 16, 100k rows improve 27 -> 3 ms.
    # Four workers still benefit from spinning (3.7 -> 3.1 ms).
    # Threadripper and M1 sweeps: docs/worker-calibration.md (#273).
    return 0 if participants > cores else _SPIN_LIMIT


def _budget_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    var address = external_call["calloc", Int](1, size_of[Atomic[Int64]]())
    return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=address)


def _budget_keep(address: Optional[Pointer[NoneType, MutUntrackedOrigin]]):
    # Never freed: a worker of another operation may still decrement it
    # while the process exits.
    pass


def _helpers() -> ref[MutAnyOrigin] Atomic[Int64]:
    """Helper threads running jobs right now, across the whole process."""
    var address = _get_global[
        "DATAFRAME_HELPER_THREADS", _budget_init, _budget_keep
    ]()
    return Pointer[Atomic[Int64], MutAnyOrigin](
        unsafe_from_address=Int(address.value())
    )[]


def _reserve_helpers(wanted: Int) -> Int:
    """Claim up to `wanted` helper threads from the process-wide budget.

    The budget is `configured_workers() - 1` helpers: with the thread that
    asks, never more than `DATAFRAME_THREADS` threads run jobs at once
    (#390). A job that starts jobs of its own -- a grouped reduce inside a
    hash bucket -- therefore gets helpers only if others are idle, and
    otherwise runs its jobs on its own thread.
    """
    if wanted <= 0:
        return 0
    var limit = Int64(configured_workers() - 1)
    ref helpers = _helpers()
    var busy = helpers.load()
    while True:
        var granted = min(Int64(wanted), limit - busy)
        if granted <= 0:
            return 0
        if helpers.compare_exchange(busy, busy + granted):
            return Int(granted)


def _release_helpers(count: Int):
    if count > 0:
        _ = _helpers().fetch_sub(Int64(count))


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
    # Workers taking part in the current round: the first `participants`
    # by index. The rest arrive at once, as the budget had no room for them.
    var participants: Atomic[Int64]
    var spin_limit: Int

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
        self.participants = Atomic[Int64](0)
        self.spin_limit = _SPIN_LIMIT

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
            spins < shared.spin_limit
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
        var participants = Int(shared.participants.load())
        if index < participants:
            if shared.mode.load() == 1:
                shared._run_claim()
            elif shared.mode.load() == 2:
                shared._run_produced()
            else:
                shared._run_share(index, participants + 1)
            # Done with this round: the reservation made for this worker
            # is returned as soon as its share is.
            _release_helpers(1)
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
    var helpers: Int

    def __init__(out self, capacity: Int):
        self.helpers = 0
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
        self.helpers = _reserve_helpers(shared.threads)
        if self.helpers == 0:
            # Every helper is busy: run each job as it is submitted.
            self.started = False
            return
        shared.participants.store(Int64(self.helpers))
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
        # Each participating worker returned its own reservation.
        self.helpers = 0
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
    # While a crew runs (`Crew`), the pool starts no threads: its rounds go
    # to the crew, at most this many threads at once (0: not on a crew).
    var _on_crew: Int

    def __init__(out self, workers: Int):
        """Start `workers` - 1 threads; the caller is the remaining worker.
        Inside an operation running a crew, no threads are started: rounds
        run on the crew's threads, which stay warm from one stage to the
        next (a thread that slept through a stage starts the next one at a
        low clock), and a produced round starts the threads then."""
        self.address = 0
        self._ids = List[UInt64]()
        self._args = 0
        self._on_crew = 0
        if workers <= 1:
            return
        if _current_crew().load() != 0:
            self._on_crew = workers
            return
        self._start(workers)

    def _start(mut self, workers: Int):
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
        shared.spin_limit = _pool_spin_limit(workers)
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
            return max(0, self._on_crew - 1)
        return Pointer[_Shared, MutAnyOrigin](
            unsafe_from_address=self.address
        )[].threads

    def run_produced[J: Job](mut self, mut jobs: _ProducedJobs[J]):
        """Start a produced round; call `submit` while discovering work."""
        if self._on_crew > 1 and self.address == 0:
            self._start(self._on_crew)
            self._on_crew = 0
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
        if self.address == 0 and self._on_crew > 1:
            _run_jobs(jobs, self._on_crew)
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
            var helpers = _reserve_helpers(min(shared.threads, len(slots) - 1))
            shared.participants.store(Int64(helpers))
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
                shared._run_share(helpers, helpers + 1)
            while shared.done.load() < Int64(len(tasks)):
                _ = external_call["sched_yield", Int32]()
            while shared.arrived.load() < Int64(shared.threads):
                _ = external_call["sched_yield", Int32]()
            # Close the round before opening the next: a late claimant must
            # see an empty round, never a half-published one.
            shared.count.store(0)
            shared.tasks.store(0)
            shared.mode.store(0)
            # Each participating worker returned its own reservation.
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


struct _Claims(Movable):
    """A completed job list that a fixed number of threads share: each takes
    the next unclaimed slot until none are left."""

    var slots: Int
    var stride: Int
    var count: Int
    var entry: Int
    var next: Atomic[Int64]
    # Jobs finished, for a caller that waits on crew threads (`_Crew`).
    var done: Atomic[Int64]

    def __init__(out self, slots: Int, stride: Int, count: Int, entry: Int):
        self.slots = slots
        self.stride = stride
        self.count = count
        self.entry = entry
        self.next = Atomic[Int64](0)
        self.done = Atomic[Int64](0)

    def run(mut self):
        var entry = Pointer(to=self.entry).unsafe_bitcast[_Entry]()[]
        while True:
            var i = Int(self.next.fetch_add(1))
            if i >= self.count:
                return
            _ = entry(self.slots + i * self.stride)
            _ = self.done.fetch_add(1)


def _claim_worker(address: Int) abi("C") -> Int:
    Pointer[_Claims, MutAnyOrigin](unsafe_from_address=address)[].run()
    # Out of work: give the thread back to the budget now, not when the
    # round ends, so a job still running can start helpers of its own.
    _release_helpers(1)
    return 0


def _helper_entry[J: Job](address: Int) abi("C") -> Int:
    """`_entry` for a helper thread that runs one job, then returns itself
    to the budget."""
    Pointer[_Slot[J], MutAnyOrigin](unsafe_from_address=address)[].run()
    _release_helpers(1)
    return 0


# --- crew: helper threads for the run_jobs calls of one operation -------
#
# `run_jobs` starts threads for each call and joins them before returning,
# about 15-35 us a thread. A query makes hundreds of calls, most of them two
# jobs inside a stream's batch (TPC-DS q96: 363 calls, 737 jobs in a 6 ms
# query), so starting threads was a large share of short queries. A crew is
# a set of threads started once for an operation -- `LazyFrame.collect` --
# that every `run_jobs` call inside it posts its jobs to instead. As with
# `Pool`, it is joined when the operation ends (see the module docstring:
# threads must not outlive the operation). Calls made while no crew is
# running, or when every posting slot is taken, start threads as before.
# A `Pool` made inside the operation sends its rounds to the crew as well,
# so the same threads run every stage and none goes cold between them.

comptime _POSTINGS = 64
# How long a crew thread looks for work before parking. A thread that sleeps
# wakes on a core the CPU governor (schedutil) has clocked down, and ran a
# group-by's jobs 1.5x slower than a thread started fresh (PDS-H q1); one
# kept busy between stages stays at full clock. Gaps inside a query are
# mostly serial stretches of a few milliseconds, so 50 ms covers them; the
# crew is joined when the operation ends, so no thread spins past it.
comptime _CREW_SPIN_NS = 50_000_000


struct _Crew(Movable):
    """Crew state at a fixed heap address: the threads' lock and signal,
    the stop flag, and `_POSTINGS` postings. Posting p is two words at
    `postings + 16p`: the address of a `_Claims` (0 when free), and a state
    word holding the helpers still wanted (high 32 bits) and the crew
    threads running its jobs now (low 32 bits), changed together so a
    caller can stop new joiners and wait for the last one."""

    var mutex: Array[UInt8, _SYNC_BYTES]
    var cond: Array[UInt8, _SYNC_BYTES]
    var stopping: Atomic[Int64]
    var postings: Int
    var threads: Int

    def __init__(out self, postings: Int):
        self.mutex = Array[UInt8, _SYNC_BYTES](fill=0)
        self.cond = Array[UInt8, _SYNC_BYTES](fill=0)
        self.stopping = Atomic[Int64](0)
        self.postings = postings
        self.threads = 0

    def _mutex(self) -> Int:
        return Int(self.mutex.unsafe_ptr())

    def _cond(self) -> Int:
        return Int(self.cond.unsafe_ptr())

    def _claims(self, p: Int) -> ref[MutAnyOrigin] Atomic[Int64]:
        return Pointer[Atomic[Int64], MutAnyOrigin](
            unsafe_from_address=self.postings + 16 * p
        )[]

    def _state(self, p: Int) -> ref[MutAnyOrigin] Atomic[Int64]:
        return Pointer[Atomic[Int64], MutAnyOrigin](
            unsafe_from_address=self.postings + 16 * p + 8
        )[]

    def _take(mut self) -> Int:
        """Join a posting that still wants a helper: its index, or -1."""
        for p in range(_POSTINGS):
            var state = self._state(p).load()
            while (state >> 32) > 0:
                # One fewer wanted, one more running, in one step.
                if self._state(p).compare_exchange(
                    state, state - (Int64(1) << 32) + 1
                ):
                    return p
                state = self._state(p).load()
        return -1

    def _wanted(mut self) -> Bool:
        for p in range(_POSTINGS):
            if (self._state(p).load() >> 32) > 0:
                return True
        return False

    def post(mut self, claims: Int, helpers: Int) -> Int:
        """Offer a job list to `helpers` crew threads; the posting's index,
        or -1 when every posting is taken."""
        for p in range(_POSTINGS):
            var free = Int64(0)
            if self._claims(p).compare_exchange(free, Int64(claims)):
                self._state(p).store(Int64(helpers) << 32)
                _ = external_call["pthread_mutex_lock", Int32](self._mutex())
                _ = external_call["pthread_cond_broadcast", Int32](self._cond())
                _ = external_call["pthread_mutex_unlock", Int32](self._mutex())
                return p
        return -1

    def retire(mut self, p: Int, claims: Int) -> Int:
        """Wait until the posting's jobs are done and no crew thread is in
        it, then free it. New helpers are turned away first; returns how
        many never joined, whose budget the caller still holds."""
        ref list = Pointer[_Claims, MutAnyOrigin](unsafe_from_address=claims)[]
        var unjoined = 0
        while True:
            var state = self._state(p).load()
            if self._state(p).compare_exchange(state, state & 0xFFFFFFFF):
                unjoined = Int(state >> 32)
                break
        while self._state(p).load() != 0 or list.done.load() < Int64(
            list.count
        ):
            _ = external_call["sched_yield", Int32]()
        self._claims(p).store(0)
        return unjoined


def _crew_worker(address: Int) abi("C") -> Int:
    ref crew = Pointer[_Crew, MutAnyOrigin](unsafe_from_address=address)[]
    while True:
        if crew.stopping.load() != 0:
            return 0
        var p = crew._take()
        if p < 0:
            var deadline = perf_counter_ns() + _CREW_SPIN_NS
            while p < 0 and crew.stopping.load() == 0:
                for _ in range(64):
                    p = crew._take()
                    if p >= 0:
                        break
                if perf_counter_ns() > deadline:
                    break
        if p < 0:
            _ = external_call["pthread_mutex_lock", Int32](crew._mutex())
            while not crew._wanted() and crew.stopping.load() == 0:
                _ = external_call["pthread_cond_wait", Int32](
                    crew._cond(), crew._mutex()
                )
            _ = external_call["pthread_mutex_unlock", Int32](crew._mutex())
            continue
        Pointer[_Claims, MutAnyOrigin](
            unsafe_from_address=Int(crew._claims(p).load())
        )[].run()
        # Out of work: give the thread back to the budget now, as
        # `_claim_worker` does, so a job still running can start helpers.
        _release_helpers(1)
        _ = crew._state(p).fetch_sub(1)


def _crew_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    var address = external_call["calloc", Int](1, size_of[Atomic[Int64]]())
    return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=address)


def _crew_keep(address: Optional[Pointer[NoneType, MutUntrackedOrigin]]):
    pass


def _current_crew() -> ref[MutAnyOrigin] Atomic[Int64]:
    """The address of the running crew's state, or 0."""
    var address = _get_global["DATAFRAME_CREW", _crew_init, _crew_keep]()
    return Pointer[Atomic[Int64], MutAnyOrigin](
        unsafe_from_address=Int(address.value())
    )[]


struct Crew(Movable):
    """Helper threads for the `run_jobs` calls of one operation. `start`
    begins one unless one is already running (then this one does nothing),
    and `release` must be called when the operation ends, error or not: it
    joins every thread."""

    var address: Int
    var ids: List[UInt64]

    def __init__(out self):
        self.address = 0
        self.ids = List[UInt64]()

    @staticmethod
    def start() -> Crew:
        var crew = Crew()
        var helpers = configured_workers() - 1
        if helpers <= 0 or _current_crew().load() != 0:
            return crew^
        var address = external_call["malloc", Int](size_of[_Crew]())
        var postings = external_call["calloc", Int](_POSTINGS, 16)
        if address == 0 or postings == 0:
            return crew^
        var pointer = Pointer[_Crew, MutAnyOrigin](unsafe_from_address=address)
        pointer.unsafe_write(_Crew(postings))
        ref state = pointer[]
        _ = external_call["pthread_mutex_init", Int32](state._mutex(), 0)
        _ = external_call["pthread_cond_init", Int32](state._cond(), 0)
        var entry: _Entry = _crew_worker
        var entry_address = Pointer(to=entry).unsafe_bitcast[Int]()[]
        var ids = List[UInt64](length=helpers, fill=0)
        var started = 0
        for t in range(helpers):
            var rc = external_call["pthread_create", Int32](
                Int(ids.unsafe_ptr()) + 8 * t, 0, entry_address, address
            )
            if rc != 0:
                break
            started += 1
        state.threads = started
        ids.resize(started, 0)
        crew.address = address
        crew.ids = ids^
        var none = Int64(0)
        if not _current_crew().compare_exchange(none, Int64(address)):
            # Another operation started one first: stop ours.
            crew.release()
        return crew^

    def release(mut self):
        """Stop and join the threads, and free the state. Idempotent."""
        if self.address == 0:
            return
        ref state = Pointer[_Crew, MutAnyOrigin](
            unsafe_from_address=self.address
        )[]
        var mine = Int64(self.address)
        _ = _current_crew().compare_exchange(mine, Int64(0))
        _ = external_call["pthread_mutex_lock", Int32](state._mutex())
        state.stopping.store(1)
        _ = external_call["pthread_cond_broadcast", Int32](state._cond())
        _ = external_call["pthread_mutex_unlock", Int32](state._mutex())
        for t in range(len(self.ids)):
            _ = external_call["pthread_join", Int32](self.ids[t], 0)
        _ = external_call["free", NoneType](state.postings)
        _ = external_call["free", NoneType](self.address)
        self.address = 0
        self.ids = List[UInt64]()


def _run_on_crew[
    J: Job
](mut slots: List[_Slot[J]], entry: Int, threads: Int) -> Bool:
    """Run every slot with crew threads and the caller; False (nothing
    run) when no crew is running, no helper is free in the budget, or every
    posting is taken."""
    var address = Int(_current_crew().load())
    if address == 0:
        return False
    var helpers = _reserve_helpers(min(len(slots), threads) - 1)
    if helpers == 0:
        return False
    ref crew = Pointer[_Crew, MutAnyOrigin](unsafe_from_address=address)[]
    var claims = _Claims(
        Int(slots.unsafe_ptr()), size_of[_Slot[J]](), len(slots), entry
    )
    var claims_address = Int(Pointer(to=claims))
    var p = crew.post(claims_address, helpers)
    if p < 0:
        _release_helpers(helpers)
        return False
    claims.run()
    _release_helpers(crew.retire(p, claims_address))
    _ = claims^
    return True


def run_jobs[J: Job](mut jobs: List[J]) raises:
    """Run every job and return them, in order, with their results.

    Threads come from the process-wide budget (`_reserve_helpers`), so at
    most `configured_workers()` threads run jobs at once, across nested and
    concurrent calls (#390). When the budget covers a thread per job, the
    caller runs the last one. Otherwise the threads it does cover, and the
    caller, claim jobs one at a time until none are left: a caller may cut
    work finer than the thread count for balance, and a job that starts
    jobs while every helper is busy runs them on its own thread.
    """
    _run_jobs(jobs, Int.MAX)


def _run_jobs[J: Job](mut jobs: List[J], limit: Int) raises:
    """`run_jobs` on at most `limit` threads, the caller included."""
    if len(jobs) == 0:
        return
    var slots = List[_Slot[J]](capacity=len(jobs))
    while len(jobs) > 0:
        slots.append(_Slot[J](jobs.pop(0)))
    var entry: _Entry = _entry[J]
    var entry_address = Pointer(to=entry).unsafe_bitcast[Int]()[]
    if len(slots) > 1 and _run_on_crew(slots, entry_address, limit):
        for t in range(len(slots)):
            if slots[t].failed:
                raise Error(slots[t].message)
        while len(slots) > 0:
            jobs.append(slots.pop(0).into_job())
        return
    var helper: _Entry = _helper_entry[J]
    var helper_address = Pointer(to=helper).unsafe_bitcast[Int]()[]
    var spawned = _reserve_helpers(min(len(slots), limit) - 1)
    # With a thread per job, the caller runs the last one; with fewer, the
    # threads and the caller claim jobs until none are left.
    var claiming = spawned < len(slots) - 1
    var claims = _Claims(
        Int(slots.unsafe_ptr()), size_of[_Slot[J]](), len(slots), entry_address
    )
    var threads = List[UInt64](length=spawned, fill=0)
    var started = 0
    var spawn_error = String()
    var claim_entry: _Entry = _claim_worker
    var claim_address = Pointer(to=claim_entry).unsafe_bitcast[Int]()[]
    for t in range(spawned):
        var rc: Int32
        if claiming:
            rc = external_call["pthread_create", Int32](
                Int(threads.unsafe_ptr()) + 8 * t,
                0,
                claim_address,
                Int(Pointer(to=claims)),
            )
        else:
            rc = external_call["pthread_create", Int32](
                Int(threads.unsafe_ptr()) + 8 * t,
                0,
                helper_address,
                Int(Pointer(to=slots[t])),
            )
        if rc != 0:
            spawn_error = "pthread_create failed with code " + String(rc)
            break
        started += 1
    if claiming:
        # The caller claims too; if a thread failed to start, it and the
        # threads that did start still finish every job.
        claims.run()
    else:
        # The caller does the last job itself (or every unstarted job on
        # error).
        for t in range(started, len(slots)):
            slots[t].run()
    for t in range(started):
        _ = external_call["pthread_join", Int32](threads[t], 0)
    # Started threads returned their own; these never started.
    _release_helpers(spawned - started)
    _ = claims^
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
