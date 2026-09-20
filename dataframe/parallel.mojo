"""Worker threads for partitioned execution (POSIX threads via libc).

Mojo 1.1's standard library has no thread pool, so jobs run on threads
created with `pthread_create` through the C FFI. There is no new dependency:
libc provides pthreads on Linux and macOS.

A job is a self-contained value implementing `Job`: it owns (reference-counted)
copies of its inputs and writes only its own outputs, so no locking is
needed. `run_jobs` runs every job, joins the batch, and re-raises the first
job error on the caller (errors cannot unwind across threads). Shared column
buffers are safe to read concurrently: their reference counts are atomic and
they are never mutated while shared.

Threads are created once, on the first batch, and reused: a process-wide
pool of `num_physical_cores() - 1` workers sleeps on a condition variable
between batches, so a batch costs a wake-up (tens of microseconds for
32 workers) rather than thread creation (over a millisecond). The pool is
found again through the compiler runtime's keyed global registry, since Mojo
has no global variables. It is never destroyed: workers sleep until the
process exits.

A batch is a queue of tasks that workers and the calling thread claim with
an atomic counter, so any number of jobs works with any number of workers
and the caller never idles while work remains. Only one batch runs at a
time. A `run_jobs` call made while a batch is running -- from inside a job,
or from another thread -- runs its jobs inline on the calling thread, which
is how nested submission avoids deadlock.

`worker_count` sizes a batch: `DATAFRAME_THREADS` (1 disables parallelism),
else the physical core count, capped so each job gets at least
`MIN_ROWS_PER_WORKER` rows (small inputs stay single-threaded).
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import Pointer
from std.os import getenv
from std.sys import num_physical_cores, size_of

comptime MIN_ROWS_PER_WORKER = 8192

# Where the pool registers itself. The runtime keeps one entry per name for
# the life of the process.
comptime _POOL_KEY = "dataframe_mojo.thread_pool"

# pthread_mutex_t and pthread_cond_t are opaque and platform-sized (up to 64
# bytes on the platforms built for); over-allocating is harmless.
comptime _SYNC_BYTES = 128


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


@fieldwise_init
struct _Task(Copyable, Movable):
    """One job of a batch: the typed entry function and its slot."""

    var entry: Int
    var slot: Int

    def run(self):
        var entry = Pointer(to=self.entry).unsafe_bitcast[_Entry]()[]
        _ = entry(self.slot)


struct _Pool(Movable):
    """The process-wide worker pool. Lives on the C heap, never freed."""

    var mutex: Array[UInt8, _SYNC_BYTES]
    var cond: Array[UInt8, _SYNC_BYTES]
    # Bumped under the mutex to wake sleeping workers.
    var generation: Atomic[Int64]
    # The current batch: the address of its List[_Task], its length, the
    # next unclaimed task, and how many have finished. `tasks` and `count`
    # are written before `next` is reset to zero, which is what opens the
    # batch to claimants. Every field is atomic and read after the claim:
    # a claimant may have entered the loop during the previous batch, and a
    # plain `count` held in a register from then would make it drop a task
    # it had just claimed (the first version of this pool lost exactly one
    # task per hang that way).
    var tasks: Atomic[Int64]
    var count: Atomic[Int64]
    var next: Atomic[Int64]
    var done: Atomic[Int64]
    # Nonzero while a batch is running; claimed by test-and-increment.
    var busy: Atomic[Int64]
    var threads: Int

    def __init__(out self):
        self.mutex = Array[UInt8, _SYNC_BYTES](fill=0)
        self.cond = Array[UInt8, _SYNC_BYTES](fill=0)
        self.generation = Atomic[Int64](0)
        self.tasks = Atomic[Int64](0)
        self.count = Atomic[Int64](0)
        self.next = Atomic[Int64](0)
        self.done = Atomic[Int64](0)
        self.busy = Atomic[Int64](0)
        self.threads = 0

    def _mutex(self) -> Int:
        return Int(self.mutex.unsafe_ptr())

    def _cond(self) -> Int:
        return Int(self.cond.unsafe_ptr())

    def _claim_and_run(mut self):
        """Run tasks of the current batch until none are left to claim."""
        while True:
            var i = Int(self.next.fetch_add(1))
            # Loaded after the claim, so they describe the batch the index
            # came from (see the field comments).
            if i >= Int(self.count.load()):
                return
            Pointer[List[_Task], MutAnyOrigin](
                unsafe_from_address=Int(self.tasks.load())
            )[][i].run()
            _ = self.done.fetch_add(1)


def _worker(address: Int) abi("C") -> Int:
    ref pool = Pointer[_Pool, MutAnyOrigin](unsafe_from_address=address)[]
    var seen = Int64(0)
    while True:
        _ = external_call["pthread_mutex_lock", Int32](pool._mutex())
        while pool.generation.load() == seen:
            _ = external_call["pthread_cond_wait", Int32](
                pool._cond(), pool._mutex()
            )
        seen = pool.generation.load()
        _ = external_call["pthread_mutex_unlock", Int32](pool._mutex())
        pool._claim_and_run()


def _pool() raises -> Pointer[_Pool, MutAnyOrigin]:
    """The process-wide pool, created on first use."""
    var key = String(_POOL_KEY)
    var address = external_call["KGEN_CompilerRT_GetGlobalOrNull", Int](
        Int(key.unsafe_ptr()), key.byte_length()
    )
    if address != 0:
        return Pointer[_Pool, MutAnyOrigin](unsafe_from_address=address)

    address = external_call["malloc", Int](size_of[_Pool]())
    var pointer = Pointer[_Pool, MutAnyOrigin](unsafe_from_address=address)
    pointer.unsafe_write(_Pool())
    ref pool = pointer[]
    _ = external_call["pthread_mutex_init", Int32](pool._mutex(), 0)
    _ = external_call["pthread_cond_init", Int32](pool._cond(), 0)

    var entry: _Entry = _worker
    var entry_address = Pointer(to=entry).unsafe_bitcast[Int]()[]
    var wanted = max(0, num_physical_cores() - 1)
    var thread = UInt64(0)
    for _ in range(wanted):
        var rc = external_call["pthread_create", Int32](
            Int(Pointer(to=thread)), 0, entry_address, address
        )
        if rc != 0:
            break
        # Workers are never joined; detaching lets the OS reclaim them.
        _ = external_call["pthread_detach", Int32](thread)
        pool.threads += 1
    _ = external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        Int(key.unsafe_ptr()), key.byte_length(), address
    )
    _ = key^
    return pointer


def run_jobs[J: Job](mut jobs: List[J]) raises:
    """Run every job on the worker pool and return them with their results.

    Jobs come back in the order given. If any job raised, the first error
    (in job order) is re-raised after the whole batch has finished.
    """
    if len(jobs) == 0:
        return
    var slots = List[_Slot[J]](capacity=len(jobs))
    while len(jobs) > 0:
        slots.append(_Slot[J](jobs.pop(0)))

    if len(slots) == 1:
        slots[0].run()
    else:
        var entry: _Entry = _entry[J]
        var entry_address = Pointer(to=entry).unsafe_bitcast[Int]()[]
        var tasks = List[_Task](capacity=len(slots))
        for t in range(len(slots)):
            tasks.append(_Task(entry_address, Int(Pointer(to=slots[t]))))
        var pool = _pool()
        ref shared = pool[]
        # Test-and-increment: whoever moves `busy` from 0 owns the batch;
        # anyone else backs out and runs inline.
        var acquired = shared.threads > 0 and shared.busy.fetch_add(1) == 0
        if shared.threads > 0 and not acquired:
            _ = shared.busy.fetch_sub(1)
        if not acquired:
            # Nested or concurrent submission, or no workers: run inline.
            for t in range(len(tasks)):
                tasks[t].run()
        else:
            shared.tasks.store(Int64(Int(Pointer(to=tasks))))
            shared.count.store(Int64(len(tasks)))
            shared.done.store(0)
            shared.next.store(0)
            _ = external_call["pthread_mutex_lock", Int32](shared._mutex())
            _ = shared.generation.fetch_add(1)
            _ = external_call["pthread_cond_broadcast", Int32](shared._cond())
            _ = external_call["pthread_mutex_unlock", Int32](shared._mutex())
            shared._claim_and_run()
            while shared.done.load() < Int64(len(tasks)):
                _ = external_call["sched_yield", Int32]()
            # Close the batch before releasing it: a late claimant must see
            # an empty batch, never the next caller's half-published one.
            shared.count.store(0)
            shared.tasks.store(0)
            shared.busy.store(0)
        # `tasks` and `slots` must outlive every worker's use of them.
        _ = tasks^

    for t in range(len(slots)):
        if slots[t].failed:
            raise Error(slots[t].message)
    while len(slots) > 0:
        jobs.append(slots.pop(0).into_job())


def worker_count(rows: Int) -> Int:
    """Threads to use for `rows` rows (see the module docstring)."""
    var configured = num_physical_cores()
    var setting = getenv("DATAFRAME_THREADS")
    if setting:
        try:
            configured = Int(setting)
        except:
            pass
    return max(1, min(configured, rows // MIN_ROWS_PER_WORKER))


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
