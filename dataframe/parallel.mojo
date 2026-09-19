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
"""
from std.ffi import external_call
from std.memory import Pointer
from std.os import getenv
from std.sys import num_physical_cores

comptime MIN_ROWS_PER_WORKER = 65536


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
