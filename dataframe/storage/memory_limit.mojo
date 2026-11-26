"""The process-wide memory limit and the count of memory held against it.

Without a limit, a query that needs more memory than the machine (or the
container it runs in) allows is stopped by the operating system: the kernel
kills the process outright, with no error, and every table the program held
is lost with it. The engine cannot avoid needing the memory, but it can
notice that it is about to cross a line and fail the query with an ordinary
error the program can catch, leaving the process and its other tables
intact.

So the large allocations the engine holds -- column buffers, the row and
heap blocks of hash tables and aggregates, the freed blocks kept for reuse
-- are counted in one process-wide total (`charge` when they are made,
`uncharge` when they are freed). Small allocations (a vector's scratch,
names, plans) are not counted: they are a small, roughly fixed share of the
process, which the default limit's headroom covers.

The limit is checked where an error can be raised: before the engine makes
an allocation whose size it knows in advance, and between the pieces of
work every worker thread does (each morsel of a pipeline, each scheduled
task, each round of jobs). Crossing it raises an error that starts with
`out of memory`.

The count misses what is not counted -- small allocations, a library's
own buffers, string dictionaries -- and on a small machine or a tight
container that share is large enough to matter. So a check also compares
the process's resident anonymous memory (the memory the operating system
charges it for) with the limit, read from `/proc/self/statm`. Reading it is
a system call, so it is read at most once a millisecond and checks in
between use the last reading: a process cannot grow by more than a few tens
of MB in a millisecond (memory bandwidth bounds how fast pages are
written), which is small against any limit.

The default limit is 80% of the memory available to the process: physical
memory, or the control group's limit when the process runs under one (a
container, a systemd scope) and that is lower. The remaining fifth is
headroom for uncounted allocations and for what is allocated between two
checks. `DATAFRAME_MEMORY_LIMIT` sets it at startup and `set_memory_limit`
at any time.
"""
from std.atomic import Atomic
from std.ffi import _get_global, external_call
from std.memory import Layout, Pointer, alloc
from std.os import getenv
from std.sys import CompilationTarget
from std.time import monotonic

# Linux `sysconf` names.
comptime _SC_PAGESIZE = 30
comptime _SC_PHYS_PAGES = 85

# The limit while unset (computed on first use) and when there is none.
comptime _UNSET = Int64(-1)
comptime UNLIMITED = Int64.MAX

# The share of available memory the default limit allows, in percent.
comptime DEFAULT_LIMIT_PERCENT = 80


struct _MemoryState(Movable):
    """Bytes held (`used`) and the limit, shared by every thread. Both are
    atomics reached by address (`_memory_state`), never through a `mut`
    argument, so every thread sees every other's changes."""

    var used: Atomic[Int64]
    var limit: Atomic[Int64]
    # The last reading of resident anonymous memory and when it was taken
    # (`monotonic` nanoseconds; 0 before the first).
    var resident: Atomic[Int64]
    var resident_at: Atomic[Int64]

    def __init__(out self):
        self.used = Atomic[Int64](0)
        self.limit = Atomic[Int64](_UNSET)
        self.resident = Atomic[Int64](0)
        self.resident_at = Atomic[Int64](0)


def _state_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    var pointer = alloc(Layout[_MemoryState](length=1)).unsafe_leak()
    pointer.unsafe_write(_MemoryState())
    return pointer.unsafe_bitcast[NoneType]()


def _state_keep(address: Optional[Pointer[NoneType, MutUntrackedOrigin]]):
    # Never freed: buffers are still released while the process exits.
    pass


@always_inline
def _memory_state() -> ref[MutAnyOrigin] _MemoryState:
    var address = _get_global[
        "DATAFRAME_MEMORY_STATE", _state_init, _state_keep
    ]()
    return Pointer[_MemoryState, MutAnyOrigin](
        unsafe_from_address=Int(address.value())
    )[]


@always_inline
def charge(bytes: Int):
    """Count `bytes` the engine now holds. Never fails: the caller has the
    memory already, and the next check raises if it went over the limit.
    """
    if bytes > 0:
        _ = _memory_state().used.fetch_add(Int64(bytes))


@always_inline
def uncharge(bytes: Int):
    """`bytes` counted by `charge` were freed."""
    if bytes > 0:
        _ = _memory_state().used.fetch_sub(Int64(bytes))


def memory_usage() -> Int:
    """Bytes the engine holds against the memory limit: column buffers,
    hash table and aggregate blocks, and freed blocks kept for reuse."""
    return Int(_memory_state().used.load())


def memory_limit() -> Int:
    """The memory limit in bytes (`Int64.MAX` when there is none): the
    last `set_memory_limit`, else `DATAFRAME_MEMORY_LIMIT`, else 80% of
    the memory available to the process (see `available_memory`)."""
    ref limit = _memory_state().limit
    var value = limit.load()
    if value != _UNSET:
        return Int(value)
    var computed = Int64(_default_limit())
    var expected = _UNSET
    if limit.compare_exchange(expected, computed):
        return Int(computed)
    return Int(limit.load())


def set_memory_limit(bytes: Int):
    """Limit the memory the engine holds to `bytes`; 0 or less removes the
    limit. Work that would go over it raises an `out of memory` error.
    Tables already held count toward it, so a limit below what the program
    holds makes the next query fail."""
    _memory_state().limit.store(Int64(bytes) if bytes > 0 else UNLIMITED)


def set_memory_limit(text: String) raises:
    """Set the limit from text: a size such as `"4GB"`, `"512MiB"` or
    `"1000000"` (bytes), a share of available memory such as `"50%"`, or
    `"none"` for no limit (see `parse_memory_limit`)."""
    set_memory_limit(parse_memory_limit(text))


def parse_memory_limit(text: String) raises -> Int:
    """Bytes for a memory limit written as text, 0 for none.

    Sizes take a decimal number and a unit: `B`, `KB`/`MB`/`GB`/`TB`
    (powers of 1000) or `KiB`/`MiB`/`GiB`/`TiB` (powers of 1024), any
    case, spaces allowed; a bare number is bytes. `N%` is that share of
    `available_memory()`. `none`, `unlimited` and `-1` mean no limit."""
    var s = String(text.strip()).lower()
    if s == "none" or s == "unlimited" or s == "-1" or s == "":
        return 0
    var bytes = s.as_bytes()
    var end = 0
    var seen_point = False
    while end < len(bytes):
        var c = Int(bytes[end])
        if c >= 48 and c <= 57:
            end += 1
        elif c == 46 and not seen_point:
            seen_point = True
            end += 1
        else:
            break
    if end == 0:
        raise Error("invalid memory limit '" + text + "'")
    var number = Float64(String(s[byte=0:end]))
    var unit = String(String(s[byte=end:]).strip())
    var scale: Float64
    if unit == "%":
        if number > 100.0:
            raise Error("invalid memory limit '" + text + "': above 100%")
        return Int(Float64(available_memory()) * number / 100.0)
    elif unit == "" or unit == "b":
        scale = 1.0
    elif unit == "k" or unit == "kb":
        scale = 1e3
    elif unit == "m" or unit == "mb":
        scale = 1e6
    elif unit == "g" or unit == "gb":
        scale = 1e9
    elif unit == "t" or unit == "tb":
        scale = 1e12
    elif unit == "kib":
        scale = 1024.0
    elif unit == "mib":
        scale = 1024.0 * 1024.0
    elif unit == "gib":
        scale = 1024.0 * 1024.0 * 1024.0
    elif unit == "tib":
        scale = 1024.0 * 1024.0 * 1024.0 * 1024.0
    else:
        raise Error(
            "invalid memory limit '"
            + text
            + "': unknown unit (use B, KB, MB, GB, TB, KiB, MiB, GiB, TiB"
            " or %)"
        )
    return Int(number * scale)


def _default_limit() -> Int:
    var setting = getenv("DATAFRAME_MEMORY_LIMIT")
    if setting:
        try:
            var bytes = parse_memory_limit(setting)
            return bytes if bytes > 0 else Int(UNLIMITED)
        except:
            # An unreadable setting falls back to the default rather than
            # failing every query.
            pass
    var available = available_memory()
    if available <= 0:
        return Int(UNLIMITED)
    return available // 100 * DEFAULT_LIMIT_PERCENT


def physical_memory() -> Int:
    """The machine's physical memory in bytes, 0 where unknown."""
    comptime if CompilationTarget.is_linux():
        var pages = Int(external_call["sysconf", Int64](Int32(_SC_PHYS_PAGES)))
        var size = Int(external_call["sysconf", Int64](Int32(_SC_PAGESIZE)))
        if pages <= 0 or size <= 0:
            return 0
        return pages * size
    elif CompilationTarget.is_macos():
        var bytes = Int64(0)
        var length = Int(8)
        var name = String("hw.memsize")
        var c_name = name.as_c_string_span()
        var status = external_call["sysctlbyname", Int32](
            c_name.ptr(),
            Pointer(to=bytes),
            Pointer(to=length),
            Int(0),
            Int(0),
        )
        _ = name^
        if status != 0:
            return 0
        return Int(bytes)
    else:
        return 0


def available_memory() -> Int:
    """Memory the process may use: physical memory, or the memory limit of
    the control group it runs in (and of every group above it) when that is
    lower. 0 where neither is known."""
    var physical = physical_memory()
    var group = cgroup_memory_limit()
    if group > 0 and (physical <= 0 or group < physical):
        return group
    return physical


def _read_file(path: String) -> String:
    try:
        with open(path, "r") as file:
            return file.read()
    except:
        return String()


def _parse_bytes_value(text: String) -> Int:
    """A control group limit file's value: a byte count, or 0 for `max`,
    an empty file or anything else unreadable."""
    var value = 0
    var digits = 0
    for b in text.as_bytes():
        var c = Int(b)
        if c >= 48 and c <= 57:
            if value > (Int(Int64.MAX) - 9) // 10:
                return 0
            value = 10 * value + c - 48
            digits += 1
        elif c == 10 or c == 32:
            break
        else:
            return 0
    return value if digits > 0 else 0


def _smaller_limit(current: Int, found: Int) -> Int:
    if found <= 0:
        return current
    if current <= 0 or found < current:
        return found
    return current


def cgroup_memory_limit(
    proc_cgroup: String = "/proc/self/cgroup",
    root: String = "/sys/fs/cgroup",
) -> Int:
    """The lowest memory limit of the control group this process runs in
    and of the groups above it, in bytes; 0 when there is none.

    `/proc/self/cgroup` names the group: a `0::PATH` line under cgroup v2,
    a `N:...memory...:PATH` line under v1. The limit is in `memory.max`
    (v2) or `memory.limit_in_bytes` (v1) of that group's directory. A
    limit set on a parent group binds the child too, so every directory up
    to the root is read and the smallest limit wins. A container often
    mounts its own group as the root, so the root directory's file is
    read as well. The arguments exist for tests."""
    comptime if not CompilationTarget.is_linux():
        return 0
    var text = _read_file(proc_cgroup)
    var limit = 0
    for line_slice in text.split("\n"):
        var line = String(line_slice)
        # hierarchy-id:controllers:path, where the path may hold colons.
        var first = line.find(":").or_else(-1)
        if first < 0:
            continue
        var second = line.find(":", first + 1).or_else(-1)
        if second < 0:
            continue
        var hierarchy = String(line[byte=0:first])
        var controllers = String(line[byte = first + 1 : second])
        var path = String(line[byte = second + 1 :])
        var directory: String
        var file: String
        if hierarchy == "0" and controllers == "":
            directory = root
            file = "memory.max"
        else:
            var has_memory = False
            for c in controllers.split(","):
                if String(c) == "memory":
                    has_memory = True
            if not has_memory:
                continue
            directory = root + "/memory"
            file = "memory.limit_in_bytes"
        # The group's own directory, each parent's, then the root's.
        var current = path
        while True:
            var dir = directory + (current if current != "/" else "")
            limit = _smaller_limit(
                limit, _parse_bytes_value(_read_file(dir + "/" + file))
            )
            if current == "/" or current == "":
                break
            var slash = current.rfind("/").or_else(-1)
            current = String(current[byte=0:slash]) if slash > 0 else "/"
    return limit


def format_bytes(bytes: Int) -> String:
    """`bytes` for a message: `512 B`, `1.5 KiB`, `3.2 GiB`."""
    if bytes < 1024:
        return String(bytes) + " B"
    var units = ["KiB", "MiB", "GiB", "TiB", "PiB"]
    var value = Float64(bytes) / 1024.0
    var u = 0
    while value >= 1024.0 and u < len(units) - 1:
        value /= 1024.0
        u += 1
    var tenths = Int(value * 10.0 + 0.5)
    return String(tenths // 10) + "." + String(tenths % 10) + " " + units[u]


# How long a reading of resident memory stands in for the next one.
comptime _RESIDENT_NANOSECONDS = 1_000_000


def resident_memory() -> Int:
    """The process's resident anonymous memory in bytes: resident pages
    that no file backs (heap and anonymous mappings), from
    `/proc/self/statm`; 0 where that is not readable. Mapped files are
    left out because the system can drop their pages instead of failing.
    """
    comptime if not CompilationTarget.is_linux():
        return 0
    var text = _read_file("/proc/self/statm")
    # size resident shared text lib data dt, in pages.
    var fields = text.split(" ")
    if len(fields) < 3:
        return 0
    var resident = _parse_bytes_value(String(fields[1]))
    var shared = _parse_bytes_value(String(fields[2]))
    var page = Int(external_call["sysconf", Int64](Int32(_SC_PAGESIZE)))
    return max(0, resident - shared) * max(page, 0)


@no_inline
def _recent_resident() -> Int:
    """The last reading of resident memory, read again when it is older
    than _RESIDENT_NANOSECONDS. One thread reads; the others use the old
    value meanwhile."""
    ref state = _memory_state()
    var now = Int64(monotonic())
    var at = state.resident_at.load()
    if at == 0 or now - at > _RESIDENT_NANOSECONDS:
        if state.resident_at.compare_exchange(at, now):
            state.resident.store(Int64(resident_memory()))
    return Int(state.resident.load())


def refresh_resident():
    """Read resident memory now, for a check right after memory was freed."""
    ref state = _memory_state()
    state.resident.store(Int64(resident_memory()))
    state.resident_at.store(Int64(monotonic()))


def _in_use() -> Int:
    return max(memory_usage(), Int(_memory_state().resident.load()))


def out_of_memory(requested: Int) -> Error:
    """The error raised when the memory limit is crossed: for an
    allocation of `requested` bytes, or 0 when the engine already holds
    more than the limit."""
    var used = _in_use()
    var limit = memory_limit()
    var what: String
    if requested > 0:
        what = "could not allocate " + format_bytes(requested) + " ("
    else:
        what = "the memory limit was crossed ("
    return Error(
        "out of memory: "
        + what
        + format_bytes(used)
        + " in use of the "
        + format_bytes(limit)
        + " limit). Free tables the program no longer needs, or raise the"
        " limit with set_memory_limit or DATAFRAME_MEMORY_LIMIT."
    )


@always_inline
def over_limit(extra: Int = 0) -> Bool:
    """Whether holding `extra` more bytes would go over the limit, by the
    count of held memory or by the process's resident memory."""
    ref state = _memory_state()
    var limit = state.limit.load()
    if limit == _UNSET:
        limit = Int64(memory_limit())
    if state.used.load() + Int64(extra) > limit:
        return True
    if limit == UNLIMITED:
        return False
    return Int64(_recent_resident() + extra) > limit


@always_inline
def over_limit_reserved(bytes: Int) -> Bool:
    """`over_limit` for `bytes` already counted (`charge`) but not yet
    allocated: they are in the count, and not yet resident."""
    ref state = _memory_state()
    var limit = state.limit.load()
    if limit == _UNSET:
        limit = Int64(memory_limit())
    if state.used.load() > limit:
        return True
    if limit == UNLIMITED:
        return False
    return Int64(_recent_resident() + bytes) > limit
