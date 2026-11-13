"""Facts about join keys that planning reads before any join runs, and the
dense integer lookup `replace` uses.

- `int64_progression`: whether a key ascends by one step (kept with its
  buffer, `_BufferFacts.progression`). No join reads it since the hash
  join port; it stays with the buffer facts it publishes.
- `_perfect_slots`: integer keys' rows by `key - least` within
  `_PERFECT_SPAN`, the maximum direct-address slot span.
"""
from dataframe.series import Series
from dataframe.storage.column import Column
from std.collections import Optional


@always_inline
def int64_progression(key: Series) -> Tuple[Bool, Int64, UInt64]:
    """Base and step when the keys ascend from base by one positive step.

    The key must have physical Int64 storage. Every row must be valid and
    each key must exceed the previous one by the same amount, so row r holds
    base + r * step. Sorted sequential IDs and calendar grids have this
    shape. The scan stops at the first null or irregular step.

    Whether a whole buffer is one is found once and kept with it
    (`_BufferFacts.progression`): a window onto a known progression is
    checked only where it meets the chunk before.
    """
    if len(key) == 0:
        return (False, Int64(0), UInt64(1))
    var base = Int64(0)
    var previous = Int64(0)
    var stride = UInt64(0)
    var row = 0
    for part in key.chunks():
        ref column = part._data[Column[Int64]]
        var n = len(column)
        if n == 0:
            continue
        var known = column._facts[].progression_state()
        if (
            known == 0
            and column._offset == 0
            and column._length == len(column._data[])
        ):
            # The window is the whole buffer: what it finds holds for
            # every later window onto it.
            known = 1 if _progression_scan(column)[0] else 2
            column._facts[].publish_progression(known == 1)
            if known == 2:
                return (False, Int64(0), UInt64(1))
        if known != 1 and len(column._bits[]) > 0 and column.null_count() > 0:
            # Every row must be valid (a Parquet column carries a bitmap
            # even when none is null; it is counted, not read per row).
            return (False, Int64(0), UInt64(1))
        var values = column.unsafe_values()
        var start = 0
        if row == 0:
            base = values.unsafe_load(0)
            previous = base
            start = 1
        if stride == 0 and start < n:
            var first = values.unsafe_load(start)
            if first <= previous:
                return (False, Int64(0), UInt64(1))
            stride = UInt64(first) - UInt64(previous)
        if known == 1:
            # A window onto a progression: one step throughout, so only
            # where it meets the chunk before and its own step are read.
            var first = values.unsafe_load(0)
            if start == 0 and (
                first <= previous or UInt64(first) - UInt64(previous) != stride
            ):
                return (False, Int64(0), UInt64(1))
            if n >= 2 and (
                UInt64(values.unsafe_load(1)) - UInt64(first) != stride
            ):
                return (False, Int64(0), UInt64(1))
            previous = values.unsafe_load(n - 1)
            row += n
            continue
        # The values read straight from storage, checked a block at a
        # time: a checked read per row costs a branch and a bounds test
        # per key, which over a large key column dominates the join.
        var i = start
        while i < n:
            var stop = min(n, i + 1024)
            var bad = False
            for k in range(i, stop):
                var value = values.unsafe_load(k)
                # Unsigned subtraction is the exact distance of value >
                # previous; `|` keeps the loop free of branches.
                bad = (
                    bad
                    | (value <= previous)
                    | (UInt64(value) - UInt64(previous) != stride)
                )
                previous = value
            if bad:
                return (False, Int64(0), UInt64(1))
            i = stop
        row += n
    return (True, base, max(stride, UInt64(1)))


def _progression_scan(column: Column[Int64]) -> Tuple[Bool, UInt64]:
    """Whether the column's rows are valid and ascend by one positive step,
    and the step (1 for a single row)."""
    var n = len(column)
    if len(column._bits[]) > 0 and column.null_count() > 0:
        return (False, UInt64(1))
    if n < 2:
        return (n == 1, UInt64(1))
    var values = column.unsafe_values()
    var previous = values.unsafe_load(0)
    var first = values.unsafe_load(1)
    if first <= previous:
        return (False, UInt64(1))
    var stride = UInt64(first) - UInt64(previous)
    var i = 1
    while i < n:
        var stop = min(n, i + 1024)
        var bad = False
        for k in range(i, stop):
            var value = values.unsafe_load(k)
            bad = (
                bad
                | (value <= previous)
                | (UInt64(value) - UInt64(previous) != stride)
            )
            previous = value
        if bad:
            return (False, UInt64(1))
        i = stop
    return (True, stride)


def _l2_cache_bytes() -> Int:
    """A core's L2 cache size, from Linux's sysfs; 1 MiB where that is not
    readable."""
    try:
        with open("/sys/devices/system/cpu/cpu0/cache/index2/size", "r") as f:
            var text = f.read()
            var value = 0
            var scale = 1
            for b in text.as_bytes():
                var byte = Int(b)
                if byte >= 48 and byte <= 57:
                    value = 10 * value + byte - 48
                elif byte == 75:
                    scale = 1024
                elif byte == 77:
                    scale = 1024 * 1024
            if value > 0:
                return value * scale
    except:
        pass
    return 1 << 20


# Bound direct-address allocation even when key values are sparse.
comptime _PERFECT_SPAN = 1_048_576


def _perfect_slots(key: Series) -> Optional[Tuple[Int64, List[Int32]]]:
    """A build key's rows by `key - least`, when its values are integers,
    never null, unique, and span at most `_PERFECT_SPAN` in slots that stay
    in cache; None otherwise."""
    if key.null_count() > 0 or len(key) == 0 or len(key) > Int(Int32.MAX):
        return None
    ref column = key._data[Column[Int64]]
    var values = column._ptr()
    var n = len(column)
    var least = Int64.MAX
    var most = Int64.MIN
    for i in range(n):
        var v = values[unsafe_offset=i]
        least = min(least, v)
        most = max(most, v)
    # The span in unsigned arithmetic: keys spread over more than 2**63
    # (random 64-bit ids) overflow a signed `most - least`.
    var span = UInt64(most) - UInt64(least)
    if span >= UInt64(_PERFECT_SPAN):
        return None
    # The slots must stay in cache while every probe row reads one at
    # random: no larger than the hash table they replace (about 16 bytes
    # a build row), or within half a core's L2 cache. A few hundred keys
    # spread over a wide span would otherwise miss on nearly every probe.
    var slot_bytes = 4 * (Int(span) + 1)
    if slot_bytes > max(16 * n, _l2_cache_bytes() // 2):
        return None
    var slots = List[Int32](length=Int(span) + 1, fill=-1)
    var table = slots.unsafe_ptr()
    for i in range(n):
        var at = Int(UInt64(values[unsafe_offset=i]) - UInt64(least))
        if table[unsafe_offset=at] >= 0:
            return None
        table[unsafe_offset=at] = Int32(i)
    return (least, slots^)
