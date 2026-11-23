"""Native reduction inputs, shared timing and checked chunk grid geometry."""
from dataframe.series import Series
from std.time import perf_counter_ns
from .typed_storage import (
    NativeColumn,
    upload_native,
    wrap_native,
    coalesced_native_chunks,
)
from .query_scope import Scope
from .filter_sum import BLOCK_SIZE
from dataframe.execution_observation import DeviceObservation
from std.memory import ArcPointer

comptime MAX_PARTIALS = 1024
comptime MIN_CHUNK_PARTIALS = 64


struct HandoffTimings(Copyable, Movable):
    """Host wall time by phase, in nanoseconds. `kernel_ns` includes queue
    submission, device work, copies and waits; it is not GPU-only time."""

    var context_ns: Int
    var upload_ns: Int
    var kernel_ns: Int
    var finish_ns: Int

    def __init__(out self):
        self.context_ns = 0
        self.upload_ns = 0
        self.kernel_ns = 0
        self.finish_ns = 0

    def record(self, observation: Optional[ArcPointer[DeviceObservation]]):
        if observation:
            observation.value()[].provider = "metal"
            observation.value()[].phases(
                self.context_ns, self.upload_ns, self.kernel_ns, self.finish_ns
            )


def _place[
    dtype: DType
](scope: Scope, series: Series, zero_copy: Bool) raises -> NativeColumn[dtype]:
    """The column on the device: its own memory wrapped in place when
    `zero_copy` and the column is one chunk, else a copy."""
    if zero_copy:
        var wrapped = wrap_native[dtype](scope, series, wrap_validity=True)
        if wrapped:
            return wrapped.take()
    return upload_native[dtype](scope, series)


def _place_chunks[
    dtype: DType
](
    scope: Scope, series: Series, zero_copy: Bool, *, coalesce: Bool = False
) raises -> List[NativeColumn[dtype]]:
    """Keep independent native chunks separate, including their bitmap offsets.

    Adjacent windows with the same owners can share one zero-copy mapping.

    The caller keeps `series` alive through the batched reduction's final
    synchronization. Explicit copy mode remains a coalesced reference path.
    """
    var columns = List[NativeColumn[dtype]]()
    if zero_copy:
        var logical = series.chunks()
        var next = 0
        for part in coalesced_native_chunks[dtype](series):
            var mapped = _place[dtype](scope, part, True)
            if coalesce:
                columns.append(mapped^)
            else:
                # Share the mapping while retaining the original work ranges.
                # Changing reduction geometry can hurt throughput and changes
                # floating reduction order even though payloads are identical.
                var offset = 0
                while offset < len(part):
                    while next < len(logical) and len(logical[next]) == 0:
                        next += 1
                    if next >= len(logical):
                        raise Error("Metal native chunk ranges are incomplete")
                    var length = len(logical[next])
                    columns.append(mapped.view(offset, length))
                    offset += length
                    next += 1
    else:
        columns.append(_place[dtype](scope, series, False))
    return columns^


def _check_ran(counts: List[UInt32]) raises:
    """Raise unless every block wrote its count: a block holds fewer than
    2**31 rows, so the sentinel can only remain where a block never ran."""
    for count in counts:
        if count == UInt32.MAX:
            raise Error(
                "Mojo GPU hand-off kernel did not complete on the device"
            )


def _partials(rows: Int) -> Int:
    """Blocks to launch for `rows` rows: one per BLOCK_SIZE rows, capped
    at MAX_PARTIALS, with a grid stride covering the rest."""
    return max(1, min(MAX_PARTIALS, (rows + BLOCK_SIZE - 1) // BLOCK_SIZE))


def _chunk_partials(rows: Int, total_rows: Int) -> Int:
    """Share the grid budget across chunks instead of multiplying it.

    A small floor keeps a chunk's launch parallel; short chunks still use
    only the blocks their rows need. A single chunk keeps the original grid.
    """
    var share = (rows * MAX_PARTIALS + total_rows - 1) // total_rows
    return min(_partials(rows), max(MIN_CHUNK_PARTIALS, share))
