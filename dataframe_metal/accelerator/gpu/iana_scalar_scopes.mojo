"""Device-only scalar observation masks for certified CPU evaluation batches.

The IANA interpreter prefix probe supplies incoming active flags. Only the
ANY reduction lives here; no source payload, predicate or interval is
computed on the host. Dense scalar slots retain independent cache owners.
"""
from max.gpu import global_idx
from std.memory import Pointer
from dataframe_metal.accelerator.memory import checked_add, checked_mul


@fieldwise_init
struct Resources(Copyable):
    var batches: Int
    var scratch_bytes: Int
    var cache_bytes: Int
    var workspace_bytes: Int


def resources(
    rows: Int, scalar_slots: Int, batch_rows: Int = 8192
) raises -> Resources:
    if (
        rows < 0
        or rows > Int(Int32.MAX)
        or scalar_slots < 0
        or scalar_slots > 128
        or batch_rows < 1
        or batch_rows > Int(UInt32.MAX)
    ):
        raise Error("IANA scalar observation geometry is invalid")
    var batches = rows // batch_rows + Int(rows % batch_rows != 0)
    var scratch_bytes = rows if scalar_slots else 0
    var cache_bytes = checked_mul(scalar_slots, batches)
    return Resources(
        batches,
        scratch_bytes,
        cache_bytes,
        checked_add(scratch_bytes, cache_bytes),
    )


def reduce_scalar_any(
    active: Pointer[UInt8, ImmutAnyOrigin],
    cache: Pointer[UInt8, MutAnyOrigin],
    rows: UInt32,
    batch_rows: UInt32,
    batches: UInt32,
    slot: UInt32,
):
    var batch = global_idx.x
    if not batch_rows or batch >= Int(batches):
        return
    var begin = batch * Int(batch_rows)
    var end = min(Int(rows), begin + Int(batch_rows))
    var seen = False
    for row in range(begin, end):
        seen |= active[unsafe_offset=row] != 0
    cache[unsafe_offset=Int(slot) * Int(batches) + batch] = UInt8(seen)
