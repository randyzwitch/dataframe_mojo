"""Stable GPU row ordering over shared fixed-width dataframe words.

Sort descriptors are the shared RowPlan code and aligned physical type array.
Short merge passes rank each row against the opposite sorted run. Wider passes
use bounded output tiles and the same comparator. Source row indices break
ties, preserving NaN payloads, signed zeros and nullable values.
"""
from max.gpu import global_idx, thread_idx, block_idx, barrier
from max.gpu.primitives.block import prefix_sum
from std.memory import Pointer, unsafe_stack_allocation
from . import floating, decimal
from .rows import unsigned_type
from dataframe.accelerator.rows import ROW_GROUP_KEY

comptime MERGE_TILE_ROWS = 2048
comptime MERGE_MIN_RUN_WIDTH = 1024


@always_inline
def _word_less(a: UInt64, b: UInt64, dtype: Int64) -> Bool:
    if dtype == 11:
        return floating.less(a, b)
    if dtype == 1:
        # Compare encodings without native floating arithmetic: Float32
        # subnormals must not be flushed to zero by the device.
        var x = UInt32(a)
        var y = UInt32(b)
        var ax = x & UInt32(0x7FFFFFFF)
        var ay = y & UInt32(0x7FFFFFFF)
        if ax > UInt32(0x7F800000) or ay > UInt32(0x7F800000):
            return False
        if ax == 0 and ay == 0:
            return False
        var sx = x >> 31 != 0
        var sy = y >> 31 != 0
        return sx if sx != sy else (x > y if sx else x < y)
    if unsigned_type(dtype) or dtype == 4:
        return a < b
    return a.cast[DType.int64]() < b.cast[DType.int64]()


@always_inline
def _nan(word: UInt64, dtype: Int64) -> Bool:
    if dtype == 11:
        return floating.is_nan(word)
    if dtype == 1:
        return (UInt32(word) & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000)
    return False


@always_inline
def order_less(
    a: UInt32,
    b: UInt32,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
    literals: Pointer[UInt64, ImmutAnyOrigin] = Pointer[UInt64, ImmutAnyOrigin](
        unsafe_from_address=0
    ),
) -> Bool:
    for key in range(Int(keys)):
        var node = Int(start) + key
        var base = Int(code[unsafe_offset=4 * node + 1]) * Int(stride)
        var ai = base + Int(a)
        var bi = base + Int(b)
        var av = validity[unsafe_offset=ai] != 0
        var bv = validity[unsafe_offset=bi] != 0
        var nulls_last = (code[unsafe_offset=4 * node + 3] & 1) != 0
        if av != bv:
            return av if nulls_last else not av
        if not av:
            continue
        var aw = words[unsafe_offset=ai]
        var bw = words[unsafe_offset=bi]
        var dtype = types[unsafe_offset=node]
        if code[unsafe_offset=4 * node] != ROW_GROUP_KEY and (
            code[unsafe_offset=4 * node + 3] & 2
        ):
            var high_base = Int(stride) * Int(slots)
            var offset = Int(literals[unsafe_offset=node])
            aw = literals[
                unsafe_offset=offset
                + Int(words[unsafe_offset=high_base + ai] - 1)
            ]
            bw = literals[
                unsafe_offset=offset
                + Int(words[unsafe_offset=high_base + bi] - 1)
            ]
        if (
            code[unsafe_offset=4 * node] == ROW_GROUP_KEY
            and code[unsafe_offset=4 * node + 3] == 2
        ):
            var high_base = Int(stride) * Int(slots)
            var ah = words[unsafe_offset=high_base + ai]
            var bh = words[unsafe_offset=high_base + bi]
            if ah != bh:
                return ah < bh
            continue
        if dtype == 12:
            var high_base = Int(stride) * Int(slots)
            var ah = words[unsafe_offset=high_base + ai]
            var bh = words[unsafe_offset=high_base + bi]
            if ah != bh or aw != bw:
                var less = decimal.signed_less(
                    decimal.Wide(aw, ah), decimal.Wide(bw, bh)
                )
                return (
                    not less if code[unsafe_offset=4 * node + 2] != 0 else less
                )
            continue
        var an = _nan(aw, dtype)
        var bn = _nan(bw, dtype)
        var descending = code[unsafe_offset=4 * node + 2] != 0
        # NaN is the largest value: after the numbers ascending, before them
        # descending. NaN payload and sign do not participate in ordering;
        # later keys can still break ties.
        if an != bn:
            return an if descending else not an
        var less = _word_less(bw, aw, dtype) if descending else _word_less(
            aw, bw, dtype
        )
        if less:
            return True
        var greater = _word_less(aw, bw, dtype) if descending else _word_less(
            bw, aw, dtype
        )
        if greater:
            return False
    return a < b


def order_init(index: Pointer[UInt32, MutAnyOrigin], rows: UInt32):
    var row = global_idx.x
    if row < Int(rows):
        index[unsafe_offset=row] = UInt32(row)


def merge_grid(rows: Int, run_width: Int) -> Int:
    """Grid size for the row-ranked or bounded tiled merge kernel."""
    if rows <= 0:
        return 0
    if run_width < MERGE_MIN_RUN_WIDTH:
        return (rows + 255) // 256
    var pair_width = 2 * run_width
    var pairs = (rows + pair_width - 1) // pair_width
    var tiles_per_pair = (pair_width + MERGE_TILE_ROWS - 1) // MERGE_TILE_ROWS
    return pairs * tiles_per_pair


@always_inline
def _merge_corank(
    diagonal: Int,
    a_start: Int,
    a_end: Int,
    b_start: Int,
    b_end: Int,
    index: Pointer[UInt32, ImmutAnyOrigin],
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    stride: UInt32,
    slots: UInt32,
    literals: Pointer[UInt64, ImmutAnyOrigin],
) -> Tuple[Int, Int]:
    """Stable co-rank in two sorted index ranges at one output diagonal.

    The shared `order_less` comparator ends ties with distinct source row
    indices, so the merge order is strict even when sort keys compare equal.
    """
    var a_count = a_end - a_start
    var b_count = b_end - b_start
    var low = max(0, diagonal - b_count)
    var high = min(diagonal, a_count)
    while low <= high:
        var a = low + (high - low) // 2
        var b = diagonal - a
        var move_left = False
        if a > 0 and b < b_count:
            move_left = order_less(
                index[unsafe_offset=b_start + b],
                index[unsafe_offset=a_start + a - 1],
                words,
                validity,
                code,
                types,
                start,
                keys,
                stride,
                slots,
                literals,
            )
        if move_left:
            high = a - 1
        else:
            var move_right = False
            if b > 0 and a < a_count:
                move_right = order_less(
                    index[unsafe_offset=a_start + a],
                    index[unsafe_offset=b_start + b - 1],
                    words,
                    validity,
                    code,
                    types,
                    start,
                    keys,
                    stride,
                    slots,
                    literals,
                )
            if move_right:
                low = a + 1
            else:
                return (a, b)
    return (low, diagonal - low)


def order_merge(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    target: Pointer[UInt32, MutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    rows: UInt32,
    stride: UInt32,
    run_width: UInt32,
    slots: UInt32 = 0,
    literals: Pointer[UInt64, ImmutAnyOrigin] = Pointer[UInt64, ImmutAnyOrigin](
        unsafe_from_address=0
    ),
):
    var width = Int(run_width)
    if width < MERGE_MIN_RUN_WIDTH:
        var row = global_idx.x
        if row >= Int(rows):
            return
        # Int arithmetic avoids overflow in the final pass at Int32.MAX rows.
        var base = (row // (2 * width)) * (2 * width)
        var middle = min(base + width, Int(rows))
        var end = min(base + 2 * width, Int(rows))
        var left = row < middle
        var first = middle if left else base
        var lo = first
        var hi = end if left else middle
        var own = index[unsafe_offset=row]
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            if order_less(
                index[unsafe_offset=mid],
                own,
                words,
                validity,
                code,
                types,
                start,
                keys,
                stride,
                slots,
                literals,
            ):
                lo = mid + 1
            else:
                hi = mid
        var own_offset = row - (base if left else middle)
        target[unsafe_offset=base + own_offset + lo - first] = own
        return

    # At the first tiled pass a complete pair is one 2048-row output tile.
    # Later blocks cover contiguous tiles within each complete merge pair.
    var pair_width = 2 * width
    var tiles_per_pair = (pair_width + MERGE_TILE_ROWS - 1) // MERGE_TILE_ROWS
    var tile_work = Int(block_idx.x)
    var pair = tile_work // tiles_per_pair
    var tile = tile_work % tiles_per_pair
    var base = pair * pair_width
    var middle = min(base + width, Int(rows))
    var end = min(base + pair_width, Int(rows))
    var pair_rows = end - base
    var tile_start = tile * MERGE_TILE_ROWS
    if tile_start >= pair_rows:
        return
    var tile_end = min(tile_start + MERGE_TILE_ROWS, pair_rows)
    var left_start = base
    var left_end = middle
    var right_start = middle
    var right_end = end
    var bounds = unsafe_stack_allocation[
        4, DType.uint32, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        var tile_left = _merge_corank(
            tile_start,
            left_start,
            left_end,
            right_start,
            right_end,
            index,
            words,
            validity,
            code,
            types,
            start,
            keys,
            stride,
            slots,
            literals,
        )
        var tile_right = _merge_corank(
            tile_end,
            left_start,
            left_end,
            right_start,
            right_end,
            index,
            words,
            validity,
            code,
            types,
            start,
            keys,
            stride,
            slots,
            literals,
        )
        bounds[unsafe_offset=0] = UInt32(tile_left[0])
        bounds[unsafe_offset=1] = UInt32(tile_left[1])
        bounds[unsafe_offset=2] = UInt32(tile_right[0])
        bounds[unsafe_offset=3] = UInt32(tile_right[1])
    barrier()
    var a_start = left_start + Int(bounds[unsafe_offset=0])
    var b_start = right_start + Int(bounds[unsafe_offset=1])
    var a_end = left_start + Int(bounds[unsafe_offset=2])
    var b_end = right_start + Int(bounds[unsafe_offset=3])
    var lane = Int(thread_idx.x)
    var lane_diagonal = lane * (MERGE_TILE_ROWS // 256)
    if lane_diagonal >= tile_end - tile_start:
        return
    var local = _merge_corank(
        lane_diagonal,
        a_start,
        a_end,
        b_start,
        b_end,
        index,
        words,
        validity,
        code,
        types,
        start,
        keys,
        stride,
        slots,
        literals,
    )
    var a = a_start + local[0]
    var b = b_start + local[1]
    var remaining = min(
        MERGE_TILE_ROWS // 256, tile_end - tile_start - lane_diagonal
    )
    for offset in range(remaining):
        var from_a = False
        if a < a_end:
            if b >= b_end:
                from_a = True
            else:
                from_a = order_less(
                    index[unsafe_offset=a],
                    index[unsafe_offset=b],
                    words,
                    validity,
                    code,
                    types,
                    start,
                    keys,
                    stride,
                    slots,
                    literals,
                )
        target[
            unsafe_offset=base + tile_start + lane_diagonal + offset
        ] = index[unsafe_offset=a] if from_a else index[unsafe_offset=b]
        if from_a:
            a += 1
        else:
            b += 1


def order_key_errors(
    errors: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    failures: Pointer[UInt32, MutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    var row = global_idx.x
    var failed = UInt32(0)
    if row < Int(rows):
        for key in range(Int(keys)):
            var node = Int(start) + key
            var base = Int(code[unsafe_offset=4 * node + 1]) * Int(stride)
            failed |= errors[unsafe_offset=base + row]
    var count = prefix_sum[block_size=256](UInt32(failed != 0))
    if thread_idx.x == 255:
        failures[unsafe_offset=block_idx.x] = count


def order_gather(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    gathers: Pointer[Int64, ImmutAnyOrigin],
    count: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var source = Int(index[unsafe_offset=row])
    for column in range(Int(count)):
        var base = Int(gathers[unsafe_offset=column]) * Int(stride)
        output_words[unsafe_offset=base + row] = words[
            unsafe_offset=base + source
        ]
        if slots:
            var high_base = Int(stride) * Int(slots)
            output_words[unsafe_offset=high_base + base + row] = words[
                unsafe_offset=high_base + base + source
            ]
        output_validity[unsafe_offset=base + row] = validity[
            unsafe_offset=base + source
        ]
        output_errors[unsafe_offset=base + row] = errors[
            unsafe_offset=base + source
        ]
