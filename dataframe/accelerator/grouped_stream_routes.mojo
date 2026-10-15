"""Original CPU grouped-stream certificates captured before device projection.

Only metadata is inspected here. The device separately proves the CPU key
sample selects inserted tables, using original canonical row group IDs.
"""
from dataframe import DataFrame, DataType
from dataframe.binding import bind
from dataframe.expr import COL, SUM, MEAN, MIN, MAX
from std.collections import Optional
from dataframe.lazy import LazyFrame, SCAN_FRAME, AGG
from dataframe.pipeline import _flush_groups
from dataframe.gather import can_filter_aligned_chunks
from dataframe.accelerator.memory import checked_add, checked_mul

comptime BASIC = 1


@fieldwise_init
struct StreamRoute(Copyable):
    var query: LazyFrame
    var frame: DataFrame
    var family: Int
    var configured_workers: Int
    var jobs: Int
    var ranges: List[UInt32]
    var flush_groups: Int
    var rows: Int
    var capture_peak: Int


def terminal_family(query: LazyFrame) raises -> Int:
    var original = query.copy()
    if not len(original._nodes):
        return 0
    original._reorder(len(original._nodes) - 1)
    if (
        len(original._nodes) != 2
        or original._nodes[0].kind != SCAN_FRAME
        or original._nodes[1].kind != AGG
        or original._nodes[1].left != 0
        or not original._nodes[1].maintain_order
        or len(original._nodes[1].names) != 1
        or not len(original._nodes[1].exprs)
    ):
        return 0
    ref scan = original._nodes[0]
    ref frame = original._frames[scan.offset]
    if frame.column(original._nodes[1].names[0]).dtype() != DataType.INT32:
        return 0
    var family = 0
    for expression in original._nodes[1].exprs:
        var bound = bind(expression, frame._columns)
        var size = len(bound.expr._nodes)
        ref node = bound.expr._nodes[size - 1]
        if size != 2 or node.left != 0 or bound.expr._nodes[0].op != COL:
            return 0
        var dtype = bound.dtypes[node.left]
        if dtype != DataType.FLOAT32 and dtype != DataType.FLOAT64:
            return 0
        if node.op in [SUM, MEAN, MIN, MAX]:
            family |= BASIC
        else:
            return 0
    return family


@fieldwise_init
struct CapturedRanges(Movable):
    var values: List[UInt32]
    var peak: Int

    def take_values(mut self) -> List[UInt32]:
        # Reinitialize the field so the capture remains wholly destructible.
        # Moving the List preserves its allocation and exact charged capacity.
        var values = self.values^
        self.values = List[UInt32]()
        return values^


def _capture_ranges(
    frame: DataFrame, morsel_rows: Int, workers: Int, budget: Int
) raises -> CapturedRanges:
    """CPU _morsel_ranges geometry with bounded allocation capacities.

    The piece arithmetic and 64-row rounding are copied from pipeline;
    count output entries first so capture never grows an unchecked List.
    """
    # These bounds keep every copied CPU piece/count expression in Int.
    # The public route declines settings whose CPU overflow is unmodeled.
    if (
        morsel_rows < 1
        or morsel_rows > Int.MAX // 2
        or workers < 1
        or workers > (1 << 30)
    ):
        raise Error(
            "Metal unsupported [grouped float stream]: morsel settings exceed certified arithmetic bounds"
        )
    if not frame.height():
        return CapturedRanges(List[UInt32](), 0)
    if frame.height() < 0 or frame.height() >= (1 << 30):
        raise Error("Grouped float source extent is unsupported")
    var aligned = can_filter_aligned_chunks(frame._columns)
    var chunks = len(
        frame._columns[0]._chunked.value()[].ends
    ) if aligned else 1
    var scratch = checked_mul(24, chunks)
    if budget >= 0 and scratch > budget:
        raise Error(
            "Mojo GPU resident query exceeds memory budget before submission"
        )
    var starts = List[Int](length=chunks, fill=0)
    var lengths = List[Int](length=chunks, fill=0)
    var pieces = List[Int](length=chunks, fill=0)
    var count = 0
    var total = 0
    var start = 0
    for at in range(chunks):
        var end = (
            frame._columns[0]
            ._chunked.value()[]
            .ends[at] if aligned else frame.height()
        )
        var length = end - start
        if length > 0:
            starts[count] = start
            lengths[count] = length
            pieces[count] = max(
                1, (2 * length + morsel_rows) // (2 * morsel_rows)
            )
            total += pieces[count]
            count += 1
        start = end
    var lanes = max(1, workers)
    if total == 1 and lanes > 1:
        var share = max(2048, morsel_rows // lanes)
        var most = min(lanes, lengths[0] // share)
        if most > 1:
            pieces[0] = most
            total = most
    if total > 1 and total < 4 * lanes and total % lanes != 0:
        var wanted = total + lanes - total % lanes
        var scaled = 0
        for c in range(count):
            pieces[c] = min(lengths[c], pieces[c] * wanted // total)
            scaled += pieces[c]
        while scaled < wanted:
            var longest = -1
            for c in range(count):
                if pieces[c] < lengths[c] and (
                    longest < 0
                    or lengths[c] // pieces[c]
                    > lengths[longest] // pieces[longest]
                ):
                    longest = c
            if longest < 0:
                break
            pieces[longest] += 1
            scaled += 1
    var entries = 0
    for c in range(count):
        for part in range(pieces[c]):
            var low = lengths[c] * part // pieces[c]
            var high = lengths[c] * (part + 1) // pieces[c]
            if part > 0:
                low = low // 64 * 64
            if part + 1 < pieces[c]:
                high = high // 64 * 64
            if high > low:
                entries += 2
    var peak = checked_add(scratch, checked_mul(4, entries))
    if budget >= 0 and peak > budget:
        raise Error(
            "Mojo GPU resident query exceeds memory budget before submission"
        )
    var ranges = List[UInt32](length=entries, fill=0)
    var entry = 0
    for c in range(count):
        for part in range(pieces[c]):
            var low = lengths[c] * part // pieces[c]
            var high = lengths[c] * (part + 1) // pieces[c]
            if part > 0:
                low = low // 64 * 64
            if part + 1 < pieces[c]:
                high = high // 64 * 64
            if high > low:
                ranges[entry] = UInt32(starts[c] + low)
                ranges[entry + 1] = UInt32(high - low)
                entry += 2
    return CapturedRanges(ranges^, peak)


def capture(
    query: LazyFrame,
    configured_workers: Int,
    batch_size: Int,
    *,
    optimize: Bool,
    streaming: Bool,
    original_source: Bool,
    budget: Int = -1,
    require_single_job: Bool = True,
) raises -> StreamRoute:
    if not original_source or not streaming:
        raise Error(
            "Metal unsupported [grouped float stream]: original CPU streaming route required"
        )
    if configured_workers < 1 or batch_size < 1:
        raise Error("Grouped float stream execution settings are invalid")
    # The public CPU collector performs this optimization before its source
    # projection and pipeline selection. Direct SCAN→AGG has no joins or
    # shared subplans requiring execution during later optimizer passes.
    var cpu_query = query._optimized() if optimize else query.copy()
    cpu_query._reorder(len(cpu_query._nodes) - 1)
    cpu_query._compact_scan_slots()
    var family = terminal_family(cpu_query)
    if not family:
        raise Error(
            "Metal unsupported [grouped float stream]: original terminal shape is not certified"
        )
    ref scan = cpu_query._nodes[0]
    if scan.source_failure:
        raise Error(scan.source_failure)
    if len(scan.names2):
        raise Error(
            "Metal unsupported [grouped float stream]: rewritten source provenance is not certified"
        )
    var frame = cpu_query._frames[scan.offset].copy()
    if len(scan.names):
        frame = frame.select(scan.names)
    # Keep describe and execute on the same metadata certificate. Only raw
    # float columns are covered by the pinned update/merge witnesses.
    for expression in cpu_query._nodes[1].exprs:
        var bound = bind(expression, frame._columns)
        var size = len(bound.expr._nodes)
        ref node = bound.expr._nodes[size - 1]
        if size != 2 or node.left != 0 or bound.expr._nodes[0].op != COL:
            raise Error(
                "Metal unsupported [grouped float stream]: direct floating column operands required"
            )
    var captured = _capture_ranges(
        frame, batch_size, configured_workers, budget
    )
    var capture_peak = captured.peak
    var ranges = captured.take_values()
    # run_pipeline launches one shared-cursor job per morsel at most. A
    # configured crew greater than one is still exact when only one job
    # exists; multiple jobs have no fixed morsel-to-worker assignment.
    var jobs = max(1, min(configured_workers, len(ranges) // 2))
    if require_single_job and jobs != 1:
        raise Error(
            "Metal unsupported [grouped float stream]: multiple CPU pipeline jobs have no certified morsel ownership"
        )
    var rows = frame.height()
    return StreamRoute(
        cpu_query^,
        frame^,
        family,
        configured_workers,
        jobs,
        ranges^,
        _flush_groups(),
        rows,
        capture_peak,
    )


def candidate(
    query: LazyFrame,
    configured_workers: Int,
    batch_size: Int,
    *,
    optimize: Bool,
    streaming: Bool,
    original_source: Bool,
    budget: Int = -1,
) raises -> Optional[StreamRoute]:
    """Select only this prototype's original streaming single-job route.

    Eager, rewritten, computed and actual multi-job queries retain the main
    executor. Key sample admission remains a separate on-device certificate.
    """
    if not streaming or not original_source or not terminal_family(query):
        return None
    var route = capture(
        query,
        configured_workers,
        batch_size,
        optimize=optimize,
        streaming=True,
        original_source=True,
        budget=budget,
        require_single_job=False,
    )
    if route.jobs != 1:
        return None
    return route^
