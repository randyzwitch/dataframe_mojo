"""Probe and gather one block of probe rows at a time.

A join used to probe every left row into per-worker row lists, copy those
into two row lists of the whole output, and only then gather every column
from them: three passes over index lists as long as the output, which fall
out of cache. Polars' streaming join (`partition_and_probe` in
polars-stream/src/nodes/joins/equi_join.rs) probes one morsel, gathers that
morsel's build and probe payload straight from its match lists
(`DataFrameBuilder::gather_extend`), and emits it, so its index lists stay
as long as one morsel and no list of the whole output exists.

Here a block of probe rows is a job: it probes its rows with the route's
probe job, gathers every output column from its own match lists, and keeps
the gathered pieces. The pieces of each column, in block order, become the
column's chunks (a column with a single block stays one array).
"""
from std.memory import ArcPointer

from .gather import _RechunkJob, take_parallel
from .parallel import Job, run_jobs, worker_count
from .series import Series

# Probe rows per block. Polars' morsel is about 100K rows, but a block's output
# columns are fresh buffers whose first touch faults every page, and only a
# buffer that holds whole 2 MB pages (an 8-byte column of 1M rows is 8 MB) can
# be backed by huge pages (`huge_pages.mojo`); with 128K-row blocks the
# faults came back and the join was slower than with whole-output lists.
comptime PROBE_BLOCK_ROWS = 1 << 20


def probe_blocks(rows: Int) -> Int:
    """Blocks to cut `rows` probe rows into: none longer than the block
    size, and a whole number per worker so no worker runs a round alone."""
    var workers = worker_count(rows)
    var blocks = (rows + PROBE_BLOCK_ROWS - 1) // PROBE_BLOCK_ROWS
    if blocks <= workers:
        return workers
    return (blocks + workers - 1) // workers * workers


trait MatchJob(Job):
    """A probe job that leaves its matches as row lists to take."""

    def take_matches(mut self, mut left: List[Int], mut right: List[Int]) -> Bool:
        """Move the matched rows out. True means every probe row in
        `row_range` matched once, in order, and `left` is left empty."""
        ...

    def row_range(self) -> Tuple[Int, Int]:
        ...


struct FusedJoin(Copyable, Movable):
    """What a join asks of the fused route, and what it returns."""

    var enabled: Bool
    var left_sources: List[Series]
    var right_sources: List[Series]
    var done: Bool
    # Every probe row matched once, in order: `left` is then empty and the
    # join's left columns are its own.
    var identity: Bool
    var left: List[Series]
    var right: List[Series]
    var height: Int

    def __init__(out self):
        self.enabled = False
        self.left_sources = List[Series]()
        self.right_sources = List[Series]()
        self.done = False
        self.identity = False
        self.left = List[Series]()
        self.right = List[Series]()
        self.height = 0


struct _FusedJob[P: MatchJob](Job):
    var probe: Self.P
    var left: ArcPointer[List[Series]]
    var right: ArcPointer[List[Series]]
    var include_unmatched: Bool
    var identity: Bool
    var left_out: List[Series]
    var right_out: List[Series]
    var height: Int

    def __init__(
        out self,
        var probe: Self.P,
        left: ArcPointer[List[Series]],
        right: ArcPointer[List[Series]],
        include_unmatched: Bool,
    ):
        self.probe = probe^
        self.left = left.copy()
        self.right = right.copy()
        self.include_unmatched = include_unmatched
        self.identity = False
        self.left_out = List[Series]()
        self.right_out = List[Series]()
        self.height = 0

    def run(mut self) raises:
        self.probe.run()
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        self.identity = self.probe.take_matches(left_rows, right_rows)
        self.height = len(right_rows)
        var span = self.probe.row_range()
        if not self.identity and len(left_rows) == span[1] - span[0]:
            # Every probe row once, in order: its rows are the block's own.
            var rows = left_rows.unsafe_ptr()
            var first = span[0]
            var same = True
            for k in range(len(left_rows)):
                if rows[unsafe_offset=k] != first + k:
                    same = False
                    break
            self.identity = same
        if self.identity:
            for column in self.left[]:
                self.left_out.append(column.slice(span[0], span[1] - span[0]))
        else:
            self.left_out = take_parallel(
                self.left[], left_rows^, 1, or_null=False
            )
        self.right_out = take_parallel(
            self.right[], right_rows^, 1, or_null=self.include_unmatched
        )


def _contiguous(var columns: List[Series]) raises -> List[Series]:
    """Each column as one array, rechunked once on all workers, so the
    blocks gather from them without each rechunking its own copy."""
    var positions = List[Int]()
    var jobs = List[_RechunkJob]()
    for c in range(len(columns)):
        if columns[c].is_chunked():
            positions.append(c)
            jobs.append(_RechunkJob(columns[c]))
    if len(jobs) == 1:
        jobs[0].run()
    elif len(jobs) > 1:
        run_jobs(jobs)
    while len(jobs) > 0:
        columns[positions.pop(0)] = jobs.pop(0).into_result()
    return columns^


def _assemble(columns: Int, pieces: List[List[Series]]) raises -> List[Series]:
    var out = List[Series](capacity=columns)
    for c in range(columns):
        var parts = List[Series](capacity=len(pieces))
        for block in pieces:
            parts.append(block[c].copy())
        out.append(Series._from_chunks(parts))
    return out^


def fused_gather[
    P: MatchJob
](
    mut probes: List[P],
    mut request: FusedJoin,
    include_unmatched: Bool,
) raises:
    """Run every probe job and gather its block's columns from
    `request.left_sources` and `request.right_sources`; the joined columns
    are left in `request`."""
    var left = ArcPointer(_contiguous(request.left_sources.copy()))
    var right = ArcPointer(_contiguous(request.right_sources.copy()))
    var jobs = List[_FusedJob[P]](capacity=len(probes))
    while len(probes) > 0:
        jobs.append(
            _FusedJob[P](probes.pop(0), left, right, include_unmatched)
        )
    run_jobs(jobs)
    var identity = True
    var height = 0
    var left_pieces = List[List[Series]](capacity=len(jobs))
    var right_pieces = List[List[Series]](capacity=len(jobs))
    for j in range(len(jobs)):
        ref job = jobs[j]
        identity = identity and job.identity
        height += job.height
        left_pieces.append(job.left_out.copy())
        right_pieces.append(job.right_out.copy())
    request.identity = identity
    request.height = height
    # A block that matched every row once left its columns as slices of the
    # source; they are chunks like any other gathered piece.
    if not identity:
        request.left = _assemble(len(left[]), left_pieces)
    request.right = _assemble(len(right[]), right_pieces)
    request.done = True
