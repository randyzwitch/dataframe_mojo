"""A join build's finish: its columns and key index."""
from dataframe.frame import (
    DataFrame,
)
from dataframe.joins.hash_table import (
    JoinBuildLayout,
    JoinBuildSink,
    JoinHashTable,
    _FrameSinkJob,
    join_build_layout,
)
from dataframe.parallel import (
    Rounds,
    partitions,
    round_width,
    run_rounds,
    worker_count,
)
from dataframe.series import (
    Series,
)
from dataframe.vector.types import (
    STANDARD_VECTOR_SIZE,
)
from std.collections import (
    Optional,
)
from std.memory import (
    ArcPointer,
)


def finish_build(
    mut built: DataFrame,
    keys: List[String],
    how: Int,
    coalesce: Bool = True,
    probe_estimate: Int = -1,
    build_filtered: Bool = False,
    defer_finalize: Bool = False,
) raises -> Optional[JoinHashTable]:
    """Finish a join build by retaining its columns and constructing its key
    index (`BuildFinish`, its rounds run here). Return no index for absent
    or unsupported keys. Semi/anti joins need no build payload because
    they emit only the preserved side. probe_estimate describes the
    expected probe cardinality, and build_filtered whether the build side
    filters its rows. `defer_finalize`: the caller's finish event runs the
    table's finalize tasks (`JoinHashTable.finalize_tasks`)."""
    var finish = BuildFinish(
        built.copy(), keys, how, coalesce, probe_estimate, build_filtered
    )
    finish.defer_finalize = defer_finalize
    run_rounds(finish)
    swap(built, finish.frame)
    var table = Optional[JoinHashTable]()
    swap(table, finish.table)
    return table^


# The stages of a `BuildFinish`.
comptime _BUILD_BEGIN = 0
comptime _BUILD_RECHUNK = 1
comptime _BUILD_SINK = 2
comptime _BUILD_DONE = 3


struct BuildFinish(Movable, Rounds):
    """A join build's finish over its whole frame as rounds of tasks
    (`next_round`, `round_task`): its chunked columns made one chunk each,
    a task per column; then its rows sunk into the table's partitioned
    rows, each task a contiguous run of whole vectors into a sink of its
    own, whose partitions combine in task order so each keeps its rows in
    build order. Once `next_round` returns 0, `frame` holds the columns
    and `table` the key index (none for absent or unsupported keys)."""

    var frame: DataFrame
    var keys: List[String]
    var how: Int
    var coalesce: Bool
    var probe_estimate: Int
    var build_filtered: Bool
    var defer_finalize: Bool
    var stage: Int
    # The chunked columns and their single-chunk copies; task t takes
    # columns t, t + width, ...
    var chunked: List[Int]
    var whole: List[Series]
    var width: Int
    var layout: Optional[ArcPointer[JoinBuildLayout]]
    var jobs: List[_FrameSinkJob]
    var table: Optional[JoinHashTable]

    def __init__(
        out self,
        var frame: DataFrame,
        keys: List[String],
        how: Int,
        coalesce: Bool,
        probe_estimate: Int,
        build_filtered: Bool,
    ):
        self.frame = frame^
        self.keys = keys.copy()
        self.how = how
        self.coalesce = coalesce
        self.probe_estimate = probe_estimate
        self.build_filtered = build_filtered
        self.defer_finalize = True
        self.stage = _BUILD_BEGIN
        self.chunked = List[Int]()
        self.whole = List[Series]()
        self.width = 0
        self.layout = None
        self.jobs = List[_FrameSinkJob]()
        self.table = None

    def next_round(mut self) raises -> Int:
        if self.stage == _BUILD_BEGIN:
            for c in range(self.frame.width()):
                if self.frame._columns[c].is_chunked():
                    self.chunked.append(c)
                    self.whole.append(self.frame._columns[c].copy())
            if len(self.chunked) > 0:
                self.stage = _BUILD_RECHUNK
                self.width = round_width(len(self.chunked))
                return self.width
            return self._begin_sink()
        if self.stage == _BUILD_RECHUNK:
            for k in range(len(self.chunked)):
                self.frame._columns[self.chunked[k]] = self.whole[k].copy()
            self.whole = List[Series]()
            return self._begin_sink()
        if self.stage == _BUILD_SINK:
            var sinks = List[JoinBuildSink](capacity=len(self.jobs))
            for j in range(len(self.jobs)):
                sinks.append(self.jobs[j].sink.pop())
            self.jobs = List[_FrameSinkJob]()
            ref layout = self.layout.value()
            var keys = List[Series](capacity=len(layout[].key_columns))
            for c in layout[].key_columns:
                keys.append(self.frame._columns[c].copy())
            self.table = JoinHashTable(
                layout, sinks^, self.defer_finalize, keys^
            )
            self.stage = _BUILD_DONE
        return 0

    def _begin_sink(mut self) raises -> Int:
        """The round sinking the rows: a task per run of whole vectors, one
        a thread; none for absent or unsupported keys."""
        self.stage = _BUILD_DONE
        var count = self.frame.height()
        if len(self.keys) == 0 or count > Int(Int32.MAX):
            return 0
        var layout = join_build_layout(
            self.frame._columns,
            self.keys,
            self.how,
            self.coalesce,
            self.probe_estimate,
            False,
            build_filtered=self.build_filtered,
        )
        if not layout:
            return 0
        var shared = ArcPointer(layout.take())
        var workers = worker_count(count)
        var bounds = partitions(count, workers, STANDARD_VECTOR_SIZE)
        for w in range(workers):
            if bounds[w] < bounds[w + 1]:
                self.jobs.append(
                    _FrameSinkJob(
                        self.frame._columns,
                        shared,
                        bounds[w],
                        bounds[w + 1] - bounds[w],
                    )
                )
        self.layout = shared^
        self.stage = _BUILD_SINK
        if len(self.jobs) == 0:
            # No rows: the table over no sink.
            return self.next_round()
        return len(self.jobs)

    def round_task(mut self, task: Int) raises:
        if self.stage == _BUILD_RECHUNK:
            var k = task
            while k < len(self.chunked):
                self.whole[k] = self.whole[k].rechunk()
                k += self.width
        else:
            self.jobs[task].run()
