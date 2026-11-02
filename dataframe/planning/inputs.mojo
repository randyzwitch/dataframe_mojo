"""Independent input jobs executed through the query runtime port."""
from dataframe.frame import (
    DataFrame,
)
from dataframe.kernels.join_hash import (
    PreparedHashIndex,
    prepare_hash_index,
)
from dataframe.parallel import (
    Job,
)
from dataframe.planning.context import (
    QueryContext,
    _ReadyIndex,
)
from dataframe.series import (
    Series,
)
from std.collections import (
    Dict,
    Optional,
)
from std.memory import (
    ArcPointer,
)


def finish_build(
    mut built: DataFrame, keys: List[String], probe_rows: Int
) raises -> Optional[PreparedHashIndex]:
    """A join's build side, finished as its build pipeline's last stage:
    every column in one chunk, then the hash index over `keys` (none
    when `keys` is empty, or when a key cannot be hashed this way), as
    DuckDB finalizes a hash join's table at the end of its build
    pipeline. A probe gathers the build's payload through
    `take_parallel`, which rechunks a chunked column every time it is
    called, once per morsel or batch (PDS-H q2's 159K-row build with five
    string columns: 3.8 ms a probe). `probe_rows` bounds the rows that
    will probe it (see `prepare_hash_index`)."""
    for column in built._columns:
        if column.is_chunked():
            built = built.rechunk()
            break
    if len(keys) == 0:
        return None
    var sources = List[Series]()
    var supported = built.height() <= Int(Int32.MAX)
    for name in keys:
        var key = built[name]
        supported = supported and not key.dtype().is_nested()
        sources.append(key.copy())
    if not supported:
        return None
    return prepare_hash_index(sources, probe_rows=probe_rows)


struct _SubplanJob(Job):
    """One subplan (a union's or a join's input, a stream's breaker input),
    executed from its own copy of the plan. A join's build side given its
    keys also finishes as a build (`finish_build`), on this job's thread,
    so the hash index of one build is made while sibling inputs still
    run, as DuckDB's build pipelines each finalize on their own
    dependencies (`BuildPipelineSchedule`)."""

    var plan: QueryContext
    var index: Int
    var streaming: Bool
    var batch_size: Int
    var result: DataFrame
    # Build keys and the rows bound to probe them; no keys for an input
    # that is not a hash join's build side.
    var keys: List[String]
    var probe_rows: Int
    var built: Optional[PreparedHashIndex]

    def __init__(
        out self,
        var plan: QueryContext,
        index: Int,
        streaming: Bool,
        batch_size: Int,
        keys: List[String] = List[String](),
        probe_rows: Int = -1,
    ) raises:
        self.plan = plan^
        # Its own memo: nested prefetches on other threads never share one.
        self.plan._prefetched = Optional(ArcPointer(Dict[Int, DataFrame]()))
        self.plan._prefetched_indexes = Optional(
            ArcPointer(Dict[Int, _ReadyIndex]())
        )
        self.index = index
        self.streaming = streaming
        self.batch_size = batch_size
        self.result = DataFrame(List[Series]())
        self.keys = keys.copy()
        self.probe_rows = probe_rows
        self.built = None

    def run(mut self) raises:
        self.result = self.plan._execute(
            self.index, False, self.streaming, self.batch_size
        )
        if len(self.keys) > 0:
            self.built = finish_build(self.result, self.keys, self.probe_rows)
