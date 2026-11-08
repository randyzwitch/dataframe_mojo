"""Independent input jobs executed through the query runtime port."""
from dataframe.frame import (
    DataFrame,
)
from dataframe.joins.hash_table import (
    JoinHashTable,
    keys_only_join,
)
from dataframe.dtype import DataType
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_INNER,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
)
from dataframe.scheduler import (
    SchedulablePipeline,
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
    mut built: DataFrame,
    keys: List[String],
    how: Int,
    coalesce: Bool = True,
    probe_estimate: Int = -1,
) raises -> Optional[JoinHashTable]:
    """A join's build side, finished as its build pipeline's last stage:
    every column in one chunk, then the hash table over `keys` with the
    build's other columns as its rows' payload (none when `keys` is
    empty, or when a key cannot be hashed this way), as DuckDB finalizes
    a hash join's table at the end of its build pipeline. Semi and anti
    joins read no build column, so their rows have no payload.
    `probe_estimate` is the probe side's estimated rows."""
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
    var emits = how != JOIN_SEMI and how != JOIN_ANTI
    var payload = List[Series]()
    var output_columns = List[Int]()
    var output_names = List[String]()
    var output_types = List[DataType]()
    for i in range(built.width()):
        ref column = built._columns[i]
        var key_at = -1
        for k in range(len(keys)):
            if keys[k] == column.name():
                key_at = k
        if key_at < 0:
            if emits:
                payload.append(column.copy())
                output_columns.append(len(keys) + len(payload) - 1)
                output_names.append(column.name())
                output_types.append(column.dtype())
            continue
        # A key column: read from its own field where the join outputs it.
        if not emits:
            continue
        if coalesce and how != JOIN_RIGHT_SEMI and how != JOIN_RIGHT_ANTI:
            continue
        output_columns.append(key_at)
        output_names.append(column.name())
        output_types.append(column.dtype())
    return JoinHashTable(
        sources,
        payload=Optional(payload^),
        output_columns=output_columns,
        output_names=output_names,
        output_types=output_types,
        perfect=how == JOIN_INNER,
        keys_only=keys_only_join(how),
        probe_estimate=probe_estimate,
    )


struct _SubplanPipeline(SchedulablePipeline):
    """One subplan (a union's or a join's input, a stream's build side or
    breaker input), executed from its own copy of the plan as one
    scheduled pipeline (`run_pipelines`). A join's build side given its
    keys finishes as a build (`finish_build`) in its finish event, as
    DuckDB's hash join finalizes its table in its build pipeline's
    `PipelineFinishEvent`."""

    var plan: QueryContext
    var index: Int
    var streaming: Bool
    var batch_size: Int
    var result: DataFrame
    # Build keys and the join's type; no keys for an input that is not a
    # hash join's build side.
    var keys: List[String]
    var how: Int
    var coalesce: Bool
    var probe_estimate: Int
    var built: Optional[JoinHashTable]

    def __init__(
        out self,
        var plan: QueryContext,
        index: Int,
        streaming: Bool,
        batch_size: Int,
        keys: List[String] = List[String](),
        how: Int = JOIN_INNER,
        coalesce: Bool = True,
        probe_estimate: Int = -1,
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
        self.how = how
        self.coalesce = coalesce
        self.probe_estimate = probe_estimate
        self.built = None

    def initialize(mut self) raises:
        pass

    def execute_tasks(self) -> Int:
        return 1

    def execute(mut self, task: Int) raises:
        self.result = self.plan._execute(
            self.index, False, self.streaming, self.batch_size
        )

    def prepare_finalize(mut self) raises:
        pass

    def finalize(mut self) raises:
        if len(self.keys) > 0:
            self.built = finish_build(
                self.result,
                self.keys,
                self.how,
                self.coalesce,
                self.probe_estimate,
            )
