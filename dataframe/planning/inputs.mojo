"""Independent input jobs executed through the query runtime port."""
from dataframe.frame import (
    DataFrame,
)
from dataframe.joins.hash_table import (
    JoinHashTable,
    build_from_columns,
    join_build_layout,
)
from dataframe.join_type import JOIN_INNER
from dataframe.parallel import configured_workers
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
    defer_finalize: Bool = False,
) raises -> Optional[JoinHashTable]:
    """Finish a join build by retaining its columns and constructing its key
    index. Return no index for absent or unsupported keys. Semi/anti joins
    need no build payload because they emit only the preserved side.
    probe_estimate describes the expected probe cardinality.
    `defer_finalize`: the caller's finish event runs the table's finalize
    tasks (`JoinHashTable.finalize_tasks`)."""
    for column in built._columns:
        if column.is_chunked():
            built = built.rechunk()
            break
    if len(keys) == 0 or built.height() > Int(Int32.MAX):
        return None
    var layout = join_build_layout(
        built._columns, keys, how, coalesce, probe_estimate, False
    )
    if not layout:
        return None
    return build_from_columns(
        ArcPointer(layout.take()), built._columns, defer_finalize
    )


struct _SubplanPipeline(SchedulablePipeline):
    """A scheduled input plan whose finish event prepares its columns and
    optional join-key index. The dependent probe starts only after this
    event completes."""

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

    def max_execute_tasks(self) -> Int:
        return 1

    def execute_tasks(mut self) raises -> Int:
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
                defer_finalize=True,
            )

    def max_finalize_tasks(self) -> Int:
        return configured_workers() if len(self.keys) > 0 else 0

    def finalize_tasks(mut self) raises -> Int:
        if not self.built:
            return 0
        return self.built.value().finalize_tasks()

    def finalize_task(mut self, task: Int) raises:
        self.built.value().finalize_task(task)

    def finish_finalize(mut self) raises:
        if self.built:
            self.built.value().finish_finalize()
