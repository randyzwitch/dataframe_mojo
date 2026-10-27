"""Independent input jobs executed through the query runtime port."""
from dataframe.frame import (
    DataFrame,
)
from dataframe.parallel import (
    Job,
)
from dataframe.planning.context import (
    QueryContext,
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


struct _SubplanJob(Job):
    """One subplan (a union's or a join's input, a stream's breaker input),
    executed from its own copy of the plan."""

    var plan: QueryContext
    var index: Int
    var streaming: Bool
    var batch_size: Int
    var result: DataFrame

    def __init__(
        out self,
        var plan: QueryContext,
        index: Int,
        streaming: Bool,
        batch_size: Int,
    ) raises:
        self.plan = plan^
        # Its own memo: nested prefetches on other threads never share one.
        self.plan._prefetched = Optional(ArcPointer(Dict[Int, DataFrame]()))
        self.index = index
        self.streaming = streaming
        self.batch_size = batch_size
        self.result = DataFrame(List[Series]())

    def run(mut self) raises:
        self.result = self.plan._execute(
            self.index, False, self.streaming, self.batch_size
        )
