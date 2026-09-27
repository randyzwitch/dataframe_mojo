"""Merge aggregate states across ordered batches without finalizing partial sums."""
from std.memory import ArcPointer
from .aggregate import Reducer
from .binding import bind, AGGREGATE
from .dtype import DataType
from .execution import _ReduceJob
from .expr import Expr, col, is_reduction, subtree
from .frame import DataFrame, concat, _expand_struct_keys, _pack_struct_keys
from .hashing import encode_rows
from .series import Series


struct _StreamReduction(Movable):
    var keys: DataFrame
    var states: List[Reducer]
    var names: List[String]
    var dtypes: List[DataType]
    var outputs: List[Expr]
    var grouped: Bool

    def __init__(
        out self, frame: DataFrame, expressions: List[Expr], names: List[String]
    ) raises:
        self.grouped = len(names) > 0
        var ids = List[Int]()
        var count = 1
        if self.grouped:
            var keys = _expand_struct_keys(frame._subset_keys(names))
            var groups = encode_rows(keys, nulls_equal=True)
            count = groups.count()
            ids = groups.ids.copy()
            var columns = List[Series]()
            for key in keys:
                columns.append(key.take(groups.representatives.copy()))
            self.keys = DataFrame(columns^, height=count)
        else:
            self.keys = DataFrame(List[Series](), height=1)
        self.outputs = List[Expr]()
        self.states = List[Reducer]()
        self.names = List[String]()
        self.dtypes = List[DataType]()
        var shared = ArcPointer(ids^)
        for expression in expressions:
            var bound = bind(expression, frame._columns)
            if bound.shape() != AGGREGATE:
                raise Error(
                    "Streaming aggregate requires scalar aggregate expressions"
                )
            if expression._name in names:
                raise Error(
                    "Aggregate output name collides with grouping key: "
                    + expression._name
                )
            var rewritten = expression.copy()
            for i in range(len(expression._nodes)):
                if not is_reduction(expression._nodes[i].op):
                    continue
                var job = _ReduceJob[8](
                    bound,
                    frame._columns,
                    List[Series](),
                    expression._nodes[i],
                    0,
                    frame.height(),
                    1024,
                    self.grouped,
                    shared,
                    count,
                )
                job.run()
                var name = "__stream_reduction_" + String(len(self.states))
                self.states.append(job^.into_reducer())
                self.names.append(name)
                self.dtypes.append(bound.dtypes[i])
                rewritten._nodes[i] = col(name)._nodes[0].copy()
            self.outputs.append(subtree(rewritten, len(rewritten._nodes) - 1))

    def merge(mut self, other: Self) raises:
        var mapping = List[Int]()
        var count = 1
        if self.grouped:
            var old = self.keys.height()
            var combined = concat([self.keys.copy(), other.keys.copy()])
            var groups = encode_rows(combined._columns, nulls_equal=True)
            count = groups.count()
            for i in range(other.keys.height()):
                mapping.append(groups.ids[old + i])
            self.keys = combined.take(groups.representatives^)
        for i in range(len(self.states)):
            self.states[i].grow(count)
            self.states[i].merge(other.states[i], mapping)

    def finish(self) raises -> DataFrame:
        var reduced = List[Series]()
        for i in range(len(self.states)):
            reduced.append(
                self.states[i]
                .finish()
                .with_dtype(self.dtypes[i])
                .renamed(self.names[i])
            )
        var values = DataFrame(
            reduced^, height=self.keys.height()
        ).select_exprs(self.outputs)
        var columns = self.keys._columns.copy()
        for column in values._columns:
            columns.append(column.copy())
        return _pack_struct_keys(DataFrame(columns^, height=self.keys.height()))
