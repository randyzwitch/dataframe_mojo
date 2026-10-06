"""Observed execution counters for a lazy plan (#439).

`LazyFrame.profile()` attaches a report to a copy of the plan and both
executors record into it: one record per plan node that ran, with the
rows it read and wrote, how many times it ran, and for a join the rows it
built an index over, how many indexes it built, the build side and the
algorithm. The counts are observed, never estimated, so tests can assert
one build per streaming join and one execution per input without timing.
Without a report attached, the executors only test a flag.
"""
from std.memory import ArcPointer

from .column import Column
from .dtype import DataType
from .frame import DataFrame
from .series import Series


@fieldwise_init
struct OperatorRecord(Copyable, Movable):
    """What one plan node did when it ran."""

    var node: Int
    var operator: String  # the label `explain` prints for the node
    var executor: String  # "streaming" or "eager"
    var algorithm: String  # a join's index: hash_index, progression, cross,
    # eager_hash; empty for other operators
    var build_side: String  # "left" or "right" for a join, else empty
    var input_rows: Int  # rows read from the left (probe) input
    var build_rows: Int  # rows of the right (build) input, joins only
    var output_rows: Int
    var builds: Int  # indexes built for the join
    var executions: Int  # times the node ran


struct ExecutionReport(Movable):
    """Records in node order, one per node that ran."""

    var records: List[OperatorRecord]

    def __init__(out self):
        self.records = List[OperatorRecord]()

    def _slot(mut self, node: Int, operator: String, executor: String) -> Int:
        for i in range(len(self.records)):
            if self.records[i].node == node:
                return i
        self.records.append(
            OperatorRecord(node, operator, executor, "", "", 0, 0, 0, 0, 0)
        )
        return len(self.records) - 1

    def record(
        mut self,
        node: Int,
        operator: String,
        executor: String,
        input_rows: Int,
        output_rows: Int,
        *,
        executions: Int = 1,
        build_rows: Int = 0,
        builds: Int = 0,
        algorithm: String = "",
        build_side: String = "",
    ):
        """Add an execution of `node`; rows accumulate across executions
        and across the batches of a stream."""
        var at = self._slot(node, operator, executor)
        ref item = self.records[at]
        item.executor = executor
        item.input_rows += input_rows
        item.output_rows += output_rows
        item.build_rows += build_rows
        item.builds += builds
        item.executions += executions
        if algorithm:
            item.algorithm = algorithm
        if build_side:
            item.build_side = build_side

    def find(self, node: Int) -> Optional[OperatorRecord]:
        for item in self.records:
            if item.node == node:
                return Optional(item.copy())
        return None

    def frame(self) raises -> DataFrame:
        """One row per node, in node order (inputs before the operators
        that read them)."""
        var ordered = self.records.copy()
        for i in range(1, len(ordered)):
            var j = i
            while j > 0 and ordered[j - 1].node > ordered[j].node:
                var held = ordered[j].copy()
                ordered[j] = ordered[j - 1].copy()
                ordered[j - 1] = held^
                j -= 1
        var nodes = List[Int64]()
        var operators = List[String]()
        var executors = List[String]()
        var algorithms = List[String]()
        var sides = List[String]()
        var inputs = List[Int64]()
        var builds_rows = List[Int64]()
        var outputs = List[Int64]()
        var builds = List[Int64]()
        var executions = List[Int64]()
        for item in ordered:
            nodes.append(Int64(item.node))
            operators.append(item.operator)
            executors.append(item.executor)
            algorithms.append(item.algorithm)
            sides.append(item.build_side)
            inputs.append(Int64(item.input_rows))
            builds_rows.append(Int64(item.build_rows))
            outputs.append(Int64(item.output_rows))
            builds.append(Int64(item.builds))
            executions.append(Int64(item.executions))
        return DataFrame(
            [
                Series("node", Column[Int64](nodes^)),
                Series("operator", Column[String](operators^)),
                Series("executor", Column[String](executors^)),
                Series("algorithm", Column[String](algorithms^)),
                Series("build_side", Column[String](sides^)),
                Series("input_rows", Column[Int64](inputs^)),
                Series("build_rows", Column[Int64](builds_rows^)),
                Series("output_rows", Column[Int64](outputs^)),
                Series("builds", Column[Int64](builds^)),
                Series("executions", Column[Int64](executions^)),
            ]
        )

    def lines(self) raises -> List[String]:
        """Tab-separated rows for a trace, in node order."""
        var table = self.frame()
        var out = List[String]()
        for r in range(table.height()):
            var parts = List[String]()
            for name in table.columns():
                var cell = table.item(r, name)
                parts.append(
                    String(cell.int64()) if cell.dtype()
                    == DataType.INT64 else cell.string()
                )
            out.append(String("\t").join(parts))
        return out^
