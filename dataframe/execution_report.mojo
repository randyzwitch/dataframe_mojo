"""Observed execution counters for a lazy plan (#439).

`LazyFrame.profile()` attaches a report to a copy of the plan and both
executors record into it: one record per plan node that ran, with the
rows it read and wrote, how many times it ran, and for a join the rows it
built an index over, how many indexes it built, the build side and the
algorithm, and the time it took. The counts are observed, never
estimated, so tests can assert one build per streaming join and one
execution per input without timing. The recorded durations help identify
where a query spends its time.
Without a report attached, the executors only test a flag.
"""

from ._build_identity import CPU_BUILD_ID
from dataframe.storage.column import Column
from .frame import DataFrame
from .series import Series
from .execution_observation import DeviceObservation
from std.memory import ArcPointer
from std.time import perf_counter_ns
from std.os import getenv
from .dtype import DataType
from .json.value import json_quote


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
    var busy_ns: Int  # a streamed operation's time summed over its
    # batches (busy time across workers)
    var wall_ns: Int  # an eager node's wall time including its inputs; a
    # stream's last node: the whole stream's wall time
    var pipeline: Int
    var device: Optional[ArcPointer[DeviceObservation]]
    var plan_id: Int


struct ExecutionReport(Movable):
    """Records in node order, one per node that ran."""

    var records: List[OperatorRecord]
    var query_id: Int
    var default_reason: String

    def __init__(out self):
        self.records = List[OperatorRecord]()
        self.query_id = Int(perf_counter_ns())
        self.default_reason = ""

    def _slot(mut self, node: Int, operator: String, executor: String) -> Int:
        for i in range(len(self.records)):
            if self.records[i].node == node:
                return i
        self.records.append(
            OperatorRecord(
                node,
                operator,
                executor,
                "",
                "",
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                node,
                None,
                0,
            )
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
        busy_ns: Int = 0,
        wall_ns: Int = 0,
        pipeline: Int = -1,
        plan_id: Int = 0,
    ):
        """Add an execution of `node`; rows and time accumulate across
        executions and across the batches of a stream."""
        var at = self._slot(node, operator, executor)
        ref item = self.records[at]
        if item.operator != operator and item.executions == 0:
            item.operator = operator
            item.device = None
        item.executor = executor
        item.plan_id = plan_id
        item.input_rows += input_rows
        item.output_rows += output_rows
        item.build_rows += build_rows
        item.builds += builds
        item.executions += executions
        item.busy_ns += busy_ns
        item.wall_ns += wall_ns
        if pipeline >= 0:
            item.pipeline = pipeline
        if algorithm:
            item.algorithm = algorithm
        if build_side:
            item.build_side = build_side

    def observe(
        mut self, node: Int, operator: String
    ) -> ArcPointer[DeviceObservation]:
        var at = self._slot(node, operator, "placement")
        if not self.records[at].device:
            self.records[at].device = ArcPointer(DeviceObservation())
        return self.records[at].device.value().copy()

    def find(self, node: Int) -> Optional[OperatorRecord]:
        for item in self.records:
            if item.node == node:
                return Optional(item.copy())
        return None

    def frame(self) raises -> DataFrame:
        """One row per node, in node order. Shared producers use IDs after
        the consumer plan's IDs, so their work remains distinguishable."""
        var ordered = List[OperatorRecord]()
        for item in self.records:
            if item.executions:
                ordered.append(item.copy())
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
        var busy = List[Float64]()
        var wall = List[Float64]()
        var pipelines = List[Int64]()
        var plans = List[Int64]()
        var devices = List[String]()
        var statuses = List[String]()
        var reasons = List[String]()
        var launches = List[Int64]()
        var completed = List[Int64]()
        var uploads = List[Int64]()
        var downloads = List[Int64]()
        var waits = List[Int64]()
        var context = List[Float64]()
        var input = List[Float64]()
        var device_work = List[Float64]()
        var finish = List[Float64]()
        var synchronization = List[Float64]()
        var timing_kind = List[String]()
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
            busy.append(Float64(item.busy_ns) / 1e6)
            wall.append(Float64(item.wall_ns) / 1e6)
            pipelines.append(Int64(item.pipeline))
            plans.append(Int64(item.plan_id))
            var observed = item.device.value() if item.device else ArcPointer(
                DeviceObservation()
            )
            devices.append(observed[].result_device)
            statuses.append(
                "cpu_fallback" if not item.device
                and self.default_reason else observed[].status
            )
            reasons.append(
                observed[].reason if item.device else self.default_reason
            )
            launches.append(observed[].launches.load())
            completed.append(observed[].completed_launches.load())
            uploads.append(observed[].upload_bytes.load())
            downloads.append(observed[].download_bytes.load())
            waits.append(observed[].waits.load())
            var phases = observed[].phase_samples > 0
            context.append(
                Float64(observed[].context_ns) / 1e6 if phases else -1
            )
            input.append(Float64(observed[].input_ns) / 1e6 if phases else -1)
            device_work.append(
                Float64(observed[].device_work_ns) / 1e6 if phases else -1
            )
            finish.append(Float64(observed[].finish_ns) / 1e6 if phases else -1)
            synchronization.append(Float64(observed[].wait_ns.load()) / 1e6)
            timing_kind.append("host_wall" if phases else "unavailable")
        return DataFrame(
            [
                Series(
                    "cpu_build_id",
                    Column[String](
                        List[String](length=len(ordered), fill=CPU_BUILD_ID)
                    ),
                ),
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
                Series("busy_ms", Column[Float64](busy^)),
                Series("wall_ms", Column[Float64](wall^)),
                Series("pipeline", Column[Int64](pipelines^)),
                Series("result_device", Column[String](devices^)),
                Series("device_status", Column[String](statuses^)),
                Series("fallback_reason", Column[String](reasons^)),
                Series("gpu_launches", Column[Int64](launches^)),
                Series("gpu_completed_launches", Column[Int64](completed^)),
                Series("upload_bytes", Column[Int64](uploads^)),
                Series("download_bytes", Column[Int64](downloads^)),
                Series("device_waits", Column[Int64](waits^)),
                Series("context_wall_ms", Column[Float64](context^)),
                Series("input_wall_ms", Column[Float64](input^)),
                Series("device_work_wall_ms", Column[Float64](device_work^)),
                Series("finish_wall_ms", Column[Float64](finish^)),
                Series(
                    "synchronize_wall_ms", Column[Float64](synchronization^)
                ),
                Series("phase_timing_kind", Column[String](timing_kind^)),
                Series(
                    "query_id",
                    Column[Int64](
                        List[Int64](
                            length=len(ordered), fill=Int64(self.query_id)
                        )
                    ),
                ),
                Series(
                    "execution_attempt",
                    Column[Int64](List[Int64](length=len(ordered), fill=0)),
                ),
                Series(
                    "result_used",
                    Column[Bool](List[Bool](length=len(ordered), fill=True)),
                ),
                Series("plan_id", Column[Int64](plans^)),
            ]
        )


def execution_trace_path() -> String:
    return getenv("DATAFRAME_EXECUTION_TRACE")


def pipeline_record_json(
    report: DataFrame, row: Int, status: String = "completed"
) raises -> String:
    var text = String(
        '{"format":"dataframe-pipeline-observation-v1","query_status":'
    ) + json_quote(status)
    for name in report.columns():
        text += "," + json_quote(name) + ":"
        var value = report.item(row, name)
        var dtype = value.dtype()
        if dtype == DataType.INT64:
            text += String(value.int64())
        elif dtype == DataType.FLOAT64:
            text += String(value.float64())
        elif dtype == DataType.BOOL:
            var used = value.bool()
            if name == "result_used" and status in [
                "failed",
                "cpu_adjudication_failed",
            ]:
                used = False
            text += "true" if used else "false"
        else:
            text += json_quote(value.string())
    return text + "}"


def trace_execution(report: DataFrame, status: String = "completed") raises:
    """Append one terminal record per observed pipeline, after execution."""
    var path = execution_trace_path()
    if not path:
        return
    var text = String()
    for row in range(report.height()):
        if (
            report.item(row, "node").int64()
            == report.item(row, "pipeline").int64()
        ):
            text += pipeline_record_json(report, row, status) + "\n"
    with open(path, "a") as target:
        target.write(text)
