"""Apple Metal execution through Mojo kernels inside the dataframe library.

Apple silicon shares memory between the CPU and the GPU, so Metal runs
inside the CPU executor: `execute` runs the plan on the CPU with each node
offered to Metal (`LazyFrame._with_handoff`), and Metal runs the regions it
supports (`handoff.mojo`) on the columns the CPU computed below them. The
rest of the query stays on the CPU; `describe` and profiles say which nodes
Metal ran (docs/apple-metal.md). Construction, import and explanation do not
initialize a device.
"""
from std.os import getenv

from dataframe.storage.column import Column
from dataframe.execution_settings import ExecutionSettings
from dataframe.frame import DataFrame
from dataframe.lazy import AcceleratorBackend, LazyFrame
from dataframe.series import Series
from .accelerator.faults import is_data_fault, DATA_FAULT
from .accelerator.gpu.query_scope import DeviceService
from .accelerator.gpu.target import METAL_TARGET


comptime UNSUPPORTED = "Metal unsupported [rebuild]: "


def default_runtime() raises -> MetalRuntime:
    """The runtime the default provider uses, configured from the environment."""
    var device = 0
    var limit = -1
    var configured_device = getenv("DATAFRAME_ACCEL_DEVICE")
    var configured_limit = getenv("DATAFRAME_ACCEL_MEMORY_LIMIT")
    if configured_device:
        device = Int(configured_device)
        if device < 0:
            raise Error("DATAFRAME_ACCEL_DEVICE must be nonnegative")
    if configured_limit:
        limit = Int(configured_limit)
        if limit < 0:
            raise Error("DATAFRAME_ACCEL_MEMORY_LIMIT must be nonnegative")
    return MetalRuntime(device, memory_limit_bytes=limit)


def metal_installed() -> Bool:
    """Whether this target supports Mojo Metal; no device is initialized."""
    from .accelerator.gpu.identity import available

    return available()


def _unsupported(query: LazyFrame) -> String:
    """Why automatic placement does not hand Metal the whole `query`."""
    _ = query
    return (
        UNSUPPORTED
        + "Metal never runs a whole plan; the CPU executor places each node"
        " Metal supports by measured cost when DATAFRAME_HANDOFF_CALIBRATION"
        ' names a calibration, and engine="accel" hands every one over'
    )


struct MetalRuntime(AcceleratorBackend):
    """Apple Metal provider with Mojo-owned kernels, buffers and query lifetime."""

    var _device: Int
    var _budget: Int
    var _device_service: DeviceService
    var _execution_settings: ExecutionSettings

    def __init__(
        out self, device_id: Int = 0, *, memory_limit_bytes: Int = -1
    ) raises:
        if device_id < 0 or memory_limit_bytes < -1:
            raise Error(
                "Metal requires nonnegative device_id and memory_limit_bytes"
                " >= -1"
            )
        self._device = device_id
        self._budget = memory_limit_bytes
        self._execution_settings = ExecutionSettings(True, True, 65536)
        self._device_service = DeviceService(device_id)

    @staticmethod
    def device_count() raises -> Int:
        from .accelerator.gpu.identity import device_count

        return device_count()

    def execute(self, query: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        return self._execute(query, False)

    def execute_profiled(
        self, query: LazyFrame
    ) raises -> Tuple[DataFrame, DataFrame]:
        return self._execute(query, True)

    def describe(self, query: LazyFrame) -> String:
        """Which nodes Metal runs; the CPU runs the rest."""
        try:
            var nodes = query._handoff_nodes()
            if len(nodes) == 0:
                return (
                    "ENGINE accel: CPU executor; Metal supports no node of"
                    " this plan (unsupported)\n"
                )
            var text = "ENGINE accel: CPU executor; Metal runs:\n"
            for label in nodes:
                text += "  HANDOFF metal: " + label + "\n"
            return text^
        except error:
            return "ENGINE accel: " + String(error) + "\n"

    def _execute(
        self, query: LazyFrame, profiling: Bool
    ) raises -> Tuple[DataFrame, DataFrame]:
        comptime if METAL_TARGET:
            var configured = self._configured(query)
            var fault: String
            try:
                return configured._run(query, profiling)
            except error:
                if not is_data_fault(error):
                    raise error^
                fault = String(error)
            return configured._adjudicate(query, profiling, fault)
        else:
            raise Error(
                "Mojo Metal execution requires an Apple Metal compilation"
                " target; " + self.describe(query)
            )

    def _run(
        self, query: LazyFrame, profiling: Bool
    ) raises -> Tuple[DataFrame, DataFrame]:
        """Run `query` on the CPU executor, handing Metal the nodes it
        supports."""
        var settings = self._execution_settings.copy()
        var plan = query._with_handoff()
        if profiling:
            return plan.profile(
                engine="cpu",
                optimize=settings.optimize,
                streaming=settings.streaming,
                batch_size=settings.batch_size,
            )
        var result = plan.collect(
            engine="cpu",
            optimize=settings.optimize,
            streaming=settings.streaming,
            batch_size=settings.batch_size,
        )
        return (result^, DataFrame(List[Series]()))

    def _adjudicate(
        self, query: LazyFrame, profiling: Bool, fault: String
    ) raises -> Tuple[DataFrame, DataFrame]:
        """Decide a query whose kernels observed a data fault on the CPU.

        Metal evaluates every checked value, including values the CPU
        executor never computes (rows a head or filter prunes first), and
        does not reproduce the CPU's evaluation order. The CPU therefore
        decides the outcome: the error it raises, or its result when the
        faulting values are never evaluated. The report says so, and a CPU
        error keeps its message first, then names the Metal fault.
        """
        try:
            return self._decide_on_cpu(query, profiling)
        except cpu_error:
            var message = String(cpu_error)
            var observed = fault.replace(DATA_FAULT, "")
            if observed in message:
                raise cpu_error^
            raise Error(message + " [Metal data fault: " + observed + "]")

    def _decide_on_cpu(
        self, query: LazyFrame, profiling: Bool
    ) raises -> Tuple[DataFrame, DataFrame]:
        var settings = self._execution_settings.copy()
        if not profiling:
            var result = query.collect(
                engine="cpu",
                optimize=settings.optimize,
                streaming=settings.streaming,
                batch_size=settings.batch_size,
            )
            return (
                result^,
                DataFrame(
                    [Series("fault_adjudication", Column[String](["cpu"]))]
                ),
            )
        var measured = query.profile(
            engine="cpu",
            optimize=settings.optimize,
            streaming=settings.streaming,
            batch_size=settings.batch_size,
        )
        var marker = Series(
            "fault_adjudication",
            Column[String](
                List[String](length=measured[1].height(), fill="cpu")
            ),
        )
        return (measured[0].copy(), measured[1].with_column(marker^))

    def _configured(self, query: LazyFrame) -> Self:
        var configured = self.copy()
        if query._plan._accelerator_settings:
            configured._execution_settings = (
                query._plan._accelerator_settings.value().copy()
            )
        return configured^
