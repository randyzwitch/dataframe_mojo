"""Optional native Apple Metal provider for the shared accelerator row planner.

Importing this module loads no driver. Float64 arithmetic, floating sum/mean,
and Int64 sum remain CPU operations because their accumulators require more
precision than Apple GPU hardware provides. No arithmetic is emulated.
"""
from std.ffi import external_call
from std.memory import Pointer
from std.os import getenv
from std.os.path import exists
from std.sys import CompilationTarget
from std.time import perf_counter_ns
from .accelerator.capabilities import RowCapabilities
from .accelerator.rows import RowPlan, lower_rows
from .arrow import _c_string, _read_c_string
from .bool_column import BoolColumn
from .column import Column
from .dtype import DataType
from .execution_report import ExecutionReport
from .expr import (
    COL,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_NULL,
    ADD,
    SUB,
    MUL,
    NEG,
    GT,
    EQ,
    LT,
    GE,
    LE,
    NE,
    AND,
    OR,
    XOR,
    FILL_NULL,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    SUM,
    COUNT,
    LEN,
)
from .frame import DataFrame
from .lazy import AcceleratorBackend, LazyFrame
from .series import Series

comptime _Create = def(Int64, Int64, Int) thin abi("C") -> Int
comptime _Release = def(Int) thin abi("C") -> None
comptime _Execute = def(Int, Int, Int, Int) thin abi("C") -> Int32
comptime _Estimate = def(Int, Int, Int) thin abi("C") -> Int32
comptime _Info = def(Int, Int, Int) thin abi("C") -> Int32
comptime _Version = def() thin abi("C") -> Int64
comptime _Count = def() thin abi("C") -> Int64
comptime _Build = def() thin abi("C") -> Int


def _create_call(callback: Int, device: Int, mut error: Int) -> Int:
    return Pointer(to=callback).unsafe_bitcast[_Create]()[](
        Int64(device), Int64(1), Int(Pointer(to=error))
    )


def _estimate_call(
    callback: Int, request: List[Int64], mut memory: List[Int64], mut error: Int
) -> Int32:
    return Pointer(to=callback).unsafe_bitcast[_Estimate]()[](
        Int(request.unsafe_ptr()),
        Int(memory.unsafe_ptr()),
        Int(Pointer(to=error)),
    )


def _info_call(
    callback: Int, context: Int, mut info: List[Int64], mut error: Int
) -> Int32:
    return Pointer(to=callback).unsafe_bitcast[_Info]()[](
        context, Int(info.unsafe_ptr()), Int(Pointer(to=error))
    )


def _execute_call(
    callback: Int,
    context: Int,
    request: List[Int64],
    mut stats: List[Int64],
    mut error: Int,
) -> Int32:
    return Pointer(to=callback).unsafe_bitcast[_Execute]()[](
        context,
        Int(request.unsafe_ptr()),
        Int(stats.unsafe_ptr()),
        Int(Pointer(to=error)),
    )


def metal_library_candidates() -> List[String]:
    var candidates = List[String]()
    var configured = getenv("DATAFRAME_METAL_LIBRARY")
    if configured:
        candidates.append(configured)
        return candidates^
    var prefix = getenv("CONDA_PREFIX")
    if prefix:
        candidates.append(prefix + "/lib/libdfmetal.dylib")
    candidates.append("build/dfmetal/libdfmetal.dylib")
    return candidates^


def metal_installed() -> Bool:
    """Library presence only; this does not open Metal or discover devices."""
    comptime if CompilationTarget.is_macos():
        for path in metal_library_candidates():
            if exists(path):
                return True
    return False


struct _Library(Movable):
    var handle: Int
    var create: Int
    var release: Int
    var execute: Int
    var estimate: Int
    var info: Int
    var count: Int
    var build: Int
    var free: Int

    def __init__(out self) raises:
        self.handle = 0
        self.create = 0
        self.release = 0
        self.execute = 0
        self.estimate = 0
        self.info = 0
        self.count = 0
        self.build = 0
        self.free = 0
        comptime if not CompilationTarget.is_macos():
            raise Error("Metal requires Apple Silicon macOS")
        var path = String()
        for candidate in metal_library_candidates():
            if exists(candidate):
                path = candidate
                break
        if not path:
            raise Error(
                "Metal runtime is not installed; run native/dfmetal/build.sh or set DATAFRAME_METAL_LIBRARY"
            )
        var name = _c_string(path)
        # Darwin RTLD_NOW | RTLD_NODELETE. Every open is balanced by close;
        # the native process cache and its code remain valid between queries.
        self.handle = external_call["dlopen", Int](
            name.unsafe_ptr(), Int32(0x82)
        )
        if self.handle == 0:
            raise Error(
                "Could not load Metal runtime: "
                + _read_c_string(external_call["dlerror", Int]())
            )
        var version = self.symbol("dfm_abi_version")
        if Pointer(to=version).unsafe_bitcast[_Version]()[]() != 1:
            raise Error("Metal ABI version mismatch; rebuild native/dfmetal")
        self.create = self.symbol("dfm_context_create")
        self.release = self.symbol("dfm_context_release")
        self.execute = self.symbol("dfm_execute")
        self.estimate = self.symbol("dfm_estimate")
        self.info = self.symbol("dfm_context_info")
        self.count = self.symbol("dfm_device_count")
        self.build = self.symbol("dfm_build_id")
        self.free = self.symbol("dfm_free")

    def __deinit__(deinit self):
        if self.handle:
            _ = external_call["dlclose", Int32](self.handle)

    def symbol(self, name: String) raises -> Int:
        var c_name = _c_string(name)
        var value = external_call["dlsym", Int](
            self.handle, c_name.unsafe_ptr()
        )
        if not value:
            raise Error("Metal runtime lacks " + name)
        return value

    def check(self, status: Int32, error: Int) raises:
        if status:
            var message = _read_c_string(
                error
            ) if error else "missing native error"
            if error:
                var callback = self.free
                Pointer(to=callback).unsafe_bitcast[_Release]()[](error)
            raise Error(message)


struct _Context(Movable):
    var library: _Library
    var address: Int

    def __init__(out self, var library: _Library, device: Int) raises:
        self.address = 0
        self.library = library^
        var error = 0
        var callback = self.library.create
        self.address = _create_call(callback, device, error)
        self.library.check(Int32(self.address == 0), error)

    def __deinit__(deinit self):
        if self.address:
            var callback = self.library.release
            Pointer(to=callback).unsafe_bitcast[_Release]()[](self.address)


def _capabilities() -> RowCapabilities:
    return RowCapabilities("Metal", float64=False, wide_integer=False)


def _type(dtype: DataType) raises -> Int64:
    if dtype == DataType.FLOAT32:
        return 1
    if dtype == DataType.INT32:
        return 2
    if dtype == DataType.INT64:
        return 3
    if dtype == DataType.BOOL:
        return 4
    raise Error("Metal unsupported dtype")


def _op(op: Int) raises -> Int64:
    # The native ABI has its own versioned opcodes.
    if op == COL:
        return 0
    if op == LIT_INT:
        return 1
    if op == LIT_FLOAT:
        return 2
    if op == LIT_BOOL:
        return 3
    if op == ADD:
        return 5
    if op == SUB:
        return 6
    if op == MUL:
        return 7
    if op == GT:
        return 8
    if op == EQ:
        return 9
    if op == SUM:
        return 10
    if op == COUNT:
        return 11
    if op == LIT_NULL:
        return 12
    if op == LT:
        return 20
    if op == GE:
        return 21
    if op == LE:
        return 22
    if op == NE:
        return 23
    if op == AND:
        return 30
    if op == OR:
        return 31
    if op == XOR:
        return 32
    if op == FILL_NULL:
        return 33
    if op == NEG:
        return 50
    if op == NOT:
        return 60
    if op == IS_NULL:
        return 61
    if op == IS_NOT_NULL:
        return 62
    if op == LEN:
        return 90
    raise Error("Metal unsupported expression opcode")


def _input[T: DType](column: Column[Scalar[T]], dtype: Int64) -> List[Int64]:
    var offset = column.validity_offset()
    var has_bits = len(column._bits[]) != 0
    return [
        Int64(Int(column.unsafe_values())),
        Int64(
            Int(column.unsafe_validity().unsafe_offset(offset // 8))
        ) if has_bits else Int64(0),
        dtype,
        Int64(offset % 8),
        Int64(has_bits),
    ]


struct _Descriptors(Movable):
    var inputs: List[Int64]
    var code: List[Int64]
    var literals: List[Float64]
    var gathers: List[Int64]
    var steps: List[Int64]
    var outputs: List[Int64]

    def __init__(out self, plan: RowPlan) raises:
        self.inputs = List[Int64]()
        self.code = plan.code.copy()
        self.literals = plan.literals.copy()
        self.gathers = plan.gathers.copy()
        self.steps = List[Int64]()
        self.outputs = List[Int64]()
        for i in range(0, len(self.code), 4):
            self.code[i] = _op(Int(self.code[i]))
        for column in plan.source._columns:
            if column.dtype() == DataType.BOOL:
                var source = column.bool()
                var offset = source.validity_offset()
                var has_bits = len(source._bits[]) != 0
                self.inputs.extend(
                    [
                        Int64(
                            Int(
                                source.unsafe_values().unsafe_offset(
                                    offset // 8
                                )
                            )
                        ),
                        Int64(
                            Int(
                                source.unsafe_validity().unsafe_offset(
                                    offset // 8
                                )
                            )
                        ) if has_bits else Int64(0),
                        Int64(4),
                        Int64(offset % 8),
                        Int64(has_bits),
                    ]
                )
            elif column.dtype() == DataType.FLOAT32:
                self.inputs.extend(_input(column.float32(), 1))
            elif column.dtype() == DataType.INT32:
                self.inputs.extend(_input(column.int32(), 2))
            else:
                self.inputs.extend(_input(column.int64(), 3))
        for step in plan.steps:
            self.steps.extend(
                [
                    Int64(step.start),
                    Int64(step.nodes),
                    Int64(step.slot),
                    Int64(step.filter),
                    Int64(step.gather_start),
                    Int64(step.gather_count if step.filter else 0),
                ]
            )
        for output in plan.outputs:
            self.outputs.extend(
                [
                    Int64(0),
                    Int64(0),
                    _type(output.dtype),
                    Int64(output.slot),
                    _op(output.reduction) if output.reduction
                    >= 0 else Int64(-1),
                    Int64(output.min_count),
                ]
            )

    def request(
        self, plan: RowPlan, budget: Int, profiling: Bool
    ) raises -> List[Int64]:
        return [
            Int64(1),
            Int64(plan.source.height()),
            _type(plan.dtype),
            Int64(plan.slots),
            Int64(plan.source.width()),
            Int64(len(self.code)),
            Int64(len(plan.literals)),
            Int64(len(plan.steps)),
            Int64(len(plan.gathers)),
            Int64(len(plan.outputs)),
            Int64(plan.limit),
            Int64(plan.reductions),
            Int64(profiling),
            Int64(budget),
            Int64(Int(self.inputs.unsafe_ptr())),
            Int64(Int(self.code.unsafe_ptr())),
            Int64(Int(self.literals.unsafe_ptr())),
            Int64(Int(self.steps.unsafe_ptr())),
            Int64(Int(self.gathers.unsafe_ptr())),
            Int64(Int(self.outputs.unsafe_ptr())),
        ]


struct _Result[D: DType](Movable):
    var values: List[Scalar[Self.D]]
    var counts: List[Int64]
    var bools: List[UInt8]
    var bits: List[UInt8]

    def __init__(out self, dtype: DataType, capacity: Int):
        self.values = List[Scalar[Self.D]]()
        self.counts = List[Int64]()
        self.bools = List[UInt8]()
        self.bits = List[UInt8](length=(capacity + 7) // 8, fill=0)
        if dtype == DataType.BOOL:
            self.bools = List[UInt8](length=(capacity + 7) // 8, fill=0)
        elif dtype == DataType.INT64 and Self.D != DType.int64:
            self.counts = List[Int64](length=capacity, fill=0)
        else:
            self.values = List[Scalar[Self.D]](length=capacity, fill=0)

    def address(self, dtype: DataType) -> Int:
        if dtype == DataType.BOOL:
            return Int(self.bools.unsafe_ptr())
        if dtype == DataType.INT64 and Self.D != DType.int64:
            return Int(self.counts.unsafe_ptr())
        return Int(self.values.unsafe_ptr())

    def finish(
        deinit self, name: String, dtype: DataType, rows: Int
    ) raises -> Series:
        self.bits.resize((rows + 7) // 8, 0)
        if dtype == DataType.BOOL:
            self.bools.resize((rows + 7) // 8, 0)
            return Series(
                name,
                BoolColumn(values=self.bools^, bits=self.bits^, length=rows),
            )
        if dtype == DataType.INT64 and Self.D != DType.int64:
            self.counts.resize(rows, 0)
            return Series(
                name, Column[Int64](values=self.counts^, bits=self.bits^)
            )
        self.values.resize(rows, 0)
        return Series(
            name, Column[Scalar[Self.D]](values=self.values^, bits=self.bits^)
        )


def _execute[
    D: DType
](
    plan: RowPlan, device: Int, budget: Int, profiling: Bool, start: Int
) raises -> Tuple[DataFrame, DataFrame]:
    var descriptors = _Descriptors(plan)
    var request = descriptors.request(plan, budget, profiling)
    var library = _Library()
    var memory = List[Int64](length=8, fill=0)
    var error = 0
    var estimate = library.estimate
    var status = _estimate_call(estimate, request, memory, error)
    library.check(status, error)
    if budget >= 0 and Int(memory[2]) > budget:
        raise Error("Metal request exceeds memory budget before submission")
    var initialized = perf_counter_ns()
    var context = _Context(library^, device)
    var initialization_ns = Int(perf_counter_ns() - initialized)
    var info = List[Int64](length=6, fill=0)
    var info_fn = context.library.info
    status = _info_call(info_fn, context.address, info, error)
    context.library.check(status, error)
    var headroom = max(0, Int(info[2] - info[3]))
    if Int(memory[2]) > headroom:
        raise Error("Metal request exceeds current working set headroom")
    var capacity = 1 if plan.reductions else plan.source.height()
    var storage = List[_Result[D]]()
    for i in range(len(plan.outputs)):
        var output = _Result[D](plan.outputs[i].dtype, capacity)
        descriptors.outputs[6 * i] = Int64(
            output.address(plan.outputs[i].dtype)
        )
        descriptors.outputs[6 * i + 1] = Int64(Int(output.bits.unsafe_ptr()))
        storage.append(output^)
    var stats = List[Int64](length=17, fill=0)
    var execute = context.library.execute
    status = _execute_call(execute, context.address, request, stats, error)
    # Raw ABI addresses carry no Mojo origins. Explicitly retain the descriptor
    # buffers and request until the synchronous native call has returned.
    _ = descriptors^
    _ = request^
    context.library.check(status, error)
    var columns = List[Series]()
    for i in range(len(plan.outputs)):
        var output = storage.pop(0)
        columns.append(
            output
            ^.finish(plan.outputs[i].name, plan.outputs[i].dtype, Int(stats[8]))
        )
    var result = DataFrame(columns^)
    var report = ExecutionReport()
    report.record(
        plan.root,
        "RESIDENT ROW EXPRESSIONS",
        "metal",
        plan.source.height(),
        result.height(),
        algorithm="specialized_native_metal_stable_compaction",
        wall_ns=Int(perf_counter_ns() - start),
    )
    var fields = report.frame()._columns.copy()
    fields.append(Series("device_id", Column[Int64]([Int64(device)])))
    fields.append(
        Series("device_name", Column[String]([_read_c_string(Int(info[0]))]))
    )
    fields.append(Series("upload_bytes", Column[Int64]([stats[3]])))
    fields.append(Series("download_bytes", Column[Int64]([stats[9]])))
    fields.append(Series("shared_buffer_bytes", Column[Int64]([stats[0]])))
    fields.append(Series("host_result_bytes", Column[Int64]([stats[1]])))
    fields.append(Series("peak_requested_bytes", Column[Int64]([stats[2]])))
    fields.append(Series("kernel_launches", Column[Int64]([stats[4]])))
    fields.append(Series("synchronizations", Column[Int64]([stats[10]])))
    fields.append(Series("pipeline_cache_hit", Column[Bool]([stats[11] != 0])))
    fields.append(
        Series(
            "pipeline_compile_ms", Column[Float64]([Float64(stats[12]) / 1e6])
        )
    )
    fields.append(
        Series("staging_copy_ms", Column[Float64]([Float64(stats[13]) / 1e6]))
    )
    fields.append(
        Series("submit_wait_ms", Column[Float64]([Float64(stats[14]) / 1e6]))
    )
    fields.append(
        Series("result_copy_ms", Column[Float64]([Float64(stats[15]) / 1e6]))
    )
    fields.append(
        Series(
            "gpu_ms",
            Column[Float64]([Float64(stats[16]) / 1e6], [stats[16] >= 0]),
        )
    )
    fields.append(
        Series(
            "initialization_ms",
            Column[Float64]([Float64(initialization_ns) / 1e6]),
        )
    )
    var build_fn = context.library.build
    fields.append(
        Series(
            "runtime_build_id",
            Column[String](
                [
                    _read_c_string(
                        Pointer(to=build_fn).unsafe_bitcast[_Build]()[]()
                    )
                ]
            ),
        )
    )
    fields.append(
        Series(
            "boundaries",
            Column[String](
                [
                    "shared-buffer staging -> native Metal kernels -> synchronized CPU result"
                ]
            ),
        )
    )
    return (result^, DataFrame(fields^))


struct MetalRuntime(AcceleratorBackend):
    """Apple Metal provider; copies retain configuration, native default cache is shared.

    Construction validates configuration only. The entire query is bound before
    opening the optional library or initializing a device. Native submission is
    synchronized and serialized per cached context, including exceptional exits.
    """

    var _device: Int
    var _budget: Int

    def __init__(
        out self, device_id: Int = 0, *, memory_limit_bytes: Int = -1
    ) raises:
        if device_id < 0 or memory_limit_bytes < -1:
            raise Error(
                "Metal requires nonnegative device_id and memory_limit_bytes >= -1"
            )
        self._device = device_id
        self._budget = memory_limit_bytes

    @staticmethod
    def device_count() raises -> Int:
        var library = _Library()
        var callback = library.count
        return Int(Pointer(to=callback).unsafe_bitcast[_Count]()[]())

    def execute(self, query: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        return self._run(query, False)

    def execute_profiled(
        self, query: LazyFrame
    ) raises -> Tuple[DataFrame, DataFrame]:
        return self._run(query, True)

    def _run(
        self, query: LazyFrame, profiling: Bool
    ) raises -> Tuple[DataFrame, DataFrame]:
        var start = Int(perf_counter_ns())
        var plan = lower_rows(query, _capabilities())
        if plan.dtype == DataType.FLOAT32:
            return _execute[DType.float32](
                plan, self._device, self._budget, profiling, start
            )
        if plan.dtype == DataType.INT32:
            return _execute[DType.int32](
                plan, self._device, self._budget, profiling, start
            )
        return _execute[DType.int64](
            plan, self._device, self._budget, profiling, start
        )

    def describe(self, query: LazyFrame) -> String:
        try:
            var plan = lower_rows(query, _capabilities())
            return (
                "ENGINE accel: Metal native row expressions; "
                + String(len(plan.steps))
                + " steps, "
                + String(plan.slots)
                + " slots; optional runtime "
                + ("present" if metal_installed() else "missing")
                + "\n"
            )
        except error:
            return "ENGINE accel: " + String(error) + "\n"
