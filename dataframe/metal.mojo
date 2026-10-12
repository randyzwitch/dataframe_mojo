"""Optional native Apple Metal provider for the shared accelerator row planner.

Importing this module loads no driver. Binary64 arithmetic uses integer-only
GPU emulation, including floating sum/mean accumulation. Int64/UInt64 sums and
integer means require a GPU proof that all exact integer partials fit the native
range. Decimal32/64 sums use exact two-word accumulation with Decimal128
results. Automatic placement requires separate calibration for emulated work.
"""
from .accelerator.metadata import schema_frame
from std.ffi import external_call
from std.math import isfinite
from std.memory import Pointer
from std.os import getenv
from std.os.path import exists
from std.sys import CompilationTarget
from std.time import perf_counter_ns
from .accelerator.cost import (
    PlanEstimate,
    CalibrationKey,
    select_auto as select_cost,
)
from .accelerator.cpu_work import cpu_features
from ._build_identity import CPU_BUILD_ID
from .parallel import configured_workers
from .json_value import JsonValue
from .accelerator.capabilities import RowCapabilities
from .accelerator.sets import has_set_operations, lower_set_operations
from .accelerator.constants import (
    has_literal_text_parsing,
    fold_literal_text_parsing,
)
from .accelerator.distinct_counts import (
    has_distinct_counts,
    lower_distinct_counts,
)
from .accelerator.windows import (
    has_partition_aggregates,
    lower_partition_aggregates,
)
from .accelerator.aggregates import (
    has_aggregate_compositions,
    lower_aggregate_compositions,
)
from .accelerator.dictionaries import (
    DictionaryCodes,
    has_dictionary_columns,
    encode_query,
    restore_frame,
    lower_dictionary_operations,
)
from .accelerator.memory import checked_add
from .accelerator.rows import RowPlan, lower_rows, ROW_SORT_KEY, ROW_RANK_KEY
from .accelerator.joins import (
    JoinDag,
    DagTask,
    DAG_ROWS,
    DAG_JOIN,
    has_join,
    has_scalar_consumers,
    lower_join_dag,
)
from .arrow import _c_string, _read_c_string
from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .execution_report import ExecutionReport
from .expr import (
    COL,
    DIV,
    DT_YEAR,
    DT_TIME,
    DT_TRUNCATE,
    FLOORDIV,
    MOD,
    POW,
    CLIP_LOW,
    CLIP_HIGH,
    FILL_NAN,
    KEEP_NULLS,
    ABS,
    SQRT,
    FLOOR,
    CEIL,
    ROUND,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    WHEN,
    INT_IS_IN,
    CAST,
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
    MEAN,
    MIN,
    MAX,
    COUNT,
    LEN,
    FIRST,
    LAST,
    ANY,
    ALL,
    NULL_COUNT,
    ARG_MIN,
    ARG_MAX,
)
from .frame import DataFrame
from .lazy import AcceleratorBackend, LazyFrame
from .accelerator.rows import (
    ROW_GROUP_HEADER,
    ROW_GROUP_HEADER_LAST,
    ROW_GROUP_HEADER_WINDOW,
    ROW_SLICE,
    ROW_GROUP_KEY,
    ROW_GROUP_OUTPUT,
)
from .series import Series

comptime _Create = def(Int64, Int64, Int) thin abi("C") -> Int
comptime _Release = def(Int) thin abi("C") -> None
comptime _Execute = def(Int, Int, Int, Int) thin abi("C") -> Int32
comptime _Estimate = def(Int, Int, Int) thin abi("C") -> Int32
comptime _Probe = def(Int64, Int, Int, Int) thin abi("C") -> Int32
comptime _Info = def(Int, Int, Int) thin abi("C") -> Int32
comptime _Version = def() thin abi("C") -> Int64
comptime _Count = def() thin abi("C") -> Int64
comptime _Build = def() thin abi("C") -> Int
comptime _PlanFeatures = def(Int, Int) thin abi("C") -> Int64
comptime _JoinPrepare = def(Int, Int, Int, Int, Int) thin abi("C") -> Int32
comptime _JoinFinish = def(Int, Int, Int64, Int, Int) thin abi("C") -> Int32
comptime _QueryCreate = def(Int, Int64, Int) thin abi("C") -> Int
comptime _ResidentRun = def(
    Int, Int, Int, Int64, Int64, Int, Int, Int
) thin abi("C") -> Int32


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
                "Metal runtime is not installed; run native/dfmetal/build.sh or"
                " set DATAFRAME_METAL_LIBRARY"
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
        if Pointer(to=version).unsafe_bitcast[_Version]()[]() != 2:
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
        var value = self.optional_symbol(name)
        if not value:
            raise Error("Metal runtime lacks " + name)
        return value

    def optional_symbol(self, name: String) -> Int:
        var c_name = _c_string(name)
        var value = external_call["dlsym", Int](
            self.handle, c_name.unsafe_ptr()
        )
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
    return RowCapabilities(
        "Metal",
        float64=False,
        float64_storage=True,
        float64_emulated=True,
        float_division=True,
        float_rounding=True,
        float_modulo=True,
        float_sqrt=True,
        wide_integer=False,
        bounded_integer_sum=True,
        extended_integers=True,
        mixed_types=True,
        casts=True,
        extrema=True,
        fixed_logical=True,
        row_extras=True,
        order=True,
        grouped=True,
        extra_reductions=True,
        slices=True,
        scalar_regions=True,
        temporal_fields=True,
        ranks=True,
        decimal128=True,
        reuse_column_slots=True,
        max_nodes=128,
        max_slots=128,
        max_steps=128,
        max_join_columns=128,
    )


def _type(dtype: DataType) raises -> Int64:
    if dtype.is_decimal() and dtype.decimal_width() == 128:
        return 12
    if dtype.physical() == DataType.FLOAT64:
        return 11
    if dtype.physical() == DataType.FLOAT32:
        return 1
    if dtype.physical() == DataType.INT32:
        return 2
    if dtype.physical() == DataType.INT64:
        return 3
    if dtype.physical() == DataType.BOOL:
        return 4
    if dtype.physical() == DataType.INT8:
        return 5
    if dtype.physical() == DataType.INT16:
        return 6
    if dtype.physical() == DataType.UINT8:
        return 7
    if dtype.physical() == DataType.UINT16:
        return 8
    if dtype.physical() == DataType.UINT32:
        return 9
    if dtype.physical() == DataType.UINT64:
        return 10
    raise Error("Metal unsupported dtype")


def _op(op: Int) raises -> Int64:
    if op >= DT_YEAR and op <= DT_TRUNCATE:
        return Int64(130 + op - DT_YEAR)
    if op == INT_IS_IN:
        return 172
    if (
        op == FIRST
        or op == LAST
        or op == ANY
        or op == ALL
        or op == NULL_COUNT
        or op == ARG_MIN
        or op == ARG_MAX
    ):
        return Int64(op)
    if op == ROW_GROUP_HEADER:
        return 201
    if op == ROW_GROUP_HEADER_LAST:
        return 205
    if op == ROW_GROUP_HEADER_WINDOW:
        return 209
    if op == ROW_SLICE:
        return 206
    if op == ROW_GROUP_KEY:
        return 202
    if op == ROW_GROUP_OUTPUT:
        return 203
    # The native ABI has its own versioned opcodes.
    if op == ROW_RANK_KEY:
        return 207
    if op == ROW_SORT_KEY:
        return 200
    if op == FLOORDIV:
        return 25
    if op == MOD:
        return 26
    if op == POW:
        return 27
    if op == CLIP_LOW:
        return 28
    if op == CLIP_HIGH:
        return 29
    if op == FILL_NAN:
        return 34
    if op == KEEP_NULLS:
        return 35
    if op == ABS:
        return 51
    if op == SQRT:
        return 52
    if op == FLOOR:
        return 55
    if op == CEIL:
        return 56
    if op == ROUND:
        return 57
    if op == IS_NAN:
        return 63
    if op == IS_NOT_NAN:
        return 64
    if op == IS_FINITE:
        return 65
    if op == IS_INFINITE:
        return 66
    if op == WHEN:
        return 100
    if op == CAST:
        return 79
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
    if op == DIV:
        return 24
    if op == GT:
        return 8
    if op == EQ:
        return 9
    if op == MIN:
        return 80
    if op == MAX:
        return 81
    if op == SUM:
        return 10
    if op == MEAN:
        return 82
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


def _input_series(column: Series) raises -> List[Int64]:
    if column.dtype().is_decimal() and column.dtype().decimal_width() == 128:
        return _input(column.numeric[DType.int128](), Int64(12))
    if column.dtype() == DataType.BOOL:
        var source = column.bool()
        var offset = source.validity_offset()
        var has_bits = len(source._bits[]) != 0
        return [
            Int64(Int(source.unsafe_values().unsafe_offset(offset // 8))),
            Int64(
                Int(source.unsafe_validity().unsafe_offset(offset // 8))
            ) if has_bits else Int64(0),
            Int64(4),
            Int64(offset % 8),
            Int64(has_bits),
        ]
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if column.dtype().physical() == DataType.of(D):
            return _input(column.numeric[D](), _type(column.dtype()))
    raise Error("Metal unsupported input dtype")


struct _Descriptors(Movable):
    var inputs: List[Int64]
    var code: List[Int64]
    var literals: List[Float64]
    var gathers: List[Int64]
    var steps: List[Int64]
    var outputs: List[Int64]
    var node_types: List[Int64]
    var slot_types: List[Int64]
    var typed: Bool

    def __init__(out self, plan: RowPlan) raises:
        self.inputs = List[Int64]()
        self.code = plan.code.copy()
        self.literals = plan.literals.copy()
        self.gathers = plan.gathers.copy()
        self.steps = List[Int64]()
        self.outputs = List[Int64]()
        self.node_types = List[Int64]()
        self.slot_types = List[Int64]()
        self.typed = False
        for step in plan.steps:
            self.typed |= step.order or step.grouped
        for output in plan.outputs:
            self.typed |= (
                output.reduction == MIN
                or output.reduction == MEAN
                or output.reduction == MAX
                or output.reduction == FIRST
                or output.reduction == LAST
                or output.reduction == ANY
                or output.reduction == ALL
                or output.reduction == NULL_COUNT
                or output.reduction == ARG_MIN
                or output.reduction == ARG_MAX
            )
        for dtype in plan.node_dtypes:
            self.node_types.append(_type(dtype))
            self.typed |= dtype != plan.dtype and dtype != DataType.BOOL
        for dtype in plan.slot_dtypes:
            self.slot_types.append(_type(dtype))
            self.typed |= dtype == DataType.FLOAT64
            self.typed |= dtype != plan.dtype and dtype != DataType.BOOL
            self.typed |= dtype != dtype.physical()
        for i in range(0, len(self.code), 4):
            self.typed |= (
                self.code[i] == CAST
                or self.code[i] == WHEN
                or self.code[i] == INT_IS_IN
                or self.code[i] == ROW_SLICE
                or (
                    self.code[i] == ROUND
                    and self.code[i + 3] != 0
                    and plan.node_dtypes[i // 4].is_float()
                )
            )
            if self.code[i] == ROW_GROUP_OUTPUT:
                self.code[i + 3] = Int64(208) if self.code[
                    i + 3
                ] == SUM and plan.node_dtypes[i // 4].is_decimal() else _op(
                    Int(self.code[i + 3])
                )
            self.code[i] = _op(Int(self.code[i]))
        for column in plan.source._columns:
            self.inputs.extend(_input_series(column))
        for step in plan.steps:
            self.steps.extend(
                [
                    Int64(step.start),
                    Int64(step.nodes),
                    Int64(step.slot),
                    Int64(4) if plan.code[step.start * 4]
                    == ROW_RANK_KEY else Int64(3) if step.grouped else (
                        Int64(2) if step.order else Int64(step.filter)
                    ),
                    Int64(step.gather_start),
                    Int64(
                        step.gather_count if step.filter or step.order else 0
                    ),
                ]
            )
        for output in plan.outputs:
            self.outputs.extend(
                [
                    Int64(0),
                    Int64(0),
                    _type(output.dtype),
                    Int64(output.slot),
                    Int64(208) if output.reduction == SUM
                    and output.dtype.is_decimal() else _op(
                        output.reduction
                    ) if output.reduction
                    >= 0 else Int64(-1),
                    Int64(output.min_count),
                ]
            )

    def request(
        self, plan: RowPlan, budget: Int, profiling: Bool
    ) raises -> List[Int64]:
        return [
            Int64(2),
            Int64(plan.source.height()),
            Int64(0) if self.typed else _type(plan.dtype),
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
            Int64(Int(self.node_types.unsafe_ptr())) if self.typed else Int64(
                0
            ),
            Int64(Int(self.slot_types.unsafe_ptr())) if self.typed else Int64(
                0
            ),
        ]


def _allocate_output(
    name: String, dtype: DataType, capacity: Int
) raises -> Series:
    return Series.full_null(name, dtype, capacity)


def _output_addresses(column: Series) raises -> Tuple[Int, Int]:
    if column.dtype().is_decimal() and column.dtype().decimal_width() == 128:
        var value = column.numeric[DType.int128]()
        return (Int(value.unsafe_values()), Int(value.unsafe_validity()))
    if column.dtype() == DataType.BOOL:
        var value = column.bool()
        return (Int(value.unsafe_values()), Int(value.unsafe_validity()))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if column.dtype().physical() == DataType.of(D):
            var value = column.numeric[D]()
            return (Int(value.unsafe_values()), Int(value.unsafe_validity()))
    raise Error("Metal unsupported output dtype")


def _direct_filtered_reduction(plan: RowPlan) -> Bool:
    if not plan.reductions:
        return False
    var filtered = False
    for step in plan.steps:
        if step.order or step.grouped:
            return False
        for i in range(step.start, step.start + step.nodes):
            if plan.code[4 * i] == ROW_SLICE:
                return False
        filtered |= step.filter
    for output in plan.outputs:
        if (
            output.reduction != SUM
            and output.reduction != COUNT
            and output.reduction != LEN
        ):
            return False
    return filtered


def _plan_features(
    library: _Library, request: List[Int64], join: Bool = False
) raises -> Int64:
    var callback = library.optional_symbol(
        "dfm_join_plan_features" if join else "dfm_plan_features"
    )
    if not callback:
        return 0
    var error = 0
    var features = Pointer(to=callback).unsafe_bitcast[_PlanFeatures]()[](
        Int(request.unsafe_ptr()), Int(Pointer(to=error))
    )
    library.check(Int32(features < 0), error)
    return features


def _execute(
    plan: RowPlan, device: Int, budget: Int, profiling: Bool, start: Int
) raises -> Tuple[DataFrame, DataFrame]:
    var descriptors = _Descriptors(plan)
    var request = descriptors.request(plan, budget, profiling)
    var library = _Library()
    var features = _plan_features(library, request)
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
    var idle_bytes = 0
    var workspace_info = context.library.optional_symbol("dfm_workspace_info")
    if workspace_info:
        var workspace = List[Int64](length=5, fill=0)
        status = _info_call(workspace_info, context.address, workspace, error)
        context.library.check(status, error)
        idle_bytes = Int(workspace[0])
    var headroom = max(0, Int(info[2]) - max(0, Int(info[3]) - idle_bytes))
    if Int(memory[2]) > headroom:
        raise Error("Metal request exceeds current working set headroom")
    var capacity = 1 if plan.reductions else plan.source.height()
    for step in plan.steps:
        if step.grouped:
            capacity = max(1, capacity)
    var storage = List[Series]()
    for i in range(len(plan.outputs)):
        var output = _allocate_output(
            plan.outputs[i].name, plan.outputs[i].dtype, capacity
        )
        var addresses = _output_addresses(output)
        descriptors.outputs[6 * i] = Int64(addresses[0])
        descriptors.outputs[6 * i + 1] = Int64(addresses[1])
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
    for column in storage:
        columns.append(column.slice(0, Int(stats[8])))
    var result = DataFrame(columns^)
    var report = _native_report(
        context,
        device,
        plan.root,
        "RESIDENT ROW EXPRESSIONS",
        plan.source.height(),
        result.height(),
        stats,
        info,
        initialization_ns,
        start,
        "native_metal_predicate_reductions" if features
        & 1 else "native_metal_hash_groups" if features
        & 2 else "native_metal_integer_radix_sort" if features
        & 4 else "specialized_native_metal_stable_compaction",
        (
            "shared-buffer staging -> native Metal kernels -> synchronized CPU"
            " result"
        ),
    )
    return (result^, report^)


def _native_report(
    context: _Context,
    device: Int,
    root: Int,
    label: String,
    input_rows: Int,
    output_rows: Int,
    stats: List[Int64],
    info: List[Int64],
    initialization_ns: Int,
    start: Int,
    algorithm: String,
    boundaries: String,
    *,
    build_rows: Int = 0,
    builds: Int = 0,
    build_side: String = "",
    resident_input_bytes: Int = 0,
    resident_output_bytes: Int = 0,
) raises -> DataFrame:
    var report = ExecutionReport()
    report.record(
        root,
        label,
        "metal",
        input_rows,
        output_rows,
        algorithm=algorithm,
        build_rows=build_rows,
        builds=builds,
        build_side=build_side,
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
    fields.append(
        Series(
            "resident_input_bytes", Column[Int64]([Int64(resident_input_bytes)])
        )
    )
    fields.append(
        Series(
            "resident_output_bytes",
            Column[Int64]([Int64(resident_output_bytes)]),
        )
    )
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
            Column[String]([boundaries]),
        )
    )
    return DataFrame(fields^)


struct _JoinDescriptors(Movable):
    var inputs: List[Int64]
    var keys: List[Int64]
    var outputs: List[Int64]

    def __init__(
        out self, task: DagTask, left: DataFrame, right: DataFrame
    ) raises:
        self.inputs = List[Int64]()
        self.keys = task.keys.copy()
        self.outputs = List[Int64]()
        for side in range(2):
            var frame = left.copy() if side == 0 else right.copy()
            for column in frame._columns:
                self.inputs.extend(_input_series(column))
                self.inputs.append(Int64(side))
        for field in task.fields:
            self.outputs.extend(
                [
                    Int64(0),
                    Int64(0),
                    _type(field.dtype),
                    Int64(field.left),
                    Int64(field.right),
                    Int64(field.prefer_right),
                ]
            )

    def request(
        self,
        task: DagTask,
        left: DataFrame,
        right: DataFrame,
        budget: Int,
        profiling: Bool,
    ) -> List[Int64]:
        return [
            Int64(1),
            Int64(left.height()),
            Int64(right.height()),
            Int64(left.width() + right.width()),
            Int64(len(self.keys) // 2),
            Int64(len(task.fields)),
            Int64(task.how),
            Int64(task.limit),
            Int64(budget),
            Int64(profiling),
            Int64(Int(self.inputs.unsafe_ptr())),
            Int64(Int(self.keys.unsafe_ptr())),
            Int64(Int(self.outputs.unsafe_ptr())),
        ]


def _join_finish_call(
    callback: Int,
    session: Int,
    outputs: List[Int64],
    mut stats: List[Int64],
    mut error: Int,
) -> Int32:
    return Pointer(to=callback).unsafe_bitcast[_JoinFinish]()[](
        session,
        Int(outputs.unsafe_ptr()),
        Int64(len(outputs) // 6),
        Int(stats.unsafe_ptr()),
        Int(Pointer(to=error)),
    )


struct _ResidentScope(Movable):
    var context: _Context
    var address: Int
    var release: Int
    var release_table: Int
    var rows: Int
    var join: Int
    var info: Int
    var query_info: Int
    var export: Int
    var device_info: List[Int64]
    var initialization_ns: Int

    def __init__(out self, device: Int, budget: Int) raises:
        self.address = 0
        self.release = 0
        self.release_table = 0
        self.rows = 0
        self.join = 0
        self.info = 0
        self.query_info = 0
        self.export = 0
        self.device_info = List[Int64](length=6, fill=0)
        self.initialization_ns = 0
        var library = _Library()
        var version = library.symbol("dfm_resident_abi_version")
        if Pointer(to=version).unsafe_bitcast[_Version]()[]() != 1:
            raise Error(
                "Metal resident ABI version mismatch; rebuild native/dfmetal"
            )
        var create = library.symbol("dfm_query_create")
        self.release = library.symbol("dfm_query_release")
        self.release_table = library.symbol("dfm_resident_release")
        self.rows = library.symbol("dfm_resident_rows")
        self.join = library.symbol("dfm_resident_join")
        self.info = library.symbol("dfm_resident_info")
        self.query_info = library.symbol("dfm_query_info")
        self.export = library.symbol("dfm_resident_export")
        var start = perf_counter_ns()
        self.context = _Context(library^, device)
        self.initialization_ns = Int(perf_counter_ns() - start)
        var error = 0
        var status = _info_call(
            self.context.library.info,
            self.context.address,
            self.device_info,
            error,
        )
        self.context.library.check(status, error)
        self.address = Pointer(to=create).unsafe_bitcast[_QueryCreate]()[](
            self.context.address, Int64(budget), Int(Pointer(to=error))
        )
        self.context.library.check(Int32(self.address == 0), error)

    def __deinit__(deinit self):
        if self.address:
            var callback = self.release
            Pointer(to=callback).unsafe_bitcast[_Release]()[](self.address)


struct _ResidentValue(Movable):
    var frame: DataFrame
    var address: Int
    var release: Int
    var rows: Int
    var retained: Int

    def __init__(
        out self,
        var frame: DataFrame,
        address: Int = 0,
        release: Int = 0,
        rows: Int = -1,
        retained: Int = 0,
    ):
        self.rows = frame.height() if rows < 0 else rows
        self.frame = frame^
        self.address = address
        self.release = release
        self.retained = retained

    def __deinit__(deinit self):
        if self.address:
            var callback = self.release
            Pointer(to=callback).unsafe_bitcast[_Release]()[](self.address)


def _resident_run_call(
    callback: Int,
    query: Int,
    request: List[Int64],
    refs: List[Int64],
    final: Bool,
    mut address: Int,
    mut stats: List[Int64],
    mut error: Int,
) -> Int32:
    return Pointer(to=callback).unsafe_bitcast[_ResidentRun]()[](
        query,
        Int(request.unsafe_ptr()),
        Int(refs.unsafe_ptr()),
        Int64(len(refs) // 2),
        Int64(final),
        Int(Pointer(to=address)),
        Int(stats.unsafe_ptr()),
        Int(Pointer(to=error)),
    )


def _execute_join_dag(
    dag: JoinDag,
    device: Int,
    budget: Int,
    profiling: Bool,
    query_start: Int,
) raises -> Tuple[DataFrame, DataFrame]:
    var scope = _ResidentScope(device, budget)
    var values = List[_ResidentValue]()
    var report = DataFrame(List[Series]())
    var result = DataFrame(List[Series]())
    var uses = List[Int](length=len(dag.tasks), fill=0)
    for task in dag.tasks:
        if task.source >= 0:
            uses[task.source] += 1
        if task.kind == DAG_JOIN:
            uses[task.right] += 1
    for index in range(len(dag.tasks)):
        var task = dag.tasks[index].copy()
        var start = Int(perf_counter_ns())
        var final = index == dag.root
        var input_rows: Int
        var input_bytes = 0
        var algorithm = String("native_metal_resident_rows")
        var label = String("RESIDENT ROW EXPRESSIONS")
        var stats = List[Int64](length=17, fill=0)
        var address = 0
        var error = 0
        var status: Int32
        if task.kind == DAG_JOIN:
            var left = values[task.source].frame.copy()
            var right = values[task.right].frame.copy()
            var descriptors = _JoinDescriptors(task, left, right)
            var request = descriptors.request(
                task, left, right, budget, profiling
            )
            request[1] = Int64(values[task.source].rows)
            request[2] = Int64(values[task.right].rows)
            input_rows = values[task.source].rows
            input_bytes = values[task.source].retained
            if values[task.right].address != values[task.source].address:
                input_bytes += values[task.right].retained
            var refs = List[Int64]()
            for side in range(2):
                var parent = task.source if side == 0 else task.right
                for column in range(values[parent].frame.width()):
                    refs.extend([Int64(values[parent].address), Int64(column)])
            var join_features = _plan_features(
                scope.context.library, request, True
            )
            status = _resident_run_call(
                scope.join,
                scope.address,
                request,
                refs,
                final,
                address,
                stats,
                error,
            )
            _ = descriptors^
            _ = request^
            _ = refs^
            scope.context.library.check(status, error)
            algorithm = (
                "native_metal_resident_hash_join" if join_features
                & 1 else "native_metal_resident_sorted_join"
            )
            label = "JOIN"
        else:
            if task.source >= 0:
                task.query._frames[0] = values[task.source].frame.copy()
            var plan = lower_rows(
                task.query, _capabilities(), terminal_boundary=task.terminal
            )
            plan.root = task.root
            if len(task.query._nodes) == 1 and task.source < 0:
                # Keep original CPU scans until their first consumer. This
                # avoids a standalone upload/submit/wait for an identity scan.
                values.append(_ResidentValue(plan.source.copy()))
                continue
            var descriptors = _Descriptors(plan)
            descriptors.typed = True
            var request = descriptors.request(plan, budget, profiling)
            var source_address = 0
            var source_frame = plan.source.copy()
            input_rows = plan.source.height()
            if task.source >= 0:
                source_address = values[task.source].address
                source_frame = values[task.source].frame.copy()
                input_rows = values[task.source].rows
                input_bytes = values[task.source].retained
            request[1] = Int64(input_rows)
            var features = _plan_features(scope.context.library, request)
            if features & 1:
                algorithm = "native_metal_predicate_reductions"
            elif features & 2:
                algorithm = "native_metal_hash_groups"
            elif features & 4:
                algorithm = "native_metal_integer_radix_sort"
            var refs = List[Int64]()
            for name in plan.source.columns():
                refs.extend(
                    [Int64(source_address), Int64(source_frame._index(name))]
                )
            status = _resident_run_call(
                scope.rows,
                scope.address,
                request,
                refs,
                final,
                address,
                stats,
                error,
            )
            _ = descriptors^
            _ = request^
            _ = refs^
            scope.context.library.check(status, error)
        var metadata = List[Int64](length=3, fill=0)
        status = _info_call(scope.info, address, metadata, error)
        if status:
            var release = scope.release_table
            Pointer(to=release).unsafe_bitcast[_Release]()[](address)
            scope.context.library.check(status, error)
        values.append(
            _ResidentValue(
                task.schema.copy(),
                address,
                scope.release_table,
                Int(metadata[0]),
                Int(metadata[2]),
            )
        )
        if final:
            var output_columns = List[Series]()
            var outputs = List[Int64]()
            for i in range(task.schema.width()):
                var field = task.schema._columns[i].copy()
                var column = _allocate_output(
                    field.name(), field.dtype(), Int(metadata[0])
                )
                var pointers = _output_addresses(column)
                outputs.extend(
                    [
                        Int64(pointers[0]),
                        Int64(pointers[1]),
                        _type(field.dtype()),
                        Int64(i),
                        Int64(-1),
                        Int64(0),
                    ]
                )
                output_columns.append(column^)
            status = _join_finish_call(
                scope.export, address, outputs, stats, error
            )
            _ = outputs^
            scope.context.library.check(status, error)
            result = DataFrame(output_columns^, height=Int(metadata[0]))
        var region = _native_report(
            scope.context,
            device,
            task.root,
            label,
            input_rows,
            Int(metadata[0]),
            stats,
            scope.device_info,
            scope.initialization_ns if report.width() == 0 else 0,
            start,
            algorithm,
            (
                "CPU source staging -> resident Metal regions -> synchronized"
                " final CPU result"
            ),
            build_rows=(
                values[task.source].rows if task.how
                == 2 else values[task.right].rows
            ) if task.kind
            == DAG_JOIN else 0,
            builds=Int(task.kind == DAG_JOIN),
            build_side=("left" if task.how == 2 else "right") if task.kind
            == DAG_JOIN else "",
            resident_input_bytes=input_bytes,
            resident_output_bytes=Int(metadata[2]),
        )
        report = region^ if report.width() == 0 else report.vstack(region)
        if task.source >= 0:
            uses[task.source] -= 1
            if uses[task.source] == 0:
                values[task.source] = _ResidentValue(DataFrame(List[Series]()))
        if task.kind == DAG_JOIN:
            uses[task.right] -= 1
            if uses[task.right] == 0:
                values[task.right] = _ResidentValue(DataFrame(List[Series]()))
    var query_info = List[Int64](length=3, fill=0)
    var error = 0
    var status = _info_call(scope.query_info, scope.address, query_info, error)
    scope.context.library.check(status, error)
    report = report.with_column(
        Series(
            "query_peak_requested_bytes",
            Column[Int64](
                List[Int64](length=report.height(), fill=query_info[1])
            ),
        )
    )
    var elapsed = Float64(Int(perf_counter_ns()) - query_start) / 1e6
    report = report.with_column(
        Series(
            "query_wall_ms",
            Column[Float64](
                List[Float64](length=report.height(), fill=elapsed)
            ),
        )
    )
    return (result^, report^)


def _probe_metadata(
    library: _Library, device: Int, request: List[Int64]
) raises -> List[Int64]:
    var callback = library.symbol("dfm_probe")
    var info = List[Int64](length=8, fill=0)
    var error = 0
    var status = Pointer(to=callback).unsafe_bitcast[_Probe]()[](
        Int64(device),
        Int(request.unsafe_ptr()) if len(request) else 0,
        Int(info.unsafe_ptr()),
        Int(Pointer(to=error)),
    )
    library.check(status, error)
    return info^


def _metal_calibration_key(
    library: _Library,
    metadata: List[Int64],
    optimize: Bool,
    streaming: Bool,
    batch_size: Int,
) -> CalibrationKey:
    var callback = library.build
    var native_id = _read_c_string(
        Pointer(to=callback).unsafe_bitcast[_Build]()[]()
    )
    return CalibrationKey(
        "metal",
        "registry=" + String(metadata[0]) + ";unified=" + String(metadata[3]),
        native_id^,
        CPU_BUILD_ID,
        _read_c_string(Int(metadata[7])),
        configured_workers(),
        optimize,
        streaming,
        batch_size,
    )


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
                "Metal requires nonnegative device_id and memory_limit_bytes"
                " >= -1"
            )
        self._device = device_id
        self._budget = memory_limit_bytes

    @staticmethod
    def device_count() raises -> Int:
        var library = _Library()
        var callback = library.count
        return Int(Pointer(to=callback).unsafe_bitcast[_Count]()[]())

    def estimate_auto(self, query: LazyFrame) raises -> PlanEstimate:
        """Bound numeric row work; never read values or submit device work."""
        var work = cpu_features(query)
        var plan = lower_rows(query, _capabilities())
        var descriptors = _Descriptors(plan)
        var request = descriptors.request(plan, self._budget, False)
        var library = _Library()
        var memory = List[Int64](length=8, fill=0)
        var error = 0
        var status = _estimate_call(library.estimate, request, memory, error)
        library.check(status, error)
        var metadata = _probe_metadata(library, self._device, request)
        # The request stores raw ABI addresses without Mojo origins.
        # Retain every descriptor buffer through both synchronous FFI calls.
        _ = descriptors^
        _ = request^
        var device = work[0].copy()
        device[8] = Float64(memory[4])
        device[9] = 1
        device[10] = Float64(memory[3])
        device[11] = Float64(memory[1])
        device[12] = Float64(memory[2])
        device[13] = Float64(metadata[4] == 0)
        device[14] = 0
        var family = work[1].copy()
        if plan.reused_columns:
            family += ";metal-column-slots-v1"
        for step in plan.steps:
            if plan.code[step.start * 4] == ROW_RANK_KEY:
                family += ";metal-rank-v1"
                break
        var wide_decimal = False
        for dtype in plan.node_dtypes:
            wide_decimal |= dtype.is_decimal() and dtype.decimal_width() == 128
        for output in plan.outputs:
            wide_decimal |= (
                output.dtype.is_decimal()
                and output.dtype.decimal_width() == 128
            )
        if wide_decimal:
            family += ";metal-decimal128-v1"
        var emulated = False
        for i in range(len(plan.node_dtypes)):
            if plan.node_dtypes[i] == DataType.FLOAT64:
                emulated = True
            if plan.node_dtypes[i].is_float() and (
                plan.code[i * 4] == ROUND
                or plan.code[i * 4] == SQRT
                or plan.code[i * 4] == FLOORDIV
                or plan.code[i * 4] == MOD
            ):
                emulated = True
        for output in plan.outputs:
            emulated |= output.reduction == MEAN or (
                output.reduction == SUM and output.dtype.is_float()
            )
        for step in plan.steps:
            for i in range(step.start, step.start + step.nodes):
                if (
                    plan.code[i * 4] == CAST
                    and plan.node_dtypes[i] == DataType.FLOAT32
                ):
                    var operand = Int(plan.code[i * 4 + 1])
                    if operand >= 0:
                        var source = plan.node_dtypes[step.start + operand]
                        emulated |= (
                            source == DataType.INT64
                            or source == DataType.UINT64
                        )
        for i in range(0, len(plan.code), 4):
            if plan.code[i] == ROW_GROUP_OUTPUT:
                emulated |= plan.code[i + 3] == MEAN or (
                    plan.code[i + 3] == SUM
                    and plan.slot_dtypes[Int(plan.code[i + 1])].is_float()
                )
        if emulated:
            family += ";metal-emulated-float64-v1"
        return PlanEstimate(
            work[0].copy(), device^, Int(memory[2]), family^, "accel"
        )

    def calibration_key(
        self, optimize: Bool, streaming: Bool, batch_size: Int
    ) raises -> CalibrationKey:
        var library = _Library()
        var metadata = _probe_metadata(library, self._device, List[Int64]())
        return _metal_calibration_key(
            library, metadata, optimize, streaming, batch_size
        )

    def select_auto(
        self, query: LazyFrame, optimize: Bool, streaming: Bool, batch_size: Int
    ) -> Tuple[String, String]:
        try:
            var path = getenv("DATAFRAME_ACCEL_COST_TABLE")
            if not path:
                return (
                    "cpu",
                    "auto uses CPU; no analytical calibration configured",
                )
            if CPU_BUILD_ID == "development-source-no-calibration":
                return (
                    "cpu",
                    (
                        "automatic calibration requires an identified"
                        " CPU/provider package"
                    ),
                )
            var facts = self.estimate_auto(query)
            if self._budget >= 0 and facts.peak_bytes > self._budget:
                return (
                    "cpu",
                    "estimated Metal payload exceeds configured memory limit",
                )
            var table: JsonValue
            with open(path, "r") as file:
                table = JsonValue(file.read())
            var fraction = Float64(table.get("memory_fraction").text)
            if not isfinite(fraction) or fraction <= 0 or fraction > 1:
                return ("cpu", "invalid automatic memory headroom fraction")
            var library = _Library()
            var metadata = _probe_metadata(library, self._device, List[Int64]())
            var free = max(
                0,
                Int(metadata[1]) - max(0, Int(metadata[2]) - Int(metadata[5])),
            )
            var budget = Int(Float64(free) * fraction)
            if self._budget >= 0:
                budget = min(budget, self._budget)
            var key = _metal_calibration_key(
                library, metadata, optimize, streaming, batch_size
            )
            var decision = select_cost(
                facts, table, key, budget, external_call["time", Int64](Int(0))
            )
            return (
                decision.engine.copy(),
                decision.reason
                + "; peak_bytes="
                + String(facts.peak_bytes)
                + "; budget_bytes="
                + String(budget),
            )
        except error:
            return (
                "cpu",
                "automatic analytical preflight rejected: " + String(error),
            )

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
        if has_literal_text_parsing(query):
            var executed = self._run(
                fold_literal_text_parsing(query), profiling
            )
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_partition_aggregates(query):
            var executed = self._run(
                lower_partition_aggregates(query), profiling
            )
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_aggregate_compositions(query):
            var executed = self._run(
                lower_aggregate_compositions(query), profiling
            )
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_distinct_counts(query):
            var executed = self._run(lower_distinct_counts(query), profiling)
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_set_operations(query):
            var executed = self._run(lower_set_operations(query), profiling)
            var report = executed[1].with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](
                            length=executed[1].height(),
                            fill=Float64(Int(perf_counter_ns()) - start) / 1e6,
                        )
                    ),
                )
            )
            return (executed[0].copy(), report^)
        if has_dictionary_columns(query):
            # Validate original string semantics before replacing values with
            # ordered codes. Schema execution reads no source rows.
            var schema = schema_frame(query, len(query._nodes) - 1)
            var lowered = lower_dictionary_operations(query, self._budget)
            var staged_query = lowered[0].copy()
            var dictionary = DictionaryCodes(staged_query)
            var staging_bytes = checked_add(
                dictionary.staging_bytes(staged_query), lowered[1]
            )
            var budget = self._budget
            if budget >= 0:
                if staging_bytes > budget:
                    raise Error(
                        "Metal dictionary staging exceeds payload budget"
                    )
                budget -= staging_bytes
            var encoded = encode_query(staged_query, dictionary)
            var runtime = MetalRuntime(self._device, memory_limit_bytes=budget)
            var staged_ns = Int(perf_counter_ns()) - start
            var executed = runtime._run(encoded, profiling)
            if self._budget >= 0:
                if (
                    dictionary.restoration_bytes(executed[0], schema)
                    > self._budget - staging_bytes
                ):
                    raise Error(
                        "Metal dictionary restoration exceeds payload budget"
                    )
            var decode_start = Int(perf_counter_ns())
            var result = restore_frame(executed[0], schema, dictionary)
            var decoded_ns = Int(perf_counter_ns()) - decode_start
            var report = executed[1].copy()
            var elapsed = Float64(Int(perf_counter_ns()) - start) / 1e6
            report = report.with_column(
                Series(
                    "query_wall_ms",
                    Column[Float64](
                        List[Float64](length=report.height(), fill=elapsed)
                    ),
                )
            )
            report = report.with_column(
                Series(
                    "dictionary_staging_bytes",
                    Column[Int64](
                        List[Int64](
                            length=report.height(), fill=Int64(staging_bytes)
                        )
                    ),
                )
            )
            report = report.with_column(
                Series(
                    "dictionary_operation_tables",
                    Column[Int64](
                        List[Int64](
                            length=report.height(), fill=Int64(lowered[2])
                        )
                    ),
                )
            )
            report = report.with_column(
                Series(
                    "dictionary_operation_staging_bytes",
                    Column[Int64](
                        List[Int64](
                            length=report.height(), fill=Int64(lowered[1])
                        )
                    ),
                )
            )
            report = report.with_column(
                Series(
                    "dictionary_encode_ms",
                    Column[Float64](
                        List[Float64](
                            length=report.height(),
                            fill=Float64(staged_ns) / 1e6,
                        )
                    ),
                )
            )
            report = report.with_column(
                Series(
                    "dictionary_decode_ms",
                    Column[Float64](
                        List[Float64](
                            length=report.height(),
                            fill=Float64(decoded_ns) / 1e6,
                        )
                    ),
                )
            )
            return (result^, report^)
        if has_join(query) or has_scalar_consumers(query):
            var dag = lower_join_dag(query, _capabilities())
            return _execute_join_dag(
                dag, self._device, self._budget, profiling, start
            )
        var plan = lower_rows(query, _capabilities())
        return _execute(plan, self._device, self._budget, profiling, start)

    def describe(self, query: LazyFrame) -> String:
        try:
            if has_literal_text_parsing(query):
                return "Metal literal text parsing; " + self.describe(
                    fold_literal_text_parsing(query)
                )
            if has_partition_aggregates(query):
                return "Metal resident partition aggregates; " + self.describe(
                    lower_partition_aggregates(query)
                )
            if has_aggregate_compositions(query):
                return "Metal resident aggregate expressions; " + self.describe(
                    lower_aggregate_compositions(query)
                )
            if has_distinct_counts(query):
                return "Metal resident distinct counts; " + self.describe(
                    lower_distinct_counts(query)
                )
            if has_set_operations(query):
                return "Metal resident set operations; " + self.describe(
                    lower_set_operations(query)
                )
            if has_dictionary_columns(query):
                _ = schema_frame(query, len(query._nodes) - 1)
                var metadata = query.copy()
                for i in range(len(metadata._frames)):
                    metadata._frames[i] = metadata._frames[i].slice(0, 0)
                var lowered = lower_dictionary_operations(metadata)
                var staged_query = lowered[0].copy()
                var dictionary = DictionaryCodes(staged_query)
                var encoded = encode_query(staged_query, dictionary)
                return "Metal ordered dictionary staging; " + self.describe(
                    encoded
                )
            if has_join(query) or has_scalar_consumers(query):
                var dag = lower_join_dag(query, _capabilities())
                return (
                    (
                        "ENGINE accel: Metal native joins; " if dag.joins else "ENGINE accel: Metal resident row regions; "
                    )
                    + String(dag.joins)
                    + " joins, "
                    + String(len(dag.tasks))
                    + " resident regions; only the final result materializes"
                    " CPU columns; optional runtime "
                    + ("present" if metal_installed() else "missing")
                    + "\n"
                )
            var plan = lower_rows(query, _capabilities())
            return (
                "ENGINE accel: Metal native row expressions; "
                + String(len(plan.steps))
                + " steps, "
                + String(plan.slots)
                + " slots; optional runtime "
                + ("present" if metal_installed() else "missing")
                + (
                    "; filtered reductions consume predicates without row compaction" if _direct_filtered_reduction(
                        plan
                    ) else ""
                )
                + "\n"
            )
        except error:
            return "ENGINE accel: " + String(error) + "\n"
