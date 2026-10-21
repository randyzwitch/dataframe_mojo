"""Counted provider adapters and independent expected row literals.

Portable adapters exercise real LazyFrame provider dispatch. They deliberately
execute CPU work inside the adapter and make no native device coverage claim.
"""
from std.memory import Pointer
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.lazy import AcceleratorBackend, LazyFrame


def backend_value(row: Int) -> Int64:
    return Int64(row % 11 - 5)


def backend_valid(row: Int) -> Bool:
    return row % 7 != 2


def backend_source(n: Int, offset: Int = 7) raises -> DataFrame:
    var values = List[Int64]()
    var valid = List[Bool]()
    var ids = List[UInt64]()
    for i in range(n + offset):
        values.append(backend_value(i) if backend_valid(i) else Int64.MAX)
        valid.append(backend_valid(i))
        ids.append(UInt64(i) + (UInt64(1) << 63))
    return DataFrame(
        [
            Series("value", Column[Int64](values^, valid)),
            Series("id", Column[UInt64](ids^)),
        ]
    ).slice(offset, n)


def backend_plan(n: Int, offset: Int = 7) raises -> LazyFrame:
    return (
        backend_source(n, offset)
        .lazy()
        .filter(col("value") >= 0)
        .select_exprs(
            [
                col("id"),
                (col("value") + 3).alias("result"),
            ]
        )
    )


def assert_backend_rows(
    actual: DataFrame, n: Int, offset: Int = 7, limit: Int = -1
) raises:
    var rows = List[Int]()
    for row in range(offset, offset + n):
        if backend_valid(row) and backend_value(row) >= 0:
            if limit < 0 or len(rows) < limit:
                rows.append(row)
    var names: List[String] = ["id", "result"]
    assert_equal(actual.columns(), names)
    assert_equal(actual.height(), len(rows))
    assert_true(actual.column("id").dtype() == DataType.UINT64)
    assert_true(actual.column("result").dtype() == DataType.INT64)
    for i in range(len(rows)):
        assert_equal(
            actual.item(i, "id").uint64(), UInt64(rows[i]) + (UInt64(1) << 63)
        )
        assert_equal(
            actual.item(i, "result").int64(), backend_value(rows[i]) + 3
        )


struct BackendTrace(Movable):
    var selections: Int
    var executes: Int
    var profiled: Int
    var automatic: Int
    var descriptions: Int
    var optimize: Bool
    var streaming: Bool
    var batch: Int
    var profiling: Bool
    var selected: String
    var failure: String

    def __init__(out self, selected: String = "accel", failure: String = ""):
        self.selections = 0
        self.executes = 0
        self.profiled = 0
        self.automatic = 0
        self.descriptions = 0
        self.optimize = True
        self.streaming = True
        self.batch = 65536
        self.profiling = False
        self.selected = selected
        self.failure = failure


def trace_at(address: Int) -> Pointer[BackendTrace, MutAnyOrigin]:
    return Pointer[BackendTrace, MutAnyOrigin](unsafe_from_address=address)


def provider_result(
    plan: LazyFrame, optimize: Bool, streaming: Bool, batch: Int
) raises -> Tuple[DataFrame, DataFrame]:
    var frame = plan.collect(
        engine="cpu", optimize=optimize, streaming=streaming, batch_size=batch
    )
    var report = DataFrame(
        [
            Series("executor", Column[String](["counted-provider"])),
            Series("output_rows", Column[Int64]([Int64(frame.height())])),
        ]
    )
    return (frame^, report^)


struct CountedBackend(AcceleratorBackend):
    var address: Int

    def __init__(out self, address: Int):
        self.address = address

    def select_auto(
        self, plan: LazyFrame, optimize: Bool, streaming: Bool, batch_size: Int
    ) -> Tuple[String, String]:
        ref trace = trace_at(self.address)[]
        trace.selections += 1
        trace.optimize = optimize
        trace.streaming = streaming
        trace.batch = batch_size
        return (trace.selected.copy(), "counted placement " + trace.selected)

    def execute(self, plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        ref trace = trace_at(self.address)[]
        trace.executes += 1
        if trace.failure:
            raise Error(trace.failure)
        return provider_result(plan, True, True, 65536)

    def execute_profiled(
        self, plan: LazyFrame
    ) raises -> Tuple[DataFrame, DataFrame]:
        ref trace = trace_at(self.address)[]
        trace.profiled += 1
        if trace.failure:
            raise Error(trace.failure)
        return provider_result(plan, True, True, 65536)

    def execute_auto(
        self,
        plan: LazyFrame,
        optimize: Bool,
        streaming: Bool,
        batch_size: Int,
        profiling: Bool,
    ) raises -> Tuple[DataFrame, DataFrame]:
        ref trace = trace_at(self.address)[]
        trace.automatic += 1
        trace.optimize = optimize
        trace.streaming = streaming
        trace.batch = batch_size
        trace.profiling = profiling
        if trace.failure:
            raise Error(trace.failure)
        return provider_result(plan, optimize, streaming, batch_size)

    def describe(self, plan: LazyFrame) -> String:
        trace_at(self.address)[].descriptions += 1
        return "ENGINE accel: counted provider description\n"


struct DefaultPolicyBackend(AcceleratorBackend):
    var address: Int

    def __init__(out self, address: Int):
        self.address = address

    def execute(self, plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        trace_at(self.address)[].executes += 1
        return provider_result(plan, True, True, 65536)

    def describe(self, plan: LazyFrame) -> String:
        trace_at(self.address)[].descriptions += 1
        return "ENGINE accel: default policy provider\n"
