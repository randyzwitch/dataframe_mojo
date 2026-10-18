"""Resident projections share immutable owners, values and deferred failures."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.lazy import LazyFrame
from dataframe.accelerator.gpu.residency import Scope, Table, HostMatrix
from dataframe.accelerator.gpu.runtime import (
    _preflight,
    execute_plan,
    materialize,
)
from dataframe.accelerator.gpu.target import METAL_TARGET
from tests.execution_modes.test_mojo_gpu_residency import _source
from tests.support.coverage_compare import assert_frame


def schema(name: String) raises -> DataFrame:
    return DataFrame([Series(name, Column[Int64](List[Int64]()))])


def test_mojo_gpu_projection_views_host_identity_and_guards() raises:
    var scope = Scope()
    var words: List[UInt64] = [7, 8, 9, 0, 0, 0]
    var validity: List[UInt8] = [1, 1, 1]
    var errors: List[UInt32] = [0, 1, 0]
    var source = Table.identity(
        scope,
        HostMatrix(
            schema("x"),
            3,
            words^,
            validity^,
            errors^,
        ),
    )
    var bytes = scope.info()[0]
    var view = source.project_view(scope, schema("renamed"), [0], 1)
    assert_true(source.same_storage(view))
    assert_equal(scope.info()[0], bytes)
    assert_equal(view.schema.columns()[0], "renamed")
    assert_equal(source.schema.columns()[0], "x")
    assert_equal(view.readback().words[0], UInt64(7))
    view.observe_errors()
    with assert_raises():
        source.project_view(scope, schema("all"), [0], 3).observe_errors()
    with assert_raises(contains="another query"):
        _ = source.project_view(Scope(), schema("x"), [0], 1)
    with assert_raises(contains="outside"):
        _ = source.project_view(scope, schema("x"), [1], 1)
    with assert_raises(contains="shape"):
        _ = source.project_view(scope, schema("x"), [0], 4)
    var wrong = DataFrame([Series("x", Column[Float64](List[Float64]()))])
    with assert_raises(contains="dtype"):
        _ = source.project_view(scope, wrong, [0], 1)
    assert_true(not scope.initialized())


def resident_projection(
    query: LazyFrame, source: Table, scope: Scope, terminal: Bool = True
) raises -> Tuple[Table, DataFrame]:
    comptime if METAL_TARGET:
        var plan = _preflight(
            query, source_rows=source.rows, terminal_boundary=terminal
        )
        return execute_plan(
            plan,
            scope,
            Optional[Table](source.copy()),
            terminal_boundary=terminal,
        )
    else:
        raise Error("This witness requires Apple Metal")


# requires: metal
def test_mojo_gpu_projection_views_execute_without_copy_or_submission() raises:
    var scope = Scope()
    var source = _source(scope)
    var bytes = scope.info()[0]
    var query = (
        source.schema.lazy()
        .select_exprs([col("flag").alias("keep"), col("x").alias("value")])
        .head(2)
    )
    var measured = resident_projection(query, source, scope)
    assert_true(measured[0].same_storage(source))
    assert_equal(scope.info()[0], bytes)
    assert_equal(
        measured[1].item(0, "algorithm").string(),
        "mojo_gpu_resident_projection_view",
    )
    assert_equal(measured[1].item(0, "kernel_launches").int64(), Int64(0))
    assert_equal(measured[1].item(0, "upload_bytes").int64(), Int64(0))
    assert_equal(measured[1].item(0, "synchronizations").int64(), Int64(0))
    var expected = DataFrame(
        [
            Series("keep", Column[Bool]([True, False])),
            Series("value", Column[Int64]([-7, 6])),
        ]
    )
    assert_frame(materialize(measured[0]), expected)


# requires: metal
def test_mojo_gpu_projection_views_keep_nonterminal_error_observation() raises:
    var scope = Scope()
    var source = _source(scope, True)
    var query = source.schema.lazy().select(col("x"))
    with assert_raises():
        _ = resident_projection(query, source, scope, False)
    var harmless = source.schema.lazy().select(col("flag"))
    var measured = resident_projection(harmless, source, scope, False)
    assert_true(measured[0].same_storage(source))
    assert_equal(measured[1].item(0, "kernel_launches").int64(), Int64(0))
    assert_true(measured[1].item(0, "synchronizations").int64() > 0)
