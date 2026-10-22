"""Independent public CPU/GPU witnesses for scalar mask observation batches."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.binding import bind, SCALAR
from dataframe_metal.accelerator import iana_certificate, ordered_rows
from dataframe_metal.accelerator.gpu.runtime import (
    _preflight,
    estimate_resources,
)
from dataframe_metal.accelerator.gpu import iana_scalar_scopes
from dataframe_metal.metal import MetalRuntime
from tests.support.coverage_compare import assert_frame
from tests.tooling.iana_mutation_proofs import (
    cpu_nonfused_kernel_mask_dtype_and_operand,
)


def _frame(
    rows: Int, gap_row: Int, active_row: Int, offset: Int = 0
) raises -> DataFrame:
    var height = rows + offset
    var keep = List[Bool](length=height, fill=False)
    var validity = List[Bool](length=height, fill=False)
    if active_row >= 0:
        keep[active_row + offset] = True
    validity[gap_row + offset] = True
    return DataFrame(
        [
            Series("keep", Column[Bool](keep^)).slice(offset, rows),
            Series(
                "gap",
                Column[Int64](
                    List[Int64](length=height, fill=1710037800000000), validity
                ),
            )
            .with_dtype(DataType.datetime("us"))
            .slice(offset, rows),
            Series(
                "safe",
                Column[Int64](
                    List[Int64](length=height, fill=1710034200000000)
                ),
            )
            .with_dtype(DataType.datetime("us"))
            .slice(offset, rows),
        ]
    )


def _safe() -> Expr:
    return (
        col("safe")
        .dt()
        .replace_time_zone("UTC")
        .dt()
        .convert_time_zone("America/New_York")
    )


def _expression(nested: Bool) -> Expr:
    var predicate = lit(Int64(1)).cast(DataType.BOOL, strict=False)
    if nested:
        # All three nonleaf scalar occurrences require CPU ANY masks. The
        # Float32 operations deliberately do not use the Float64 fusion path.
        predicate = (
            when(predicate)
            .then((lit(Float32(1)) + lit(Float32(1))) > lit(Float32(0)))
            .otherwise(lit(False))
        )
    var branch = (
        when(predicate)
        .then(col("gap"))
        .otherwise(col("safe"))
        .dt()
        .replace_time_zone("America/New_York")
    )
    return when(col("keep")).then(branch).otherwise(_safe()).alias("out")


def _case(
    runtime: MetalRuntime,
    rows: Int,
    gap_row: Int,
    active_row: Int,
    should_raise: Bool,
    nested: Bool = False,
    offset: Int = 0,
) raises:
    var frame = _frame(rows, gap_row, active_row, offset)
    var query = frame.lazy().select(_expression(nested))
    var expected = String("")
    var actual = String("")
    try:
        _ = query.collect(engine="cpu", optimize=False)
    except error:
        expected = String(error)
    try:
        _ = query.collect(accelerator=runtime, optimize=False)
    except error:
        actual = String(error)
    if should_raise:
        assert_true("non-existent" in expected)
        assert_equal(actual, expected)
    else:
        assert_equal(expected, "")
        assert_equal(actual, "")
        assert_frame(
            query.collect(accelerator=runtime, optimize=False),
            query.collect(engine="cpu", optimize=False),
        )
    var recovery = frame.lazy().select(_safe().alias("out"))
    assert_frame(
        recovery.collect(accelerator=runtime, optimize=False),
        recovery.collect(engine="cpu", optimize=False),
    )


def metadata_scalar_occurrence_slots_and_resources() raises:
    var frame = _frame(8194, 8192, 8193)
    var bound = bind(_expression(True), frame._columns)
    # Main's Decimal producer classifier uses the unchanged default policy.
    # Only original mode5 preflight opts in to IANA occurrence admission.
    assert_true(not ordered_rows.eligible(bound))
    assert_true(ordered_rows.eligible(bound, allow_iana=True))
    var certificate = iana_certificate.prepare(bound)
    var nodes = len(certificate.bound.expr._nodes)
    var policies = 2 + 3 * Int(certificate.words[1]) + nodes
    assert_equal(
        len(certificate.words), 2 + 3 * Int(certificate.words[1]) + 3 * nodes
    )
    var slots = 0
    for node in range(nodes):
        var nonleaf = certificate.bound.expr._nodes[node].left >= 0
        var fused = nonleaf and certificate.bound.fusible[node]
        var policy = certificate.words[policies + node]
        assert_equal(policy & 1, UInt64(fused))
        if nonleaf and not fused and certificate.bound.shapes[node] == SCALAR:
            slots += 1
            assert_equal(policy >> 1, UInt64(slots))
        else:
            assert_equal(policy >> 1, UInt64(0))
    assert_true(slots >= 4)
    assert_equal(
        iana_certificate.scalar_count(certificate.words, 0, nodes), slots
    )
    var resources = iana_scalar_scopes.resources(8194, slots)
    assert_equal(resources.workspace_bytes, 8194 + 2 * slots)
    var malformed = certificate.words.copy()
    for node in range(nodes):
        if malformed[policies + node] >> 1:
            malformed[policies + node] = UInt64(4)
            break
    with assert_raises(contains="scalar mask slot"):
        iana_certificate.validate(malformed, 0, nodes)
    var malformed_kernel = certificate.words.copy()
    malformed_kernel[policies + nodes] = UInt64(2)
    with assert_raises(contains="kernel mask policy"):
        iana_certificate.validate(malformed_kernel, 0, nodes)
    cpu_nonfused_kernel_mask_dtype_and_operand()
    var plan = _preflight(
        frame.lazy().select(_expression(True)), ordered_source=True
    )
    assert_equal(plan.ordered_mode, 5)
    var ordinary = _preflight(
        frame.lazy().select(col("keep").cast(DataType.INT64, strict=False)),
        ordered_source=True,
    )
    assert_equal(len(ordinary.iana_mask_bases), 0)
    var facts = estimate_resources(plan)
    assert_true(facts.peak_bytes >= resources.workspace_bytes)
    assert_true(facts.kernel_launches >= 2 * slots + 1)


def gpu_scalar_any_partial_all_false_and_nested() raises:
    var runtime = MetalRuntime()
    _case(runtime, 2, 0, 1, True)
    _case(runtime, 2, 0, -1, False)
    _case(runtime, 2, 0, 1, True, nested=True)
    _case(runtime, 2, 0, -1, False, nested=True)


def gpu_scalar_any_cpu_batch_boundaries() raises:
    var runtime = MetalRuntime()
    # A true in the next batch cannot activate the previous batch's scalar.
    _case(runtime, 8193, 8191, 8192, False)
    # A true in the previous batch cannot activate a one-row ALL-false tail.
    _case(runtime, 8193, 8192, 8191, False, nested=True)
    # The inactive gap at the start of batch two sees a later active row.
    _case(runtime, 8194, 8192, 8193, True, nested=True)


def gpu_scalar_any_nonzero_sliced_replay() raises:
    var runtime = MetalRuntime()
    _case(runtime, 270, 257, 258, True, nested=True, offset=3)


def gpu_scalar_any_budget_and_recovery() raises:
    var frame = _frame(270, 257, -1, 3)
    var query = frame.lazy().select(_expression(True))
    var measured = query.profile(accelerator=MetalRuntime(), optimize=False)
    var peak = Int(measured[1].item(0, "peak_requested_bytes").int64())
    assert_frame(
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), optimize=False
        ),
        measured[0],
    )
    var limited = MetalRuntime(memory_limit_bytes=peak - 1)
    var refused = False
    try:
        _ = query.collect(accelerator=limited, optimize=False)
    except error:
        refused = "memory budget" in String(error)
    assert_true(refused)
    var recovery = (
        DataFrame([Series("x", Column[Int64]([7]))]).lazy().select(col("x"))
    )
    assert_equal(recovery.collect(accelerator=limited).item().int64(), Int64(7))
