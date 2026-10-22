"""Five focused proofs for descriptors, masks, faults and public recovery."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.binding import bind, SCALAR
from dataframe_metal.accelerator import iana_mutations, iana_certificate
from dataframe_metal.accelerator.rows import lower_rows
from dataframe_metal.accelerator.gpu.runtime import capabilities, _preflight
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.metal import MetalRuntime
from tests.support.coverage_compare import assert_frame


def _typed_temporal_literal(value: Int64, dtype: DataType) -> Expr:
    var expression = lit(value)
    expression._nodes[0].dtypes = [Optional(dtype)]
    return expression^


def _frame(valid: Bool = True) raises -> DataFrame:
    return DataFrame(
        [
            Series("t", Column[Int64]([1710037800000000], [valid])).with_dtype(
                DataType.datetime("us")
            )
        ]
    )


def metadata_unique_tables_and_relocation() raises:
    var frame = _frame()
    var expression = (
        col("t")
        .dt()
        .replace_time_zone("America/New_York", non_existent="null")
        .dt()
        .offset_by("1d")
        .dt()
        .truncate("1h")
        .alias("out")
    )
    var bound = bind(expression, frame._columns)
    var capsule = iana_mutations.Capsule()
    for index in range(len(bound.expr._nodes)):
        if iana_mutations.mutation(bound, index):
            _ = capsule.append(bound, index)
    capsule.validate()
    assert_equal(len(capsule.names), 2)
    assert_equal(len(capsule.descriptors), 3)
    var query = frame.lazy().select(expression)
    var plan = lower_rows(query, capabilities(), ordered_source=True)
    assert_equal(plan.ordered_mode, 5)
    var program = Program(plan)
    assert_true(program.iana_mask_bases[0] >= len(plan.literals))
    for index in range(len(plan.literals)):
        if plan.code[4 * index] == iana_mutations.ROW_IANA_MUTATION:
            assert_true(program.code[4 * index + 2] > plan.code[4 * index + 2])


def metadata_deferred_intervals_and_unconditional_policy() raises:
    var frame = _frame(False)
    var source = col("t").dt().replace_time_zone("America/New_York")
    var bound = bind(source.dt().truncate("0h").alias("out"), frame._columns)
    var capsule = iana_mutations.Capsule()
    var base = capsule.append(bound, len(bound.expr._nodes) - 1)
    assert_equal(capsule.words[base + 15], UInt64(1))
    assert_true(capsule.diagnostics[0].byte_length() > 0)
    assert_true(
        "truncate needs one positive interval" in capsule.diagnostics[0]
    )
    # The binder validates syntax before any row masks or provider context.
    for input in [frame.copy(), frame.head(0)]:
        for operation in [
            source.dt().truncate("invalid"),
            source.dt().offset_by("invalid"),
        ]:
            var safe = (
                col("t")
                .dt()
                .replace_time_zone("UTC")
                .dt()
                .convert_time_zone("America/New_York")
            )
            for expression in [
                operation.copy(),
                when(lit(False)).then(operation.copy()).otherwise(safe),
            ]:
                var query = input.lazy().select(expression.alias("out"))
                var cpu_error = String("")
                var gpu_error = String("")
                try:
                    _ = query.collect(engine="cpu", optimize=False)
                except error:
                    cpu_error = String(error)
                try:
                    _ = _preflight(query, ordered_source=True)
                except error:
                    gpu_error = String(error)
                assert_equal(cpu_error, String("expected a number"))
                assert_equal(gpu_error, cpu_error)
    _ = (
        frame.lazy()
        .select(source.dt().truncate("0h").alias("out"))
        .collect(engine="cpu", optimize=False)
    )
    var raised = False
    try:
        var invalid = bind(
            col("t")
            .dt()
            .replace_time_zone("America/New_York", ambiguous="invalid"),
            frame._columns,
        )
        _ = capsule.append(invalid, len(invalid.expr._nodes) - 1)
    except error:
        raised = "ambiguous must be" in String(error)
    assert_true(raised)


def metadata_occurrence_scopes() raises:
    var frame = _frame()
    var expression = (
        when(lit(False))
        .then(col("t").dt().replace_time_zone("America/New_York"))
        .otherwise(
            col("t")
            .dt()
            .replace_time_zone("UTC")
            .dt()
            .convert_time_zone("America/New_York")
        )
    )
    var bound = bind(expression, frame._columns)
    var certificate = iana_certificate.prepare(bound)
    assert_equal(
        certificate.words[0], UInt64(len(certificate.bound.expr._nodes))
    )
    assert_equal(certificate.words[1], UInt64(3))
    assert_equal(len(certificate.ranks), len(certificate.bound.expr._nodes))
    var offset = 2 + 3 * Int(certificate.words[1])
    assert_equal(
        certificate.words[offset + len(certificate.ranks) - 1], UInt64(0)
    )


def cpu_unselected_replace_and_masked_operand() raises:
    var frame = _frame()
    var gap = col("t").dt().replace_time_zone("America/New_York")
    var safe = (
        col("t")
        .dt()
        .replace_time_zone("UTC")
        .dt()
        .convert_time_zone("America/New_York")
    )
    var query = frame.lazy().select(
        when(lit(False)).then(gap).otherwise(safe).alias("out")
    )
    var raised = False
    try:
        _ = query.collect(engine="cpu", optimize=False)
    except error:
        raised = "non-existent" in String(error)
    assert_true(raised)
    var literal_gap = (
        _typed_temporal_literal(1710037800000000, DataType.datetime("us"))
        .dt()
        .replace_time_zone("America/New_York")
    )
    var literal_query = frame.lazy().select(
        when(lit(False)).then(literal_gap).otherwise(safe).alias("out")
    )
    raised = False
    try:
        _ = literal_query.collect(engine="cpu", optimize=False)
    except error:
        raised = "non-existent" in String(error)
    assert_true(raised)
    _ = (
        frame.head(0)
        .lazy()
        .select(when(lit(False)).then(literal_gap).otherwise(safe).alias("out"))
        .collect(engine="cpu", optimize=False)
    )
    var masked = (
        col("t")
        .cast(DataType.INT64)
        .cast(DataType.datetime("us"))
        .dt()
        .replace_time_zone("America/New_York")
    )
    var masked_query = frame.lazy().select(
        when(lit(False)).then(masked).otherwise(safe).alias("out")
    )
    _ = masked_query.collect(engine="cpu", optimize=False)
    _ = (
        _frame(False)
        .lazy()
        .select(when(lit(False)).then(gap).otherwise(safe).alias("out"))
        .collect(engine="cpu", optimize=False)
    )


def gpu_public_fault_replay_and_recovery() raises:
    var runtime = MetalRuntime()
    var frame = _frame()
    var gap = col("t").dt().replace_time_zone("America/New_York")
    var safe = (
        col("t")
        .dt()
        .replace_time_zone("UTC")
        .dt()
        .convert_time_zone("America/New_York")
    )
    var query = frame.lazy().select(
        when(lit(False)).then(gap).otherwise(safe).alias("out")
    )
    var expected = String("")
    try:
        _ = query.collect(engine="cpu", optimize=False)
    except error:
        expected = String(error)
    var actual = String("")
    try:
        _ = query.collect(accelerator=runtime, optimize=False)
    except error:
        actual = String(error)
    assert_equal(actual, expected)
    var literal_gap = (
        _typed_temporal_literal(1710037800000000, DataType.datetime("us"))
        .dt()
        .replace_time_zone("America/New_York")
    )
    var literal_query = frame.lazy().select(
        when(lit(False)).then(literal_gap).otherwise(safe).alias("out")
    )
    expected = ""
    actual = ""
    try:
        _ = literal_query.collect(engine="cpu", optimize=False)
    except error:
        expected = String(error)
    try:
        _ = literal_query.collect(accelerator=runtime, optimize=False)
    except error:
        actual = String(error)
    assert_true(expected.byte_length() > 0)
    assert_equal(actual, expected)
    var empty_query = (
        frame.head(0)
        .lazy()
        .select(when(lit(False)).then(literal_gap).otherwise(safe).alias("out"))
    )
    assert_frame(
        empty_query.collect(accelerator=runtime, optimize=False),
        empty_query.collect(engine="cpu", optimize=False),
    )
    var recovery_frame = DataFrame(
        [
            Series(
                "t", Column[Int64]([1710034200000000, 1730597400000000, -1])
            ).with_dtype(DataType.datetime("us"))
        ]
    )
    var recovery = recovery_frame.lazy().select(
        col("t")
        .dt()
        .replace_time_zone(
            "America/New_York", ambiguous="earliest", non_existent="null"
        )
        .dt()
        .offset_by("1d")
        .dt()
        .truncate("1h")
        .alias("out")
    )
    assert_frame(
        recovery.collect(accelerator=runtime, optimize=False),
        recovery.collect(engine="cpu", optimize=False),
    )
    var masked = (
        col("t")
        .cast(DataType.INT64)
        .cast(DataType.datetime("us"))
        .dt()
        .replace_time_zone("America/New_York")
    )
    var masked_query = frame.lazy().select(
        when(lit(False)).then(masked).otherwise(safe).alias("out")
    )
    assert_frame(
        masked_query.collect(accelerator=runtime, optimize=False),
        masked_query.collect(engine="cpu", optimize=False),
    )
    var deferred = (
        _frame(False)
        .lazy()
        .select(
            col("t")
            .dt()
            .replace_time_zone("America/New_York")
            .dt()
            .truncate("0h")
            .alias("out")
        )
    )
    assert_frame(
        deferred.collect(accelerator=runtime, optimize=False),
        deferred.collect(engine="cpu", optimize=False),
    )

    var malformed = (
        _frame(False)
        .lazy()
        .select(
            col("t")
            .dt()
            .replace_time_zone("America/New_York")
            .dt()
            .truncate("invalid")
            .alias("out")
        )
    )
    actual = ""
    try:
        _ = malformed.collect(accelerator=runtime, optimize=False)
    except error:
        actual = String(error)
    assert_equal(actual, String("expected a number"))

    # A refused allocation must preserve the public runtime's recovery path.
    var measured = recovery.profile(accelerator=runtime)
    var peak = Int(measured[1].item(0, "peak_requested_bytes").int64())
    assert_frame(
        recovery.collect(accelerator=MetalRuntime(memory_limit_bytes=peak)),
        measured[0],
    )
    var limited = MetalRuntime(memory_limit_bytes=peak - 1)
    var refused = False
    try:
        _ = recovery.collect(accelerator=limited)
    except error:
        refused = "memory budget" in String(error)
    assert_true(refused)
    assert_equal(
        DataFrame([Series("x", Column[Int64]([7]))])
        .lazy()
        .select(col("x"))
        .collect(accelerator=limited)
        .item()
        .int64(),
        Int64(7),
    )


def cpu_fused_inactive_predicate_eager_replace() raises:
    """CPU fusion ignores an inactive ancestor mask before nested selection."""
    var frame = DataFrame(
        [
            Series("gap", Column[Int64]([1710037800000000])).with_dtype(
                DataType.datetime("us")
            ),
            Series("safe", Column[Int64]([1710034200000000])).with_dtype(
                DataType.datetime("us")
            ),
            Series("f", Column[Float64]([1.0])),
        ]
    )
    var nested = (
        when((col("f") + lit(Float64(1))) > lit(Float64(0)))
        .then(col("gap"))
        .otherwise(col("safe"))
        .dt()
        .replace_time_zone("America/New_York")
    )
    var safe = (
        col("safe")
        .dt()
        .replace_time_zone("UTC")
        .dt()
        .convert_time_zone("America/New_York")
    )
    var expression = when(lit(False)).then(nested).otherwise(safe).alias("out")
    var bound = bind(expression, frame._columns)
    var fused = 0
    for at in range(len(bound.fusible)):
        fused += Int(bound.fusible[at] and bound.expr._nodes[at].left >= 0)
    assert_true(fused >= 2)
    var raised = False
    try:
        _ = (
            frame.lazy()
            .select(expression)
            .collect(engine="cpu", optimize=False)
        )
    except error:
        raised = "non-existent" in String(error)
    assert_true(raised)


def _fusion_mask_frame[D: DType]() raises -> DataFrame:
    return DataFrame(
        [
            Series("gap", Column[Int64]([1710037800000000])).with_dtype(
                DataType.datetime("us")
            ),
            Series("safe", Column[Int64]([1710034200000000])).with_dtype(
                DataType.datetime("us")
            ),
            Series("f", Column[Scalar[D]]([Scalar[D](1)])),
        ]
    )


def _fusion_mask_expression[D: DType](masked_operand: Bool = False) -> Expr:
    var predicate = (col("f") + lit(Scalar[D](1))) > lit(Scalar[D](0))
    if masked_operand:
        predicate = (
            col("f").cast(DataType.FLOAT64, strict=False) + lit(Float64(1))
        ) > lit(Float64(0))
    var nested = (
        when(predicate)
        .then(col("gap"))
        .otherwise(col("safe"))
        .dt()
        .replace_time_zone("America/New_York")
    )
    var safe = (
        col("safe")
        .dt()
        .replace_time_zone("UTC")
        .dt()
        .convert_time_zone("America/New_York")
    )
    return when(lit(False)).then(nested).otherwise(safe).alias("out")


def _cpu_kernel_mask_case[
    D: DType
](masked_operand: Bool, should_raise: Bool) raises:
    var frame = _fusion_mask_frame[D]()
    var expression = _fusion_mask_expression[D](masked_operand)
    var bound = bind(expression, frame._columns)
    var fused = 0
    for at in range(len(bound.fusible)):
        fused += Int(bound.fusible[at] and bound.expr._nodes[at].left >= 0)
    comptime if D == DType.float32 or D == DType.int32:
        assert_equal(fused, 0)
    var message = String("")
    try:
        _ = (
            frame.lazy()
            .select(expression)
            .collect(engine="cpu", optimize=False)
        )
    except error:
        message = String(error)
    if should_raise:
        assert_true("non-existent" in message)
    else:
        assert_equal(message, "")


def cpu_nonfused_kernel_mask_dtype_and_operand() raises:
    # CPU Float32 binary dispatch ignores its own mask; integer dispatch fits
    # it. A nonfused Float64 parent still receives null from a masked CAST.
    _cpu_kernel_mask_case[DType.float32](False, True)
    _cpu_kernel_mask_case[DType.int32](False, False)
    _cpu_kernel_mask_case[DType.float32](True, False)


def _gpu_fusion_mask_case[
    D: DType
](runtime: MetalRuntime, masked_operand: Bool = False) raises:
    var frame = _fusion_mask_frame[D]()
    var expression = _fusion_mask_expression[D](masked_operand)
    var bound = bind(expression, frame._columns)
    var certificate = iana_certificate.prepare(bound)
    var nodes = len(certificate.bound.expr._nodes)
    var flags = 2 + 3 * Int(certificate.words[1]) + nodes
    var fused = 0
    for at in range(nodes):
        var expected = (
            certificate.bound.fusible[at]
            and certificate.bound.expr._nodes[at].left >= 0
        )
        assert_equal(certificate.words[flags + at] & 1, UInt64(expected))
        fused += Int(expected)
    comptime if D == DType.float64:
        assert_true(fused >= 2)
    else:
        assert_equal(fused, 0)
    var query = frame.lazy().select(expression)
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
    comptime if D == DType.float64 or D == DType.float32:
        if not masked_operand:
            assert_true("non-existent" in expected)
        else:
            assert_equal(expected, "")
    else:
        assert_equal(expected, "")
    assert_equal(actual, expected)
    if not expected:
        assert_frame(
            query.collect(accelerator=runtime, optimize=False),
            query.collect(engine="cpu", optimize=False),
        )
    var recovery = frame.lazy().select(
        col("safe")
        .dt()
        .replace_time_zone("UTC")
        .dt()
        .convert_time_zone("America/New_York")
        .alias("out")
    )
    assert_frame(
        recovery.collect(accelerator=runtime, optimize=False),
        recovery.collect(engine="cpu", optimize=False),
    )


def gpu_fused_inactive_predicate_eager_replace() raises:
    var runtime = MetalRuntime()
    _gpu_fusion_mask_case[DType.float64](runtime)
    _gpu_fusion_mask_case[DType.float32](runtime)
    _gpu_fusion_mask_case[DType.int32](runtime)
    _gpu_fusion_mask_case[DType.float32](runtime, masked_operand=True)


def gpu_iana_replay_selected_row() raises:
    """One-lane replay retains a nonzero logical row from a sliced source."""
    var values = List[Int64](length=270, fill=1710034200000000)
    var valid = List[Bool](length=270, fill=True)
    values[260] = 1710037800000000
    valid[5] = False
    var source = (
        Series("t", Column[Int64](values^, valid^))
        .with_dtype(DataType.datetime("us"))
        .slice(3, 263)
    )
    var frame = DataFrame([source^])
    var query = frame.lazy().select(
        col("t").dt().replace_time_zone("America/New_York").alias("out")
    )
    var expected = String("")
    var actual = String("")
    try:
        _ = query.collect(engine="cpu", optimize=False)
    except error:
        expected = String(error)
    var runtime = MetalRuntime()
    try:
        _ = query.collect(accelerator=runtime, optimize=False)
    except error:
        actual = String(error)
    assert_true("non-existent" in expected)
    assert_equal(actual, expected)
    var recovery = frame.lazy().select(
        col("t")
        .dt()
        .replace_time_zone("America/New_York", non_existent="null")
        .alias("out")
    )
    assert_frame(
        recovery.collect(accelerator=runtime, optimize=False),
        recovery.collect(engine="cpu", optimize=False),
    )


def cpu_scalar_any_inactive_predicate_eager_replace() raises:
    """A scalar cast sees ANY of its parent mask within the CPU batch."""
    for partial in [False, True]:
        var frame = DataFrame(
            [
                Series("keep", Column[Bool]([False, partial])),
                Series(
                    "gap",
                    Column[Int64](
                        [1710037800000000, 1710037800000000], [True, False]
                    ),
                ).with_dtype(DataType.datetime("us")),
                Series(
                    "safe", Column[Int64]([1710034200000000, 1710034200000000])
                ).with_dtype(DataType.datetime("us")),
            ]
        )
        var predicate = lit(Int64(1)).cast(DataType.BOOL, strict=False)
        var predicate_bound = bind(predicate, frame._columns)
        assert_equal(
            predicate_bound.shapes[len(predicate_bound.shapes) - 1], SCALAR
        )
        var nested = (
            when(predicate)
            .then(col("gap"))
            .otherwise(col("safe"))
            .dt()
            .replace_time_zone("America/New_York")
        )
        var safe = (
            col("safe")
            .dt()
            .replace_time_zone("UTC")
            .dt()
            .convert_time_zone("America/New_York")
        )
        var query = frame.lazy().select(
            when(col("keep")).then(nested).otherwise(safe).alias("out")
        )
        var prepared = _preflight(query, ordered_source=True)
        assert_equal(prepared.ordered_mode, 5)
        var message = String("")
        try:
            _ = query.collect(engine="cpu", optimize=False)
        except error:
            message = String(error)
        if partial:
            assert_true("non-existent" in message)
        else:
            assert_equal(message, "")
