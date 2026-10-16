"""Physical-width proof for checked narrow Decimal GROUP operands."""
# contract-group: execution.backends
# contract-group: execution.apple_parity
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.lazy import LazyFrame
from dataframe.accelerator import decimal_group_feed
from dataframe.accelerator.decimal_prefix import prefix_infallible
from dataframe.accelerator.constants import fold_literal_text_parsing
from dataframe.accelerator.dictionaries import DictionaryCodes, encode_query
from dataframe.binding import bind
from dataframe.accelerator.joins import lower_join_dag
from dataframe.accelerator.gpu.runtime import capabilities
from dataframe.metal import MetalRuntime
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source(nullable: Bool = False) raises -> DataFrame:
    return DataFrame(
        [
            Series("g", Column[Int32]([0, 1, 0])),
            Series(
                "price",
                Column[Int64]([100, 200, 300], [True, True, not nullable]),
            ).with_dtype(DataType.decimal(15, 2, 64)),
            Series("one", Column[Int32]([100, 100, 100])).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
            Series("discount", Column[Int32]([10, 20, 30])).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
            Series("tax", Column[Int32]([110, 110, 110])).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
        ]
    )


def _expressions() -> List[Expr]:
    var discount = col("price") * (col("one") - col("discount"))
    return [
        discount.sum().alias("total"),
        (discount * col("tax")).sum().alias("charge"),
        discount.mean().alias("average"),
        col("price").count().alias("count"),
    ]


def _collect(
    query: LazyFrame, gpu: Bool, streaming: Bool, batch: Int
) raises -> DataFrame:
    if gpu:
        return query.collect(
            engine="accel",
            accelerator=MetalRuntime(),
            optimize=False,
            streaming=streaming,
            batch_size=batch,
        )
    return query.collect(
        engine="cpu", optimize=False, streaming=streaming, batch_size=batch
    )


def _literal(frame: DataFrame, nullable: Bool = False) raises:
    assert_equal(frame.height(), 2)
    assert_equal(frame.column("g").int32().to_list(), [0, 1])
    assert_equal(frame.column("total").dtype(), DataType.decimal(38, 4))
    assert_equal(frame.column("charge").dtype(), DataType.decimal(38, 6))
    assert_equal(
        frame.column("total")._decimal_raw(0),
        Int128(9000 if nullable else 30000),
    )
    assert_equal(frame.column("total")._decimal_raw(1), Int128(16000))
    assert_equal(
        frame.column("charge")._decimal_raw(0),
        Int128(990000 if nullable else 3300000),
    )
    assert_equal(frame.column("charge")._decimal_raw(1), Int128(1760000))
    assert_equal(
        frame.column("count").int64().to_list(),
        [Int64(1 if nullable else 2), Int64(1)],
    )
    assert_equal(
        frame.column("average").float64().to_list(),
        [Float64(0.9 if nullable else 1.5), Float64(1.6)],
    )


def _money_and_prefix_certificates() raises:
    print("GROUP metadata: original money literals and signatures")
    var source = _source().with_column(
        Series("discount", Column[Int64]([10, 20, 30])).with_dtype(
            DataType.decimal(15, 2, 64)
        )
    )
    for text in ["1", "-1"]:
        # This is the benchmark money() expression, including its original
        # Arrow Decimal128 dtype. No source or literal dtype is rewritten.
        var money = lit(text).cast(DataType.decimal(15, 2))
        var query = (
            source.lazy()
            .group_by("g", maintain_order=True)
            .agg(
                (col("price") * (money - col("discount"))).sum().alias("total")
            )
        )
        var cpu = query.collect(
            engine="cpu", optimize=False, streaming=True, batch_size=1
        )
        assert_equal(
            cpu.column("total")._decimal_raw(0),
            Int128(30000 if text == "1" else -50000),
        )
        var normalized = fold_literal_text_parsing(query)
        var bound = bind(
            normalized._nodes[len(normalized._nodes) - 1].exprs[0],
            source._columns,
        )
        var literals = 0
        for at in range(len(bound.expr._nodes)):
            if (
                bound.dtypes[at].is_decimal()
                and bound.dtypes[at].decimal_width() == 128
                and bound.expr._nodes[at].left < 0
            ):
                literals += 1
                assert_true(decimal_group_feed.narrow_value(bound, at))
                var forged = bound.copy()
                forged.expr._nodes[at].text2 = (
                    "1" if forged.expr._nodes[at].integer >= 0 else "0"
                )
                assert_true(not decimal_group_feed.narrow_value(forged, at))
                assert_true(not decimal_group_feed.operand(forged)[0])
        assert_equal(literals, 1)
        var dag = lower_join_dag(
            normalized.head(0)._with_accelerator_settings(False, True, 65536),
            capabilities(),
        )
        var observed = 0
        for task in dag.tasks:
            for node in task.query._nodes:
                observed += Int(node.text == decimal_group_feed.MATERIALIZATION)
        assert_equal(observed, 1)
    var left = DataFrame(
        [
            Series("g", Column[Int32]([0, 1])),
            Series("segment", StringColumn(["BUILDING", "BUILDING"])),
        ]
    )
    var right = DataFrame(
        [
            Series("g", Column[Int32]([0, 1])),
            Series("orderdate", Column[Int64]([9000, 9100])).with_dtype(
                DataType.DATE
            ),
            Series("priority", Column[Int32]([0, 1])),
        ]
    )
    var prefix = (
        left.lazy()
        .filter(col("segment") == "BUILDING")
        .join(
            right.lazy().filter(
                col("orderdate") < lit("1995-03-15").str().to_date()
            ),
            on="g",
        )
        .join(source.lazy(), on="g")
    )
    prefix = fold_literal_text_parsing(prefix)
    assert_true(prefix_infallible(prefix, len(prefix._nodes) - 1))
    var grouped = (
        prefix.group_by(["g", "orderdate", "priority"])
        .agg(
            (
                col("price")
                * (lit("1").cast(DataType.decimal(15, 2)) - col("discount"))
            )
            .sum()
            .alias("revenue")
        )
        .sort("revenue", descending=True)
        .head(10)
    )
    var normalized = fold_literal_text_parsing(
        grouped
    )._with_accelerator_settings(False, True, 65536)
    # The public route encodes String fields before numeric DAG lowering.
    # DATE retains its original Int64 logical/physical representation; the
    # fresh String segment is the only carrier normalization needed here.
    print("GROUP metadata: public dictionary staging with Int64 DATE keys")
    var dictionary = DictionaryCodes(normalized)
    var encoded = encode_query(normalized, dictionary)
    var dag = lower_join_dag(encoded, capabilities())
    var observed = 0
    for task in dag.tasks:
        for node in task.query._nodes:
            observed += Int(node.text == decimal_group_feed.MATERIALIZATION)
    assert_equal(observed, 1)
    print(
        "GROUP metadata: original prefix fault and wide physical refusal checks"
    )
    # A fallible value in either original JOIN arm is not made infallible by
    # a successful eventual GPU input-zero observation.
    var poison = (col("g").cast(DataType.INT64) + lit(Int64.MAX)).alias(
        "unused"
    )
    for bad in [
        left.lazy().with_columns(poison).join(right.lazy(), on="g"),
        left.lazy().join(right.lazy().with_columns(poison), on="g"),
    ]:
        assert_true(not prefix_infallible(bad, len(bad._nodes) - 1))
    var controls = source.lazy().with_columns(
        col("g").cut([2.0, 1.0]).alias("unused")
    )
    assert_true(not prefix_infallible(controls, len(controls._nodes) - 1))
    var failed_source = source.lazy()
    failed_source._nodes[0].source_failure = "captured original source failure"
    assert_true(not prefix_infallible(failed_source, 0))
    assert_equal(
        failed_source._nodes[0].source_failure,
        "captured original source failure",
    )
    # Declared p15 cannot bound retagged physical128 source values.
    var wide = DataFrame(
        [
            Series("g", Column[Int32]([0])),
            Series("price", Column[Int128]([Int128.MAX])).with_dtype(
                DataType.decimal(15, 2)
            ),
            Series("discount", Column[Int64]([0])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
        ]
    )
    var wide_query = (
        wide.lazy()
        .group_by("g")
        .agg(
            (
                col("price")
                * (lit("1").cast(DataType.decimal(15, 2)) - col("discount"))
            )
            .sum()
            .alias("total")
        )
    )
    with assert_raises(contains="decimal multiplication overflow"):
        _ = wide_query.collect(engine="cpu", optimize=False, streaming=True)
    with assert_raises(contains="checked decimal"):
        _ = lower_join_dag(
            fold_literal_text_parsing(wide_query)
            .head(0)
            ._with_accelerator_settings(False, True, 65536),
            capabilities(),
        )


# requires: mojo_gpu
def test_decimal_group_feed_certificate_preserves_whole_original_agg() raises:
    _money_and_prefix_certificates()
    var source = _source()
    var query = (
        source.lazy()
        .group_by("g", maintain_order=True)
        .agg(_expressions())
        .head(0)
    )
    for streaming in [False, True]:
        var configured = query._with_accelerator_settings(False, streaming, 17)
        var dag = lower_join_dag(configured, capabilities())
        var observed = 0
        for task in dag.tasks:
            for node in task.query._nodes:
                if node.text == decimal_group_feed.MATERIALIZATION:
                    observed += 1
                    assert_true(task.terminal)
                    assert_equal(len(node.names2), 7)
                    assert_equal(node.names2[2], "g")
                    assert_equal(len(node.exprs), 4)
        assert_equal(observed, 1)
    assert_true(
        decimal_group_feed.infallible_sum(
            DataType.decimal(18, 18, 64), Int(UInt32.MAX)
        )
    )
    assert_true(
        not decimal_group_feed.infallible_sum(
            DataType.decimal(38, 0), Int(UInt32.MAX)
        )
    )
    # An implicit Decimal WHEN common cast changes stored scale and can
    # throw separately. The narrow arithmetic proof must not certify it.
    var mixed = DataFrame(
        [
            Series("g", Column[Int32]([0, 0])),
            Series("pick", Column[Int32]([1, 0])),
            Series("a", Column[Int64]([100, 999])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("b", Column[Int64]([777, 200])).with_dtype(
                DataType.decimal(15, 4, 64)
            ),
            Series("factor", Column[Int64]([1, 1])).with_dtype(
                DataType.decimal(15, 0, 64)
            ),
        ]
    )
    var aligned = (
        when(col("pick") == lit(Int32(1))).then(col("a")).otherwise(col("b"))
    )
    var mixed_query = (
        mixed.lazy()
        .group_by("g")
        .agg((aligned * col("factor")).sum().alias("total"))
    )
    var cpu = mixed_query.collect(
        engine="cpu", optimize=False, streaming=True, batch_size=1
    )
    assert_equal(cpu.column("total").dtype(), DataType.decimal(38, 4))
    assert_equal(cpu.column("total")._decimal_raw(0), Int128(10200))
    with assert_raises(contains="checked decimal"):
        _ = lower_join_dag(
            mixed_query.head(0)._with_accelerator_settings(False, True, 1),
            capabilities(),
        )
    # Unary MIN/NEG and wide precision intermediates are different categories,
    # never covered by a narrow ADD/SUB/MUL proof.
    for expression in [
        (-col("price")).sum(),
        (col("price").cast(DataType.decimal(38, 2)) * col("tax")).sum(),
    ]:
        var unsupported = (
            source.lazy()
            .group_by("g")
            .agg(expression)
            .head(0)
            ._with_accelerator_settings(False, True, 17)
        )
        with assert_raises(contains="checked decimal"):
            _ = lower_join_dag(unsupported, capabilities())


# requires: metal
def test_mojo_gpu_decimal_group_feed_literals_nulls_layouts_and_masks() raises:
    var workers = WorkerSetting(1)
    for nullable in [False, True]:
        var source = _source(nullable)
        for frame in [
            source.copy(),
            offset_layout(source),
            chunk_layout(source),
        ]:
            var query = (
                frame.lazy()
                .group_by("g", maintain_order=True)
                .agg(_expressions())
            )
            for streaming in [False, True]:
                for gpu in [False, True]:
                    _literal(_collect(query, gpu, streaming, 17), nullable)
    var text = (
        _source()
        .with_column(Series("returnflag", StringColumn(["N", "R", "N"])))
        .with_column(Series("status", StringColumn(["O", "F", "O"])))
    )
    var text_query = (
        text.lazy()
        .group_by(["returnflag", "status"], maintain_order=True)
        .agg(_expressions())
    )
    for gpu in [False, True]:
        var result = _collect(text_query, gpu, True, 17)
        assert_equal(result.column("returnflag").string().to_list(), ["N", "R"])
        assert_equal(result.column("status").string().to_list(), ["O", "F"])
        assert_equal(result.column("total")._decimal_raw(0), Int128(30000))
        assert_equal(result.column("total")._decimal_raw(1), Int128(16000))
    # An inactive checked multiply must not produce a precision fault.
    var masked = DataFrame(
        [
            Series("g", Column[Int32]([0, 0])),
            Series("a", Column[Int64]([Int64.MAX, 2])).with_dtype(
                DataType.decimal(18, 0, 64)
            ),
            Series("b", Column[Int64]([Int64.MAX, 3])).with_dtype(
                DataType.decimal(18, 0, 64)
            ),
            Series("pick", Column[Int32]([0, 1])),
        ]
    )
    # Keep the false arm valid while its true-arm operands remain extreme.
    var value = (
        when(col("pick") == lit(Int32(1)))
        .then(col("a") * col("b"))
        .otherwise(lit(Int64(0)).cast(DataType.decimal(18, 0, 64)))
    )
    for gpu in [False, True]:
        var result = _collect(
            masked.lazy().group_by("g").agg(value.sum().alias("total")),
            gpu,
            True,
            1,
        )
        assert_equal(result.column("total")._decimal_raw(0), Int128(6))
    var null_operand = masked.with_column(
        Series("a", Column[Int64]([Int64.MAX, 2], [False, True])).with_dtype(
            DataType.decimal(18, 0, 64)
        )
    )
    for gpu in [False, True]:
        var result = _collect(
            null_operand.lazy()
            .group_by("g")
            .agg((col("a") * col("b")).sum().alias("total")),
            gpu,
            True,
            1,
        )
        assert_equal(result.column("total")._decimal_raw(0), Int128(6))
    _ = workers^


# requires: metal
def test_mojo_gpu_decimal_group_feed_stream_workers_and_precision_faults() raises:
    var rows = 131072
    var groups = List[Int32](length=rows, fill=0)
    var price = List[Int64](length=rows, fill=100)
    var one = List[Int32](length=rows, fill=100)
    var discount = List[Int32](length=rows, fill=10)
    var tax = List[Int32](length=rows, fill=110)
    for row in range(rows):
        groups[row] = Int32(row % 2)
    var source = DataFrame(
        [
            Series("g", Column[Int32](groups^)),
            Series("price", Column[Int64](price^)).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("one", Column[Int32](one^)).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
            Series("discount", Column[Int32](discount^)).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
            Series("tax", Column[Int32](tax^)).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
        ]
    )
    for count in [1, 2, 8]:
        var workers = WorkerSetting(count)
        for streaming in [False, True]:
            for batch in [17, 8193]:
                var query = (
                    source.lazy()
                    .group_by("g", maintain_order=True)
                    .agg(_expressions())
                )
                for gpu in [False, True]:
                    var result = _collect(query, gpu, streaming, batch)
                    assert_equal(result.column("g").int32().to_list(), [0, 1])
                    assert_equal(
                        result.column("total")._decimal_raw(0),
                        Int128(589824000),
                    )
                    assert_equal(
                        result.column("total")._decimal_raw(1),
                        Int128(589824000),
                    )
                    assert_equal(
                        result.column("charge")._decimal_raw(0),
                        Int128(64880640000),
                    )
                    assert_equal(
                        result.column("count").int64().to_list(),
                        [Int64(65536), Int64(65536)],
                    )
        _ = workers^
    # Distinct scales and multiple original arithmetic occurrences all map
    # to precision, including physical extrema retagged beyond dtype digits.
    var bad = DataFrame(
        [
            Series("g", Column[Int32]([0, 1, 0])),
            Series("a", Column[Int64]([Int64.MAX, Int64.MIN, 1])).with_dtype(
                DataType.decimal(18, 0, 64)
            ),
            Series("b", Column[Int64]([Int64.MAX, 1, 1])).with_dtype(
                DataType.decimal(18, 0, 64)
            ),
            Series("c", Column[Int64]([Int64.MAX, Int64.MIN, 1])).with_dtype(
                DataType.decimal(18, 18, 64)
            ),
        ]
    )
    var expressions: List[Expr] = [
        (col("a") * (col("b") - col("a"))).sum().alias("total"),
        (col("a") + col("c")).sum().alias("scaled"),
        (col("a") * col("b")).sum().alias("product"),
    ]
    var empty = DataFrame([Series("g", Column[Int32](List[Int32]()))])
    # Rotate the original sibling order so ADD/SUB/MUL are each the CPU's
    # first throwing projection, including different precision/scale metadata.
    for first in range(len(expressions)):
        var ordered = List[Expr]()
        for index in range(len(expressions)):
            ordered.append(
                expressions[(first + index) % len(expressions)].copy()
            )
        var producer = bad.lazy().group_by("g").agg(ordered)
        for streaming in [False, True]:
            for gpu in [False, True]:
                with assert_raises(contains="decimal precision"):
                    _ = _collect(producer, gpu, streaming, 17)
    var producer = bad.lazy().group_by("g").agg(expressions)
    for count in [1, 2, 8]:
        var workers = WorkerSetting(count)
        for streaming in [False, True]:
            for batch in [1, 8193]:
                for query in [
                    producer.copy(),
                    producer.head(0),
                    producer.join(empty.lazy(), on="g", how="inner"),
                ]:
                    for gpu in [False, True]:
                        with assert_raises(contains="decimal precision"):
                            _ = _collect(query, gpu, streaming, batch)
        _ = workers^


# requires: metal
def test_mojo_gpu_decimal_group_feed_budget_precedes_device() raises:
    var query = (
        _source()
        .lazy()
        .group_by("g", maintain_order=True)
        .agg(_expressions())
        .head(1)
    )
    var measured = query.profile(
        engine="accel",
        accelerator=MetalRuntime(),
        optimize=False,
        streaming=True,
        batch_size=17,
    )
    var peak_field = (
        "query_peak_requested_bytes" if "query_peak_requested_bytes"
        in measured[1].columns() else "peak_requested_bytes"
    )
    var peak = Int(measured[1].item(0, peak_field).int64())
    var result = query.collect(
        engine="accel",
        accelerator=MetalRuntime(memory_limit_bytes=peak),
        optimize=False,
        streaming=True,
        batch_size=17,
    )
    assert_equal(result.column("total")._decimal_raw(0), Int128(30000))
    with assert_raises(contains="budget"):
        _ = query.collect(
            engine="accel",
            accelerator=MetalRuntime(device_id=999999, memory_limit_bytes=0),
            optimize=False,
            streaming=True,
            batch_size=17,
        )

    from dataframe.accelerator.rows import lower_rows
    from dataframe.accelerator.gpu.runtime import execute_plan
    from dataframe.accelerator.gpu.residency import Scope, HostMatrix, Table

    var source = _source()
    var producer = (
        source.lazy()
        .group_by("g")
        .agg(_expressions())
        .head(0)
        ._with_accelerator_settings(False, True, 17)
    )
    var dag = lower_join_dag(producer, capabilities())
    for task in dag.tasks:
        if (
            task.query._nodes[len(task.query._nodes) - 1].text
            != decimal_group_feed.MATERIALIZATION
        ):
            continue
        var plan = lower_rows(
            task.query, capabilities(), terminal_boundary=True
        )
        var scope = Scope()
        var cells = source.height() * plan.source.width()
        var errors = List[UInt32](length=cells, fill=0)
        errors[cells - 1] = 32
        var foreign = Table.identity(
            scope,
            HostMatrix(
                plan.source.clear(),
                source.height(),
                List[UInt64](length=2 * cells, fill=0),
                List[UInt8](length=cells, fill=1),
                errors^,
            ),
        )
        with assert_raises(contains="already observed successful inputs"):
            _ = execute_plan(
                plan,
                scope,
                resident_input=foreign.copy(),
                terminal_boundary=True,
            )


# requires: metal
def test_mojo_gpu_decimal_group_feed_coded_keys_publish_plain_strings() raises:
    from std.memory import ArcPointer
    from std.testing import assert_false
    from dataframe.categorical import decode
    from dataframe.dtype import CategoricalDictionary
    from dataframe.string_column import StringCodes

    var dictionary = CategoricalDictionary()
    for label in ["dup", "dup", "unused"]:
        dictionary.append(label)
    var dtype = DataType.categorical(dictionary^)
    var codes: List[UInt32] = [0, 1, 0, 0]
    var strings = decode(
        Series(
            "k", Column[UInt32](codes.copy(), [True, True, True, False])
        ).with_dtype(dtype)
    )
    strings._data[StringColumn]._codes = ArcPointer(
        StringCodes(ArcPointer(codes^), dtype)
    )
    var raw = DataFrame(
        [
            strings^,
            Series("a", Column[Int32]([1, 2, 4, 8])).with_dtype(
                DataType.decimal(9, 0, 32)
            ),
            Series("b", Column[Int32]([10, 10, 10, 10])).with_dtype(
                DataType.decimal(9, 0, 32)
            ),
        ]
    )
    for layout in range(3):
        var source = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        var query = (
            source.lazy()
            .group_by("k", maintain_order=True)
            .agg((col("a") * col("b")).sum().alias("total"))
        )
        for gpu in [False, True]:
            for streaming in [False, True]:
                var grouped = query.collect(
                    accelerator=MetalRuntime(),
                    optimize=False,
                    streaming=streaming,
                ) if gpu else query.collect(
                    engine="cpu", optimize=False, streaming=streaming
                )
                # Independent original-code partitions, including nulls and
                # two distinct codes sharing one decoded label.
                assert_equal(grouped.height(), 3)
                assert_equal(grouped.item(0, "k").string(), "dup")
                assert_equal(grouped.item(1, "k").string(), "dup")
                assert_true(grouped.item(2, "k").is_null())
                var expected: List[Int128] = [50, 20, 80]
                for row in range(3):
                    assert_equal(
                        grouped.column("total")._decimal_raw(row), expected[row]
                    )
                assert_equal(grouped.column("k").dtype(), DataType.STRING)
                assert_false(Bool(grouped.column("k")._dictionary_codes()))
                assert_false(Bool(grouped.column("k")._field_metadata))
                var next_query = query.group_by("k", maintain_order=True).agg(
                    col("total").sum()
                )
                var regrouped = next_query.collect(
                    accelerator=MetalRuntime(),
                    optimize=False,
                    streaming=streaming,
                ) if gpu else next_query.collect(
                    engine="cpu", optimize=False, streaming=streaming
                )
                assert_equal(regrouped.height(), 2)
                assert_equal(
                    regrouped.column("total")._decimal_raw(0), Int128(70)
                )
                assert_equal(
                    regrouped.column("total")._decimal_raw(1), Int128(80)
                )
    with assert_raises(contains="budget"):
        _ = (
            raw.lazy()
            .group_by("k")
            .agg((col("a") * col("b")).sum())
            .collect(
                accelerator=MetalRuntime(
                    device_id=999999, memory_limit_bytes=0
                ),
                streaming=True,
            )
        )
    var bad = raw.with_column(
        Series("a", Column[Int32]([Int32.MAX, 2, 4, 8])).with_dtype(
            DataType.decimal(9, 0, 32)
        )
    ).with_column(
        Series("b", Column[Int32]([Int32.MAX, 10, 10, 10])).with_dtype(
            DataType.decimal(9, 0, 32)
        )
    )
    for gpu in [False, True]:
        with assert_raises(contains="precision"):
            var failed = (
                bad.lazy()
                .group_by("k")
                .agg((col("a") * col("b")).sum())
                .head(0)
            )
            _ = failed.collect(
                accelerator=MetalRuntime()
            ) if gpu else failed.collect(engine="cpu")
