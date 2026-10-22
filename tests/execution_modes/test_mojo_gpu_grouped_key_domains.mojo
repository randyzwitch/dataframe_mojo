"""Public original-domain keys; grouping and List production remain on GPU."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
)
from dataframe.dtype import CategoricalDictionary
from dataframe_metal.metal import MetalRuntime
from dataframe.key_domains import (
    KeyDomain,
    STRING,
    GEOMETRY,
    coded_string_dtype,
)
from dataframe_metal.accelerator.gpu.calibration import estimate
from dataframe_metal.accelerator.gpu.runtime import _preflight, execute_plan
from dataframe_metal.accelerator.gpu.residency import Scope
from dataframe_metal.accelerator.gpu.grouped_schedule import (
    build as build_schedule,
)
from dataframe.plan_types import WindowKey
from dataframe.partition import _hash_column
from tests.public_contracts.test_grouped_key_domains import _key, _coded
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.lifecycle_jobs import WorkerSetting
from dataframe.parallel import configured_workers


def _expressions() -> List[Expr]:
    return [
        col("x").implode().alias("items"),
        col("x").sum().alias("sum"),
        col("k").count().alias("key_count"),
        col("k").null_count().alias("key_nulls"),
    ]


def _compare(frame: DataFrame) raises:
    var query = (
        frame.lazy().group_by("k", maintain_order=True).agg(_expressions())
    )
    var expected = query.collect(engine="cpu", optimize=False, streaming=False)
    var actual = query.collect(
        accelerator=MetalRuntime(), engine="accel", optimize=False
    )
    assert_true(actual.equals(expected))
    assert_equal(expected.column("sum").dtype(), DataType.INT32)
    assert_equal(actual.column("sum").dtype(), DataType.INT32)
    var lengths = actual.column("items").list_column()._offsets[].copy()
    assert_equal(lengths, List[Int64]([Int64(0), Int64(2), Int64(4), Int64(5)]))
    assert_equal(
        actual.column("items").list_column().child().int32().to_list(),
        List[Int32]([Int32(10), Int32(30), Int32(20), Int32(40), Int32(50)]),
    )
    assert_equal(
        actual.column("sum").int32().to_list(),
        List[Int32]([Int32(40), Int32(60), Int32(50)]),
    )
    assert_true(actual.item(2, "k").is_null())
    var cpu_key = expected.column("k")
    var gpu_key = actual.column("k")
    var input_key = frame.column("k")
    # GroupBy._coded_keys replaces coded Strings with original categorical
    # codes, then categorical.decode creates a fresh String without metadata.
    # Plain String/CAT/Binary/Geometry keys retain the take() source owner.
    if coded_string_dtype(input_key):
        assert_false(Bool(cpu_key._field_metadata))
    else:
        assert_true(Bool(cpu_key._field_metadata))
        assert_true(
            cpu_key._field_metadata.value().ptr()
            == input_key._field_metadata.value().ptr()
        )
    assert_equal(Bool(gpu_key._field_metadata), Bool(cpu_key._field_metadata))
    if cpu_key._field_metadata:
        assert_true(
            gpu_key._field_metadata.value().ptr()
            == cpu_key._field_metadata.value().ptr()
        )
    assert_true(
        actual.column("items")
        .list_column()
        .child()
        ._field_metadata.value()
        .ptr()
        == frame.column("x")._field_metadata.value().ptr()
    )
    if frame.column("k").dtype().is_categorical():
        assert_true(
            actual.column("k").dtype().dictionary().ptr()
            == frame.column("k").dtype().dictionary().ptr()
        )
        assert_equal(actual.column("k").uint32().value(0), UInt32(0))
        assert_equal(actual.column("k").uint32().value(1), UInt32(1))
    var consumed = query.select_exprs(
        [
            col("k").alias("key"),
            col("items").list().len().alias("n"),
            col("items").list().get(-1).alias("last"),
        ]
    ).head(2)
    var profiled = consumed.profile(
        accelerator=MetalRuntime(), engine="accel", optimize=False
    )
    assert_true(
        profiled[0].equals(
            consumed.collect(engine="cpu", optimize=False, streaming=False)
        )
    )
    assert_equal(profiled[1].item(0, "resident_list_exports").int64(), Int64(0))
    assert_equal(
        profiled[1].item(0, "list_descriptor_upload_bytes").int64(), Int64(0)
    )


def _check_multiple_lookup_bases() raises:
    # Three original domains occupy distinct nonzero lookup bases. The
    # categorical dictionary deliberately contains duplicate labels; null
    # poison and invalid UTF8 remain distinct from valid empty bytes.
    var dictionary = CategoricalDictionary()
    dictionary.append("same")
    dictionary.append("same")
    var bytes: List[List[UInt8]] = [[255], [0], [], []]
    var original = DataFrame(
        [
            Series(
                "s",
                StringColumn(["a", "é", "a", ""], [True, True, True, False]),
            ),
            Series(
                "c",
                Column[UInt32](
                    [0, 1, 1, UInt32.MAX], [True, True, True, False]
                ),
            ).with_dtype(DataType.categorical(dictionary^)),
            Series.binary("b", bytes^, List[Bool]([True, True, False, False])),
        ]
    )
    var literal: List[UInt64] = [
        0x9E7282CB50DCBF38,
        0x04D29BFCCF7BE747,
        0x4305D69F7C619070,
        0xDFFC52222FE74448,
    ]
    var cpu = List[UInt64](length=4, fill=0)
    for index in range(original.width()):
        var field = original._columns[index].rechunk()
        _hash_column(field, 0, 4, Int(cpu.unsafe_ptr()), index == 0)
    assert_equal(cpu, literal)
    var domains = List[KeyDomain]()
    var columns = List[Series]()
    var provenance = List[WindowKey]()
    for field in original._columns:
        var domain = KeyDomain(field)
        columns.append(domain.encoded.copy())
        provenance.append(domain.provenance.copy())
        domains.append(domain^)
    var frame = DataFrame(columns^)
    var plan = _preflight(frame.lazy().select(["s", "c", "b"]))
    var scope = Scope()
    var source = execute_plan(plan, scope, export_values=False)
    var schedule = build_schedule(
        source[0], ["s", "c", "b"], scope, 4, provenance=provenance
    )
    var actual = List[UInt64](length=4, fill=0)
    var ctx = scope.context()
    try:
        ctx.enqueue_copy(actual.unsafe_ptr(), schedule[].hashes)
        scope.synchronize()
    except error:
        scope.synchronize()
        _ = actual^
        _ = schedule^
        _ = source^
        _ = provenance^
        _ = domains^
        raise error^
    assert_equal(actual, literal)
    _ = actual^
    _ = schedule^
    _ = source^
    _ = provenance^
    _ = domains^


# requires: metal
def test_mojo_gpu_grouped_original_key_domains_duplicates_bytes_metadata_layouts() raises:
    _check_multiple_lookup_bases()
    var dictionary = CategoricalDictionary()
    dictionary.append("same")
    dictionary.append("same")
    dictionary.append("unused")
    var cat = Series(
        "k",
        Column[UInt32](
            [0, 1, 0, 1, UInt32.MAX], [True, True, True, True, False]
        ),
    ).with_dtype(DataType.categorical(dictionary^))
    var keys = List[Series]([cat.copy(), _coded(cat)])
    var bytes: List[List[UInt8]] = [[255], [97, 0, 98], [255], [97, 0, 98], []]
    keys.append(
        Series.binary("k", bytes^, List[Bool]([True, True, True, True, False]))
    )
    keys.append(
        Series(
            "k",
            StringColumn(
                ["é", "a\x00b", "é", "a\x00b", ""],
                [True, True, True, True, False],
            ),
        )
    )
    keys.append(
        _key(
            GEOMETRY,
            List[Int64]([Int64(0), Int64(4), Int64(0), Int64(4), Int64(0)]),
            List[Bool]([True, True, True, True, False]),
        )
    )
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    var owner = ArcPointer(metadata^)
    for workers in List[Int]([1, 2, 4, 8]):
        var setting = WorkerSetting(workers)
        assert_equal(configured_workers(), workers)
        for source in keys:
            var key = source.copy()
            key._field_metadata = owner.copy()
            var x = Series("x", Column[Int32]([10, 20, 30, 40, 50]))
            x._field_metadata = owner.copy()
            var frame = DataFrame([key^, x^])
            for layout in range(3):
                var input = (
                    frame.copy() if layout
                    == 0 else offset_layout(frame) if layout
                    == 1 else chunk_layout(frame)
                )
                _compare(input)
        _ = setting^
    # Plain text with duplicate labels groups by raw text; duplicate original
    # categorical codes above remain two distinct groups even with equal labels.
    var same = DataFrame(
        [
            Series(
                "k",
                StringColumn(
                    ["same", "same", "same", "same", ""],
                    [True, True, True, True, False],
                ),
            ),
            Series("x", Column[Int32]([10, 20, 30, 40, 50])),
        ]
    )
    var plain = same.lazy().group_by("k").agg(col("x").implode().alias("items"))
    var result = plain.collect(accelerator=MetalRuntime(), engine="accel")
    assert_true(result.equals(plain.collect(engine="cpu", streaming=False)))
    assert_equal(result.height(), 2)
    assert_equal(
        result.column("items").list_column()._offsets[],
        List[Int64]([Int64(0), Int64(4), Int64(5)]),
    )


# requires: metal
def test_mojo_gpu_grouped_original_key_hash_worker_bucket_heavy_fault_priority() raises:
    var n = 262145
    var hot = 2 * n // 5
    var values = List[Int64]()
    for row in range(n):
        values.append(
            Int64(4) if row
            < hot else Int64(0) if row
            == hot else Int64(row + 32)
        )
    var valid = List[Bool](length=n, fill=True)
    var integers = List[Int64](length=n, fill=0)
    integers[hot] = 128
    var exponents = List[Int64](length=n, fill=0)
    exponents[0] = -1
    for domain in range(STRING, GEOMETRY + 1):
        var key = _key(domain, values, valid)
        var frame = DataFrame(
            [
                key^,
                Series("u", Column[Int64](integers.copy())),
                Series("base", Column[Int64](List[Int64](length=n, fill=2))),
                Series("exp", Column[Int64](exponents.copy())),
            ]
        )
        var query = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("u").cast(DataType.INT8).implode().alias("first"),
                    col("base").pow(col("exp")).implode().alias("later"),
                ]
            )
        )
        for workers in List[Int]([1, 2, 4, 8]):
            var setting = WorkerSetting(workers)
            assert_equal(configured_workers(), workers)
            var power_first = workers == 8 or (
                domain == GEOMETRY and workers > 1
            )
            with assert_raises(
                contains="nonnegative exponent" if power_first else "cast from int64 to int8 failed"
            ):
                _ = query.collect(engine="cpu", optimize=False, streaming=False)
            with assert_raises(
                contains="nonnegative exponent" if power_first else "strict cast failed"
            ):
                _ = query.head(0).collect(
                    accelerator=MetalRuntime(), engine="accel", optimize=False
                )
            _ = setting^
    # Recovery after every domain's failed private target must remain usable.
    var safe = (
        DataFrame(
            [
                Series("k", StringColumn(["b", "a", "b"])),
                Series("x", Column[Int32]([1, 2, 3])),
            ]
        )
        .lazy()
        .group_by("k", maintain_order=True)
        .agg(col("x").implode().alias("items"))
    )
    assert_true(
        safe.collect(accelerator=MetalRuntime(), engine="accel").equals(
            safe.collect(engine="cpu", streaming=False)
        )
    )


# requires: metal
def test_mojo_gpu_grouped_key_staging_budget_empty_checked_feeds_and_metadata_admission() raises:
    var frame = DataFrame(
        [
            Series("k", StringColumn(["a\x00b", "é", "a\x00b", ""])),
            Series("x", Column[Int32]([1, 2, 3, 4])),
        ]
    )
    var query = (
        frame.lazy().group_by("k", maintain_order=True).agg(_expressions())
    )
    var profiled = query.profile(accelerator=MetalRuntime(), engine="accel")
    var peak = Int(profiled[1].item(0, "query_peak_requested_bytes").int64())
    assert_true(profiled[1].item(0, "grouped_key_staging_bytes").int64() > 0)
    assert_true(
        profiled[0].equals(
            query.collect(
                accelerator=MetalRuntime(memory_limit_bytes=peak),
                engine="accel",
            )
        )
    )
    with assert_raises(contains="memory budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    var empty = frame.clear().lazy().group_by("k").agg(_expressions())
    var zero = empty.profile(
        accelerator=MetalRuntime(999999, memory_limit_bytes=0), engine="accel"
    )
    assert_equal(zero[0].height(), 0)
    assert_equal(zero[0].column("k").dtype(), DataType.STRING)
    assert_equal(
        zero[1].item(0, "query_peak_requested_bytes").int64(), Int64(0)
    )
    assert_equal(zero[1].item(0, "kernel_launches").int64(), Int64(0))
    assert_true(
        MetalRuntime(999999).describe(query).find("resident grouped") >= 0
    )
    with assert_raises(contains="no analytical calibration"):
        _ = estimate(query)
    # Original text operands never acquire numeric eligibility by key recoding.
    var unsupported = (
        frame.lazy().group_by("k").agg(col("k").implode().alias("bad"))
    )
    with assert_raises(contains="transported text keys"):
        _ = unsupported.collect(
            accelerator=MetalRuntime(999999), engine="accel"
        )
    # Payload occurrence certificates survive only unchanged original numeric
    # ASTs, even though the unrelated key changes its physical carrier.
    var checked = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                col("x").cast(DataType.INT64).sum().alias("answer"),
                col("x").implode().alias("items"),
            ]
        )
    )
    assert_true(
        checked.collect(accelerator=MetalRuntime(), engine="accel").equals(
            checked.collect(engine="cpu", streaming=False)
        )
    )
    var bad = (
        frame.with_column(Series("x", Column[Int64]([128, 0, 0, 0])))
        .lazy()
        .group_by("k")
        .agg(
            [
                col("x").cast(DataType.INT8).sum().alias("answer"),
                col("x").implode().alias("items"),
            ]
        )
    )
    with assert_raises(contains="strict cast failed"):
        _ = bad.collect(accelerator=MetalRuntime(), engine="accel")
    with assert_raises(contains="source provenance"):
        _ = MetalRuntime(999999)._run(bad, False, original_entry=False)
