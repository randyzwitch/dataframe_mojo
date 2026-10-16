"""Portable accelerator eligibility and metadata are independently observable."""
# contract-group: planner.routing
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, DataType, col, lit
from dataframe.lazy import LazyFrame
from dataframe.categorical import encode
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.dictionaries import DictionaryCodes
from dataframe.accelerator.cpu_work import cpu_features
from dataframe.accelerator.cost import cost_feature_names
from dataframe.accelerator.memory import row_memory


def test_accelerator_dictionary_codes_are_query_wide_ordered_and_keep_empty_source_dictionary_identity() raises:
    var source = Series(
        "k",
        Column[String](
            ["z", "", "é", "a", "poison", "a"],
            [True, True, True, True, False, True],
        ),
    )
    var dictionary = DictionaryCodes(
        DataFrame([source.copy()]).lazy().filter(col("k") != lit("missing"))
    )
    var ordered: List[String] = ["", "a", "missing", "z", "é"]
    for at in range(len(ordered)):
        assert_equal(dictionary.code(ordered[at]), UInt32(at))
    var coded = dictionary.encode(source)
    var expected: List[UInt32] = [3, 0, 4, 1, 0, 1]
    var restored = dictionary.restore(coded, DataType.STRING)
    for row in range(6):
        assert_equal(coded.get(row).is_null(), row == 4)
        assert_equal(restored.get(row).is_null(), row == 4)
        if row != 4:
            assert_equal(coded.numeric[DType.uint32]()._get(row), expected[row])
            assert_equal(restored.get(row).string(), source.get(row).string())
    var first = encode(Series("k", Column[String](["z", "a", "z"])))
    var second = encode(Series("k", Column[String](["a", "z", "other"])))
    var shared = DictionaryCodes(
        DataFrame([first.copy()])
        .lazy()
        .join(DataFrame([second.copy()]).lazy(), "k")
    )
    var left = shared.encode(first)
    var right = shared.encode(second)
    var right_codes: List[UInt32] = [0, 2, 1]
    for row in range(3):
        assert_equal(
            left.numeric[DType.uint32]()._get(row), UInt32(2 if row != 1 else 0)
        )
        assert_equal(right.numeric[DType.uint32]()._get(row), right_codes[row])
    var original = encode(Series("k", Column[String](["unused", "a"])))
    var empty = original.slice(0, 0)
    var empty_codes = DictionaryCodes(DataFrame([empty.copy()]).lazy())
    assert_equal(empty_codes.code("a"), UInt32(0))
    assert_equal(empty_codes.code("unused"), UInt32(1))
    var decoded = empty_codes.restore(
        empty_codes.encode(empty), original.dtype()
    )
    assert_equal(decoded.dtype(), original.dtype())
    assert_equal(len(decoded), 0)
    assert_equal(len(decoded.dtype().dictionary()[]), 2)


def all_capabilities() -> RowCapabilities:
    return RowCapabilities(
        "matrix",
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
    )


def test_accelerator_feature_capabilities_are_independent_opt_in_gates() raises:
    var numeric = DataFrame(
        [
            Series("f", Column[Float32]([1, 2])),
            Series("i", Column[Int32]([1, 2])),
        ]
    )
    var narrow = DataFrame([Series("n", Column[Int8]([1, 2]))])
    var dates = DataFrame(
        [Series("d", Column[Int64]([0, 1])).with_dtype(DataType.DATE)]
    )
    var requests: List[LazyFrame] = [
        narrow.lazy(),
        numeric.lazy(),
        numeric.lazy().select(col("f").cast("int32", strict=False)),
        numeric.lazy().select(col("f").min()),
        numeric.lazy().select(col("f").abs()),
        numeric.lazy().sort("f"),
        numeric.lazy().group_by("i", maintain_order=False).agg(col("f").min()),
        numeric.lazy().select(col("f").first()),
        numeric.lazy().slice(1, 1),
        dates.lazy(),
        DataFrame([Series("w", Column[Int64]([1, 2]))])
        .lazy()
        .select(col("w") + lit(Int64(1))),
        dates.lazy().select(col("d").dt().year()),
    ]
    for at in range(len(requests)):
        var enabled = all_capabilities()
        var lowered = lower_rows(requests[at], enabled)
        assert_true(len(lowered.outputs) > 0)
        var disabled = enabled.copy()
        if at == 0:
            disabled.extended_integers = False
        elif at == 1:
            disabled.mixed_types = False
        elif at == 2:
            disabled.casts = False
        elif at == 3:
            disabled.extrema = False
        elif at == 4:
            disabled.row_extras = False
        elif at == 5:
            disabled.order = False
        elif at == 6:
            disabled.grouped = False
        elif at == 7:
            disabled.extra_reductions = False
        elif at == 8:
            disabled.slices = False
        elif at == 9:
            disabled.fixed_logical = False
        elif at == 10:
            disabled.int64_arithmetic = False
        else:
            disabled.temporal_fields = False
        with assert_raises(contains="matrix unsupported"):
            _ = lower_rows(requests[at], disabled)


def test_accelerator_cpu_work_is_value_blind_and_custom_launch_geometry_covers_each_row() raises:
    for rows in [0, 1, 7, 128, 129, 1025]:
        var values = List[Float32](length=rows, fill=1)
        var source = DataFrame([Series("x", Column[Float32](values^))])
        var query = (
            source.lazy()
            .filter(col("x") > lit(Float32(1)))
            .select((col("x") + lit(Float32(1))).sum())
        )
        var before = query.explain(engine="cpu")
        var facts = cpu_features(query)
        var changed = DataFrame(
            [
                Series(
                    "x",
                    Column[Float32](
                        List[Float32](length=rows, fill=-99),
                        List[Bool](length=rows, fill=False),
                    ),
                )
            ]
        )
        var other = cpu_features(
            changed.lazy()
            .filter(col("x") > lit(Float32(1)))
            .select((col("x") + lit(Float32(1))).sum())
        )
        assert_equal(facts[0], other[0])
        assert_equal(facts[1], "float32")
        var model: List[Float64] = [
            1,
            Float64(rows),
            Float64(rows * 4),
            Float64(rows * 3),
            Float64(rows * 3),
            Float64(rows),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
        ]
        assert_equal(len(facts[0]), len(cost_feature_names()))
        assert_equal(facts[0], model)
        assert_equal(query.explain(engine="cpu"), before)
        var plan = lower_rows(source.lazy().select(col("x")))
        for threads in [1, 7, 128, 256]:
            for maximum in [1, 2, 17]:
                var groups = 0
                for _ in range(0, rows, threads):
                    groups += 1
                var blocks = max(1, min(groups, maximum))
                var chunk = threads
                while chunk * blocks < max(1, rows):
                    chunk += threads
                var observed = row_memory(
                    plan, threads=threads, max_blocks=maximum
                )
                assert_equal(observed.blocks, blocks)
                assert_equal(observed.chunk, chunk)
                assert_true(observed.chunk * observed.blocks >= rows)
        for invalid in [0, -1]:
            with assert_raises(contains="must be positive"):
                _ = row_memory(plan, threads=invalid)
            with assert_raises(contains="must be positive"):
                _ = row_memory(plan, max_blocks=invalid)
