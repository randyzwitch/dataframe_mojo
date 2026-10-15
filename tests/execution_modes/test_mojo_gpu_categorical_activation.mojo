"""Public category activation and exact per-key decoded OVER certificates."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.categorical import decode
from dataframe.string_column import StringColumn, StringCodes
from dataframe.accelerator.dictionaries import has_category_transport
from dataframe.accelerator.code_domains import (
    CategoryOwner,
    CategoryField,
    TextField,
)
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_mojo_gpu_categorical_transport import (
    _dtype,
    _cat,
    _codes,
    _integers,
)
from tests.support.coverage_generators import offset_layout, chunk_layout


def _coded(category: Series) raises -> Series:
    var result = decode(category)
    var codes = List[UInt32]([0, 1, 0])
    result._data[StringColumn]._codes = ArcPointer(
        StringCodes(ArcPointer(codes^), category.dtype())
    )
    return result^


def test_categorical_activation_requires_reachable_logical_or_code_owners() raises:
    var category = _cat(
        "k", _dtype(["dup", "dup", "unused"]), [0, 1, 0], [True, True, True]
    )
    var frame = DataFrame([category.copy()])
    assert_true(has_category_transport(frame.lazy()))
    assert_true(
        has_category_transport(chunk_layout(frame).lazy().select(col("k")))
    )
    assert_true(has_category_transport(DataFrame([_coded(category)]).lazy()))
    var numeric = DataFrame([Series("k", Column[UInt32]([0, 1, 0]))])
    assert_false(has_category_transport(numeric.lazy()))
    assert_false(
        has_category_transport(
            DataFrame([Series("k", Column[String](["a"]))]).lazy()
        )
    )
    # An unrelated categorical scan is not a live source of the final SELECT.
    var unrelated = numeric.lazy()
    unrelated._frames.append(frame.copy())
    var dead = frame.lazy()._nodes[0].copy()
    dead.offset = 1
    unrelated._nodes.append(dead^)
    var selected = numeric.lazy().select(col("k"))._nodes[1].copy()
    selected.left = 0
    unrelated._nodes.append(selected^)
    assert_false(has_category_transport(unrelated))
    # Existing private resident lineage is authoritative; raw UInt32 is not.
    var resident = numeric.lazy()
    var owner = ArcPointer(
        CategoryOwner(category.dtype(), List[Column[UInt32]]())
    )
    resident._nodes[0].text_fields = [
        TextField("k", category.dtype(), CategoryField(owner.copy()))
    ]
    assert_true(has_category_transport(resident))


def test_mojo_gpu_category_activation_preserves_original_public_code_owners() raises:
    var category = _cat(
        "k", _dtype(["dup", "dup", "unused"]), [0, 1, 0], [True, True, True]
    )
    var input = DataFrame([category.copy()])
    var actual = (
        input.lazy()
        .select(col("k"))
        .collect(
            accelerator=MetalRuntime(),
            engine="accel",
            optimize=False,
            streaming=False,
        )
    )
    _codes(actual.column("k"), [0, 1, 0], ["dup", "dup", "unused"])
    assert_true(
        actual.column("k").dtype().dictionary().ptr()
        == category.dtype().dictionary().ptr()
    )
    var coded = _coded(category)
    var strings = (
        DataFrame([coded.copy()])
        .lazy()
        .select(col("k"))
        .collect(
            accelerator=MetalRuntime(),
            engine="accel",
            optimize=False,
            streaming=False,
        )
    )
    assert_equal(strings.column("k").dtype(), DataType.STRING)
    assert_true(Bool(strings.column("k")._dictionary_codes()))
    var retained_codes = strings.column("k")._dictionary_codes()
    _codes(
        retained_codes.take(),
        [0, 1, 0],
        ["dup", "dup", "unused"],
    )
    assert_true(strings.equals(DataFrame([coded.copy()])))
    var numeric = DataFrame([Series("k", Column[UInt32]([0, 1, 0]))])
    var numbers = (
        numeric.lazy()
        .select(col("k"))
        .collect(accelerator=MetalRuntime(), engine="accel", optimize=False)
    )
    assert_equal(numbers.column("k").dtype(), DataType.UINT32)
    assert_true(numbers.equals(numeric))


def test_mojo_gpu_categorical_self_key_over_decodes_only_captured_string_keys() raises:
    var dtype = _dtype(["dup", "dup"])
    var source = DataFrame(
        [
            _cat("k", dtype, [0, 1, 0], [True, True, True]),
            Series("v", Column[Int64]([10, 20, 30])),
        ]
    )
    for layout in range(3):
        var input = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        for gpu in [False, True]:
            print(
                "CHECK self OVER layout",
                layout,
                "GPU" if gpu else "CPU",
                "decoded read",
                flush=True,
            )
            var read_key = input.lazy().select_exprs(
                [col("k"), col("k").count().over("k").alias("n")]
            )
            var read = read_key.collect(
                accelerator=MetalRuntime(),
                engine="accel",
                optimize=False,
                streaming=False,
            ) if gpu else read_key.collect(
                engine="cpu", optimize=False, streaming=False
            )
            _integers(read, "n", [3, 3, 3])
            _codes(read.column("k"), [0, 1, 0], ["dup", "dup"])
            assert_true(
                read.column("k").dtype().dictionary().ptr()
                == dtype.dictionary().ptr()
            )
            print(
                "CHECK self OVER layout",
                layout,
                "GPU" if gpu else "CPU",
                "bare control",
                flush=True,
            )
            var bare_key = input.lazy().select(
                col("v").count().over("k").alias("n")
            )
            var bare = bare_key.collect(
                accelerator=MetalRuntime(),
                engine="accel",
                optimize=False,
                streaming=False,
            ) if gpu else bare_key.collect(
                engine="cpu", optimize=False, streaming=False
            )
            _integers(bare, "n", [2, 1, 2])
    var multiple = DataFrame(
        [
            _cat("k", dtype, [0, 1, 0, 1], [True, True, True, True]),
            _cat("j", dtype, [0, 0, 1, 1], [True, True, True, True]),
        ]
    )
    var query = multiple.lazy().select_exprs(
        [col("k"), col("j"), col("k").count().over(["k", "j"]).alias("n")]
    )
    for gpu in [False, True]:
        print(
            "CHECK multikey OVER",
            "GPU" if gpu else "CPU",
            "only k decoded",
            flush=True,
        )
        var result = query.collect(
            accelerator=MetalRuntime(),
            engine="accel",
            optimize=False,
            streaming=False,
        ) if gpu else query.collect(
            engine="cpu", optimize=False, streaming=False
        )
        _integers(result, "n", [2, 2, 2, 2])
        _codes(result.column("k"), [0, 1, 0, 1], ["dup", "dup"])
        _codes(result.column("j"), [0, 0, 1, 1], ["dup", "dup"])
