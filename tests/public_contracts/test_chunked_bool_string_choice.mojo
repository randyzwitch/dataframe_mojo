"""Boolean/String choice retains selected payloads, windows and preparation."""
# contract-group: expr.bool_string_choice
# contract-group: expr.chunk_windows
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.expr import null
from dataframe.dtype import CategoricalDictionary
from dataframe.storage.buffer import _Buffer
from dataframe.storage.string import StringCodes
from dataframe.string_view import StringViewBuilder
from dataframe.kernels.binary import choose
from tests.support.coverage_compare import assert_frame


def _booleans(
    name: String,
    values: List[Bool],
    valid: List[Bool],
    first: Int = 13,
    alternate: Bool = False,
) raises -> Series:
    var raw = List[Bool](length=first, fill=False)
    raw.extend(values.copy())
    var flags = List[Bool](length=first, fill=True)
    flags.extend(valid.copy())
    var parent = Series(name, BoolColumn(raw.copy(), flags.copy()))
    var window = parent.slice(first, len(values))
    var parts = List[Series]()
    parts.append(window.slice(0, 0))
    var at = 0
    var sizes: List[Int] = [2, 1, 7, len(values)] if alternate else [
        1,
        3,
        5,
        len(values),
    ]
    for size in sizes:
        var count = min(size, len(values) - at)
        parts.append(window.slice(at, count))
        at += count
    return Series._from_chunks(parts)


def _strings(
    name: String, values: List[String], valid: List[Bool], mode: Int = 0
) raises -> Series:
    var raw = List[String](length=13, fill="prefix")
    raw.extend(values.copy())
    var flags = List[Bool](length=13, fill=True)
    flags.extend(valid.copy())
    var offsets = Series(name, StringColumn(raw.copy(), flags.copy()))
    var builder = StringViewBuilder()
    for row in range(len(raw)):
        if flags[row]:
            builder.append(StringSlice(raw[row]))
        else:
            builder.append_null()
    var views = Series(name, StringColumn(builder^.finish()))
    var parts = List[Series]()
    parts.append(offsets.slice(13, 0))
    var at = 0
    var piece = 0
    for size in [1, 2, 3, len(values)]:
        var count = min(size, len(values) - at)
        if mode == 1 or (mode == 2 and piece % 2 == 1):
            parts.append(views.slice(13 + at, count))
        else:
            parts.append(offsets.slice(13 + at, count))
        at += count
        piece += 1
    return Series._from_chunks(parts)


def _check_bool(
    actual: Series, values: List[Bool], valid: List[Bool], raw: Bool = True
) raises:
    assert_frame(
        DataFrame([actual.copy()]),
        DataFrame(
            [Series(actual.name(), BoolColumn(values.copy(), valid.copy()))]
        ),
    )
    if raw:
        var flat = actual.rechunk()
        for row in range(len(values)):
            assert_equal(flat._data[BoolColumn]._get(row), values[row])


def _check_strings(
    actual: Series, values: List[String], valid: List[Bool]
) raises:
    assert_frame(
        DataFrame([actual.copy()]),
        DataFrame(
            [Series(actual.name(), StringColumn(values.copy(), valid.copy()))]
        ),
    )
    var flat = actual.rechunk()
    for row in range(len(values)):
        if not valid[row]:
            assert_equal(flat._data[StringColumn]._byte_length(row), 0)


def test_boolean_choice_all_raw_valid_states_offsets_and_scalar_broadcast() raises:
    var av = List[Bool]()
    var an = List[Bool]()
    var bv = List[Bool]()
    var bn = List[Bool]()
    var selected = List[Bool]()
    var values = List[Bool]()
    var valid = List[Bool]()
    for state in range(32):
        var take = state & 16 != 0
        av.append(state & 1 != 0)
        an.append(state & 2 != 0)
        bv.append(state & 4 != 0)
        bn.append(state & 8 != 0)
        selected.append(take)
        values.append((state & 1 != 0) if take else (state & 4 != 0))
        valid.append((state & 2 != 0) if take else (state & 8 != 0))
    for first in range(8):
        var a = _booleans("a", av, an, first)
        var b = _booleans("b", bv, bn, 7 - first, True)
        var actual = choose(selected, a, b)
        assert_equal(a._chunked.value()[].cache[].state.load(), Int64(0))
        assert_equal(b._chunked.value()[].cache[].state.load(), Int64(0))
        assert_equal(actual.name(), "")
        _check_bool(actual, values, valid)
    var scalar = Series("s", BoolColumn([True], [False]))
    var vector = _booleans("v", [False, True, False], [True, True, True])
    _check_bool(
        choose([True, False, True], scalar, vector),
        [True, True, True],
        [False, True, False],
    )
    _check_bool(
        choose([False, True, False], vector, scalar),
        [True, True, True],
        [False, True, False],
    )


def _owned_string_choice(mode: Int) raises -> Series:
    var a = _strings(
        "a",
        [
            "é",
            "a\0b",
            "line\nnext",
            "long_external_payload_abcdefghijklmnopqrstuvwxyz",
            "null_poison",
            "tail",
            "Z",
        ],
        [True, True, True, True, False, True, True],
        mode,
    )
    var b = _strings(
        "b",
        ["B", "C", "D", "E", "F", "G", "H"],
        [True, True, True, True, True, True, True],
        2 - mode,
    )
    var result = choose([True, False, True, True, True, False, True], a, b)
    assert_equal(a._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(b._chunked.value()[].cache[].state.load(), Int64(0))
    return result^


def test_string_choice_offsets_views_mixed_null_bytes_and_owned_output() raises:
    for mode in range(3):
        var result = _owned_string_choice(mode)
        assert_equal(result.name(), "")
        assert_true(result.dtype() == DataType.STRING)
        _check_strings(
            result,
            [
                "é",
                "C",
                "line\nnext",
                "long_external_payload_abcdefghijklmnopqrstuvwxyz",
                "",
                "G",
                "Z",
            ],
            [True, True, True, True, False, True, True],
        )
    var scalar = Series("s", StringColumn(["a\0b"]))
    var other = _strings("v", ["X", "Y", "Z"], [True, False, True], 1)
    _check_strings(
        choose([True, False, True], scalar, other),
        ["a\0b", "", "a\0b"],
        [True, False, True],
    )
    _check_strings(
        choose([False, True, False], other, scalar),
        ["a\0b", "", "a\0b"],
        [True, False, True],
    )


def test_choice_safe_empty_prefix_and_short_unused_branch_shapes() raises:
    var empty_bool = Series("e", BoolColumn(List[Bool]()))
    var bools = _booleans(
        "b", [True, False, True, False, True], [True, True, True, True, True]
    )
    _check_bool(choose([], bools, empty_bool), [], [])
    _check_bool(
        choose([True, True, True], bools, empty_bool),
        [True, False, True],
        [True, True, True],
    )
    var short_bool = _booleans("s", [False, True], [True, False])
    _check_bool(
        choose([True, True, False, False, False], short_bool, bools),
        [False, True, True, False, True],
        [True, False, True, True, True],
    )
    var empty_string = Series("e", StringColumn(List[String]()))
    var strings = _strings(
        "b", ["A", "B", "C", "D", "E"], [True, True, True, True, True], 2
    )
    _check_strings(choose([], strings, empty_string), [], [])
    _check_strings(
        choose([True, True, True], strings, empty_string),
        ["A", "B", "C"],
        [True, True, True],
    )
    var short_string = _strings("s", ["x", "hidden"], [True, False], 1)
    _check_strings(
        choose([True, True, False, False, False], short_string, strings),
        ["x", "", "C", "D", "E"],
        [True, False, True, True, True],
    )


def _cached_choice(warm_a: Bool, warm_b: Bool, strings: Bool) raises -> Series:
    var a: Series
    var b: Series
    if strings:
        a = _strings("a", ["A", "hidden", "C"], [True, False, True], 2)
        b = _strings("b", ["D", "E", "hidden"], [True, True, False], 1)
    else:
        a = _booleans("a", [True, True, False], [True, False, True])
        b = _booleans("b", [False, False, True], [True, True, False], 13, True)
    if warm_a:
        var flat = a.rechunk()
        _ = flat^
    if warm_b:
        var flat = b.rechunk()
        _ = flat^
    var result = choose([True, False, False], a, b)
    assert_equal(
        a._chunked.value()[].cache[].state.load(), Int64(2 if warm_a else 0)
    )
    assert_equal(
        b._chunked.value()[].cache[].state.load(), Int64(2 if warm_b else 0)
    )
    return result^


def _coded_source() raises -> Series:
    var labels: List[String] = [
        "zero",
        "é",
        "long_external_dictionary_payload",
        "",
    ]
    var dictionary = CategoricalDictionary()
    for label in labels:
        dictionary.append(label)
    var dtype = DataType.categorical(dictionary^)
    var values = List[String]()
    var codes = List[UInt32]()
    for row in range(7):
        values.append(labels[row % 4])
        codes.append(UInt32(row % 4))
    var column = StringColumn(
        values.copy(), [True, True, False, True, True, True, True]
    )
    column._codes = ArcPointer(StringCodes(_Buffer(codes^), dtype))
    var parent = Series("coded", column^)
    return Series._from_chunks(
        [parent.slice(0, 1), parent.slice(1, 2), parent.slice(3, 4)]
    )


def test_choice_each_operand_cache_state_lifetime_and_legal_string_codes() raises:
    for warm_a in [False, True]:
        for warm_b in [False, True]:
            _check_bool(
                _cached_choice(warm_a, warm_b, False),
                [True, False, True],
                [True, True, False],
            )
            _check_strings(
                _cached_choice(warm_a, warm_b, True),
                ["A", "E", ""],
                [True, True, False],
            )
    var coded = _coded_source()
    var result = choose(
        [True, False, True, False, True, False, True],
        coded,
        Series("s", StringColumn(["else"])),
    )
    _check_strings(
        result,
        [
            "zero",
            "else",
            "",
            "else",
            "zero",
            "else",
            "long_external_dictionary_payload",
        ],
        [True, True, False, True, True, True, True],
    )


def test_public_choice_nullable_predicate_otherwise_variants_and_batches() raises:
    for batch in [1, 3, 64]:
        var frame = DataFrame(
            [
                _strings(
                    "a",
                    ["A", "B", "C", "hidden", "E"],
                    [True, True, True, False, True],
                    2,
                ),
                _strings(
                    "b",
                    ["u", "v", "w", "x", "y"],
                    [True, True, True, True, True],
                    1,
                ),
                _booleans(
                    "take",
                    [True, True, False, True, False],
                    [True, False, True, True, True],
                ),
            ]
        )
        var explicit = (
            frame.lazy()
            .select(
                when(col("take")).then(col("a")).otherwise(col("b")).alias("x")
            )
            .collect(engine="cpu", batch_size=batch)
        )
        _check_strings(
            explicit.column("x"),
            ["A", "v", "w", "", "y"],
            [True, True, True, False, True],
        )
        var omitted = (
            frame.lazy()
            .select(when(col("take")).then(col("a")).end().alias("x"))
            .collect(engine="cpu", batch_size=batch)
        )
        _check_strings(
            omitted.column("x"),
            ["A", "", "", "", ""],
            [True, False, False, False, False],
        )
        var typed_null = (
            frame.lazy()
            .select(
                when(col("take"))
                .then(col("a"))
                .otherwise(null(DataType.STRING))
                .alias("x")
            )
            .collect(engine="cpu", batch_size=batch)
        )
        _check_strings(
            typed_null.column("x"),
            ["A", "", "", "", ""],
            [True, False, False, False, False],
        )
        var bool_frame = DataFrame(
            [
                _booleans("a", [True, True, False], [True, False, True]),
                _booleans("take", [True, False, True], [True, True, False]),
            ]
        )
        var booleans = (
            bool_frame.lazy()
            .select(
                when(col("take"))
                .then(col("a"))
                .otherwise(lit(False))
                .alias("x")
            )
            .collect(engine="cpu", batch_size=batch)
        )
        _check_bool(
            booleans.column("x"),
            [True, False, False],
            [True, True, True],
            False,
        )

        var bool_omitted = (
            bool_frame.lazy()
            .select(when(col("take")).then(col("a")).end().alias("x"))
            .collect(engine="cpu", batch_size=batch)
        )
        _check_bool(
            bool_omitted.column("x"),
            [True, False, False],
            [True, False, False],
            False,
        )
        var bool_null = (
            bool_frame.lazy()
            .select(
                when(col("take"))
                .then(col("a"))
                .otherwise(null(DataType.BOOL))
                .alias("x")
            )
            .collect(engine="cpu", batch_size=batch)
        )
        _check_bool(
            bool_null.column("x"),
            [True, False, False],
            [True, False, False],
            False,
        )


def test_choice_nested_masks_binding_errors_and_checked_branch_observation() raises:
    var frame = DataFrame(
        [
            _strings(
                "text", ["7", "bad", "9", "bad"], [True, True, True, True], 2
            ),
            _booleans(
                "outer", [True, False, True, False], [True, True, True, True]
            ),
            _booleans(
                "inner", [True, True, False, True], [True, True, True, True]
            ),
        ]
    )
    var nested = frame.select(
        when(col("outer"))
        .then(
            when(col("inner"))
            .then(col("text").cast(DataType.INT8))
            .otherwise(lit(Int8(2)))
        )
        .otherwise(lit(Int8(0)))
        .alias("x")
    )
    assert_frame(nested, DataFrame([Series("x", Column[Int8]([7, 0, 2, 0]))]))
    var chained = frame.select(
        when(col("outer"))
        .then(lit("first"))
        .when(col("inner"))
        .then(lit("second"))
        .otherwise(lit("last"))
        .alias("x")
    )
    _check_strings(
        chained.column("x"),
        ["first", "second", "first", "second"],
        [True, True, True, True],
    )
    with assert_raises(contains="failed at row 1"):
        _ = frame.select(
            when(col("inner"))
            .then(col("text").cast(DataType.INT8))
            .otherwise(lit(Int8(0)))
        )
    with assert_raises(contains="when requires a bool predicate"):
        _ = frame.select(
            when(col("text")).then(lit("yes")).otherwise(lit("no"))
        )
    with assert_raises(contains="branches require matching dtypes"):
        _ = frame.select(
            when(col("outer")).then(lit(True)).otherwise(lit("no"))
        )
    var numeric = choose(
        [True, False, True],
        Series._from_chunks(
            [
                Series("a", Column[Int64]([7])),
                Series("a", Column[Int64]([8, 9])),
            ]
        ),
        Series("b", Column[Int64]([2])),
    )
    assert_frame(
        DataFrame([numeric.renamed("x")]),
        DataFrame([Series("x", Column[Int64]([7, 2, 9]))]),
    )
