"""As-of joins select independently modeled rows and preserve full left schemas."""
from std.collections import Optional
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import AnyValue, Column, DataFrame, DataType, Series
from tests.support.numeric_model import (
    INTEGER_DTYPES,
    FLOAT_DTYPES,
    integer_bounds,
)
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.asof_reference import reference_asof, assert_asof_rows
from tests.support.join_typed_reference import typed_join_values


def _layout(frame: DataFrame, layout: Int) raises -> DataFrame:
    return (
        frame.copy() if layout
        == 0 else offset_layout(frame) if layout
        == 1 else chunk_layout(frame)
    )


def _check(
    left: DataFrame,
    right: DataFrame,
    selected: List[Int],
    strategy: String,
    exact: Bool,
    tolerance: Optional[AnyValue] = None,
) raises:
    var actual = left.join_asof(
        right,
        on="k",
        strategy=strategy,
        allow_exact_matches=exact,
        tolerance=tolerance,
    )
    assert_asof_rows(
        actual, left, right, selected, ["k", "p", "p_right"], ["p"]
    )


def test_asof_reference_has_literal_direction_duplicate_and_tolerance_anchors() raises:
    var left: List[Int64] = [0, 1, 2, 3, 4, 5]
    var right: List[Int64] = [1, 2, 2, 4]
    var lv = List[Bool](length=6, fill=True)
    var rv = List[Bool](length=4, fill=True)
    var lg = List[Int](length=6, fill=0)
    var rg = List[Int](length=4, fill=0)
    assert_equal(
        reference_asof[DType.int64](
            left, right, lv, rv, lg, rg, "backward", True
        ),
        [-1, 0, 2, 2, 3, 3],
    )
    assert_equal(
        reference_asof[DType.int64](
            left, right, lv, rv, lg, rg, "forward", True
        ),
        [0, 0, 1, 3, 3, -1],
    )
    assert_equal(
        reference_asof[DType.int64](
            left, right, lv, rv, lg, rg, "nearest", True
        ),
        [0, 0, 2, 3, 3, 3],
    )
    assert_equal(
        reference_asof[DType.int64](
            left, right, lv, rv, lg, rg, "nearest", False
        ),
        [0, 2, 0, 3, 2, 3],
    )
    assert_equal(
        reference_asof[DType.int64](
            left, right, lv, rv, lg, rg, "backward", True, True, 0, 0
        ),
        [-1, 0, 2, -1, 3, -1],
    )


def _numeric_matrix[D: DType]() raises:
    var keys_l = List[Scalar[D]]()
    var keys_r = List[Scalar[D]]()
    for value in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        keys_l.append(Scalar[D](value))
    for value in [1, 2, 2, 4, 6, 6, 9]:
        keys_r.append(Scalar[D](value))
    for strategy in ["backward", "forward", "nearest"]:
        for mode in range(4):
            var lv = List[Bool](length=9, fill=mode != 3)
            var rv = List[Bool](length=7, fill=mode != 2)
            if mode == 1:
                lv[3] = False
                # Nearest's documented null-position scan semantics have a
                # separate pinned external literal oracle below.
                if strategy != "nearest":
                    rv[1] = False
            var left = DataFrame(
                [
                    Series("k", Column[Scalar[D]](keys_l.copy(), lv.copy())),
                    Series(
                        "p", Column[Int64]([10, 11, 12, 13, 14, 15, 16, 17, 18])
                    ),
                ]
            )
            var right = DataFrame(
                [
                    Series("k", Column[Scalar[D]](keys_r.copy(), rv.copy())),
                    Series(
                        "p",
                        Column[String](
                            ["a", "b", "c", "d", "e", "f", "g"],
                            [True, False, True, True, True, True, True],
                        ),
                    ),
                ]
            )
            for layout in range(3):
                var l = _layout(left, layout)
                var r = _layout(right, layout)
                for exact in [False, True]:
                    for bound in [-1, 0, 1, 2]:
                        var selected = reference_asof[D](
                            keys_l,
                            keys_r,
                            lv,
                            rv,
                            List[Int](length=9, fill=0),
                            List[Int](length=7, fill=0),
                            strategy,
                            exact,
                            bound >= 0,
                            Int128(bound),
                            Float64(bound),
                        )
                        var tolerance = Optional[AnyValue]()
                        if bound >= 0:
                            tolerance = AnyValue(Int64(bound))
                        _check(l, r, selected, strategy, exact, tolerance)
                    _check(l.clear(), r, List[Int](), strategy, exact)
                    _check(
                        l,
                        r.clear(),
                        List[Int](length=9, fill=-1),
                        strategy,
                        exact,
                    )


def test_asof_all_numeric_widths_match_quadratic_direction_and_tolerance_reference() raises:
    comptime for i in range(len(INTEGER_DTYPES)):
        _numeric_matrix[INTEGER_DTYPES[i]]()
    comptime for i in range(len(FLOAT_DTYPES)):
        _numeric_matrix[FLOAT_DTYPES[i]]()


def _integer_extremes[D: DType]() raises:
    var bounds = integer_bounds[D]()
    var a = bounds[0]
    var b = bounds[1]
    var lraw: List[Int128] = [a, a + 1, (a + b) // 2, b - 1, b]
    var rraw: List[Int128] = [a, a + 2, b - 2, b]
    var left_values = List[Scalar[D]]()
    var right_values = List[Scalar[D]]()
    for value in lraw:
        left_values.append(value.cast[D]())
    for value in rraw:
        right_values.append(value.cast[D]())
    var left = DataFrame(
        [
            Series("k", Column[Scalar[D]](left_values.copy())),
            Series("p", Column[Int64]([0, 1, 2, 3, 4])),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Scalar[D]](right_values.copy())),
            Series("p", Column[Int64]([10, 11, 12, 13])),
        ]
    )
    for strategy in ["backward", "forward", "nearest"]:
        for exact in [False, True]:
            for limited in [False, True]:
                var selected = reference_asof[D](
                    left_values,
                    right_values,
                    List[Bool](length=5, fill=True),
                    List[Bool](length=4, fill=True),
                    List[Int](length=5, fill=0),
                    List[Int](length=4, fill=0),
                    strategy,
                    exact,
                    limited,
                    1,
                    1,
                )
                var tolerance = Optional[AnyValue]()
                if limited:
                    tolerance = AnyValue(UInt64(1))
                _check(left, right, selected, strategy, exact, tolerance)


def test_asof_integer_extrema_and_unsigned_tolerances_keep_full_precision() raises:
    comptime for i in range(len(INTEGER_DTYPES)):
        _integer_extremes[INTEGER_DTYPES[i]]()


def test_asof_temporal_units_zones_and_exact_duration_tolerances() raises:
    var lraw: List[Int64] = [0, 1, 2, 3, 4]
    var rraw: List[Int64] = [0, 2, 4]
    for dtype in [
        DataType.DATE,
        DataType.datetime("ns", "UTC"),
        DataType.datetime("us"),
        DataType.datetime("ms", "America/New_York"),
        DataType.duration("ns"),
        DataType.duration("us"),
        DataType.duration("ms"),
    ]:
        var left = DataFrame(
            [
                Series("k", Column[Int64](lraw.copy())).with_dtype(dtype),
                Series("p", Column[Int64]([0, 1, 2, 3, 4])),
            ]
        )
        var right = DataFrame(
            [
                Series("k", Column[Int64](rraw.copy())).with_dtype(dtype),
                Series("p", Column[Int64]([10, 11, 12])),
            ]
        )
        for strategy in ["backward", "forward", "nearest"]:
            for exact in [False, True]:
                for layout in range(3):
                    var l = _layout(left, layout)
                    var r = _layout(right, layout)
                    for bound in [0, 1, 2]:
                        var selected = reference_asof[DType.int64](
                            lraw,
                            rraw,
                            List[Bool](length=5, fill=True),
                            List[Bool](length=3, fill=True),
                            List[Int](length=5, fill=0),
                            List[Int](length=3, fill=0),
                            strategy,
                            exact,
                            True,
                            Int128(bound),
                            Float64(bound),
                        )
                        _check(
                            l,
                            r,
                            selected,
                            strategy,
                            exact,
                            AnyValue(Int64(bound)),
                        )
                        var ticks_per_unit = Int64(
                            86400000000000
                        ) if dtype.is_date() else Int64(
                            1000000000 // dtype.per_second()
                        )
                        _check(
                            l,
                            r,
                            selected,
                            strategy,
                            exact,
                            AnyValue.temporal(
                                DataType.duration("ns"),
                                Int64(bound) * ticks_per_unit,
                            ),
                        )
                    var exact_only = reference_asof[DType.int64](
                        lraw,
                        rraw,
                        List[Bool](length=5, fill=True),
                        List[Bool](length=3, fill=True),
                        List[Int](length=5, fill=0),
                        List[Int](length=3, fill=0),
                        strategy,
                        exact,
                        True,
                        0,
                        0,
                    )
                    _check(
                        l,
                        r,
                        exact_only,
                        strategy,
                        exact,
                        AnyValue(Float64(0.5)),
                    )


def test_asof_nearest_null_positions_follow_pinned_external_scan_fixtures() raises:
    # Independent fixture provenance: repository-pinned Polars 1.44.2,
    # join_asof(strategy="nearest", check_sortedness=False), left [1,2,3],
    # right [0,2,2,null,4], [0,2,null,2,4], [null,0,2,2,4],
    # [0,2,2,4,null]. Payload is original right row index. Null positions
    # change adjacency and exact-match scanning, not mathematical distance.
    var left = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 3])),
            Series("p", Column[Int64]([0, 1, 2])),
        ]
    )
    var keys: List[List[Int64]] = [
        [0, 2, 2, 99, 4],
        [0, 2, 99, 2, 4],
        [99, 0, 2, 2, 4],
        [0, 2, 2, 4, 99],
    ]
    var masks: List[List[Bool]] = [
        [True, True, True, False, True],
        [True, True, False, True, True],
        [False, True, True, True, True],
        [True, True, True, True, False],
    ]
    var with_exact: List[List[Int]] = [
        [2, 4, 4],
        [1, 3, 4],
        [3, 3, 4],
        [2, 2, 3],
    ]
    var without_exact: List[List[Int]] = [
        [2, 4, 4],
        [1, 4, 4],
        [3, 4, 4],
        [2, 3, 3],
    ]
    for example in range(4):
        var right = DataFrame(
            [
                Series(
                    "k",
                    Column[Int64](keys[example].copy(), masks[example].copy()),
                ),
                Series("p", Column[Int64]([0, 1, 2, 3, 4])),
            ]
        )
        for layout in range(3):
            for exact in [False, True]:
                _check(
                    _layout(left, layout),
                    _layout(right, layout),
                    with_exact[example]
                    .copy() if exact else without_exact[example]
                    .copy(),
                    "nearest",
                    exact,
                )


def test_asof_payloads_preserve_nested_values_and_logical_metadata() raises:
    for payload in typed_join_values():
        var left = DataFrame(
            [Series("k", Column[Int64]([0, 1, 2, 3, 4])), payload.renamed("p")]
        )
        var right = DataFrame(
            [Series("k", Column[Int64]([1, 2, 3, 4, 5])), payload.renamed("p")]
        )
        for layout in range(3):
            _check(
                _layout(left, layout),
                _layout(right, layout),
                [-1, 0, 1, 2, 3],
                "backward",
                True,
            )


def test_asof_interleaved_composite_groups_skip_null_members_and_unify_categories() raises:
    var lk: List[Int64] = [1, 10, 2, 11, 3, 12, 999, 4, 5, 6]
    var rk: List[Int64] = [0, 9, 2, 11, 4, 14, 7, 6, 100]
    var lv: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    var rv = List[Bool](length=9, fill=True)
    var lg: List[Int] = [0, 1, 2, 1, 0, 1, 0, 0, -1, -1]
    var rg: List[Int] = [0, 1, 2, 1, 0, 1, 0, 2, -1]
    var left_g = Series(
        "g",
        Column[String](
            ["a", "b", "a", "b", "a", "b", "a", "a", "hidden", "a"],
            [True, True, True, True, True, True, True, True, False, True],
        ),
    )
    var right_g = Series(
        "g",
        Column[String](
            ["a", "b", "a", "b", "a", "b", "a", "a", "hidden"],
            [True, True, True, True, True, True, True, True, False],
        ),
    )
    var dictionary_permutation: List[Int] = [1, 0, 2, 3, 4, 5, 6, 7, 8]
    for encoded in [False, True]:
        var gleft = left_g.cast(
            DataType.CATEGORICAL
        ) if encoded else left_g.copy()
        var gright = (
            right_g.take(dictionary_permutation)
            .cast(DataType.CATEGORICAL)
            .take(dictionary_permutation) if encoded else right_g.copy()
        )
        var left = DataFrame(
            [
                Series("k", Column[Int64](lk.copy(), lv.copy())),
                gleft^,
                Series(
                    "h",
                    Column[Bool](
                        [
                            True,
                            True,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                        ],
                        [
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                        ],
                    ),
                ),
                Series("p", Column[Int64]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
            ]
        )
        var right = DataFrame(
            [
                Series("k", Column[Int64](rk.copy())),
                gright^,
                Series(
                    "h",
                    Column[Bool](
                        [True, True, False, True, True, True, True, False, True]
                    ),
                ),
                Series(
                    "p", Column[Int64]([10, 11, 12, 13, 14, 15, 16, 17, 18])
                ),
            ]
        )
        for layout in range(3):
            var l = _layout(left, layout)
            var r = _layout(right, layout)
            for strategy in ["backward", "forward", "nearest"]:
                for exact in [False, True]:
                    for bound in [-1, 1, 3]:
                        var selected = reference_asof[DType.int64](
                            lk,
                            rk,
                            lv,
                            rv,
                            lg,
                            rg,
                            strategy,
                            exact,
                            bound >= 0,
                            Int128(bound),
                            Float64(bound),
                        )
                        var tolerance = Optional[AnyValue]()
                        if bound >= 0:
                            tolerance = AnyValue(Int64(bound))
                        var actual = l.join_asof(
                            r,
                            on="k",
                            by=["g", "h"],
                            strategy=strategy,
                            allow_exact_matches=exact,
                            tolerance=tolerance,
                        )
                        assert_asof_rows(
                            actual,
                            l,
                            r,
                            selected,
                            ["k", "g", "h", "p", "p_right"],
                            ["p"],
                        )
            var single_left = l.slice(0, 1)
            var single_right = r.slice(0, 1)
            var single = single_left.join_asof(single_right, on="k", by="g")
            assert_asof_rows(
                single,
                single_left,
                single_right,
                [0],
                ["k", "g", "h", "p", "h_right", "p_right"],
                ["h", "p"],
            )


def _assert_unsorted(
    left: DataFrame,
    right: DataFrame,
    side: String,
    row: Int,
    grouped: Bool = False,
) raises:
    var by: List[String] = ["g"] if grouped else List[String]()
    var failed = False
    try:
        _ = left.join_asof(right, on="k", by=by)
    except error:
        assert_true(String(error).find(side + " column 'k'") >= 0)
        assert_true(String(error).find("row " + String(row)) >= 0)
        failed = True
    assert_true(failed)
    # Building a lazy plan succeeds, validation belongs to execution in both
    # materialized and requested-streaming paths, independent of optimization.
    var plan = left.lazy().join_asof(right.lazy(), on="k", by=by)
    for streaming in [False, True]:
        for optimize in [False, True]:
            failed = False
            try:
                _ = plan.collect(
                    streaming=streaming,
                    optimize=optimize,
                    batch_size=2,
                    engine="cpu",
                )
            except error:
                assert_true(String(error).find(side + " column 'k'") >= 0)
                assert_true(String(error).find("row " + String(row)) >= 0)
                failed = True
            assert_true(failed)


def test_asof_sortedness_is_per_group_ignores_hidden_nulls_and_reports_execution_rows() raises:
    var sorted = DataFrame(
        [
            Series("k", Column[Int64]([0, 999, 1], [True, False, True])),
            Series("p", Column[Int64]([0, 1, 2])),
        ]
    )
    var descending = DataFrame(
        [
            Series("k", Column[Int64]([2, 999, 1], [True, False, True])),
            Series("p", Column[Int64]([10, 11, 12])),
        ]
    )
    _check(sorted, sorted, [0, -1, 2], "backward", True)
    _assert_unsorted(descending, sorted, "left", 2)
    _assert_unsorted(sorted, descending, "right", 2)
    var interleaved = DataFrame(
        [
            Series("k", Column[Int64]([0, 10, 2, 11, 1])),
            Series("g", Column[String](["a", "b", "a", "b", "a"])),
        ]
    )
    var ordered = DataFrame(
        [
            Series("k", Column[Int64]([0, 10, 1, 11, 2])),
            Series("g", Column[String](["a", "b", "a", "b", "a"])),
        ]
    )
    _assert_unsorted(interleaved, ordered, "left", 4, True)
    _assert_unsorted(ordered, interleaved, "right", 4, True)


def test_asof_key_group_tolerance_and_output_schema_errors_validate_empty_inputs() raises:
    var frame = DataFrame(
        [
            Series("k", Column[Int64]([0, 1])),
            Series("p", Column[Int64]([10, 11])),
            Series("g", Column[String](["a", "a"])),
        ]
    )
    for empty in [False, True]:
        var f = frame.clear() if empty else frame.copy()
        with assert_raises():
            _ = f.join_asof(f)
        with assert_raises():
            _ = f.join_asof(f, left_on="k")
        with assert_raises():
            _ = f.join_asof(f, on="k", left_on="k", right_on="k")
        with assert_raises():
            _ = f.join_asof(f, on="missing")
        with assert_raises():
            _ = f.join_asof(f, on="k", strategy="unknown")
        with assert_raises():
            _ = f.join_asof(f, on="k", by=["g", "g"])
        with assert_raises():
            _ = f.join_asof(f, on="k", by=["k"])
        with assert_raises():
            _ = f.join_asof(f, on="k", by=["missing"])
        with assert_raises():
            _ = f.join_asof(f, on="k", suffix="")
        var nan = Float64(0) / Float64(0)
        var bad_tolerances: List[AnyValue] = [
            AnyValue(Int64(-1)),
            AnyValue(Float64(-0.5)),
            AnyValue(nan),
            AnyValue("1"),
            AnyValue(True),
            AnyValue.temporal(DataType.duration("ns"), 1),
            AnyValue(DataType.INT64, False, 0, 0, False, ""),
        ]
        for tolerance in bad_tolerances:
            with assert_raises():
                _ = f.join_asof(f, on="k", tolerance=tolerance.copy())
        var changed = DataFrame(
            [f.column("k").cast(DataType.INT32), f.column("p"), f.column("g")]
        )
        with assert_raises():
            _ = f.join_asof(changed, on="k")
        changed = DataFrame(
            [
                f.column("k"),
                f.column("p"),
                f.column("g").cast(DataType.BOOL, strict=False),
            ]
        )
        with assert_raises():
            _ = f.join_asof(changed, on="k", by="g")
    var colliding = frame.with_column(Series("p_right", Column[Int64]([0, 1])))
    with assert_raises():
        _ = colliding.join_asof(frame, on="k", by="g")
    for source in [
        Series("k", Column[Bool]([False, True])),
        Series("k", Column[String](["a", "b"])),
        Series.binary("k", [[0], [1]]),
        Series("k", Column[Int64]([0, 1])).with_dtype(DataType.TIME),
        Series("k", Column[Int128]([0, 1])).with_dtype(DataType.decimal(8, 2)),
    ]:
        var unsupported = DataFrame([source.copy()])
        for f in [unsupported.copy(), unsupported.clear()]:
            with assert_raises():
                _ = f.join_asof(f, on="k")
    for target in [
        DataType.datetime("us", "UTC"),
        DataType.datetime("ns", "America/New_York"),
        DataType.duration("ms"),
    ]:
        var left = DataFrame(
            [
                Series("k", Column[Int64]([0, 1])).with_dtype(
                    DataType.datetime("ns", "UTC")
                )
            ]
        )
        var right = DataFrame(
            [Series("k", Column[Int64]([0, 1])).with_dtype(target)]
        )
        with assert_raises():
            _ = left.join_asof(right, on="k")
    var renamed = frame.rename({"k": "q"})
    var actual = frame.join_asof(
        renamed, left_on="k", right_on="q", by=None, suffix="_other"
    )
    assert_asof_rows(
        actual,
        frame,
        renamed,
        [0, 1],
        ["k", "p", "g", "q", "p_other", "g_other"],
        ["q", "p", "g"],
    )


def _float_specials[D: DType]() raises:
    var inf = Float64(1) / Float64(0)
    var nan = Float64(0) / Float64(0)
    var lk = List[Scalar[D]]()
    var rk = List[Scalar[D]]()
    for value in [-inf, -1.0, -0.0, 0.0, 1.0, inf, nan]:
        lk.append(value.cast[D]())
    for value in [-inf, 0.0, inf, nan]:
        rk.append(value.cast[D]())
    var left = DataFrame(
        [
            Series("k", Column[Scalar[D]](lk.copy())),
            Series("p", Column[Int64]([0, 1, 2, 3, 4, 5, 6])),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Scalar[D]](rk.copy())),
            Series("p", Column[Int64]([10, 11, 12, 13])),
        ]
    )
    # Pinned Polars 1.44.2 independently generated these row-index fixtures:
    # strategy backward/forward/nearest, exact false/true, tolerance None/1/inf.
    # Keeping literals handles IEEE distance NaNs and documented scan policy
    # without pretending extended-real nearest has a unique arithmetic law.
    var expected: List[List[Int]] = [
        [-1, 0, 0, 0, 1, 1, 2],
        [-1, -1, -1, -1, 1, -1, -1],
        [-1, 0, 0, 0, 1, 1, -1],
        [0, 0, 1, 1, 1, 2, 3],
        [-1, -1, 1, 1, 1, -1, -1],
        [-1, 0, 1, 1, 1, -1, -1],
        [1, 1, 2, 2, 2, 3, -1],
        [-1, 1, -1, -1, -1, -1, -1],
        [1, 1, 2, 2, 2, -1, -1],
        [0, 1, 1, 1, 2, 2, 3],
        [-1, 1, 1, 1, -1, -1, -1],
        [-1, 1, 1, 1, 2, -1, -1],
        [1, 1, 2, 2, 1, 1, 3],
        [-1, 1, -1, -1, 1, -1, -1],
        [1, 1, 2, 2, 1, 1, -1],
        [0, 1, 1, 1, 1, 1, 3],
        [-1, 1, 1, 1, 1, -1, -1],
        [-1, 1, 1, 1, 1, 1, -1],
    ]
    var index = 0
    for strategy in ["backward", "forward", "nearest"]:
        for exact in [False, True]:
            for bound in range(3):
                var tolerance = Optional[AnyValue]()
                if bound > 0:
                    tolerance = AnyValue(Float64(1) if bound == 1 else inf)
                for layout in range(3):
                    _check(
                        _layout(left, layout),
                        _layout(right, layout),
                        expected[index],
                        strategy,
                        exact,
                        tolerance,
                    )
                index += 1
    var finite_l: List[Scalar[D]] = [
        Scalar[D](0),
        Scalar[D](0.25),
        Scalar[D](0.5),
        Scalar[D](0.75),
        Scalar[D](1),
    ]
    var finite_r: List[Scalar[D]] = [Scalar[D](0), Scalar[D](0.5), Scalar[D](1)]
    left = DataFrame(
        [
            Series("k", Column[Scalar[D]](finite_l.copy())),
            Series("p", Column[Int64]([0, 1, 2, 3, 4])),
        ]
    )
    right = DataFrame(
        [
            Series("k", Column[Scalar[D]](finite_r.copy())),
            Series("p", Column[Int64]([10, 11, 12])),
        ]
    )
    for strategy in ["backward", "forward", "nearest"]:
        for exact in [False, True]:
            for bound in [0.0, 0.2499, 0.25]:
                var selected = reference_asof[D](
                    finite_l,
                    finite_r,
                    List[Bool](length=5, fill=True),
                    List[Bool](length=3, fill=True),
                    List[Int](length=5, fill=0),
                    List[Int](length=3, fill=0),
                    strategy,
                    exact,
                    True,
                    0,
                    bound,
                )
                _check(left, right, selected, strategy, exact, AnyValue(bound))


def test_asof_float_fractional_tolerance_and_nonfinite_external_fixtures() raises:
    _float_specials[DType.float32]()
    _float_specials[DType.float64]()


def test_asof_tolerance_extrema_and_subtick_duration_bounds_are_inclusive() raises:
    # Each interval has distance 2^64-1. MAX-1 differs by one tick that a
    # Float64 oracle cannot represent; typed unsigned tolerances must retain it.
    for unsigned in [False, True]:
        var key = Series(
            "k", Column[UInt64]([0, UInt64.MAX])
        ) if unsigned else Series("k", Column[Int64]([Int64.MIN, Int64.MAX]))
        var low = DataFrame([key.slice(0, 1), Series("p", Column[Int64]([10]))])
        var high = DataFrame(
            [key.slice(1, 1), Series("p", Column[Int64]([20]))]
        )
        for layout in range(3):
            var l = _layout(low, layout)
            var h = _layout(high, layout)
            for exact in [False, True]:
                for nearest in [False, True]:
                    _check(
                        l,
                        h,
                        [0],
                        "nearest" if nearest else "forward",
                        exact,
                        AnyValue(UInt64.MAX),
                    )
                    _check(
                        h,
                        l,
                        [0],
                        "nearest" if nearest else "backward",
                        exact,
                        AnyValue(UInt64.MAX),
                    )
                    _check(
                        l,
                        h,
                        [-1],
                        "nearest" if nearest else "forward",
                        exact,
                        AnyValue(UInt64.MAX - 1),
                    )
                    _check(
                        h,
                        l,
                        [-1],
                        "nearest" if nearest else "backward",
                        exact,
                        AnyValue(UInt64.MAX - 1),
                    )
    for dtype in [
        DataType.DATE,
        DataType.datetime("us", "UTC"),
        DataType.datetime("ms"),
        DataType.duration("us"),
        DataType.duration("ms"),
    ]:
        var tick_ns = Int64(86400000000000) if dtype.is_date() else Int64(
            1000000000 // dtype.per_second()
        )
        var left = DataFrame(
            [
                Series("k", Column[Int64]([1])).with_dtype(dtype),
                Series("p", Column[Int64]([10])),
            ]
        )
        var right = DataFrame(
            [
                Series("k", Column[Int64]([0])).with_dtype(dtype),
                Series("p", Column[Int64]([20])),
            ]
        )
        for strategy in ["backward", "nearest"]:
            _check(
                left,
                right,
                [-1],
                strategy,
                True,
                AnyValue.temporal(DataType.duration("ns"), tick_ns - 1),
            )
            _check(
                left,
                right,
                [0],
                strategy,
                True,
                AnyValue.temporal(DataType.duration("ns"), tick_ns),
            )
        for empty in [False, True]:
            var l = left.clear() if empty else left.copy()
            var r = right.clear() if empty else right.copy()
            with assert_raises():
                _ = l.join_asof(
                    r,
                    on="k",
                    tolerance=AnyValue.temporal(DataType.duration("ns"), -1),
                )
