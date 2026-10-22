"""Direct-kernel as-of fixtures with public CPU and literal reference checks."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.testing import assert_equal, assert_raises
from dataframe import AnyValue, Column, DataFrame, DataType, Series
from dataframe.dtype import NUMERIC_DTYPES
from dataframe_metal.accelerator.gpu.program import kind
from dataframe_metal.accelerator.gpu import asof as gpu
from dataframe_metal.accelerator.gpu.joins import join_init, join_merge, MISSING


def _append(
    column: Series,
    mut words: List[UInt64],
    mut highs: List[UInt64],
    mut validity: List[UInt8],
    mut descriptors: List[Int64],
) raises:
    descriptors.extend(
        [Int64(len(words)), Int64(len(column)), kind(column.dtype())]
    )
    if column.dtype().is_decimal() and column.dtype().decimal_width() == 128:
        ref values = column._data[Column[Int128]]
        for row in range(len(column)):
            var word = values._get(row)
            words.append(word.cast[DType.uint64]())
            highs.append((word >> 64).cast[DType.uint64]())
            validity.append(UInt8(values.is_valid(row)))
        return
    if column.dtype() == DataType.BOOL:
        for row in range(len(column)):
            var value = column.get(row)
            words.append(
                UInt64(value.bool()) if not value.is_null() else UInt64(0)
            )
            highs.append(0)
            validity.append(UInt8(not value.is_null()))
        return
    comptime for D in NUMERIC_DTYPES:
        if column.dtype().physical() != DataType.of(D):
            continue
        var values = column.numeric[D]()
        for row in range(len(column)):
            var value = values._get(row)
            comptime if D == DType.float64:
                words.append(bitcast[DType.uint64](value))
            elif D == DType.float32:
                words.append(UInt64(bitcast[DType.uint32](value)))
            else:
                words.append(value.cast[DType.uint64]())
            highs.append(0)
            validity.append(UInt8(values.is_valid(row)))
        return
    raise Error("Probe input dtype not implemented")


def device_matches(
    left: Series,
    right: Series,
    left_groups: List[Series],
    right_groups: List[Series],
    strategy: Int,
    exact: Bool,
    has_tolerance: Bool = False,
    tolerance_integer: UInt64 = UInt64.MAX,
    tolerance_float: UInt64 = UInt64(0x7FF0000000000000),
    error_column: Int = -1,
    error_row: Int = 0,
) raises -> Tuple[List[Int], List[UInt32]]:
    var words = List[UInt64]()
    var highs = List[UInt64]()
    var validity = List[UInt8]()
    var columns = List[Int64]()
    _append(left, words, highs, validity, columns)
    _append(right, words, highs, validity, columns)
    var keys = List[Int64]()
    for key in range(len(left_groups)):
        keys.extend([Int64(2 + 2 * key), Int64(3 + 2 * key)])
        _append(left_groups[key], words, highs, validity, columns)
        _append(right_groups[key], words, highs, validity, columns)
    var cells = len(words)
    if cells == 0:
        cells = 1
        words.append(0)
        highs.append(0)
        validity.append(0)
    words.extend(highs^)
    if not len(keys):
        keys.append(-1)
    var errors = List[UInt32](length=cells, fill=0)
    if error_column >= 0:
        errors[Int(columns[3 * error_column]) + error_row] = 32
    var outputs: List[Int64] = [0, -1, 0, 0, -1, 1, 0, 1]
    if len(left_groups):
        outputs.extend([-1, Int64(3 + 2 * (len(left_groups) - 1)), 0, 2])
    var output_count = len(outputs) // 4
    var left_rows = len(left)
    var right_rows = len(right)
    var capacity = max(1, left_rows)
    var result_words = List[UInt64](length=capacity * output_count * 2, fill=0)
    var result_valid = List[UInt8](length=capacity * output_count, fill=0)
    var result_errors = List[UInt32](length=capacity * output_count, fill=0)
    var pairs = List[UInt64](length=capacity, fill=UInt64.MAX)
    var control = List[UInt32](length=3, fill=0)
    var ctx = DeviceContext()
    var dw = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var dv = ctx.enqueue_create_buffer[DType.uint8](cells)
    var de = ctx.enqueue_create_buffer[DType.uint32](cells)
    var dc = ctx.enqueue_create_buffer[DType.int64](len(columns))
    var dk = ctx.enqueue_create_buffer[DType.int64](len(keys))
    var dout = ctx.enqueue_create_buffer[DType.int64](len(outputs))
    var li = ctx.enqueue_create_buffer[DType.uint32](capacity)
    var ln = ctx.enqueue_create_buffer[DType.uint32](capacity)
    var ri = ctx.enqueue_create_buffer[DType.uint32](max(1, right_rows))
    var rn = ctx.enqueue_create_buffer[DType.uint32](max(1, right_rows))
    var lc = ctx.enqueue_create_buffer[DType.uint64](capacity)
    var rc = ctx.enqueue_create_buffer[DType.uint64](max(1, right_rows))
    var ends = ctx.enqueue_create_buffer[DType.uint32](max(1, right_rows))
    var dp = ctx.enqueue_create_buffer[DType.uint64](capacity)
    var dm = ctx.enqueue_create_buffer[DType.uint32](3)
    var dow = ctx.enqueue_create_buffer[DType.uint64](len(result_words))
    var dov = ctx.enqueue_create_buffer[DType.uint8](len(result_valid))
    var doe = ctx.enqueue_create_buffer[DType.uint32](len(result_errors))
    ctx.enqueue_copy(dw, words.unsafe_ptr())
    ctx.enqueue_copy(dv, validity.unsafe_ptr())
    ctx.enqueue_copy(de, errors.unsafe_ptr())
    ctx.enqueue_copy(dc, columns.unsafe_ptr())
    ctx.enqueue_copy(dk, keys.unsafe_ptr())
    ctx.enqueue_copy(dout, outputs.unsafe_ptr())
    ctx.enqueue_copy(dp, pairs.unsafe_ptr())
    ctx.enqueue_copy(dow, result_words.unsafe_ptr())
    ctx.enqueue_copy(dov, result_valid.unsafe_ptr())
    ctx.enqueue_copy(doe, result_errors.unsafe_ptr())
    var left_grid = max(1, (left_rows + 255) // 256)
    var right_grid = max(1, (right_rows + 255) // 256)
    ctx.enqueue_function[join_init](
        li.unsafe_ptr(), UInt32(left_rows), grid_dim=left_grid, block_dim=256
    )
    ctx.enqueue_function[join_init](
        ri.unsafe_ptr(), UInt32(right_rows), grid_dim=right_grid, block_dim=256
    )
    var width = 1
    while width < max(left_rows, right_rows):
        if width < left_rows:
            ctx.enqueue_function[join_merge](
                dw.unsafe_ptr(),
                dv.unsafe_ptr(),
                li.unsafe_ptr(),
                ln.unsafe_ptr(),
                dc.unsafe_ptr(),
                dk.unsafe_ptr(),
                UInt32(len(left_groups)),
                UInt32(1),
                UInt32(left_rows),
                UInt32(width),
                UInt64(cells),
                grid_dim=left_grid,
                block_dim=256,
            )
            var old = li^
            li = ln^
            ln = old^
        if width < right_rows:
            ctx.enqueue_function[join_merge](
                dw.unsafe_ptr(),
                dv.unsafe_ptr(),
                ri.unsafe_ptr(),
                rn.unsafe_ptr(),
                dc.unsafe_ptr(),
                dk.unsafe_ptr(),
                UInt32(len(left_groups)),
                UInt32(0),
                UInt32(right_rows),
                UInt32(width),
                UInt64(cells),
                grid_dim=right_grid,
                block_dim=256,
            )
            var old = ri^
            ri = rn^
            rn = old^
        width *= 2
    ctx.enqueue_function[gpu.asof_validate](
        dw.unsafe_ptr(),
        dv.unsafe_ptr(),
        de.unsafe_ptr(),
        li.unsafe_ptr(),
        dc.unsafe_ptr(),
        dk.unsafe_ptr(),
        lc.unsafe_ptr(),
        Int64(0),
        UInt32(len(left_groups)),
        UInt32(1),
        UInt32(left_rows),
        UInt64(cells),
        grid_dim=left_grid,
        block_dim=256,
    )
    ctx.enqueue_function[gpu.asof_validate](
        dw.unsafe_ptr(),
        dv.unsafe_ptr(),
        de.unsafe_ptr(),
        ri.unsafe_ptr(),
        dc.unsafe_ptr(),
        dk.unsafe_ptr(),
        rc.unsafe_ptr(),
        Int64(1),
        UInt32(len(left_groups)),
        UInt32(0),
        UInt32(right_rows),
        UInt64(cells),
        grid_dim=right_grid,
        block_dim=256,
    )
    ctx.enqueue_function[gpu.asof_control](
        lc.unsafe_ptr(),
        rc.unsafe_ptr(),
        dm.unsafe_ptr(),
        UInt32(left_rows),
        UInt32(right_rows),
        grid_dim=1,
        block_dim=1,
    )
    ctx.enqueue_copy(control.unsafe_ptr(), dm)
    ctx.synchronize()
    var selected = List[Int](length=left_rows, fill=-1)
    if control[0] != MISSING or control[1] != MISSING or control[2]:
        return (selected^, control^)
    ctx.enqueue_function[gpu.asof_right_ends](
        dw.unsafe_ptr(),
        dv.unsafe_ptr(),
        ri.unsafe_ptr(),
        dc.unsafe_ptr(),
        dk.unsafe_ptr(),
        ends.unsafe_ptr(),
        Int64(1),
        UInt32(len(left_groups)),
        UInt32(right_rows),
        UInt64(cells),
        grid_dim=right_grid,
        block_dim=256,
    )
    ctx.enqueue_function[gpu.asof_match](
        dw.unsafe_ptr(),
        dv.unsafe_ptr(),
        li.unsafe_ptr(),
        ri.unsafe_ptr(),
        ends.unsafe_ptr(),
        dc.unsafe_ptr(),
        dk.unsafe_ptr(),
        dp.unsafe_ptr(),
        Int64(0),
        Int64(1),
        UInt32(len(left_groups)),
        UInt32(left_rows),
        UInt32(right_rows),
        UInt32(strategy),
        UInt32(exact),
        UInt32(has_tolerance),
        tolerance_integer,
        tolerance_float,
        UInt64(cells),
        grid_dim=left_grid,
        block_dim=256,
    )
    ctx.enqueue_function[gpu.asof_gather](
        dw.unsafe_ptr(),
        dv.unsafe_ptr(),
        de.unsafe_ptr(),
        dc.unsafe_ptr(),
        dout.unsafe_ptr(),
        dp.unsafe_ptr(),
        dow.unsafe_ptr(),
        dov.unsafe_ptr(),
        doe.unsafe_ptr(),
        UInt32(left_rows),
        UInt32(output_count),
        UInt64(cells),
        grid_dim=left_grid,
        block_dim=256,
    )
    ctx.enqueue_copy(pairs.unsafe_ptr(), dp)
    ctx.enqueue_copy(result_words.unsafe_ptr(), dow)
    ctx.enqueue_copy(result_valid.unsafe_ptr(), dov)
    ctx.enqueue_copy(result_errors.unsafe_ptr(), doe)
    ctx.synchronize()
    for row in range(left_rows):
        assert_equal(UInt32(pairs[row]), UInt32(row))
        var right_row = UInt32(pairs[row] >> 32)
        selected[row] = -1 if right_row == MISSING else Int(right_row)
        for column in range(output_count):
            var original = row if column == 0 else selected[row]
            var descriptor = (
                0 if column
                == 0 else 1 if column
                == 1 else 3 + 2 * (len(left_groups) - 1)
            )
            var source = Int(columns[3 * descriptor]) + max(0, original)
            var valid = original >= 0 and validity[source] != 0
            var at = column * capacity + row
            assert_equal(result_valid[at] != 0, valid)
            assert_equal(result_errors[at], UInt32(0))
            assert_equal(
                result_words[at], words[source] if valid else UInt64(0)
            )
            assert_equal(
                result_words[capacity * output_count + at],
                words[cells + source] if valid else UInt64(0),
            )
    # Every host upload and device owner outlives its queued reads.
    _ = words^
    _ = validity^
    _ = errors^
    _ = columns^
    _ = keys^
    _ = outputs^
    _ = dw^
    _ = dv^
    _ = de^
    _ = dc^
    _ = dk^
    _ = dout^
    _ = li^
    _ = ln^
    _ = ri^
    _ = rn^
    _ = lc^
    _ = rc^
    _ = ends^
    _ = dp^
    _ = dm^
    _ = dow^
    _ = dov^
    _ = doe^
    return (selected^, control^)


def check(
    left: Series,
    right: Series,
    strategy: Int,
    exact: Bool,
    expected: List[Int] = List[Int](),
    left_groups: List[Series] = List[Series](),
    right_groups: List[Series] = List[Series](),
    tolerance: Optional[AnyValue] = None,
    tolerance_integer: UInt64 = UInt64.MAX,
    tolerance_float: UInt64 = UInt64(0x7FF0000000000000),
) raises:
    var bounded = False
    if tolerance:
        bounded = True
    var actual = device_matches(
        left,
        right,
        left_groups,
        right_groups,
        strategy,
        exact,
        bounded,
        tolerance_integer,
        tolerance_float,
    )
    assert_equal(actual[1], [MISSING, MISSING, UInt32(0)])
    var lhs: List[Series] = [left.renamed("k")]
    var rhs: List[Series] = [
        right.renamed("k"),
        Series(
            "position", Column[Int64](List[Int64](length=len(right), fill=0))
        ),
    ]
    var ids = List[Int64]()
    for row in range(len(right)):
        ids.append(Int64(row))
    rhs[1] = Series("position", Column[Int64](ids^))
    var by = List[String]()
    for key in range(len(left_groups)):
        var name = "g" + String(key)
        by.append(name)
        lhs.append(left_groups[key].renamed(name))
        rhs.append(right_groups[key].renamed(name))
    var cpu = DataFrame(lhs^).join_asof(
        DataFrame(rhs^),
        on="k",
        by=by,
        strategy="backward" if strategy
        == 0 else "forward" if strategy
        == 1 else "nearest",
        tolerance=tolerance,
        allow_exact_matches=exact,
    )
    assert_equal(cpu.height(), len(left))
    var cpu_pairs = List[Int](capacity=len(left))
    for row in range(len(left)):
        var position = cpu.item(row, "position")
        cpu_pairs.append(-1 if position.is_null() else Int(position.int64()))
    if actual[0] != cpu_pairs or (len(expected) and actual[0] != expected):
        print(
            "ASOF CASE dtype=",
            left.dtype().name(),
            "strategy=",
            strategy,
            "exact=",
            exact,
            "bounded=",
            bounded,
            "tolerance_float_bits=",
            tolerance_float,
        )
        print("GPU pairs:", actual[0])
        print("Actual CPU pairs:", cpu_pairs)
        print("Independent literal pairs:", expected)
    assert_equal(actual[0], cpu_pairs)
    if len(expected):
        assert_equal(actual[0], expected)
    for row in range(len(left)):
        assert_equal(cpu.item(row, "position").is_null(), actual[0][row] < 0)
        if actual[0][row] >= 0:
            assert_equal(
                cpu.item(row, "position").int64(), Int64(actual[0][row])
            )


def directions_null_runs_and_integer_distances() raises:
    comptime for D in NUMERIC_DTYPES:
        var left = List[Scalar[D]]()
        var right = List[Scalar[D]]()
        for value in [1, 2, 3]:
            left.append(Scalar[D](value))
        for value in [0, 2, 2, 4]:
            right.append(Scalar[D](value))
        var l = Series("k", Column[Scalar[D]](left^))
        var r = Series("k", Column[Scalar[D]](right^))
        var expected: List[List[Int]] = [
            [0, 0, 2],
            [0, 2, 2],
            [1, 3, 3],
            [1, 1, 3],
            [2, 3, 3],
            [2, 2, 3],
        ]
        for strategy in range(3):
            for exact in [False, True]:
                check(
                    l, r, strategy, exact, expected[2 * strategy + Int(exact)]
                )
    var left = Series("k", Column[Int64]([1, 2, 3]))
    var examples: List[List[Int64]] = [
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
    var exact_pairs: List[List[Int]] = [
        [2, 4, 4],
        [1, 3, 4],
        [3, 3, 4],
        [2, 2, 3],
    ]
    var strict_pairs: List[List[Int]] = [
        [2, 4, 4],
        [1, 4, 4],
        [3, 4, 4],
        [2, 3, 3],
    ]
    for example in range(4):
        var right = Series(
            "k", Column[Int64](examples[example].copy(), masks[example].copy())
        )
        check(left, right, 2, True, exact_pairs[example])
        check(left, right, 2, False, strict_pairs[example])
    for unsigned in [False, True]:
        var low = Series("k", Column[UInt64]([0])) if unsigned else Series(
            "k", Column[Int64]([Int64.MIN])
        )
        var high = Series(
            "k", Column[UInt64]([UInt64.MAX])
        ) if unsigned else Series("k", Column[Int64]([Int64.MAX]))
        for exact in [False, True]:
            check(
                low,
                high,
                1,
                exact,
                [0],
                tolerance=AnyValue(UInt64.MAX),
                tolerance_integer=UInt64.MAX,
            )
            check(
                low,
                high,
                1,
                exact,
                [-1],
                tolerance=AnyValue(UInt64.MAX - 1),
                tolerance_integer=UInt64.MAX - 1,
            )
            check(
                high,
                low,
                0,
                exact,
                [0],
                tolerance=AnyValue(UInt64.MAX),
                tolerance_integer=UInt64.MAX,
            )
    for unit in ["ms", "us", "ns"]:
        var dtype = DataType.datetime(unit, "+05:45")
        check(
            Series("k", Column[Int64]([-1, 0, 1])).with_dtype(dtype),
            Series("k", Column[Int64]([-2, 0, 2])).with_dtype(dtype),
            2,
            True,
            [1, 1, 2],
        )
        var tick_ns = (
            Int64(1000000) if unit
            == "ms" else Int64(1000) if unit
            == "us" else Int64(1)
        )
        var left = Series("k", Column[Int64]([1])).with_dtype(dtype)
        var right = Series("k", Column[Int64]([0])).with_dtype(dtype)
        check(
            left,
            right,
            0,
            True,
            [-1],
            tolerance=AnyValue.temporal(DataType.duration("ns"), tick_ns - 1),
            tolerance_integer=0,
        )
        check(
            left,
            right,
            0,
            True,
            [0],
            tolerance=AnyValue.temporal(DataType.duration("ns"), tick_ns),
            tolerance_integer=1,
        )
    for dtype in [
        DataType.DATE,
        DataType.duration("ms"),
        DataType.duration("us"),
        DataType.duration("ns"),
    ]:
        check(
            Series("k", Column[Int64]([-1, 0, 1])).with_dtype(dtype),
            Series("k", Column[Int64]([-2, 0, 2])).with_dtype(dtype),
            2,
            True,
            [1, 1, 2],
        )


def ieee_comparisons_and_tolerance() raises:
    var expected: List[List[Int]] = [
        [-1, 0, 0, 0, 1, 1, 2],
        [-1, -1, -1, -1, 1, -1, -1],
        [-1, 0, 0, 0, 1, 1, -1],
        [0, 0, 1, 1, 1, 2, 3],
        [-1, -1, 1, 1, 1, -1, -1],
        [-1, 0, 1, 1, 1, -1, -1],
        [1, 1, 2, 2, 2, 3, -1],
        [-1, 1, -1, -1, -1, -1, -1],
        # Current CPU forward/exact=False with tolerance=inf selects right0
        # for left -inf: abs(-inf - 0) == inf <= inf. The external Polars
        # fixture used -1 for this first cell; direct CPU/GPU diagnostic3b
        # confirmed row1 for both. Retain that distinct external provenance.
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
    comptime float_types = (DType.float32, DType.float64)
    comptime for type_index in range(len(float_types)):
        comptime D = float_types[type_index]
        var left = List[Scalar[D]]()
        var right = List[Scalar[D]]()
        var inf = Float64(1) / Float64(0)
        var nan = Float64(0) / Float64(0)
        for value in [-inf, -1.0, -0.0, 0.0, 1.0, inf, nan]:
            left.append(value.cast[D]())
        for value in [-inf, 0.0, inf, nan]:
            right.append(value.cast[D]())
        var l = Series("k", Column[Scalar[D]](left^))
        var r = Series("k", Column[Scalar[D]](right^))
        var index = 0
        for strategy in range(3):
            for exact in [False, True]:
                for bound in range(3):
                    var tolerance = Optional[AnyValue]()
                    var bits = bitcast[DType.uint64](
                        Float64(1) if bound == 1 else inf
                    )
                    if bound:
                        tolerance = AnyValue(Float64(1) if bound == 1 else inf)
                    check(
                        l,
                        r,
                        strategy,
                        exact,
                        expected[index],
                        tolerance=tolerance,
                        tolerance_float=bits,
                    )
                    index += 1
        var finite_left = Series(
            "k",
            Column[Scalar[D]](
                [
                    Scalar[D](0),
                    Scalar[D](0.25),
                    Scalar[D](0.5),
                    Scalar[D](0.75),
                    Scalar[D](1),
                ]
            ),
        )
        var finite_right = Series(
            "k", Column[Scalar[D]]([Scalar[D](0), Scalar[D](0.5), Scalar[D](1)])
        )
        for strategy in range(3):
            for exact in [False, True]:
                for tolerance in [0.0, 0.2499, 0.25]:
                    check(
                        finite_left,
                        finite_right,
                        strategy,
                        exact,
                        tolerance=AnyValue(Float64(tolerance)),
                        tolerance_float=bitcast[DType.uint64](
                            Float64(tolerance)
                        ),
                    )
        var minimum = Scalar[D](0)
        comptime if D == DType.float32:
            minimum = bitcast[DType.float32](UInt32(1)).cast[D]()
        else:
            minimum = bitcast[DType.float64](UInt64(1)).cast[D]()
        var subnormal_left = Series(
            "k",
            Column[Scalar[D]](
                [-minimum, Scalar[D](-0.0), Scalar[D](0), minimum]
            ),
        )
        var subnormal_right = Series(
            "k", Column[Scalar[D]]([-minimum, Scalar[D](0), minimum])
        )
        for strategy in range(3):
            check(subnormal_left, subnormal_right, strategy, True, [0, 1, 1, 2])


def grouped_keys_sortedness_and_error_observation() raises:
    var l = Series("k", Column[Int64]([1, 1, 2, 2, 3, 3, 4, 4]))
    var r = Series("k", Column[Int64]([0, 0, 2, 2, 2, 2, 4, 4]))
    var groups = Series(
        "g",
        Column[Int32](
            [1, 0, 1, 0, 1, 0, 1, 0],
            [True, True, True, True, True, True, False, False],
        ),
    )
    var lf = List[Float64]()
    var rf = List[Float64]()
    var wide = List[Int128]()
    for row in range(8):
        lf.append(
            bitcast[DType.float64](UInt64(0x7FF8000000000001)) if row % 2
            == 0 else bitcast[DType.float64](UInt64(0x8000000000000000))
        )
        rf.append(
            bitcast[DType.float64](UInt64(0xFFF8000000000002)) if row % 2
            == 0 else Float64(0)
        )
        wide.append((Int128(1) << 80) if row % 2 == 0 else -(Int128(1) << 80))
    var left_groups: List[Series] = [
        groups.copy(),
        Series("f", Column[Float64](lf^)),
        Series("d", Column[Int128](wide.copy())).with_dtype(
            DataType.decimal(38, 0)
        ),
    ]
    var right_groups: List[Series] = [
        groups.copy(),
        Series("f", Column[Float64](rf^)),
        Series("d", Column[Int128](wide^)).with_dtype(DataType.decimal(38, 0)),
    ]
    for strategy in range(3):
        for exact in [False, True]:
            check(
                l,
                r,
                strategy,
                exact,
                left_groups=left_groups,
                right_groups=right_groups,
            )
    var many_left = List[Int64]()
    var many_right = List[Int64]()
    var left_keys = List[UInt64]()
    var right_keys = List[UInt64]()
    var left_valid = List[Bool]()
    var right_valid = List[Bool]()
    for row in range(263):
        many_left.append(Int64(row // 3 - 4))
        left_keys.append(UInt64.MAX - UInt64(row % 3))
        left_valid.append(row % 17 != 5)
    for row in range(257):
        many_right.append(Int64(row // 3 - 4))
        right_keys.append(UInt64.MAX - UInt64(row % 3))
        right_valid.append(row % 13 != 3)
    var large_left = Series("k", Column[Int64](many_left^, left_valid^))
    var large_right = Series("k", Column[Int64](many_right^, right_valid^))
    large_left = Series._from_chunks(
        [
            large_left.head(129),
            large_left.slice(129, 0),
            large_left.slice(129, 134),
        ]
    )
    large_right = Series._from_chunks(
        [
            large_right.head(129),
            large_right.slice(129, 0),
            large_right.slice(129, 128),
        ]
    )
    for strategy in range(3):
        for exact in [False, True]:
            check(
                large_left,
                large_right,
                strategy,
                exact,
                left_groups=[Series("g", Column[UInt64](left_keys.copy()))],
                right_groups=[Series("g", Column[UInt64](right_keys.copy()))],
            )
    var empty = Series("k", Column[Int64](List[Int64]()))
    var unsorted = Series("k", Column[Int64]([2, 1]))
    var null_left = Series("g", Column[Int64](List[Int64]()))
    var null_right = Series("g", Column[Int64]([7, 7], [False, False]))
    var validation = device_matches(
        empty, unsorted, [null_left.copy()], [null_right.copy()], 0, True
    )
    assert_equal(validation[1], [MISSING, UInt32(1), UInt32(0)])
    with assert_raises(contains="not sorted within its group"):
        _ = DataFrame([empty.copy(), null_left.copy()]).join_asof(
            DataFrame([unsorted.copy(), null_right.copy()]), on="k", by=["g"]
        )
    check(empty, r, 2, True)
    check(l, empty, 2, True, List[Int](length=len(l), fill=-1))
    check(empty, empty, 2, True)
    var left_bad = Series("k", Column[Int64]([1, 3, 2]))
    validation = device_matches(left_bad, unsorted, [], [], 2, True)
    assert_equal(validation[1], [UInt32(2), UInt32(1), UInt32(0)])
    var hidden = Series("k", Column[Int64]([2, -99, 3], [True, False, True]))
    check(Series("k", Column[Int64]([2])), hidden, 2, True, [2])
    validation = device_matches(
        l, r, [], [], 0, True, error_column=1, error_row=7
    )
    assert_equal(validation[1], [MISSING, MISSING, UInt32(32)])
    validation = device_matches(
        l, r, left_groups, right_groups, 0, True, error_column=2, error_row=7
    )
    assert_equal(validation[1], [MISSING, MISSING, UInt32(32)])
    assert_equal(gpu.workspace_bytes(0, 0), 56)
