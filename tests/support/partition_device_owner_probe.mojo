"""Resident source mapping/offset/ownership and exact OVER scheduler proof."""
from std.memory import ArcPointer, bitcast
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
from dataframe.hashing import encode_rows
from dataframe.partition import _hash_bytes
from dataframe.accelerator.window_keys import WindowKey
from dataframe.accelerator.gpu.residency import Scope, Table
from dataframe.accelerator.gpu.runtime import _preflight, execute_plan
from dataframe.accelerator.gpu.partition_layout import (
    KeyInput,
    build,
    estimate_bytes,
)
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.partition_schedule_oracle import expected


def _stage(frame: DataFrame, scope: Scope) raises -> Table:
    var columns = List[Expr](capacity=frame.width())
    for field in frame._columns:
        columns.append(col(field.name()))
    var plan = _preflight(frame.lazy().select_exprs(columns))
    var result = execute_plan(plan, scope, export_values=False)
    return result[0].copy()


def _levels(value: Int) -> Int:
    var count = 0
    var width = 1
    while width < value:
        width *= 2
        count += 1
    return count


def _check(
    kind: Int,
    n: Int,
    configured: Int,
    allow_contiguous: Bool,
    cached_layout: Bool = False,
) raises:
    var setting = WorkerSetting(configured)
    var offset = 3 if kind % 2 == 0 else 0
    var capacity = n + 5
    var masks = List[Bool](capacity=capacity)
    var codes = List[UInt32](capacity=capacity)
    var signed = List[Int64](capacity=capacity)
    var floating = List[Float64](capacity=capacity)
    var wide = List[Int128](capacity=capacity)
    var text = List[String](capacity=capacity)
    var labels: List[String] = ["a", "z", "é🙂", ""]
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000042))
    for at in range(capacity):
        var row = max(0, at - offset)
        var key = row % 2
        if kind == 1:
            key = Int(row >= 192)
        elif kind == 3:
            key = row % 4
        elif kind == 5:
            key = row * 1000000007
        codes.append(UInt32(key & 3))
        signed.append(Int64(key))
        floating.append(nan if row % 7 == 0 else Float64(row % 3 - 1))
        wide.append(
            bitcast[DType.int128](
                (UInt128(1) << UInt128(100))
                + UInt128(row % 5)
                + (UInt128(row % 3) << UInt128(64))
            )
        )
        text.append(labels[key & 3])
        masks.append(kind < 2 or row % 17 != 0)
    var physical: Series
    var logical: Series
    var key_values = List[UInt64]()
    var original_codes = False
    if kind == 2:
        physical = Series("k", Column[Float64](floating^, masks.copy()))
        logical = physical.copy()
    elif kind == 3:
        physical = Series("k", Column[UInt32](codes^, masks.copy()))
        logical = Series("k", StringColumn(text^, masks.copy()))
        for label in labels:
            var raw = label.as_bytes()
            var bytes = Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=raw.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                length=len(raw),
            )
            key_values.append(_hash_bytes(bytes))
    elif kind == 4:
        var dictionary = CategoricalDictionary()
        dictionary.append("same")
        dictionary.append("same")
        var dtype = DataType.categorical(dictionary^)
        physical = Series("k", Column[UInt32](codes^, masks.copy()))
        logical = physical.with_dtype(dtype)
        key_values = [0, 1]
        original_codes = True
    elif kind == 6:
        physical = Series("k", Column[Int128](wide^, masks.copy())).with_dtype(
            DataType.decimal(38, 0)
        )
        logical = physical.copy()
    else:
        physical = Series("k", Column[Int64](signed^, masks.copy()))
        logical = physical.copy()
    var cpu_key = logical.slice(offset, n)
    var equality = encode_rows([cpu_key.copy()], nulls_equal=True)
    var wanted = expected(
        [cpu_key.copy()],
        equality.ids,
        equality.representatives,
        configured,
        allow_contiguous,
        cached_layout,
    )
    var scope = Scope()
    var frame = DataFrame(
        [
            Series(
                "payload", Column[Int64](List[Int64](length=capacity, fill=7))
            ),
            physical^,
        ]
    )
    var source = _stage(frame, scope)
    source.offset = offset
    source.rows = n
    if offset:
        var schema = DataFrame(
            [
                source.schema._columns[1].copy(),
                source.schema._columns[0].copy(),
            ]
        )
        var columns: List[Int64] = [source.columns[1], source.columns[0]]
        source.schema = schema^
        source.columns = columns^
    var key = WindowKey("k", logical.dtype(), kind == 3)
    key.values = ArcPointer(key_values^)
    var keys: List[KeyInput] = [
        KeyInput("k", key, original_codes=original_codes)
    ]
    var before = scope.transfers()
    var live = scope.info()[0]
    var schedule = build(
        source,
        keys,
        scope,
        configured,
        allow_contiguous=allow_contiguous,
        cached_layout=cached_layout,
    )
    var after = scope.transfers()
    assert_equal(scope.info()[0] - live, 32 * n + 20)
    assert_equal(schedule[].source.offset, offset)
    assert_equal(schedule[].source.columns[0], source.columns[0])
    assert_equal(schedule[].source.columns[1], source.columns[1])
    assert_equal(after[1] - before[1], 4 * n * source.schema.width() + 32)
    assert_equal(after[3] - before[3], 3)
    assert_equal(
        after[2] - before[2], 12 + _levels(n) + 2 * _levels(equality.count())
    )
    var groups = schedule[].groups
    assert_equal(groups, equality.count())
    for at in range(4):
        assert_equal(schedule[].facts[at], wanted.facts[at])
    var index = scope.read_control(schedule[].index)
    var starts = scope.read_control(
        schedule[].starts.create_sub_buffer[DType.uint32](0, groups + 1)
    )
    var group_rows = scope.read_control(schedule[].group_by_row)
    var phase = scope.read_control(
        schedule[].phase_by_group.create_sub_buffer[DType.uint32](0, groups)
    )
    var rank = scope.read_control(
        schedule[].rank_by_group.create_sub_buffer[DType.uint32](0, groups)
    )
    var offsets = scope.read_control(
        schedule[].offset_by_group.create_sub_buffer[DType.uint32](0, groups)
    )
    var row_phase = scope.read_control(schedule[].phase_by_row)
    var ordinal = scope.read_control(schedule[].ordinal_by_row)
    var canonical_to_device = List[UInt32](length=groups, fill=UInt32.MAX)
    for group in range(groups):
        var original = equality.ids[Int(index[Int(starts[group])])]
        canonical_to_device[original] = UInt32(group)
        assert_equal(phase[group], wanted.phase[original])
        assert_equal(rank[group], wanted.rank[original])
        assert_equal(offsets[group], wanted.offset[original])
    for row in range(n):
        assert_equal(group_rows[row], canonical_to_device[equality.ids[row]])
        assert_equal(row_phase[row], wanted.row_phase[row])
        assert_equal(ordinal[row], wanted.ordinal[row])
    var foreign = Scope(99999)
    with assert_raises(contains="different query scope"):
        _ = build(source, keys, foreign, configured)
    assert_false(foreign.initialized())
    _ = schedule^
    assert_equal(scope.info()[0], live)
    _ = source^
    _ = setting^
    print(
        "PASS resident OVER owner",
        kind,
        n,
        configured,
        allow_contiguous,
        cached_layout,
    )


def _invalid_lookup_and_recovery() raises:
    var scope = Scope()
    var source = _stage(
        DataFrame(
            [
                Series("k", Column[UInt32]([0, 2, 1])),
            ]
        ),
        scope,
    )
    var key = WindowKey("k", DataType.STRING, True)
    var lookup: List[UInt64] = [1, 2]
    key.values = ArcPointer(lookup^)
    var keys: List[KeyInput] = [KeyInput("k", key)]
    with assert_raises(contains="key lookup or source value is invalid"):
        _ = build(source, keys, scope, 1)
    assert_equal(scope.transfers()[1], 4 * 3 + 16)
    var good = _stage(
        DataFrame(
            [
                Series("k", Column[UInt32]([0, 1, 0])),
            ]
        ),
        scope,
    )
    var recovered = build(good, keys, scope, 1)
    assert_equal(recovered[].groups, 2)
    assert_equal(recovered[].facts[2], UInt32(1))
    with assert_raises(contains="resource geometry"):
        _ = estimate_bytes(0, 1, 0, 1, 1)
    with assert_raises(contains="worker count"):
        _ = build(good, keys, scope, 0)


def run_probe() raises:
    for configured in [1, 4]:
        for allow_contiguous in [False, True]:
            for kind in range(7):
                _check(
                    kind,
                    131073 if kind == 5 else 8192,
                    configured,
                    allow_contiguous,
                )
    # A later/nested OVER on these keys reuses an earlier sibling's layout
    # even though its own child disables contiguous execution.
    _check(3, 8192, 4, False, cached_layout=True)
    _check(5, 8192, 4, False, cached_layout=True)
    _invalid_lookup_and_recovery()
    print("PASS30 resident OVER owners plus lookup guards/recovery")
