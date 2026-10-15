"""Metadata-only planning for joins between bounded accelerator row regions.

Every reachable branch is bound before device initialization. Child regions
are not terminal boundaries for checked expressions. Executing these tasks
may materialize intermediate frames; the provider reports those boundaries.
"""
from .metadata import prune_projections, schema_frame
from .qcuts import is_qcut_marker, controls as qcut_controls
from .capabilities import RowCapabilities
from .rows import lower_rows
from .explodes import schema as explode_schema
from dataframe.dtype import DataType
from dataframe.frame import DataFrame
from dataframe.lazy import (
    LazyFrame,
    SCAN_FRAME,
    SELECT,
    WITH_COLUMNS,
    FILTER,
    SLICE,
    DROP,
    SORT,
    AGG,
    JOIN,
    ASOF,
    EXPLODE,
)
from dataframe.series import Series
from dataframe.expr import is_reduction, ADD, SUB, MUL, DIV, NEG, ABS
from dataframe.binding import bind, ROWS
from .integer_ranges import proven_integer_nodes
from . import ordered_rows, implode_ordered

comptime DECIMAL_MATERIALIZATION = "metal_decimal_materialization"


def _decimal_producer(query: LazyFrame, index: Int) raises -> Bool:
    ref operation = query._nodes[index]
    if operation.kind != SELECT and operation.kind != WITH_COLUMNS:
        return False
    var possible = False
    for expression in operation.exprs:
        for node in expression._nodes:
            possible |= node.op in [ADD, SUB, MUL, DIV, NEG, ABS]
    if not possible:
        return False
    var schema = schema_frame(query, operation.left)
    var checked = False
    for expression in operation.exprs:
        var bound = bind(expression, schema._columns)
        if bound.shape() != ROWS or not ordered_rows.eligible(bound):
            return False
        var proven = proven_integer_nodes(bound)
        for at in range(len(bound.expr._nodes)):
            var op = bound.expr._nodes[at].op
            checked |= (
                bound.dtypes[at].is_decimal()
                and (
                    op == ADD
                    or op == SUB
                    or op == MUL
                    or op == DIV
                    or op == NEG
                    or op == ABS
                )
                and not proven[at]
            )
    return checked


def has_decimal_consumers(query: LazyFrame) raises -> Bool:
    # A region boundary observes each expression before any later row
    # selection, join, ordering or stateful consumer can suppress its fault.
    for index in range(len(query._nodes)):
        if _decimal_producer(query, index):
            return True
    return False


def _mark_decimal_producers(mut query: LazyFrame) raises:
    for index in range(len(query._nodes)):
        if not query._nodes[index].text and _decimal_producer(query, index):
            var schema = schema_frame(query, query._nodes[index].left)
            query._nodes[index].text = DECIMAL_MATERIALIZATION
            query._nodes[index].names2 = [DECIMAL_MATERIALIZATION]
            for expression in query._nodes[index].exprs:
                query._nodes[index].names2.append(
                    implode_ordered.signature(bind(expression, schema._columns))
                )


comptime DAG_ROWS = 0
comptime DAG_JOIN = 1
comptime DAG_ASOF = 2
comptime DAG_SOURCE_FIELDS = 3
comptime DAG_EXPLODE = 4


def _scalar_reduction(query: LazyFrame, index: Int) -> Bool:
    ref operation = query._nodes[index]
    if operation.kind != SELECT:
        return False
    for expression in operation.exprs:
        if len(expression._nodes) and is_reduction(
            expression._nodes[len(expression._nodes) - 1].op
        ):
            return True
    return False


def has_scalar_consumers(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        if operation.left >= 0 and _scalar_reduction(query, operation.left):
            return True
        if operation.right >= 0 and _scalar_reduction(query, operation.right):
            return True
    return False


@fieldwise_init
struct JoinField(Copyable):
    var name: String
    var dtype: DataType
    var left: Int
    var right: Int
    var prefer_right: Bool


@fieldwise_init
struct DagTask(Copyable):
    var kind: Int
    var root: Int
    var query: LazyFrame
    var source: Int
    var right: Int
    var schema: DataFrame
    var terminal: Bool
    var how: Int
    var limit: Int
    var keys: List[Int64]
    var fields: List[JoinField]
    var equal_rows: Bool


@fieldwise_init
struct JoinDag(Movable):
    var tasks: List[DagTask]
    var root: Int
    var joins: Int


def has_join(query: LazyFrame) -> Bool:
    for node in query._nodes:
        if node.kind == JOIN or node.kind == ASOF:
            return True
    return False


def _validate_sources(query: LazyFrame, capabilities: RowCapabilities) raises:
    if len(query._nodes) == 0:
        capabilities.reject("plan", "empty plan")
    var pending = List[Int]([len(query._nodes) - 1])
    var visited = List[Bool](length=len(query._nodes), fill=False)
    while len(pending):
        var index = pending.pop()
        if index < 0 or index >= len(query._nodes):
            capabilities.reject("plan", "invalid child index")
        if visited[index]:
            continue
        visited[index] = True
        ref node = query._nodes[index]
        if node.kind == SCAN_FRAME:
            if node.offset < 0 or node.offset >= len(query._frames):
                capabilities.reject("plan", "invalid in-memory source")
            continue
        if (
            node.kind != JOIN
            and node.kind != ASOF
            and node.kind != SELECT
            and node.kind != WITH_COLUMNS
            and node.kind != FILTER
            and node.kind != SLICE
            and node.kind != DROP
            and node.kind != SORT
            and node.kind != AGG
            and node.kind != EXPLODE
        ):
            capabilities.reject(
                "plan",
                "join regions require in-memory scans and supported row steps",
            )
        if node.kind == EXPLODE and not capabilities.primitive_list_explode:
            capabilities.reject(
                "explode", "primitive List expansion is unavailable"
            )
        if node.kind == SLICE and node.length < -1:
            raise Error("Slice length must be nonnegative")
        # Public plans are topological; check this before projection analysis,
        # which otherwise can read a file or follow an invalid/cyclic branch.
        if node.left < 0 or node.left >= index:
            capabilities.reject(
                "plan", "join regions require a topological DAG"
            )
        pending.append(node.left)
        if node.kind == JOIN or node.kind == ASOF:
            if node.right < 0 or node.right >= index:
                capabilities.reject(
                    "plan", "join regions require two input branches"
                )
            pending.append(node.right)


def _region_query(
    query: LazyFrame,
    chain: List[Int],
    source: DataFrame,
    source_failure: String = "",
) -> LazyFrame:
    var region = LazyFrame(source)
    region._nodes[0].source_failure = source_failure
    for reverse in range(len(chain)):
        var node = query._nodes[chain[len(chain) - reverse - 1]].copy()
        node.left = len(region._nodes) - 1
        node.right = -1
        region._nodes.append(node^)
    return region^


def _join_task(
    query: LazyFrame,
    root: Int,
    terminal: Bool,
    limit: Int,
    capabilities: RowCapabilities,
    mut tasks: List[DagTask],
) raises -> Int:
    ref node = query._nodes[root]
    if node.how < 0 or node.how > 6:
        raise Error(
            "Join how must be inner, left, right, full, semi, anti, or cross"
        )
    if node.how == 6:
        if len(node.names) or len(node.right_keys):
            raise Error(
                "A cross join takes no keys; use join(right, how='cross')"
            )
    elif len(node.names) == 0 or len(node.names) != len(node.right_keys):
        raise Error(
            "Join requires the same nonzero number of left and right keys"
        )
    var equal_rows = len(node.flags) > 1 and node.flags[1]
    var left_id = _task(query, node.left, equal_rows, capabilities, tasks)
    var right_id = _task(query, node.right, equal_rows, capabilities, tasks)
    var left = tasks[left_id].schema.copy()
    var right = tasks[right_id].schema.copy()
    if left.width() + right.width() > capabilities.max_join_columns:
        capabilities.reject(
            "plan",
            "join supports at most "
            + String(capabilities.max_join_columns)
            + " combined input columns",
        )
    var keys = List[Int64]()
    var left_keys = List[Int]()
    var right_keys = List[Int]()
    var seen = List[String]()
    for i in range(len(node.names)):
        if node.names[i] in seen:
            raise Error("Duplicate join key: " + node.names[i])
        seen.append(node.names[i])
        var l = left._index(node.names[i])
        var r = right._index(node.right_keys[i])
        var lt = left._columns[l].dtype()
        var rt = right._columns[r].dtype()
        capabilities.require_key(lt)
        capabilities.require_key(rt)
        if lt != rt:
            raise Error(
                "Join key dtypes differ: "
                + node.names[i]
                + " is "
                + lt.name()
                + " but "
                + node.right_keys[i]
                + " is "
                + rt.name()
            )
        left_keys.append(l)
        right_keys.append(r)
        keys.extend([Int64(l), Int64(left.width() + r)])
    var fields = List[JoinField]()
    var names = left.columns()
    var keep_right_keys = not node.coalesce
    for c in range(left.width()):
        var fallback = -1
        if not keep_right_keys and (node.how == 2 or node.how == 3):
            for k in range(len(left_keys)):
                if left_keys[k] == c:
                    fallback = left.width() + right_keys[k]
        fields.append(
            JoinField(
                left._columns[c].name(),
                left._columns[c].dtype(),
                c,
                fallback,
                node.how == 2 and fallback >= 0,
            )
        )
    if node.how != 4 and node.how != 5:
        var suffix = node.names2[0] if len(node.names2) else String("_right")
        var left_names = names.copy()
        for c in range(right.width()):
            if not keep_right_keys and c in right_keys:
                continue
            var name = right._columns[c].name()
            if name in left_names:
                name += suffix
            if name in names:
                raise Error("Join output name collision: " + name)
            names.append(name)
            fields.append(
                JoinField(
                    name, right._columns[c].dtype(), -1, left.width() + c, False
                )
            )
    if len(fields) == 0 or len(fields) > capabilities.max_join_columns:
        capabilities.reject(
            "plan",
            "join requires 1 to "
            + String(capabilities.max_join_columns)
            + " output columns",
        )
    var schema = List[Series]()
    for field in fields:
        capabilities.require_dtype(field.dtype)
        var column = Series.full_null(field.name, field.dtype, 0)
        # Match CPU output ownership: RIGHT coalesced keys come from the
        # right input; FULL coalesced keys are newly chosen values whose
        # metadata is cleared by expr_kernels.choose. Other fields transport
        # their source metadata even when the gathered values are all null.
        if field.prefer_right or field.left < 0:
            column._field_metadata = right._columns[
                field.right - left.width()
            ]._field_metadata
        elif not (node.how == 3 and field.right >= 0):
            column._field_metadata = left._columns[field.left]._field_metadata
        schema.append(column^)
    var frame = DataFrame(schema^)
    var metadata = frame.lazy()
    metadata._nodes[0].code_domain = node.code_domain
    metadata._nodes[0].text_fields = node.text_fields.copy()
    var id = len(tasks)
    tasks.append(
        DagTask(
            DAG_JOIN,
            root,
            metadata^,
            left_id,
            right_id,
            frame^,
            terminal,
            7 if node.how == 1
            and len(node.flags)
            and node.flags[0] else node.how,
            limit,
            keys^,
            fields^,
            equal_rows,
        )
    )
    return id


def _asof_task(
    query: LazyFrame,
    root: Int,
    terminal: Bool,
    limit: Int,
    capabilities: RowCapabilities,
    mut tasks: List[DagTask],
) raises -> Int:
    if not capabilities.asof_joins:
        capabilities.reject("plan", "as-of joins require backend support")
    ref node = query._nodes[root]
    if len(node.names) != 1 or len(node.right_keys) != 1 or not len(node.flags):
        raise Error("join_asof requires one ordered key on each side")
    # CPU materializes and observes complete left then right producers before
    # validating sortedness. A final head cannot hide producer value errors.
    var left_id = _task(query, node.left, True, capabilities, tasks)
    var right_id = _task(query, node.right, True, capabilities, tasks)
    var left = tasks[left_id].schema.copy()
    var right = tasks[right_id].schema.copy()
    if left.width() + right.width() > capabilities.max_join_columns:
        capabilities.reject("plan", "as-of join exceeds combined column limit")
    var by = List[String]()
    for index in range(1, len(node.names2)):
        by.append(node.names2[index])
    var suffix = node.names2[0] if len(node.names2) else String("_right")
    # Empty child schemas run the authoritative dtype/options/name checks;
    # this touches no source values and cannot perform a CPU data join.
    var frame = left.join_asof(
        right,
        left_on=node.names[0],
        right_on=node.right_keys[0],
        by=by,
        strategy=node.text,
        tolerance=node.asof_tolerance,
        suffix=suffix,
        allow_exact_matches=node.flags[0],
    )
    capabilities.require_key(left.column(node.names[0]).dtype())
    var keys = List[Int64]()
    for name in by:
        var l = left._index(name)
        var r = right._index(name)
        capabilities.require_key(left._columns[l].dtype())
        keys.extend([Int64(l), Int64(left.width() + r)])
    var fields = List[JoinField]()
    for column in range(left.width()):
        ref field = left._columns[column]
        fields.append(JoinField(field.name(), field.dtype(), column, -1, False))
    var left_names = left.columns()
    for column in range(right.width()):
        ref field = right._columns[column]
        var name = field.name()
        if name in by or (
            name == node.right_keys[0] and node.names[0] == node.right_keys[0]
        ):
            continue
        if name in left_names:
            name += suffix
        fields.append(
            JoinField(name, field.dtype(), -1, left.width() + column, False)
        )
    if not len(fields) or len(fields) > capabilities.max_join_columns:
        capabilities.reject("plan", "as-of join exceeds output column limit")
    for field in fields:
        capabilities.require_dtype(field.dtype)
    var metadata = frame.lazy()
    var marker = node.copy()
    marker.left = 0
    marker.right = 0
    metadata._nodes.append(marker^)
    var id = len(tasks)
    tasks.append(
        DagTask(
            DAG_ASOF,
            root,
            metadata^,
            left_id,
            right_id,
            frame^,
            terminal,
            0 if node.text
            == "backward" else 1 if node.text
            == "forward" else 2,
            limit,
            keys^,
            fields^,
            False,
        )
    )
    return id


def _cached_rows(tasks: List[DagTask], root: Int) -> Int:
    for i in range(len(tasks)):
        if (
            tasks[i].kind == DAG_ROWS or tasks[i].kind == DAG_EXPLODE
        ) and tasks[i].root == root:
            return i
    return -1


def _explode_task(
    query: LazyFrame,
    root: Int,
    capabilities: RowCapabilities,
    mut tasks: List[DagTask],
) raises -> Int:
    ref operation = query._nodes[root]
    ref scan = query._nodes[operation.left]
    if scan.kind != SCAN_FRAME:
        capabilities.reject(
            "explode",
            "initial scoped List producer requires a direct in-memory source",
        )
    if scan.code_domain or len(scan.text_fields):
        capabilities.reject(
            "explode",
            "private encoded source domains are not primitive List sources",
        )
    var source = query._frames[scan.offset].copy()
    if len(scan.names):
        source = source.select(scan.names)
    var schema = explode_schema(source, operation.names, capabilities)
    var source_id = -1
    for index in range(len(tasks)):
        if (
            tasks[index].kind == DAG_SOURCE_FIELDS
            and tasks[index].root == operation.left
        ):
            source_id = index
            break
    if source_id < 0:
        var input = LazyFrame(source)
        input._nodes[0] = scan.copy()
        input._nodes[0].offset = 0
        input._nodes[0].names = List[String]()
        source_id = len(tasks)
        tasks.append(
            DagTask(
                DAG_SOURCE_FIELDS,
                operation.left,
                input^,
                -1,
                -1,
                source.clear(),
                True,
                -1,
                -1,
                List[Int64](),
                List[JoinField](),
                False,
            )
        )
    var metadata = LazyFrame(source.clear())
    var marker = operation.copy()
    marker.left = 0
    marker.right = -1
    metadata._nodes.append(marker^)
    var id = len(tasks)
    # Expansion observes the complete producer and all logical counts before
    # any later projection, filter or head can suppress a cardinality failure.
    tasks.append(
        DagTask(
            DAG_EXPLODE,
            root,
            metadata^,
            source_id,
            -1,
            schema^,
            True,
            -1,
            -1,
            List[Int64](),
            List[JoinField](),
            False,
        )
    )
    return id


def _task(
    query: LazyFrame,
    root: Int,
    terminal: Bool,
    capabilities: RowCapabilities,
    mut tasks: List[DagTask],
) raises -> Int:
    var boundary = (
        terminal
        or query._nodes[root].text == "metal_window_materialization"
        or query._nodes[root].text == "metal_distribution_materialization"
        or query._nodes[root].text == "metal_implode_materialization"
        or query._nodes[root].text == DECIMAL_MATERIALIZATION
    )
    var cached = _cached_rows(tasks, root)
    if cached >= 0:
        if terminal:
            tasks[cached].terminal = True
        return cached
    if query._nodes[root].kind == EXPLODE:
        return _explode_task(query, root, capabilities, tasks)
    if query._nodes[root].kind == JOIN:
        return _join_task(query, root, boundary, -1, capabilities, tasks)
    if query._nodes[root].kind == ASOF:
        return _asof_task(query, root, boundary, -1, capabilities, tasks)
    # A direct head is a join output limit, so large fanout can be bounded
    # without allocating the unobserved Cartesian-sized result.
    ref top = query._nodes[root]
    if (
        top.kind == SLICE
        and top.offset == 0
        and query._nodes[top.left].kind == JOIN
    ):
        return _join_task(
            query, top.left, boundary, top.length, capabilities, tasks
        )
    if (
        top.kind == SLICE
        and top.offset == 0
        and query._nodes[top.left].kind == ASOF
    ):
        return _asof_task(
            query, top.left, boundary, top.length, capabilities, tasks
        )
    var chain = List[Int]()
    var cursor = root
    while (
        query._nodes[cursor].kind != SCAN_FRAME
        and _cached_rows(tasks, cursor) < 0
        and query._nodes[cursor].kind != JOIN
        and query._nodes[cursor].kind != ASOF
        and query._nodes[cursor].kind != EXPLODE
        and (cursor == root or query._nodes[cursor].kind != SLICE)
        and (
            cursor == root or query._nodes[root].text != DECIMAL_MATERIALIZATION
        )
        and (
            cursor == root
            or (
                query._nodes[cursor].text != "metal_window_materialization"
                and query._nodes[cursor].text
                != "metal_distribution_materialization"
                and query._nodes[cursor].text != "metal_implode_materialization"
                and query._nodes[cursor].text != DECIMAL_MATERIALIZATION
            )
        )
        and (
            cursor == root or query._nodes[cursor].text != "metal_shared_prefix"
        )
        and (
            cursor == root
            or not capabilities.scalar_regions
            or not _scalar_reduction(query, cursor)
        )
    ):
        chain.append(cursor)
        cursor = query._nodes[cursor].left
    var source = -1
    var frame: DataFrame
    cached = _cached_rows(tasks, cursor)
    if cached >= 0:
        source = cached
        frame = tasks[source].schema.copy()
    elif (
        query._nodes[cursor].kind == JOIN
        or (
            cursor != root
            and query._nodes[root].text == DECIMAL_MATERIALIZATION
            and query._nodes[cursor].kind != SCAN_FRAME
        )
        or query._nodes[cursor].kind == ASOF
        or query._nodes[cursor].kind == EXPLODE
        or query._nodes[cursor].kind == SLICE
        or (
            cursor != root
            and (
                query._nodes[cursor].text == "metal_window_materialization"
                or query._nodes[cursor].text
                == "metal_distribution_materialization"
                or query._nodes[cursor].text == "metal_implode_materialization"
                or query._nodes[cursor].text == DECIMAL_MATERIALIZATION
            )
        )
        or (
            cursor != root
            and query._nodes[cursor].text == "metal_shared_prefix"
        )
        or (
            capabilities.scalar_regions
            and _scalar_reduction(query, cursor)
            and cursor != root
        )
    ):
        source = _task(query, cursor, False, capabilities, tasks)
        frame = tasks[source].schema.copy()
    else:
        ref scan = query._nodes[cursor]
        frame = query._frames[scan.offset].copy()
        if len(scan.names):
            frame = frame.select(scan.names)
    var region = _region_query(
        query,
        chain,
        frame,
        query._nodes[cursor].source_failure if source < 0 else "",
    )
    region._nodes[0].code_domain = query._nodes[cursor].code_domain
    region._nodes[0].text_fields = query._nodes[cursor].text_fields.copy()
    if is_qcut_marker(region):
        # This private control phase has its own resident kernels/resources.
        # Bind its harmless placeholder only for metadata; never compile or
        # execute that placeholder as a generic row interpreter program.
        _ = qcut_controls(region)
        if source < 0:
            raise Error("QCUT marker requires an observed resident producer")
        var schema = schema_frame(region, len(region._nodes) - 1).clear()
        var id = len(tasks)
        tasks.append(
            DagTask(
                DAG_ROWS,
                root,
                region^,
                source,
                -1,
                schema^,
                True,
                -1,
                -1,
                List[Int64](),
                List[JoinField](),
                False,
            )
        )
        return id
    var plan = lower_rows(region, capabilities, terminal_boundary=boundary)
    # Pure projections preserve the prefix. Push their final head into a join
    # only after binding proves that no reduction or row reordering intervenes.
    var prefix = plan.limit >= 0 and not plan.reductions
    for step in plan.steps:
        prefix &= not step.filter and not step.order and not step.grouped
    if prefix and source >= 0:
        if tasks[source].kind == DAG_JOIN or tasks[source].kind == DAG_ASOF:
            var existing = tasks[source].limit
            tasks[source].limit = (
                min(existing, plan.limit) if existing >= 0 else plan.limit
            )
    var columns = List[Series]()
    for output in plan.outputs:
        var column = Series.full_null(output.name, output.dtype, 0)
        column._field_metadata = output.field_metadata
        columns.append(column^)
    var id = len(tasks)
    tasks.append(
        DagTask(
            DAG_ROWS,
            root,
            region^,
            source,
            -1,
            DataFrame(columns^),
            boundary,
            -1,
            -1,
            List[Int64](),
            List[JoinField](),
            False,
        )
    )
    return id


def lower_join_dag(
    query: LazyFrame, capabilities: RowCapabilities
) raises -> JoinDag:
    """Validate every branch, prune unused inputs, then bind every region."""
    _validate_sources(query, capabilities)
    var projected = query.copy()
    var optimize = (
        not query._accelerator_settings
        or query._accelerator_settings.value().optimize
    )
    if optimize and has_decimal_consumers(query):
        projected._push_slices()
    prune_projections(projected)
    _mark_decimal_producers(projected)
    var tasks = List[DagTask]()
    var root = _task(
        projected, len(projected._nodes) - 1, True, capabilities, tasks
    )
    var joins = 0
    for task in tasks:
        joins += Int(task.kind == DAG_JOIN or task.kind == DAG_ASOF)
    return JoinDag(tasks^, root, joins)
