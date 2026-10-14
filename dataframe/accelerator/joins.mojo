"""Metadata-only planning for joins between bounded accelerator row regions.

Every reachable branch is bound before device initialization. Child regions
are not terminal boundaries for checked expressions. Executing these tasks
may materialize intermediate frames; the provider reports those boundaries.
"""
from .metadata import prune_projections
from .capabilities import RowCapabilities
from .rows import lower_rows
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
)
from dataframe.series import Series
from dataframe.expr import is_reduction

comptime DAG_ROWS = 0
comptime DAG_JOIN = 1


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


@fieldwise_init
struct JoinDag(Movable):
    var tasks: List[DagTask]
    var root: Int
    var joins: Int


def has_join(query: LazyFrame) -> Bool:
    for node in query._nodes:
        if node.kind == JOIN:
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
            and node.kind != SELECT
            and node.kind != WITH_COLUMNS
            and node.kind != FILTER
            and node.kind != SLICE
            and node.kind != DROP
            and node.kind != SORT
            and node.kind != AGG
        ):
            capabilities.reject(
                "plan",
                "join regions require in-memory scans and supported row steps",
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
        if node.kind == JOIN:
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
    var left_id = _task(query, node.left, False, capabilities, tasks)
    var right_id = _task(query, node.right, False, capabilities, tasks)
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
        schema.append(Series.full_null(field.name, field.dtype, 0))
    var frame = DataFrame(schema^)
    var id = len(tasks)
    tasks.append(
        DagTask(
            DAG_JOIN,
            root,
            frame.lazy(),
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
        )
    )
    return id


def _cached_rows(tasks: List[DagTask], root: Int) -> Int:
    for i in range(len(tasks)):
        if tasks[i].kind == DAG_ROWS and tasks[i].root == root:
            return i
    return -1


def _task(
    query: LazyFrame,
    root: Int,
    terminal: Bool,
    capabilities: RowCapabilities,
    mut tasks: List[DagTask],
) raises -> Int:
    var boundary = (
        terminal or query._nodes[root].text == "metal_window_materialization"
    )
    var cached = _cached_rows(tasks, root)
    if cached >= 0:
        return cached
    if query._nodes[root].kind == JOIN:
        return _join_task(query, root, boundary, -1, capabilities, tasks)
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
    var chain = List[Int]()
    var cursor = root
    while (
        query._nodes[cursor].kind != SCAN_FRAME
        and _cached_rows(tasks, cursor) < 0
        and query._nodes[cursor].kind != JOIN
        and (cursor == root or query._nodes[cursor].kind != SLICE)
        and (
            cursor == root
            or query._nodes[cursor].text != "metal_window_materialization"
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
        or query._nodes[cursor].kind == SLICE
        or (
            cursor != root
            and query._nodes[cursor].text == "metal_window_materialization"
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
    var plan = lower_rows(region, capabilities, terminal_boundary=boundary)
    # Pure projections preserve the prefix. Push their final head into a join
    # only after binding proves that no reduction or row reordering intervenes.
    var prefix = plan.limit >= 0 and not plan.reductions
    for step in plan.steps:
        prefix &= not step.filter and not step.order and not step.grouped
    if prefix and source >= 0:
        if tasks[source].kind == DAG_JOIN:
            var existing = tasks[source].limit
            tasks[source].limit = (
                min(existing, plan.limit) if existing >= 0 else plan.limit
            )
    var columns = List[Series]()
    for output in plan.outputs:
        columns.append(Series.full_null(output.name, output.dtype, 0))
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
        )
    )
    return id


def lower_join_dag(
    query: LazyFrame, capabilities: RowCapabilities
) raises -> JoinDag:
    """Validate every branch, prune unused inputs, then bind every region."""
    _validate_sources(query, capabilities)
    var projected = query.copy()
    prune_projections(projected)
    var tasks = List[DagTask]()
    var root = _task(
        projected, len(projected._nodes) - 1, True, capabilities, tasks
    )
    var joins = 0
    for task in tasks:
        joins += Int(task.kind == DAG_JOIN)
    return JoinDag(tasks^, root, joins)
