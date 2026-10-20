"""Metadata-only planning for joins between bounded accelerator row regions.

Every reachable branch is bound before device initialization. Child regions
are not terminal boundaries for checked expressions. Executing these tasks
may materialize intermediate frames; the provider reports those boundaries.
"""
from std.memory import ArcPointer
from .metadata import prune_projections, schema_frame
from .selectors import _live_nodes
from .grouped_lists import GroupedListsPlan, prepare as prepare_grouped_lists
from .qcuts import is_qcut_marker, controls as qcut_controls
from .cuts import has_cut_materializations
from .partition_cuts import is_marker as is_partition_cut_marker, marker_capture
from .capabilities import RowCapabilities
from .rows import lower_rows
from . import (
    producer_phases,
    decimal_feed,
    decimal_group_feed,
    decimal_scalar_feed,
    decimal_sum_receipts,
    decimal_row_feed,
)
from .decimal_prefix import prefix_infallible
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
from dataframe.expr import (
    is_reduction,
    is_window,
    OVER,
    ADD,
    SUB,
    MUL,
    DIV,
    NEG,
    ABS,
    FLOORDIV,
    POW,
    SUM,
    DT_YEAR,
    COL,
    CAST,
    MOD,
    subtree,
)
from dataframe.binding import bind, BoundExpr, ROWS
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


def _decimal_date_producer(query: LazyFrame, index: Int) raises -> Bool:
    ref operation = query._nodes[index]
    if operation.kind != SELECT and operation.kind != WITH_COLUMNS:
        return False
    var year = False
    for expression in operation.exprs:
        for part in expression._nodes:
            year |= part.op == DT_YEAR
    if not year or not prefix_infallible(query, operation.left):
        return False
    if not decimal_row_feed.consumers_infallible(query, index):
        return False
    var schema = schema_frame(query, operation.left)
    var bounds = List[BoundExpr]()
    for expression in operation.exprs:
        bounds.append(bind(expression, schema._columns))
    return decimal_row_feed.eligible(bounds)


def _decimal_feed_producer(query: LazyFrame, index: Int) raises -> Bool:
    ref operation = query._nodes[index]
    # Eager SELECT uses frame.select_exprs' 8192-row operand batches. Streaming
    # morsels and group/OVER feeds require their own captured phase geometry.
    if (
        operation.kind != SELECT
        or len(operation.exprs) != 1
        or not query._accelerator_settings
        or query._accelerator_settings.value().streaming
    ):
        return False
    if (
        not len(operation.exprs[0]._nodes)
        or operation.exprs[0]._nodes[len(operation.exprs[0]._nodes) - 1].op
        != SUM
    ):
        return False
    var possible = False
    for node in operation.exprs[0]._nodes:
        possible |= node.op in [ADD, SUB, MUL, DIV, NEG, ABS]
    if not possible:
        return False
    var schema = schema_frame(query, operation.left)
    return decimal_feed.eligible(
        bind(operation.exprs[0], schema._columns), schema._columns
    )


def _decimal_group_producer(query: LazyFrame, index: Int) raises -> Bool:
    return decimal_group_feed.producer(query, index)


def has_decimal_consumers(query: LazyFrame) raises -> Bool:
    # A region boundary observes each expression before any later row
    # selection, join, ordering or stateful consumer can suppress its fault.
    for index in range(len(query._nodes)):
        if (
            _decimal_producer(query, index)
            or _decimal_date_producer(query, index)
            or _decimal_feed_producer(query, index)
            or _decimal_group_producer(query, index)
            or _decimal_scalar_producer(query, index)
        ):
            return True
    return False


def _decimal_scalar_producer(query: LazyFrame, index: Int) raises -> Bool:
    ref operation = query._nodes[index]
    if operation.kind != SELECT or not len(operation.exprs):
        return False
    for expression in operation.exprs:
        if (
            not len(expression._nodes)
            or expression._nodes[len(expression._nodes) - 1].op != SUM
        ):
            return False
    if not prefix_infallible(query, operation.left):
        return False
    var schema = schema_frame(query, operation.left)
    var bounds = List[BoundExpr]()
    for expression in operation.exprs:
        bounds.append(bind(expression, schema._columns))
    return decimal_scalar_feed.eligible(bounds, schema._columns)


def _mark_decimal_date_producers(mut query: LazyFrame) raises:
    # Issue this proof on the retained original graph before input pruning can
    # replace a JOIN arm with a synthetic selected/normalized source region.
    for index in range(len(query._nodes)):
        if not query._nodes[index].text and _decimal_date_producer(
            query, index
        ):
            var schema = schema_frame(query, query._nodes[index].left)
            query._nodes[index].text = decimal_row_feed.MATERIALIZATION
            query._nodes[index].names2 = [decimal_row_feed.MATERIALIZATION]
            for expression in query._nodes[index].exprs:
                query._nodes[index].names2.append(
                    implode_ordered.signature(bind(expression, schema._columns))
                )


def _mark_decimal_producers(mut query: LazyFrame) raises:
    for index in range(len(query._nodes)):
        if not query._nodes[index].text and _decimal_scalar_producer(
            query, index
        ):
            var schema = schema_frame(query, query._nodes[index].left)
            query._nodes[index].text = decimal_scalar_feed.MATERIALIZATION
            query._nodes[index].names2 = [decimal_scalar_feed.MATERIALIZATION]
            for expression in query._nodes[index].exprs:
                query._nodes[index].names2.append(
                    implode_ordered.signature(bind(expression, schema._columns))
                )
        if not query._nodes[index].text and _decimal_group_producer(
            query, index
        ):
            var schema = schema_frame(query, query._nodes[index].left)
            query._nodes[index].text = decimal_group_feed.MATERIALIZATION
            query._nodes[index].names2 = [
                decimal_group_feed.MATERIALIZATION,
                String(query._nodes[index].maintain_order),
            ]
            for name in query._nodes[index].names:
                query._nodes[index].names2.append(name)
            for expression in query._nodes[index].exprs:
                query._nodes[index].names2.append(
                    implode_ordered.signature(bind(expression, schema._columns))
                )
        if not query._nodes[index].text and _decimal_feed_producer(
            query, index
        ):
            var schema = schema_frame(query, query._nodes[index].left)
            query._nodes[index].text = decimal_feed.MATERIALIZATION
            query._nodes[index].names2 = [
                decimal_feed.MATERIALIZATION,
                implode_ordered.signature(
                    bind(query._nodes[index].exprs[0], schema._columns)
                ),
            ]
        if not query._nodes[index].text and _decimal_producer(query, index):
            var schema = schema_frame(query, query._nodes[index].left)
            query._nodes[index].text = DECIMAL_MATERIALIZATION
            query._nodes[index].names2 = [DECIMAL_MATERIALIZATION]
            for expression in query._nodes[index].exprs:
                query._nodes[index].names2.append(
                    implode_ordered.signature(bind(expression, schema._columns))
                )


comptime INTEGER_MATERIALIZATION = "metal_integer_materialization"
comptime GROUP_MATERIALIZATION = "metal_group_materialization"
comptime SUM_MATERIALIZATION = "metal_sum_materialization"


def _checked_scalar_sum(query: LazyFrame, index: Int) raises -> Bool:
    ref operation = query._nodes[index]
    if operation.kind != SELECT or not len(operation.exprs):
        return False
    var possible = False
    for expression in operation.exprs:
        if not len(expression._nodes):
            return False
        ref root = expression._nodes[len(expression._nodes) - 1]
        if not is_reduction(root.op):
            return False
        possible |= root.op == SUM
    if not possible:
        return False
    var schema = schema_frame(query, operation.left)
    var checked = False
    for expression in operation.exprs:
        var bound = bind(expression, schema._columns)
        var top = len(bound.expr._nodes) - 1
        ref root = bound.expr._nodes[top]
        if root.left < 0 or root.right >= 0 or bound.shapes[root.left] != ROWS:
            return False
        var operand = bind(subtree(bound.expr, root.left), schema._columns)
        if (
            not ordered_rows.eligible(operand)
            or operand.dtypes[len(operand.dtypes) - 1].is_decimal()
        ):
            return False
        var proven = proven_integer_nodes(bound)
        # CPU completes each aggregate before evaluating the next operand.
        # Preparing infallible operands together cannot reorder checked faults.
        for at in range(top):
            ref node = bound.expr._nodes[at]
            if is_reduction(node.op):
                return False
            # A Decimal source converted to Float64 is an infallible operand,
            # while Decimal arithmetic and Decimal reductions keep their own
            # observation contracts.
            if bound.dtypes[at].is_decimal() and node.op != COL:
                return False
            var float_cast = (
                node.op == CAST
                and node.left >= 0
                and bound.dtypes[at] == DataType.FLOAT64
                and bound.dtypes[node.left].is_numeric()
            )
            if (
                node.op == CAST
                and node.integer
                and not proven[at]
                and not float_cast
            ):
                return False
            if (
                bound.dtypes[at].is_integer()
                and node.op
                in [ADD, SUB, MUL, DIV, FLOORDIV, MOD, POW, NEG, ABS]
                and not proven[at]
            ):
                return False
        var input = bound.dtypes[root.left]
        checked |= root.op == SUM and (
            bound.dtypes[top] == DataType.INT32
            or bound.dtypes[top] == DataType.UINT32
            or input.physical() == DataType.INT64
            or input == DataType.UINT64
        )
    return checked


def _optimize_scalar_sums(mut query: LazyFrame, optimize: Bool) raises:
    if not optimize:
        return
    var live = _live_nodes(query)
    for index in range(len(query._nodes) - 1):
        if live[index] and _checked_scalar_sum(query, index):
            # Canonical CPU predicate folding includes constant-false branch
            # elimination. Issue observation markers only for surviving work.
            # Its empty-schema execution reads no source row payload.
            query._push_predicates()
            return


def _mark_scalar_sums(mut query: LazyFrame) raises:
    var live = _live_nodes(query)
    for index in range(len(query._nodes) - 1):
        if (
            live[index]
            and not query._nodes[index].text
            and _checked_scalar_sum(query, index)
        ):
            var schema = schema_frame(query, query._nodes[index].left)
            query._nodes[index].text = SUM_MATERIALIZATION
            query._nodes[index].names2 = [SUM_MATERIALIZATION]
            for expression in query._nodes[index].exprs:
                query._nodes[index].names2.append(
                    implode_ordered.signature(bind(expression, schema._columns))
                )


def _integer_producer(query: LazyFrame, index: Int) raises -> Bool:
    ref operation = query._nodes[index]
    if operation.kind != SELECT and operation.kind != WITH_COLUMNS:
        return False
    # Reject infallible consumers before executing ancestor metadata. CUT/QCUT
    # control failures are deferred until their observed operand completes.
    var possible = False
    for expression in operation.exprs:
        for node in expression._nodes:
            if is_reduction(node.op) or is_window(node.op) or node.op == OVER:
                return False
            possible |= node.op in [ADD, SUB, MUL, NEG, ABS, FLOORDIV, POW]
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
            checked |= (
                bound.dtypes[at].is_integer()
                and bound.expr._nodes[at].op
                in [ADD, SUB, MUL, NEG, ABS, FLOORDIV, POW]
                and not proven[at]
            )
    return checked


def _checked_group(query: LazyFrame, index: Int) raises -> Bool:
    ref operation = query._nodes[index]
    if operation.kind != AGG:
        return False
    var possible = False
    for expression in operation.exprs:
        possible |= (
            len(expression._nodes) > 0
            and expression._nodes[len(expression._nodes) - 1].op == SUM
        )
    if not possible:
        return False
    var schema = schema_frame(query, operation.left)
    # Match lower_rows' checked-SUM families. Widened narrow integer sums
    # already have an exact accumulator and retain their fused consumer region.
    for expression in operation.exprs:
        var bound = bind(expression, schema._columns)
        var last = len(bound.expr._nodes) - 1
        if (
            last >= 0
            and bound.expr._nodes[last].op == SUM
            and bound.dtypes[last].is_integer()
            and (
                bound.dtypes[last] == DataType.INT32
                or bound.dtypes[last] == DataType.UINT32
                or (
                    not bound.dtypes[bound.expr._nodes[last].left].is_decimal()
                    and (
                        bound.dtypes[bound.expr._nodes[last].left].physical()
                        == DataType.INT64
                        or bound.dtypes[bound.expr._nodes[last].left]
                        == DataType.UINT64
                    )
                )
            )
        ):
            return True
    return False


def has_checked_integer_consumers(query: LazyFrame) raises -> Bool:
    var live = _live_nodes(query)
    for index in range(len(query._nodes) - 1):
        if live[index] and (
            _integer_producer(query, index)
            or _checked_group(query, index)
            or _checked_scalar_sum(query, index)
        ):
            return True
    return False


def _observed_marker(text: String) -> Bool:
    return (
        text == DECIMAL_MATERIALIZATION
        or text == decimal_feed.MATERIALIZATION
        or text == decimal_group_feed.MATERIALIZATION
        or text == decimal_scalar_feed.MATERIALIZATION
        or text == decimal_row_feed.MATERIALIZATION
        or text == INTEGER_MATERIALIZATION
        or text == GROUP_MATERIALIZATION
        or text == SUM_MATERIALIZATION
    )


def _mark_integer_producers(mut query: LazyFrame) raises:
    var live = _live_nodes(query)
    # The public terminal already observes its own errors. Cut only producers
    # with a later consumer; a head remains in that later consumer's region.
    for index in range(len(query._nodes) - 1):
        if not live[index] or query._nodes[index].text:
            continue
        var grouped = _checked_group(query, index)
        if not grouped and not _integer_producer(query, index):
            continue
        var schema = schema_frame(query, query._nodes[index].left)
        var marker = (
            GROUP_MATERIALIZATION if grouped else INTEGER_MATERIALIZATION
        )
        query._nodes[index].text = marker
        query._nodes[index].names2 = [marker]
        if grouped:
            query._nodes[index].names2.append(
                String(query._nodes[index].maintain_order)
            )
            for name in query._nodes[index].names:
                query._nodes[index].names2.append(name)
        for expression in query._nodes[index].exprs:
            query._nodes[index].names2.append(
                implode_ordered.signature(bind(expression, schema._columns))
            )


def prepare_temporal_prefix(
    query: LazyFrame, optimize: Bool
) raises -> LazyFrame:
    """Finalize existing CPU-equivalent bridges before temporal signatures."""
    var result = query.copy()
    _optimize_scalar_sums(result, optimize)
    _mark_decimal_date_producers(result)
    if optimize:
        result._push_slices()
        prune_projections(result)
    _mark_decimal_producers(result)
    _mark_integer_producers(result)
    _mark_scalar_sums(result)
    return result^


comptime DAG_ROWS = 0
comptime DAG_JOIN = 1
comptime DAG_ASOF = 2
comptime DAG_SOURCE_FIELDS = 3
comptime DAG_EXPLODE = 4
comptime DAG_GROUPED_FIELDS = 5


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
    # Keyed joins may opt into matching null keys. Keep this separately from
    # `how`, since left-join null matching has a legacy internal how code.
    var nulls_equal: Bool
    var limit: Int
    var keys: List[Int64]
    var fields: List[JoinField]
    var equal_rows: Bool
    var grouped: Optional[ArcPointer[GroupedListsPlan]]
    var sum_receipt: Optional[ArcPointer[decimal_sum_receipts.Receipt]]


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
    source_task: Int = -1,
) raises -> LazyFrame:
    var region = LazyFrame(source)
    region._nodes[0].source_failure = source_failure
    for reverse in range(len(chain)):
        var node = query._nodes[chain[len(chain) - reverse - 1]].copy()
        node.left = len(region._nodes) - 1
        node.right = -1
        if node.producer_origin:
            node.producer_region = producer_phases.bind_region(
                query, chain[len(chain) - reverse - 1], source, source_task
            )
        region._nodes.append(node^)
    return region^


def _join_task(
    query: LazyFrame,
    root: Int,
    terminal: Bool,
    limit: Int,
    capabilities: RowCapabilities,
    mut tasks: List[DagTask],
    *,
    sum_origins: List[Optional[ArcPointer[decimal_sum_receipts.Origin]]] = List[
        Optional[ArcPointer[decimal_sum_receipts.Origin]]
    ](),
    ordered_source: Bool = False,
    eager_feed: Bool = False,
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
    var left_id = _task(
        query,
        node.left,
        equal_rows,
        capabilities,
        tasks,
        sum_origins=sum_origins,
        ordered_source=ordered_source,
        eager_feed=eager_feed,
    )
    var right_id = _task(
        query,
        node.right,
        equal_rows,
        capabilities,
        tasks,
        sum_origins=sum_origins,
        ordered_source=ordered_source,
        eager_feed=eager_feed,
    )
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
            len(node.flags) > 0 and node.flags[0],
            limit,
            keys^,
            fields^,
            equal_rows,
            None,
            None,
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
    *,
    sum_origins: List[Optional[ArcPointer[decimal_sum_receipts.Origin]]] = List[
        Optional[ArcPointer[decimal_sum_receipts.Origin]]
    ](),
    ordered_source: Bool = False,
    eager_feed: Bool = False,
) raises -> Int:
    if not capabilities.asof_joins:
        capabilities.reject("plan", "as-of joins require backend support")
    ref node = query._nodes[root]
    if len(node.names) != 1 or len(node.right_keys) != 1 or not len(node.flags):
        raise Error("join_asof requires one ordered key on each side")
    # CPU materializes and observes complete left then right producers before
    # validating sortedness. A final head cannot hide producer value errors.
    var left_id = _task(
        query,
        node.left,
        True,
        capabilities,
        tasks,
        sum_origins=sum_origins,
        ordered_source=ordered_source,
        eager_feed=eager_feed,
    )
    var right_id = _task(
        query,
        node.right,
        True,
        capabilities,
        tasks,
        sum_origins=sum_origins,
        ordered_source=ordered_source,
        eager_feed=eager_feed,
    )
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
            False,
            limit,
            keys^,
            fields^,
            False,
            None,
            None,
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


def _grouped_fields_task(
    query: LazyFrame,
    root: Int,
    capabilities: RowCapabilities,
    mut tasks: List[DagTask],
    *,
    sum_origins: List[Optional[ArcPointer[decimal_sum_receipts.Origin]]] = List[
        Optional[ArcPointer[decimal_sum_receipts.Origin]]
    ](),
    ordered_source: Bool = False,
    eager_feed: Bool = False,
) raises -> Int:
    if not capabilities.grouped or not capabilities.primitive_list_explode:
        capabilities.reject(
            "grouped explode",
            "grouped primitive List publication is unavailable",
        )
    for index in range(len(tasks)):
        if (
            tasks[index].root == root
            and tasks[index].kind == DAG_GROUPED_FIELDS
        ):
            return index
    var producer = query.copy()
    producer._reorder(root)
    producer._compact_scan_slots()
    if len(producer._nodes) < 2 or producer._nodes[0].kind != SCAN_FRAME:
        capabilities.reject(
            "grouped explode", "direct original grouped List producer required"
        )
    var failure = producer._nodes[0].source_failure
    producer._nodes[0].source_failure = ""
    var prepared = prepare_grouped_lists(
        producer, ordered_source=ordered_source, eager_feed=eager_feed
    )
    producer._nodes[0].source_failure = failure
    prepared.scan.source_failure = failure
    prepared.source.source_failure = failure
    if not prepared.original_keys and (
        len(prepared.key_provenance)
        or prepared.source.source_domain
        or len(prepared.source.source_text)
    ):
        capabilities.reject(
            "grouped explode",
            "encoded keys require a retained typed domain owner",
        )
    var schema = prepared.schema.copy()
    var id = len(tasks)
    tasks.append(
        DagTask(
            DAG_GROUPED_FIELDS,
            root,
            producer^,
            -1,
            -1,
            schema^,
            True,
            -1,
            False,
            -1,
            List[Int64](),
            List[JoinField](),
            False,
            ArcPointer(prepared^),
            None,
        )
    )
    return id


def _explode_task(
    query: LazyFrame,
    root: Int,
    capabilities: RowCapabilities,
    mut tasks: List[DagTask],
    *,
    sum_origins: List[Optional[ArcPointer[decimal_sum_receipts.Origin]]] = List[
        Optional[ArcPointer[decimal_sum_receipts.Origin]]
    ](),
    ordered_source: Bool = False,
    eager_feed: Bool = False,
) raises -> Int:
    ref operation = query._nodes[root]
    ref scan = query._nodes[operation.left]
    var source_id = -1
    var source: DataFrame
    if scan.kind == SCAN_FRAME:
        if scan.code_domain or len(scan.text_fields):
            capabilities.reject(
                "explode",
                "private encoded source domains are not primitive List sources",
            )
        source = query._frames[scan.offset].copy()
        if len(scan.names):
            source = source.select(scan.names)
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
                    False,
                    -1,
                    List[Int64](),
                    List[JoinField](),
                    False,
                    None,
                    None,
                )
            )
    else:
        source_id = _grouped_fields_task(
            query,
            operation.left,
            capabilities,
            tasks,
            sum_origins=sum_origins,
            ordered_source=ordered_source,
            eager_feed=eager_feed,
        )
        source = tasks[source_id].schema.copy()
    var schema = explode_schema(source, operation.names, capabilities)
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
            False,
            -1,
            List[Int64](),
            List[JoinField](),
            False,
            None,
            None,
        )
    )
    return id


def _task(
    query: LazyFrame,
    root: Int,
    terminal: Bool,
    capabilities: RowCapabilities,
    mut tasks: List[DagTask],
    *,
    sum_origins: List[Optional[ArcPointer[decimal_sum_receipts.Origin]]] = List[
        Optional[ArcPointer[decimal_sum_receipts.Origin]]
    ](),
    ordered_source: Bool = False,
    eager_feed: Bool = False,
) raises -> Int:
    var boundary = (
        terminal
        or query._nodes[root].text == "metal_window_materialization"
        or query._nodes[root].text == "metal_distribution_materialization"
        or query._nodes[root].text == "metal_implode_materialization"
        or _observed_marker(query._nodes[root].text)
        or Bool(query._nodes[root].producer_origin)
    )
    if query._nodes[root].text in [
        decimal_group_feed.MATERIALIZATION,
        decimal_scalar_feed.MATERIALIZATION,
        decimal_row_feed.MATERIALIZATION,
    ] and not prefix_infallible(query, query._nodes[root].left):
        capabilities.reject(
            "decimal group feed", "original infallible prefix changed"
        )
    if query._nodes[
        root
    ].text == decimal_row_feed.MATERIALIZATION and not decimal_row_feed.consumers_infallible(
        query, root
    ):
        capabilities.reject(
            "decimal date rows", "original infallible consumers changed"
        )
    var cached = _cached_rows(tasks, root)
    if cached >= 0:
        if terminal:
            tasks[cached].terminal = True
        return cached
    if query._nodes[root].kind == EXPLODE:
        return _explode_task(
            query,
            root,
            capabilities,
            tasks,
            sum_origins=sum_origins,
            ordered_source=ordered_source,
            eager_feed=eager_feed,
        )
    if query._nodes[root].kind == JOIN:
        return _join_task(
            query,
            root,
            boundary,
            -1,
            capabilities,
            tasks,
            sum_origins=sum_origins,
            ordered_source=ordered_source,
            eager_feed=eager_feed,
        )
    if query._nodes[root].kind == ASOF:
        return _asof_task(
            query,
            root,
            boundary,
            -1,
            capabilities,
            tasks,
            sum_origins=sum_origins,
            ordered_source=ordered_source,
            eager_feed=eager_feed,
        )
    # A direct head is a join output limit, so large fanout can be bounded
    # without allocating the unobserved Cartesian-sized result.
    ref top = query._nodes[root]
    if (
        top.kind == SLICE
        and top.offset == 0
        and query._nodes[top.left].kind == JOIN
    ):
        return _join_task(
            query,
            top.left,
            boundary,
            top.length,
            capabilities,
            tasks,
            sum_origins=sum_origins,
            ordered_source=ordered_source,
            eager_feed=eager_feed,
        )
    if (
        top.kind == SLICE
        and top.offset == 0
        and query._nodes[top.left].kind == ASOF
    ):
        return _asof_task(
            query,
            top.left,
            boundary,
            top.length,
            capabilities,
            tasks,
            sum_origins=sum_origins,
            ordered_source=ordered_source,
            eager_feed=eager_feed,
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
        and (cursor == root or not _observed_marker(query._nodes[root].text))
        and (cursor == root or not query._nodes[root].producer_origin)
        and (cursor == root or not query._nodes[cursor].producer_origin)
        and (
            cursor == root
            or (
                query._nodes[cursor].text != "metal_window_materialization"
                and query._nodes[cursor].text
                != "metal_distribution_materialization"
                and query._nodes[cursor].text != "metal_implode_materialization"
                and not _observed_marker(query._nodes[cursor].text)
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
            and (
                _observed_marker(query._nodes[root].text)
                or Bool(query._nodes[root].producer_origin)
            )
            and query._nodes[cursor].kind != SCAN_FRAME
        )
        or Bool(query._nodes[cursor].producer_origin)
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
                or _observed_marker(query._nodes[cursor].text)
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
        source = _task(
            query,
            cursor,
            Bool(query._nodes[root].producer_origin),
            capabilities,
            tasks,
            sum_origins=sum_origins,
            ordered_source=ordered_source,
            eager_feed=eager_feed,
        )
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
        source,
    )
    region._nodes[0].code_domain = query._nodes[cursor].code_domain
    region._nodes[0].text_fields = query._nodes[cursor].text_fields.copy()
    if is_qcut_marker(region) or is_partition_cut_marker(region):
        # This private control phase has its own resident kernels/resources.
        # Bind its harmless placeholder only for metadata; never compile or
        # execute that placeholder as a generic row interpreter program.
        if is_partition_cut_marker(region):
            _ = marker_capture(region)
        else:
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
                False,
                -1,
                List[Int64](),
                List[JoinField](),
                False,
                None,
                None,
            )
        )
        return id
    var sum_receipt: Optional[ArcPointer[decimal_sum_receipts.Receipt]] = None
    if (
        source >= 0
        and cursor >= 0
        and cursor < len(sum_origins)
        and sum_origins[cursor]
    ):
        var origin = sum_origins[cursor].value()
        if tasks[source].root == cursor and decimal_sum_receipts.matches(
            origin[], query, cursor
        ):
            sum_receipt = decimal_sum_receipts.bind_receipt(origin, source)
            decimal_sum_receipts.validate(
                sum_receipt.value()[], source, tasks[source].query, region
            )
    var plan = lower_rows(
        region,
        capabilities,
        terminal_boundary=boundary,
        sum_receipt=sum_receipt,
    )
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
            False,
            -1,
            List[Int64](),
            List[JoinField](),
            False,
            None,
            sum_receipt,
        )
    )
    return id


def lower_join_dag(
    query: LazyFrame,
    capabilities: RowCapabilities,
    *,
    ordered_source: Bool = False,
    eager_feed: Bool = False,
) raises -> JoinDag:
    """Validate every branch, prune unused inputs, then bind every region."""
    # Capture actual original GROUP occurrences before projection pruning.
    # UInt32 is the existing admitted GROUP-input row bound, never a base
    # SCAN count (a JOIN may multiply rows). The receipt is only consumed
    # behind the original successful observed GROUP task.
    var original = query.copy()
    var candidates = List[Bool](length=len(query._nodes), fill=False)
    for at in range(len(original._nodes)):
        if _decimal_group_producer(
            original, at
        ) and decimal_sum_receipts.needs_bounds(original, at):
            candidates[at] = True
            for ancestor in producer_phases._ancestors(original, at):
                if not original._nodes[ancestor].lineage_anchor:
                    original._nodes[ancestor].lineage_anchor = ArcPointer(
                        ancestor
                    )
    var sum_origins = List[Optional[ArcPointer[decimal_sum_receipts.Origin]]](
        length=len(query._nodes), fill=None
    )
    for at in range(len(query._nodes)):
        if candidates[at]:
            sum_origins[at] = decimal_sum_receipts.capture(original, at)
    var projected = original.copy()
    var optimize = (
        not query._accelerator_settings
        or query._accelerator_settings.value().optimize
    )
    if not producer_phases.has_certified(projected):
        _optimize_scalar_sums(projected, optimize)
    _validate_sources(projected, capabilities)
    if producer_phases.has_certified(projected):
        # Original-entry normalization records the annotations as part of
        # each immutable prefix. No optimizer or marker may mutate it later.
        for index in range(len(projected._nodes)):
            if projected._nodes[index].producer_origin:
                producer_phases.validate_prefix(projected, index)
    else:
        _mark_decimal_date_producers(projected)
        if (
            optimize
            and not has_cut_materializations(projected)
            and (
                has_decimal_consumers(query)
                or has_checked_integer_consumers(query)
            )
        ):
            # Lowered CUT/QCUT markers are eager observation boundaries.
            projected._push_slices()
        prune_projections(projected)
        _mark_decimal_producers(projected)
        _mark_integer_producers(projected)
        _mark_scalar_sums(projected)
    var mapped_origins = List[
        Optional[ArcPointer[decimal_sum_receipts.Origin]]
    ](length=len(projected._nodes), fill=None)
    for origin in sum_origins:
        if not origin:
            continue
        for at in range(len(projected._nodes)):
            if decimal_sum_receipts.matches(origin.value()[], projected, at):
                mapped_origins[at] = origin.value()
    var tasks = List[DagTask]()
    var root = _task(
        projected,
        len(projected._nodes) - 1,
        True,
        capabilities,
        tasks,
        sum_origins=mapped_origins,
        ordered_source=ordered_source,
        eager_feed=eager_feed,
    )
    var joins = 0
    for task in tasks:
        joins += Int(task.kind == DAG_JOIN or task.kind == DAG_ASOF)
    return JoinDag(tasks^, root, joins)
