"""Carry List and Struct columns through accelerator row movement by row id.

DuckDB moves a nested vector by its row references and leaves the child data
in place; this does the same for nested columns that a plan only passes
through (bare column references moved by filters, sorts, joins, slices and
concatenation). Each such source column is replaced by an Int64 id column of
the same name: `owner << 40 | row`, null where the value is null. The
accelerator moves ids like any other column. The original output schema says
which result columns were nested, and `restore` rebuilds them from their host
owners with `take_or_null`, so every child type (String, nested List or
Struct) is preserved without device transport.

A plan that computes on a nested value (a List/Struct expression, a key, a
predicate, an aggregate, EXPLODE or UNNEST) is not a pass-through and keeps
its existing route.
"""
from dataframe import Column, DataFrame, DataType, Series
from dataframe.binding import bind, BoundExpr
from std.collections import Dict
from dataframe.expr import (
    COL,
    SELECTOR,
    Expr,
    Node,
    LIST_LEN,
    LIST_GET,
    LIST_CONTAINS,
    LIST_JOIN,
    LIST_SUM,
    LIST_MIN,
    LIST_MAX,
    LIST_MEAN,
    STRUCT_FIELD,
    col,
    subtree,
)
from dataframe.lazy import (
    LazyFrame,
    _plan_node,
    SCAN_FRAME,
    SCAN_CSV,
    SCAN_PARQUET,
    FILTER,
    SELECT,
    WITH_COLUMNS,
    AGG,
    JOIN,
    SORT,
    UNIQUE,
    EXPLODE,
    UNNEST,
    ASOF,
)
from dataframe.nested_column import ListColumn, StructColumn
from .metadata import schema_frame
from .faults import data_fault

comptime OWNER_SHIFT = 40
comptime ROW_MASK = (Int64(1) << OWNER_SHIFT) - 1


@fieldwise_init
struct Transport(Movable):
    var query: LazyFrame
    var owners: List[Series]
    var schema: DataFrame


def _has_nested(frame: DataFrame) -> Bool:
    for column in frame._columns:
        if column.dtype().is_nested():
            return True
    return False


def _live(query: LazyFrame) -> List[Bool]:
    var live = List[Bool](length=len(query._nodes), fill=False)
    if not len(query._nodes):
        return live^
    var pending = List[Int]([len(query._nodes) - 1])
    while len(pending):
        var current = pending.pop()
        if current < 0 or live[current]:
            continue
        live[current] = True
        pending.append(query._nodes[current].left)
        pending.append(query._nodes[current].right)
    return live^


def _bare(expression: Expr) -> Bool:
    """A plain column reference, possibly renamed."""
    return len(expression._nodes) == 1 and expression._nodes[0].op == COL


def _named(frame: DataFrame, names: List[String]) -> Bool:
    """Whether any named column of frame is nested."""
    for name in names:
        for column in frame._columns:
            if column.name() == name and column.dtype().is_nested():
                return True
    return False


def _computes_on_nested(expression: Expr, frame: DataFrame) raises -> Bool:
    """Any node other than a bare root reference reads or yields a nested value.

    An expression the binder rejects is not a pass-through either.
    """
    var bound: BoundExpr
    try:
        bound = bind(expression, frame._columns)
    except:
        return True
    if _bare(bound.expr):
        return False
    for index in range(len(bound.dtypes)):
        if bound.dtypes[index].is_nested():
            return True
    return False


def applies(query: LazyFrame) raises -> Bool:
    """Nested source columns exist and every use only moves them."""
    var live = _live(query)
    var nested = False
    for index in range(len(query._nodes)):
        if not live[index]:
            continue
        ref node = query._nodes[index]
        if node.kind == SCAN_FRAME:
            if node.offset >= 0 and node.offset < len(query._frames):
                nested |= _has_nested(query._frames[node.offset])
        elif node.kind == SCAN_CSV or node.kind == SCAN_PARQUET:
            # File scans stage through their own decoder route.
            if _has_nested(schema_frame(query, index)):
                return False
    if not nested:
        return False
    for index in range(len(query._nodes)):
        if not live[index]:
            continue
        ref node = query._nodes[index]
        if node.kind == EXPLODE or node.kind == UNNEST:
            return False
        if node.left < 0:
            continue
        var input = schema_frame(query, node.left)
        if node.kind == JOIN or node.kind == ASOF:
            var right = schema_frame(query, node.right)
            if _named(input, node.names) or _named(right, node.right_keys):
                return False
            continue
        if not _has_nested(input):
            continue
        if node.kind == SORT or node.kind == AGG:
            if _named(input, node.names):
                return False
        if node.kind == UNIQUE:
            # Equal lists under different ids would stay distinct.
            if not len(node.names) or _named(input, node.names):
                return False
        for expression in node.exprs:
            for part in expression._nodes:
                # Selectors expand later; they keep their own route.
                if part.op == SELECTOR:
                    return False
            if _computes_on_nested(expression, input):
                return False
            var bound = bind(expression, input._columns)
            var root = bound.dtypes[len(bound.dtypes) - 1]
            # A bare nested key or grouped value is a computation too.
            if (
                root.is_nested()
                and node.kind != SELECT
                and (node.kind != WITH_COLUMNS)
            ):
                return False
    return True


def _ids(column: Series, owner: Int) raises -> Series:
    var source = column.rechunk() if column.is_chunked() else column.copy()
    var rows = len(source)
    var ids = List[Int64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var base = Int64(owner) << OWNER_SHIFT
    for row in range(rows):
        var present: Bool
        if source.dtype().is_list():
            present = source._data[ListColumn]._valid(row)
        else:
            present = source._data[StructColumn]._valid(row)
        ids.append(base | Int64(row))
        valid.append(present)
    return Series(column.name(), Column[Int64](ids^, valid^))


def lower(query: LazyFrame) raises -> Transport:
    """Replace nested scan columns by ids; record owners and output schema."""
    var schema = schema_frame(query, len(query._nodes) - 1)
    var rewritten = query.copy()
    var owners = List[Series]()
    var done = List[Int]()
    var live = _live(query)
    for index in range(len(query._nodes)):
        ref node = query._nodes[index]
        if not live[index] or node.kind != SCAN_FRAME or node.offset in done:
            continue
        done.append(node.offset)
        var frame = query._frames[node.offset].copy()
        var columns = List[Series](capacity=frame.width())
        for column in frame._columns:
            if column.dtype().is_nested():
                if len(owners) >= (1 << (63 - OWNER_SHIFT)):
                    raise Error("Metal nested transport has too many owners")
                columns.append(_ids(column, len(owners)))
                owners.append(
                    column.rechunk() if column.is_chunked() else column.copy()
                )
            else:
                columns.append(column.copy())
        rewritten._frames[node.offset] = DataFrame(
            columns^, height=frame.height()
        )
    return Transport(rewritten^, owners^, schema^)


def restore(result: DataFrame, transport: Transport) raises -> DataFrame:
    """Rebuild every originally nested output column from its owners."""
    if result.width() != transport.schema.width():
        raise Error("Metal nested transport output width differs")
    var columns = List[Series](capacity=result.width())
    for index in range(result.width()):
        ref column = result._columns[index]
        ref expected = transport.schema._columns[index]
        if not expected.dtype().is_nested():
            columns.append(column.copy())
            continue
        if column.dtype() != DataType.INT64:
            raise Error("Metal nested transport lost its row ids")
        ref ids = column._data[Column[Int64]]
        var owner = -1
        var mixed = False
        for row in range(len(column)):
            if ids._valid(row):
                var current = Int(ids._get(row) >> OWNER_SHIFT)
                if owner >= 0 and current != owner:
                    mixed = True
                owner = current
        var rebuilt: Series
        if mixed:
            # Concatenated sources: lay this column's owners end to end.
            var bases = List[Int](length=len(transport.owners), fill=-1)
            var combined = Optional[Series]()
            for row in range(len(column)):
                if not ids._valid(row):
                    continue
                var at = Int(ids._get(row) >> OWNER_SHIFT)
                if bases[at] >= 0:
                    continue
                if combined:
                    bases[at] = len(combined.value())
                    combined = combined.value().append(transport.owners[at])
                else:
                    bases[at] = 0
                    combined = transport.owners[at].copy()
            var rows = List[Int](capacity=len(column))
            for row in range(len(column)):
                if ids._valid(row):
                    var id = ids._get(row)
                    rows.append(
                        bases[Int(id >> OWNER_SHIFT)] + Int(id & ROW_MASK)
                    )
                else:
                    rows.append(-1)
            rebuilt = combined.value().take_or_null(rows)
        else:
            var rows = List[Int](capacity=len(column))
            for row in range(len(column)):
                rows.append(
                    Int(ids._get(row) & ROW_MASK) if ids._valid(row) else -1
                )
            if owner < 0:
                # Every row is null: any owner of the right dtype will do.
                for at in range(len(transport.owners)):
                    if transport.owners[at].dtype() == expected.dtype():
                        owner = at
                        break
                if owner < 0:
                    raise Error("Metal nested transport owner is missing")
            rebuilt = transport.owners[owner].take_or_null(rows)
        columns.append(rebuilt.renamed(column.name()))
    return DataFrame(columns^, height=result.height())


def _is_function(op: Int) -> Bool:
    return (
        op == LIST_LEN
        or op == LIST_GET
        or op == LIST_CONTAINS
        or op == LIST_JOIN
        or op == LIST_SUM
        or op == LIST_MIN
        or op == LIST_MAX
        or op == LIST_MEAN
        or op == STRUCT_FIELD
    )


def _replace(expression: Expr, targets: Dict[Int, String]) -> Expr:
    """Replace each target subtree by a column; drop unreachable nodes."""
    var count = len(expression._nodes)
    var keep = List[Bool](length=count, fill=False)
    keep[count - 1] = True
    for reverse in range(count):
        var i = count - 1 - reverse
        if not keep[i] or i in targets:
            continue
        ref node = expression._nodes[i]
        for child in [node.left, node.right, node.extra]:
            if child >= 0:
                keep[child] = True
    var position = List[Int](length=count, fill=-1)
    var nodes = List[Node]()
    for i in range(count):
        if not keep[i]:
            continue
        position[i] = len(nodes)
        if i in targets:
            try:
                nodes.append(col(targets[i])._nodes[0].copy())
            except:
                pass
            continue
        var node = expression._nodes[i].copy()
        if node.left >= 0:
            node.left = position[node.left]
        if node.right >= 0:
            node.right = position[node.right]
        if node.extra >= 0:
            node.extra = position[node.extra]
        nodes.append(node^)
    return Expr(nodes^, expression._name)


def _only_reads(expression: Expr, name: String) -> Bool:
    for node in expression._nodes:
        if node.op == COL and node.text != name:
            return False
        if node.op == SELECTOR:
            return False
    return True


def lower_functions(
    query: LazyFrame, metadata_only: Bool = False
) raises -> Optional[LazyFrame]:
    """Evaluate List/Struct functions of source columns while staging.

    A List or Struct function of one source column (len, get, contains,
    join, sum/min/max/mean, field) depends only on that column's value, so
    the CPU kernels evaluate it once per source row and the plan then moves
    the scalar result like any other column. The nested column itself, if
    still used, travels by row id. A name that is ambiguous across scans,
    or that the plan renames or redefines, keeps its existing route.

    Staging evaluates every source row, including rows the CPU executor may
    prune first, so a failing evaluation is a data fault that the CPU
    decides (accelerator/faults.mojo). With metadata_only, functions are
    bound over empty frames: planning never reads values.
    """
    var live = _live(query)
    var sources = Dict[String, Int]()  # name -> frame offset, -1 ambiguous
    for index in range(len(query._nodes)):
        ref node = query._nodes[index]
        if not live[index] or node.kind != SCAN_FRAME:
            continue
        if node.offset < 0 or node.offset >= len(query._frames):
            continue
        for column in query._frames[node.offset]._columns:
            if not column.dtype().is_nested():
                continue
            var name = column.name()
            if name in sources and sources[name] != node.offset:
                sources[name] = -1
            else:
                sources[name] = node.offset
    if not len(sources):
        return None
    # A node that redefines a name (alias or computed column) hides the
    # source column from every node above it, not from its own inputs.
    var hidden = List[List[String]](
        length=len(query._nodes), fill=List[String]()
    )
    for index in range(len(query._nodes)):
        ref node = query._nodes[index]
        var names = List[String]()
        for child in [node.left, node.right]:
            if child < 0:
                continue
            for name in hidden[child]:
                names.append(name)
            ref below = query._nodes[child]
            if below.kind == FILTER:
                continue
            for expression in below.exprs:
                if expression._name in sources and not (
                    _bare(expression)
                    and expression._nodes[0].text == expression._name
                ):
                    names.append(expression._name)
        hidden[index] = names^
    var rewritten = query.copy()
    var synthetic = Dict[String, String]()  # signature -> column name
    var additions = Dict[Int, List[Series]]()
    var used = Dict[String, Bool]()
    for frame in query._frames:
        for name in frame.columns():
            used[name] = True
    var changed = False
    for index in range(len(query._nodes)):
        if not live[index]:
            continue
        for e in range(len(query._nodes[index].exprs)):
            ref expression = query._nodes[index].exprs[e]
            var targets = Dict[Int, String]()
            for i in range(len(expression._nodes)):
                ref node = expression._nodes[i]
                if not _is_function(node.op) or node.left < 0:
                    continue
                ref child = expression._nodes[node.left]
                if child.op != COL or child.text not in sources:
                    continue
                if child.text in hidden[index]:
                    continue
                var offset = sources[child.text]
                if offset < 0:
                    continue
                var function = subtree(expression, i)
                if not _only_reads(function, child.text):
                    continue
                # One synthetic column per occurrence.
                var signature = (
                    String(index) + ":" + String(e) + ":" + String(i)
                )
                if signature not in synthetic:
                    var name = "__dataframe_nested_" + String(len(synthetic))
                    while name in used:
                        name += "_"
                    used[name] = True
                    var frame = (
                        query._frames[offset]
                        .clear() if (metadata_only) else query._frames[offset]
                        .copy()
                    )
                    var values: Series
                    try:
                        values = frame.select_exprs(
                            [function.alias(name)]
                        ).column(name)
                    except error:
                        raise data_fault(String(error))
                    if offset not in additions:
                        additions[offset] = List[Series]()
                    additions[offset].append(values^)
                    synthetic[signature] = name
                targets[i] = synthetic[signature]
            if len(targets):
                rewritten._nodes[index].exprs[e] = _replace(expression, targets)
                changed = True
    if not changed:
        return None
    for offset in additions.keys():
        var columns = rewritten._frames[offset]._columns.copy()
        for added in additions[offset]:
            columns.append(added.copy())
        rewritten._frames[offset] = DataFrame(
            columns^, height=rewritten._frames[offset].height()
        )
    # Staged columns stay private: project back to the original outputs,
    # since a with_columns above a scan keeps every scan column.
    var outputs = List[Expr]()
    for name in schema_frame(query, len(query._nodes) - 1).columns():
        outputs.append(col(name))
    rewritten._nodes.append(
        _plan_node(SELECT, len(rewritten._nodes) - 1, exprs=outputs^)
    )
    # A scan that projects names must read the new columns too.
    for index in range(len(rewritten._nodes)):
        ref node = rewritten._nodes[index]
        if (
            node.kind == SCAN_FRAME
            and len(node.names)
            and (node.offset in additions)
        ):
            for added in additions[node.offset]:
                rewritten._nodes[index].names.append(added.name())
    return rewritten^


def direct_projection(query: LazyFrame) -> Bool:
    """A direct scan and one SELECT: the shape the resident routes accept."""
    return (
        len(query._nodes) == 2
        and query._nodes[0].kind == SCAN_FRAME
        and query._nodes[1].kind == SELECT
        and query._nodes[1].left == 0
    )
