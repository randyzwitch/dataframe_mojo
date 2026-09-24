"""Lazy query plans over the eager operators.

A LazyFrame records operations as a flat list of plan nodes (children always
precede parents). Nothing reads data until `collect`. Optimization rewrites
the plan before execution:

- predicate pushdown: filters move below with_columns/select that do not
  produce the columns they read, below a sort when they are row-local, and
  into the side of an inner join (or the left side of a left/semi/anti join)
  that owns every column they read;
  row-local filters directly above unrestricted CSV scans run per decode range;
- projection pushdown: scans read only the columns the rest of the plan uses
  (CSV scans decode only those fields);
- slice pushdown: a head/slice directly over a CSV scan becomes `n_rows`.

Filters never move past a slice, unique, group_by, or right/full join,
because that would change which rows those operators see.
"""
from std.collections import Dict, Optional
from .csv import CsvSchema, read_csv
from .csv_reader import read_csv_explicit, read_csv_inferred
from .expr import COL, OVER, SELECTOR, Expr, col, is_reduction, is_window
from .frame import DataFrame, GroupBy
from .series import Series

comptime SCAN_FRAME = 0
comptime SCAN_CSV = 1
comptime FILTER = 2
comptime SELECT = 3
comptime WITH_COLUMNS = 4
comptime AGG = 5
comptime JOIN = 6
comptime SORT = 7
comptime SLICE = 8
comptime UNIQUE = 9
comptime DROP = 10


@fieldwise_init
struct PlanNode(Copyable):
    var kind: Int
    var left: Int
    var right: Int
    var exprs: List[Expr]
    var names: List[String]
    var names2: List[String]
    var flags: List[Bool]
    var text: String
    var offset: Int
    var length: Int
    var maintain_order: Bool


def _plan_node(
    kind: Int,
    left: Int = -1,
    right: Int = -1,
    exprs: List[Expr] = List[Expr](),
    names: List[String] = List[String](),
    names2: List[String] = List[String](),
    flags: List[Bool] = List[Bool](),
    text: String = "",
    offset: Int = 0,
    length: Int = -1,
    maintain_order: Bool = False,
) -> PlanNode:
    return PlanNode(
        kind,
        left,
        right,
        exprs.copy(),
        names.copy(),
        names2.copy(),
        flags.copy(),
        text,
        offset,
        length,
        maintain_order,
    )


def _references(expr: Expr) -> Optional[List[String]]:
    """Columns an expression reads, or None when a selector reads any."""
    var names = List[String]()
    for node in expr._nodes:
        if node.op == SELECTOR:
            return None
        if node.op == COL:
            names.append(node.text)
        if node.text2.byte_length() > 0 and node.op >= 120:
            # over() partitions read their key columns.
            for part in node.text2.split("\x1f"):
                names.append(String(part))
    return names^


def _row_local(exprs: List[Expr]) -> Bool:
    """True when every output row depends only on the same input row."""
    for e in exprs:
        for node in e._nodes:
            if is_reduction(node.op) or is_window(node.op) or node.op == OVER:
                return False
    return True


def _output_names(exprs: List[Expr]) -> List[String]:
    var names = List[String]()
    for e in exprs:
        names.append(e._name)
    return names^


struct LazyFrame(Copyable):
    """A deferred query; build it with DataFrame.lazy() or scan_csv()."""

    var _nodes: List[PlanNode]
    var _frames: List[DataFrame]
    var _schemas: List[Optional[CsvSchema]]

    def __init__(out self, frame: DataFrame):
        self._frames = [frame.copy()]
        self._schemas = [Optional[CsvSchema]()]
        self._nodes = [_plan_node(SCAN_FRAME, offset=0)]

    def __init__(
        out self,
        var nodes: List[PlanNode],
        var frames: List[DataFrame],
        var schemas: List[Optional[CsvSchema]],
    ):
        self._nodes = nodes^
        self._frames = frames^
        self._schemas = schemas^

    def _push(self, var node: PlanNode) -> Self:
        var result = self.copy()
        node.left = len(result._nodes) - 1
        result._nodes.append(node^)
        return result^

    def filter(self, predicate: Expr) -> Self:
        return self._push(_plan_node(FILTER, exprs=[predicate.copy()]))

    def select(self, expr: Expr) -> Self:
        return self.select_exprs([expr.copy()])

    def select(self, names: List[String]) -> Self:
        var exprs = List[Expr]()
        for name in names:
            exprs.append(col(name))
        return self.select_exprs(exprs)

    def select_exprs(self, exprs: List[Expr]) -> Self:
        return self._push(_plan_node(SELECT, exprs=exprs))

    def with_columns(self, expr: Expr) -> Self:
        return self.with_columns([expr.copy()])

    def with_columns(self, exprs: List[Expr]) -> Self:
        return self._push(_plan_node(WITH_COLUMNS, exprs=exprs))

    def group_by(
        self, keys: List[String], *, maintain_order: Bool = False
    ) -> LazyGroupBy:
        return LazyGroupBy(self.copy(), keys.copy(), maintain_order)

    def group_by(
        self, key: String, *, maintain_order: Bool = False
    ) -> LazyGroupBy:
        return LazyGroupBy(self.copy(), [key], maintain_order)

    def sort(
        self,
        by: List[String],
        descending: Bool = False,
        nulls_last: Bool = True,
    ) -> Self:
        var n = len(by)
        return self._push(
            _plan_node(
                SORT,
                names=by,
                flags=List[Bool](length=n, fill=descending)
                + List[Bool](length=n, fill=nulls_last),
            )
        )

    def sort(self, by: String, descending: Bool = False) -> Self:
        return self.sort([by], descending)

    def slice(self, offset: Int, length: Int = -1) -> Self:
        return self._push(_plan_node(SLICE, offset=offset, length=length))

    def head(self, n: Int = 5) -> Self:
        return self.slice(0, n)

    def limit(self, n: Int = 5) -> Self:
        return self.head(n)

    def unique(
        self,
        subset: List[String] = List[String](),
        *,
        keep: String = "any",
        maintain_order: Bool = False,
    ) -> Self:
        return self._push(
            _plan_node(
                UNIQUE, names=subset, text=keep, maintain_order=maintain_order
            )
        )

    def drop(self, names: List[String]) -> Self:
        return self._push(_plan_node(DROP, names=names))

    def join(
        self,
        other: Self,
        on: List[String],
        how: String = "inner",
        suffix: String = "_right",
    ) -> Self:
        """Join with another lazy plan; see DataFrame.join."""
        var result = self.copy()
        var left_root = len(result._nodes) - 1
        var shift = len(result._nodes)
        var frame_shift = len(result._frames)
        for frame in other._frames:
            result._frames.append(frame.copy())
        for schema in other._schemas:
            result._schemas.append(schema.copy())
        for node in other._nodes:
            var copied = node.copy()
            if copied.left >= 0:
                copied.left += shift
            if copied.right >= 0:
                copied.right += shift
            if copied.kind == SCAN_FRAME or copied.kind == SCAN_CSV:
                copied.offset += frame_shift
            result._nodes.append(copied^)
        var join = _plan_node(
            JOIN, left_root, len(result._nodes) - 1, names=on, text=how
        )
        join.names2 = [suffix]
        result._nodes.append(join^)
        return result^

    def join(
        self,
        other: Self,
        on: String,
        how: String = "inner",
        suffix: String = "_right",
    ) -> Self:
        return self.join(other, [on], how, suffix)

    def collect(self, *, optimize: Bool = True) raises -> DataFrame:
        """Optimize (unless disabled) and execute the plan."""
        var plan = self._optimized() if optimize else self.copy()
        return plan._execute(len(plan._nodes) - 1, False)

    def fetch(self, n: Int = 5) raises -> DataFrame:
        """Collect only the first n rows of the result."""
        return self.head(n).collect()

    def collect_schema(self) raises -> List[String]:
        """Output names and dtypes as "name: dtype", computed without reading
        rows: every scan yields zero rows, then the plan runs as usual, so
        binding validates each expression exactly as collect would."""
        var plan = self._optimized()
        var frame = plan._execute(len(plan._nodes) - 1, True)
        var out = List[String]()
        for field in frame.schema():
            out.append(field.name + ": " + field.dtype.name())
        return out^

    def explain(self, *, optimize: Bool = True) raises -> String:
        """The (optimized) plan, one operator per line, root first."""
        var plan = self._optimized() if optimize else self.copy()
        var out = String()
        plan._describe(len(plan._nodes) - 1, 0, out)
        return out^

    # --- execution -----------------------------------------------------

    def _execute(self, index: Int, empty: Bool) raises -> DataFrame:
        ref node = self._nodes[index]
        if node.kind == SCAN_FRAME:
            var frame = self._frames[node.offset].copy()
            if len(node.names) > 0:
                frame = frame.select(node.names)
            return frame.clear() if empty else frame^
        if node.kind == SCAN_CSV:
            var rows = 0 if empty else node.length
            if self._schemas[node.offset]:
                return read_csv(
                    node.text,
                    self._schemas[node.offset].value(),
                    n_rows=rows,
                    columns=node.names,
                )
            return read_csv(node.text, n_rows=rows, columns=node.names)
        # Keep a sort's row permutation until after a plain projection, so
        # columns used only as sort keys are never gathered into output.
        if node.kind == SELECT and not empty:
            ref sorted = self._nodes[node.left]
            if sorted.kind == SORT:
                var plain = True
                for expression in node.exprs:
                    plain = plain and len(expression._nodes) == 1
                    if len(expression._nodes) == 1:
                        plain = plain and expression._nodes[0].op == COL
                if plain:
                    var source = self._execute(sorted.left, False)
                    var n = len(sorted.names)
                    var descending = List[Bool]()
                    var nulls_last = List[Bool]()
                    for i in range(n):
                        descending.append(sorted.flags[i])
                        nulls_last.append(sorted.flags[n + i])
                    var order = source.arg_sort(
                        sorted.names,
                        descending=descending,
                        nulls_last=nulls_last,
                    )
                    return source.select_exprs(node.exprs).take(order^)
        # Filter each decoded CSV range before assembling the scan result.
        # A row limit and a whole-column predicate require the original
        # materialization order, so retain the eager path for those cases.
        if node.kind == FILTER and not empty and _row_local(node.exprs):
            ref source = self._nodes[node.left]
            if source.kind == SCAN_CSV and source.length < 0:
                if self._schemas[source.offset]:
                    return read_csv_explicit(
                        source.text,
                        self._schemas[source.offset].value(),
                        columns=source.names,
                        predicate=Optional(node.exprs[0].copy()),
                    )
                return read_csv_inferred(
                    source.text,
                    columns=source.names,
                    predicate=Optional(node.exprs[0].copy()),
                )
        var input = self._execute(node.left, empty)
        if node.kind == FILTER:
            return input.filter(node.exprs[0])
        if node.kind == SELECT:
            return input.select_exprs(node.exprs)
        if node.kind == WITH_COLUMNS:
            return input.with_columns(node.exprs)
        if node.kind == AGG:
            return input.group_by(
                node.names, maintain_order=node.maintain_order
            ).agg(node.exprs)
        if node.kind == JOIN:
            var right = self._execute(node.right, empty)
            return input.join(right, node.names, node.text, node.names2[0])
        if node.kind == SORT:
            var n = len(node.names)
            var descending = List[Bool]()
            var nulls_last = List[Bool]()
            for i in range(n):
                descending.append(node.flags[i])
                nulls_last.append(node.flags[n + i])
            return input.sort(
                node.names, descending=descending, nulls_last=nulls_last
            )
        if node.kind == SLICE:
            return input.slice(node.offset, node.length)
        if node.kind == UNIQUE:
            return input.unique(
                node.names, keep=node.text, maintain_order=node.maintain_order
            )
        return input.drop(node.names)

    # --- optimization --------------------------------------------------

    def _optimized(self) raises -> Self:
        var plan = self.copy()
        plan._push_predicates()
        plan._push_slices()
        plan._push_projections()
        return plan^

    def _parents(self) -> List[Int]:
        var parents = List[Int](length=len(self._nodes), fill=-1)
        for i in range(len(self._nodes)):
            if self._nodes[i].left >= 0:
                parents[self._nodes[i].left] = i
            if self._nodes[i].right >= 0:
                parents[self._nodes[i].right] = i
        return parents^

    def _columns_of(self, index: Int) raises -> List[String]:
        return self._execute(index, True).columns()

    def _push_predicates(mut self) raises:
        """Swap each filter below operators that cannot change its result."""
        var changed = True
        while changed:
            changed = False
            for i in range(len(self._nodes)):
                if self._nodes[i].kind != FILTER:
                    continue
                var reads = _references(self._nodes[i].exprs[0])
                if not reads:
                    continue
                var child = self._nodes[i].left
                ref below = self._nodes[child]
                if below.kind == WITH_COLUMNS or below.kind == SELECT:
                    # Safe when the filter reads only unchanged input columns.
                    var produced = _output_names(below.exprs)
                    var ok = True
                    for name in reads.value():
                        for p in produced:
                            if p == name:
                                ok = False
                    if below.kind == SELECT:
                        for e in below.exprs:
                            var r = _references(e)
                            if (
                                not r
                                or len(r.value()) != 1
                                or r.value()[0] != e._name
                                or len(e._nodes) != 1
                            ):
                                ok = False
                    if ok:
                        self._swap_down(i, child, False)
                        changed = True
                        break
                elif below.kind == SORT and _row_local(self._nodes[i].exprs):
                    # Stable sorting and a row-local filter commute: the
                    # surviving rows retain the same relative sort order.
                    self._swap_down(i, child, False)
                    changed = True
                    break
                elif below.kind == JOIN and (
                    below.text == "inner"
                    or below.text == "left"
                    or below.text == "semi"
                    or below.text == "anti"
                ):
                    var left_cols = self._columns_of(below.left)
                    var right_cols = self._columns_of(below.right)
                    var side = -1
                    if _covers(left_cols, reads.value()):
                        side = 0
                    elif below.text == "inner" and _covers_exclusive(
                        right_cols, left_cols, reads.value()
                    ):
                        side = 1
                    if side >= 0:
                        self._swap_down(i, child, side == 1)
                        changed = True
                        break

    def _swap_down(mut self, filter: Int, child: Int, right_side: Bool):
        """Move filter from above child to directly above child's input."""
        var parents = self._parents()
        var parent = parents[filter]
        var target = (
            self._nodes[child].right if right_side else self._nodes[child].left
        )
        self._nodes[filter].left = target
        if right_side:
            self._nodes[child].right = filter
        else:
            self._nodes[child].left = filter
        if parent >= 0:
            if self._nodes[parent].left == filter:
                self._nodes[parent].left = child
            else:
                self._nodes[parent].right = child
        self._reorder()

    def _reorder(mut self):
        """Restore children-before-parents order after rewiring."""
        var root = -1
        var parents = self._parents()
        for i in range(len(self._nodes)):
            if parents[i] < 0:
                root = i
        var order = List[Int]()
        var stack = List[Int]()
        var visited = List[Bool](length=len(self._nodes), fill=False)
        stack.append(root)
        while len(stack) > 0:
            var top = stack[len(stack) - 1]
            var pending = False
            for child in [self._nodes[top].left, self._nodes[top].right]:
                if child >= 0 and not visited[child]:
                    stack.append(child)
                    pending = True
            if not pending:
                _ = stack.pop()
                if not visited[top]:
                    visited[top] = True
                    order.append(top)
        var position = List[Int](length=len(self._nodes), fill=-1)
        for k in range(len(order)):
            position[order[k]] = k
        var nodes = List[PlanNode]()
        for k in range(len(order)):
            var node = self._nodes[order[k]].copy()
            if node.left >= 0:
                node.left = position[node.left]
            if node.right >= 0:
                node.right = position[node.right]
            nodes.append(node^)
        self._nodes = nodes^

    def _push_slices(mut self):
        """Move leading slices below row-local projections; a leading slice
        directly over a CSV scan also becomes the reader's n_rows."""
        var changed = True
        while changed:
            changed = False
            for i in range(len(self._nodes)):
                ref node = self._nodes[i]
                if node.kind != SLICE or node.offset != 0 or node.length < 0:
                    continue
                ref below = self._nodes[node.left]
                if (
                    below.kind == SELECT or below.kind == WITH_COLUMNS
                ) and _row_local(below.exprs):
                    self._swap_down(i, node.left, False)
                    changed = True
                    break
        for i in range(len(self._nodes)):
            ref node = self._nodes[i]
            if node.kind != SLICE or node.offset != 0 or node.length < 0:
                continue
            ref scan = self._nodes[node.left]
            if scan.kind == SCAN_CSV and (
                scan.length < 0 or scan.length > node.length
            ):
                self._nodes[node.left].length = node.length

    def _push_projections(mut self) raises:
        """Scans read only columns that some operator above them uses."""
        var needed = List[Optional[List[String]]](
            length=len(self._nodes), fill=Optional[List[String]]()
        )
        var all_needed = List[Bool](length=len(self._nodes), fill=False)
        all_needed[len(self._nodes) - 1] = True
        for reverse in range(len(self._nodes)):
            var i = len(self._nodes) - 1 - reverse
            ref node = self._nodes[i]
            if node.kind == SCAN_FRAME or node.kind == SCAN_CSV:
                if not all_needed[i] and needed[i]:
                    var columns = self._columns_of(i)
                    var keep = List[String]()
                    for c in columns:
                        for want in needed[i].value():
                            if want == c:
                                keep.append(c)
                                break
                    if len(keep) > 0 and len(keep) < len(columns):
                        self._nodes[i].names = keep^
                continue
            # What this node reads from its input(s).
            var reads = List[String]()
            var reads_all = all_needed[i] and (
                node.kind == FILTER
                or node.kind == SORT
                or node.kind == SLICE
                or node.kind == UNIQUE
                or node.kind == WITH_COLUMNS
                or node.kind == DROP
                or node.kind == JOIN
            )
            if node.kind == UNIQUE and len(node.names) == 0:
                reads_all = True
            if not reads_all:
                if needed[i]:
                    for n in needed[i].value():
                        reads.append(n)
                for e in node.exprs:
                    var r = _references(e)
                    if not r:
                        reads_all = True
                    else:
                        for n in r.value():
                            reads.append(n)
                for n in node.names:
                    reads.append(n)
            for child in [node.left, node.right]:
                if child < 0:
                    continue
                if reads_all or node.kind == JOIN:
                    all_needed[child] = True
                elif needed[child]:
                    var merged = needed[child].value().copy()
                    for n in reads:
                        merged.append(n)
                    needed[child] = merged^
                else:
                    needed[child] = reads.copy()

    def _describe(self, index: Int, depth: Int, mut out: String):
        ref node = self._nodes[index]
        var pad = String("  ") * depth
        var label: String
        if node.kind == SCAN_FRAME:
            label = "SCAN frame"
        elif node.kind == SCAN_CSV:
            label = "SCAN CSV " + node.text
        elif node.kind == FILTER:
            label = "FILTER"
        elif node.kind == SELECT:
            label = "SELECT " + _joined(_output_names(node.exprs))
        elif node.kind == WITH_COLUMNS:
            label = "WITH_COLUMNS " + _joined(_output_names(node.exprs))
        elif node.kind == AGG:
            label = (
                "GROUP_BY "
                + _joined(node.names)
                + " AGG "
                + _joined(_output_names(node.exprs))
            )
        elif node.kind == JOIN:
            label = "JOIN " + node.text + " on " + _joined(node.names)
        elif node.kind == SORT:
            label = "SORT " + _joined(node.names)
        elif node.kind == SLICE:
            label = "SLICE " + String(node.offset) + " " + String(node.length)
        elif node.kind == UNIQUE:
            label = "UNIQUE " + _joined(node.names)
        else:
            label = "DROP " + _joined(node.names)
        if node.kind == SCAN_FRAME or node.kind == SCAN_CSV:
            if len(node.names) > 0:
                label += " [project " + _joined(node.names) + "]"
            if node.kind == SCAN_CSV and node.length >= 0:
                label += " [n_rows " + String(node.length) + "]"
        out += pad + label + "\n"
        if node.left >= 0:
            self._describe(node.left, depth + 1, out)
        if node.right >= 0:
            self._describe(node.right, depth + 1, out)


def _joined(names: List[String]) -> String:
    var out = String()
    for i in range(len(names)):
        if i > 0:
            out += ", "
        out += names[i]
    return out^


def _covers(columns: List[String], reads: List[String]) -> Bool:
    for name in reads:
        var found = False
        for c in columns:
            found = found or c == name
        if not found:
            return False
    return True


def _covers_exclusive(
    columns: List[String], other: List[String], reads: List[String]
) -> Bool:
    """Right-side names that the join output does not rename or shadow."""
    for name in reads:
        for c in other:
            if c == name:
                return False
    return _covers(columns, reads)


@fieldwise_init
struct LazyGroupBy(Copyable):
    """A pending lazy grouping; finish it with agg."""

    var _frame: LazyFrame
    var _keys: List[String]
    var _maintain_order: Bool

    def agg(self, exprs: List[Expr]) -> LazyFrame:
        return self._frame._push(
            _plan_node(
                AGG,
                exprs=exprs,
                names=self._keys,
                maintain_order=self._maintain_order,
            )
        )

    def agg(self, expr: Expr) -> LazyFrame:
        return self.agg([expr.copy()])


def scan_csv(path: String) raises -> LazyFrame:
    """Lazily read a CSV file with an inferred schema. Nothing is read until
    collect; projection and head() are pushed into the reader."""
    # Scan slots are parallel: frames[k] and schemas[k] belong to one scan.
    return LazyFrame(
        [_plan_node(SCAN_CSV, text=path, offset=0)],
        [DataFrame([])],
        [Optional[CsvSchema]()],
    )


def scan_csv(path: String, schema: CsvSchema) raises -> LazyFrame:
    return LazyFrame(
        [_plan_node(SCAN_CSV, text=path, offset=0)],
        [DataFrame([])],
        [Optional[CsvSchema](schema.copy())],
    )
