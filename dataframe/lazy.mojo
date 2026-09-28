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
  (CSV and Parquet scans decode only those fields); a join passes each input
  only its keys and the columns read above it from that side;
- slice pushdown: a head/slice directly over a CSV scan becomes `n_rows`;
- row-group pruning: a row-local filter directly above a Parquet scan reads
  the footer statistics first and decodes only the row groups that can
  hold a match (see `parquet._pruned_row_groups`).

Filters never move past a slice, unique, group_by, or right/full join,
because that would change which rows those operators see.
"""
from std.collections import Dict, Optional
from std.memory import ArcPointer
from .csv_reader import _CsvBatches, _DecodeJob
from .csv_types import _map_file
from .parquet import _ParquetBatches
from .parallel import Job, Pool, configured_workers
from .streaming import _StreamReduction
from .expr import (
    SUM,
    COUNT,
    MIN,
    MAX,
    MEAN,
    FIRST,
    LAST,
    STD,
    VAR,
    LEN,
    ANY,
    ALL,
    NULL_COUNT,
    N_UNIQUE,
)
from .frame import concat
from .csv import CsvSchema, read_csv
from .csv_reader import read_csv_explicit, read_csv_inferred
from .expr import (
    COL,
    OVER,
    SELECTOR,
    Expr,
    col,
    is_reduction,
    is_window,
    subtree,
)
from .frame import DataFrame, GroupBy, _row_local, _stream_reductions
from .parquet import (
    _pruned_row_groups,
    parquet_row_group_statistics,
    read_parquet,
)
from .series import Series
from .join_hash import (
    PreparedHashIndex,
    prepare_hash_index,
    prepare_progression_index,
    prefer_left_build,
)

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
comptime SCAN_PARQUET = 11
comptime EXPLODE = 12
comptime UNNEST = 13


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


def _output_names(exprs: List[Expr]) -> List[String]:
    var names = List[String]()
    for e in exprs:
        names.append(e._name)
    return names^


def _stream_rows(node: PlanNode) -> Bool:
    if node.kind == FILTER or node.kind == WITH_COLUMNS:
        return _row_local(node.exprs)
    if node.kind == SELECT:
        if not _row_local(node.exprs):
            return False
        # An all-scalar select produces one row for the whole input.
        for expression in node.exprs:
            for item in expression._nodes:
                if item.op == COL or item.op == SELECTOR:
                    return True
        return False
    return node.kind == DROP or node.kind == EXPLODE or node.kind == UNNEST


struct _StreamJob(Job):
    var decode: List[_DecodeJob]
    var frame: DataFrame
    var operations: List[PlanNode]
    var joins: ArcPointer[List[DataFrame]]
    var indexes: ArcPointer[List[Optional[PreparedHashIndex]]]
    var expressions: List[Expr]
    var keys: List[String]
    var reduced: List[_StreamReduction]

    def __init__(
        out self,
        var frame: DataFrame,
        operations: List[PlanNode],
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[PreparedHashIndex]]],
        expressions: List[Expr],
        keys: List[String],
    ):
        self.decode = List[_DecodeJob]()
        self.frame = frame^
        self.operations = operations.copy()
        self.joins = joins.copy()
        self.indexes = indexes.copy()
        self.expressions = expressions.copy()
        self.keys = keys.copy()
        self.reduced = List[_StreamReduction]()

    def run(mut self) raises:
        if len(self.decode):
            self.decode[0].run()
            self.frame = self.decode.pop().into_frame()
        for node in self.operations:
            if node.kind == JOIN and self.indexes[][node.offset]:
                self.frame = self.frame._join_impl(
                    self.joins[][node.offset],
                    left_on=node.names,
                    right_on=node.names,
                    how=node.text,
                    suffix=node.names2[0],
                    prepared=self.indexes[][node.offset],
                )
            elif node.kind == JOIN:
                self.frame = self.frame.join(
                    self.joins[][node.offset],
                    node.names,
                    node.text,
                    node.names2[0],
                )
            elif node.kind == FILTER:
                self.frame = self.frame.filter(node.exprs[0])
            elif node.kind == SELECT:
                self.frame = self.frame.select_exprs(node.exprs)
            elif node.kind == WITH_COLUMNS:
                self.frame = self.frame.with_columns(node.exprs)
            elif node.kind == DROP:
                self.frame = self.frame.drop(node.names)
            elif node.kind == EXPLODE:
                self.frame = self.frame.explode(node.names)
            elif node.kind == UNNEST:
                self.frame = self.frame.unnest(node.text)
        if len(self.expressions):
            self.reduced.append(
                _StreamReduction(self.frame, self.expressions, self.keys)
            )
            self.frame = self.frame.clear()


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

    def explode(self, columns: List[String]) -> Self:
        """One row per list element; see DataFrame.explode."""
        return self._push(_plan_node(EXPLODE, names=columns))

    def explode(self, column: String) -> Self:
        return self.explode([column])

    def unnest(self, column: String) -> Self:
        """Replace a struct column with its fields; see DataFrame.unnest."""
        return self._push(_plan_node(UNNEST, text=column))

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
            if _is_scan(copied.kind):
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

    def collect(
        self,
        *,
        optimize: Bool = True,
        streaming: Bool = True,
        batch_size: Int = 65536,
    ) raises -> DataFrame:
        """Optimize (unless disabled) and execute the plan.

        Streaming batches default to 65,536 rows. Set streaming=False to use
        the materializing executor. Stateful/global operations retain their
        documented boundaries; collecting still retains the final output.
        """
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        var plan = self._optimized() if optimize else self.copy()
        return plan._execute(len(plan._nodes) - 1, False, streaming, batch_size)

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

    def explain(
        self, *, optimize: Bool = True, streaming: Bool = True
    ) raises -> String:
        """The (optimized) plan, one operator per line, root first.

        Streaming annotations show batch-capable operators, aggregate state
        and materialization boundaries. streaming=False omits annotations.
        """
        var plan = self._optimized() if optimize else self.copy()
        var out = String()
        plan._describe(len(plan._nodes) - 1, 0, out, streaming)
        return out^

    # --- execution -----------------------------------------------------

    def _known_height(self, index: Int) -> Int:
        """Exact cheap cardinalities only; -1 means execution is required."""
        ref node = self._nodes[index]
        if node.kind == SCAN_FRAME:
            return self._frames[node.offset].height()
        if (
            node.kind == DROP
            or node.kind == UNNEST
            or (
                (node.kind == SELECT or node.kind == WITH_COLUMNS)
                and _stream_rows(node)
            )
        ):
            return self._known_height(node.left)
        if node.kind == SLICE and node.offset >= 0:
            var height = self._known_height(node.left)
            if height >= 0:
                height = max(0, height - node.offset)
                return min(height, node.length) if node.length >= 0 else height
        return -1

    def _stream_execute(
        self, index: Int, batch_size: Int
    ) raises -> Optional[DataFrame]:
        var cursor = index
        var expressions = List[Expr]()
        var keys = List[String]()
        var skip = 0
        var limit = -1
        ref terminal = self._nodes[index]
        if terminal.kind == AGG or terminal.kind == SELECT:
            if _stream_reductions(terminal.exprs):
                expressions = terminal.exprs.copy()
                if terminal.kind == AGG:
                    keys = terminal.names.copy()
                    if len(keys) == 0:
                        return None
                cursor = terminal.left
        if terminal.kind == SLICE:
            if terminal.offset < 0:
                return None
            skip = terminal.offset
            limit = terminal.length
            cursor = terminal.left
        var operations = List[PlanNode]()
        var joins = List[DataFrame]()
        var indexes = List[Optional[PreparedHashIndex]]()
        while cursor >= 0:
            ref node = self._nodes[cursor]
            if _stream_rows(node):
                operations.append(node.copy())
            elif node.kind == JOIN and node.text in [
                "inner",
                "left",
                "semi",
                "anti",
                "cross",
            ]:
                var prepared = Optional[PreparedHashIndex]()
                if (
                    self._known_height(node.left) <= batch_size
                    and (node.text == "inner" or node.text == "left")
                    and prefer_left_build(
                        self._known_height(node.left),
                        self._known_height(node.right),
                    )
                ):
                    # This left input already fits one stream batch, so
                    # physical reversal does not increase the materialized
                    # join output over the existing batch bound. Keep larger
                    # left inputs on the bounded streaming executor.
                    # A directly available progression already has a cheaper
                    # compact lookup. Validate once and retain its prepared
                    # state instead of falling back to eager progression scans.
                    ref build_node = self._nodes[node.right]
                    if build_node.kind == SCAN_FRAME and len(node.names) == 1:
                        var sources: List[Series] = [
                            self._frames[build_node.offset][
                                node.names[0]
                            ].copy()
                        ]
                        prepared = prepare_progression_index(sources)
                    if not prepared:
                        break
                var operation = node.copy()
                operation.offset = len(joins)
                joins.append(self._execute(node.right, False, True, batch_size))
                if not prepared and node.text != "cross" and len(node.names):
                    ref build = joins[len(joins) - 1]
                    var sources = List[Series]()
                    var supported = build.height() <= Int(Int32.MAX)
                    for name in node.names:
                        var key = build[name]
                        supported = supported and not key.dtype().is_nested()
                        sources.append(key.copy())
                    if supported:
                        prepared = prepare_hash_index(sources)
                indexes.append(prepared^)
                operations.append(operation^)
            else:
                break
            cursor = node.left
        if (
            cursor < 0
            or cursor == index
            and not _is_scan(self._nodes[cursor].kind)
        ):
            return None
        operations.reverse()
        var shared_joins = ArcPointer(joins^)
        var shared_indexes = ArcPointer(indexes^)
        ref source = self._nodes[cursor]
        var csv = List[_CsvBatches]()
        var parquet = List[_ParquetBatches]()
        var input = DataFrame(List[Series](), height=0)
        if source.kind == SCAN_CSV:
            var mapping = _map_file(source.text)
            if mapping.address == 0:
                return None
            csv.append(
                _CsvBatches(
                    mapping^,
                    self._schemas[source.offset],
                    source.names,
                    source.length,
                    batch_size,
                )
            )
        elif source.kind == SCAN_PARQUET:
            var groups = Optional[List[Int]]()
            if len(operations) and operations[0].kind == FILTER:
                groups = _pruned_row_groups(
                    parquet_row_group_statistics(source.text),
                    operations[0].exprs[0],
                )
            parquet.append(
                _ParquetBatches(source.text, source.names, groups, batch_size)
            )
        elif source.kind == SCAN_FRAME:
            input = self._frames[source.offset].copy()
            if len(source.names):
                input = input.select(source.names)
        else:
            input = self._execute(cursor, False, True, batch_size)
        var workers = configured_workers()
        var pool = Pool(1)
        var pool_ready = False
        var offset = 0
        var emitted = False
        var ended = False
        var outputs = List[DataFrame]()
        var reductions = List[_StreamReduction]()
        while not ended:
            var jobs = List[_StreamJob]()
            for _ in range(workers):
                var frame = DataFrame(List[Series](), height=0)
                var decode = List[_DecodeJob]()
                if len(csv):
                    var next = csv[0].next()
                    if not next:
                        ended = True
                        break
                    decode.append(next.take())
                elif len(parquet):
                    var next = parquet[0].next()
                    if not next:
                        ended = True
                        break
                    frame = next.take()
                else:
                    if emitted and offset >= input.height():
                        ended = True
                        break
                    frame = input.slice(offset, batch_size)
                    offset += frame.height()
                    emitted = True
                var job = _StreamJob(
                    frame^,
                    operations,
                    shared_joins,
                    shared_indexes,
                    expressions,
                    keys,
                )
                job.decode = decode^
                jobs.append(job^)
            if len(jobs) == 0:
                break
            if not pool_ready:
                pool = Pool(len(jobs))
                pool_ready = True
            pool.run(jobs, claim=True)
            # Pool returns jobs in submission order, independently of worker
            # completion order. Merge states and assemble rows in that order.
            for i in range(len(jobs)):
                if len(expressions):
                    if len(reductions) == 0:
                        reductions.append(jobs[i].reduced.pop())
                    else:
                        reductions[0].merge(jobs[i].reduced[0])
                else:
                    var part = jobs[i].frame.copy()
                    var dropped = min(skip, part.height())
                    skip -= dropped
                    part = part.slice(dropped, limit)
                    if limit >= 0:
                        limit -= part.height()
                    outputs.append(part^)
            if len(csv):
                csv[0].discard()
            if limit == 0:
                ended = True
        pool.release()
        if len(reductions):
            return reductions[0].finish()
        if len(outputs):
            return concat(outputs)
        return None

    def _execute(
        self,
        index: Int,
        empty: Bool,
        streaming: Bool = False,
        batch_size: Int = 65536,
    ) raises -> DataFrame:
        if streaming and not empty:
            var streamed = self._stream_execute(index, batch_size)
            if streamed:
                return streamed.take()
        ref node = self._nodes[index]
        if node.kind == SCAN_FRAME:
            var frame = self._frames[node.offset].copy()
            if len(node.names) > 0:
                frame = frame.select(node.names)
            return frame.clear() if empty else frame^
        if node.kind == SCAN_CSV:
            if empty:
                # Metadata probes must not use eager n_rows=0: that API
                # intentionally decodes a source chunk before truncation.
                var mapping = _map_file(node.text)
                if mapping.address != 0:
                    var batches = _CsvBatches(
                        mapping^,
                        self._schemas[node.offset],
                        node.names,
                        0,
                        batch_size,
                    )
                    var next = batches.next()
                    var job = next.take()
                    job.run()
                    return job^.into_frame()
            var rows = 0 if empty else node.length
            if self._schemas[node.offset]:
                return read_csv(
                    node.text,
                    self._schemas[node.offset].value(),
                    n_rows=rows,
                    columns=node.names,
                )
            return read_csv(node.text, n_rows=rows, columns=node.names)
        if node.kind == SCAN_PARQUET:
            if empty:
                return read_parquet(
                    node.text, columns=node.names, row_groups=List[Int]()
                )
            return read_parquet(node.text, columns=node.names)
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
                    var source = self._execute(
                        sorted.left, False, streaming, batch_size
                    )
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
            # Decode only the row groups whose footer bounds admit a match;
            # the filter still runs on what was read.
            if source.kind == SCAN_PARQUET:
                var groups = _pruned_row_groups(
                    parquet_row_group_statistics(source.text), node.exprs[0]
                )
                return read_parquet(
                    source.text, columns=source.names, row_groups=groups
                ).filter(node.exprs[0])
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
        var input = self._execute(node.left, empty, streaming, batch_size)
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
            var right = self._execute(node.right, empty, streaming, batch_size)
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
        if node.kind == EXPLODE:
            return input.explode(node.names)
        if node.kind == UNNEST:
            return input.unnest(node.text)
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
        """Scans read only columns that some operator above them uses.

        A join passes each input only its keys and the columns the plan
        above reads from that side, so unused payload columns are never
        read or gathered. A join input that is not a scan gets a plain
        column selection above it.
        """
        var needed = List[Optional[List[String]]](
            length=len(self._nodes), fill=Optional[List[String]]()
        )
        var all_needed = List[Bool](length=len(self._nodes), fill=False)
        all_needed[len(self._nodes) - 1] = True
        # (join index, right side?, columns) for join inputs to narrow.
        var narrow_joins = List[Int]()
        var narrow_sides = List[Bool]()
        var narrow_columns = List[List[String]]()
        for reverse in range(len(self._nodes)):
            var i = len(self._nodes) - 1 - reverse
            ref node = self._nodes[i]
            if node.kind == JOIN and not all_needed[i] and needed[i]:
                var sides = self._join_input_columns(i, needed[i].value())
                if sides:
                    for side in range(2):
                        var child = node.right if side == 1 else node.left
                        ref keep = sides.value()[side]
                        needed[child] = keep.copy()
                        narrow_joins.append(i)
                        narrow_sides.append(side == 1)
                        narrow_columns.append(keep.copy())
                    continue
            if _is_scan(node.kind):
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
                or node.kind == EXPLODE
                or node.kind == UNNEST
            )
            if node.kind == UNNEST:
                reads.append(node.text)
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
        self._narrow_join_inputs(narrow_joins, narrow_sides, narrow_columns)

    def _join_input_columns(
        self, index: Int, wanted: List[String]
    ) raises -> Optional[List[List[String]]]:
        """Columns each join input must supply for the wanted output names.

        Keys are always kept. A right column whose output name carries the
        suffix keeps its colliding left column too, so no output is renamed.
        Returns None when the unpruned join would reject its output names,
        so execution still raises that error.
        """
        ref node = self._nodes[index]
        var left_cols = self._columns_of(node.left)
        var right_cols = self._columns_of(node.right)
        var suffix = node.names2[0]
        var membership = node.text == "semi" or node.text == "anti"
        var outputs = Dict[String, Bool]()
        for c in left_cols:
            outputs[c] = True
        var left_names = outputs.copy()
        var want = Dict[String, Bool]()
        for w in wanted:
            want[w] = True
        var keep_left = Dict[String, Bool]()
        var keep_right = Dict[String, Bool]()
        for k in node.names:
            keep_left[k] = True
            keep_right[k] = True
        for c in left_cols:
            if c in want:
                keep_left[c] = True
        if not membership:
            for c in right_cols:
                if c in keep_right:
                    continue
                var name = c + suffix if c in left_names else c
                if name in outputs:
                    return None
                outputs[name] = True
                if name in want:
                    keep_right[c] = True
                    if c in left_names:
                        keep_left[c] = True
        var left_keep = List[String]()
        for c in left_cols:
            if c in keep_left:
                left_keep.append(c)
        var right_keep = List[String]()
        for c in right_cols:
            if c in keep_right:
                right_keep.append(c)
        var sides = List[List[String]]()
        sides.append(left_keep^)
        sides.append(right_keep^)
        return sides^

    def _narrow_join_inputs(
        mut self,
        joins: List[Int],
        right_sides: List[Bool],
        columns: List[List[String]],
    ) raises:
        """Select only the kept columns above join inputs that are not scans.

        Scans already read only those columns. Other inputs (a filter, a
        computed column) would otherwise hand the join every column they
        produce. The selection is only added when it drops a column.
        """
        var added = False
        for k in range(len(joins)):
            var join = joins[k]
            var child = self._nodes[join].left
            if right_sides[k]:
                child = self._nodes[join].right
            if _is_scan(self._nodes[child].kind):
                continue
            var produced = self._columns_of(child)
            if len(columns[k]) >= len(produced):
                continue
            var exprs = List[Expr]()
            for name in columns[k]:
                exprs.append(col(name))
            self._nodes.append(_plan_node(SELECT, child, exprs=exprs))
            if right_sides[k]:
                self._nodes[join].right = len(self._nodes) - 1
            else:
                self._nodes[join].left = len(self._nodes) - 1
            added = True
        if added:
            self._reorder()

    def _describe(
        self, index: Int, depth: Int, mut out: String, streaming: Bool = True
    ):
        ref node = self._nodes[index]
        var pad = String("  ") * depth
        var label: String
        if node.kind == SCAN_FRAME:
            label = "SCAN frame"
        elif node.kind == SCAN_CSV:
            label = "SCAN CSV " + node.text
        elif node.kind == SCAN_PARQUET:
            label = "SCAN PARQUET " + node.text
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
        elif node.kind == EXPLODE:
            label = "EXPLODE " + _joined(node.names)
        elif node.kind == UNNEST:
            label = "UNNEST " + node.text
        else:
            label = "DROP " + _joined(node.names)
        if _is_scan(node.kind):
            if len(node.names) > 0:
                label += " [project " + _joined(node.names) + "]"
            if node.kind == SCAN_CSV and node.length >= 0:
                label += " [n_rows " + String(node.length) + "]"
        if streaming:
            if _is_scan(node.kind) or _stream_rows(node):
                label += " [stream]"
            elif (
                node.kind == AGG or node.kind == SELECT
            ) and _stream_reductions(node.exprs):
                label += " [stream aggregate state]"
            elif node.kind == JOIN and node.text in [
                "inner",
                "left",
                "semi",
                "anti",
                "cross",
            ]:
                if (
                    self._known_height(node.left) <= 65536
                    and (node.text == "inner" or node.text == "left")
                    and prefer_left_build(
                        self._known_height(node.left),
                        self._known_height(node.right),
                    )
                ):
                    label += " [small left input; materialize unless compact progression]"
                else:
                    label += " [stream probe; materialize build]"
            elif node.kind == SLICE and node.offset >= 0:
                label += " [ordered slice]"
            else:
                label += " [materialize]"
        out += pad + label + "\n"
        if node.left >= 0:
            self._describe(node.left, depth + 1, out, streaming)
        if node.right >= 0:
            self._describe(node.right, depth + 1, out, streaming)


def _is_scan(kind: Int) -> Bool:
    return kind == SCAN_FRAME or kind == SCAN_CSV or kind == SCAN_PARQUET


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


def scan_parquet(path: String) raises -> LazyFrame:
    """Lazily read a Parquet file. Nothing is read until collect; projection
    is pushed into the reader, and a filter directly above the scan decodes
    only the row groups whose footer statistics can hold a match."""
    return LazyFrame(
        [_plan_node(SCAN_PARQUET, text=path, offset=0)],
        [DataFrame([])],
        [Optional[CsvSchema]()],
    )
