"""Lazy query plans over the eager operators.

A LazyFrame records operations as a flat list of plan nodes (children always
precede parents). Nothing reads data until `collect`. Optimization rewrites
the plan before execution:

- filter splitting: a filter on `a & b` moves as two filters, and one on an
  OR of ANDs also gets the filters it implies on each input's columns
  (`(a1 & b1) | (a2 & b2)` implies `a1 | a2`); filters that end up
  together merge back into one;
- predicate pushdown: filters move below with_columns/select that do not
  produce the columns they read, below a sort when they are row-local, and
  into the side of an inner join (or the left side of a left/semi/anti join)
  that owns every column they read;
  row-local filters directly above unrestricted CSV scans run per decode range;
- projection pushdown: scans read only the columns the rest of the plan uses
  (CSV and Parquet scans decode only those fields); a join passes each input
  only its keys and the columns read above it from that side;
- slice pushdown: a head/slice directly over a CSV scan becomes `n_rows`;
- join order, when the plan runs: in a chain of inner joins, inputs that
  filters narrow run first, a selective join moves into the input holding
  its keys, and the most selective joins run first (`_order_joins`);
- row-group pruning: a row-local filter directly above a Parquet scan reads
  the footer statistics first and decodes only the row groups that can
  hold a match (see `parquet._pruned_row_groups`).

Filters never move past a slice, unique, group_by, or right/full join,
because that would change which rows those operators see.
"""
from std.os import getenv
from std.collections import Dict, Optional
from std.memory import ArcPointer
from .csv_reader import _CsvBatches, _DecodeJob
from .csv_types import _map_file
from .parquet import _ParquetBatches
from .parallel import Job, Pool, configured_workers, run_jobs
from .streaming import _StreamReduction, _StreamMergeJob, _finish_parts
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
    AND,
    COL,
    Node,
    OR,
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
from .hashing import encode_rows
from .trace import trace_path
from .join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_SEMI,
    join_code,
)
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
    # A JOIN's type code (see join_type); -1 for an unknown name, which the
    # eager join reports from the original text when the plan executes.
    var how: Int
    # A JOIN's right-side key names, paired with `names` (the left keys),
    # and whether key columns coalesce into the left names.
    var right_keys: List[String]
    var coalesce: Bool


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
    how: Int = -1,
    right_keys: List[String] = List[String](),
    coalesce: Bool = True,
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
        how,
        right_keys.copy(),
        coalesce,
    )


def _references(expr: Expr) -> Optional[List[String]]:
    """Columns an expression reads, or None when a selector reads any."""
    var names = List[String]()
    for node in expr._nodes:
        if node.op == SELECTOR:
            return None
        if node.op == COL:
            names.append(node.text)
        if node.op == OVER and node.text2.byte_length() > 0:
            # over() partitions read their key columns. Other operators
            # keep other text there (a strptime dtype, cut labels).
            for part in node.text2.split("\x1f"):
                names.append(String(part))
    return names^


def _output_names(exprs: List[Expr]) -> List[String]:
    var names = List[String]()
    for e in exprs:
        names.append(e._name)
    return names^


def _boolean_parts(expr: Expr, op: Int) -> List[Expr]:
    """The operands of a chain of `op` (AND or OR) at the root of expr, left
    to right; expr alone when its root is another operator."""
    var parts = List[Expr]()
    _collect_parts(expr, len(expr._nodes) - 1, op, parts)
    return parts^


def _collect_parts(expr: Expr, index: Int, op: Int, mut parts: List[Expr]):
    ref node = expr._nodes[index]
    if node.op == op and node.left >= 0 and node.right >= 0:
        _collect_parts(expr, node.left, op, parts)
        _collect_parts(expr, node.right, op, parts)
    else:
        parts.append(subtree(expr, index))


def _implied_filters(predicate: Expr) -> List[Expr]:
    """Filters that an OR of ANDs implies on fewer columns.

    `(a1 & b1) | (a2 & b2)` keeps a row only if some branch holds, and so
    only if `a1 | a2` holds, where a1 and a2 read one set of columns: that
    filter can move into the join input owning those columns, as DuckDB
    derives it. A branch with no part on those columns implies nothing.
    Under SQL's three-valued logic a true branch has every part true, so
    the derived filter keeps every row the predicate keeps.
    """
    var out = List[Expr]()
    var branches = _boolean_parts(predicate, OR)
    if len(branches) < 2:
        return out^
    var atoms = List[List[Expr]]()
    var reads = List[List[List[String]]]()
    for branch in branches:
        var parts = _boolean_parts(branch, AND)
        var part_reads = List[List[String]]()
        for part in parts:
            var r = _references(part)
            if not r:
                return out^
            part_reads.append(r.value().copy())
        atoms.append(parts^)
        reads.append(part_reads^)
    var tried = List[List[String]]()
    for a in range(len(atoms[0])):
        ref columns = reads[0][a]
        if len(columns) == 0:
            continue
        var seen = False
        for t in tried:
            seen = seen or (_covers(t, columns) and _covers(columns, t))
        if seen:
            continue
        tried.append(columns.copy())
        var implied = Expr(List[Node](), "")
        var found = True
        var partial = False
        for b in range(len(atoms)):
            var kept = Expr(List[Node](), "")
            for j in range(len(atoms[b])):
                if len(reads[b][j]) > 0 and _covers(columns, reads[b][j]):
                    kept = (
                        atoms[b][j].copy() if len(kept._nodes)
                        == 0 else kept & atoms[b][j]
                    )
                else:
                    partial = True
            if len(kept._nodes) == 0:
                found = False
                break
            implied = kept^ if len(implied._nodes) == 0 else implied | kept
        # A derived filter equal to the predicate adds nothing.
        if found and partial:
            out.append(implied^)
    return out^


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


def _stream_split_groups() -> Int:
    """Groups in the first batch at which streaming aggregation splits its
    state into hash parts. DATAFRAME_STREAM_SPLIT_GROUPS overrides it, so
    tests can exercise the split path on small inputs."""
    var setting = getenv("DATAFRAME_STREAM_SPLIT_GROUPS")
    if setting.byte_length() > 0:
        try:
            return max(1, Int(setting))
        except:
            pass
    return 4096


def _stream_split_bits(workers: Int) -> Int:
    """Hash parts for a split state: a power of two, at least the workers
    (so every worker has a part to merge) and at least 2."""
    var bits = 1
    while (1 << bits) < workers and bits < 8:
        bits += 1
    return bits


def _merge_parts(
    mut states: List[_StreamReduction],
    mut pending: List[List[_StreamReduction]],
    mut pending_groups: List[Int],
    workers: Int,
    force: Bool,
) raises:
    """Merge each part whose pending groups reached its accumulated count
    (or 64 batches), or every part when forced. One state merges in place;
    parts are spread over `workers` jobs, each merging its parts in turn."""
    var parts = len(states)
    var due = List[Bool](capacity=parts)
    var any_due = False
    for p in range(parts):
        var ready = len(pending[p]) > 0 and (
            force
            or states[p].indexing == 1
            or pending_groups[p] >= states[p].group_count()
            or len(pending[p]) >= 64
        )
        due.append(ready)
        any_due = any_due or ready
    if not any_due:
        return
    if parts == 1:
        states[0].merge_all(pending[0])
        pending[0] = List[_StreamReduction]()
        pending_groups[0] = 0
        return
    var job_count = max(1, min(workers, parts))
    var jobs = List[_StreamMergeJob](capacity=job_count)
    for _ in range(job_count):
        jobs.append(_StreamMergeJob())
    # Part p goes to job p % job_count; parts not due keep their pending
    # batches here and ride along with nothing to merge.
    var carried = List[List[_StreamReduction]](capacity=parts)
    for p in range(parts):
        var state = states.pop(0)
        var waiting = pending.pop(0)
        ref job = jobs[p % job_count]
        job.states.append(state^)
        if due[p]:
            job.pending.append(waiting^)
            carried.append(List[_StreamReduction]())
            pending_groups[p] = 0
        else:
            job.pending.append(List[_StreamReduction]())
            carried.append(waiting^)
    run_jobs(jobs)
    for p in range(parts):
        states.append(jobs[p % job_count].states.pop(0))
        pending.append(carried.pop(0))


struct _StreamJob(Job):
    var decode: List[_DecodeJob]
    var frame: DataFrame
    var operations: List[PlanNode]
    var joins: ArcPointer[List[DataFrame]]
    var indexes: ArcPointer[List[Optional[PreparedHashIndex]]]
    var expressions: List[Expr]
    var keys: List[String]
    var reduced: List[_StreamReduction]
    # Split a grouped state into 2^bits hash parts (0: keep one state), and
    # the rows it was reduced from.
    var bits: Int
    var rows: Int

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
        self.bits = 0
        self.rows = 0

    def take_reduced(mut self) -> List[_StreamReduction]:
        var out = self.reduced^
        self.reduced = List[_StreamReduction]()
        return out^

    def run(mut self) raises:
        if len(self.decode):
            self.decode[0].run()
            self.frame = self.decode.pop().into_frame()
        for node in self.operations:
            if node.kind == JOIN and self.indexes[][node.offset]:
                self.frame = self.frame._join_impl(
                    self.joins[][node.offset],
                    left_on=node.names,
                    right_on=node.right_keys,
                    how=node.how,
                    suffix=node.names2[0],
                    coalesce=node.coalesce,
                    prepared=self.indexes[][node.offset],
                )
            elif node.kind == JOIN:
                self.frame = self.frame.join(
                    self.joins[][node.offset],
                    left_on=node.names,
                    right_on=node.right_keys,
                    how=node.text,
                    suffix=node.names2[0],
                    coalesce=node.coalesce,
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
            elif node.kind == SORT:
                # A sort limited to its first rows (#332): this batch's own
                # first rows, selected on this worker alone.
                var n = len(node.names)
                self.frame = self.frame.take(
                    self.frame._arg_sort_head(
                        node.names,
                        List[Bool](node.flags[:n]),
                        List[Bool](node.flags[n:]),
                        node.length,
                        threads=1,
                    )
                )
        if len(self.expressions):
            var reduction = _StreamReduction(
                self.frame, self.expressions, self.keys
            )
            self.rows = reduction.rows
            if self.bits > 0 and reduction.grouped:
                self.reduced = reduction.split(self.bits)
            else:
                self.reduced.append(reduction^)
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

    def sort(
        self,
        by: List[String],
        *,
        descending: List[Bool],
        nulls_last: List[Bool],
    ) raises -> Self:
        """Sort with a direction and null placement per key, as the eager
        DataFrame.sort takes them."""
        if len(descending) != len(by) or len(nulls_last) != len(by):
            raise Error("sort needs one descending and nulls_last flag per key")
        return self._push(
            _plan_node(SORT, names=by, flags=descending + nulls_last.copy())
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
        coalesce: Bool = True,
    ) -> Self:
        """Join with another lazy plan on keys named alike on both sides;
        see DataFrame.join."""
        return self.join(
            other,
            left_on=on,
            right_on=on,
            how=how,
            suffix=suffix,
            coalesce=coalesce,
        )

    def join(
        self,
        other: Self,
        *,
        left_on: List[String],
        right_on: List[String],
        how: String = "inner",
        suffix: String = "_right",
        coalesce: Bool = True,
    ) -> Self:
        """Join with another lazy plan on paired keys: left_on[i] matches
        right_on[i], as in DataFrame.join. Projection and predicate
        pushdown see both key lists, so each input reads only its own keys
        and the columns used above the join."""
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
            JOIN,
            left_root,
            len(result._nodes) - 1,
            names=left_on,
            text=how,
            how=join_code(how),
            right_keys=right_on,
            coalesce=coalesce,
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
        coalesce: Bool = True,
    ) -> Self:
        return self.join(other, [on], how, suffix, coalesce)

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
        if optimize:
            plan._order_joins(streaming, batch_size)
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

    def _sampled_groups(self, keys: List[String], frame: Int) raises -> Int:
        """Distinct grouping keys among 4,096 evenly spaced rows of an
        in-memory frame, or -1 when the frame does not hold every key (one
        made by an earlier step).
        """
        ref source = self._frames[frame]
        var height = source.height()
        var sample = min(height, 4096)
        if sample == 0:
            return -1
        var rows = List[Int](capacity=sample)
        for k in range(sample):
            rows.append(k * height // sample)
        var picked = List[Series](capacity=len(keys))
        for name in keys:
            if name not in source.columns():
                return -1
            picked.append(source.column(name).take(rows))
        return encode_rows(picked, nulls_equal=True).count()

    def _joins_below(self, index: Int) -> Bool:
        """Whether a join sits under `index` along its left inputs, through
        the row-local steps a stream applies."""
        var cursor = index
        while cursor >= 0:
            ref node = self._nodes[cursor]
            if node.kind == JOIN:
                return True
            if _is_scan(node.kind):
                return False
            cursor = node.left
        return False

    def _height_bound(self, index: Int) -> Int:
        """An upper bound on the rows a node yields without executing it:
        the scanned frame's height through steps that only drop or keep
        rows; -1 when unknown (joins, aggregations, file scans)."""
        ref node = self._nodes[index]
        if node.kind == SCAN_FRAME:
            return self._frames[node.offset].height()
        if (
            node.kind == FILTER
            or node.kind == DROP
            or (
                (node.kind == SELECT or node.kind == WITH_COLUMNS)
                and _stream_rows(node)
            )
        ):
            return self._height_bound(node.left)
        if node.kind == SLICE:
            var height = self._height_bound(node.left)
            if height >= 0 and node.length >= 0:
                return min(height, node.length)
            return height
        return -1

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
        # A sort limited to its first rows (#332) streams: each batch keeps
        # its own first rows, which contain the overall first rows, since a
        # row among those is among its own batch's. Batches arrive in input
        # order, so ties keep the stable sort's order.
        var top = -1
        var top_node = -1
        var top_names = List[String]()
        var top_descending = List[Bool]()
        var top_nulls_last = List[Bool]()
        if (
            terminal.kind == SLICE
            and cursor >= 0
            and self._nodes[cursor].kind == SORT
            and self._nodes[cursor].length >= 0
        ):
            top = self._nodes[cursor].length
            top_node = cursor
            top_names = self._nodes[cursor].names.copy()
            var n = len(top_names)
            for k in range(n):
                top_descending.append(self._nodes[cursor].flags[k])
                top_nulls_last.append(self._nodes[cursor].flags[n + k])
            cursor = self._nodes[cursor].left
        var candidates = List[DataFrame]()
        var candidate_rows = 0
        var operations = List[PlanNode]()
        var joins = List[DataFrame]()
        var indexes = List[Optional[PreparedHashIndex]]()
        while cursor >= 0:
            ref node = self._nodes[cursor]
            if _stream_rows(node):
                operations.append(node.copy())
            elif node.kind == JOIN and node.how in [
                JOIN_INNER,
                JOIN_LEFT,
                JOIN_SEMI,
                JOIN_ANTI,
                JOIN_CROSS,
            ]:
                var prepared = Optional[PreparedHashIndex]()
                # A left input that can only be small next to a large known
                # right may be the side to hash: decline streaming, so the
                # plan runs eagerly and `DataFrame.join` builds on the smaller
                # side, as Polars' `det_hash_prone_order` and DuckDB's
                # build/probe optimizer do (#377). Streaming would build on
                # the right: PDS-H q17 hashed 6M lineitem rows to match the
                # few hundred parts its filter keeps.
                var bound = self._height_bound(node.left)
                var right_rows = self._known_height(node.right)
                # A left input made by another join has no bound; how many
                # rows it holds is known only once it has run.
                var unbounded = bound < 0 and self._joins_below(node.left)
                if (
                    (node.how == JOIN_INNER or node.how == JOIN_LEFT)
                    and self._known_height(node.left) < 0
                    and right_rows >= 524288
                    and (unbounded or (bound >= 0 and 4 * bound <= right_rows))
                ):
                    # The bound only says the left side may be small; the
                    # eager join then decides with the real heights (a
                    # filter often keeps far fewer rows than it scans).
                    if len(joins) == 0:
                        return None
                    # Joins above this one stream faster than they run
                    # eagerly (PDS-H q8 and q9 were 1.5 times slower eager),
                    # so only this join leaves the stream: it becomes the
                    # stream's source, executed as a plan of its own, where
                    # the rule above runs it eagerly. PDS-H q8 hashed 6M
                    # lineitem rows to match the 1,300 parts its filter
                    # keeps. Leaving the stream costs the joins above their
                    # prepared probe of this one's output, so it takes a
                    # wider margin: PDS-H q16 (200K parts at most, 800K
                    # partsupp rows) was 10% slower leaving it. A left input
                    # made by other joins has no bound and leaves too: PDS-H
                    # q5 hashed 6M lineitem rows for about 45K orders, and
                    # q18, q11, q7 and q2 gained 15-44%. Where that input is
                    # large, streaming was faster: q9's partsupp and orders
                    # joins (about 320K rows on the left) are 14% slower.
                    if unbounded or 8 * bound <= right_rows:
                        break
                if (
                    self._known_height(node.left) <= batch_size
                    and (node.how == JOIN_INNER or node.how == JOIN_LEFT)
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
                    if (
                        build_node.kind == SCAN_FRAME
                        and len(node.right_keys) == 1
                    ):
                        var sources: List[Series] = [
                            self._frames[build_node.offset][
                                node.right_keys[0]
                            ].copy()
                        ]
                        prepared = prepare_progression_index(sources)
                    if not prepared:
                        break
                var operation = node.copy()
                operation.offset = len(joins)
                var built = self._execute(node.right, False, True, batch_size)
                if (
                    (node.how == JOIN_SEMI or node.how == JOIN_ANTI)
                    and built.height() >= 524288
                    and self._known_height(node.left) < 0
                ):
                    # A semi or anti join whose built right side turned out
                    # large: run the left too, and decide from both real
                    # sizes, as DuckDB does. A left at most a quarter of the
                    # right is joined eagerly, hashing the left (PDS-H q4:
                    # about 57K orders against 3.8M late lineitem rows);
                    # otherwise the stream continues from the left's rows.
                    # Either way both results replace their nodes in a copy
                    # of the plan, so neither side runs twice.
                    var left_frame = self._execute(
                        node.left, False, True, batch_size
                    )
                    var plan = self.copy()
                    if 4 * left_frame.height() <= built.height():
                        var joined = left_frame.join(
                            built,
                            left_on=node.names,
                            right_on=node.right_keys,
                            how="semi" if node.how == JOIN_SEMI else "anti",
                        )
                        plan._frames.append(joined^)
                        plan._schemas.append(Optional[CsvSchema]())
                        plan._nodes[cursor] = _plan_node(
                            SCAN_FRAME, offset=len(plan._frames) - 1
                        )
                    else:
                        plan._frames.append(left_frame^)
                        plan._schemas.append(Optional[CsvSchema]())
                        plan._nodes.append(
                            _plan_node(SCAN_FRAME, offset=len(plan._frames) - 1)
                        )
                        plan._nodes[cursor].left = len(plan._nodes) - 1
                        plan._frames.append(built^)
                        plan._schemas.append(Optional[CsvSchema]())
                        plan._nodes.append(
                            _plan_node(SCAN_FRAME, offset=len(plan._frames) - 1)
                        )
                        plan._nodes[cursor].right = len(plan._nodes) - 1
                    return plan._execute(index, False, True, batch_size)
                joins.append(built^)
                if not prepared and node.how != JOIN_CROSS and len(node.names):
                    ref build = joins[len(joins) - 1]
                    var sources = List[Series]()
                    var supported = build.height() <= Int(Int32.MAX)
                    for name in node.right_keys:
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
        # Grouped aggregations over an in-memory frame, filtered or
        # projected at most: whether the eager group-by is faster.
        var eager = False
        if (
            len(expressions)
            and self._nodes[cursor].kind == SCAN_FRAME
            and _row_steps_only(operations)
        ):
            if _counts_distinct(expressions):
                eager = True
            elif len(keys) > 0:
                if _filters(operations):
                    eager = True
                else:
                    var frame = self._nodes[cursor].offset
                    var sampled = self._sampled_groups(keys, frame)
                    var sample = min(self._frames[frame].height(), 4096)
                    eager = (
                        sampled < 0 or sampled <= 192 or 2 * sampled > sample
                    )
        if eager:
            # Over a frame already in memory, filtered or projected at most,
            # batches save no memory, and most grouped aggregations run
            # faster eagerly, measured on 10M ClickBench rows (#381):
            # - n_unique keeps per-group sets in batch states that the eager
            #   aggregation replaces with hash partitions on every worker
            #   (#336).
            # - After a filter, the eager group-by only sees the kept rows,
            #   where every batch state holds and merges its groups again:
            #   non-empty SearchPhrase counts take 66 ms against 358
            #   streamed, and q39's five keys 99 against 371.
            # - Keys made by an earlier step, which the sample cannot read:
            #   q18's (UserID, minute, SearchPhrase) take 354 ms against
            #   1,645 streamed.
            # - Few groups (at most 192 in the sample) take 33 ms against
            #   64; ResolutionWidth (134 in the sample) 35 against 67.
            # - Many groups (more than half the sample) are encoded once by
            #   hash partition, where every batch state would hold most of
            #   them: on 1M rows URL takes 40 ms against 63 streamed.
            # Unfiltered keys in between keep streaming, which is faster
            # there: RegionID (364 in the sample) takes 75 ms against 99,
            # and SearchPhrase (574, one dominant value) 353 against 375,
            # or on 1M rows (212) 8 against 13.
            # Ungrouped reductions stream, as do plans with a join, whose
            # batched probes beat the eager join.
            return None
        if top >= 0:
            # Over a frame already in memory with nothing to apply first,
            # batches save no memory, and the eager selection already works
            # in cache-sized chunks on every worker.
            if len(operations) == 0 and self._nodes[cursor].kind == SCAN_FRAME:
                return None
            operations.append(self._nodes[top_node].copy())
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
        # Aggregate state (#326). Batch states wait in `pending` and are
        # merged together once their groups reach the accumulated count (or
        # 64 batches), so each merge covers at least as much new work as old
        # and total merge work stays linear, with pending state about the
        # size of the accumulated state. When the first batch shows many
        # groups, every batch state is split into hash parts that are merged
        # on separate workers and interleaved by first occurrence at the end.
        var states = List[_StreamReduction]()
        var pending = List[List[_StreamReduction]]()
        var pending_groups = List[Int]()
        var bits = 0
        var rows_seen = 0
        var split_groups = _stream_split_groups()
        # Batches of a large in-memory input that streams through joins grow
        # up to four times the default: each batch probes every join, and
        # fewer, larger batches cost less per row (PDS-H q21's two joins of
        # 3.8M rows against 1.5M-row builds: 186 -> 148 ms). They grow only
        # while every worker still gets a batch: a 150K-row customer table
        # in one batch ran on one worker (q10, q13 and q22 were 7-23% slower).
        var rows_per_batch = batch_size
        if len(shared_joins[]) > 0 and input.height() > 0:
            rows_per_batch = max(
                batch_size, min(4 * batch_size, input.height() // workers)
            )
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
                    frame = input.slice(offset, rows_per_batch)
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
                job.bits = bits
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
                    var pieces = jobs[i].take_reduced()
                    for k in range(len(pieces)):
                        pieces[k].shift_firsts(rows_seen)
                    rows_seen += jobs[i].rows
                    if (
                        len(states) == 0
                        and pieces[0].grouped
                        and pieces[0].group_count() >= split_groups
                    ):
                        bits = _stream_split_bits(workers)
                    if bits > 0 and len(pieces) == 1 and pieces[0].grouped:
                        var whole = pieces.pop()
                        pieces = whole.split(bits)
                    if len(states) == 0:
                        for _ in range(len(pieces)):
                            pending.append(List[_StreamReduction]())
                            pending_groups.append(0)
                        states = pieces^
                        continue
                    for p in range(len(pending)):
                        var piece = pieces.pop(0)
                        pending_groups[p] += piece.group_count()
                        pending[p].append(piece^)
                elif top >= 0:
                    # The job has already cut its batch to its first rows.
                    candidate_rows += jobs[i].frame.height()
                    candidates.append(jobs[i].frame.copy())
                else:
                    var part = jobs[i].frame.copy()
                    var dropped = min(skip, part.height())
                    skip -= dropped
                    part = part.slice(dropped, limit)
                    if limit >= 0:
                        limit -= part.height()
                    outputs.append(part^)
            if len(states):
                _merge_parts(states, pending, pending_groups, workers, False)
            if top >= 0 and len(candidates) > 1 and candidate_rows > 4 * top:
                var merged = concat(candidates)
                merged = merged.take(
                    merged._arg_sort_head(
                        top_names, top_descending, top_nulls_last, top
                    )
                )
                candidate_rows = merged.height()
                candidates = [merged^]
            if len(csv):
                csv[0].discard()
            if limit == 0 and top < 0:
                ended = True
        pool.release()
        if len(states):
            _merge_parts(states, pending, pending_groups, workers, True)
            if len(states) == 1:
                return states[0].finish()
            return _finish_parts(states^)
        if top >= 0:
            if len(candidates) == 0:
                return None
            var merged = concat(candidates)
            merged = merged.take(
                merged._arg_sort_head(
                    top_names, top_descending, top_nulls_last, top
                )
            )
            return merged.slice(skip, limit)
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
                    var order = source._arg_sort_head(
                        sorted.names, descending, nulls_last, sorted.length
                    ) if sorted.length >= 0 else source.arg_sort(
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
            return input.join(
                right,
                left_on=node.names,
                right_on=node.right_keys,
                how=node.text,
                suffix=node.names2[0],
                coalesce=node.coalesce,
            )
        if node.kind == SORT:
            var n = len(node.names)
            var descending = List[Bool]()
            var nulls_last = List[Bool]()
            for i in range(n):
                descending.append(node.flags[i])
                nulls_last.append(node.flags[n + i])
            if node.length >= 0:
                return input.take(
                    input._arg_sort_head(
                        node.names, descending, nulls_last, node.length
                    )
                )
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
        plan._split_filters()
        plan._push_predicates()
        plan._merge_filters()
        plan._push_slices()
        plan._fuse_top_k()
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

    def _split_filters(mut self):
        """A row-local filter on `a & b` becomes a filter on b over one on
        a, so predicate pushdown moves each part as far as it can go; a
        filter on an OR of ANDs also gets the single-input filters it
        implies (`_implied_filters`). `_merge_filters` rejoins the parts
        that end up together."""
        var count = len(self._nodes)
        var added = False
        for i in range(count):
            if self._nodes[i].kind != FILTER:
                continue
            if not _row_local(self._nodes[i].exprs):
                continue
            var parts = _boolean_parts(self._nodes[i].exprs[0], AND)
            var extra = List[Expr]()
            for part in parts:
                for implied in _implied_filters(part):
                    extra.append(implied.copy())
            if len(parts) + len(extra) < 2:
                continue
            var below = self._nodes[i].left
            for implied in extra:
                self._nodes.append(
                    _plan_node(FILTER, below, exprs=[implied.copy()])
                )
                below = len(self._nodes) - 1
            for k in range(len(parts) - 1, 0, -1):
                self._nodes.append(
                    _plan_node(FILTER, below, exprs=[parts[k].copy()])
                )
                below = len(self._nodes) - 1
            self._nodes[i].left = below
            self._nodes[i].exprs = [parts[0].copy()]
            added = True
        if added:
            self._reorder()

    def _merge_filters(mut self):
        """Join a row-local filter directly over another into one filter
        on both predicates, lower first, so a scan below sees them all (CSV
        range filters, Parquet row-group pruning)."""
        var changed = True
        while changed:
            changed = False
            for i in range(len(self._nodes)):
                ref node = self._nodes[i]
                if node.kind != FILTER or node.left < 0:
                    continue
                ref below = self._nodes[node.left]
                if below.kind != FILTER:
                    continue
                if not (_row_local(node.exprs) and _row_local(below.exprs)):
                    continue
                var merged = below.exprs[0] & node.exprs[0]
                var child = below.left
                self._nodes[i].exprs = [merged^]
                self._nodes[i].left = child
                self._reorder()
                changed = True
                break

    def _order_joins(mut self, streaming: Bool, batch_size: Int) raises:
        """Order each chain of inner joins by how much their inputs narrow.

        A chain is a run of inner joins, filters and plain column selections
        along left inputs, over one base input. Run when the plan executes,
        not in `explain`: a join's right input is executed first when it
        is a known table narrowed by filters or joins, and how many of the
        table's rows it kept is its selectivity, as DuckDB's join order
        optimizer estimates it from statistics. Then:

        - A join whose keys come from one earlier join's right input, that
          keeps under half its table and holds at most a quarter as many
          rows as that input's table, joins that input instead (a bushy
          plan): PDS-H q7 joins customer to the two nations its
          filter allows, and orders to those customers, before any
          lineitem row is probed.
        - The remaining joins run most selective first, each once its keys
          are available, with every filter as soon as its columns are.

        Inputs run here replace their subtrees with their results, so
        nothing runs twice. Projection pushdown runs again afterwards, as
        the chain's column selections are dropped and the chain ends with
        one selecting its original output.
        """
        var parents = self._parents()
        var changed = False
        for top in range(len(parents)):
            if not _chain_member(self._nodes[top]):
                continue
            var parent = parents[top]
            if (
                parent >= 0
                and self._nodes[parent].left == top
                and _chain_member(self._nodes[parent])
            ):
                continue
            if self._order_chain(top, streaming, batch_size):
                changed = True
        if changed:
            self._reorder()
            self._push_projections()

    def _spine_rows(self, index: Int) -> Int:
        """Rows of the table under `index` along steps that keep, drop or
        join rows (a filter, a projection, an inner join's left input);
        -1 through anything else (an aggregation) or a file scan."""
        var cursor = index
        while cursor >= 0:
            ref node = self._nodes[cursor]
            if node.kind == SCAN_FRAME:
                return self._frames[node.offset].height()
            if (
                node.kind == FILTER
                or node.kind == DROP
                or (node.kind == JOIN and node.how == JOIN_INNER)
                or (
                    (node.kind == SELECT or node.kind == WITH_COLUMNS)
                    and _stream_rows(node)
                )
            ):
                cursor = node.left
                continue
            return -1
        return -1

    def _measurable(self, index: Int) -> Bool:
        """Whether `_selectivity` would run node `index`: a known table,
        narrowed by steps above it."""
        return (
            self._nodes[index].kind != SCAN_FRAME
            and self._spine_rows(index) > 0
        )

    def _selectivity(
        mut self, index: Int, streaming: Bool, batch_size: Int
    ) raises -> Float64:
        """The share of its table's rows that node `index` keeps, executing
        it and replacing it with its result; 1 when it is a plain table or
        no table is known."""
        var table = self._spine_rows(index)
        if table <= 0 or self._nodes[index].kind == SCAN_FRAME:
            return 1.0
        var frame = self._execute(index, False, streaming, batch_size)
        var rows = frame.height()
        self._frames.append(frame^)
        self._schemas.append(Optional[CsvSchema]())
        self._nodes[index] = _plan_node(
            SCAN_FRAME, offset=len(self._frames) - 1
        )
        return Float64(rows) / Float64(table)

    def _order_chain(
        mut self, top: Int, streaming: Bool, batch_size: Int
    ) raises -> Bool:
        """Reorder the chain whose last node is `top`; see `_order_joins`.
        False, leaving it unchanged, when nothing would move or column
        names could resolve differently in another order."""
        var spine = List[Int]()
        var cursor = top
        while cursor >= 0 and _chain_member(self._nodes[cursor]):
            spine.append(cursor)
            cursor = self._nodes[cursor].left
        var base = cursor
        if base < 0:
            return False
        # Steps bottom to top, without the column selections.
        var steps = List[Int]()
        var join_count = 0
        for k in range(len(spine)):
            var i = spine[len(spine) - 1 - k]
            if self._nodes[i].kind == SELECT:
                continue
            steps.append(i)
            if self._nodes[i].kind == JOIN:
                join_count += 1
        if join_count < 2:
            return False
        var top_columns = self._columns_of(top)
        # Which step supplies each column (-1 the base). Every name must be
        # unique across the chain, so no join renames a column in any order.
        var provider = Dict[String, Int]()
        for c in self._columns_of(base):
            provider[c] = -1
        var outputs = List[List[String]]()
        for p in range(len(steps)):
            ref node = self._nodes[steps[p]]
            var produced = List[String]()
            if node.kind == JOIN:
                for c in self._columns_of(node.right):
                    if c in node.right_keys:
                        continue
                    if c in provider:
                        return False
                    provider[c] = p
                    produced.append(c)
            outputs.append(produced^)
        var needs = List[List[String]]()
        for p in range(len(steps)):
            ref node = self._nodes[steps[p]]
            if node.kind == JOIN:
                needs.append(node.names.copy())
            else:
                var reads = _references(node.exprs[0])
                if not reads:
                    return False
                needs.append(reads.value().copy())
            for name in needs[p]:
                if name not in provider:
                    return False
        # owner[p]: the step whose output now carries step p's columns.
        var owner = List[Int](capacity=len(steps))
        for p in range(len(steps)):
            owner.append(p)
        var selectivity = List[Float64](length=len(steps), fill=1.0)
        var measured = List[Bool](length=len(steps), fill=False)
        var pushed = False
        # Measuring runs inputs, and an input with a known size changes
        # how the stream executes joins; an unchanged chain is restored.
        var saved = self._nodes.copy()
        for reverse in range(len(steps)):
            var p = len(steps) - 1 - reverse
            if self._nodes[steps[p]].kind != JOIN:
                continue
            var keys = self._nodes[steps[p]].names.copy()
            var host = -2
            for name in keys:
                var from_step = provider[name]
                var o = owner[from_step] if from_step >= 0 else -1
                if host == -2:
                    host = o
                elif host != o:
                    host = -1
            if host < 0:
                continue
            # The host's right input must not hold a name this join adds.
            var host_columns = self._columns_of(self._nodes[steps[host]].right)
            var clash = False
            for c in outputs[p]:
                clash = clash or c in host_columns
            if clash:
                continue
            # The pushed input must be small next to the host's table, as a
            # dimension of it: joining 1.5M lineitem rows into 57K orders
            # first made PDS-H q10 12% slower. Its table bounds its rows
            # before it runs, and its rows bound them after.
            var host_rows = self._spine_rows(self._nodes[steps[host]].right)
            var table = self._spine_rows(self._nodes[steps[p]].right)
            if host_rows < 0 or table < 0 or 4 * table > host_rows:
                continue
            var share = self._selectivity(
                self._nodes[steps[p]].right, streaming, batch_size
            )
            selectivity[p] = share
            measured[p] = True
            if share >= 0.5:
                continue
            var rows = self._known_height(self._nodes[steps[p]].right)
            if rows < 0 or 4 * rows > host_rows:
                continue
            var inner = self._nodes[steps[p]].copy()
            inner.left = self._nodes[steps[host]].right
            self._nodes.append(inner^)
            self._nodes[steps[host]].right = len(self._nodes) - 1
            for q in range(len(steps)):
                if owner[q] == p:
                    owner[q] = host
            pushed = True
        # Selectivity of the joins that stay in the chain, measured only
        # where it can change the order: a narrowed input whose join could
        # run before one that precedes it, and that one. A plain table keeps
        # all its rows and never moves ahead.
        var need = List[Bool](length=len(steps), fill=False)
        var done = List[Bool](length=len(steps), fill=False)
        for p in range(len(steps)):
            if owner[p] != p:
                done[p] = True
        for p in range(len(steps)):
            if owner[p] != p:
                continue
            if self._nodes[steps[p]].kind == JOIN:
                for q in range(p + 1, len(steps)):
                    if (
                        owner[q] != q
                        or self._nodes[steps[q]].kind != JOIN
                        or not self._available(needs[q], provider, owner, done)
                    ):
                        continue
                    if self._measurable(self._nodes[steps[q]].right):
                        need[q] = True
                        need[p] = need[p] or self._measurable(
                            self._nodes[steps[p]].right
                        )
            done[p] = True
        for p in range(len(steps)):
            if not need[p] or measured[p]:
                continue
            selectivity[p] = self._selectivity(
                self._nodes[steps[p]].right, streaming, batch_size
            )
        # Greedy order: filters as soon as their columns are available, then
        # the most selective available join (ties keep the original order).
        var placed = List[Bool](length=len(steps), fill=False)
        for p in range(len(steps)):
            if owner[p] != p:
                placed[p] = True
        var order = List[Int]()
        var remaining = 0
        for p in range(len(steps)):
            if not placed[p]:
                remaining += 1
        while len(order) < remaining:
            var progress = False
            for p in range(len(steps)):
                if placed[p] or self._nodes[steps[p]].kind != FILTER:
                    continue
                if self._available(needs[p], provider, owner, placed):
                    placed[p] = True
                    order.append(p)
                    progress = True
            var best = -1
            for p in range(len(steps)):
                if placed[p] or self._nodes[steps[p]].kind != JOIN:
                    continue
                if not self._available(needs[p], provider, owner, placed):
                    continue
                if best < 0 or selectivity[p] < selectivity[best]:
                    best = p
            if best >= 0:
                placed[best] = True
                order.append(best)
                progress = True
            if not progress:
                self._nodes = saved^
                return False
        var moved = pushed
        var expected = 0
        for p in range(len(steps)):
            if owner[p] != p:
                continue
            if order[expected] != p:
                moved = True
            expected += 1
        if not moved:
            self._nodes = saved^
            return False
        var current = base
        for p in order:
            var node = self._nodes[steps[p]].copy()
            node.left = current
            self._nodes.append(node^)
            current = len(self._nodes) - 1
        var keep = List[Expr]()
        for name in top_columns:
            keep.append(col(name))
        self._nodes[top] = _plan_node(SELECT, current, exprs=keep)
        trace_path("lazy.join_order")
        return True

    def _available(
        self,
        names: List[String],
        provider: Dict[String, Int],
        owner: List[Int],
        placed: List[Bool],
    ) raises -> Bool:
        """Whether every name comes from the base or a placed step."""
        for name in names:
            var p = provider[name]
            if p >= 0 and not placed[owner[p]]:
                return False
        return True

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
                    below.how == JOIN_INNER
                    or below.how == JOIN_LEFT
                    or below.how == JOIN_SEMI
                    or below.how == JOIN_ANTI
                ):
                    var left_cols = self._columns_of(below.left)
                    var right_cols = List[String]()
                    for c in self._columns_of(below.right):
                        # Coalesced right keys are not in the join output,
                        # so no filter above can mean them.
                        if not (below.coalesce and c in below.right_keys):
                            right_cols.append(c)
                    var side = -1
                    if _covers(left_cols, reads.value()):
                        side = 0
                    elif below.how == JOIN_INNER and _covers_exclusive(
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

    def _fuse_top_k(mut self):
        """A slice directly above a sort needs only the sort's first
        offset + length rows (SQL's ORDER BY ... LIMIT), so the sort records
        that limit and selects those rows instead of sorting them all
        (#332). The slice stays to apply its offset; a sort holds its limit
        in `length`, which it otherwise leaves at -1."""
        for i in range(len(self._nodes)):
            ref node = self._nodes[i]
            if node.kind != SLICE or node.offset < 0 or node.length < 0:
                continue
            ref below = self._nodes[node.left]
            if below.kind != SORT:
                continue
            var limit = node.offset + node.length
            if below.length < 0 or limit < below.length:
                self._nodes[node.left].length = limit

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
        var membership = node.how == JOIN_SEMI or node.how == JOIN_ANTI
        var outputs = Dict[String, Bool]()
        for c in left_cols:
            outputs[c] = True
        var left_names = outputs.copy()
        var want = Dict[String, Bool]()
        for w in wanted:
            want[w] = True
        if not node.coalesce:
            # Right keys then appear in the output as well; do not prune.
            return None
        var keep_left = Dict[String, Bool]()
        var keep_right = Dict[String, Bool]()
        for k in node.names:
            keep_left[k] = True
        for k in node.right_keys:
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
            label = "JOIN " + node.text + " on " + _key_pairs(node)
        elif node.kind == SORT and node.length >= 0:
            label = (
                "TOP_K " + String(node.length) + " by " + _joined(node.names)
            )
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
            elif node.kind == JOIN and node.how in [
                JOIN_INNER,
                JOIN_LEFT,
                JOIN_SEMI,
                JOIN_ANTI,
                JOIN_CROSS,
            ]:
                if (
                    self._known_height(node.left) <= 65536
                    and (node.how == JOIN_INNER or node.how == JOIN_LEFT)
                    and prefer_left_build(
                        self._known_height(node.left),
                        self._known_height(node.right),
                    )
                ):
                    label += " [small left input; materialize unless compact progression]"
                else:
                    label += " [stream probe; materialize build]"
            elif node.kind == SORT and node.length >= 0:
                label += " [stream top-k state]"
            elif node.kind == SLICE and node.offset >= 0:
                label += " [ordered slice]"
            else:
                label += " [materialize]"
        out += pad + label + "\n"
        if node.left >= 0:
            self._describe(node.left, depth + 1, out, streaming)
        if node.right >= 0:
            self._describe(node.right, depth + 1, out, streaming)


def _row_steps_only(operations: List[PlanNode]) -> Bool:
    for operation in operations:
        if operation.kind not in [FILTER, SELECT, WITH_COLUMNS, DROP]:
            return False
    return True


def _filters(operations: List[PlanNode]) -> Bool:
    for operation in operations:
        if operation.kind == FILTER:
            return True
    return False


def _counts_distinct(expressions: List[Expr]) -> Bool:
    for expression in expressions:
        for node in expression._nodes:
            if node.op == N_UNIQUE:
                return True
    return False


def _chain_member(node: PlanNode) -> Bool:
    """A step `_order_chain` can move: an inner join with coalesced keys, a
    row-local filter, or a selection of plain columns."""
    if node.kind == JOIN:
        return node.how == JOIN_INNER and node.coalesce and len(node.names) > 0
    if node.kind == FILTER:
        return _row_local(node.exprs)
    if node.kind == SELECT:
        for e in node.exprs:
            if len(e._nodes) != 1 or e._nodes[0].op != COL:
                return False
            if e._nodes[0].text != e._name:
                return False
        return True
    return False


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


def _key_pairs(node: PlanNode) -> String:
    """`a` for keys named alike, `a = b` for differently named pairs."""
    var out = String()
    for i in range(len(node.names)):
        if i > 0:
            out += ", "
        out += node.names[i]
        if i < len(node.right_keys) and node.right_keys[i] != node.names[i]:
            out += " = " + node.right_keys[i]
    return out^


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
