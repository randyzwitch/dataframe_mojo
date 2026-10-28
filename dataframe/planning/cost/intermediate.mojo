"""Bounded-work intermediate statistics for cost choices, never safety proofs.

Each request owns its memo. Unknown statistics stay unknown; computed
expressions are not executed to discover distributions. Source NDVs use at most 4,096 jittered observations; filters retain the
existing sample of at most 65,536 rows.
"""
from std.collections import Dict
from std.io import RawFileDescriptor
from std.os import getenv
from dataframe.aggregation.expressions import _row_local, _stream_reductions
from dataframe.frame import DataFrame
from dataframe.dtype import DataType
from dataframe.kernels.hashing import encode_rows
from dataframe.series import Series
from dataframe.expr import COL
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_SEMI,
)
from dataframe.planning.context import QueryContext
from dataframe.planning.expression_inputs import _references
from dataframe.planning.model import (
    AGG,
    CONCAT,
    DROP,
    FILTER,
    JOIN,
    SCAN_FRAME,
    SELECT,
    SLICE,
    SORT,
    UNIQUE,
    WITH_COLUMNS,
)
from dataframe.planning.properties import _stream_rows
from dataframe.storage.boolean import true_count


@fieldwise_init
struct RowEstimate(Copyable, Movable):
    """A cost fact. Only `exact` facts describe observed/proven cardinality."""

    var rows: Int
    var reason: String
    var exact: Bool


def _cost_rows(value: Float64) -> Int:
    # Saturate before converting to Int, including many-to-many products.
    # The ceiling is a cost-model limit, not a result/capacity bound.
    return Int(min(max(value, 0.0), Float64(Int.MAX // 2)))


def frame_column_width(frame: DataFrame, name: String) raises -> Float64:
    """Payload bytes/value from buffer metadata, with no distinct sampling."""
    var column = frame.column(name)
    if column.dtype() == DataType.STRING or column.dtype()._is_bytes():
        return 16.0 + Float64(column._text_bytes()) / Float64(
            max(1, frame.height())
        )
    if column.dtype().is_nested():
        return -1.0
    return Float64(max(1, column.dtype().bit_width() // 8))


def bounded_distinct(column: Series) raises -> Int:
    """At most 4,096 observations, including frames without table statistics.

    One deterministically jittered row per stratum avoids locking every
    observation to the same offset in periodic data. Singleton/doubleton
    frequencies estimate unseen values, rather than scaling the observed
    distinct fraction by the entire population.
    This is a cost estimate, never a uniqueness or capacity proof.
    """
    var height = len(column)
    var count = min(height, 4096)
    if count == 0:
        return 0
    var sample = column.copy()
    if height > count:
        var rows = List[Int](capacity=count)
        var step = height // count
        var extra = height % count
        for k in range(count):
            var first = k * step + min(k, extra)
            var width = step + (1 if k < extra else 0)
            var hash = UInt64(k + 1) * UInt64(0x9E3779B97F4A7C15)
            hash ^= hash >> 30
            hash *= UInt64(0xBF58476D1CE4E5B9)
            hash ^= hash >> 27
            rows.append(first + Int(hash % UInt64(width)))
        if column.is_chunked():
            # Series.take currently rechunks its entire input. Borrow only
            # the sampled windows so estimate preparation does not copy the
            # full payload of a fragmented intermediate.
            var chunks = column.chunks()
            var parts = List[Series](capacity=count)
            var chunk = 0
            var first = 0
            for row in rows:
                while chunk < len(chunks) and row >= first + len(chunks[chunk]):
                    first += len(chunks[chunk])
                    chunk += 1
                parts.append(chunks[chunk].slice(row - first, 1))
            sample = Series._from_chunks(parts)
        else:
            sample = column.take(rows)
    var codes = encode_rows([sample^], nulls_equal=False)
    var unique = codes.count()
    if height == count:
        return unique
    var frequencies = List[Int](length=unique, fill=0)
    var valid = 0
    for code in codes.ids:
        if code >= 0:
            frequencies[code] += 1
            valid += 1
    var singletons = 0
    var doubletons = 0
    for frequency in frequencies:
        singletons += 1 if frequency == 1 else 0
        doubletons += 1 if frequency == 2 else 0
    if singletons == valid:
        # No repeated non-null value was observed. Extrapolate the valid
        # fraction, without treating this as a uniqueness proof.
        return min(
            height,
            _cost_rows(Float64(height) * Float64(valid) / Float64(count)),
        )
    # Bias-corrected Chao1, adapted as a bounded-sample cost heuristic.
    # Stratification and finite populations do not make this a semantic
    # lower/upper bound. All decisions still validate actual inputs.
    var unseen = (
        Float64(singletons)
        * Float64(singletons - 1)
        / (2.0 * Float64(doubletons + 1))
    )
    return min(height, _cost_rows(Float64(unique) + unseen))


def frame_payload_width(frame: DataFrame) raises -> Float64:
    var width = 0.0
    for name in frame.columns():
        var bytes = frame_column_width(frame, name)
        width += bytes if bytes >= 0 else 16.0
    return max(1.0, width)


def trace_join_cost(
    fact: RowEstimate,
    left_width: Float64,
    right_bytes: Float64,
    prefer_left: Bool,
):
    if getenv("DATAFRAME_TRACE_PATHS"):
        print(
            "dataframe-strategy: join_build; side="
            + ("left" if prefer_left else "right")
            + "; estimated_rows="
            + String(fact.rows)
            + "; estimate_reason="
            + fact.reason
            + "; exact_rows="
            + String(fact.exact)
            + "; left_width="
            + String(left_width)
            + "; right_bytes="
            + String(right_bytes),
            file=RawFileDescriptor(2),
        )


struct IntermediateEstimates(Movable):
    """Execution-independent, request-scoped memo for a fixed logical plan."""

    var facts: List[RowEstimate]
    var columns: Dict[String, Tuple[Int, Float64]]
    var widths: Dict[String, Float64]
    var observed: Dict[Int, DataFrame]
    var used: Bool

    def __init__(out self, count: Int):
        self.facts = List[RowEstimate]()
        for _ in range(count):
            self.facts.append(RowEstimate(-2, "not_estimated", False))
        self.columns = Dict[String, Tuple[Int, Float64]]()
        self.widths = Dict[String, Float64]()
        self.observed = Dict[Int, DataFrame]()
        self.used = False

    def observe(mut self, index: Int, frame: DataFrame) raises:
        """Supply an already-executed input before requesting any estimates."""
        if self.used:
            raise Error("Observed inputs must be supplied before estimation")
        self.observed[index] = frame.copy()
        self.facts[index] = RowEstimate(
            frame.height(), "observed_intermediate", True
        )

    def _set(
        mut self, index: Int, rows: Int, reason: String, exact: Bool = False
    ) -> Int:
        self.facts[index] = RowEstimate(rows, reason, exact)
        return rows

    def column(
        mut self, plan: QueryContext, index: Int, name: String
    ) raises -> Tuple[Int, Float64]:
        """Estimated NDV and bytes/value; -1 means no usable lineage."""
        var key = String(index) + "/" + name
        self.used = True
        if key in self.columns:
            return self.columns[key]
        var result = self._column(plan, index, name)
        self.columns[key] = result
        return result

    def _column(
        mut self, plan: QueryContext, index: Int, name: String
    ) raises -> Tuple[Int, Float64]:
        ref node = plan._nodes[index]
        if index in self.observed:
            ref frame = self.observed[index]
            if (
                name not in frame.columns()
                or frame.column(name).dtype().is_nested()
            ):
                return (-1, -1.0)
            return (
                bounded_distinct(frame.column(name)),
                frame_column_width(frame, name),
            )
        if node.kind == SCAN_FRAME:
            ref frame = plan._frames[node.source_slot()]
            if name not in frame.columns():
                return (-1, -1.0)
            if frame.column(name).dtype().is_nested():
                return (-1, -1.0)
            return (
                bounded_distinct(frame.column(name)),
                frame_column_width(frame, name),
            )
        if node.kind == DROP and name in node.names:
            return (-1, -1.0)
        if node.kind == SELECT or node.kind == WITH_COLUMNS:
            for expression in node.exprs:
                if expression._name == name:
                    if (
                        len(expression._nodes) == 1
                        and expression._nodes[0].op == COL
                    ):
                        return self.column(
                            plan, node.left, expression._nodes[0].text
                        )
                    return (-1, -1.0)
            if node.kind == SELECT:
                return (-1, -1.0)
        if node.kind == AGG and name not in node.names:
            return (-1, -1.0)
        if node.kind == JOIN:
            if (
                node.coalesce
                and name in node.right_keys
                and name not in node.names
            ):
                return (-1, -1.0)
            var left = self.column(plan, node.left, name)
            if left[0] >= 0 or left[1] >= 0:
                return left
            if node.how == JOIN_SEMI or node.how == JOIN_ANTI:
                return (-1, -1.0)
            # Coalesced keys inherit their corresponding source statistics.
            for k in range(len(node.names)):
                if node.coalesce and name == node.names[k]:
                    return self.column(plan, node.right, node.right_keys[k])
            return self.column(plan, node.right, name)
        if node.kind in [
            DROP,
            SELECT,
            WITH_COLUMNS,
            FILTER,
            SORT,
            SLICE,
            AGG,
            UNIQUE,
        ]:
            var result = self.column(plan, node.left, name)
            var rows = (
                self.rows(plan, index) if node.kind
                in [FILTER, SLICE, AGG, UNIQUE] else -1
            )
            if rows >= 0 and result[0] >= 0:
                return (min(rows, result[0]), result[1])
            return result
        return (-1, -1.0)

    def rows(mut self, plan: QueryContext, index: Int) raises -> Int:
        self.used = True
        if self.facts[index].rows != -2:
            return self.facts[index].rows
        ref node = plan._nodes[index]
        if node.kind == SCAN_FRAME:
            return self._set(
                index,
                plan._frames[node.source_slot()].height(),
                "observed_frame",
                True,
            )
        if node.kind == SELECT and _stream_reductions(node.exprs):
            return self._set(index, 1, "scalar_reduction", True)
        if node.kind in [DROP, SORT] or (
            node.kind in [SELECT, WITH_COLUMNS] and _stream_rows(node)
        ):
            var rows = self.rows(plan, node.left)
            return self._set(
                index, rows, "row_preserving", self.facts[node.left].exact
            )
        if node.kind == SLICE:
            var rows = self.rows(plan, node.left)
            if rows >= 0:
                var offset = node.slice_offset()
                if offset >= 0:
                    rows = max(0, rows - offset)
                elif offset > -rows:
                    rows = -offset
                if node.slice_length() >= 0:
                    rows = min(rows, node.slice_length())
            return self._set(index, rows, "slice", self.facts[node.left].exact)
        if node.kind == CONCAT:
            var left = self.rows(plan, node.left)
            var right = self.rows(plan, node.right)
            if left >= 0 and right >= 0:
                return self._set(
                    index,
                    _cost_rows(Float64(left) + Float64(right)),
                    "union_all",
                )
        if node.kind == FILTER and _row_local(node.exprs):
            ref source = plan._nodes[node.left]
            if source.kind == SCAN_FRAME or node.left in self.observed:
                var height = self.rows(plan, node.left)
                var reads = _references(node.exprs[0])
                if reads:
                    var names = List[String]()
                    for name in reads.value():
                        if name not in names:
                            names.append(name)
                    var frame = (
                        self.observed[node.left].copy() if node.left
                        in self.observed else plan._frames[
                            source.source_slot()
                        ].copy()
                    )
                    var sample = frame.select(names)
                    var budget = 64 * 1024
                    if height > budget:
                        sample = frame._sampled(names, 64, 1024)
                    var kept = true_count(
                        sample._selection(node.exprs[0], 8192)
                    )
                    return self._set(
                        index,
                        _cost_rows(
                            Float64(kept)
                            / Float64(max(1, min(height, budget)))
                            * Float64(height)
                        ),
                        "sampled_filter",
                        height <= budget,
                    )
            return self._set(
                index, self.rows(plan, node.left), "unknown_filter_selectivity"
            )
        if node.kind == AGG or node.kind == UNIQUE:
            var rows = self.rows(plan, node.left)
            var groups = 1.0
            if len(node.names) == 0:
                return self._set(index, -1, "unknown_group_keys")
            for name in node.names:
                var count = self.column(plan, node.left, name)[0]
                if count < 0:
                    return self._set(index, rows, "unknown_group_domain")
                groups = min(
                    Float64(max(rows, 0)), groups * Float64(max(count, 1))
                )
            return self._set(
                index, _cost_rows(groups) if rows >= 0 else -1, "group_domain"
            )
        if node.kind == JOIN:
            var left = self.rows(plan, node.left)
            var right = self.rows(plan, node.right)
            if left < 0 or right < 0:
                return self._set(index, -1, "unknown_join_input")
            if node.how == JOIN_CROSS:
                return self._set(
                    index,
                    _cost_rows(Float64(left) * Float64(right)),
                    "cross_product",
                )
            var domain = 1
            var known = False
            var empty_domain = False
            for k in range(len(node.names)):
                var a = self.column(plan, node.left, node.names[k])[0]
                var b = self.column(plan, node.right, node.right_keys[k])[0]
                empty_domain = empty_domain or a == 0 or b == 0
                if a > 0 and b > 0:
                    domain = max(domain, max(a, b))
                    known = True
            var matched = (
                Float64(left)
                * Float64(right)
                / Float64(domain) if known else Float64(max(left, right))
            )
            if left == 0 or right == 0 or empty_domain:
                matched = 0.0
            if node.how == JOIN_SEMI:
                matched = min(Float64(left), matched)
            elif node.how == JOIN_ANTI:
                matched = max(0.0, Float64(left) - min(Float64(left), matched))
            elif node.how == JOIN_LEFT:
                matched = max(Float64(left), matched)
            elif node.how == JOIN_RIGHT:
                matched = max(Float64(right), matched)
            elif node.how == JOIN_FULL:
                matched = max(Float64(max(left, right)), matched)
            elif node.how != JOIN_INNER:
                return self._set(index, -1, "unsupported_join")
            return self._set(
                index,
                _cost_rows(matched),
                "join_key_domain" if known else "unknown_join_domain",
            )
        return self._set(index, -1, "unknown_operator")

    def payload_width(
        mut self, plan: QueryContext, index: Int, names: List[String]
    ) raises -> Float64:
        var width = 0.0
        for name in names:
            var bytes = self.column_width(plan, index, name)
            width += bytes if bytes >= 0 else 16.0
        return max(width, 1.0)

    def column_width(
        mut self, plan: QueryContext, index: Int, name: String
    ) raises -> Float64:
        var key = String(index) + "/" + name
        self.used = True
        if key in self.widths:
            return self.widths[key]
        var width = self._column_width(plan, index, name)
        self.widths[key] = width
        return width

    def _column_width(
        mut self, plan: QueryContext, index: Int, name: String
    ) raises -> Float64:
        ref node = plan._nodes[index]
        if index in self.observed:
            ref frame = self.observed[index]
            return (
                frame_column_width(frame, name) if name
                in frame.columns() else -1.0
            )
        if node.kind == SCAN_FRAME:
            ref frame = plan._frames[node.source_slot()]
            return (
                frame_column_width(frame, name) if name
                in frame.columns() else -1.0
            )
        if node.kind == DROP and name in node.names:
            return -1.0
        if node.kind in [SELECT, WITH_COLUMNS]:
            for expression in node.exprs:
                if expression._name == name:
                    if (
                        len(expression._nodes) == 1
                        and expression._nodes[0].op == COL
                    ):
                        return self.column_width(
                            plan, node.left, expression._nodes[0].text
                        )
                    return -1.0
            if node.kind == SELECT:
                return -1.0
        if node.kind == AGG and name not in node.names:
            return -1.0
        if node.kind == JOIN:
            if (
                node.coalesce
                and name in node.right_keys
                and name not in node.names
            ):
                return -1.0
            var left = self.column_width(plan, node.left, name)
            if left >= 0:
                return left
            if node.how in [JOIN_SEMI, JOIN_ANTI]:
                return -1.0
            return self.column_width(plan, node.right, name)
        if node.kind in [
            DROP,
            SELECT,
            WITH_COLUMNS,
            AGG,
            FILTER,
            SORT,
            SLICE,
            UNIQUE,
        ]:
            return self.column_width(plan, node.left, name)
        return -1.0
