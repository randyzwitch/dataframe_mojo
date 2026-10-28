"""Central streaming-reduction strategy selection and cost preferences."""
from dataframe.aggregation.group_by import (
    group_by_workers,
    range_states,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.expr import (
    Expr,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.kernels.hashing import (
    encode_rows,
)
from dataframe.join_type import (
    JOIN_INNER,
    JOIN_RIGHT,
)
from dataframe.planning.cost.intermediate import (
    IntermediateEstimates,
    frame_column_width,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.cost.model import (
    AggregationDecision,
    BATCH_STREAM,
    EAGER_FRAME,
    MORSEL_PIPELINE,
    PipelineEligibility,
    TABLE_PIPELINE,
)
from dataframe.planning.cost.sampling import (
    _batch_shows_many_groups,
    _key_group_bound,
    _keys_repeat,
    _stream_split_groups,
    _unique_key_column,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from dataframe.planning.model import (
    AGG,
    JOIN,
    PlanNode,
    SCAN_FRAME,
)
from dataframe.planning.cost.eligibility import (
    _counts_distinct,
    _filters,
    _has_quantile,
    _key_read,
    _order_sensitive,
    _row_merge_reductions,
    _row_steps_only,
)
from dataframe.series import (
    Series,
)
from dataframe.trace import (
    trace_path,
)


def plan_sampled_groups(
    self: QueryContext, keys: List[String], frame: Int
) raises -> Int:
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


def plan_batch_states_win(
    self: QueryContext, keys: List[String], frame: Int, batch_size: Int
) raises -> Bool:
    """Whether an unfiltered grouped reduction over in-memory `frame`
    runs faster as batches, each reduced into a state of its own, than
    through the eager group-by (#608). Each route's own tests decide,
    not a count of groups:
    - The eager group-by keeps private per-worker states when its keys
      have few values (`range_states`); batch states are the same work
      plus batching and a merge per batch.
    - Otherwise it hash-partitions every row, which costs more than
      batch states while a batch holds few of its rows' groups. A
      batch that shows many (`_batch_shows_many_groups`, the stream's
      own test) makes the stream collect its batches into that same
      partitioned group-by, which then also pays the copy.
    - The stream reduces one batch per worker, so with fewer batches
      than the eager group-by has workers (or one worker) it leaves
      threads idle.
    Measured on generated int, string and two-key frames of 100K to
    10M rows, 50 to 5,000 groups, uniform and skewed, one and six
    aggregates, 1 and 8 threads: batch states won only where all three
    hold (int and string keys of about 190 to 1,000 groups at 1M and
    10M rows, 8 threads: 0.4-0.85 of the eager time); everywhere else
    the eager route was as fast or faster (up to 2.6 times)."""
    ref source = self._frames[frame]
    var height = source.height()
    var columns = List[Series](capacity=len(keys))
    for name in keys:
        if name not in source.columns():
            # Made by an earlier step: ClickBench q18's (UserID,
            # minute, SearchPhrase) took 354 ms eagerly, 1,645 batched.
            return False
        columns.append(source.column(name).copy())
    var workers = group_by_workers(height)
    if workers <= 1 or (height + batch_size - 1) // batch_size < workers:
        return False
    if range_states(columns) or _batch_shows_many_groups(
        source.slice(0, min(height, batch_size)),
        keys,
        _stream_split_groups(),
    ):
        return False
    trace_path("stream.batch_states")
    return True


def plan_row_number_shape(
    self: QueryContext,
    source: Int,
    keys: List[String],
    carried: List[String],
    holder: Int,
    joins: List[DataFrame],
    operations: List[PlanNode],
    input_index: Int = -1,
) raises -> Tuple[Float64, Float64]:
    """Estimate input rows and first-pass groups using observed join builds.

    A build's actual result replaces only cost facts; no source is executed
    to obtain statistics. Duplicate build keys can multiply input rows.
    Reached holder rows include those duplicates, rather than equating the
    number of reached keys with the number of reached build rows.
    """
    var input = input_index
    if (
        input < 0
        and len(self._nodes) > 0
        and self._nodes[len(self._nodes) - 1].kind == AGG
    ):
        input = self._nodes[len(self._nodes) - 1].left
    if input < 0:
        return (-1.0, -1.0)
    var estimates = IntermediateEstimates(len(self._nodes))
    for op in operations:
        if op.kind == JOIN:
            estimates.observe(op.right, joins[op.join_build_slot()])
    var estimated_rows = estimates.rows(self, input)
    if estimated_rows < 0:
        return (-1.0, -1.0)
    var rows = Float64(estimated_rows)
    var reached = Float64(joins[holder].height())
    for op in operations:
        if op.kind != JOIN or op.join_build_slot() != holder:
            continue
        var probe = -1
        var build = -1
        for k in range(len(op.names)):
            probe = max(probe, estimates.column(self, op.left, op.names[k])[0])
            build = max(
                build, estimates.column(self, op.right, op.right_keys[k])[0]
            )
        if probe == 0 or build == 0:
            reached = 0.0
        elif probe > 0 and build > 0:
            reached *= min(1.0, Float64(probe) / Float64(build))
    var groups = reached
    for key in keys:
        if key in carried:
            continue
        var count = estimates.column(self, input, key)[0]
        if count < 0:
            return (-1.0, -1.0)
        groups = min(rows, groups * Float64(max(1, count)))
    return (rows, min(rows, groups))


def _key_payload_bytes(
    source: DataFrame, joins: List[DataFrame], names: List[String]
) raises -> Float64:
    var bytes = 0.0
    for name in names:
        var width = -1.0
        if name in source.columns():
            width = frame_column_width(source, name)
        else:
            for build in joins:
                if name in build.columns():
                    width = frame_column_width(build, name)
                    break
        bytes += width if width >= 0 else 16.0
    return bytes


def select_aggregation_strategy(
    self: QueryContext,
    cursor: Int,
    batch_size: Int,
    keys: List[String],
    expressions: List[Expr],
    operations: List[PlanNode],
    joins: List[DataFrame],
    distinct: Bool,
    ordered: Bool,
    top: Int,
    grouping: Bool,
    late_names: List[String],
    late_builds: List[Int],
    eligibility: PipelineEligibility,
    inline_table_eligible: Bool,
    input_index: Int = -1,
) raises -> AggregationDecision:
    """Choose one route, retaining the original preference and sampling order.

    Semantic pipeline/inline-state eligibility is supplied before cost
    preferences. Key coding and row-number grouping are returned as a plan;
    this function does not rewrite operations or execute a source.
    """
    var choice = AggregationDecision(
        MORSEL_PIPELINE if eligibility.allowed else BATCH_STREAM,
        eligibility.reason,
    )
    if inline_table_eligible:
        choice.choose(TABLE_PIPELINE, "inline_aggregate_states")
    if not choice.uses_aggregate_table() and (
        grouping or _has_quantile(expressions)
    ):
        choice.choose(EAGER_FRAME, "inline_table_required")
        return choice^
    if (
        choice.uses_pipeline()
        and len(keys) > 0
        and not choice.uses_aggregate_table()
    ):
        var frame = self._nodes[cursor].source_slot()
        ref source_frame = self._frames[frame]
        var fixed = True
        # String keys group exactly in the insert tables (a long
        # string's bytes compared, `_PartIndex`); collected, they stay
        # on the eager route (below).
        var string_keys = False
        var used = keys.copy()
        for expression in expressions:
            var reads = _references(expression)
            if not reads:
                fixed = False
                break
            for name in reads.value():
                if name not in used:
                    used.append(name)
        for name in used:
            if not fixed:
                break
            if name in late_names:
                # Grouped by codes if the plan inserts (below).
                continue
            var dtype = DataType.INT64
            if name in source_frame.columns():
                dtype = source_frame.column(name).dtype()
            else:
                # A join's build column, or made by a step (then its
                # dtype is not known here).
                var found = False
                for build in joins:
                    if name in build.columns():
                        dtype = build.column(name).dtype()
                        found = True
                        break
                if not found:
                    continue
            if (
                dtype.physical() == DataType.STRING
                and name in keys
                and not _key_read(name, expressions)
            ):
                string_keys = True
            elif (
                dtype.is_nested()
                or dtype.is_decimal()
                or dtype.physical() == DataType.STRING
            ):
                fixed = False
        if not fixed and len(operations) > 0:
            choice.choose(BATCH_STREAM, "variable_width_payload")
        elif fixed and not _order_sensitive(expressions):
            # Sampled only when the mode is still open: the sample
            # takes and encodes 4,096 key rows (a millisecond on
            # strings, an eighth of ClickBench q38).
            var sample = min(source_frame.height(), 4096)
            # A key from a join's build side is not in the source to
            # sample (and the source's key columns are not taken).
            var sampled = -1 if len(late_names) > 0 else (
                -1 if string_keys else plan_sampled_groups(self, keys, frame)
            )
            if string_keys:
                # String keys insert behind a filter, which the eager
                # route pays for by gathering the kept rows' strings
                # (ClickBench q37: 22.7 -> 15.9 ms). Without one, the
                # eager route reads the keys in place, and inserting
                # many groups was slower (q16 119 -> 176 ms, q33 189
                # -> 235).
                # Whether the keys repeat is sampled once the input
                # is narrowed (below); a plan with joins keeps the
                # eager route, which would redo their builds if the
                # sample sent it back (TPC-DS q45: 5.9 -> 16.8 ms).
                choice.choose_insertion(
                    _filters(operations) and len(joins) == 0,
                    "filtered_string_keys",
                )
                choice.pending_string_sample = choice.inserts_groups()
            elif sampled >= 0:
                choice.sample_rows = sample
                choice.sample_groups = sampled
                choice.choose_insertion(
                    4 * sampled <= sample, "sampled_key_density"
                )
            else:
                # The groups' bound from where each key is stored,
                # held to the sampled rule's density: at most a
                # quarter of the sample's rows.
                var bound = _key_group_bound(
                    keys, source_frame, joins, sample // 4
                )
                choice.group_bound = bound
                choice.choose_insertion(bound >= 0, "bounded_group_domain")
        if string_keys and not choice.inserts_groups() and len(operations) > 0:
            # Collected strings gather slower than the eager route's
            # per-chunk filter (ClickBench q12: 85 ms against 48).
            choice.choose(BATCH_STREAM, "string_key_collection")
    if distinct and choice.inserts_groups():
        # Distinct counts are collected for the eager group-by.
        choice.choose(MORSEL_PIPELINE, "distinct_requires_collection")

    var late_table_codes_pay = False
    if choice.uses_aggregate_table() and len(late_names) > 0:
        ref source_frame = self._frames[self._nodes[cursor].source_slot()]
        var sample = min(source_frame.height(), 4096)
        late_table_codes_pay = (
            _key_group_bound(keys, source_frame, joins, sample // 4) >= 0
        )
    choice.code_late_keys = len(late_names) > 0 and (
        choice.uses_pipeline()
        and (choice.inserts_groups() or late_table_codes_pay)
    )
    if len(late_names) > 0 and not choice.code_late_keys:
        # Coded keys pay only in the insert tables: collecting them
        # for the finish's group-by ran slower than the eager route
        # their strings take (TPC-DS q45 5.9 -> 13.4 ms, q24 40 -> 81),
        # so those plans take the strings' route, as before; the
        # aggregate table groups the strings themselves.
        if len(operations) > 0 and not choice.uses_aggregate_table():
            choice.choose(BATCH_STREAM, "uncoded_build_keys")
        # Several of these keys from one build side, under the aggregate
        # table: the sink groups by that build's row number alone
        # (DuckDB carries a build row's payload by its address through
        # the join and reads it once, at the output), and the finished
        # groups, far fewer than the joined rows, read their key values
        # from the build and merge where two build rows hold the same
        # values.
        if (
            choice.uses_aggregate_table()
            and choice.uses_pipeline()
            and not distinct
            and top < 0
        ):
            var best = -1
            var best_count = 1
            for h in late_builds:
                var count = 0
                for other in late_builds:
                    if other == h:
                        count += 1
                if count > best_count:
                    best = h
                    best_count = count
            var carried = List[String]()
            for k in range(len(late_names)):
                if best >= 0 and late_builds[k] == best:
                    carried.append(late_names[k])
            if best >= 0:
                var shape = plan_row_number_shape(
                    self,
                    self._nodes[cursor].source_slot(),
                    keys,
                    carried,
                    best,
                    joins,
                    operations,
                    input_index,
                )
                var merge = _row_merge_reductions(expressions)
                var rows = shape[0]
                var groups = shape[1]
                ref source_frame = self._frames[
                    self._nodes[cursor].source_slot()
                ]
                var carried_bytes = _key_payload_bytes(
                    joins[best], joins, carried
                )
                var key_bytes = _key_payload_bytes(source_frame, joins, keys)
                var build = Float64(joins[best].height())
                # A row id still has to be carried through the first pass.
                var saved = rows * max(0.0, carried_bytes - 8.0)
                # Bound the preparation read separately from output work.
                var setup = Float64(min(source_frame.height(), 4096)) * 8.0
                var proof = 2.0 * build * carried_bytes
                var finish_unique = groups * (carried_bytes + 8.0)
                var unique = False
                # Avoid a full uniqueness proof when even the optimistic
                # no-regroup route cannot repay it.
                if rows >= 0 and saved > setup + proof + finish_unique:
                    unique = _unique_key_column(joins[best], carried)
                    setup += proof
                choice.key_input_rows = rows
                choice.key_finish_groups = groups
                choice.key_saved_work = saved
                if rows >= 0 and unique:
                    choice.key_paid_work = setup + finish_unique
                    choice.key_cost_reason = "unique_key_gather"
                    if saved > choice.key_paid_work:
                        choice.row_holder = best
                        choice.row_unique = True
                        choice.row_names = carried^
                elif rows >= 0 and merge:
                    # Read/hash/partition/compare output keys; copy and merge
                    # states; carry row ids and assemble the final output.
                    var state_bytes = Float64(max(1, len(expressions)) * 8)
                    var finish = groups * (
                        4.0 * key_bytes + 2.0 * state_bytes + 16.0
                    )
                    choice.key_paid_work = setup + finish
                    choice.key_cost_reason = "final_regroup_and_state_copy"
                    if saved > choice.key_paid_work:
                        choice.row_holder = best
                        choice.row_merge = merge.take()
                        choice.row_names = carried^
                else:
                    choice.key_cost_reason = (
                        "unknown_shape_or_unmergeable_states"
                    )

    # Right joins require their unmatched build rows to be emitted by a
    # morsel pipeline. A row-only in-memory fallback may prefer eager work.
    if not choice.uses_pipeline():
        for operation in operations:
            if operation.kind == JOIN and operation.how == JOIN_RIGHT:
                choice.choose(EAGER_FRAME, "right_join_requires_pipeline")
                return choice^
    if (
        not choice.uses_pipeline()
        and len(expressions)
        and self._nodes[cursor].kind == SCAN_FRAME
        and _row_steps_only(operations)
    ):
        if _counts_distinct(expressions):
            choice.choose(EAGER_FRAME, "in_memory_distinct_reduction")
            return choice^
        elif len(keys) > 0:
            if _filters(operations):
                choice.choose(EAGER_FRAME, "filtered_frame_grouping")
                return choice^
            if not plan_batch_states_win(
                self, keys, self._nodes[cursor].source_slot(), batch_size
            ):
                choice.choose(EAGER_FRAME, "eager_grouping_preferred")
                return choice^
    if (
        top >= 0
        and len(operations) == 0
        and self._nodes[cursor].kind == SCAN_FRAME
    ):
        choice.choose(EAGER_FRAME, "in_memory_topk_without_steps")
    return choice^


def refine_string_sample(
    var choice: AggregationDecision,
    input: DataFrame,
    keys: List[String],
    operations: List[PlanNode],
) raises -> AggregationDecision:
    """Judge deferred string-key density after the original input narrowing."""
    if choice.pending_string_sample:
        if not _keys_repeat(input, keys, operations):
            choice.choose(EAGER_FRAME, "filtered_strings_mostly_distinct")
        choice.pending_string_sample = False
    return choice^
