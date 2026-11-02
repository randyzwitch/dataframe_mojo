"""Eager join strategy dispatch and output assembly."""
from dataframe.aggregation.keys import (
    _expand_struct_keys,
)
from dataframe.categorical import (
    unify,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.expr import (
    MAX,
)
from dataframe.kernels.binary import (
    choose,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.kernels.gather import (
    SORTED_GATHER_MIN_CHUNKS,
    can_filter_aligned_chunks,
    take_parallel,
    take_sorted_chunked,
)
from dataframe.kernels.join_fuse import (
    FusedJoin,
)
from dataframe.kernels.join_hash import (
    PreparedHashIndex,
    direct_hash_join_rows,
    direct_hash_semi_anti_rows,
    prefer_left_build,
    prepare_hash_index,
    prepared_hash_join_gather,
    prepared_hash_join_rows,
    prepared_hash_semi_anti_rows,
)
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_SEMI,
    join_name,
)
from dataframe.joins.keys import (
    _joint_key_ids,
)
from dataframe.joins.range import (
    _bounded_int64_join_rows,
    _dense_right_int64_rows,
    _join_range_rows,
    _range_int64_membership_chunks,
    _range_int64_membership_rows,
)
from dataframe.joins.rows import (
    _group_index,
    _group_rows,
    _parallel_csr_fits,
    _parallel_full_join_rows,
    _parallel_group_rows,
    _parallel_join_rows,
    _parallel_left_major,
)
from dataframe.parallel import (
    worker_count,
)
from dataframe.kernels.partition import (
    low_cardinality,
)
from dataframe.series import (
    Series,
)
from dataframe.trace import (
    trace_path,
)
from std.collections import (
    Dict,
    Optional,
)
from std.memory import (
    Pointer,
)


def join_frames(
    self: DataFrame,
    right: DataFrame,
    *,
    left_on: List[String],
    right_on: List[String],
    how: Int = JOIN_INNER,
    suffix: String = "_right",
    coalesce: Bool = True,
    prepared: Optional[PreparedHashIndex] = None,
    range_filtered: Bool = False,
    keep_order: Bool = True,
    workers: Int = 0,
    marks: Int = 0,
) raises -> DataFrame:
    """Execute a validated eager join with the existing strategy order."""
    # A categorical key joins on codes, which compare only on a shared
    # dictionary (#106): both key columns move onto the union of theirs
    # (a String key is encoded into it), leaving the left codes as they
    # are. Unchanged pairs join as before.
    var unified = False
    var left_frame = self.copy()
    var right_frame = right.copy()
    for i in range(len(left_on)):
        var l = self._index(left_on[i])
        var r = right._index(right_on[i])
        var ltype = self._columns[l].dtype()
        var rtype = right._columns[r].dtype()
        if (ltype.is_categorical() or rtype.is_categorical()) and not (
            ltype.is_categorical() and rtype.is_categorical() and ltype == rtype
        ):
            var lkey = left_frame._columns[l].copy()
            var rkey = right_frame._columns[r].copy()
            if ltype.is_categorical() or ltype.physical() == DataType.STRING:
                unify(lkey, rkey)
                left_frame._columns[l] = lkey^
                right_frame._columns[r] = rkey^
                unified = True
    if unified:
        return left_frame._join_impl(
            right_frame,
            left_on=left_on,
            right_on=right_on,
            how=how,
            suffix=suffix,
            coalesce=coalesce,
            prepared=None,
            range_filtered=range_filtered,
            keep_order=keep_order,
        )
    var left_keys = List[Int]()
    var right_keys = List[Int]()
    var seen = Dict[String, Bool]()
    for i in range(len(left_on)):
        if left_on[i] in seen:
            raise Error("Duplicate join key: " + left_on[i])
        seen[left_on[i]] = True
        left_keys.append(self._index(left_on[i]))
        right_keys.append(right._index(right_on[i]))
        var ltype = self._columns[left_keys[i]].dtype()
        var rtype = right._columns[right_keys[i]].dtype()
        if ltype != rtype:
            raise Error(
                "Join key dtypes differ: "
                + left_on[i]
                + " is "
                + ltype.name()
                + " but "
                + right_on[i]
                + " is "
                + rtype.name()
            )
    var has_struct_key = False
    for k in left_keys:
        if self._columns[k].dtype().is_struct():
            has_struct_key = True
    if has_struct_key:
        # Join on the struct's fields and validity as ordinary key
        # columns, then drop those helpers. The right struct column is
        # dropped up front so it does not come through as an extra
        # output column.
        if (
            how != JOIN_INNER
            and how != JOIN_LEFT
            and how != JOIN_SEMI
            and how != JOIN_ANTI
        ):
            raise Error(
                "struct join keys support inner, left, semi and anti"
                " joins; found " + join_name(how)
            )
        var left_side = self.copy()
        var right_side = right.copy()
        var new_left_on = List[String]()
        var new_right_on = List[String]()
        var helpers = List[String]()
        for i in range(len(left_on)):
            if not self._columns[left_keys[i]].dtype().is_struct():
                new_left_on.append(left_on[i])
                new_right_on.append(right_on[i])
                continue
            var left_parts = _expand_struct_keys(
                [self._columns[left_keys[i]].copy()]
            )
            var right_parts = _expand_struct_keys(
                [right._columns[right_keys[i]].renamed(left_on[i])]
            )
            right_side = right_side.drop([right_on[i]])
            for p in range(len(left_parts)):
                left_side = left_side.with_column(left_parts[p].copy())
                right_side = right_side.with_column(right_parts[p].copy())
                new_left_on.append(left_parts[p].name())
                new_right_on.append(left_parts[p].name())
                helpers.append(left_parts[p].name())
        return left_side._join_impl(
            right_side,
            left_on=new_left_on,
            right_on=new_right_on,
            how=how,
            suffix=suffix,
            coalesce=coalesce,
            keep_order=keep_order,
        ).drop(helpers)
    var keep_right_keys = not coalesce and how != JOIN_SEMI and how != JOIN_ANTI
    var right_output = List[Int]()
    var right_names = List[String]()
    if how != JOIN_SEMI and how != JOIN_ANTI:
        var names = Dict[String, Bool]()
        for column in self._columns:
            names[column.name()] = True
        var left_names = names.copy()
        for i in range(right.width()):
            if not keep_right_keys and i in right_keys:
                continue
            var name = right._columns[i].name()
            if name in left_names:
                name += suffix
            if name in names:
                raise Error("Join output name collision: " + name)
            names[name] = True
            right_output.append(i)
            right_names.append(name)
    if (
        not range_filtered
        and not prepared
        and (how == JOIN_INNER or how == JOIN_LEFT)
        and prefer_left_build(self.height(), right.height())
    ):
        for k in range(len(left_keys)):
            var selected = _join_range_rows(
                self._columns[left_keys[k]], right._columns[right_keys[k]]
            )
            if selected:
                var filtered = right._filter_rows(selected.take())
                return self._join_impl(
                    filtered,
                    left_on=left_on,
                    right_on=right_on,
                    how=how,
                    suffix=suffix,
                    coalesce=coalesce,
                    range_filtered=True,
                    keep_order=keep_order,
                )
    if prepared:
        var sources = List[Series](capacity=len(left_keys))
        for k in left_keys:
            sources.append(self._columns[k].copy())
        # `workers` > 0 fixes the probe and gather lanes: a pipeline
        # worker joins its morsel alone, and gathers in one piece, so
        # no later merge of per-lane string pieces (PDS-H q2: 13.9 ms
        # against 11.6 with lanes per morsel).
        if how == JOIN_SEMI or how == JOIN_ANTI:
            return self._filter_rows(
                prepared_hash_semi_anti_rows(
                    sources, prepared.value(), how == JOIN_SEMI, workers
                )
            )
        # A full join probes as a left join (its unmatched probe rows
        # with no build row) and flags the build rows it matched, as a
        # right join does: DuckDB's FULL OUTER probe, then its scan of
        # the build rows no probe found.
        var outer = how == JOIN_LEFT or how == JOIN_FULL
        var pairs = prepared_hash_join_rows(
            sources,
            prepared.value(),
            outer,
            omit_identity=True,
            workers=workers,
        )
        var lanes = workers if workers > 0 else worker_count(len(pairs[1]))
        var columns = self._columns.copy()
        var identity = pairs[2] or len(pairs[0]) == self.height()
        if identity and not pairs[2]:
            for i in range(len(pairs[0])):
                if pairs[0][i] != i:
                    identity = False
                    break
        # The row lists are swapped out of the tuple: copying them
        # doubled the index traffic of a 10M-row join (#335).
        var output_height = len(pairs[1])
        if marks != 0 and (how == JOIN_RIGHT or how == JOIN_FULL):
            var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=marks)
            var matched = pairs[1].unsafe_ptr()
            for r in range(output_height):
                var row = matched[unsafe_offset=r]
                if row >= 0:
                    flags[unsafe_offset=row] = 1
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        swap(left_rows, pairs[0])
        swap(right_rows, pairs[1])
        if not identity:
            columns = take_parallel(columns, left_rows^, lanes, or_null=False)
        var right_sources = List[Series]()
        for c in right_output:
            right_sources.append(right._columns[c].copy())
        var gathered = take_parallel(
            right_sources,
            right_rows^,
            lanes,
            or_null=outer,
        )
        for k in range(len(right_output)):
            columns.append(gathered[k].renamed(right_names[k]))
        return DataFrame(columns^, height=output_height)
    if (how == JOIN_INNER or how == JOIN_LEFT) and len(left_keys) == 1:
        var dense = _dense_right_int64_rows(
            self._columns[left_keys[0]],
            right._columns[right_keys[0]],
            how == JOIN_LEFT,
        )
        if dense[0]:
            var left_rows = dense[1].copy()
            var right_rows = dense[2].copy()
            var workers = worker_count(len(left_rows))
            var columns = self._columns.copy()
            var left_identity = len(left_rows) == self.height()
            if left_identity:
                for i in range(len(left_rows)):
                    if left_rows[i] != i:
                        left_identity = False
                        break
            var output_height = len(left_rows)
            if not left_identity:
                var ordered_chunks = how == JOIN_INNER
                for column in columns:
                    if (
                        not column.is_chunked()
                        or column.n_chunks() < SORTED_GATHER_MIN_CHUNKS
                    ):
                        ordered_chunks = False
                if ordered_chunks:
                    columns = take_sorted_chunked(
                        columns,
                        left_rows^,
                        workers,
                        allow_repeats=True,
                    )
                else:
                    columns = take_parallel(
                        columns, left_rows^, workers, or_null=False
                    )
            var right_sources = List[Series]()
            for c in right_output:
                right_sources.append(right._columns[c].copy())
            var gathered = take_parallel(
                right_sources,
                right_rows^,
                workers,
                or_null=how == JOIN_LEFT,
            )
            for k in range(len(right_output)):
                columns.append(gathered[k].renamed(right_names[k]))
            return DataFrame(columns^, height=output_height)
    # A small left side against a large right one: hash the left keys
    # and scan the right rows, marking the left rows they match, as
    # DuckDB builds on the smaller side. PDS-H q4 hashed 3.8M late
    # lineitem rows to test about 57K orders. Checked before the
    # bounded-domain path, which scans every right row on one thread
    # to build its presence table (TPC-DS q82: 34 items against 2.9M
    # store sales, 28 ms).
    if (
        (how == JOIN_SEMI or how == JOIN_ANTI)
        and right.height() >= 524288
        and self.height() > 0
        and 4 * self.height() <= right.height()
        and self.height() <= Int(Int32.MAX)
    ):
        trace_path("join.membership_left_build")
        var left_sources = List[Series](capacity=len(left_keys))
        var right_sources = List[Series](capacity=len(right_keys))
        for k in range(len(left_keys)):
            left_sources.append(self._columns[left_keys[k]].copy())
            right_sources.append(right._columns[right_keys[k]].copy())
        var pairs = direct_hash_join_rows(right_sources, left_sources, False)
        var marked = List[Bool](length=self.height(), fill=False)
        for row in pairs[1]:
            marked[row] = True
        var keep = how == JOIN_SEMI
        var rows = List[Int]()
        for i in range(self.height()):
            if marked[i] == keep:
                rows.append(i)
        return self._filter_rows(rows^)
    if (how == JOIN_SEMI or how == JOIN_ANTI) and len(left_keys) == 1:
        var aligned_chunks = can_filter_aligned_chunks(self._columns)
        if aligned_chunks:
            var chunk_membership = _range_int64_membership_chunks(
                self,
                left_keys[0],
                right._columns[right_keys[0]],
                how == JOIN_SEMI,
            )
            if chunk_membership[0]:
                var selected = chunk_membership[1].copy()
                return DataFrame(selected^, height=len(selected[0]))
        else:
            var membership = _range_int64_membership_rows(
                self._columns[left_keys[0]],
                right._columns[right_keys[0]],
                how == JOIN_SEMI,
            )
            if membership[0]:
                return self._filter_rows(membership[1].copy())
    # Semi and anti joins on keys the bounded path declined: probe a
    # right-row hash index for membership only. The dictionary path
    # below would encode both inputs and group every right row first.
    if (
        (how == JOIN_SEMI or how == JOIN_ANTI)
        and worker_count(self.height()) > 1
        and right.height() <= Int(Int32.MAX)
    ):
        var left_sources = List[Series](capacity=len(left_keys))
        var right_sources = List[Series](capacity=len(right_keys))
        for k in range(len(left_keys)):
            left_sources.append(self._columns[left_keys[k]].copy())
            right_sources.append(right._columns[right_keys[k]].copy())
        if not low_cardinality(right_sources):
            return self._filter_rows(
                direct_hash_semi_anti_rows(
                    left_sources, right_sources, how == JOIN_SEMI
                )
            )
    # Dense ids over both inputs materialize and re-encode every key.
    # For high-cardinality right keys, a row index probes the original
    # columns directly and preserves exact equality across collisions.
    # Every inner and left join takes the direct route. It used to need
    # a left of more than one worker's rows (65,536), which sent a left
    # of 65K to 131K rows to the dictionary route below: on a 2M-row
    # right, 125K left rows took 114 ms there against 44 ms here.
    var build_left = _left_builds(self, right)
    if how == JOIN_INNER or how == JOIN_LEFT:
        var left_sources = List[Series](capacity=len(left_keys))
        var right_sources = List[Series](capacity=len(right_keys))
        for k in range(len(left_keys)):
            left_sources.append(self._columns[left_keys[k]].copy())
            right_sources.append(right._columns[right_keys[k]].copy())
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        var direct = False
        var direct_identity = False
        # Probe and gather one block of rows at a time (`fused_gather`),
        # unless the left columns are many chunks and an inner join's
        # ascending rows gather from them in place, without a rechunk.
        var fused = FusedJoin()
        if not build_left:
            var in_place = how == JOIN_INNER
            for column in self._columns:
                if (
                    not column.is_chunked()
                    or column.n_chunks() < SORTED_GATHER_MIN_CHUNKS
                ):
                    in_place = False
            if not in_place:
                fused.enabled = True
                fused.left_sources = self._columns.copy()
                for c in right_output:
                    fused.right_sources.append(right._columns[c].copy())
        # Every route returns the pairs in left row order except a
        # left build whose caller ignores order, which returns them in
        # the order the right side probed (a left join's unmatched
        # left rows after them).
        var left_ascending = not build_left or keep_order
        if build_left:
            var pairs = _smaller_build_join_rows(
                left_sources,
                right_sources,
                how == JOIN_LEFT,
                keep_order,
            )
            swap(left_rows, pairs[0])
            swap(right_rows, pairs[1])
            direct = True
        if not direct and len(left_keys) == 1:
            direct = _bounded_int64_join_rows(
                left_sources[0],
                right_sources[0],
                how == JOIN_LEFT,
                left_rows,
                right_rows,
                fused,
            )
        if (
            not direct
            and right.height() <= Int(Int32.MAX)
            and not low_cardinality(right_sources)
        ):
            if fused.enabled:
                prepared_hash_join_gather(
                    left_sources,
                    prepare_hash_index(
                        right_sources,
                        allow_progression=False,
                        probe_rows=len(left_sources[0]),
                    ),
                    how == JOIN_LEFT,
                    True,
                    fused,
                )
                direct = True
            else:
                var pairs = direct_hash_join_rows(
                    left_sources,
                    right_sources,
                    how == JOIN_LEFT,
                    omit_identity=True,
                )
                direct = True
                direct_identity = pairs[2]
                swap(left_rows, pairs[0])
                swap(right_rows, pairs[1])
        if fused.done:
            var columns = (
                self._columns.copy() if fused.identity else fused.left.copy()
            )
            for k in range(len(right_output)):
                columns.append(fused.right[k].renamed(right_names[k]))
            return DataFrame(columns^, height=fused.height)
        if direct:
            var workers = worker_count(len(right_rows))
            var columns = self._columns.copy()
            var left_identity = (
                direct_identity or len(left_rows) == self.height()
            )
            if left_identity and not direct_identity:
                for i in range(len(left_rows)):
                    if left_rows[i] != i:
                        left_identity = False
                        break
            if not left_identity:
                # The chunk-by-chunk gather reads left rows in ascending
                # order only.
                var ordered_chunks = how == JOIN_INNER and left_ascending
                for column in columns:
                    if (
                        not column.is_chunked()
                        or column.n_chunks() < SORTED_GATHER_MIN_CHUNKS
                    ):
                        ordered_chunks = False
                if ordered_chunks:
                    columns = take_sorted_chunked(
                        columns,
                        left_rows^,
                        workers,
                        allow_repeats=True,
                    )
                else:
                    columns = take_parallel(
                        columns, left_rows^, workers, or_null=False
                    )
            var right_output_sources = List[Series]()
            for c in right_output:
                right_output_sources.append(right._columns[c].copy())
            var output_height = len(right_rows)
            var gathered = take_parallel(
                right_output_sources,
                right_rows^,
                workers,
                or_null=how == JOIN_LEFT,
            )
            for k in range(len(right_output)):
                columns.append(gathered[k].renamed(right_names[k]))
            return DataFrame(columns^, height=output_height)
    # A right join is a left-major probe from the right input. Build on
    # the original left rows for high-cardinality keys, then swap the
    # resulting row lists back to the public output column order.
    var left_rows = List[Int]()
    var right_rows = List[Int]()
    var direct_right = False
    if self.height() == 0 and (
        how == JOIN_FULL or (how == JOIN_RIGHT and keep_right_keys)
    ):
        # No probe rows: every build row comes out once, in order, beside
        # null probe columns, with nothing hashed or gathered (a
        # pipeline's unmatched build rows join an empty probe side, as
        # DuckDB's outer scan reads them straight from its build).
        var m = right.height()
        var columns = List[Series](capacity=self.width() + len(right_output))
        for c in range(self.width()):
            ref column = self._columns[c]
            var key = -1
            if not keep_right_keys:
                for k in range(len(left_keys)):
                    if left_keys[k] == c:
                        key = right_keys[k]
            if key >= 0:
                columns.append(
                    right._columns[key]
                    .with_dtype(column.dtype())
                    .renamed(column.name())
                )
            else:
                columns.append(
                    Series.full_null(column.name(), column.dtype(), m)
                )
        for k in range(len(right_output)):
            columns.append(
                right._columns[right_output[k]].renamed(right_names[k])
            )
        return DataFrame(columns^, height=m)
    if how == JOIN_RIGHT and worker_count(right.height()) > 1:
        var right_probe_keys = List[Series](capacity=len(right_keys))
        var left_build_keys = List[Series](capacity=len(left_keys))
        for k in range(len(left_keys)):
            right_probe_keys.append(right._columns[right_keys[k]].copy())
            left_build_keys.append(self._columns[left_keys[k]].copy())
        var unfused = FusedJoin()
        if len(left_keys) == 1:
            direct_right = _bounded_int64_join_rows(
                right_probe_keys[0],
                left_build_keys[0],
                True,
                right_rows,
                left_rows,
                unfused,
            )
        if (
            not direct_right
            and self.height() <= Int(Int32.MAX)
            and not low_cardinality(left_build_keys)
        ):
            var pairs = direct_hash_join_rows(
                right_probe_keys, left_build_keys, True
            )
            direct_right = True
            swap(right_rows, pairs[0])
            swap(left_rows, pairs[1])
    if how == JOIN_FULL and right.height() <= Int(Int32.MAX):
        # A full join is a left join plus the right rows nothing
        # matched (DuckDB's FULL OUTER probe, then its scan of the
        # unmatched build rows): the left join's direct route, then
        # those rows in order, as the dictionary route below orders
        # them, without encoding every key of both inputs.
        var left_sources = List[Series](capacity=len(left_keys))
        var right_sources = List[Series](capacity=len(right_keys))
        for k in range(len(left_keys)):
            left_sources.append(self._columns[left_keys[k]].copy())
            right_sources.append(right._columns[right_keys[k]].copy())
        if not low_cardinality(right_sources):
            trace_path("join.full_direct")
            var pairs = direct_hash_join_rows(
                left_sources, right_sources, True, omit_identity=False
            )
            swap(left_rows, pairs[0])
            swap(right_rows, pairs[1])
            var matched = List[Bool](length=right.height(), fill=False)
            var at = right_rows.unsafe_ptr()
            for k in range(len(right_rows)):
                var j = at[unsafe_offset=k]
                if j >= 0:
                    matched[j] = True
            for j in range(right.height()):
                if not matched[j]:
                    left_rows.append(-1)
                    right_rows.append(j)
            direct_right = True
    if not direct_right:
        trace_path("join.dictionary")
        var ids = _joint_key_ids(self, right, left_keys, right_keys)
        var left_ids = ids[0].copy()
        var right_ids = ids[1].copy()
        var count = ids[2]
        # Rows per key id in a flat CSR layout: one list per distinct key
        # would be hundreds of thousands of heap allocations on a
        # high-cardinality join. Ids appear in increasing row order, so a
        # counting sort preserves the match order the contract documents.
        var right_starts = List[Int]()
        var right_flat = List[Int]()
        if how != JOIN_RIGHT:
            right_starts = _group_index(right_ids, count)
            var csr_workers = worker_count(len(right_ids))
            # Use the output/cursor working set and key cardinality to
            # decide whether the extra stable scatter passes pay off.
            right_flat = _parallel_group_rows(
                right_ids, right_starts, csr_workers
            ) if (
                (how == JOIN_INNER or how == JOIN_FULL)
                and _parallel_csr_fits(right_ids, count, csr_workers)
            ) else _group_rows(
                right_ids, right_starts
            )
        if how == JOIN_SEMI or how == JOIN_ANTI:
            for i in range(len(left_ids)):
                var id = left_ids[i]
                var matched = (
                    id >= 0 and right_starts[id + 1] > right_starts[id]
                )
                if matched == (how == JOIN_SEMI):
                    left_rows.append(i)
            return self.take(left_rows)
        if how == JOIN_RIGHT:
            var left_starts = _group_index(left_ids, count)
            var left_workers = worker_count(len(left_ids))
            var left_flat = _parallel_group_rows(
                left_ids, left_starts, left_workers
            ) if (
                _parallel_csr_fits(left_ids, count, left_workers)
            ) else _group_rows(
                left_ids, left_starts
            )
            var right_workers = worker_count(len(right_ids))
            if right_workers > 1:
                var pairs = _parallel_join_rows(
                    right_ids, left_starts, left_flat, right_workers, True
                )
                swap(right_rows, pairs[0])
                swap(left_rows, pairs[1])
            else:
                for j in range(len(right_ids)):
                    var id = right_ids[j]
                    if id >= 0 and left_starts[id + 1] > left_starts[id]:
                        for k in range(left_starts[id], left_starts[id + 1]):
                            left_rows.append(left_flat[k])
                            right_rows.append(j)
                    else:
                        left_rows.append(-1)
                        right_rows.append(j)
        elif how == JOIN_FULL and worker_count(len(left_ids)) > 1:
            var pairs = _parallel_full_join_rows(
                left_ids,
                right_ids,
                right_starts,
                right_flat,
                count,
                worker_count(len(left_ids)),
            )
            swap(left_rows, pairs[0])
            swap(right_rows, pairs[1])
        elif (how == JOIN_INNER or how == JOIN_LEFT) and worker_count(
            len(left_ids)
        ) > 1:
            var pairs = _parallel_join_rows(
                left_ids,
                right_starts,
                right_flat,
                worker_count(len(left_ids)),
                how == JOIN_LEFT,
            )
            swap(left_rows, pairs[0])
            swap(right_rows, pairs[1])
        else:
            var right_matched = List[Bool](length=len(right_ids), fill=False)
            for i in range(len(left_ids)):
                var id = left_ids[i]
                if id >= 0 and right_starts[id + 1] > right_starts[id]:
                    for k in range(right_starts[id], right_starts[id + 1]):
                        var j = right_flat[k]
                        left_rows.append(i)
                        right_rows.append(j)
                        right_matched[j] = True
                elif how != JOIN_INNER:
                    left_rows.append(i)
                    right_rows.append(-1)
            if how == JOIN_FULL:
                for j in range(len(right_ids)):
                    if not right_matched[j]:
                        left_rows.append(-1)
                        right_rows.append(j)
    if how == JOIN_RIGHT and keep_right_keys:
        # Right keys kept apart: every left column, keys included, as
        # the left rows have them (null where none matched), then the
        # right columns, keys among them.
        var gather_workers = worker_count(len(left_rows))
        var left_sources = List[Series](capacity=self.width())
        for c in range(self.width()):
            left_sources.append(self._columns[c].copy())
        var columns = take_parallel(
            left_sources, left_rows.copy(), gather_workers, or_null=True
        )
        var right_sources = List[Series]()
        for c in right_output:
            right_sources.append(right._columns[c].copy())
        var gathered = take_parallel(
            right_sources, right_rows^, gather_workers, or_null=False
        )
        for k in range(len(right_output)):
            columns.append(gathered[k].renamed(right_names[k]))
        return DataFrame(columns^, height=len(left_rows))
    if how == JOIN_RIGHT:
        # Every output row has a right row. Its key is the coalesced key
        # even when no left row matched, so never gather a left key or
        # choose between duplicate key columns after materialization.
        var gather_workers = worker_count(len(left_rows))
        var left_sources = List[Series]()
        for c in range(self.width()):
            if c not in left_keys:
                left_sources.append(self._columns[c].copy())
        var left_identity = len(left_rows) == self.height()
        if left_identity:
            for i in range(len(left_rows)):
                if left_rows[i] != i:
                    left_identity = False
                    break
        var left_gathered = left_sources^
        if not left_identity:
            left_gathered = take_parallel(
                left_gathered,
                left_rows.copy(),
                gather_workers,
                or_null=True,
            )
        var right_sources = List[Series]()
        for k in range(len(right_keys)):
            right_sources.append(right._columns[right_keys[k]].copy())
        for c in right_output:
            right_sources.append(right._columns[c].copy())
        var right_gathered = take_parallel(
            right_sources, right_rows^, gather_workers, or_null=False
        )
        var columns = List[Series](capacity=self.width() + len(right_output))
        var left_source = 0
        for c in range(self.width()):
            var key = -1
            for k in range(len(left_keys)):
                if left_keys[k] == c:
                    key = k
                    break
            if key >= 0:
                columns.append(
                    right_gathered[key].renamed(self._columns[c].name())
                )
            else:
                columns.append(left_gathered[left_source].copy())
                left_source += 1
        for k in range(len(right_output)):
            columns.append(
                right_gathered[len(right_keys) + k].renamed(right_names[k])
            )
        return DataFrame(columns^, height=len(left_rows))
    # Assembling the output is about half of a join, and it used to
    # gather one column at a time on the calling thread. take_parallel
    # writes disjoint output ranges, so every column of both sides goes
    # at once; `or_null` is what lets it carry the -1 that means "no row
    # on this side".
    var gather_workers = worker_count(len(left_rows))
    var sides_mixed = how == JOIN_RIGHT or how == JOIN_FULL
    var left_sources = List[Series](capacity=self.width())
    for c in range(self.width()):
        left_sources.append(self._columns[c].copy())
    # When every left row appears once in input order, its columns are
    # already the exact output. Sharing them avoids a full-frame gather.
    var left_identity = len(left_rows) == self.height()
    if left_identity:
        for i in range(len(left_rows)):
            if left_rows[i] != i:
                left_identity = False
                break
    var columns = left_sources^
    if not left_identity:
        columns = take_parallel(
            columns,
            left_rows.copy(),
            gather_workers,
            or_null=sides_mixed,
        )

    # Right-side columns: the non-key output columns, plus any key
    # column that has to be coalesced with its left counterpart.
    var right_sources = List[Series]()
    var coalesced = List[Int]()
    for c in range(self.width()):
        for k in range(len(left_keys)):
            if left_keys[k] == c and sides_mixed and not keep_right_keys:
                coalesced.append(c)
                right_sources.append(right._columns[right_keys[k]].copy())
    for k in range(len(right_output)):
        right_sources.append(right._columns[right_output[k]].copy())
    var from_right = take_parallel(
        right_sources,
        right_rows.copy(),
        gather_workers,
        or_null=how == JOIN_LEFT or how == JOIN_FULL,
    )

    for j in range(len(coalesced)):
        var c = coalesced[j]
        var use_left = List[Bool](capacity=len(left_rows))
        for i in left_rows:
            use_left.append(i >= 0)
        columns[c] = (
            choose(use_left, columns[c], from_right[j])
            .with_dtype(self._columns[c].dtype())
            .renamed(self._columns[c].name())
        )
    for k in range(len(right_output)):
        columns.append(from_right[len(coalesced) + k].renamed(right_names[k]))
    return DataFrame(columns^, height=len(left_rows))


def _smaller_build_join_rows(
    left: List[Series],
    right: List[Series],
    include_unmatched: Bool,
    keep_order: Bool = True,
) raises -> Tuple[List[Int], List[Int]]:
    """Build on logical left, then stably restore left-major/right-input order.

    The physical probe emits right-major pairs. Stable counting scatter by
    logical left row preserves the original right-row ordering within every
    left group, including duplicate keys on either side. Without
    `keep_order` (a join whose consumers ignore order) the pairs are
    returned right-major as probed: the scatter cost more than the probe
    (PDS-H q5: 750K pairs over 187K orders, 18 ms against 13). A left join
    then emits the unmatched build rows after the pairs, as Polars'
    `emit_unmatched_build` and DuckDB's right-outer scan of a swapped join
    do: the build rows no probe row matched are flagged while the pairs are
    taken, and appended with no right row.
    """
    trace_path("join.smaller_build")
    var pairs = direct_hash_join_rows(right, left, False)
    if not keep_order:
        trace_path("join.smaller_build_unordered")
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        swap(left_rows, pairs[1])
        swap(right_rows, pairs[0])
        if include_unmatched:
            var count = len(left[0])
            var matched = List[Bool](length=count, fill=False)
            var seen = matched.unsafe_ptr()
            var rows = left_rows.unsafe_ptr()
            for k in range(len(left_rows)):
                seen[unsafe_offset=rows[unsafe_offset=k]] = True
            for i in range(count):
                if not seen[unsafe_offset=i]:
                    left_rows.append(i)
                    right_rows.append(-1)
        return (left_rows^, right_rows^)
    # A counting sort of the pairs by left row, written straight into the
    # output: the probe's right rows land in their left group in probe
    # order, and the left row of every output position follows from the
    # group starts. Through pointers, in two passes over the pairs: the
    # list-indexed three passes before cost 19 ns a pair (PDS-H q2, 159K
    # pairs, 3 ms on one thread).
    var count = len(left[0])
    var n = len(pairs[0])
    # On several workers when there are at least as many pairs a worker
    # as left rows: each worker keeps a count per left row, so more
    # workers than that would count more than they scatter.
    var lanes = min(worker_count(n), max(1, n // max(count, 1)))
    if lanes > 1 and not include_unmatched:
        return _parallel_left_major(pairs[1], pairs[0], count, lanes)
    var pairs_left = pairs[1].unsafe_ptr()
    var pairs_right = pairs[0].unsafe_ptr()
    var starts = _group_index(pairs[1], count)
    var cursor = starts.copy()
    var unmatched = 0
    if include_unmatched:
        for i in range(count):
            if starts[i] == starts[i + 1]:
                unmatched += 1
    var total = n + unmatched
    var left_rows = List[Int](unsafe_uninit_length=total)
    var right_rows = List[Int](unsafe_uninit_length=total)
    var out_left = left_rows.unsafe_ptr()
    var out_right = right_rows.unsafe_ptr()
    if unmatched == 0:
        var at = cursor.unsafe_ptr()
        for k in range(n):
            var i = pairs_left[unsafe_offset=k]
            var position = at[unsafe_offset=i]
            at[unsafe_offset=i] = position + 1
            out_right[unsafe_offset=position] = pairs_right[unsafe_offset=k]
        var group_start = starts.unsafe_ptr()
        for i in range(count):
            for position in range(
                group_start[unsafe_offset=i], group_start[unsafe_offset=i + 1]
            ):
                out_left[unsafe_offset=position] = i
        return (left_rows^, right_rows^)
    # With unmatched left rows an output position shifts by the unmatched
    # rows before its group: the group starts move accordingly.
    var shifted = List[Int](unsafe_uninit_length=count + 1)
    var shift = shifted.unsafe_ptr()
    var seen = 0
    for i in range(count):
        shift[unsafe_offset=i] = starts[i] + seen
        if starts[i] == starts[i + 1]:
            out_left[unsafe_offset=starts[i] + seen] = i
            out_right[unsafe_offset=starts[i] + seen] = -1
            seen += 1
    shift[unsafe_offset=count] = n + seen
    var at = List[Int](unsafe_uninit_length=count)
    var fill = at.unsafe_ptr()
    for i in range(count):
        fill[unsafe_offset=i] = shift[unsafe_offset=i]
    for k in range(n):
        var i = pairs_left[unsafe_offset=k]
        var position = fill[unsafe_offset=i]
        fill[unsafe_offset=i] = position + 1
        out_left[unsafe_offset=position] = i
        out_right[unsafe_offset=position] = pairs_right[unsafe_offset=k]
    return (left_rows^, right_rows^)


def _left_builds(left: DataFrame, right: DataFrame) -> Bool:
    """Whether a join builds its hash table on the left input: choose the
    input with fewer rows times columns, choosing the right on a tie, as
    the lazy planner does. For example, a 747-row left input and a 159K-row
    right input (PDS-H q2) select the right side to build."""
    var left_rows = left.height()
    if left_rows == 0 or left_rows > Int(Int32.MAX):
        return False
    return left_rows * max(left.width(), 1) < right.height() * max(
        right.width(), 1
    )


def cross_join_frames(
    self: DataFrame, right: DataFrame, total: Int, right_names: List[String]
) raises -> DataFrame:
    """Materialize validated cross-join pairs in left-major order."""
    var left_rows = List[Int](capacity=total)
    var right_rows = List[Int](capacity=total)
    for i in range(self._height):
        for j in range(right._height):
            left_rows.append(i)
            right_rows.append(j)
    var columns = List[Series]()
    for column in self._columns:
        columns.append(column.take(left_rows))
    for k in range(right.width()):
        columns.append(
            right._columns[k].take(right_rows).renamed(right_names[k])
        )
    return DataFrame(columns^, height=total)
