"""Eager joins: DuckDB's hash join (`joins/hash_table.mojo`) over two
frames, and the output's assembly."""
from dataframe.aggregation.keys import (
    _expand_struct_keys,
)
from dataframe.categorical import (
    unify,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.kernels.binary import (
    choose,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.kernels.gather import (
    take_parallel,
)
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
    join_name,
)
from dataframe.joins.hash_table import (
    JoinHashTable,
    keys_only_join,
    probe_pairs,
)
from dataframe.parallel import (
    worker_count,
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


def join_frames(
    self: DataFrame,
    right: DataFrame,
    *,
    left_on: List[String],
    right_on: List[String],
    how: Int = JOIN_INNER,
    suffix: String = "_right",
    coalesce: Bool = True,
    prepared: Optional[JoinHashTable] = None,
    keep_order: Bool = True,
    workers: Int = 0,
) raises -> DataFrame:
    """Execute a validated join. `prepared` is a hash table already built
    over `right`'s keys, which `self` probes (a pipeline worker's morsel;
    `workers` > 0 keeps the probe on this thread)."""
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
    # A coalesced right join keeps the right keys, under their names and
    # in their places among the right columns, after the left's other
    # columns, as Polars' right join (a left join with the sides
    # swapped) names them: the left keys are what it drops.
    var right_coalesced = how == JOIN_RIGHT and coalesce
    var right_output = List[Int]()
    var right_names = List[String]()
    if how != JOIN_SEMI and how != JOIN_ANTI:
        var names = Dict[String, Bool]()
        for c in range(self.width()):
            if right_coalesced and c in left_keys:
                continue
            names[self._columns[c].name()] = True
        var left_names = names.copy()
        for i in range(right.width()):
            if not keep_right_keys and not right_coalesced and i in right_keys:
                continue
            var name = right._columns[i].name()
            if name in left_names:
                name += suffix
            if name in names:
                raise Error("Join output name collision: " + name)
            names[name] = True
            right_output.append(i)
            right_names.append(name)
    var left_rows = List[Int]()
    var right_rows = List[Int]()
    if prepared:
        # A pipeline worker's morsel probing the shared table of the build
        # (`right`): right and full joins flag the build rows they match
        # in the table, and the pipeline emits the unmatched ones after
        # every morsel (`scan_full_outer`), so a right join emits only its
        # matched pairs here.
        var probe_keys = List[Series](capacity=len(left_keys))
        for k in left_keys:
            probe_keys.append(self._columns[k].copy())
        var pairs = probe_pairs(
            prepared.value(),
            probe_keys,
            how,
            1 if workers > 0 else 0,
            scan=False,
        )
        if how == JOIN_SEMI or how == JOIN_ANTI:
            if len(pairs[0]) == self.height():
                return self.copy()
            return self._filter_rows(pairs[0].copy())
        var output_height = len(pairs[0])
        var lanes = workers if workers > 0 else worker_count(output_height)
        var columns = self._columns.copy()
        var identity = output_height == self.height()
        if identity:
            for i in range(output_height):
                if pairs[0][i] != i:
                    identity = False
                    break
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
            or_null=how == JOIN_LEFT or how == JOIN_FULL,
        )
        for k in range(len(right_output)):
            columns.append(gathered[k].renamed(right_names[k]))
        return DataFrame(columns^, height=output_height)
    if self.height() == 0 and (how == JOIN_FULL or how == JOIN_RIGHT):
        # No probe rows: every build row comes out once, in order, beside
        # null probe columns, with nothing hashed or gathered (a
        # pipeline's unmatched build rows join an empty probe side, as
        # DuckDB's outer scan reads them straight from its build).
        var m = right.height()
        var columns = List[Series](capacity=self.width() + len(right_output))
        for c in range(self.width()):
            ref column = self._columns[c]
            if right_coalesced and c in left_keys:
                continue
            var key = -1
            if not keep_right_keys and not right_coalesced:
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
    # The join's sides, as DuckDB's hash join runs it: the probe streams
    # through the table built on the other side. A right join probes its
    # right input against the left (a left join with the sides swapped,
    # Polars' order); every other kind probes the left. The side with
    # the smaller hash table builds (`BuildProbeSideOptimizer`, on exact
    # sizes here): a flipped join probes the other input with the inverse
    # join type, and a semi or anti join becomes DuckDB's RIGHT_SEMI or
    # RIGHT_ANTI, which flags the build rows and emits them by a scan.
    var probe_left = how != JOIN_RIGHT
    var left_cost = build_size(self)
    var right_cost = build_size(right)
    var flip = right_cost > left_cost if probe_left else left_cost > right_cost
    if flip:
        trace_path("join.build_flipped")
        probe_left = not probe_left
    var physical = how
    if flip:
        if how == JOIN_LEFT:
            physical = JOIN_RIGHT
        elif how == JOIN_SEMI:
            physical = JOIN_RIGHT_SEMI
        elif how == JOIN_ANTI:
            physical = JOIN_RIGHT_ANTI
    elif how == JOIN_RIGHT:
        physical = JOIN_LEFT
    var probe_keys = List[Series](capacity=len(left_keys))
    var build_keys = List[Series](capacity=len(left_keys))
    for k in range(len(left_keys)):
        if probe_left:
            probe_keys.append(self._columns[left_keys[k]].copy())
            build_keys.append(right._columns[right_keys[k]].copy())
        else:
            probe_keys.append(right._columns[right_keys[k]].copy())
            build_keys.append(self._columns[left_keys[k]].copy())
    trace_path("join.hash_table")
    var table = JoinHashTable(
        build_keys,
        perfect=physical == JOIN_INNER,
        keys_only=keys_only_join(physical),
    )
    var pairs = Tuple[List[Int], List[Int]](List[Int](), List[Int]())
    if table.count == 0 and (
        physical == JOIN_INNER
        or physical == JOIN_SEMI
        or physical == JOIN_RIGHT_SEMI
    ):
        # An empty build matches no probe row (DuckDB's
        # `EmptyResultIfRHSIsEmpty`: the hash join's sink finalizes as
        # NO_OUTPUT_POSSIBLE), so nothing is probed.
        trace_path("join.empty_build")
    else:
        pairs = probe_pairs(table, probe_keys, physical, 0)
    if how == JOIN_SEMI or how == JOIN_ANTI:
        var kept = List[Int]()
        if probe_left:
            swap(kept, pairs[0])
        else:
            swap(kept, pairs[1])
        if len(kept) == self.height():
            return self.copy()
        return self._filter_rows(kept^)
    if probe_left:
        swap(left_rows, pairs[0])
        swap(right_rows, pairs[1])
    else:
        swap(left_rows, pairs[1])
        swap(right_rows, pairs[0])
    if flip and keep_order:
        # A caller that reads the rows in order gets them as the unflipped
        # join emits them: by the row of the side it probes (a right join's
        # right input, any other's left), rows with none at the end.
        if how == JOIN_RIGHT:
            _order_pairs(right_rows, left_rows, right.height())
        else:
            _order_pairs(left_rows, right_rows, self.height())
    if how == JOIN_RIGHT:
        # Every left column, as the left rows have them (null where none
        # matched), but the keys of a coalesced join, then the right
        # columns, keys among them.
        var gather_workers = worker_count(len(left_rows))
        var left_sources = List[Series](capacity=self.width())
        for c in range(self.width()):
            if not (right_coalesced and c in left_keys):
                left_sources.append(self._columns[c].copy())
        var left_identity = len(left_rows) == self.height()
        if left_identity:
            for i in range(len(left_rows)):
                if left_rows[i] != i:
                    left_identity = False
                    break
        var columns = left_sources^
        if not left_identity:
            columns = take_parallel(
                columns, left_rows.copy(), gather_workers, or_null=True
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
                # One output column takes rows of both keys, as a
                # concatenation does: their Arrow field metadata must agree.
                self._columns[c]._check_metadata(right._columns[right_keys[k]])
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


# DuckDB's BuildProbeSideOptimizer: a column's bytes in a hash table row
# beyond its TupleDataLayout width, and the pointer table's bytes a row
# (about three ht_entry_t, the average of NextPowerOfTwo(2 * count)).
comptime _COLUMN_COUNT_PENALTY = 2
comptime _VARCHAR_PENALTY = 8
comptime _NESTED_PENALTY = 32
comptime _ENTRY_BYTES_PER_ROW = 3 * 8


def build_size(frame: DataFrame) -> Float64:
    """The bytes a hash table built on `frame` holds
    (BuildProbeSideOptimizer::GetBuildSize): its row width
    (`build_row_width`) times its rows."""
    return max(Float64(build_row_width(frame)) * Float64(frame.height()), 8.0)


def build_row_width(frame: DataFrame) -> Int:
    """The bytes a row of `frame`'s columns takes in a hash table
    (BuildProbeSideOptimizer::GetBuildSize): its row width (validity
    bytes, each column's width, the hash) with penalties for strings,
    nested columns and every column, and the pointer table's share. Only
    the columns' types are read, so an empty frame of a plan's schema
    gives the width its rows will have."""
    var width = (frame.width() + 1 + 7) // 8 + 8
    for column in frame._columns:
        var dtype = column.dtype()
        if dtype.is_nested():
            width += 16 + _NESTED_PENALTY
        elif dtype.physical() == DataType.STRING:
            width += 16 + _VARCHAR_PENALTY
        else:
            width += max(dtype.physical().bit_width() // 8, 1)
        width += _COLUMN_COUNT_PENALTY
    return width + _ENTRY_BYTES_PER_ROW


def _order_pairs(mut major: List[Int], mut minor: List[Int], rows: Int):
    """Reorder the pairs stably by `major` (a row among `rows`), pairs
    whose major row is -1 last: a counting sort."""
    var n = len(major)
    var starts = List[Int](length=rows + 2, fill=0)
    var at = starts.unsafe_ptr()
    var by = major.unsafe_ptr()
    for k in range(n):
        var row = by[unsafe_offset=k]
        at[unsafe_offset=(row if row >= 0 else rows) + 1] += 1
    for r in range(rows + 1):
        at[unsafe_offset=r + 1] += at[unsafe_offset=r]
    var sorted_major = List[Int](unsafe_uninit_length=n)
    var sorted_minor = List[Int](unsafe_uninit_length=n)
    var out_major = sorted_major.unsafe_ptr()
    var out_minor = sorted_minor.unsafe_ptr()
    var other = minor.unsafe_ptr()
    for k in range(n):
        var row = by[unsafe_offset=k]
        var slot = row if row >= 0 else rows
        var position = at[unsafe_offset=slot]
        at[unsafe_offset=slot] = position + 1
        out_major[unsafe_offset=position] = row
        out_minor[unsafe_offset=position] = other[unsafe_offset=k]
    major = sorted_major^
    minor = sorted_minor^


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
