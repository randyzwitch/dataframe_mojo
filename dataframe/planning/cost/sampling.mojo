"""Shared strategy observations, sampling rules, and memory budgets."""
from dataframe.aggregation.expressions import (
    _row_local,
)
from dataframe.aggregation.state import (
    _StreamReduction,
)
from dataframe.distinct_statistics import (
    distinct_estimate,
)
from dataframe.expr import (
    Expr,
)
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.hashing import (
    encode_rows,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from dataframe.planning.model import (
    FILTER,
    PlanNode,
)
from dataframe.series import (
    Series,
)
from std.os import (
    getenv,
)


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


def _stream_collect_bytes() -> Int:
    """Bytes of collected batches a many-group streaming aggregation holds
    before reducing them into a mergeable state and continuing, so its
    memory stays bounded whatever the input's length. Default 4 GiB;
    DATAFRAME_STREAM_COLLECT_BYTES overrides it, so tests can exercise the
    bound on small inputs."""
    var setting = getenv("DATAFRAME_STREAM_COLLECT_BYTES")
    if setting.byte_length() > 0:
        try:
            return max(1, Int(setting))
        except:
            pass
    return 4 << 30


def _frame_bytes(frame: DataFrame) -> Int:
    """Approximate bytes a frame's columns hold: fixed widths by dtype,
    strings at 16 bytes a row plus their descriptors (an estimate; the
    exact text length of a view-backed window costs a pass over it)."""
    var total = 0
    for column in frame._columns:
        var bits = column.dtype().bit_width()
        if bits > 0:
            total += (frame.height() * bits + 7) // 8
        else:
            total += 32 * frame.height()
    return total


def _held_groups(
    held: List[DataFrame], keys: List[String], limit: Int
) raises -> Tuple[Int, Int]:
    """Distinct key rows among the first `limit` rows of the held batches,
    and how many rows that was."""
    var parts = List[DataFrame](capacity=len(held))
    var rows = 0
    for frame in held:
        if rows >= limit:
            break
        var take = min(frame.height(), limit - rows)
        parts.append(frame.select(keys).slice(0, take))
        rows += take
    var whole = concat(parts).rechunk()
    return (encode_rows(whole._columns, nulls_equal=True).count(), rows)


def _many_groups(groups: Int, rows: Int, held: Int, split_groups: Int) -> Bool:
    """Whether `groups` distinct keys among `rows` sampled of `held` rows
    mean a many-group stream: `split_groups` groups at one row in eight
    starting one, which take eight times `split_groups` rows to show. A
    sample of at least half the rows is read as exact (the stream may
    be about this long: PDS-H q16 at a tenth scale has 2.8K groups in
    11.6K rows, few); a smaller one extrapolates to a quarter of eight
    times `split_groups` rows."""
    if 2 * rows >= held:
        return groups >= split_groups and 8 * groups >= rows
    return 4 * groups >= split_groups and 8 * groups >= rows


def _batch_shows_many_groups(
    frame: DataFrame, keys: List[String], split_groups: Int
) raises -> Bool:
    """A long batch's own judgement from its first 2 * split_groups rows,
    by the stream's rules (`_judge_held`): a batch that shows many groups
    keeps only its frame, since the stream will most likely collect and
    a state built now would be dropped (TPC-DS q39: 10 ms for 95K rows);
    one that shows few builds its state in this job, where a replay later
    would cost a round."""
    var sample = frame.slice(0, min(frame.height(), 2 * split_groups))
    if _key_text_bytes(sample.slice(0, 1024), keys) > 64 * 1024:
        return False
    var groups = encode_rows(
        sample.select(keys)._columns, nulls_equal=True
    ).count()
    return _many_groups(groups, sample.height(), frame.height(), split_groups)


def _judge_held(
    held: List[DataFrame],
    states: List[List[_StreamReduction]],
    keys: List[String],
    held_rows: Int,
    split_groups: Int,
    distinct_inputs: List[String] = List[String](),
) raises -> Bool:
    """Whether the held batches show many groups (`_many_groups`) with
    keys short enough to collect (`_key_text_bytes`). The held states'
    group counts bound the union's from above: when even that bound is
    few groups, no sample is encoded. A distinct count's state is a set
    of values per group, which merging repeats the distinct aggregation
    over: its (key, value) pairs count as groups (PDS-H q16 probing 800K
    partsupp rows in batches showed few part groups but nearly a pair a
    row, and spent 29 ms merging its batches' sets)."""
    if _key_text_bytes(held[0].slice(0, 1024), keys) > 64 * 1024:
        return False
    for name in distinct_inputs:
        var pair = keys.copy()
        if name not in pair:
            pair.append(name)
        var sampled = _held_groups(held, pair, 2 * split_groups)
        if _many_groups(sampled[0], sampled[1], held_rows, split_groups):
            return True
    # One batch's state showing many groups on its own settles it.
    for h in range(len(states)):
        if len(states[h]) == 1 and _many_groups(
            states[h][0].group_count(),
            held[h].height(),
            held[h].height(),
            split_groups,
        ):
            return True
    var groups = _held_group_bound(states)
    var rows = held_rows
    if not (_all_reduced(states) and 8 * groups < rows):
        var sampled = _held_groups(held, keys, 2 * split_groups)
        groups = sampled[0]
        rows = sampled[1]
    return _many_groups(groups, rows, held_rows, split_groups)


def _held_group_bound(states: List[List[_StreamReduction]]) -> Int:
    """The held states' group counts summed: at least the distinct groups
    among their rows."""
    var total = 0
    for h in range(len(states)):
        for k in range(len(states[h])):
            total += states[h][k].group_count()
    return total


def _all_reduced(states: List[List[_StreamReduction]]) -> Bool:
    for h in range(len(states)):
        if len(states[h]) == 0:
            return False
    return True


def _key_text_bytes(frame: DataFrame, keys: List[String]) raises -> Int:
    """Bytes of string text among a batch's key columns. Composite keys
    with long strings keep the batch states: the eager bucket encode
    confirms such keys row by row through scattered reads (#486), and
    PDS-H q10 (seven keys, c_comment among them, about 140 text bytes a
    row) was 22% slower collected, where TPC-DS q39 (one short name
    beside three integers) gains 40%."""
    var total = 0
    for name in keys:
        var column = frame.column(name)
        if column.dtype().bit_width() == 0:
            total += column._text_bytes()
    return total


def _keys_repeat(
    input: DataFrame, keys: List[String], operations: List[PlanNode]
) raises -> Bool:
    """Whether the plan's grouping keys repeat: among the rows of eight
    evenly spaced runs of 512 rows of `input` that pass the plan's leading
    row-local filters, at most half as many distinct keys as rows. Runs,
    not single rows, see the repeats a morsel does. False when too few
    rows pass to say (under 64) or a filter reads what `input` lacks."""
    var height = input.height()
    if height == 0:
        return False
    var names = keys.copy()
    var filters = List[Expr]()
    for operation in operations:
        if operation.kind != FILTER or not _row_local(operation.exprs):
            break
        filters.append(operation.exprs[0].copy())
        var reads = _references(operation.exprs[0])
        if not reads:
            return False
        for name in reads.value():
            if name not in names:
                names.append(name)
    for name in names:
        if name not in input.columns():
            return False
    var narrowed = input.select(names)
    var runs = 8
    var length = 512
    var rows = List[Int]()
    if height <= runs * length:
        for r in range(height):
            rows.append(r)
    else:
        for k in range(runs):
            var start = k * (height - length) // (runs - 1)
            for r in range(start, start + length):
                rows.append(r)
    var picked = narrowed.take(rows)
    for f in filters:
        picked = picked.filter(f)
    if picked.height() < 64:
        return False
    var columns = List[Series](capacity=len(keys))
    for name in keys:
        columns.append(picked.column(name))
    var distinct = encode_rows(columns, nulls_equal=True).count()
    return 2 * distinct <= picked.height()


def _key_group_bound(
    keys: List[String], source: DataFrame, builds: List[DataFrame], limit: Int
) raises -> Int:
    """An upper bound on the groups of `keys`, or -1 once it passes `limit`
    or when a key is in no frame (made by a step). Keys held by one frame
    count together, as that frame's distinct key rows; the frames' counts
    multiply. A frame's height bounds its count for free, so frames are
    taken smallest first and sampled (evenly spaced rows, a sample with
    every row distinct standing for the whole height) only where the
    height alone would pass the limit."""
    # Each key's holder: -2 the source, else a build's index.
    var holders = List[Int]()
    for name in keys:
        var holder = -1
        if name in source.columns():
            holder = -2
        else:
            for b in range(len(builds)):
                if name in builds[b].columns():
                    holder = b
                    break
        if holder == -1:
            return -1
        holders.append(holder)
    var frames = List[Int]()
    for h in holders:
        if h not in frames:
            frames.append(h)
    # Smallest frame first.
    var heights = List[Int]()
    for h in frames:
        heights.append(source.height() if h == -2 else builds[h].height())
    for i in range(1, len(frames)):
        var j = i
        while j > 0 and heights[j - 1] > heights[j]:
            heights.swap_elements(j - 1, j)
            frames.swap_elements(j - 1, j)
            j -= 1
    var bound = 1
    for f in range(len(frames)):
        var height = heights[f]
        var count = max(height, 1)
        if bound * count > limit:
            # Only whether this frame's count passes what is left of the
            # limit matters, so the sample is twice that remainder: a
            # sample with every row distinct ends the bound (q46's three
            # source keys over 2.9M rows: six rows, not 4,096, whose
            # strided reads took 1.3 ms).
            var left = limit // bound
            var sample = min(height, 4096, 2 * (left + 1))
            var rows = List[Int](capacity=sample)
            for k in range(sample):
                rows.append(k * height // sample)
            var picked = List[Series]()
            for i in range(len(keys)):
                if holders[i] != frames[f]:
                    continue
                if frames[f] == -2:
                    picked.append(source.column(keys[i]).take(rows))
                else:
                    picked.append(builds[frames[f]].column(keys[i]).take(rows))
            var distinct = encode_rows(picked, nulls_equal=True).count()
            count = height if distinct == sample else max(distinct, 1)
        bound *= count
        if bound > limit:
            return -1
    return bound


def _unique_key_column(build: DataFrame, names: List[String]) raises -> Bool:
    """Whether one of the build's columns `names` holds no null and no value
    twice, so no two rows of the build hold the same values for all of
    them. A column whose sampled distinct count is under nine tenths of the
    rows is not checked."""
    var height = build.height()
    for name in names:
        var column = build.column(name)
        if column.null_count() > 0:
            continue
        if 10 * distinct_estimate(column) < 9 * height:
            continue
        if column.n_unique() == height:
            return True
    return False
