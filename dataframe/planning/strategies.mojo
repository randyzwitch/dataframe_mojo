"""Sampling and memory-budget rules shared by physical planning and streaming."""
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.hashing import (
    encode_rows,
)
from dataframe.aggregation.state import (
    _StreamReduction,
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
