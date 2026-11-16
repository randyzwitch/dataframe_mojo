"""A join build's finish: its columns and key index."""
from dataframe.frame import (
    DataFrame,
)
from dataframe.joins.hash_table import (
    JoinHashTable,
    build_from_columns,
    join_build_layout,
)
from std.collections import (
    Optional,
)
from std.memory import (
    ArcPointer,
)


def finish_build(
    mut built: DataFrame,
    keys: List[String],
    how: Int,
    coalesce: Bool = True,
    probe_estimate: Int = -1,
    build_filtered: Bool = False,
    defer_finalize: Bool = False,
) raises -> Optional[JoinHashTable]:
    """Finish a join build by retaining its columns and constructing its key
    index. Return no index for absent or unsupported keys. Semi/anti joins
    need no build payload because they emit only the preserved side.
    probe_estimate describes the expected probe cardinality, and
    build_filtered whether the build side filters its rows.
    `defer_finalize`: the caller's finish event runs the table's finalize
    tasks (`JoinHashTable.finalize_tasks`)."""
    for column in built._columns:
        if column.is_chunked():
            built = built.rechunk()
            break
    if len(keys) == 0 or built.height() > Int(Int32.MAX):
        return None
    var layout = join_build_layout(
        built._columns,
        keys,
        how,
        coalesce,
        probe_estimate,
        False,
        build_filtered=build_filtered,
    )
    if not layout:
        return None
    return build_from_columns(
        ArcPointer(layout.take()), built._columns, defer_finalize
    )
