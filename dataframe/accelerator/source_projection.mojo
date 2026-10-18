"""Narrow captured frame payloads before expensive host/device preparation."""
from .metadata import prune_projections
from .selectors import _live_nodes
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame, SCAN_FRAME


def project_sources(query: LazyFrame) raises -> LazyFrame:
    """Apply shared projection analysis and retain each scan's own frame view.

    No expression or source row is evaluated. Separate views matter when two
    scans of the same immutable frame need different columns. Unreachable
    scans retain only their schema, so dictionary staging cannot inspect their
    payloads. Callers validate the original schema before this transformation
    and keep certified execution graphs on their existing preparation path.
    """
    var result = query.copy()
    prune_projections(result)
    var live = _live_nodes(result)
    result._frames = List[DataFrame]()
    for index in range(len(result._nodes)):
        ref node = result._nodes[index]
        if node.kind != SCAN_FRAME:
            continue
        var frame = query._frames[node.offset].copy()
        if not live[index]:
            frame = frame.clear()
        elif len(node.names):
            frame = frame.select(node.names)
        node.offset = len(result._frames)
        result._frames.append(frame^)
    return result^
