"""Metadata admission for scoped primitive List EXPLODE regions."""
from dataframe import DataFrame, Series
from dataframe.lazy import LazyFrame, EXPLODE
from .capabilities import RowCapabilities
from .selectors import _live_nodes
from .source_lists import fixed, child_prototype


def has_explode(query: LazyFrame) -> Bool:
    var live = _live_nodes(query)
    for index in range(len(query._nodes)):
        if live[index] and query._nodes[index].kind == EXPLODE:
            return True
    return False


def schema(
    source: DataFrame, names: List[String], capabilities: RowCapabilities
) raises -> DataFrame:
    if not capabilities.primitive_list_explode:
        capabilities.reject(
            "explode", "primitive List expansion is unavailable"
        )
    if not len(names):
        raise Error("explode needs at least one column")
    var selected = List[Bool](length=source.width(), fill=False)
    for name in names:
        var index = source._index(name)
        if selected[index]:
            raise Error("explode: column listed twice: " + name)
        if not source._columns[index].dtype().is_list():
            raise Error(
                "explode needs list columns; "
                + name
                + " is "
                + source._columns[index].dtype().name()
            )
        selected[index] = True
    if source.width() > capabilities.max_join_columns:
        capabilities.reject("explode", "primitive output exceeds column limit")
    var columns = List[Series]()
    for index in range(source.width()):
        ref input = source._columns[index]
        if input.dtype().is_list():
            if not selected[index]:
                capabilities.reject(
                    "explode",
                    "unselected nested outputs require a recursive field owner",
                )
            if not fixed(input.dtype().inner()):
                capabilities.reject(
                    "explode", "primitive fixed-width List children required"
                )
            capabilities.require_dtype(input.dtype().inner())
            columns.append(child_prototype(input).renamed(input.name()))
        else:
            if not fixed(input.dtype()):
                capabilities.reject(
                    "explode", "ordinary fields require fixed-width storage"
                )
            capabilities.require_dtype(input.dtype())
            columns.append(input.slice(0, 0))
    return DataFrame(columns^)
