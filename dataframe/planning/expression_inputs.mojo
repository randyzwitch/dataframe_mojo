"""Expression input references, independent of schema and strategy policy."""
from dataframe.expr import (
    COL,
    Expr,
    OVER,
    SELECTOR,
    split_names,
)
from std.collections import (
    Optional,
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
            for part in split_names(node.text2):
                names.append(part)
    return names^
