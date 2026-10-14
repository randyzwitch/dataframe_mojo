"""Terminal struct packing over primitive accelerator output columns.

CPU STRUCT_PACK always produces valid parents, even when every child is null.
This adapter implements that contract only. Nullable source structs, nested
children and intermediate struct consumers require resident field mappings.
"""
from std.collections import Dict
from dataframe.frame import DataFrame
from dataframe.dtype import DataType
from dataframe.series import Series
from dataframe.binding import bind
from dataframe.expr import (
    COL,
    STRUCT_PACK,
    STRUCT_FIELD,
    Expr,
    null,
    subtree,
    struct_pack_children,
)
from dataframe.lazy import LazyFrame, SELECT, SCAN_FRAME
from dataframe.nested_column import StructColumn
from .aggregates import _in_memory_sources
from .metadata import schema_frame
from .selectors import _live_nodes


@fieldwise_init
struct StructOutput(Copyable):
    var prototype: Series
    var fields: List[String]


@fieldwise_init
struct StructOutputs(Movable):
    var query: LazyFrame
    var schema: DataFrame
    var outputs: List[StructOutput]


def has_terminal_structs(query: LazyFrame) raises -> Bool:
    """Detect a struct result without initializing or importing a device SDK."""
    if not len(query._nodes):
        return False
    var live = _live_nodes(query)
    var candidate = False
    for index in range(len(query._nodes)):
        if not live[index]:
            continue
        ref operation = query._nodes[index]
        for expression in operation.exprs:
            for node in expression._nodes:
                candidate |= node.op == STRUCT_PACK or node.op == STRUCT_FIELD
        if operation.kind == SCAN_FRAME and 0 <= operation.offset < len(
            query._frames
        ):
            for column in query._frames[operation.offset]._columns:
                candidate |= column.dtype().is_struct()
    if not candidate:
        return False
    var schema = schema_frame(query, len(query._nodes) - 1)
    for column in schema._columns:
        if column.dtype().is_struct():
            return True
    return False


def _name(mut names: Dict[String, Bool], output: Int, field: Int) -> String:
    var name = "__dataframe_struct_" + String(output) + "_" + String(field)
    while name in names:
        name += "_"
    names[name] = True
    return name^


def lower_terminal_structs(query: LazyFrame) raises -> StructOutputs:
    """Flatten terminal as_struct fields; retain only schema restoration data.

    A field's GPU expression and logical dtype are unchanged. Scalar fields
    keep normal projection broadcasting. No source values are evaluated.
    """
    _in_memory_sources(query)
    if (
        not len(query._nodes)
        or query._nodes[len(query._nodes) - 1].kind != SELECT
    ):
        raise Error("Metal unsupported [struct]: terminal SELECT required")
    var root = len(query._nodes) - 1
    var expected = schema_frame(query, root).clear()
    var input = schema_frame(query, query._nodes[root].left)
    var live = _live_nodes(query)
    var names = Dict[String, Bool]()
    for frame in query._frames:
        for name in frame.columns():
            names[name] = True
    for operation in query._nodes:
        for expression in operation.exprs:
            names[expression._name] = True
            for node in expression._nodes:
                if node.op == COL:
                    names[node.text] = True
    for index in range(root):
        if not live[index]:
            continue
        for expression in query._nodes[index].exprs:
            for node in expression._nodes:
                if node.op == STRUCT_PACK or node.op == STRUCT_FIELD:
                    raise Error(
                        "Metal unsupported [struct]: intermediate struct consumers"
                    )
    var expressions = List[Expr]()
    var outputs = List[StructOutput]()
    var packed = False
    for index in range(len(query._nodes[root].exprs)):
        var bound = bind(query._nodes[root].exprs[index], input._columns)
        var top = len(bound.expr._nodes) - 1
        var prototype = expected._columns[index].copy()
        var fields = List[String]()
        if bound.dtypes[top].is_struct():
            if bound.expr._nodes[top].op != STRUCT_PACK:
                raise Error(
                    "Metal unsupported [struct]: nullable source struct transport"
                )
            packed = True
            for child in struct_pack_children(bound.expr._nodes[top]):
                if bound.dtypes[child].is_nested():
                    raise Error(
                        "Metal unsupported [struct]: nested child fields"
                    )
                var expression = subtree(bound.expr, child)
                for node in expression._nodes:
                    if node.op == STRUCT_PACK or node.op == STRUCT_FIELD:
                        raise Error(
                            "Metal unsupported [struct]: intermediate struct consumers"
                        )
                if (
                    len(expression._nodes) == 1
                    and expression._nodes[0].op == COL
                ):
                    for column in input._columns:
                        if (
                            column.name() == expression._nodes[0].text
                            and column.dtype().is_categorical()
                        ):
                            # A child extracted from its original multi-node
                            # expression must retain categorical->String
                            # binding rather than revert to a bare code column.
                            expression = expression.fill_null(
                                null(DataType.STRING)
                            )
                var name = _name(names, index, len(fields))
                fields.append(name)
                expressions.append(expression.alias(name))
        else:
            if bound.dtypes[top].is_nested():
                raise Error(
                    "Metal unsupported [struct]: nested output siblings"
                )
            for node in bound.expr._nodes:
                if node.op == STRUCT_PACK or node.op == STRUCT_FIELD:
                    raise Error(
                        "Metal unsupported [struct]: intermediate struct consumers"
                    )
            var name = _name(names, index, 0)
            fields.append(name)
            expressions.append(bound.expr.alias(name))
        outputs.append(StructOutput(prototype^, fields^))
    if not packed:
        raise Error("Metal unsupported [struct]: terminal as_struct required")
    var flattened = query.copy()
    flattened._nodes[root].exprs = expressions^
    # Validate field names and dtypes against metadata before allocating any
    # GPU buffers. Empty-source scalar identities keep their normal height.
    var schema = schema_frame(flattened, root)
    var at = 0
    for output in outputs:
        if output.prototype.dtype().is_struct():
            var fields = output.prototype.struct_column()
            if fields.field_count() != len(output.fields):
                raise Error("Metal struct output field count differs")
            for field in range(fields.field_count()):
                if schema._columns[at].dtype() != fields.field(field).dtype():
                    raise Error("Metal struct output field dtype differs")
                at += 1
        else:
            if schema._columns[at].dtype() != output.prototype.dtype():
                raise Error("Metal struct output dtype differs")
            at += 1
    return StructOutputs(flattened^, expected^, outputs^)


def assemble(flat: DataFrame, plan: StructOutputs) raises -> DataFrame:
    """Wrap exported fields without evaluating expressions or copying values."""
    var columns = List[Series](capacity=len(plan.outputs))
    var at = 0
    for output in plan.outputs:
        var column: Series
        if output.prototype.dtype().is_struct():
            var prototypes = output.prototype.struct_column()
            var fields = List[Series](capacity=prototypes.field_count())
            for field in range(prototypes.field_count()):
                var prototype = prototypes.field(field)
                if (
                    at >= flat.width()
                    or flat._columns[at].name() != output.fields[field]
                    or flat._columns[at].dtype() != prototype.dtype()
                ):
                    raise Error("Metal struct exported field schema differs")
                var value = flat._columns[at].renamed(prototype.name())
                value._field_metadata = prototype._field_metadata
                fields.append(value^)
                at += 1
            # STRUCT_PACK's parent has no null bitmap. Child validity belongs
            # solely to its field and must never be ANDed into parent validity.
            column = Series(output.prototype.name(), StructColumn(fields^))
        else:
            if (
                at >= flat.width()
                or flat._columns[at].name() != output.fields[0]
                or flat._columns[at].dtype() != output.prototype.dtype()
            ):
                raise Error("Metal struct exported sibling schema differs")
            column = flat._columns[at].renamed(output.prototype.name())
            at += 1
        column._field_metadata = output.prototype._field_metadata
        columns.append(column^)
    if at != flat.width():
        raise Error("Metal struct exported output width differs")
    return DataFrame(columns^, height=flat.height())
