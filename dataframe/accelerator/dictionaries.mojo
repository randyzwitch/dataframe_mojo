"""Query-wide ordered dictionary codes for accelerator string operations.

Code equality is string equality across all resident inputs. Code order is byte
lexicographic string order. Providers retain this owner until final decoding;
encoding and decoding are host staging work and belong in end-to-end costs.
"""
from .metadata import schema_frame
from std.collections import Dict
from dataframe.categorical import _values_of, decode, recode
from dataframe.dtype import CategoricalDictionary, DataType
from dataframe.expr import (
    LIT_STRING,
    LIT_INT,
    CAST,
    LIT_NULL,
    COL,
    SELECTOR,
    Expr,
    _node,
    col,
    is_string_op,
    subtree,
    STR_PAD,
    STR_REPLACE,
    DT_STRPTIME,
    STR_CONCAT,
)
from dataframe.binding import bind, ROWS
from dataframe.frame import DataFrame
from std.collections import Optional
from .memory import checked_add, checked_mul
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    _plan_node,
    SCAN_FRAME,
    SELECT,
    WITH_COLUMNS,
    JOIN,
)
from dataframe.join_type import JOIN_LEFT
from dataframe.series import Series
from dataframe.string_column import StringColumn
from dataframe.str_kernels import string_op


def dictionary_dtype(dtype: DataType) -> Bool:
    return dtype == DataType.STRING or dtype.is_categorical()


struct DictionaryCodes(Movable):
    var dtype: DataType
    var lookup: Dict[String, UInt32]
    var bytes: Int

    def __init__(out self, query: LazyFrame) raises:
        var seen = Dict[String, Bool]()
        var values = List[String]()
        for frame in query._frames:
            for column in frame._columns:
                if not dictionary_dtype(column.dtype()):
                    continue
                # Include unused categorical members too: output metadata and
                # literal comparisons must not depend on which rows are valid.
                var dictionary = _values_of(column)
                for code in range(len(dictionary)):
                    var text = String(dictionary.get(code))
                    if text not in seen:
                        seen[text] = True
                        values.append(text^)
        for operation in query._nodes:
            for expr in operation.exprs:
                for node in expr._nodes:
                    if node.op == LIT_STRING and node.text not in seen:
                        seen[node.text] = True
                        values.append(node.text)
        sort(values)
        if len(values) > Int(UInt32.MAX):
            raise Error("Accelerator dictionary exceeds UInt32 code capacity")
        var dictionary = CategoricalDictionary()
        self.lookup = Dict[String, UInt32]()
        for code in range(len(values)):
            dictionary.append(values[code])
            self.lookup[values[code]] = UInt32(code)
        self.bytes = len(dictionary.bytes) + 8 * len(dictionary.offsets)
        self.dtype = DataType.categorical(dictionary^)

    def staging_bytes(self, query: LazyFrame) raises -> Int:
        var total = self.bytes
        for frame in query._frames:
            for column in frame._columns:
                if dictionary_dtype(column.dtype()):
                    total = checked_add(total, checked_mul(len(column), 4))
                    total = checked_add(total, checked_add(len(column), 7) // 8)
        return total

    def restoration_bytes(
        self, frame: DataFrame, schema: DataFrame
    ) raises -> Int:
        # Native code outputs remain alive until restoration finishes. String
        # views own a copy of dictionary text; categorical recoding also holds
        # those views while building the original dictionary's result codes.
        var total = 0
        var rows = frame.height()
        var bitmap = checked_add(rows, 7) // 8
        for i in range(frame.width()):
            var physical = frame._columns[i].dtype().physical()
            var width = max(1, physical.bit_width() // 8)
            total = checked_add(total, checked_mul(rows, width))
            total = checked_add(total, bitmap)
            var dtype = schema._columns[i].dtype()
            if dictionary_dtype(dtype):
                total = checked_add(total, checked_mul(rows, 24))
                total = checked_add(total, checked_mul(self.bytes, 2))
                total = checked_add(
                    total, checked_mul(len(self.dtype.dictionary()[]), 16)
                )
        return total

    def code(self, text: String) raises -> UInt32:
        return self.lookup[text]

    def encode(self, column: Series) raises -> Series:
        if not dictionary_dtype(column.dtype()):
            return column.copy()
        return recode(column, self.dtype)

    def restore(self, codes: Series, dtype: DataType) raises -> Series:
        if not dictionary_dtype(dtype):
            return codes.copy()
        var categorical = codes.with_dtype(self.dtype)
        if dtype == DataType.STRING:
            return decode(categorical)
        # The query dictionary can contain values from other inputs that are
        # absent from this output's original dictionary. Recode only actual
        # output values, rather than every unused query dictionary member.
        return recode(decode(categorical), dtype)


def has_dictionary_columns(query: LazyFrame) -> Bool:
    for frame in query._frames:
        for column in frame._columns:
            if dictionary_dtype(column.dtype()):
                return True
    for operation in query._nodes:
        for expression in operation.exprs:
            for node in expression._nodes:
                if node.op == LIT_STRING:
                    return True
                if node.op == LIT_NULL:
                    for dtype in node.dtypes:
                        if dtype and dictionary_dtype(dtype.value()):
                            return True
    return False


def _dictionary_host_bytes(column: Series) raises -> Int:
    # Lookup tables are created during collection, unlike original retained
    # CPU inputs. Charge their host words/views/offsets and text explicitly.
    var rows = len(column)
    var total = checked_add(checked_mul(rows, 32), (rows + 7) // 8)
    if column.dtype() == DataType.STRING:
        total = checked_add(total, column._data[StringColumn]._value_bytes())
    return total


def lower_dictionary_operations(
    query: LazyFrame, budget: Int = -1
) raises -> Tuple[LazyFrame, Int, Int]:
    """Apply unary String operations through unique-value GPU lookup joins.

    The host evaluates the shared String kernel once per dictionary value,
    never per source row. Native left joins apply the mapping to resident row
    codes, preserve nulls and input order, and feed downstream GPU operators.
    New string values join the query-wide dictionary before any row encoding.
    """
    var needed = False
    for operation in query._nodes:
        for expression in operation.exprs:
            for node in expression._nodes:
                needed |= is_string_op(node.op)
    if not needed:
        return (query.copy(), 0, 0)
    var seen = Dict[String, Bool]()
    var pool = List[String]()
    var names = Dict[String, Bool]()
    for frame in query._frames:
        for column in frame._columns:
            names[column.name()] = True
            if dictionary_dtype(column.dtype()):
                var values = _values_of(column)
                for i in range(len(values)):
                    var value = String(values.get(i))
                    if value not in seen:
                        seen[value] = True
                        pool.append(value^)
    for operation in query._nodes:
        for name in operation.names:
            names[name] = True
        for expression in operation.exprs:
            names[expression._name] = True
            for node in expression._nodes:
                if node.op == COL:
                    names[node.text] = True
                if node.op == LIT_STRING and node.text not in seen:
                    seen[node.text] = True
                    pool.append(node.text)
    var result = query.copy()
    result._nodes = List[PlanNode]()
    var roots = List[Int]()
    var generated_bytes = 0
    var lookups = 0
    for index in range(len(query._nodes)):
        var operation = query._nodes[index].copy()
        if operation.left >= 0:
            operation.left = roots[operation.left]
        if operation.right >= 0:
            operation.right = roots[operation.right]
        var changed = False
        for e in range(len(operation.exprs)):
            var expression = operation.exprs[e].copy()
            var has_operation = False
            for node in expression._nodes:
                has_operation |= is_string_op(node.op)
            if not has_operation:
                continue
            # A selector can expand to several expressions. Do not let hidden
            # mapping columns change that expansion; keep this case rejected.
            for node in expression._nodes:
                if node.op == SELECTOR:
                    raise Error(
                        "Metal unsupported [dictionary]: String operations with"
                        " selectors"
                    )
            var original_input = schema_frame(query, query._nodes[index].left)
            for name in original_input.columns():
                names[name] = True
            var bound = bind(expression, original_input._columns)
            expression = bound.expr.copy()
            for n in range(len(expression._nodes)):
                var node = expression._nodes[n].copy()
                if not is_string_op(node.op):
                    continue
                if bound.shapes[n] != ROWS:
                    raise Error(
                        "Metal unsupported [dictionary]: scalar String"
                        " operation"
                    )
                var key_name = "__dataframe_metal_dictionary_key_" + String(
                    lookups
                )
                var value_name = "__dataframe_metal_dictionary_value_" + String(
                    lookups
                )
                while key_name in names or value_name in names:
                    key_name += "_"
                    value_name += "_"
                names[key_name] = True
                names[value_name] = True
                var keys = Series(key_name, StringColumn(pool.copy()))
                if budget >= 0:
                    var reserve = checked_mul(_dictionary_host_bytes(keys), 8)
                    if node.op == STR_PAD:
                        reserve = checked_add(
                            reserve,
                            checked_mul(
                                checked_mul(len(pool), max(0, node.min_count)),
                                4,
                            ),
                        )
                    elif node.op == STR_REPLACE:
                        reserve = checked_mul(
                            reserve, max(1, node.text2.byte_length())
                        )
                    if reserve > budget - generated_bytes:
                        raise Error(
                            "Metal dictionary operation staging exceeds payload"
                            " budget"
                        )
                var mapped = string_op(node, keys).renamed(value_name)
                generated_bytes = checked_add(
                    generated_bytes,
                    checked_add(
                        _dictionary_host_bytes(keys),
                        _dictionary_host_bytes(mapped),
                    ),
                )
                if budget >= 0 and generated_bytes > budget:
                    raise Error(
                        "Metal dictionary operation staging exceeds payload"
                        " budget"
                    )
                if mapped.dtype() == DataType.STRING:
                    var values = _values_of(mapped)
                    for i in range(len(values)):
                        var value = String(values.get(i))
                        if value not in seen:
                            seen[value] = True
                            pool.append(value^)
                var source = subtree(expression, node.left).alias(key_name)
                result._nodes.append(
                    _plan_node(WITH_COLUMNS, operation.left, exprs=[source^])
                )
                var left = len(result._nodes) - 1
                var frame = len(result._frames)
                result._frames.append(DataFrame([keys^, mapped^]))
                result._nodes.append(_plan_node(SCAN_FRAME, offset=frame))
                result._nodes.append(
                    _plan_node(
                        JOIN,
                        left,
                        len(result._nodes) - 1,
                        names=[key_name],
                        names2=["_right"],
                        text="left",
                        how=JOIN_LEFT,
                        right_keys=[key_name],
                    )
                )
                operation.left = len(result._nodes) - 1
                expression._nodes[n] = _node(COL, text=value_name)
                lookups += 1
                changed = True
            operation.exprs[e] = expression^
        result._nodes.append(operation^)
        if changed:
            # Filter/with-columns preserve their input columns. Restore exactly
            # the original visible schema and column order after the operation.
            var schema = schema_frame(query, index)
            var projected = List[Expr]()
            for name in schema.columns():
                projected.append(col(name))
            result._nodes.append(
                _plan_node(SELECT, len(result._nodes) - 1, exprs=projected)
            )
        roots.append(len(result._nodes) - 1)
    return (result^, generated_bytes, lookups)


def encode_query(
    query: LazyFrame, dictionary: DictionaryCodes
) raises -> LazyFrame:
    """Stage ordered UInt32 codes while retaining the original plan structure."""
    var encoded = query.copy()
    # Reject operations that still require text before replacing text operands
    # with integer codes. Binding encoded text would otherwise report a bogus
    # dtype error instead of the provider's actual capability boundary.
    for operation in query._nodes:
        for expression in operation.exprs:
            for node in expression._nodes:
                if (
                    node.op == DT_STRPTIME
                    or node.op == STR_CONCAT
                    or is_string_op(node.op)
                ):
                    raise Error(
                        "Metal unsupported [dictionary]: unlowered text"
                        " operation"
                    )
    for f in range(len(encoded._frames)):
        ref frame = encoded._frames[f]
        var columns = List[Series]()
        for column in frame._columns:
            var codes = dictionary.encode(column)
            if dictionary_dtype(column.dtype()):
                codes = codes.with_dtype(DataType.UINT32)
            columns.append(codes^)
        frame = DataFrame(columns^)
    for o in range(len(encoded._nodes)):
        ref operation = encoded._nodes[o]
        for e in range(len(operation.exprs)):
            ref expression = operation.exprs[e]
            for n in range(len(expression._nodes)):
                ref node = expression._nodes[n]
                if node.op == LIT_STRING:
                    node.op = LIT_INT
                    node.integer = Int64(dictionary.code(node.text))
                    node.text = "uint32"
                    node.dtypes = [Optional[DataType](DataType.UINT32)]
                elif node.op == CAST:
                    # Numeric-to-string formatting and string parsing are not
                    # dictionary operations and cannot run on numeric codes.
                    var schema = schema_frame(query, operation.left)
                    var bound = bind(query._nodes[o].exprs[e], schema._columns)
                    if dictionary_dtype(bound.dtypes[n]) or dictionary_dtype(
                        bound.dtypes[node.left]
                    ):
                        raise Error(
                            "Accelerator dictionary plans do not support"
                            " String/categorical casts"
                        )
                elif node.op == LIT_NULL:
                    for d in range(len(node.dtypes)):
                        if node.dtypes[d] and dictionary_dtype(
                            node.dtypes[d].value()
                        ):
                            node.dtypes[d] = Optional[DataType](DataType.UINT32)
    return encoded^


def restore_frame(
    frame: DataFrame, schema: DataFrame, dictionary: DictionaryCodes
) raises -> DataFrame:
    var columns = List[Series]()
    for i in range(len(frame._columns)):
        columns.append(
            dictionary.restore(frame._columns[i], schema._columns[i].dtype())
        )
    return DataFrame(columns^)
