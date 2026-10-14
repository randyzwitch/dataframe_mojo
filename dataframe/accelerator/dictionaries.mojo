"""Query-wide ordered dictionary codes for accelerator string operations.

Code equality is string equality across all resident inputs. Code order is byte
lexicographic string order. Providers retain this owner until final decoding;
encoding and decoding are host staging work and belong in end-to-end costs.
"""
from .metadata import schema_frame
from .window_keys import WindowKey
from std.collections import Dict
from std.memory import ArcPointer
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
from dataframe.str_kernels import string_op, concat_strings
from dataframe.partition import _hash_bytes


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
        # Each captured key retains one immutable lookup table throughout
        # normalized DAG execution. Plan/task copies share that owner; row
        # lowering and the device separately charge their flattened copies.
        var lookup_bytes = checked_mul(len(self.dtype.dictionary()[]), 8)
        for operation in query._nodes:
            for key in operation.window_keys:
                if dictionary_dtype(key.dtype):
                    total = checked_add(total, lookup_bytes)
        return total

    def restoration_bytes(
        self, frame: DataFrame, schema: DataFrame
    ) raises -> Int:
        # Encoded outputs remain alive until restoration finishes. String
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

    def partition_values(self, key: WindowKey) raises -> List[UInt64]:
        """Map staged codes to CPU raw partition keys, without source rows.

        Unused values from other input dictionaries have no representative
        for this key. Their zero entries are never read by valid key rows.
        """
        var dictionary = self.dtype.dictionary()
        var values = List[UInt64](length=len(dictionary[]), fill=0)
        if key.string_hash:
            for code in range(len(dictionary[])):
                values[code] = _hash_bytes(dictionary[].get(code).as_bytes())
        elif key.dtype.is_categorical():
            var original = key.dtype.dictionary()
            for code in range(len(original[])):
                var staged = Int(self.code(String(original[].get(code))))
                values[staged] = UInt64(code)
        return values^

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


def _concat_pool(
    operand: Expr, original: LazyFrame, staged: LazyFrame, pool: List[String]
) raises -> List[String]:
    # A retained column's dictionary is a safe superset of its row values only
    # when no expression redefines that name. Derived/renamed join columns use
    # the whole query dictionary, which also includes previous lookup outputs.
    var values = List[String]()
    var seen = Dict[String, Bool]()
    for node in operand._nodes:
        if node.op == LIT_STRING:
            if node.text not in seen:
                seen[node.text] = True
                values.append(node.text)
        elif node.op == COL:
            var redefined = False
            for operation in original._nodes:
                for expression in operation.exprs:
                    if expression._name != node.text:
                        continue
                    ref root = expression._nodes[len(expression._nodes) - 1]
                    redefined |= root.op != COL or root.text != node.text
            var found = False
            if not redefined:
                for frame in staged._frames:
                    for column in frame._columns:
                        if column.name() != node.text or not dictionary_dtype(
                            column.dtype()
                        ):
                            continue
                        found = True
                        var dictionary = _values_of(column)
                        for i in range(len(dictionary)):
                            var value = String(dictionary.get(i))
                            if value not in seen:
                                seen[value] = True
                                values.append(value^)
            if not found:
                return pool.copy()
    return values^


def _concat_tables(
    left: List[String],
    right: List[String],
    separator: String,
    key_name: String,
    right_name: String,
    value_name: String,
    generated_bytes: Int,
    budget: Int,
) raises -> Tuple[Series, Series, Series]:
    # Check the Cartesian allocation before materializing even the key arrays.
    # The independent ceiling also protects callers with an unlimited payload
    # budget. This is dictionary staging, never row-wise CPU execution.
    var pairs = checked_mul(len(left), len(right))
    if pairs > 1_048_576:
        raise Error("Metal unsupported [dictionary]: concat pair capacity")
    var left_bytes = 0
    var right_bytes = 0
    for text in left:
        left_bytes = checked_add(left_bytes, text.byte_length())
    for text in right:
        right_bytes = checked_add(right_bytes, text.byte_length())
    var key_bytes = checked_add(
        checked_mul(left_bytes, len(right)),
        checked_mul(right_bytes, len(left)),
    )
    var text_bytes = checked_add(
        checked_mul(key_bytes, 2), checked_mul(pairs, separator.byte_length())
    )
    var reserve = checked_mul(
        checked_add(checked_mul(pairs, 128), text_bytes), 8
    )
    if reserve > 536_870_912:
        raise Error("Metal unsupported [dictionary]: concat staging capacity")
    if budget >= 0 and reserve > budget - generated_bytes:
        raise Error("Metal dictionary operation staging exceeds payload budget")
    var a = List[String]()
    var b = List[String]()
    for x in left:
        for y in right:
            a.append(x)
            b.append(y)
    var keys = Series(key_name, StringColumn(a^))
    var right_keys = Series(right_name, StringColumn(b^))
    var mapped = concat_strings(keys, right_keys, separator).renamed(value_name)
    return (keys^, right_keys^, mapped^)


def lower_dictionary_operations(
    query: LazyFrame, budget: Int = -1
) raises -> Tuple[LazyFrame, Int, Int]:
    """Apply String operations through unique-value GPU lookup joins.

    The host evaluates the shared String kernel once per dictionary value or concatenation pair,
    never per source row. Native left joins apply the mapping to resident row
    codes, preserve nulls and input order, and feed downstream GPU operators.
    New string values join the query-wide dictionary before any row encoding.
    """
    var needed = False
    for operation in query._nodes:
        for expression in operation.exprs:
            for node in expression._nodes:
                needed |= is_string_op(node.op) or node.op == STR_CONCAT
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
                has_operation |= is_string_op(node.op) or node.op == STR_CONCAT
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
                if not is_string_op(node.op) and node.op != STR_CONCAT:
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
                var right_name = key_name + "_rhs"
                while (
                    key_name in names
                    or value_name in names
                    or right_name in names
                ):
                    key_name += "_"
                    value_name += "_"
                    right_name += "_"
                names[key_name] = True
                names[value_name] = True
                names[right_name] = True
                var keys: Series
                var right_keys = Series(
                    right_name, StringColumn(List[String]())
                )
                var mapped = Series(value_name, StringColumn(List[String]()))
                if node.op == STR_CONCAT:
                    var left_pool = _concat_pool(
                        subtree(expression, node.left), query, result, pool
                    )
                    var right_pool = _concat_pool(
                        subtree(expression, node.right), query, result, pool
                    )
                    var tables = _concat_tables(
                        left_pool,
                        right_pool,
                        node.text,
                        key_name,
                        right_name,
                        value_name,
                        generated_bytes,
                        budget,
                    )
                    keys = tables[0].copy()
                    right_keys = tables[1].copy()
                    mapped = tables[2].copy()
                else:
                    keys = Series(key_name, StringColumn(pool.copy()))
                if node.op != STR_CONCAT and budget >= 0:
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
                if node.op != STR_CONCAT:
                    mapped = string_op(node, keys).renamed(value_name)
                else:
                    generated_bytes = checked_add(
                        generated_bytes, _dictionary_host_bytes(right_keys)
                    )
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
                var sources = List[Expr](
                    [subtree(expression, node.left).alias(key_name)]
                )
                var key_names = List[String]([key_name])
                if node.op == STR_CONCAT:
                    sources.append(
                        subtree(expression, node.right).alias(right_name)
                    )
                    key_names.append(right_name)
                result._nodes.append(
                    _plan_node(WITH_COLUMNS, operation.left, exprs=sources^)
                )
                var left = len(result._nodes) - 1
                var frame = len(result._frames)
                var columns = List[Series]([keys^, mapped^])
                if node.op == STR_CONCAT:
                    columns.append(right_keys^)
                result._frames.append(DataFrame(columns^))
                result._nodes.append(_plan_node(SCAN_FRAME, offset=frame))
                result._nodes.append(
                    _plan_node(
                        JOIN,
                        left,
                        len(result._nodes) - 1,
                        names=key_names,
                        names2=["_right"],
                        text="left",
                        how=JOIN_LEFT,
                        right_keys=key_names,
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
    # The captured dtype owns the original categorical dictionary, even
    # when the source is a metadata-only resident table. Populate lookups
    # before frame recoding erases its physical code provenance.
    for o in range(len(encoded._nodes)):
        ref operation = encoded._nodes[o]
        for k in range(len(operation.window_keys)):
            ref key = operation.window_keys[k]
            if dictionary_dtype(key.dtype):
                key.values = ArcPointer(dictionary.partition_values(key))
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
        frame = DataFrame(columns^, height=frame.height())
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
    return DataFrame(columns^, height=frame.height())
