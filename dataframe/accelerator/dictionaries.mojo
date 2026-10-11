"""Query-wide ordered dictionary codes for accelerator string operations.

Code equality is string equality across all resident inputs. Code order is byte
lexicographic string order. Providers retain this owner until final decoding;
encoding and decoding are host staging work and belong in end-to-end costs.
"""
from std.collections import Dict
from dataframe.categorical import _values_of, decode, recode
from dataframe.dtype import CategoricalDictionary, DataType
from dataframe.expr import LIT_STRING, LIT_INT, CAST, LIT_NULL
from dataframe.frame import DataFrame
from std.collections import Optional
from .memory import checked_add, checked_mul
from dataframe.lazy import LazyFrame
from dataframe.series import Series


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

    def restoration_bytes(self, frame: DataFrame, schema: DataFrame) raises -> Int:
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
                total = checked_add(total, checked_mul(len(self.dtype.dictionary()[]), 16))
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
    return False


def encode_query(query: LazyFrame, dictionary: DictionaryCodes) raises -> LazyFrame:
    """Stage ordered UInt32 codes while retaining the original plan structure."""
    var encoded = query.copy()
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
                    raise Error("Accelerator dictionary plans do not support casts")
                elif node.op == LIT_NULL:
                    for d in range(len(node.dtypes)):
                        if node.dtypes[d] and dictionary_dtype(node.dtypes[d].value()):
                            node.dtypes[d] = Optional[DataType](DataType.UINT32)
    return encoded^


def restore_frame(
    frame: DataFrame, schema: DataFrame, dictionary: DictionaryCodes
) raises -> DataFrame:
    var columns = List[Series]()
    for i in range(len(frame._columns)):
        columns.append(dictionary.restore(frame._columns[i], schema._columns[i].dtype()))
    return DataFrame(columns^)
