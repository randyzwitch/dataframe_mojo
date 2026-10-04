"""Stable row sorting with bounded prefixes and full string tie comparison.

Only rows are sorted. Numeric and decimal keys retain direct encodings;
long strings compare their remaining bytes before moving to the next key.
"""
from std.ffi import external_call
from std.memory import Pointer
from .series import Series
from .string_column import StringColumn
from .dtype import DataType
from .row_encode import encode_sort_keys, STRING_PREFIX_BYTES
from .comparison_sort import KeyOrder, comparison_arg_sort
from .parallel import Job, run_jobs, partitions, worker_count


struct _RowOrder(KeyOrder):
    var first: List[Int]
    var rows: List[Int]
    var width: Int
    var fallback: List[Int]
    var columns: List[Series]
    var descending: List[Bool]

    def __init__(
        out self,
        columns: List[Series],
        descending: List[Bool],
        nulls_last: List[Bool],
    ) raises:
        self.first = List[Int]()
        self.fallback = List[Int]()
        self.columns = List[Series]()
        self.descending = descending.copy()
        var has_nulls = List[Bool]()
        for k in range(len(columns)):
            var column = columns[k].rechunk()
            if column.dtype().is_categorical():
                raise Error("categorical sort keys must be normalized first")
            var count = 1
            if column.dtype().physical() == DataType.STRING:
                count = STRING_PREFIX_BYTES // 8 + 1
            elif column.dtype().is_decimal():
                count = 2
            var nullable = column.null_count() > 0
            has_nulls.append(nullable)
            if nullable or column.dtype().physical() in [
                DataType.FLOAT32,
                DataType.FLOAT64,
            ]:
                count += 1
            for w in range(count):
                self.fallback.append(
                    k if w == count - 1
                    and column.dtype().physical() == DataType.STRING else -1
                )
            self.columns.append(column^)
        self.width = len(self.fallback)
        var n = len(columns[0])
        self.first = List[Int](unsafe_uninit_length=n)
        self.rows = List[Int](unsafe_uninit_length=n * self.width)
        var bounds = partitions(n, worker_count(n), 1)
        var jobs = List[_EncodeSortJob]()
        for w in range(len(bounds) - 1):
            if bounds[w + 1] > bounds[w]:
                jobs.append(
                    _EncodeSortJob(
                        self.columns,
                        descending,
                        nulls_last,
                        has_nulls,
                        Int(self.rows.unsafe_ptr()),
                        Int(self.first.unsafe_ptr()),
                        bounds[w],
                        bounds[w + 1] - bounds[w],
                    )
                )
        run_jobs(jobs)

    @always_inline
    def less(self, a: Int, b: Int) -> Bool:
        var values = self.rows.unsafe_ptr()
        var left_row = a * self.width
        var right_row = b * self.width
        for w in range(self.width):
            var left_word = values[unsafe_offset=left_row + w]
            var right_word = values[unsafe_offset=right_row + w]
            if left_word != right_word:
                return left_word < right_word
            var k = self.fallback[w]
            if k >= 0:
                var long_length = (
                    ~(STRING_PREFIX_BYTES + 1) if self.descending[
                        k
                    ] else STRING_PREFIX_BYTES
                    + 1
                )
                # Equal capped length words identify two valid long values.
                # Their first 24 bytes already compared equal; compare the
                # remaining bytes once, including embedded NULs.
                if left_word == long_length:
                    ref strings = self.columns[k]._data[StringColumn]
                    var left = strings._row_bytes(a)
                    var right = strings._row_bytes(b)
                    var count = min(len(left), len(right)) - STRING_PREFIX_BYTES
                    var compared = external_call["memcmp", Int32](
                        Int(left.unsafe_ptr()) + STRING_PREFIX_BYTES,
                        Int(right.unsafe_ptr()) + STRING_PREFIX_BYTES,
                        count,
                    )
                    if compared != 0:
                        return (
                            compared > 0 if self.descending[k] else compared < 0
                        )
                    if len(left) != len(right):
                        return (
                            len(left)
                            > len(right) if self.descending[k] else len(left)
                            < len(right)
                        )
        return a < b


def row_arg_sort(
    columns: List[Series], descending: List[Bool], nulls_last: List[Bool]
) raises -> List[Int]:
    if len(columns) == 0:
        raise Error("sort requires at least one column")
    if len(descending) != len(columns) or len(nulls_last) != len(columns):
        raise Error(
            "descending and nulls_last must have one entry per sort column"
        )
    var ranks = _RowOrder(columns, descending, nulls_last)
    var first = ranks.first^
    ranks.first = List[Int]()
    return comparison_arg_sort(ranks^, len(columns[0]), first)


struct _EncodeSortJob(Job):
    var columns: List[Series]
    var descending: List[Bool]
    var nulls_last: List[Bool]
    var known_nulls: List[Bool]
    var output: Int
    var first: Int
    var start: Int
    var length: Int

    def __init__(
        out self,
        columns: List[Series],
        descending: List[Bool],
        nulls_last: List[Bool],
        known_nulls: List[Bool],
        output: Int,
        first: Int,
        start: Int,
        length: Int,
    ):
        self.columns = columns.copy()
        self.descending = descending.copy()
        self.nulls_last = nulls_last.copy()
        self.known_nulls = known_nulls.copy()
        self.output = output
        self.first = first
        self.start = start
        self.length = length

    def run(mut self) raises:
        var target = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.output)
        var leading = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.first)
        # Bound temporary word-major encoding to a cache-sized batch per
        # worker. Only the final row-major keys grow with input height.
        for start in range(self.start, self.start + self.length, 2048):
            var length = min(2048, self.start + self.length - start)
            var words = encode_sort_keys(
                self.columns,
                self.descending,
                self.nulls_last,
                prefix_only=True,
                offset=start,
                length=length,
                known_nulls=self.known_nulls,
            )
            var width = len(words)
            for row in range(length):
                leading[unsafe_offset=start + row] = words[0][row]
                for w in range(width):
                    target[unsafe_offset=(start + row) * width + w] = words[w][
                        row
                    ]
