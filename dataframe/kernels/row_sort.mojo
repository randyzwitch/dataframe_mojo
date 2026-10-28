"""Stable row sorting with bounded prefixes and full string tie comparison.

Only rows are sorted. Numeric and decimal keys retain direct encodings;
long strings compare their remaining bytes before moving to the next key.
"""
from std.ffi import external_call
from std.memory import Pointer
from dataframe.series import Series
from dataframe.storage.string import StringColumn
from dataframe.dtype import DataType
from dataframe.kernels.row_encode import (
    encodable,
    encode_sort_keys,
    STRING_PREFIX_BYTES,
)
from dataframe.kernels.comparison_sort import KeyOrder, comparison_arg_sort
from dataframe.parallel import Job, run_jobs, partitions, worker_count
from dataframe.kernels.packed_sort import selection_ranges


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
        _row_layout(columns, self.columns, self.fallback, has_nulls)
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
                var tail = _long_tail_order(
                    self.columns[k], self.descending[k], left_word, a, b
                )
                if tail != 0:
                    return tail < 0
        return a < b


def _row_layout(
    columns: List[Series],
    mut contiguous: List[Series],
    mut fallback: List[Int],
    mut has_nulls: List[Bool],
) raises:
    """The row-major word layout of `encode_sort_keys(prefix_only=True)`.

    Appends each key rechunked to `contiguous`, whether it holds nulls to
    `has_nulls`, and one entry per word to `fallback`: the key's index on a
    string's capped length word, whose long values still need their
    remaining bytes compared, else -1.
    """
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
            fallback.append(
                k if w == count - 1
                and column.dtype().physical() == DataType.STRING else -1
            )
        contiguous.append(column^)


@always_inline
def _long_tail_order(
    column: Series, descending: Bool, length_word: Int, a: Int, b: Int
) -> Int:
    """The order of rows `a` and `b` past the string prefix, once their
    words so far are equal: negative when `a` comes first, positive when
    `b` does, 0 when the tail does not separate them."""
    var long_length = (
        ~(STRING_PREFIX_BYTES + 1) if descending else STRING_PREFIX_BYTES + 1
    )
    # Equal capped length words identify two valid long values. Their first
    # 24 bytes already compared equal; compare the remaining bytes once,
    # including embedded NULs.
    if length_word != long_length:
        return 0
    ref strings = column._data[StringColumn]
    var left = strings._row_bytes(a)
    var right = strings._row_bytes(b)
    var count = min(len(left), len(right)) - STRING_PREFIX_BYTES
    var compared = Int(
        external_call["memcmp", Int32](
            Int(left.unsafe_ptr()) + STRING_PREFIX_BYTES,
            Int(right.unsafe_ptr()) + STRING_PREFIX_BYTES,
            count,
        )
    )
    if compared == 0:
        compared = len(left) - len(right)
    return -compared if descending else compared


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


# Rows a top-k worker encodes at a time: the words of one batch stay in
# cache while its rows are compared against the heap.
comptime _TOP_ROWS_BATCH = 2048


struct _TopRowsJob(Job):
    """The first `k` rows of one range under `_RowOrder`'s order.

    DuckDB's top-N keeps one heap per thread (physical_top_n.cpp) and
    Polars selects over row-encoded keys (`_arg_bottom_k`); neither ranks
    whole columns. Here each batch of rows is encoded to sort words once,
    and a row joins the bounded max-heap only when it comes before the
    worst row held, so the heap holds k rows of words whatever the range's
    length. A row is compared straight from its batch and copied only when
    it is kept.
    """

    var columns: List[Series]
    var descending: List[Bool]
    var nulls_last: List[Bool]
    var known_nulls: List[Bool]
    var fallback: List[Int]
    # Whether any key can hold a string past the prefix; when none can, a
    # compare is the word loop alone.
    var long: Bool
    var start: Int
    var length: Int
    var k: Int
    var rows: List[Int]

    def __init__(
        out self,
        columns: List[Series],
        descending: List[Bool],
        nulls_last: List[Bool],
        known_nulls: List[Bool],
        fallback: List[Int],
        start: Int,
        length: Int,
        k: Int,
    ):
        self.columns = columns.copy()
        self.descending = descending.copy()
        self.nulls_last = nulls_last.copy()
        self.known_nulls = known_nulls.copy()
        self.fallback = fallback.copy()
        self.long = False
        for key in fallback:
            self.long = self.long or key >= 0
        self.start = start
        self.length = length
        self.k = k
        self.rows = List[Int]()

    @always_inline
    def _tail_less(self, w: Int, word: Int, a: Int, b: Int) -> Int:
        """`_long_tail_order` for word `w` when it is a string's length
        word, else 0."""
        var k = self.fallback[w]
        if k < 0:
            return 0
        return _long_tail_order(self.columns[k], self.descending[k], word, a, b)

    @always_inline
    def _less(
        self,
        words: Pointer[Int, MutAnyOrigin],
        owner: Pointer[Int, MutAnyOrigin],
        x: Int,
        y: Int,
    ) -> Bool:
        """Whether the row in slot `x` comes before the row in slot `y`."""
        var width = len(self.fallback)
        var a = owner[unsafe_offset=x]
        var b = owner[unsafe_offset=y]
        for w in range(width):
            var left = words[unsafe_offset=x * width + w]
            var right = words[unsafe_offset=y * width + w]
            if left != right:
                return left < right
            if self.long:
                var tail = self._tail_less(w, left, a, b)
                if tail != 0:
                    return tail < 0
        return a < b

    @always_inline
    def _before(
        self,
        encoded: Pointer[Int, MutAnyOrigin],
        i: Int,
        row: Int,
        words: Pointer[Int, MutAnyOrigin],
        owner: Pointer[Int, MutAnyOrigin],
        slot: Int,
    ) -> Bool:
        """Whether batch row `i` (input row `row`) comes before the row in
        `slot`, read from the batch's words so a losing row is never
        copied. `encoded` holds the batch's word list addresses."""
        var width = len(self.fallback)
        var b = owner[unsafe_offset=slot]
        for w in range(width):
            var left = Pointer[Int, MutAnyOrigin](
                unsafe_from_address=encoded[unsafe_offset=w]
            )[unsafe_offset=i]
            var right = words[unsafe_offset=slot * width + w]
            if left != right:
                return left < right
            if self.long:
                var tail = self._tail_less(w, left, row, b)
                if tail != 0:
                    return tail < 0
        return row < b

    def run(mut self) raises:
        var k = min(self.k, self.length)
        if k == 0:
            return
        var width = len(self.fallback)
        # Slot s holds one row's words at s * width and the row in owner[s].
        # The heap orders slots worst first; the spare slot takes the row
        # being considered. Reads go through pointers: these loops are the
        # whole cost, and checked indexing was most of it.
        var word_list = List[Int](length=(k + 1) * width, fill=0)
        var owner_list = List[Int](length=k + 1, fill=0)
        var heap_list = List[Int](length=k, fill=0)
        var words = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=Int(word_list.unsafe_ptr())
        )
        var owner = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=Int(owner_list.unsafe_ptr())
        )
        var heap = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=Int(heap_list.unsafe_ptr())
        )
        var held = 0
        var spare = k
        var addresses = List[Int](length=width, fill=0)
        var end = self.start + self.length
        for batch in range(self.start, end, _TOP_ROWS_BATCH):
            var count = min(_TOP_ROWS_BATCH, end - batch)
            var encoded = encode_sort_keys(
                self.columns,
                self.descending,
                self.nulls_last,
                prefix_only=True,
                offset=batch,
                length=count,
                known_nulls=self.known_nulls,
            )
            for w in range(width):
                addresses[w] = Int(encoded[w].unsafe_ptr())
            var columns = Pointer[Int, MutAnyOrigin](
                unsafe_from_address=Int(addresses.unsafe_ptr())
            )
            for i in range(count):
                var full = held == k
                if full and not self._before(
                    columns, i, batch + i, words, owner, heap[unsafe_offset=0]
                ):
                    continue
                var slot = spare if full else held
                for w in range(width):
                    words[unsafe_offset=slot * width + w] = Pointer[
                        Int, MutAnyOrigin
                    ](unsafe_from_address=columns[unsafe_offset=w])[
                        unsafe_offset=i
                    ]
                owner[unsafe_offset=slot] = batch + i
                if not full:
                    heap[unsafe_offset=held] = slot
                    held += 1
                    var child = held - 1
                    while child > 0:
                        var parent = (child - 1) // 2
                        if not self._less(
                            words,
                            owner,
                            heap[unsafe_offset=parent],
                            heap[unsafe_offset=child],
                        ):
                            break
                        var moved = heap[unsafe_offset=parent]
                        heap[unsafe_offset=parent] = heap[unsafe_offset=child]
                        heap[unsafe_offset=child] = moved
                        child = parent
                    continue
                spare = heap[unsafe_offset=0]
                heap[unsafe_offset=0] = slot
                var parent = 0
                while True:
                    var largest = parent
                    var left = 2 * parent + 1
                    var right = left + 1
                    if left < k and self._less(
                        words,
                        owner,
                        heap[unsafe_offset=largest],
                        heap[unsafe_offset=left],
                    ):
                        largest = left
                    if right < k and self._less(
                        words,
                        owner,
                        heap[unsafe_offset=largest],
                        heap[unsafe_offset=right],
                    ):
                        largest = right
                    if largest == parent:
                        break
                    var moved = heap[unsafe_offset=parent]
                    heap[unsafe_offset=parent] = heap[unsafe_offset=largest]
                    heap[unsafe_offset=largest] = moved
                    parent = largest
            # The words were read through their addresses only.
            _ = encoded^
        for h in range(held):
            self.rows.append(owner[unsafe_offset=heap[unsafe_offset=h]])
        _ = word_list^
        _ = owner_list^
        _ = heap_list^
        _ = addresses^


def row_top_rows(
    columns: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    k: Int,
    threads: Int = 0,
) raises -> List[Int]:
    """The first `k` rows of `row_arg_sort`'s order, in order, without
    sorting every row.

    Each range keeps its own first `k` rows (`_TopRowsJob`); a row among
    the overall first `k` is among its range's, so sorting the candidates
    finishes the selection. Candidates are taken in row order, so ties keep
    the stable order. `threads` caps the workers (0: the configured count),
    for callers that already run on a worker.
    """
    if len(columns) == 0:
        raise Error("sort requires at least one column")
    if len(descending) != len(columns) or len(nulls_last) != len(columns):
        raise Error(
            "descending and nulls_last must have one entry per sort column"
        )
    if k < 0:
        raise Error("k must be nonnegative")
    var contiguous = List[Series]()
    var fallback = List[Int]()
    var has_nulls = List[Bool]()
    _row_layout(columns, contiguous, fallback, has_nulls)
    # A string key whose values all fit in the prefix never needs its tail
    # compared; one pass over the lengths spares every compare the check.
    for w in range(len(fallback)):
        if fallback[w] >= 0 and encodable(contiguous[fallback[w]]):
            fallback[w] = -1
    var n = len(contiguous[0])
    var take = min(k, n)
    if take == 0:
        return List[Int]()
    var ranges = selection_ranges(n, threads)
    var jobs = List[_TopRowsJob](capacity=ranges)
    for r in range(ranges):
        var start = n * r // ranges
        jobs.append(
            _TopRowsJob(
                contiguous,
                descending,
                nulls_last,
                has_nulls,
                fallback,
                start,
                n * (r + 1) // ranges - start,
                take,
            )
        )
    if len(jobs) == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    var candidates = List[Int]()
    for r in range(len(jobs)):
        candidates.extend(Span(jobs[r].rows))
    sort(candidates)
    var gathered = List[Series](capacity=len(contiguous))
    for column in contiguous:
        gathered.append(column.take(candidates))
    var order = row_arg_sort(gathered, descending, nulls_last)
    var rows = List[Int](capacity=take)
    for i in range(take):
        rows.append(candidates[order[i]])
    return rows^
