"""Filters that keep each partition's first k rows by ordinal rank.

`col(v).rank("ordinal", descending=...).over(keys) <= k` keeps, in every
partition, the k rows whose (value, row) come first, the "top N per group"
idiom. Ranking every row sorts every partition and scatters the whole
column; here each partition only keeps its k best rows while the rows are
scanned once, and the filter's mask marks them. A row whose value is null
has a null rank and is not kept, as the comparison would have it.

Partitions are numbered as `over()` numbers them; workers split the
partition ids, and each keeps a small sorted list per partition of its
share. The order is `rank`'s own key (`rank._key`), so NaN and -0.0 rank
as they do there, and ties keep the earlier row, as ordinal rank does.
"""
from std.collections import Optional
from std.memory import Pointer

from .binding import BoundExpr
from .bool_column import BoolColumn
from .column import Column
from .dtype import NUMERIC_DTYPES
from .expr import COL, LE, LIT_INT, LT, OVER, RANK, SEP
from .hashing import RowKeys, encode_rows, encode_rows_parallel
from .parallel import Job, run_jobs, worker_count
from .partition import encode_partitioned, low_cardinality
from .rank import _key
from .series import Series


struct _TopKJob[D: DType](Job):
    """Keep the k best rows of partitions [lo, hi)."""

    var values: Int
    var valid: Int
    var nulls: Bool
    var ids: Int
    var n: Int
    var lo: Int
    var hi: Int
    var k: Int
    var descending: Bool
    var mask: Int

    def __init__(
        out self,
        values: Int,
        valid: Int,
        nulls: Bool,
        ids: Int,
        n: Int,
        lo: Int,
        hi: Int,
        k: Int,
        descending: Bool,
        mask: Int,
    ):
        self.values = values
        self.valid = valid
        self.nulls = nulls
        self.ids = ids
        self.n = n
        self.lo = lo
        self.hi = hi
        self.k = k
        self.descending = descending
        self.mask = mask

    def run(mut self) raises:
        ref column = Pointer[Column[Scalar[Self.D]], MutAnyOrigin](
            unsafe_from_address=self.values
        )[]
        var ids = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ids)
        var mask = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.mask)
        var width = self.hi - self.lo
        var k = self.k
        var keys = List[UInt64](length=width * k, fill=0)
        var rows = List[Int](length=width * k, fill=-1)
        var counts = List[Int](length=width, fill=0)
        var data = column._ptr()
        for i in range(self.n):
            var g = ids[unsafe_offset=i]
            if g < self.lo or g >= self.hi:
                continue
            if self.nulls and not column._valid(i):
                continue
            var key = _key[Self.D](data[unsafe_offset=i], self.descending)
            var base = (g - self.lo) * k
            var count = counts[g - self.lo]
            # Rows arrive in order, so an equal key is a later row and
            # ranks after: only a strictly smaller key displaces the worst.
            if count == k and key >= keys[base + k - 1]:
                continue
            var at = count if count < k else k - 1
            while at > 0 and keys[base + at - 1] > key:
                if at < k:
                    keys[base + at] = keys[base + at - 1]
                    rows[base + at] = rows[base + at - 1]
                at -= 1
            keys[base + at] = key
            rows[base + at] = i
            if count < k:
                counts[g - self.lo] = count + 1
        for p in range(width):
            for j in range(counts[p]):
                mask[unsafe_offset=rows[p * k + j]] = True


def _pattern(bound: BoundExpr) -> Tuple[Int, Int, Int]:
    """(column node, over node, k) of `rank(ordinal).over(...) <= k` (or
    `< k + 1`), or (-1, -1, 0)."""
    ref nodes = bound.expr._nodes
    var miss = (-1, -1, 0)
    var root = len(nodes) - 1
    if root < 0:
        return miss
    ref compare = nodes[root]
    if compare.op != LE and compare.op != LT:
        return miss
    if compare.left < 0 or compare.right < 0:
        return miss
    ref limit = nodes[compare.right]
    if limit.op != LIT_INT:
        return miss
    ref over = nodes[compare.left]
    if over.op != OVER or over.left < 0:
        return miss
    ref rank = nodes[over.left]
    if rank.op != RANK or rank.text != "ordinal" or rank.left < 0:
        return miss
    if nodes[rank.left].op != COL:
        return miss
    var k = Int(limit.integer) if compare.op == LE else Int(limit.integer) - 1
    return (rank.left, compare.left, k)


def top_k_mask(
    bound: BoundExpr, columns: List[Series], height: Int
) raises -> Optional[BoolColumn]:
    """The filter mask of a top-k-per-partition predicate, or None when
    the predicate has another shape."""
    var found = _pattern(bound)
    var column_node = found[0]
    if column_node < 0:
        return None
    ref nodes = bound.expr._nodes
    var k = found[2]
    var value = columns[bound.sources[column_node]].copy()
    var descending = nodes[nodes[found[1]].left].min_count == 1
    var keys = List[Series]()
    for name in nodes[found[1]].text2.split(SEP):
        var matched = False
        for column in columns:
            if column.name() == String(name):
                keys.append(column.copy())
                matched = True
        if not matched:
            return None
    var mask = List[Bool](length=height, fill=False)
    if k <= 0 or height == 0:
        return BoolColumn(mask^)
    var whole = value.rechunk() if value.is_chunked() else value.copy()
    var workers = worker_count(height)
    var partitions = encode_rows(keys, nulls_equal=True) if workers <= 1 else (
        encode_rows_parallel(keys, True, workers) if low_cardinality(
            keys
        ) else encode_partitioned(keys, workers, nulls_equal=True)
    )
    var count = partitions.count()
    var shares = max(1, min(workers, count))
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if whole._data.isa[Column[Scalar[D]]]():
            ref data = whole._data[Column[Scalar[D]]]
            var jobs = List[_TopKJob[D]](capacity=shares)
            for w in range(shares):
                jobs.append(
                    _TopKJob[D](
                        Int(Pointer(to=data)),
                        0,
                        data.null_count() > 0,
                        Int(partitions.ids.unsafe_ptr()),
                        height,
                        count * w // shares,
                        count * (w + 1) // shares,
                        k,
                        descending,
                        Int(mask.unsafe_ptr()),
                    )
                )
            run_jobs(jobs)
            _ = partitions^
            return BoolColumn(mask^)
    return None
