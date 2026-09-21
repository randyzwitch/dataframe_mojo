"""Order-preserving keys for sorting, one Int per key word.

Sorting needs only to compare rows, not to know a value's rank among the
distinct values. `Series._sort_ranks` computes the stronger thing: it sorts
each key column to assign dense ranks, so an n-column sort does n sorts
before the one that orders the rows. That is where most of a sort went --
107 ms of a 147 ms two-key sort at 1M rows.

This computes the weaker, cheaper thing. Every fixed-width value maps to an
Int whose signed order is the value's order, in one linear pass and no
sort at all. The result has the same shape `sort_indices` already takes --
one Int per row per word, compared lexicographically -- so the parallel
sort itself is unchanged.

The mapping is Polars' row encoding (`crates/polars-row`), with one change:
Polars encodes to a byte string and compares it with `memcmp`, which is
vectorised in Rust. Comparing bytes one at a time here cost 1,000 ms for
1M rows, so keys go into whole 64-bit words instead -- big-endian byte
order and word order agree, so a word compare gives byte-string order at
one comparison per eight bytes.

Per key:

- Signed integers and temporals are already ordered as Int64.
- Unsigned integers narrower than 64 bits widen into Int64 unchanged;
  UInt64 has its top bit toggled, since values at or above 2**63 would
  otherwise be negative.
- Floats use the IEEE-754 total order (flip every bit of a negative, the
  sign bit of a positive), which also puts NaN above every number, the
  order sorting already documents. Getting this half right -- the total
  order without the sign toggle -- sorts every positive below every
  negative, so the tests below check signs explicitly.
- Booleans are 0 and 1.
- `descending` inverts the bits rather than negating, because negating
  Int64.MIN overflows while inversion is total and order-reversing.
- A column with nulls, or any float column, gets a rank word *before* its
  value word, so placement is decided before the value is compared: 0 for
  an ordinary value, 1 for NaN, and 2 or -1 for a null according to
  `nulls_last`. NaN needs its own tier because sorting here documents
  "NaN ranks after the numbers in either direction", where the IEEE total
  order would flip NaN to the front of a descending sort. The rank word is
  never inverted, so both NaN and null placement survive `descending`.
  Columns that can hold neither skip the word entirely.
"""
from std.memory import bitcast

from .bool_column import BoolColumn
from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .series import Series
from .string_column import StringColumn


def encodable(dtype: DataType) -> Bool:
    """Whether `dtype` has a fixed-width order-preserving Int encoding.

    Strings do not: a variable-length key needs an escaped byte encoding to
    keep a short string from being a prefix of a longer one, so string keys
    still take the dense-rank path.
    """
    return dtype != DataType.STRING


def _float_order[D: DType](value: Scalar[D]) -> Int:
    """IEEE-754 total order as an Int whose signed order is float order.

    The value is canonicalised first, as Polars' encoder does: sorting
    treats -0.0 as equal to 0.0 and every NaN as equal to every other, but
    their bit patterns differ, so encoding the raw bits would order -0.0
    below 0.0. The reference comparator in test_sort.mojo caught exactly
    that.
    """
    comptime if D == DType.float32:
        var f = rebind[Float32](value)
        if f != f:
            # One quiet NaN stands for all of them.
            f = bitcast[DType.float32](Int32(0x7FC0_0000))
        elif f == 0:
            f = 0  # collapses -0.0
        var s = bitcast[DType.int32](f)
        return Int(
            s ^ ((s >> 31).cast[DType.uint32]() >> 1).cast[DType.int32]()
        )
    var d = rebind[Float64](value)
    if d != d:
        d = bitcast[DType.float64](Int64(0x7FF8_0000_0000_0000))
    elif d == 0:
        d = 0
    var s = bitcast[DType.int64](d)
    return Int(s ^ ((s >> 63).cast[DType.uint64]() >> 1).cast[DType.int64]())


def _value_order[D: DType](value: Scalar[D]) -> Int:
    """The value as an Int ordered exactly as the value is."""
    comptime if D.is_floating_point():
        return _float_order[D](value)
    comptime if D == DType.uint64:
        # Toggle the top bit: unsigned order then matches signed order.
        return Int(
            bitcast[DType.int64](
                rebind[UInt64](value) ^ UInt64(0x8000_0000_0000_0000)
            )
        )
    return Int(value)


def encode_sort_keys(
    columns: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
) raises -> List[List[Int]]:
    """Order-preserving words for `columns`, lexicographic across the list.

    Every column must satisfy `encodable`. The result is what
    `sort_indices` compares: `result[w][row]`, word-major.
    """
    if len(columns) == 0:
        raise Error("sort requires at least one column")
    var rows = len(columns[0])
    var words = List[List[Int]]()
    for k in range(len(columns)):
        ref column = columns[k]
        var flip = descending[k]
        var nulls = column.null_count() > 0
        var floating = column.dtype().physical() in (
            DataType.FLOAT64,
            DataType.FLOAT32,
        )
        var null_rank = 2 if nulls_last[k] else -1
        var ranked = List[Int](length=rows, fill=0)
        var values = List[Int](length=rows, fill=0)

        var filled = False
        comptime for t in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[t]
            if column._data.isa[Column[Scalar[D]]]():
                ref typed = column._data[Column[Scalar[D]]]
                for i in range(rows):
                    if nulls and not typed._valid(i):
                        ranked[i] = null_rank
                        continue
                    var value = typed._get(i)
                    comptime if D.is_floating_point():
                        if value != value:
                            ranked[i] = 1
                            continue
                    var order = _value_order[D](value)
                    values[i] = ~order if flip else order
                filled = True
        if not filled:
            if not column._data.isa[BoolColumn]():
                raise Error("row encoding requires a fixed-width dtype")
            ref typed = column._data[BoolColumn]
            for i in range(rows):
                if nulls and not typed._valid(i):
                    ranked[i] = null_rank
                    continue
                var order = Int(typed._get(i))
                values[i] = ~order if flip else order

        if nulls or floating:
            words.append(ranked^)
        words.append(values^)
    return words^
