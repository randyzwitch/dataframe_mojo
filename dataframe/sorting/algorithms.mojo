"""Standard-library sorting over packed, normalized key records."""
from std.memory import Pointer, unsafe_memcmp
from std.sys import size_of

# A key record held in registers: up to four 64-bit words (the widest
# layout, `PAYLOAD_VARIABLE_32` or `PAYLOAD_FIXED_32`, is 32 bytes).
comptime Rec = SIMD[DType.uint64, 4]


struct SortKeys[W: Int, VAR: Bool](ImplicitlyCopyable, Movable):
    """Key records at `base`, `R` words each (sort_key.hpp's SortKey with
    a payload). Fixed: `W` key words, then the payload (the row id).
    Variable (`VAR`): the key's first 8 bytes, its byte size, the address
    of the whole key, then the payload. Key words are stored so that
    unsigned word order is the key bytes' order."""

    comptime R = 4 if Self.VAR else Self.W + 1
    var base: Int

    def __init__(out self, base: Int):
        self.base = base

    @always_inline
    def addr(self, i: Int) -> Int:
        return self.base + i * (Self.R * 8)

    @always_inline
    def get(self, i: Int) -> Rec:
        var p = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.addr(i))
        var r = Rec(0)
        comptime for k in range(Self.R):
            r[k] = p[unsafe_offset=k]
        return r

    @always_inline
    def put(self, i: Int, r: Rec):
        var p = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.addr(i))
        comptime for k in range(Self.R):
            p[unsafe_offset=k] = r[k]

    @always_inline
    def part0(self, i: Int) -> UInt64:
        return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.addr(i))[]

    @always_inline
    def payload(self, i: Int) -> Int:
        return Int(
            Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.addr(i))[
                unsafe_offset=Self.R - 1
            ]
        )

    @staticmethod
    @always_inline
    def less(a: Rec, b: Rec) -> Bool:
        """`operator<` of the key (the payload never takes part)."""
        comptime if Self.VAR:
            # VariableSortKey: the inlined bytes first; when they are
            # equal and either key is no longer than them, the shorter one
            # is less; otherwise the whole keys.
            if a[0] != b[0]:
                return a[0] < b[0]
            if a[1] <= 8 or b[1] <= 8:
                return a[1] < b[1]
            var n = min(a[1], b[1])
            var c = unsafe_memcmp(
                Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=Int(a[2])),
                Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=Int(b[2])),
                Int(n),
            )
            return c < 0 or (c == 0 and a[1] < b[1])
        else:
            comptime for k in range(Self.W):
                if a[k] != b[k]:
                    return a[k] < b[k]
            return False

    @always_inline
    def less_at(self, i: Int, j: Int) -> Bool:
        return Self.less(self.get(i), self.get(j))


def sort_keys[W: Int, VAR: Bool](keys: SortKeys[W, VAR], begin: Int, end: Int):
    """Sort a packed record span without copying or widening its storage.

    Key encoding already includes null placement, direction, and the row
    tie-breaker when stable order is requested. The payload never compares.
    """
    if end - begin <= 1:
        return
    comptime K = SortKeys[W, VAR]
    comptime Record = Array[UInt64, K.R]
    comptime assert size_of[Record]() == K.R * size_of[UInt64]()
    var records = Span(
        unsafe_ptr=Pointer[Record, MutAnyOrigin](
            unsafe_from_address=keys.addr(begin)
        ),
        length=end - begin,
    )

    @always_inline
    def less(a: Record, b: Record) -> Bool:
        var left = Rec(0)
        var right = Rec(0)
        comptime for i in range(K.R):
            left[i] = a[i]
            right[i] = b[i]
        return K.less(left, right)

    sort(records, less)
