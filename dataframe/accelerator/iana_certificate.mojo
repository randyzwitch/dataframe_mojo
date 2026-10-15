"""Explicit CPU branch-mask scopes for cloned expression occurrences.

A scope records a predicate occurrence, its parent scope and chosen sense.
Column and literal validity survive masks. Replacement ignores its own mask,
while its cast/arithmetic operands still receive the scope. Fault observation
is consequently a separate phase from selecting the WHEN output.
"""
from dataframe.binding import BoundExpr, SCALAR
from dataframe.expr import (
    WHEN,
    ADD,
    SUB,
    MUL,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    CLIP_LOW,
    CLIP_HIGH,
    GT,
    LT,
    GE,
    LE,
    EQ,
    NE,
    NEG,
    ABS,
    SQRT,
    EXP,
    LOG,
    FLOOR,
    CEIL,
    ROUND,
    AND,
    OR,
    XOR,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    FILL_NULL,
    FILL_NAN,
    KEEP_NULLS,
)
from dataframe import DataType
from . import ordered_rows


@fieldwise_init
struct Certificate(Movable):
    var bound: BoundExpr
    var ranks: List[UInt32]
    var words: List[UInt64]


def kernel_ignores_mask(bound: BoundExpr, at: Int) -> Bool:
    """Own-kernel validity policy from CPU expr_kernels dispatch.

    Operands still receive the occurrence's incoming mask. This is separate
    from retained whole-subtree Float64 fusion and from scalar mask fitting.
    """
    ref node = bound.expr._nodes[at]
    if node.left < 0:
        return False
    if (
        node.op == AND
        or node.op == OR
        or node.op == XOR
        or node.op == NOT
        or node.op == IS_NULL
        or node.op == IS_NOT_NULL
        or node.op == IS_NAN
        or node.op == IS_NOT_NAN
        or node.op == IS_FINITE
        or node.op == IS_INFINITE
        or node.op == FILL_NULL
        or node.op == FILL_NAN
        or node.op == KEEP_NULLS
    ):
        return True
    var input = bound.dtypes[node.left]
    var comparison = (
        node.op == GT
        or node.op == LT
        or node.op == GE
        or node.op == LE
        or node.op == EQ
        or node.op == NE
    )
    if input.is_float() and (
        comparison
        or node.op == ADD
        or node.op == SUB
        or node.op == MUL
        or node.op == DIV
        or node.op == FLOORDIV
        or node.op == MOD
        or node.op == POW
        or node.op == CLIP_LOW
        or node.op == CLIP_HIGH
        or node.op == NEG
        or node.op == ABS
        or node.op == SQRT
        or node.op == EXP
        or node.op == LOG
        or node.op == FLOOR
        or node.op == CEIL
        or node.op == ROUND
    ):
        return True
    return input == DataType.BOOL and comparison


def prepare(bound: BoundExpr, max_nodes: Int = 128) raises -> Certificate:
    var cloned = ordered_rows.occurrences(bound, max_nodes)
    var occurrences = cloned[0].copy()
    var ranks = cloned[1].copy()
    var scopes = List[UInt64]()
    # Scope zero is unconditional. Negative predicate marks its terminator.
    scopes.extend([UInt64.MAX, UInt64(0), UInt64(0)])
    var assignments = List[UInt64](length=len(occurrences.expr._nodes), fill=0)
    var pending = List[Tuple[Int, Int]]()
    pending.append((len(occurrences.expr._nodes) - 1, 0))
    while len(pending):
        var visit = pending.pop()
        var index = visit[0]
        var scope = visit[1]
        assignments[index] = UInt64(scope)
        ref node = occurrences.expr._nodes[index]
        if node.op == WHEN:
            pending.append((node.left, scope))
            var then_scope = len(scopes) // 3
            scopes.extend([UInt64(node.left), UInt64(scope), UInt64(1)])
            pending.append((node.right, then_scope))
            if node.extra >= 0:
                var other_scope = len(scopes) // 3
                scopes.extend([UInt64(node.left), UInt64(scope), UInt64(0)])
                pending.append((node.extra, other_scope))
        else:
            for child in [node.left, node.right, node.extra]:
                if child >= 0:
                    pending.append((child, scope))
    var words = List[UInt64]()
    words.extend([UInt64(len(assignments)), UInt64(len(scopes) // 3)])
    words.extend(scopes^)
    words.extend(assignments^)
    # CPU execution returns a fused nonleaf directly, without its branch mask.
    # Retain that exact original-bound policy per occurrence; do not infer it
    # from physical storage or globally exempt arithmetic from masking.
    var scalar_slots = 0
    for at in range(len(occurrences.expr._nodes)):
        var nonleaf = occurrences.expr._nodes[at].left >= 0
        var fused = occurrences.fusible[at] and nonleaf
        var policy = UInt64(fused)
        if nonleaf and not fused and occurrences.shapes[at] == SCALAR:
            scalar_slots += 1
            policy |= UInt64(scalar_slots) << 1
        words.append(policy)
    # CPU nonfused kernels can preserve validity without preserving masks on
    # their operands. Keep this distinct from both bit0 fusion and upper slots.
    for at in range(len(occurrences.expr._nodes)):
        words.append(UInt64(kernel_ignores_mask(occurrences, at)))
    return Certificate(occurrences^, ranks^, words^)


def validate(words: List[UInt64], base: Int, nodes: Int) raises:
    if nodes < 1 or nodes > 128:
        raise Error("IANA mask certificate node count is invalid")
    if base < 0 or base > len(words) or len(words) - base < 5:
        raise Error("IANA mask certificate header is out of range")
    if words[base] != UInt64(nodes) or words[base + 1] > UInt64(257):
        raise Error("IANA mask certificate geometry is invalid")
    var scopes = Int(words[base + 1])
    var end = base + 2 + 3 * scopes
    if not scopes or end > len(words) or 3 * nodes > len(words) - end:
        raise Error("IANA mask certificate plane is out of range")
    if words[base + 2] != UInt64.MAX or words[base + 3] or words[base + 4]:
        raise Error("IANA mask certificate root scope is invalid")
    for scope in range(1, scopes):
        var at = base + 2 + 3 * scope
        if (
            words[at] >= UInt64(nodes)
            or words[at + 1] >= UInt64(scope)
            or words[at + 2] > 1
        ):
            raise Error("IANA mask certificate scope is invalid")
    var scalar_slots = 0
    for node in range(nodes):
        if words[end + 2 * nodes + node] > 1:
            raise Error("IANA CPU kernel mask policy flag is invalid")
        var policy = words[end + nodes + node]
        var slot = policy >> 1
        if slot:
            scalar_slots += 1
            if (policy & 1) or slot != UInt64(scalar_slots):
                raise Error("IANA scalar mask slot policy is invalid")
        var scope_word = words[end + node]
        if scope_word >= UInt64(scopes):
            raise Error("IANA mask certificate assignment is invalid")
        var scope = Int(scope_word)
        while scope:
            var at = base + 2 + 3 * scope
            if words[at] >= UInt64(node):
                raise Error(
                    "IANA mask predicate must precede its branch occurrence"
                )
            scope = Int(words[at + 1])


def mask_words(bound: BoundExpr, max_nodes: Int = 128) raises -> List[UInt64]:
    """Consume the certificate owner before its plane enters an outer loop."""
    var certificate = prepare(bound, max_nodes)
    return certificate.words.copy()


def scalar_count(words: List[UInt64], base: Int, nodes: Int) raises -> Int:
    """Read validated dense scalar slots without constructing metadata owners."""
    validate(words, base, nodes)
    var policies = base + 2 + 3 * Int(words[base + 1]) + nodes
    var count = 0
    for node in range(nodes):
        count = max(count, Int(words[policies + node] >> 1))
    return count
