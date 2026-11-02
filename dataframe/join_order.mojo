"""Choose an inner-join order using estimated cardinality and cost.

Relations form vertices in a graph; join keys and column comparisons form
edges. Each relation starts with its row count after local filters. To
estimate a connected set, divide the product of its relation row counts by
the product of edge domains on a spanning tree, choosing edges in descending
domain order. Columns equated by joins or filters share an equivalence set.
Without distinct-count statistics, an equivalence set's domain is the
smallest row count among its relations. With statistics, use the largest
distinct-count estimate among its columns. A cross-relation equality filter
is an edge that divides by its domain; another comparison divides by the
domain raised to 2/3. Additional edges after connecting the set each
contribute another domain factor. A plan's cost is the sum of estimated
output rows for its joins.

Semi, anti and left joins are operators of the graph (`JoinOperator`), as
in DuckDB 2.0's join order optimizer: each one's right input is a relation
of its own, and conflict rules from CD-C ("On the Correct and Complete
Enumeration of the Core Search Space", Moerkotte, Fender and Eich) say
which splits may apply it. A semi join can then run before inner joins
its keys do not read. A semi or anti join keeps a fifth of its left
input's rows (DuckDB's `DEFAULT_SEMI_ANTI_SELECTIVITY`); a left join keeps
at least its left input's.

Below `EXACT_RELATIONS`, dynamic programming tries every connected split
of each connected set. At or above the threshold, a greedy strategy
repeatedly merges the cheapest connected pair.
"""
from std.collections import Dict
from std.bit import pop_count

# DuckDB's THRESHOLD_TO_SWAP_TO_APPROXIMATE.
comptime EXACT_RELATIONS = 12
# DuckDB's DEFAULT_SEMI_ANTI_SELECTIVITY: a semi or anti join keeps one
# left row in five.
comptime SEMI_ANTI_SELECTIVITY = 5.0
# Operator kinds (`JoinOperator.kind`).
comptime OPERATOR_INNER = 0
comptime OPERATOR_LEFT = 1
comptime OPERATOR_SEMI = 2
comptime OPERATOR_ANTI = 3


struct JoinEdge(Copyable, Movable):
    """Relations `a` and `b` joined on `a_keys[i] = b_keys[i]`. `owner` is
    the operator (`JoinOperator`) that must apply these keys, or -1 for
    keys any split joining `a` and `b` may use."""

    var a: Int
    var b: Int
    var a_keys: List[String]
    var b_keys: List[String]
    var owner: Int

    def __init__(
        out self,
        a: Int,
        b: Int,
        var a_keys: List[String],
        var b_keys: List[String],
        owner: Int = -1,
    ):
        self.a = a
        self.b = b
        self.a_keys = a_keys^
        self.b_keys = b_keys^
        self.owner = owner


struct JoinOperator(Copyable, Movable):
    """A join of the written plan, as DuckDB's `JoinOrderOperator`: its
    kind, the relations under its left and right inputs, the relations its
    keys read (the syntactic eligibility set), and the operators under
    each input. `build_conflicts` fills in the conflict rules (`triggers`
    and `requirements`, pairwise), the total eligibility set's two sides
    and whether the join must be applied as itself rather than as free
    key pairs."""

    var kind: Int
    var left_relations: Int
    var right_relations: Int
    var syntactic: Int
    var left_operators: List[Int]
    var right_operators: List[Int]
    var triggers: List[Int]
    var requirements: List[Int]
    var left_total: Int
    var right_total: Int
    var must_apply: Bool

    def __init__(
        out self,
        kind: Int,
        left_relations: Int,
        right_relations: Int,
        syntactic: Int,
        var left_operators: List[Int],
        var right_operators: List[Int],
    ):
        self.kind = kind
        self.left_relations = left_relations
        self.right_relations = right_relations
        self.syntactic = syntactic
        self.left_operators = left_operators^
        self.right_operators = right_operators^
        self.triggers = List[Int]()
        self.requirements = List[Int]()
        self.left_total = syntactic & left_relations
        self.right_total = syntactic & right_relations
        self.must_apply = kind != OPERATOR_INNER

    def applicable(self, left: Int, right: Int) -> Bool:
        """Whether this join may join relation sets `left` and `right`
        (`IsApplicable`): each side holds its side of the total
        eligibility set (either way round for an inner join), and every
        conflict rule a side triggers has its requirement in the two."""
        var direct = (
            left & self.left_total == self.left_total
            and right & self.right_total == self.right_total
        )
        var inverted = (
            self.kind == OPERATOR_INNER
            and left & self.right_total == self.right_total
            and right & self.left_total == self.left_total
        )
        if not (direct or inverted):
            return False
        var both = left | right
        for i in range(len(self.triggers)):
            if (
                self.triggers[i] & both != 0
                and self.requirements[i] & both != self.requirements[i]
            ):
                return False
        return True

    def completed_by(self, relations: Int) -> Bool:
        """Whether `relations` holds this join's total eligibility set
        (`IsCompletedBy`)."""
        var total = self.left_total | self.right_total
        return relations & total == total


def _associative(
    left: JoinOperator, right: JoinOperator, shared_null_free: Bool
) -> Bool:
    """CD-C's assoc(left, right) for these kinds (Table 2): an inner join
    below is always associative; two semi joins are when the upper one's
    keys reject nulls of the shared input."""
    if left.kind == OPERATOR_INNER:
        return True
    if left.kind == OPERATOR_SEMI and right.kind == OPERATOR_SEMI:
        return shared_null_free
    return False


def _rule_requirement(input: Int, syntactic: Int) -> Int:
    """The input's relations the child's keys read, or the whole input
    when they read none of it (`PredicateIntersectionOrInput`)."""
    var both = input & syntactic
    return both if both != 0 else input


def _add_rule(mut op: JoinOperator, trigger: Int, requirement: Int):
    for i in range(len(op.triggers)):
        if op.triggers[i] == trigger and op.requirements[i] == requirement:
            return
    op.triggers.append(trigger)
    op.requirements.append(requirement)


def build_conflicts(mut ops: List[JoinOperator]):
    """Conflict rules and total eligibility sets for every operator
    (`JoinOrderConflictDetector::Build`). Equality keys never match
    nulls, so a join reading a shared input's column rejects its nulls."""
    for o in range(len(ops)):
        var op = ops[o].copy()
        var lefts = op.left_operators.copy()
        var rights = op.right_operators.copy()
        for c in lefts:
            ref child = ops[c]
            var shared = child.right_relations
            if not _associative(child, op, op.syntactic & shared != 0):
                _add_rule(
                    op,
                    child.right_relations,
                    _rule_requirement(child.left_relations, child.syntactic),
                )
            # Left associative-commutative holds for every pair of these
            # kinds (Table 3).
        for c in rights:
            ref child = ops[c]
            if not _associative(
                op, child, child.syntactic & child.left_relations != 0
            ):
                _add_rule(
                    op,
                    child.left_relations,
                    _rule_requirement(child.right_relations, child.syntactic),
                )
            if not (op.kind == OPERATOR_INNER and child.kind == OPERATOR_INNER):
                _add_rule(
                    op,
                    child.right_relations,
                    _rule_requirement(child.left_relations, child.syntactic),
                )
        # Decided before the rules simplify: a rule the eligibility set
        # absorbs still pins the join to its operator.
        op.must_apply = op.kind != OPERATOR_INNER or len(op.triggers) > 0
        # Rules a side of the total set triggers join their requirement
        # to it (`SimplifyConflictRules`).
        var total = op.syntactic
        var grew = True
        while grew:
            grew = False
            for i in range(len(op.triggers)):
                if (
                    op.triggers[i] & total != 0
                    and op.requirements[i] & total != op.requirements[i]
                ):
                    total |= op.requirements[i]
                    grew = True
        var triggers = List[Int]()
        var requirements = List[Int]()
        for i in range(len(op.triggers)):
            if op.requirements[i] & total != op.requirements[i]:
                triggers.append(op.triggers[i])
                requirements.append(op.requirements[i])
        op.triggers = triggers^
        op.requirements = requirements^
        op.left_total = total & op.left_relations
        op.right_total = total & op.right_relations
        ops[o] = op^


@fieldwise_init
struct FilterEdge(Copyable, Movable):
    """A cross-relation filter comparing `a_column` in relation `a` with
    `b_column` in relation `b`. It contributes an edge to the join graph
    for cardinality estimation but retains no join keys; the filter runs
    once its relations are joined."""

    var a: Int
    var b: Int
    var a_column: String
    var b_column: String
    var equal: Bool


@fieldwise_init
struct JoinTree(Copyable, Movable):
    """The chosen plan: entries 0 .. n-1 are the relations; each later
    entry joins entries `left[i]` and `right[i]` over relation set
    `mask[i]` with estimated `rows[i]`, by operator `operator[i]` (an
    index into the graph's operators, or -1 for free key pairs and for
    the relations). The last entry is the root."""

    var left: List[Int]
    var right: List[Int]
    var mask: List[Int]
    var rows: List[Float64]
    var operator: List[Int]


struct JoinGraph(Movable):
    var rows: List[Float64]
    var edges: List[JoinEdge]
    var filters: List[FilterEdge]
    # Each edge's total domain, then each filter's.
    var tdom: List[Float64]
    var estimates: Dict[Int, Float64]
    # Whether each edge's (then filter's) total domain is a distinct count
    # (DuckDB's `has_distinct_count_hll`) rather than table rows.
    var counted: List[Bool]
    var ops: List[JoinOperator]
    # The right relations of semi and anti joins, then of left joins.
    var semi_rights: Int
    var left_rights: Int
    # Left joins' right relations of no known rows: estimates leave them
    # out, as their joins keep every left row.
    var uncounted: Int
    var inner_estimates: Dict[Int, Float64]

    def __init__(
        out self,
        rows: List[Float64],
        domains: List[Float64],
        edges: List[JoinEdge],
        distinct: Dict[String, Float64] = Dict[String, Float64](),
        filters: List[FilterEdge] = List[FilterEdge](),
        ops: List[JoinOperator] = List[JoinOperator](),
        uncounted: Int = 0,
    ) raises:
        self.counted = List[Bool]()
        self.ops = ops.copy()
        self.uncounted = uncounted
        self.semi_rights = 0
        self.left_rights = 0
        for op in self.ops:
            if op.kind == OPERATOR_SEMI or op.kind == OPERATOR_ANTI:
                self.semi_rights |= op.right_relations
            elif op.kind == OPERATOR_LEFT:
                self.left_rights |= op.right_relations
        self.inner_estimates = Dict[Int, Float64]()
        self.rows = rows.copy()
        self.edges = edges.copy()
        self.filters = filters.copy()
        self.tdom = List[Float64]()
        self.estimates = Dict[Int, Float64]()
        # Equivalence sets of columns, by (relation, name), through each
        # key pair and filter (`InitEquivalentRelations`).
        var ids = Dict[String, Int]()
        var parent = List[Int]()

        def find(mut parent: List[Int], x: Int) -> Int:
            var r = x
            while parent[r] != r:
                r = parent[r]
            return r

        def id_of(
            mut ids: Dict[String, Int], mut parent: List[Int], key: String
        ) raises -> Int:
            if key in ids:
                return ids[key]
            var id = len(parent)
            ids[key] = id
            parent.append(id)
            return id

        # Every compared pair: the edges' key pairs, then the filters'.
        var compared = List[Tuple[Int, String, Int, String]]()
        for e in self.edges:
            for i in range(len(e.a_keys)):
                compared.append((e.a, e.a_keys[i], e.b, e.b_keys[i]))
        for f in self.filters:
            compared.append((f.a, f.a_column, f.b, f.b_column))
        for pair in compared:
            var x = id_of(ids, parent, String(pair[0]) + "/" + pair[1])
            var y = id_of(ids, parent, String(pair[2]) + "/" + pair[3])
            var rx = find(parent, x)
            var ry = find(parent, y)
            if rx != ry:
                parent[rx] = ry
        # A set's total domain (`UpdateTotalDomains`): the largest distinct
        # count among its columns where a table keeps statistics
        # (`distinct`, by "relation/column"), else the fewest table rows
        # (`domains`) of any relation holding one of its columns.
        var domain = Dict[Int, Float64]()
        var counted = Dict[Int, Float64]()
        for pair in compared:
            for side in range(2):
                var rel = pair[0] if side == 0 else pair[2]
                var name = pair[1] if side == 0 else pair[3]
                var key = String(rel) + "/" + name
                var root = find(parent, ids[key])
                if key in distinct:
                    var d = max(distinct[key], 1.0)
                    if root not in counted or d > counted[root]:
                        counted[root] = d
                var r = max(domains[rel], 1.0)
                if root not in domain or r < domain[root]:
                    domain[root] = r
        for root in counted.keys():
            domain[root] = counted[root]
        # An edge of several key pairs merges its relations once, at the
        # largest of its pairs' domains (the first the descending walk
        # meets); the others are left unused, which with no distinct-count
        # statistics adds no factor.
        for e in self.edges:
            var best = 1.0
            var from_count = False
            for i in range(len(e.a_keys)):
                var root = find(parent, ids[String(e.a) + "/" + e.a_keys[i]])
                if domain[root] > best:
                    best = domain[root]
                    from_count = root in counted
            self.tdom.append(best)
            self.counted.append(from_count)
        for f in self.filters:
            var root = find(parent, ids[String(f.a) + "/" + f.a_column])
            self.tdom.append(domain[root])
            self.counted.append(root in counted)

    def estimate(mut self, mask: Int) raises -> Float64:
        """Estimated rows of joining the relations in `mask`
        (`EstimateCardinalityWithSet`): the inner joins' estimate without
        the right relations of semi and anti joins, each of which keeps a
        fifth of the rows, and no fewer rows than without the right
        relations of left joins (`CalculateLeftJoinDenomInfo`)."""
        if mask in self.estimates:
            return self.estimates[mask]
        var base = mask & ~self.semi_rights & ~self.uncounted
        var result = self._inner_estimate(base)
        if base & self.left_rights != 0:
            result = max(result, self._inner_estimate(base & ~self.left_rights))
        var semis = Int(pop_count(mask & self.semi_rights))
        result = max(result / SEMI_ANTI_SELECTIVITY**semis, 1.0)
        self.estimates[mask] = result
        return result

    def _inner_estimate(mut self, mask: Int) raises -> Float64:
        """The estimate of `mask` with every edge an inner join's."""
        if mask in self.inner_estimates:
            return self.inner_estimates[mask]
        var numerator = 1.0
        var n = len(self.rows)
        for r in range(n):
            if mask & (1 << r) != 0:
                numerator *= max(self.rows[r], 1.0)
        # Edges and filters inside the set by descending total domain, each
        # merging two components divides once: an edge or an equality by
        # its domain, another comparison by the domain to the power 2/3.
        var order = List[Int]()
        var joins = len(self.edges)
        for k in range(joins + len(self.filters)):
            var a = self.edges[k].a if k < joins else self.filters[k - joins].a
            var b = self.edges[k].b if k < joins else self.filters[k - joins].b
            if mask & (1 << a) != 0 and mask & (1 << b) != 0:
                order.append(k)
        for i in range(1, len(order)):
            var j = i
            while j > 0 and self.tdom[order[j]] > self.tdom[order[j - 1]]:
                var t = order[j]
                order[j] = order[j - 1]
                order[j - 1] = t
                j -= 1
        var component = List[Int](length=n, fill=0)
        for r in range(n):
            component[r] = r
        var denominator = 1.0
        # Edges met once every relation is connected, by their distinct
        # domains: each adds one to the denominator's multiplier, DuckDB's
        # "slight penalty to cardinality for unused edges" (`GetDenominator`).
        var unused = List[Float64]()
        var parts = Int(pop_count(mask))
        for k in order:
            if parts == 1:
                if self.counted[k] and self.tdom[k] not in unused:
                    unused.append(self.tdom[k])
                continue
            var a = self.edges[k].a if k < joins else self.filters[k - joins].a
            var b = self.edges[k].b if k < joins else self.filters[k - joins].b
            var ca = component[a]
            var cb = component[b]
            if ca == cb:
                continue
            if k < joins or self.filters[k - joins].equal:
                denominator *= self.tdom[k]
            else:
                denominator *= self.tdom[k] ** (2.0 / 3.0)
            parts -= 1
            for r in range(n):
                if component[r] == cb:
                    component[r] = ca
        denominator *= 1.0 + Float64(len(unused))
        var result = max(numerator / denominator, 1.0)
        self.inner_estimates[mask] = result
        return result

    def connects(self, a: Int, b: Int) -> Bool:
        """Whether a free edge (no operator's) joins relation set `a` to
        relation set `b`."""
        for e in self.edges:
            if e.owner >= 0:
                continue
            if (a & (1 << e.a) != 0 and b & (1 << e.b) != 0) or (
                a & (1 << e.b) != 0 and b & (1 << e.a) != 0
            ):
                return True
        return False

    def candidate(self, a: Int, b: Int) -> Int:
        """How relation sets `a` and `b` may join: -1 by free edges, an
        operator's index when that operator joins them, -2 when they
        cannot (`IsJoinOrderCandidate`). An operator that must apply as
        itself and that the two complete together, but neither alone,
        is the only way to join them, and it must be applicable."""
        var both = a | b
        var chosen = -1
        for k in range(len(self.ops)):
            ref op = self.ops[k]
            if not op.must_apply:
                continue
            if (
                op.completed_by(a)
                or op.completed_by(b)
                or not op.completed_by(both)
            ):
                continue
            if chosen >= 0:
                return -2
            chosen = k
        var free = self.connects(a, b)
        if chosen < 0:
            return -1 if free else -2
        ref op = self.ops[chosen]
        if not (op.applicable(a, b) or op.applicable(b, a)):
            return -2
        # Free keys across a semi, anti or left join would have no join
        # to go to.
        if free and op.kind != OPERATOR_INNER:
            return -2
        return chosen


def _connected(graph: JoinGraph, mask: Int) -> Bool:
    var n = len(graph.rows)
    var first = -1
    for r in range(n):
        if mask & (1 << r) != 0:
            first = r
            break
    if first < 0:
        return False
    var reached = 1 << first
    var grew = True
    while grew:
        grew = False
        # An operator joins its whole total eligibility set at once.
        for op in graph.ops:
            var total = op.left_total | op.right_total
            if total & mask != total:
                continue
            if reached & total != 0 and reached & total != total:
                reached |= total
                grew = True
        for e in graph.edges:
            var a = 1 << e.a
            var b = 1 << e.b
            if mask & a == 0 or mask & b == 0:
                continue
            if reached & a != 0 and reached & b == 0:
                reached |= b
                grew = True
            elif reached & b != 0 and reached & a == 0:
                reached |= a
                grew = True
    return reached == mask


def _build(
    mut tree: JoinTree,
    mut entry: Dict[Int, Int],
    mut graph: JoinGraph,
    split: Dict[Int, Int],
    via: Dict[Int, Int],
    mask: Int,
) raises -> Int:
    """The tree entry joining relation set `mask` by its chosen splits."""
    if mask in entry:
        return entry[mask]
    var left_mask = split[mask]
    var left = _build(tree, entry, graph, split, via, left_mask)
    var right = _build(tree, entry, graph, split, via, mask ^ left_mask)
    tree.left.append(left)
    tree.right.append(right)
    tree.mask.append(mask)
    tree.rows.append(graph.estimate(mask))
    tree.operator.append(via[mask])
    entry[mask] = len(tree.mask) - 1
    return entry[mask]


def order_joins(
    rows: List[Float64],
    domains: List[Float64],
    edges: List[JoinEdge],
    distinct: Dict[String, Float64] = Dict[String, Float64](),
    filters: List[FilterEdge] = List[FilterEdge](),
    ops: List[JoinOperator] = List[JoinOperator](),
    uncounted: Int = 0,
) raises -> Optional[JoinTree]:
    """The cheapest join tree over these relations (`rows` after their own
    filters, `domains` their tables' rows before them) and operators
    (`build_conflicts` applied), or None when the edges leave them
    disconnected (a cross product, which is not reordered)."""
    var graph = JoinGraph(
        rows, domains, edges, distinct, filters, ops, uncounted
    )
    var n = len(rows)
    var full = (1 << n) - 1
    if n < 2 or not _connected(graph, full):
        return None
    var tree = JoinTree(
        List[Int](length=n, fill=-1),
        List[Int](length=n, fill=-1),
        List[Int](),
        List[Float64](),
        List[Int](length=n, fill=-1),
    )
    for r in range(n):
        tree.mask.append(1 << r)
        tree.rows.append(max(rows[r], 1.0))
    if n < EXACT_RELATIONS:
        # Every connected set from the smallest up: its cheapest split into
        # two connected, joined halves (`SolveJoinOrderExactly`).
        var cost = Dict[Int, Float64]()
        var split = Dict[Int, Int]()
        var via = Dict[Int, Int]()
        for r in range(n):
            cost[1 << r] = 0.0
        for size in range(2, n + 1):
            for mask in range(1, full + 1):
                if Int(pop_count(mask)) != size or not _connected(graph, mask):
                    continue
                var low = mask & -mask
                var best = -1.0
                var best_left = 0
                var best_via = -1
                # Submasks holding the set's lowest relation, so each split
                # is seen once.
                var sub = (mask - 1) & mask
                while sub > 0:
                    if sub & low != 0:
                        var other = mask ^ sub
                        if sub in cost and other in cost:
                            var how = graph.candidate(sub, other)
                            if how >= -1:
                                var c = (
                                    graph.estimate(mask)
                                    + cost[sub]
                                    + cost[other]
                                )
                                if best < 0 or c < best:
                                    best = c
                                    best_left = sub
                                    best_via = how
                    sub = (sub - 1) & mask
                if best >= 0:
                    cost[mask] = best
                    split[mask] = best_left
                    via[mask] = best_via
        if full not in split:
            return None
        # The tree, built from the splits down.
        var entry = Dict[Int, Int]()
        for r in range(n):
            entry[1 << r] = r

        _ = _build(tree, entry, graph, split, via, full)
        return tree^
    # Greedy: merge the cheapest connected pair of sets until one remains.
    var sets = List[Int]()
    var nodes = List[Int]()
    var costs = List[Float64]()
    for r in range(n):
        sets.append(1 << r)
        nodes.append(r)
        costs.append(0.0)
    while len(sets) > 1:
        var bi = -1
        var bj = -1
        var best = -1.0
        var best_via = -1
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                var how = graph.candidate(sets[i], sets[j])
                if how < -1:
                    continue
                var c = graph.estimate(sets[i] | sets[j]) + costs[i] + costs[j]
                if best < 0 or c < best:
                    best = c
                    bi = i
                    bj = j
                    best_via = how
        if bi < 0:
            return None
        var merged = sets[bi] | sets[bj]
        tree.left.append(nodes[bi])
        tree.right.append(nodes[bj])
        tree.mask.append(merged)
        tree.rows.append(graph.estimate(merged))
        tree.operator.append(best_via)
        sets[bi] = merged
        nodes[bi] = len(tree.mask) - 1
        costs[bi] = best
        _ = sets.pop(bj)
        _ = nodes.pop(bj)
        _ = costs.pop(bj)
    return tree^
