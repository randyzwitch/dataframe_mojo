"""Join order by estimated cost, after DuckDB's join order optimizer
(src/optimizer/join_order, v1.5.5).

Inner joins form a graph: relations (each with its row count after its own
filters) and edges between two relations (the key pairs that join them).
A set of relations has an estimated row count (`CardinalityEstimator`):
the product of its relations' rows over the product of the total domains
of the edges on a spanning tree of the set, taken in descending total
domain (Kruskal). An edge's total domain is that of the equivalence set of
columns its keys belong to (columns made equal by any join are one set),
which with no distinct-count statistics (DuckDB's case for Parquet) is the
smallest table row count among the set's relations: a column's distinct
count is taken as its table's rows before any filter
(`RelationStatisticsHelper::ExtractGetStats`), so joining to a filtered
table divides by the whole table and keeps its share of the rows. A plan's cost is the sum of
the estimated rows of every join it makes (`CostModel::ComputeCost`).

Below `EXACT_RELATIONS` relations every connected split of every connected
set is tried (dynamic programming over subsets, as DPhyp enumerates them);
at or above it the cheapest connected pair is merged until one set
remains (greedy operator ordering, `SolveJoinOrderApproximately`).
"""
from std.collections import Dict
from std.bit import pop_count

# DuckDB's THRESHOLD_TO_SWAP_TO_APPROXIMATE.
comptime EXACT_RELATIONS = 12


@fieldwise_init
struct JoinEdge(Copyable, Movable):
    """Relations `a` and `b` joined on `a_keys[i] = b_keys[i]`."""

    var a: Int
    var b: Int
    var a_keys: List[String]
    var b_keys: List[String]


@fieldwise_init
struct JoinTree(Copyable, Movable):
    """The chosen plan: entries 0 .. n-1 are the relations; each later
    entry joins entries `left[i]` and `right[i]` over relation set
    `mask[i]` with estimated `rows[i]`. The last entry is the root."""

    var left: List[Int]
    var right: List[Int]
    var mask: List[Int]
    var rows: List[Float64]


struct JoinGraph(Movable):
    var rows: List[Float64]
    var edges: List[JoinEdge]
    # Each edge's total domain.
    var tdom: List[Float64]
    var estimates: Dict[Int, Float64]

    def __init__(
        out self,
        rows: List[Float64],
        domains: List[Float64],
        edges: List[JoinEdge],
    ) raises:
        self.rows = rows.copy()
        self.edges = edges.copy()
        self.tdom = List[Float64]()
        self.estimates = Dict[Int, Float64]()
        # Equivalence sets of columns, by (relation, name), through each
        # key pair (`InitEquivalentRelations`).
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

        var pairs = List[Tuple[Int, Int]]()
        for e in self.edges:
            for i in range(len(e.a_keys)):
                var x = id_of(ids, parent, String(e.a) + "/" + e.a_keys[i])
                var y = id_of(ids, parent, String(e.b) + "/" + e.b_keys[i])
                pairs.append((x, y))
                var rx = find(parent, x)
                var ry = find(parent, y)
                if rx != ry:
                    parent[rx] = ry
        # A set's total domain: with no distinct counts, the fewest table
        # rows (`domains`) of any relation holding one of its columns
        # (`UpdateTotalDomains`).
        var domain = Dict[Int, Float64]()
        for e in self.edges:
            for i in range(len(e.a_keys)):
                for side in range(2):
                    var rel = e.a if side == 0 else e.b
                    var name = e.a_keys[i] if side == 0 else e.b_keys[i]
                    var root = find(parent, ids[String(rel) + "/" + name])
                    var r = max(domains[rel], 1.0)
                    if root not in domain or r < domain[root]:
                        domain[root] = r
        # An edge of several key pairs merges its relations once, at the
        # largest of its pairs' domains (the first the descending walk
        # meets); the others are left unused, which with no distinct-count
        # statistics adds no factor.
        for e in self.edges:
            var best = 1.0
            for i in range(len(e.a_keys)):
                var root = find(parent, ids[String(e.a) + "/" + e.a_keys[i]])
                best = max(best, domain[root])
            self.tdom.append(best)

    def estimate(mut self, mask: Int) raises -> Float64:
        """Estimated rows of joining the relations in `mask`
        (`EstimateCardinalityWithSet`)."""
        if mask in self.estimates:
            return self.estimates[mask]
        var numerator = 1.0
        var n = len(self.rows)
        for r in range(n):
            if mask & (1 << r) != 0:
                numerator *= max(self.rows[r], 1.0)
        # Edges inside the set by descending total domain, each merging two
        # components divides once by its domain.
        var order = List[Int]()
        for k in range(len(self.edges)):
            ref e = self.edges[k]
            if mask & (1 << e.a) != 0 and mask & (1 << e.b) != 0:
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
        for k in order:
            ref e = self.edges[k]
            var ca = component[e.a]
            var cb = component[e.b]
            if ca == cb:
                continue
            denominator *= self.tdom[k]
            for r in range(n):
                if component[r] == cb:
                    component[r] = ca
        var result = max(numerator / denominator, 1.0)
        self.estimates[mask] = result
        return result

    def connects(self, a: Int, b: Int) -> Bool:
        """Whether an edge joins relation set `a` to relation set `b`."""
        for e in self.edges:
            if (a & (1 << e.a) != 0 and b & (1 << e.b) != 0) or (
                a & (1 << e.b) != 0 and b & (1 << e.a) != 0
            ):
                return True
        return False


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
    mask: Int,
) raises -> Int:
    """The tree entry joining relation set `mask` by its chosen splits."""
    if mask in entry:
        return entry[mask]
    var left_mask = split[mask]
    var left = _build(tree, entry, graph, split, left_mask)
    var right = _build(tree, entry, graph, split, mask ^ left_mask)
    tree.left.append(left)
    tree.right.append(right)
    tree.mask.append(mask)
    tree.rows.append(graph.estimate(mask))
    entry[mask] = len(tree.mask) - 1
    return entry[mask]


def order_joins(
    rows: List[Float64], domains: List[Float64], edges: List[JoinEdge]
) raises -> Optional[JoinTree]:
    """The cheapest join tree over these relations (`rows` after their own
    filters, `domains` their tables' rows before them), or None when the
    edges leave them disconnected (a cross product, which is not
    reordered)."""
    var graph = JoinGraph(rows, domains, edges)
    var n = len(rows)
    var full = (1 << n) - 1
    if n < 2 or not _connected(graph, full):
        return None
    var tree = JoinTree(
        List[Int](length=n, fill=-1),
        List[Int](length=n, fill=-1),
        List[Int](),
        List[Float64](),
    )
    for r in range(n):
        tree.mask.append(1 << r)
        tree.rows.append(max(rows[r], 1.0))
    if n < EXACT_RELATIONS:
        # Every connected set from the smallest up: its cheapest split into
        # two connected, joined halves (`SolveJoinOrderExactly`).
        var cost = Dict[Int, Float64]()
        var split = Dict[Int, Int]()
        for r in range(n):
            cost[1 << r] = 0.0
        for size in range(2, n + 1):
            for mask in range(1, full + 1):
                if Int(pop_count(mask)) != size or not _connected(graph, mask):
                    continue
                var low = mask & -mask
                var best = -1.0
                var best_left = 0
                # Submasks holding the set's lowest relation, so each split
                # is seen once.
                var sub = (mask - 1) & mask
                while sub > 0:
                    if sub & low != 0:
                        var other = mask ^ sub
                        if (
                            sub in cost
                            and other in cost
                            and graph.connects(sub, other)
                        ):
                            var c = (
                                graph.estimate(mask) + cost[sub] + cost[other]
                            )
                            if best < 0 or c < best:
                                best = c
                                best_left = sub
                    sub = (sub - 1) & mask
                if best >= 0:
                    cost[mask] = best
                    split[mask] = best_left
        if full not in split:
            return None
        # The tree, built from the splits down.
        var entry = Dict[Int, Int]()
        for r in range(n):
            entry[1 << r] = r

        _ = _build(tree, entry, graph, split, full)
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
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if not graph.connects(sets[i], sets[j]):
                    continue
                var c = graph.estimate(sets[i] | sets[j]) + costs[i] + costs[j]
                if best < 0 or c < best:
                    best = c
                    bi = i
                    bj = j
        if bi < 0:
            return None
        var merged = sets[bi] | sets[bj]
        tree.left.append(nodes[bi])
        tree.right.append(nodes[bj])
        tree.mask.append(merged)
        tree.rows.append(graph.estimate(merged))
        sets[bi] = merged
        nodes[bi] = len(tree.mask) - 1
        costs[bi] = best
        _ = sets.pop(bj)
        _ = nodes.pop(bj)
        _ = costs.pop(bj)
    return tree^
