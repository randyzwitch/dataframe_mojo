"""The cells of the operator parity harness (bench_operators.py).

A cell is one operator over one synthetic input: a dataset spec (what
generate.py writes), a query spec (what every engine runs) and the cell's
parameters. The matrices sweep data properties, not benchmark queries: key
values pass through random bijections, keys are never progressions or dense
surrogates unless the cell asks for dense keys, and the join and aggregate
matrices are Latin-hypercube samples with a fixed seed, so every level of
every factor appears about equally often without running the full product.

Matrices: join, agg, filter (filter -> project -> sum), sort (full and
LIMIT 100), window, sched (three hash-join builds feeding one probe).
"""

import random

SEED = 20261102

M = 1_000_000

# Resource caps, not tuning: a join cell's probe side and its expected
# output stay within what three engines can hold and compare on one host.
PROBE_CAP = 8 * M
OUTPUT_CAP = 16 * M

JOIN_FACTORS = {
    "key": [
        "int32",
        "int64",
        "date",
        "str_short",
        "str_long",
        "decimal",
        "int_int",
        "int_str",
    ],
    "build_rows": [1_000, 64_000, M, 8 * M],
    "ratio": ["1:8", "1:1", "8:1", "64:1"],
    "match": [0.01, 0.5, 1.0],
    "dups": ["1", "4", "zipf"],
    "payload": [0, 1, 8],
    "nulls": [0.0, 0.1],
    "order": ["shuffled", "sorted"],
    "how": ["inner", "left", "right", "full", "semi", "anti"],
}

AGG_FACTORS = {
    "key": [
        "int32_dense",
        "int64_sparse",
        "decimal",
        "str_short",
        "str_long",
        "int_str",
        "mixed4",
    ],
    "groups": ["10", "1K", "100K", "1M", "N/2"],
    "skew": ["uniform", "zipf1.1", "one50"],
    "rows": [M, 10 * M],
    "nulls": [0.0, 0.1],
    "aggs": [
        "count",
        "sum_mean_int",
        "sum_mean_float",
        "sum_mean_decimal",
        "str_min_max",
        "first_last",
        "n_unique_int",
        "n_unique_str",
        "mixed6",
    ],
}

# Aggregate sets: (function, column) pairs. `fd` is a function of the
# group (constant within it), so first/last are determined in every
# engine; `u_i` and `u_s` hold no nulls, so n_unique means the same in
# SQL and Polars.
AGG_SETS = {
    "count": [("len", "")],
    "sum_mean_int": [("sum", "v_i"), ("mean", "v_i")],
    "sum_mean_float": [("sum", "v_f"), ("mean", "v_f")],
    "sum_mean_decimal": [("sum", "v_d"), ("mean", "v_d")],
    "str_min_max": [("min", "v_s"), ("max", "v_s")],
    "first_last": [("first", "fd"), ("last", "fd")],
    "n_unique_int": [("n_unique", "u_i")],
    "n_unique_str": [("n_unique", "u_s")],
    "mixed6": [
        ("len", ""),
        ("sum", "v_i"),
        ("mean", "v_f"),
        ("max", "v_d"),
        ("min", "v_s"),
        ("n_unique", "u_i"),
    ],
}

AGG_KEY_COLUMNS = {
    "int32_dense": ["k_i32"],
    "int64_sparse": ["k_i64"],
    "decimal": ["k_dec"],
    "str_short": ["k_ss"],
    "str_long": ["k_sl"],
    "int_str": ["k_a", "k_s"],
    "mixed4": ["m_i", "m_s", "m_d", "m_dec"],
}

JOIN_KEY_WIDTH = {"int_int": 2, "int_str": 2}

# Payload column types, in order, for 1 and 8 payload columns per side.
PAYLOAD_TYPES = [
    "int64",
    "str_short",
    "float64",
    "decimal",
    "int64",
    "str_short",
    "float64",
    "int32",
]


def latin_hypercube(factors, n, seed):
    """n cells over discrete factors: each factor's levels repeated evenly
    (n / levels times, the remainder drawn at random) and shuffled
    independently, so every one-factor margin is balanced."""
    rng = random.Random(seed)
    columns = {}
    for name, levels in factors.items():
        reps, extra = divmod(n, len(levels))
        column = [level for level in levels for _ in range(reps)]
        column += rng.sample(levels, extra)
        rng.shuffle(column)
        columns[name] = column
    return [{name: columns[name][i] for name in factors} for i in range(n)]


def _ratio(text):
    probe, build = (int(x) for x in text.split(":"))
    return probe / build


def mean_dups(dups):
    """Expected build rows per distinct key (zipf: see generate.py)."""
    return {"1": 1.0, "4": 4.0, "zipf": 3.5}[dups]


def join_cell(index, p):
    build = p["build_rows"]
    dup = mean_dups(p["dups"])
    probe = max(1, int(build * _ratio(p["ratio"])))
    probe = min(probe, PROBE_CAP, int(OUTPUT_CAP / dup))
    width = JOIN_KEY_WIDTH.get(p["key"], 1)
    lk = [f"k{i}" for i in range(width)]
    rk = [f"bk{i}" for i in range(width)]
    pp = [f"p{i}" for i in range(p["payload"])]
    bp = [f"b{i}" for i in range(p["payload"])]
    how = p["how"]
    if how in ("semi", "anti"):
        select = lk + pp
    elif how == "right":
        select = rk + pp + bp
    elif how == "full":
        select = lk + rk + pp + bp
    else:
        select = lk + pp + bp
    dataset = {
        "kind": "join",
        "key": p["key"],
        "build_rows": build,
        "probe_rows": probe,
        "match": p["match"],
        "dups": p["dups"],
        "payload": p["payload"],
        "nulls": p["nulls"],
        "order": p["order"],
    }
    query = {
        "op": "join",
        "left": "probe",
        "right": "build",
        "lk": lk,
        "rk": rk,
        "how": how,
        "select": select,
    }
    params = dict(p, probe_rows=probe)
    return _cell("join", index, params, dataset, query)


def _groups(text, rows):
    if text == "N/2":
        return rows // 2
    return {"10": 10, "1K": 1_000, "100K": 100_000, "1M": M}[text]


def agg_cell(index, p):
    dataset = {
        "kind": "agg",
        "key": p["key"],
        "groups": _groups(p["groups"], p["rows"]),
        "skew": p["skew"],
        "rows": p["rows"],
        "nulls": p["nulls"],
    }
    aggs = [
        (fn, column, f"{fn}_{column}" if column else fn)
        for fn, column in AGG_SETS[p["aggs"]]
    ]
    query = {
        "op": "agg",
        "t": "t",
        "keys": AGG_KEY_COLUMNS[p["key"]],
        "aggs": aggs,
    }
    return _cell("agg", index, dict(p), dataset, query)


def filter_cells():
    cells = []
    for rows in (M, 10 * M):
        for kind in ("int", "float", "decimal"):
            for share in (0.01, 0.1, 0.5, 0.9):
                p = {"rows": rows, "type": kind, "keep": share}
                dataset = {"kind": "filter", "rows": rows, "type": kind}
                query = {
                    "op": "filter_sum",
                    "t": "t",
                    "f": "f",
                    "lt": int(round(share * 1000)),
                    "a": "a",
                    "b": "b",
                }
                cells.append(_cell("filter", len(cells), p, dataset, query))
    return cells


SORT_KEYS = {
    "int": (["s_i"], [False]),
    "decimal": (["s_dec"], [False]),
    "str25": (["s_s25"], [False]),
    "mixed3": (["s_a", "s_b", "s_c"], [False, True, False]),
}


def sort_cells():
    cells = []
    for rows in (M, 10 * M):
        for key, (by, desc) in SORT_KEYS.items():
            for limit in (0, 100):
                p = {
                    "rows": rows,
                    "key": key,
                    "kind": "top100" if limit else "full",
                }
                dataset = {"kind": "sort", "rows": rows}
                query = {
                    "op": "sort",
                    "t": "t",
                    "by": by,
                    "desc": desc,
                    "limit": limit,
                    "select": by + ["p_i", "p_s"],
                }
                cells.append(_cell("sort", len(cells), p, dataset, query))
    return cells


WINDOW_ORDER = {
    # unique order keys for row_number / cum_sum / rolling, and a tied
    # version (each value twice) for rank
    "int": (["o_i"], ["t_i"]),
    "string": (["o_s"], ["t_s"]),
    "int_str": (["o_a", "o_b"], ["t_a", "t_b"]),
}


def window_cells():
    cells = []
    for rows in (M, 10 * M):
        for parts in ("1", "100", "100K", "N/2"):
            groups = {"1": 1, "100": 100, "100K": 100_000, "N/2": rows // 2}[
                parts
            ]
            for fn in ("row_number", "rank", "cum_sum", "sum", "rolling"):
                orders = ["-"] if fn == "sum" else list(WINDOW_ORDER)
                for order in orders:
                    if fn == "sum":
                        by = []
                    else:
                        unique, tied = WINDOW_ORDER[order]
                        by = tied if fn == "rank" else unique
                    part = [] if groups == 1 else ["g"]
                    p = {
                        "rows": rows,
                        "partitions": parts,
                        "fn": fn,
                        "order": order,
                    }
                    dataset = {"kind": "window", "rows": rows, "groups": groups}
                    query = {
                        "op": "window",
                        "t": "t",
                        "fn": fn,
                        "value": "v",
                        "part": part,
                        "order": by,
                        "select": part + by + ["v", "w"],
                    }
                    cells.append(_cell("window", len(cells), p, dataset, query))
    return cells


def sched_cells():
    cells = []
    for probe, build in ((M, 64_000), (M, M), (8 * M, 64_000), (8 * M, M)):
        p = {"probe_rows": probe, "build_rows": build}
        dataset = {"kind": "sched", "probe_rows": probe, "build_rows": build}
        query = {
            "op": "sched",
            "t": "t",
            "builds": ["b1", "b2", "b3"],
            "lk": ["k1", "k2", "k3"],
            "rk": ["bk1", "bk2", "bk3"],
            "select": ["k1", "k2", "k3", "p", "x1", "x2", "x3"],
        }
        cells.append(_cell("sched", len(cells), p, dataset, query))
    return cells


def _cell(matrix, index, params, dataset, query):
    return {
        "id": f"{matrix}-{index:03d}",
        "matrix": matrix,
        "params": params,
        "dataset": dataset,
        "query": query,
    }


def matrices(join_cells=200, agg_cells=150):
    """Every matrix's cells, in a fixed order."""
    join = [
        join_cell(i, p)
        for i, p in enumerate(latin_hypercube(JOIN_FACTORS, join_cells, SEED))
    ]
    agg = [
        agg_cell(i, p)
        for i, p in enumerate(
            latin_hypercube(AGG_FACTORS, agg_cells, SEED + 1)
        )
    ]
    return {
        "join": join,
        "agg": agg,
        "filter": filter_cells(),
        "sort": sort_cells(),
        "window": window_cells(),
        "sched": sched_cells(),
    }


MATRICES = ("join", "agg", "filter", "sort", "window", "sched")
