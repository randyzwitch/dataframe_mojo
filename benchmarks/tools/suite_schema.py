"""Reviewed SQL/API schema compatibility; original receipts remain unchanged.

DuckDB transports HUGEINT integer SUM outputs through Arrow as DECIMAL(38,0).
Only the declared integer-reduction positions may use that representation.
The conversion below checks exact integrality and never rounds or uses float.
"""
from decimal import Decimal

from suite_answers import Answer


INTEGER_SUMS = {
    ("h2o_groupby", "q1"): (1,),
    ("h2o_groupby", "q2"): (2,),
    ("h2o_groupby", "q3"): (1,),
    ("h2o_groupby", "q5"): (1, 2),
    ("pdsh", "q12"): (1, 2),
    ("tpcds", "q14"): (5,),
    ("tpcds", "q21"): (2, 3),
    ("tpcds", "q29"): (4, 5, 6),
    ("tpcds", "q48"): (0,),
    ("tpcds", "q50"): (10, 11, 12, 13, 14),
    ("tpcds", "q62"): (3, 4, 5, 6, 7),
    ("tpcds", "q72"): (3, 4),
    ("tpcds", "q75"): (6, 7, 8),
    ("tpcds", "q78"): (4, 7),
    ("tpcds", "q83"): (1, 3, 5),
    ("tpcds", "q97"): (0, 1, 2),
    ("tpcds", "q99"): (3, 4, 5, 6, 7),
    ("clickbench", "q2"): (0,),
    ("clickbench", "q9"): (1,),
    ("clickbench", "q29"): tuple(range(90)),
    ("clickbench", "q30"): (3,),
    ("clickbench", "q31"): (3,),
    ("clickbench", "q32"): (3,),
}

# Intentional output labels in the native/Polars translations versus DuckDB's
# upstream SQL labels. Each alias is reviewed per query; arbitrary renaming
# or case folding is never a schema compatibility rule.
ALIASES = {
    **{
        ("h2o_join", query): {f"{name}_1": f"{name}_right" for name in names}
        for query, names in {
            "q1": ("id4",),
            "q2": ("id1", "id4", "id5"),
            "q3": ("id1", "id4", "id5"),
            "q4": ("id1", "id2", "id4"),
            "q5": ("id1", "id2", "id4", "id5", "id6"),
        }.items()
    },
    **{
        ("clickbench", q): {"count_star()": "count"}
        for q in ("q0", "q1", "q20")
    },
    ("clickbench", "q2"): {
        "sum(AdvEngineID)": "s",
        "count_star()": "c",
        "avg(ResolutionWidth)": "a",
    },
    ("clickbench", "q3"): {"avg(UserID)": "UserID"},
    ("clickbench", "q4"): {"count(DISTINCT UserID)": "UserID"},
    ("clickbench", "q5"): {"count(DISTINCT SearchPhrase)": "SearchPhrase"},
    ("clickbench", "q6"): {"min(EventDate)": "min", "max(EventDate)": "max"},
    **{
        ("clickbench", q): {"count_star()": "c"}
        for q in ("q7", "q15", "q16", "q17", "q18")
    },
    ("clickbench", "q9"): {
        "sum(AdvEngineID)": "s",
        "avg(ResolutionWidth)": "a",
        "count(DISTINCT UserID)": "u",
    },
    ("clickbench", "q21"): {"min(URL)": "url"},
    ("clickbench", "q22"): {
        "min(URL)": "url",
        "min(Title)": "title",
        "count(DISTINCT UserID)": "u",
    },
    ("clickbench", "q28"): {"min(Referer)": "Referer"},
    ("clickbench", "q29"): {
        "sum(ResolutionWidth)": "s0",
        **{f"sum((ResolutionWidth + {i}))": f"s{i}" for i in range(1, 90)},
    },
    **{
        ("clickbench", q): {"sum(IsRefresh)": "r", "avg(ResolutionWidth)": "w"}
        for q in ("q30", "q31", "q32")
    },
    ("clickbench", "q34"): {"1": "one"},
    ("clickbench", "q35"): {f"(ClientIP - {i})": f"m{i}" for i in range(1, 4)},
    ("pdsh", "q18"): {"sum(l_quantity)": "sum_qty"},
    ("tpcds", "q2"): {"round((sat_sales1 / sat_sales2), 2)": "r7"},
    ("tpcds", "q13"): {"sum(ss_ext_wholesale_cost)": "sum4"},
    ("tpcds", "q15"): {"sum(cs_sales_price)": "total"},
    ("tpcds", "q32"): {"excess discount amount": "excess"},
    ("tpcds", "q35"): {
        "min(cd_dep_college_count)": "min3",
        "max(cd_dep_college_count)": "max3",
        "avg(cd_dep_college_count)": "avg3",
    },
    ("tpcds", "q39"): {
        "w_warehouse_sk": "wsk2",
        "i_item_sk": "isk2",
        "d_moy": "dmoy2",
        "mean": "mean2",
        "cov": "cov2",
    },
    ("tpcds", "q42"): {"sum(ss_ext_sales_price)": "total"},
    ("tpcds", "q45"): {"sum(ws_sales_price)": "total"},
    ("tpcds", "q48"): {"sum(ss_quantity)": "total"},
    ("tpcds", "q54"): {"SEGMENT": "segment"},
    ("tpcds", "q61"): {
        "((CAST(promotions AS DECIMAL(15, 4)) / CAST(total AS DECIMAL(15, 4))) * 100)": "share"
    },
    ("tpcds", "q79"): {'"substring"(s_city, 1, 30)': "city"},
    ("tpcds", "q85"): {
        '"substring"(r_reason_desc, 1, 20)': "reason",
        "avg(wr_fee)": "avg3",
    },
    ("tpcds", "q92"): {"Excess Discount Amount": "excess"},
    ("tpcds", "q96"): {"count_star()": "count"},
}


def normalize(answer, suite, query):
    """Apply only declared transport equivalences, preserving exact values."""
    types, rows = list(answer.types), list(answer.rows)
    for position in INTEGER_SUMS.get((suite, query), ()):
        if position >= len(types):
            raise ValueError("declared integer reduction is missing")
        dtype = types[position]
        if dtype not in {
            "int64",
            "uint64",
            "decimal(38, 0)",
            "decimal128(38, 0)",
        }:
            raise ValueError("integer reduction has an unexpected dtype")
        converted = []
        for row in rows:
            values = list(row)
            value = values[position]
            if value is not None:
                if type(value) not in (int, Decimal) or value != int(value):
                    raise ValueError(
                        "integer reduction contains a non-integer value"
                    )
                values[position] = int(value)
            converted.append(tuple(values))
        rows = converted
        types[position] = "integer_sum"
    aliases = ALIASES.get((suite, query), {})
    names = [aliases.get(n, n) for n in answer.names]
    if len(set(names)) != len(names):
        raise ValueError("schema alias mapping produces duplicate columns")
    # GROUPING tags are computed integers, represented as INTEGER by DuckDB
    # and Int64 by the native expression API. Copied input keys stay exact.
    grouping = {("tpcds", "q27"): 2, ("tpcds", "q36"): 3}.get((suite, query))
    if grouping is not None:
        if grouping >= len(types) or types[grouping] not in {"int32", "int64"}:
            raise ValueError("grouping tag has an unexpected dtype")
        types[grouping] = "int64"
    # The upstream SQL spells the projected literal as INTEGER; lit(1) uses
    # Int64 in Mojo. This is a literal-only compatibility rule, with its value
    # checked independently rather than accepting arbitrary integer columns.
    if (suite, query) == ("clickbench", "q34"):
        if (
            not types
            or types[0] not in {"int32", "int64"}
            or any(r[0] != 1 for r in rows)
        ):
            raise ValueError("q34 literal column does not contain integer one")
        types[0] = "int64"
    return Answer(names, types, rows, answer.original_schema)
