"""Exercise SQL translation edge cases using the already-built suite runners.

Run after bench_suites.py --correctness-only, selecting the same CI suites.
These fixtures include populated decimal results where smoke queries are empty.
"""
import argparse
from datetime import date
from decimal import Decimal
from pathlib import Path
import sys
import tempfile

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/tools"))
import bench_suites as bench


def fixtures():
    groups = ["missing"] * 2 + ["single"] * 2 + ["constant"] * 2 + ["paired"] * 3
    h2o = pl.DataFrame({
        "id1": groups, "id2": groups, "id3": groups,
        "id4": [1] * 9, "id5": [1] * 9, "id6": [1] * 9,
        "v1": [None, None, 1, None, 1, 1, 1, 2, 3],
        "v2": [None, None, 2, 3, 2, 2, 2, 4, 6],
        "v3": [None, None, 1., 2., 3., 4., 5., 6., 7.],
    })
    sales = pl.DataFrame({
        "ss_item_sk": [1] * 5, "ss_ticket_number": list(range(1, 6)),
        "ss_customer_sk": [1] * 5, "ss_sold_date_sk": [1] * 5,
        "ss_quantity": [10] * 5,
        "ss_sales_price": [Decimal("20.01")] * 5,
    }).with_columns(pl.col("ss_sales_price").cast(pl.Decimal(7, 2)))
    tpcds = {
        "store_sales": sales,
        "store_returns": pl.DataFrame({
            "sr_item_sk": [1], "sr_ticket_number": [1],
            "sr_return_quantity": [2], "sr_reason_sk": [1],
        }),
        "reason": pl.DataFrame({"r_reason_sk": [1], "r_reason_desc": ["reason 28"]}),
        "date_dim": pl.DataFrame({
            "d_date_sk": [1], "d_date": [date(2000, 2, 1)],
            "d_year": [2000], "d_moy": [2],
        }),
        "item": pl.DataFrame({"i_item_sk": [1], "i_item_desc": ["frequent item"]}),
        "customer": pl.DataFrame({
            "c_customer_sk": [1], "c_last_name": ["Example"], "c_first_name": ["Pat"],
        }),
    }
    for table, prefix, quantity, price in (
        ("catalog_sales", "cs", 3, "12.34"),
        ("web_sales", "ws", 2, "56.78"),
    ):
        tpcds[table] = pl.DataFrame({
            prefix + "_item_sk": [1], prefix + "_bill_customer_sk": [1],
            prefix + "_sold_date_sk": [1], prefix + "_quantity": [quantity],
            prefix + "_list_price": [Decimal(price)],
        }).with_columns(pl.col(prefix + "_list_price").cast(pl.Decimal(7, 2)))
    return (
        ("h2o_groupby", ["q1", "q2", "q3", "q5", "q9", "q10"], {"x": h2o}),
        ("tpcds", ["q23", "q93"], tpcds),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", required=True)
    args = parser.parse_args()
    selected = args.suites.split(",")
    runners = {"mojo": {
        name: ROOT / "build/suites" / ("suite_" + name) for name in ("h2o", "tpcds")
    }}
    for suite, queries, tables in fixtures():
        if suite not in selected:
            continue
        with tempfile.TemporaryDirectory(prefix="suite-translation-inputs-") as tmp:
            paths = {name: Path(tmp) / (name + ".parquet") for name in tables}
            for name, frame in tables.items():
                frame.write_parquet(paths[name])
            result = {"provenance": {"rounds": 1, "reps": 0}, "runs": []}
            for engine in ("mojo", "duckdb"):
                outcomes, stderr = bench.run_worker(
                    bench.command(engine, runners, suite, queries, 0, paths),
                    bench.environment(4), 180, True, queries,
                )
                for query, outcome in outcomes.items():
                    expected_height = (
                        2 if query == "q23" else 1 if query in {"q5", "q93"} else 4
                    )
                    if outcome.get("summary", {}).get("height") != expected_height:
                        raise AssertionError((suite, engine, query, outcome, stderr))
                    result["runs"].append(dict(
                        outcome, suite=suite, variant="fixture", query=query,
                        engine=engine, round=0,
                    ))
            cells = bench.evaluate(result)
            failures = {str(k): v for k, v in cells.items() if v["status"] != "ok"}
            if failures:
                raise AssertionError((failures, stderr))
            print(f"{suite}: {len(queries)} edge-case answers match independent SQL")


if __name__ == "__main__":
    main()
