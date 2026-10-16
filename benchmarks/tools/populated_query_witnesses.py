#!/usr/bin/env python3
"""Separate populated inputs for ten existing external-suite query cells.

Never generates or changes canonical data. Copies complete observed Parquet
schemas, supplies small qualifying/distractor rows, and checks independent SQL
answers against literal rows before publishing a supplemental manifest. No Mojo
compiler, benchmark worker, or accelerator is launched by this tool.
"""
import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BACKLOG = ROOT / "benchmarks/suites/populated_witness_backlog.json"
CANONICAL = [
    "benchmarks/suites/tpcds.mojo",
    "benchmarks/suites/clickbench.mojo",
    "benchmarks/suites/datagen.py",
    "benchmarks/metal_suite_common.mojo",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_rows(suite, query):
    """Only input rows; no implementation of a canonical query."""
    if suite == "clickbench":
        if query == "q19":
            return {
                "hits": [
                    {"WatchID": 1, "UserID": 435090932899640449},
                    {"WatchID": 2, "UserID": 435090932899640449},
                    {"WatchID": 3, "UserID": 435090932899640448},
                    {"WatchID": 4, "UserID": None},
                ]
            }
        if query == "q41":
            common = {
                "CounterID": 62,
                "EventDate": "2013-07-15",
                "IsRefresh": 0,
                "DontCountHits": 0,
                "URLHash": 2868770270353813622,
            }
            hits = [
                {
                    **common,
                    "WatchID": 2 * group + duplicate + 1,
                    "WindowClientWidth": group,
                    "WindowClientHeight": 1,
                }
                for group in range(10000)
                for duplicate in range(2)
            ]
            hits.append(
                {
                    **common,
                    "WatchID": 20001,
                    "WindowClientWidth": 20000,
                    "WindowClientHeight": 2,
                }
            )
            return {"hits": hits}
    if query in {"q3", "q19"}:
        dates = [
            {"d_date_sk": 1, "d_year": 1998, "d_moy": 11},
            {"d_date_sk": 2, "d_year": 1999, "d_moy": 11},
            {"d_date_sk": 3, "d_year": 1998, "d_moy": 10},
        ]
        items = [
            {
                "i_item_sk": 1,
                "i_brand_id": 10,
                "i_brand": "Alpha",
                "i_manufact_id": 128,
                "i_manufact": "MakerA",
                "i_manager_id": 8,
            },
            {
                "i_item_sk": 2,
                "i_brand_id": 20,
                "i_brand": "Beta",
                "i_manufact_id": 128,
                "i_manufact": "MakerB",
                "i_manager_id": 8,
            },
            {
                "i_item_sk": 3,
                "i_brand_id": 30,
                "i_brand": "Excluded",
                "i_manufact_id": 999,
                "i_manufact": "MakerC",
                "i_manager_id": 9,
            },
            {
                "i_item_sk": 4,
                "i_brand_id": 40,
                "i_brand": None,
                "i_manufact_id": 128,
                "i_manufact": "MakerD",
                "i_manager_id": 8,
            },
        ]

        def sale(item=1, day=1, price="10.00", customer=1):
            return {
                "ss_item_sk": item,
                "ss_sold_date_sk": day,
                "ss_ext_sales_price": price,
                "ss_customer_sk": customer,
                "ss_store_sk": 1,
            }

        sales = [
            sale(),
            sale(price="20.00"),
            sale(price=None),
            sale(item=2, day=2 if query == "q3" else 1, price="5.00"),
            sale(item=3, price="1000.00"),
            sale(day=3, price="1000.00"),
            sale(item=999, price="1000.00"),
            sale(day=None, price="1000.00"),
        ]
        if query == "q3":
            sales.append(sale(item=4, price="7.00"))
        result = {"date_dim": dates, "item": items, "store_sales": sales}
        if query == "q19":
            sales.extend(
                [
                    sale(customer=2, price="1000.00"),
                    sale(customer=3, price="1000.00"),
                    sale(customer=999, price="1000.00"),
                ]
            )
            result.update(
                {
                    "customer": [
                        {"c_customer_sk": i, "c_current_addr_sk": i}
                        for i in range(1, 4)
                    ],
                    "customer_address": [
                        {"ca_address_sk": 1, "ca_zip": "22222-0000"},
                        {"ca_address_sk": 2, "ca_zip": "11111-9999"},
                        {"ca_address_sk": 3, "ca_zip": None},
                    ],
                    "store": [{"s_store_sk": 1, "s_zip": "11111-0000"}],
                }
            )
        return result
    if query == "q82":
        attributes = [
            ("A", "Low", "62.00", 129),
            ("B", "High", "92.00", 270),
            ("C", "Below", "61.99", 129),
            ("D", "Above", "92.01", 129),
            ("E", "Wrong maker", "70.00", 999),
            ("F", "Null maker", "70.00", None),
            ("G", "Low stock", "75.00", 821),
            ("H", "Wrong date", "75.00", 423),
            ("I", "Unsold", "75.00", 129),
            ("J", "Null price", None, 129),
            ("K", "Null stock", "80.00", 423),
        ]
        return {
            "item": [
                {
                    "i_item_sk": i,
                    "i_item_id": name,
                    "i_item_desc": desc,
                    "i_current_price": price,
                    "i_manufact_id": maker,
                }
                for i, (name, desc, price, maker) in enumerate(attributes, 1)
            ],
            "date_dim": [
                {"d_date_sk": i, "d_date": day}
                for i, day in enumerate(
                    ["2000-05-25", "2000-07-24", "2000-05-24", "2000-07-25"], 1
                )
            ],
            "inventory": [
                {
                    "inv_item_sk": i,
                    "inv_date_sk": 3 if i == 8 else 1,
                    "inv_quantity_on_hand": 99 if i
                    == 7 else None if i
                    == 11 else 100,
                }
                for i in range(1, 12)
            ]
            + [
                {
                    "inv_item_sk": 1,
                    "inv_date_sk": 2,
                    "inv_quantity_on_hand": 500,
                },
                {
                    "inv_item_sk": 2,
                    "inv_date_sk": 2,
                    "inv_quantity_on_hand": 500,
                },
            ],
            "store_sales": [
                {"ss_item_sk": i} for i in [1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
            ],
        }
    if query == "q24":

        def sale(ticket, item, paid, store=1, customer=1):
            return {
                "ss_ticket_number": ticket,
                "ss_item_sk": item,
                "ss_net_paid": paid,
                "ss_store_sk": store,
                "ss_customer_sk": customer,
            }

        return {
            "store_sales": [
                sale(100, 1, "10.00"),
                sale(100, 1, "20.00"),
                sale(200, 2, "100.00"),
                sale(300, 3, None),
                sale(999, 1, "1000.00"),
                sale(400, 1, "1000.00", store=2),
                sale(500, 1, "1000.00", customer=2),
                sale(600, 1, "1000.00", customer=3),
            ],
            "store_returns": [
                {"sr_ticket_number": ticket, "sr_item_sk": item}
                for ticket, item in [
                    (100, 1),
                    (200, 2),
                    (300, 3),
                    (400, 1),
                    (500, 1),
                    (600, 1),
                ]
            ],
            "store": [
                {
                    "s_store_sk": 1,
                    "s_market_id": 8,
                    "s_store_name": "Main",
                    "s_zip": "11111",
                    "s_state": "CA",
                },
                {
                    "s_store_sk": 2,
                    "s_market_id": 9,
                    "s_store_name": "Excluded",
                    "s_zip": "11111",
                    "s_state": "CA",
                },
            ],
            "item": [
                {
                    "i_item_sk": i,
                    "i_color": color,
                    "i_current_price": price,
                    "i_manager_id": i + 7,
                    "i_units": "unit",
                    "i_size": "S",
                }
                for i, color, price in [
                    (1, "peach", "10.00"),
                    (2, "blue", "20.00"),
                    (3, "peach", "30.00"),
                ]
            ],
            "customer": [
                {
                    "c_customer_sk": i,
                    "c_current_addr_sk": i,
                    "c_birth_country": country,
                    "c_last_name": "Smith",
                    "c_first_name": "Ada",
                }
                for i, country in [(1, "CANADA"), (2, "USA"), (3, None)]
            ],
            "customer_address": [
                {
                    "ca_address_sk": i,
                    "ca_zip": "11111",
                    "ca_state": "CA",
                    "ca_country": "usa",
                }
                for i in range(1, 4)
            ],
        }
    raise ValueError((suite, query))


SQL = {
    "q3": """SELECT d.d_year, i.i_brand_id AS brand_id, i.i_brand AS brand,
        SUM(s.ss_ext_sales_price) AS sum_agg FROM store_sales s
        JOIN date_dim d ON s.ss_sold_date_sk=d.d_date_sk
        JOIN item i ON s.ss_item_sk=i.i_item_sk
        WHERE d.d_moy=11 AND i.i_manufact_id=128
        GROUP BY d.d_year,i.i_brand_id,i.i_brand
        ORDER BY d.d_year,sum_agg DESC,brand_id LIMIT 100""",
    "q19": """SELECT i.i_brand_id AS brand_id,i.i_brand AS brand,
        i.i_manufact_id,i.i_manufact,SUM(s.ss_ext_sales_price) AS ext_price
        FROM store_sales s JOIN date_dim d ON s.ss_sold_date_sk=d.d_date_sk
        JOIN item i ON s.ss_item_sk=i.i_item_sk
        JOIN customer c ON s.ss_customer_sk=c.c_customer_sk
        JOIN customer_address a ON c.c_current_addr_sk=a.ca_address_sk
        JOIN store st ON s.ss_store_sk=st.s_store_sk
        WHERE d.d_moy=11 AND d.d_year=1998 AND i.i_manager_id=8
        AND SUBSTR(a.ca_zip,1,5)<>SUBSTR(st.s_zip,1,5)
        GROUP BY i.i_brand_id,i.i_brand,i.i_manufact_id,i.i_manufact
        ORDER BY ext_price DESC,brand,brand_id,i_manufact_id,i_manufact LIMIT 100""",
    "q82": """SELECT DISTINCT i_item_id,i_item_desc,i_current_price FROM item i
        WHERE i_current_price BETWEEN 62 AND 92 AND i_manufact_id IN(129,270,821,423)
        AND EXISTS(SELECT 1 FROM inventory v JOIN date_dim d ON v.inv_date_sk=d.d_date_sk
            WHERE v.inv_item_sk=i.i_item_sk AND v.inv_quantity_on_hand BETWEEN 100 AND 500
            AND d.d_date BETWEEN DATE '2000-05-25' AND DATE '2000-07-24')
        AND EXISTS(SELECT 1 FROM store_sales s WHERE s.ss_item_sk=i.i_item_sk)
        ORDER BY i_item_id LIMIT 100""",
    "q24": """WITH ssales AS(SELECT c.c_last_name,c.c_first_name,st.s_store_name,
        a.ca_state,st.s_state,i.i_color,i.i_current_price,i.i_manager_id,i.i_units,i.i_size,
        SUM(s.ss_net_paid) AS netpaid FROM store_sales s
        JOIN store_returns r ON s.ss_ticket_number=r.sr_ticket_number AND s.ss_item_sk=r.sr_item_sk
        JOIN store st ON s.ss_store_sk=st.s_store_sk
        JOIN item i ON s.ss_item_sk=i.i_item_sk
        JOIN customer c ON s.ss_customer_sk=c.c_customer_sk
        JOIN customer_address a ON c.c_current_addr_sk=a.ca_address_sk AND st.s_zip=a.ca_zip
        WHERE st.s_market_id=8 AND c.c_birth_country<>UPPER(a.ca_country)
        GROUP BY c.c_last_name,c.c_first_name,st.s_store_name,a.ca_state,st.s_state,
                 i.i_color,i.i_current_price,i.i_manager_id,i.i_units,i.i_size)
        SELECT c_last_name,c_first_name,s_store_name,SUM(netpaid) AS paid FROM ssales
        WHERE i_color='peach' GROUP BY c_last_name,c_first_name,s_store_name
        HAVING SUM(netpaid)>0.05*(SELECT AVG(CAST(netpaid AS DOUBLE)) FROM ssales)
        ORDER BY c_last_name,c_first_name,s_store_name""",
    "clickbench/q19": "SELECT UserID FROM hits WHERE UserID=435090932899640449",
    "clickbench/q41": """SELECT WindowClientWidth,WindowClientHeight,COUNT(*) AS PageViews
        FROM hits WHERE CounterID=62 AND EventDate BETWEEN DATE '2013-07-01' AND DATE '2013-07-31'
        AND IsRefresh=0 AND DontCountHits=0 AND URLHash=2868770270353813622
        GROUP BY WindowClientWidth,WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000""",
}


def literal_rows(suite, query, decimal):
    money = Decimal if decimal else float
    if suite == "clickbench":
        return [
            (435090932899640449,),
            (435090932899640449,),
        ] if query == "q19" else [(20000, 2, 1)]
    return {
        "q3": [
            (1998, 10, "Alpha", money("30.00")),
            (1998, 40, None, money("7.00")),
            (1999, 20, "Beta", money("5.00")),
        ],
        "q19": [
            (10, "Alpha", 128, "MakerA", money("30.00")),
            (20, "Beta", 128, "MakerB", money("5.00")),
        ],
        "q82": [("A", "Low", money("62.00")), ("B", "High", money("92.00"))],
        "q24": [("Smith", "Ada", "Main", money("30.00"))],
    }[query]


def normalize(value, dtype, pl):
    if value is None:
        return None
    if dtype.base_type() == pl.Decimal:
        return Decimal(str(value))
    if dtype == pl.Date:
        return dt.date.fromisoformat(value) if isinstance(value, str) else value
    if dtype.base_type() == pl.Datetime:
        return dt.datetime.fromisoformat(value) if isinstance(
            value, str
        ) else value
    if dtype.is_float():
        return float(value)
    if dtype.is_integer():
        return int(value)
    return value


def json_value(value):
    return {"decimal": str(value)} if isinstance(value, Decimal) else value


def generate(canonical_root, output):
    import duckdb
    import polars as pl

    canonical_root, output = canonical_root.resolve(), output.resolve()
    if output == canonical_root or output.is_relative_to(canonical_root):
        raise ValueError(
            "Supplemental output must be outside canonical data root"
        )
    if (output / "manifest.json").exists():
        raise ValueError(
            "Published supplemental fixtures are immutable; choose a fresh output directory"
        )
    source_hashes = {name: sha(ROOT / name) for name in CANONICAL}
    backlog = json.loads(BACKLOG.read_text())
    cases, metadata = [], {}
    connection = duckdb.connect()
    specs = [
        ("tpcds", variant, q)
        for variant in ["base", "decimal"]
        for q in ["q3", "q19", "q82", "q24"]
    ] + [("clickbench", "base", "q19"), ("clickbench", "base", "q41")]
    for suite, variant, query in specs:
        witness_id = f"{suite}-{variant}-{query}-populated-v1"
        folder = output / witness_id
        folder.mkdir(parents=True, exist_ok=True)
        tables = {}
        for name, rows in fixture_rows(suite, query).items():
            observed = canonical_root / (
                "clickbench/hits_1.parquet" if suite
                == "clickbench" else f"tpcds/sf0.01{'_decimal' if variant == 'decimal' else ''}/{name}.parquet"
            )
            schema = pl.read_parquet_schema(observed)
            for row in rows:
                unknown = set(row) - set(schema)
                if unknown:
                    raise ValueError(
                        f"Fixture columns absent from original schema: {unknown}"
                    )
            values = [
                {
                    column: normalize(row.get(column), dtype, pl)
                    for column, dtype in schema.items()
                }
                for row in rows
            ]
            frame = pl.DataFrame(values, schema=schema, strict=True)
            path = folder / f"{name}.parquet"
            frame.write_parquet(path)
            if pl.read_parquet_schema(path) != schema:
                raise ValueError("Supplemental schema changed")
            connection.read_parquet(str(path)).create_view(name, replace=True)
            tables[name] = {
                "path": str(path),
                "relative_path": str(path.relative_to(output)),
                "sha256": sha(path),
                "rows": len(rows),
                "schema": {
                    column: str(dtype) for column, dtype in schema.items()
                },
                "observed_schema_source": str(observed),
                "observed_schema_sha256": hashlib.sha256(
                    json.dumps({k: str(v) for k, v in schema.items()}).encode()
                ).hexdigest(),
            }
        sql = SQL[f"clickbench/{query}" if suite == "clickbench" else query]
        expected = literal_rows(suite, query, variant == "decimal")
        actual = connection.execute(sql).fetchall()
        if actual != expected or not actual:
            raise ValueError(
                f"Independent oracle differs {witness_id}: {actual} != {expected}"
            )
        expected_path = folder / "expected.parquet"
        # SQL is an independent input oracle, not a rewrite of the executed Mojo query.
        connection.sql(sql).write_parquet(str(expected_path))
        hybrid = suite == "tpcds" and query == "q24"
        source_path = f"benchmarks/suites/{suite}.mojo"
        boundary = "hybrid" if hybrid else "gpu_collects"
        reason = (
            "Canonical q24 retains GPU collect -> eager CPU mean/item -> GPU collect; all internal materializations included in timing." if hybrid else "Canonical query collect boundaries explicitly forcedMetal by existing derivation; no query CPU fallback."
        )
        case_metadata = {
            "workload_kind": "supplemental",
            "execution_boundary": boundary,
            "audit_reason": reason,
            "source_path": source_path,
            "source_sha256": source_hashes[source_path],
        }
        # Existing suite/query wildcard applies to both variants; separate run receipts avoid collisions.
        metadata[f"{suite}/{query}"] = case_metadata
        cases.append(
            {
                **case_metadata,
                "suite": suite,
                "variant": variant,
                "query": query,
                "witness_id": witness_id,
                "canonical_query_id": query,
                "tables": tables,
                "expected_output_rows": len(expected),
                "expected_literal_rows": [
                    [json_value(x) for x in row] for row in expected
                ],
                "oracle_sql": sql,
                "oracle_sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
                "oracle_result_path": str(expected_path),
                "oracle_result_relative_path": str(
                    expected_path.relative_to(output)
                ),
                "oracle_result_sha256": sha(expected_path),
                "oracle_validation": "DuckDB result exactly equals independent literal ordered rows",
                "cpu_mojo_status": "untested",
                "apple_gpu_status": "untested",
                "actual_output_rows": None,
                "ordered_equal": None,
                "gpu_profile_counters": None,
                "before_output_rows": 0,
                "before_dataset": "canonical-smoke0c11",
                "after_dataset": "supplemental-populated",
                "timing_scope": "whole canonical query including CPU mean/item" if hybrid else "whole canonical query",
                "worker_argv": [suite, query, "2"]
                + [f"{name}={item['path']}" for name, item in tables.items()],
                "oracle_worker_argv": {
                    engine: [
                        witness_id,
                        suite,
                        query,
                        engine,
                        str(expected_path),
                    ]
                    + [
                        f"{name}={item['path']}"
                        for name, item in tables.items()
                    ]
                    for engine in ["cpu", "metal"]
                },
            }
        )
        for name in tables:
            connection.execute(f'DROP VIEW "{name}"')
    connection.close()
    selected = {
        (c["suite"], c["variant"], c["query"])
        for c in cases
        if c["execution_boundary"] != "hybrid"
    }
    backlog_keys = {
        (c["suite"], c["variant"], c["query"]) for c in backlog["cases"]
    }
    if (
        len(backlog_keys) != 49
        or len(selected) != 8
        or not selected <= backlog_keys
    ):
        raise ValueError("Backlog/selection denominator changed")
    if {name: sha(ROOT / name) for name in CANONICAL} != source_hashes:
        raise ValueError("Canonical source changed during generation")
    manifest = {
        "format": "dataframe-populated-query-witnesses-v1",
        "workload_kind": "supplemental",
        "canonical_denominator_changed": False,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "canonical_source_sha256": source_hashes,
        "generator_sha256": sha(__file__),
        "schema_policy": "Full original observed schemas; unreferenced columns null. No canonical data/query/datagen/comparator changes.",
        "prior_nonhybrid_empty_count": 49,
        "selected_nonhybrid_cells": 8,
        "remaining_nonhybrid_empty_count": 41,
        "additional_known_hybrid_cells": 2,
        "remaining_backlog": [
            c
            for c in backlog["cases"]
            if (c["suite"], c["variant"], c["query"]) not in selected
        ],
        "validation": "Ten independent DuckDB+literal oracle checks pass; MojoCPU/AppleGPU not executed.",
        "versions": {"duckdb": duckdb.__version__, "polars": pl.__version__},
        "cases": cases,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "case-metadata.json").write_text(
        json.dumps(
            {"format": "dataframe-metal-case-metadata-v1", "cases": metadata},
            indent=2,
        )
        + "\n"
    )
    return manifest


def audit(manifest_path, receipts, oracle_logs=()):
    """Join separate supplemental measurements without promoting fixture generation."""
    manifest = json.loads(manifest_path.read_text())
    expected = {
        (c["suite"], c["variant"], c["query"]): c for c in manifest["cases"]
    }
    observed = {}
    for receipt in receipts:
        data = json.loads(receipt.read_text())
        originals = data.get("provenance", {}).get("original_source_sha256", {})
        for case in data["cases"]:
            key = case["suite"], case["variant"], case["query"]
            if key not in expected or key in observed:
                raise ValueError(
                    f"Unexpected/duplicate supplemental cell: {key}"
                )
            source = expected[key]["source_path"]
            if originals.get(source) != expected[key]["source_sha256"]:
                raise ValueError(
                    "Measured canonical query source identity differs"
                )
            if set(case.get("inputs", {})) != set(expected[key]["tables"]):
                raise ValueError("Measured fixture table set differs")
            for name, item in expected[key]["tables"].items():
                if (
                    case.get("inputs", {}).get(name, {}).get("sha256")
                    != item["sha256"]
                ):
                    raise ValueError(
                        f"Measured input identity differs: {key}/{name}"
                    )
            observed[key] = case
    oracles = {}
    identities = {c["witness_id"]: c for c in manifest["cases"]}
    for path in oracle_logs:
        for line in path.read_text().splitlines():
            if not line.startswith("witness_oracle\t"):
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                raise ValueError("Malformed oracle receipt")
            _, identity, engine, count, matched = fields
            key = identity, engine
            if (
                identity not in identities
                or engine not in {"cpu", "metal"}
                or key in oracles
            ):
                raise ValueError("Unknown/duplicate oracle receipt")
            if (
                matched != "True"
                or int(count) != identities[identity]["expected_output_rows"]
            ):
                raise ValueError("Independent target oracle differs")
            oracles[key] = True
    rows = []
    for key, fixture in expected.items():
        measured = observed.get(key, {})
        if (
            measured.get("status") == "ok"
            and measured.get("output_rows") != fixture["expected_output_rows"]
        ):
            raise ValueError("Populated measurement has wrong output row count")
        rows.append(
            {
                "suite": key[0],
                "variant": key[1],
                "query": key[2],
                "witness_id": fixture["witness_id"],
                "workload_kind": "supplemental",
                "execution_boundary": fixture["execution_boundary"],
                "before_output_rows": fixture["before_output_rows"],
                "expected_output_rows": fixture["expected_output_rows"],
                "actual_output_rows": measured.get("output_rows"),
                "parity_status": measured.get("status", "untested"),
                "ordered_equal": measured.get("ordered_equal"),
                "diagnostic_cpu_collection_ns": None,
                "diagnostic_gpu_collection_ns": measured.get("collection_ns"),
                "performance_evidence": False,
                "literal_oracle_cpu_checked": oracles.get(
                    (fixture["witness_id"], "cpu"), False
                ),
                "literal_oracle_metal_checked": oracles.get(
                    (fixture["witness_id"], "metal"), False
                ),
            }
        )
    return {
        "format": "dataframe-populated-query-audit-v1",
        "canonical_denominator_changed": False,
        "remaining_nonhybrid_empty_count": 41,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    create = sub.add_parser("generate")
    create.add_argument("--canonical-data-root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    review = sub.add_parser("audit")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--receipt", type=Path, action="append", default=[])
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--oracle-log", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.operation == "generate":
        manifest = generate(args.canonical_data_root, args.output)
        print(
            f"10 supplementary fixtures;8/49 empty cells selected,41 remain;2q24 hybrid. Oracle checks pass; MojoCPU/AppleGPU untested. {args.output / 'manifest.json'}"
        )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                audit(args.manifest, args.receipt, args.oracle_log), indent=2
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
