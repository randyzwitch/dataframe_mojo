"""Complete, hash-pinned suite answers; summaries never certify correctness."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from metal_numerical_compare import DEFAULT_ATOL, DEFAULT_RTOL, _perfect_match


FORMAT = "suite-answer-v1"


def tolerant_columns(suite, query, oracle):
    """Reuse only the reviewed aggregate/arithmetic field policy, never keys."""
    policy_path = Path(__file__).with_name("suite_numerical_policy.json")
    policy = json.loads(policy_path.read_text())
    from metal_numerical_audit import validate_policy

    validate_policy(policy)
    root = Path(__file__).resolve().parents[2]
    for source, digest in policy["source_sha256"].items():
        if sha256(root / source) != digest:
            raise ValueError("numerical policy query source identity changed")
    fields = policy["cases"].get(f"{suite}/{query}", [])
    if (
        fields
        and suite == "clickbench"
        and sha256(root / "benchmarks/suites/clickbench.mojo")
        != policy.get("clickbench_source_sha256")
    ):
        raise ValueError("ClickBench numerical policy source identity changed")
    sql_hash = policy.get("duckdb_sql_sha256", {}).get(f"{suite}/{query}")
    if (
        sql_hash
        and oracle.original_schema.get("oracle", {}).get("sql_sha256")
        != sql_hash
    ):
        raise ValueError("numerical policy independent SQL identity changed")
    return [
        n
        for n, t in zip(oracle.names, oracle.types)
        if n in fields and t == "double"
    ]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin_export(record, directory, query):
    """Accept only this worker's paths and record immutable content identity."""
    if (
        not isinstance(record, dict)
        or type(record.get("repetition")) is not int
    ):
        raise ValueError("invalid answer repetition")
    repetition = record["repetition"]
    if (
        repetition < 0
        or not isinstance(query, str)
        or not query.startswith("q")
        or not query[1:].isdigit()
    ):
        raise ValueError("invalid answer identity")
    directory = Path(directory).resolve()
    out = dict(record)
    for key, suffix in (("path", ".parquet"), ("schema_path", ".schema.json")):
        expected = directory / f"{query}-{repetition}{suffix}"
        if (
            not isinstance(record.get(key), str)
            or Path(record[key]).resolve() != expected
        ):
            raise ValueError("answer path is not owned by this worker")
        out[key] = str(expected)
        out[key + "_sha256"] = sha256(expected)
    out.update(query=query, format=FORMAT)
    return out


@dataclass
class Answer:
    """One complete answer: logical column types and a PyArrow table.

    Comparisons stay columnar. A full-tier answer can hold ten million rows,
    which as Python tuples took tens of GB and most of an hour to check.
    """

    names: list[str]
    types: list[str]
    table: Any
    original_schema: dict

    @property
    def height(self):
        return self.table.num_rows


def _logical_type(dtype, pa):
    # Arrow's large-string offset width is a transport detail. Preserve all
    # integer widths, decimal precision/scale, and temporal units/timezones.
    # Arrow decimal32/64/128 storage widths can represent the same logical
    # precision and scale; the original receipt still records their storage.
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "string"
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return "binary"
    if pa.types.is_decimal(dtype):
        return f"decimal({dtype.precision}, {dtype.scale})"
    return str(dtype)


def _pinned_schema(record):
    """Check content hashes and the sidecar identity; return it and names."""
    if record.get("format") != FORMAT:
        raise ValueError("missing complete answer format")
    for key in ("path", "schema_path"):
        if sha256(record[key]) != record.get(key + "_sha256"):
            raise ValueError("answer content hash changed")
    schema = json.loads(Path(record["schema_path"]).read_text())
    if (
        schema.get("format") != FORMAT
        or schema.get("query") != record.get("query")
        or type(schema.get("repetition")) is not int
        or schema["repetition"] != record.get("repetition")
        or type(schema.get("rows")) is not int
        or schema["rows"] < 0
    ):
        raise ValueError("answer schema identity mismatch")
    columns = schema.get("columns")
    if not isinstance(columns, list) or any(
        not isinstance(c, dict) or not isinstance(c.get("name"), str)
        for c in columns
    ):
        raise ValueError("invalid answer schema columns")
    names = [c["name"] for c in columns]
    if len(set(names)) != len(names):
        raise ValueError("duplicate answer column names")
    return schema, names


def _check_parquet_schema(arrow_schema, height, schema, names):
    import pyarrow as pa
    from metal_numerical_audit import _physical_dtype_matches

    if names != arrow_schema.names or schema["rows"] != height:
        raise ValueError("Parquet names/height differ from original schema")
    for field, column in zip(arrow_schema, schema["columns"]):
        if "arrow" in column:
            valid = column["arrow"] == str(field.type)
        else:
            valid = _physical_dtype_matches(column.get("dtype"), field.type, pa)
        if not valid:
            raise ValueError(
                f"Parquet dtype differs from original schema: {field.name}"
            )


def verify_answer(record):
    """Every identity check of load_answer, reading only the Parquet footer.

    The content hash pins the bytes, so an answer whose bytes were already
    compared needs no second read. Returns the original schema sidecar.
    """
    import pyarrow.parquet as pq

    schema, names = _pinned_schema(record)
    with pq.ParquetFile(record["path"]) as parquet:
        _check_parquet_schema(
            parquet.schema_arrow, parquet.metadata.num_rows, schema, names
        )
    return schema


def load_answer(record):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema, names = _pinned_schema(record)
    table = pq.read_table(record["path"])
    _check_parquet_schema(table.schema, table.num_rows, schema, names)
    types = [_logical_type(f.type, pa) for f in table.schema]
    # Temporal values compare as their physical integers, which keeps every
    # unit exact (Python datetime used to lose nanoseconds here).
    for position, field in enumerate(table.schema):
        if pa.types.is_temporal(field.type):
            storage = (
                pa.int32() if pa.types.is_date32(field.type)
                or pa.types.is_time32(field.type) else pa.int64()
            )
            table = table.set_column(
                position, field.name, table.column(position).cast(storage)
            )
    return Answer(names, types, table, schema)


def _check_tolerance_fields(tolerant, expected):
    if any(
        name not in expected.names
        or expected.types[expected.names.index(name)] != "double"
        for name in tolerant
    ):
        raise ValueError("tolerance requires a declared Float64 result column")


def compare_answers(actual, expected, *, tolerant_columns=()):
    """Compare complete schemas and unordered cells, with field-level policy."""
    if actual.names != expected.names or actual.types != expected.types:
        return {"status": "different", "reason": "schema_mismatch"}
    tolerant = list(tolerant_columns)
    _check_tolerance_fields(tolerant, expected)
    with _Engine() as engine:
        return engine.equal(actual.table, expected.table, tolerant)


class OracleCheck:
    """Compare answers against one independent full relation.

    The rank order and LIMIT pools are built once, on the first answer, and
    reused for every later answer of the same query.
    """

    def __init__(self, oracle, tolerant_columns=()):
        self.oracle = oracle
        self.tolerant = list(tolerant_columns)
        self._engine = None
        self._prepared = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._engine is not None:
            self._engine.close()
        self._engine = self._prepared = None

    def _spec(self):
        oracle = self.oracle
        spec = oracle.original_schema.get("oracle")
        if not isinstance(spec, dict):
            return None
        width, order = spec.get("width"), spec.get("order")
        limit, offset = spec.get("limit"), spec.get("offset")
        if (
            type(width) is not int
            or width < 0
            or not isinstance(order, list)
            or len(oracle.names) != width + len(order)
            or any(
                not isinstance(o, dict)
                or type(o.get("descending")) is not bool
                or type(o.get("nulls_first")) is not bool
                for o in order
            )
            or (limit is not None and (type(limit) is not int or limit < 0))
            or type(offset) is not int
            or offset < 0
        ):
            raise ValueError("invalid independent selection metadata")
        return width, order, limit, offset

    def compare(self, actual):
        """Full relation membership and exact ordered ranks, including hidden
        keys.

        Ties can choose different rows. Match each output position only
        against rows with the independently established rank at that position.
        """
        spec = self._spec()
        if spec is None:
            return {
                "status": "unverified",
                "reason": "missing independent relation metadata",
            }
        width, order, limit, offset = spec
        oracle = self.oracle
        names, types = oracle.names[:width], oracle.types[:width]
        if actual.names != names or actual.types != types:
            return {"status": "different", "reason": "schema_mismatch"}
        if any(
            n not in names or types[names.index(n)] != "double"
            for n in self.tolerant
        ):
            raise ValueError(
                "tolerance requires a declared Float64 result column"
            )
        count = max(0, oracle.height - offset)
        if limit is not None:
            count = min(count, limit)
        if actual.height != count:
            return {
                "status": "different",
                "reason": "selection_row_count_mismatch",
            }
        if self._engine is None:
            self._engine = _Engine()
        engine = self._engine
        if not order and limit is None and offset == 0:
            return engine.equal(
                actual.table,
                oracle.table.select(list(range(width))),
                self.tolerant,
            )
        if self._prepared is None:
            self._prepared = engine.prepare_pool(
                oracle.table, width, order, offset, count
            )
        groups = self._prepared
        result = engine.subset(actual.table, groups, self.tolerant)
        if result["status"] == "equal" and order:
            result["reason"] = "valid_selection_and_order"
        return result


def compare_selection(actual, oracle, *, tolerant_columns=()):
    """Full relation membership and exact ordered ranks, including hidden keys.

    One-off form of OracleCheck(oracle, tolerant_columns).compare(actual).
    """
    with OracleCheck(oracle, tolerant_columns) as check:
        return check.compare(actual)


# --- columnar comparison -------------------------------------------------------
#
# Semantics are those of metal_numerical_compare.compare_rows (whole relations)
# and the injective LIMIT/rank pool matching that used to run over Python rows:
#   * Rows fall into buckets by every non-tolerant column. Equality is exact,
#     with nulls equal to nulls, NaN equal to NaN and -0.0 equal to 0.0.
#   * Whole relations need the same buckets with the same multiplicity;
#     selections need each output bucket to fit inside the oracle's bucket of
#     the same rank group.
#   * Tolerant Float64 fields must then admit a perfect matching inside each
#     bucket under abs(a-b) <= max(atol, rtol*max(|a|,|b|)). A bucket with
#     one row per side is checked in SQL; larger buckets use the bipartite
#     matcher on just their rows.
#   * Each bucket costs rows x candidates matching pairs, charged in the order
#     the row-based check visited buckets (first occurrence in the oracle for
#     whole relations, in the output for selections, per rank group). Passing
#     1,000,000 pairs is "unverified", exactly as before.

PAIR_BUDGET = 1_000_000


def _q(name):
    return '"' + name.replace('"', '""') + '"'


def _float_literal(value):
    return f"CAST('{float(value)!r}' AS DOUBLE)"


def _canonical(column):
    """Physical forms that DuckDB reads with the same logical value."""
    import pyarrow as pa

    dtype = column.type
    if pa.types.is_dictionary(dtype):
        column = column.cast(dtype.value_type)
        dtype = column.type
    if pa.types.is_decimal(dtype):
        if dtype.precision <= 38:
            if not pa.types.is_decimal128(dtype):
                column = column.cast(pa.decimal128(dtype.precision, dtype.scale))
        else:
            # Same scale on both sides (types match), so text is exact.
            column = column.cast(pa.large_string())
    elif pa.types.is_float16(dtype):
        column = column.cast(pa.float32())
    return column


def _is_nested(dtype):
    import pyarrow as pa

    return pa.types.is_nested(dtype)


def _is_struct(dtype):
    import pyarrow as pa

    return pa.types.is_struct(dtype)


def _key_sql(column, dtype, prefix):
    """(select expressions, group keys) giving exact-token equality."""
    import pyarrow as pa

    c = _q(column)
    if pa.types.is_floating(dtype):
        tag, value = f"{prefix}t", f"{prefix}v"
        return [
            f"CASE WHEN {c} IS NULL THEN 0 WHEN isnan({c}) THEN 1 ELSE 2 END"
            f" AS {tag}",
            f"CASE WHEN {c} = 0 THEN 0 WHEN isnan({c}) THEN NULL ELSE {c} END"
            f" AS {value}",
        ], [tag, value]
    return [f"{c} AS {prefix}v"], [f"{prefix}v"]


def _close_sql(a, p):
    rtol, atol = _float_literal(DEFAULT_RTOL), _float_literal(DEFAULT_ATOL)
    return (
        f"(({a} IS NULL AND {p} IS NULL) OR ({a} IS NOT NULL AND {p} IS NOT"
        f" NULL AND CASE WHEN isnan({a}) OR isnan({p}) THEN isnan({a}) AND"
        f" isnan({p}) WHEN isinf({a}) OR isinf({p}) THEN {a} = {p} ELSE"
        f" abs({a} - {p}) <= greatest({atol}, {rtol} * greatest(abs({a}),"
        f" abs({p}))) END))"
    )


class _Engine:
    """One DuckDB connection for one oracle at a time."""

    def __init__(self):
        import duckdb
        import tempfile

        self._spill = tempfile.mkdtemp(prefix="suite-answer-check-")
        self.con = duckdb.connect(
            config={
                "threads": min(8, os.cpu_count() or 1),
                "memory_limit": "8GB",
                "temp_directory": self._spill,
                "preserve_insertion_order": False,
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        import shutil

        if self.con is not None:
            self.con.close()
            self.con = None
        shutil.rmtree(self._spill, ignore_errors=True)

    def _sql(self, sql, fetch="all"):
        """Run one statement and fetch its whole result (a connection holds
        one open result at a time)."""
        import duckdb

        try:
            result = self.con.execute(sql)
            return result.fetchnumpy() if fetch == "numpy" else (
                result.fetchall()
            )
        except duckdb.Error as error:
            raise ValueError(f"columnar answer comparison failed: {error}")

    def _register(self, name, table, groups=None):
        """Columns c0.. plus row index i and rank group g."""
        import numpy as np
        import pyarrow as pa

        columns = [_canonical(c) for c in table.columns]
        height = table.num_rows
        extra = [pa.array(np.arange(height, dtype=np.int64))]
        names = [f"c{k}" for k in range(len(columns))] + ["i"]
        if groups is not None:
            extra.append(pa.array(groups, type=pa.int64()))
            names.append("g")
        relation = pa.Table.from_arrays(columns + extra, names=names)
        self.con.register(name, relation)
        return relation.schema

    def _unregister(self, *names):
        for name in names:
            self.con.unregister(name)

    def _release(self, *registered):
        """Drop one comparison's relations; the oracle pool stays."""
        if self.con is None:
            return
        self._unregister(*registered)
        self._sql("DROP VIEW IF EXISTS u")
        self._sql("DROP TABLE IF EXISTS b")

    @staticmethod
    def _nested_values(table, positions):
        for k in positions:
            column = table.column(k)
            if _is_nested(column.type) and column.null_count < len(column):
                kind = "dict" if _is_struct(column.type) else "list"
                raise ValueError(
                    f"exact comparison value is unhashable: {kind}"
                )

    # Whole relations: metal_numerical_compare.compare_rows.
    def equal(self, actual, expected, tolerant):
        if actual.num_rows != expected.num_rows:
            return {"status": "different", "reason": "row_count_mismatch"}
        if actual.num_rows == 0:
            return {"status": "equal", "reason": None}
        names = [f.name for f in expected.schema]
        tol = [k for k, n in enumerate(names) if n in set(tolerant)]
        exact = [k for k in range(len(names)) if k not in tol]
        self._nested_values(actual, exact)
        self._nested_values(expected, exact)
        schema = self._register("a", actual)
        self._register("p", expected)
        try:
            return self._buckets(schema, exact, tol, subset=False, grouped=False)
        finally:
            self._release("a", "p")

    # Selections: rank groups and their oracle pools, built once per oracle.
    def prepare_pool(self, oracle, width, order, offset, count):
        import numpy as np
        import pyarrow as pa

        if count == 0:
            return {"count": 0}
        # Exact keys and ranks must be hashable values, as Python tokens were.
        self._nested_values(oracle, range(oracle.num_columns))
        if not order:
            self._register("o", oracle.select(list(range(width))))
            self._sql(
                "CREATE OR REPLACE TEMP TABLE pool AS SELECT *, 0 AS g FROM o"
            )
            self._unregister("o")
            return {"count": count, "groups": np.zeros(count, np.int64)}
        schema = self._register("o", oracle)
        selects, keys, ordering = [], [], []
        for j, rule in enumerate(order):
            column = f"c{width + j}"
            dtype = schema.field(column).type
            c = _q(column)
            direction = "DESC" if rule["descending"] else "ASC"
            selects.append(f"({c} IS NULL) AS r{j}n")
            keys.append(f"r{j}n")
            ordering.append(
                f"r{j}n {'DESC' if rule['nulls_first'] else 'ASC'}"
            )
            # DuckDB sorts NaN above every non-NaN numeric value.
            if pa.types.is_floating(dtype):
                selects.append(f"isnan({c}) AS r{j}f")
                keys.append(f"r{j}f")
                ordering.append(f"r{j}f {direction} NULLS LAST")
                selects.append(
                    f"CASE WHEN {c} = 0 THEN 0 WHEN isnan({c}) THEN NULL"
                    f" ELSE {c} END AS r{j}v"
                )
            else:
                selects.append(f"{c} AS r{j}v")
            keys.append(f"r{j}v")
            ordering.append(f"r{j}v {direction} NULLS LAST")
        key_list, order_sql = ", ".join(keys), ", ".join(ordering)
        self._sql(
            "CREATE OR REPLACE TEMP TABLE ranks AS SELECT *, row_number()"
            f" OVER (ORDER BY {order_sql}) - 1 AS g FROM (SELECT {key_list},"
            f" count(*) AS n FROM (SELECT {', '.join(selects)} FROM o) GROUP"
            f" BY {key_list} ORDER BY {order_sql} LIMIT {offset + count})"
        )
        sizes = (
            self._sql("SELECT n FROM ranks ORDER BY g", "numpy")["n"]
            .astype(np.int64)
        )
        # Position p of the sorted relation holds the rank whose rows end
        # after p; OFFSET/LIMIT select positions, so ties at the edges keep
        # their whole pool.
        groups = np.searchsorted(
            np.cumsum(sizes),
            offset + np.arange(count, dtype=np.int64),
            side="right",
        ).astype(np.int64)
        values = [f"c{k}" for k in range(width)]
        picked = ", ".join([f"x.{c}" for c in values] + ["x.i", "r.g"])
        inner = ", ".join(values + ["i"] + selects)
        join = " AND ".join(f"x.{k} IS NOT DISTINCT FROM r.{k}" for k in keys)
        self._sql(
            f"CREATE OR REPLACE TEMP TABLE pool AS SELECT {picked} FROM"
            f" (SELECT {inner} FROM o) x JOIN ranks r ON {join} WHERE r.g >="
            f" {int(groups[0])}"
        )
        self._unregister("o")
        self._sql("DROP TABLE ranks")
        return {"count": count, "groups": groups}

    def subset(self, actual, prepared, tolerant):
        if prepared["count"] == 0:
            return {"status": "equal", "reason": "valid_relation_membership"}
        names = [f.name for f in actual.schema]
        tol = [k for k, n in enumerate(names) if n in set(tolerant)]
        exact = [k for k in range(len(names)) if k not in tol]
        self._nested_values(actual, exact)
        schema = self._register("a", actual, prepared["groups"])
        try:
            return self._buckets(schema, exact, tol, subset=True, grouped=True)
        finally:
            self._release("a")

    def _buckets(self, schema, exact, tol, *, subset, grouped):
        width = len(exact) + len(tol)
        values = "".join(f", c{k}" for k in range(width))
        pool = "pool" if subset else "p"
        group = "g" if grouped else "0 AS g"
        self._sql(
            f"CREATE OR REPLACE TEMP VIEW u AS SELECT 0 AS s, {group}, i"
            f"{values} FROM a UNION ALL SELECT 1 AS s, {group}, i{values}"
            f" FROM {pool}"
        )
        selects, keys = [], ["g"]
        for k in exact:
            sql, key = _key_sql(f"c{k}", schema.field(f"c{k}").type, f"k{k}")
            selects += sql
            keys += key
        tolerant = "".join(f", c{k}" for k in tol)
        inner = (
            f"SELECT s, g, i{''.join(', ' + s for s in selects)}{tolerant}"
            " FROM u"
        )
        pairs = "".join(
            f", min(c{k}) FILTER (WHERE s = 0) AS a{k}, min(c{k}) FILTER"
            f" (WHERE s = 1) AS p{k}"
            for k in tol
        )
        group_by = ", ".join(keys)
        self._sql(
            "CREATE OR REPLACE TEMP TABLE b AS SELECT g, count(*) FILTER"
            " (WHERE s = 0) AS na, count(*) FILTER (WHERE s = 1) AS np,"
            " min(i) FILTER (WHERE s = 0) AS fa, min(i) FILTER (WHERE s = 1)"
            f" AS fp{pairs} FROM ({inner}) GROUP BY {group_by}"
        )
        ok = " AND ".join(_close_sql(f"a{k}", f"p{k}") for k in tol) or "TRUE"
        single = "(na = 1 AND np = 1)"
        missing, uneven, over, bad, multi, cost = self._sql(
            "SELECT count(*) FILTER (WHERE na = 0 OR np = 0), count(*) FILTER"
            " (WHERE na <> np), count(*) FILTER (WHERE na > np), count(*)"
            f" FILTER (WHERE {single} AND NOT ({ok})), count(*) FILTER"
            f" (WHERE na > 0 AND na <= np AND NOT {single}), (SELECT max(c)"
            " FROM (SELECT sum(na * np) AS c FROM b WHERE na > 0 GROUP BY g))"
            " FROM b"
        )[0]
        if subset:
            membership = "invalid_relation_membership_or_multiplicity"
            unmatched, equal = "invalid_relation_values", (
                "valid_relation_membership"
            )
            if not tol:
                if over:
                    return {"status": "different", "reason": membership}
                return {"status": "equal", "reason": equal}
        else:
            if missing:
                return {"status": "different", "reason": "exact_key_mismatch"}
            if uneven:
                return {
                    "status": "different",
                    "reason": "exact_key_multiplicity_mismatch",
                }
            unmatched, equal, membership = "no_tolerant_perfect_matching", None, None
            if not tol:
                return {"status": "equal", "reason": equal}
        if not over and not bad and not multi and (cost or 0) <= PAIR_BUDGET:
            return {"status": "equal", "reason": equal}
        # Visit buckets in the row-based order and stop at the first that
        # fails membership, exhausts its group's pair budget, or has no
        # tolerant matching.
        first = "fa" if subset else "fp"
        events = self._sql(
            f"SELECT g, f, na, np, cum, ok FROM (SELECT g, {first} AS f, na,"
            f" np, na * np AS cost, {ok} AS ok, sum(na * np) OVER (PARTITION"
            f" BY g ORDER BY {first} ROWS BETWEEN UNBOUNDED PRECEDING AND"
            " CURRENT ROW) AS cum FROM b WHERE na > 0) WHERE cum - cost <="
            f" {PAIR_BUDGET} AND (na > np OR cum > {PAIR_BUDGET} OR NOT"
            f" {single} OR NOT ok) ORDER BY g, f"
        )
        needed = [f for g, f, na, np_, cum, good in events if not (
            na == 1 and np_ == 1
        ) and na <= np_ and cum <= PAIR_BUDGET]
        rows = self._bucket_rows(needed, keys, inner, tol, subset)
        for g, f, na, np_, cum, good in events:
            if na > np_:
                return {"status": "different", "reason": membership}
            if cum > PAIR_BUDGET:
                return {
                    "status": "unverified",
                    "reason": "candidate_pair_budget_exceeded",
                }
            if na == 1 and np_ == 1:
                if not good:
                    return {"status": "different", "reason": unmatched}
                continue
            left, right = rows[f]
            matched, _, _ = _perfect_match(
                left if subset else right,
                right if subset else left,
                range(len(tol)),
                DEFAULT_RTOL,
                DEFAULT_ATOL,
                PAIR_BUDGET,
            )
            if not matched:
                return {"status": "different", "reason": unmatched}
        return {"status": "equal", "reason": equal}

    def _bucket_rows(self, needed, keys, inner, tol, subset):
        """Tolerant values of the given buckets: {f: (actual, oracle) rows}."""
        import pyarrow as pa

        if not needed:
            return {}
        self.con.register("need", pa.table({"f": pa.array(needed, pa.int64())}))
        side = 0 if subset else 1
        values = ", ".join(f"c{k}" for k in tol)
        try:
            found = self._sql(
                f"SELECT f, s, {values} FROM (SELECT s, i, {values}, min(i)"
                f" FILTER (WHERE s = {side}) OVER (PARTITION BY"
                f" {', '.join(keys)}) AS f FROM ({inner})) WHERE f IN (SELECT"
                " f FROM need) ORDER BY f, s, i"
            )
        finally:
            self._unregister("need")
        rows = {f: ([], []) for f in needed}
        for f, s, *cells in found:
            rows[f][s].append(tuple(cells))
        return rows
