#!/usr/bin/env bash
# PREPARED ONLY: do not run until the Mac's live engine gate and initial typed
# compile have completed and the parent explicitly starts the suite baseline.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"

HARNESS=/Users/randyzwitch/github/dataframe_mojo-metal-868-coverage-audit
ENGINE="$HARNESS"
MAIN=/Users/randyzwitch/github/dataframe_mojo-metal-868
DATA=/Users/randyzwitch/GitHub/dataframe_mojo-metal-868_benchdata
OUT="$HARNESS/build/metal-868-dev-coverage-a52334e7"
ENGINE_PIN=a52334e758b7138c547883eb776d744442a1a0db
HARNESS_HEAD=50137ecc2e27f76298559a25e0aa4ecca31d4e96
LOCK_SHA=098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277
MANIFEST_SHA=8e16cbd03a3d80315967267358391f8b1a84c9f0bc8afc1aa0cd0de9fd2105ac
PARQUET_SHA=8508c1d8a7c8e86c335ddbe6db0f76288dd69fca927cc573156e6248b379cf51

# Pin data and native-library resolution, rather than relying on worktree lookup.
export DATAFRAME_BENCH_DATA="$DATA"
export DATAFRAME_PARQUET_LIBRARY="$MAIN/build/dfparquet/libdfparquet.dylib"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[ -d "$DATA" ] || fail "missing prepared dataset: $DATA"
# Assert all 77 dev input files directly; do not call datagen helpers here,
# since those deliberately generate absent files.
python3 - "$DATA" <<'PYDATA'
from pathlib import Path
import sys
root = Path(sys.argv[1])
expected = []
for k, nas, sorted_rows in [(100, 0, 0), (10, 0, 0), (2, 0, 0), (100, 5, 0), (100, 0, 1)]:
    name = f"G1_{1_000_000:.0e}_{k:.0e}_{nas}_{sorted_rows}.parquet".replace("+", "")
    expected.append(root / "h2o" / name)
rows = 1_000_000
for nas in (0, 5):
    tag = f"{rows:.0e}_NA_0_{nas}".replace("+", "")
    for suffix in ("0_0", f"{max(10, rows // 1_000_000):.0e}_0", f"{max(100, rows // 1_000):.0e}_0", f"{rows:.0e}_0"):
        expected.append(root / "h2o" / f"J1_{tag}_{suffix}.parquet".replace("+", ""))
pdsh = "customer lineitem nation orders part partsupp region supplier".split()
tpcds = "call_center catalog_page catalog_returns catalog_sales customer customer_address customer_demographics date_dim household_demographics income_band inventory item promotion reason ship_mode store store_returns store_sales time_dim warehouse web_page web_returns web_sales web_site".split()
for suite, tables in (("pdsh", pdsh), ("tpcds", tpcds)):
    for variant in ("sf0.1", "sf0.1_decimal"):
        expected.extend(root / suite / variant / f"{table}.parquet" for table in tables)
missing = [str(path) for path in expected if not path.is_file()]
if len(expected) != 77 or missing:
    raise SystemExit("prepared dev dataset incomplete; missing=" + repr(missing[:40]))
print(f"prepared dev inputs present: {len(expected)} files")
PYDATA
[ -f "$HARNESS/pixi.toml" ] && [ -f "$MAIN/pixi.toml" ] || fail 'Pixi manifest missing'
[ "$(shasum -a 256 "$HARNESS/pixi.toml" | awk '{print $1}')" = "$MANIFEST_SHA" ] || fail 'harness pixi.toml differs from prepared environment'
[ "$(shasum -a 256 "$MAIN/pixi.toml" | awk '{print $1}')" = "$MANIFEST_SHA" ] || fail 'main pixi.toml differs from prepared environment'
[ -f "$HARNESS/pixi.lock" ] && [ -f "$MAIN/pixi.lock" ] || fail 'pixi.lock missing'
[ "$(shasum -a 256 "$HARNESS/pixi.lock" | awk '{print $1}')" = "$LOCK_SHA" ] || fail 'harness pixi.lock differs from prepared environment'
[ "$(shasum -a 256 "$MAIN/pixi.lock" | awk '{print $1}')" = "$LOCK_SHA" ] || fail 'main pixi.lock differs from prepared environment'
[ "$(shasum -a 256 "$HARNESS/benchmarks/tools/bench_metal_suites.py" | awk '{print $1}')" = 8b177c151366fb7416e42a6485a8cbc754ce54817ef4d84de19d4674751b40ed ] || fail 'coverage driver hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/tools/bench_apple_performance.py" | awk '{print $1}')" = f2cdb0c2a94e9534ec35f32e57e0b25b61d0d29300deba542508562a4e02cad5 ] || fail 'performance driver hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/tools/bench_suites.py" | awk '{print $1}')" = ad1272f7b058d707a7da95f78d89878f97689c9e22271e7d98a6248f40ad1298 ] || fail 'answer-check catalog hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/suites/engines.py" | awk '{print $1}')" = c1b79b0d4fce9ae97d1a96800505bc3b5a384ec255651868c7ce5678fe906af1 ] || fail 'DuckDB engine hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/tools/metal_query_report.py" | awk '{print $1}')" = 04c1e1666b537f2577794584adf09ebdbde77fb8a0d5cb7e69b3e6960eb138e6 ] || fail 'coverage verifier hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/tools/metal_answer_exports.py" | awk '{print $1}')" = e1b6afc01ea8d137cec6a1050ae4cc184323a1b083d0e221bbf8661c9c1673f8 ] || fail 'answer export verifier hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/tools/metal_materialization_adapter.py" | awk '{print $1}')" = ff270eb2535089646659ee5cbe652a03ba85421866906e87e85f587b988bbe41 ] || fail 'materialization adapter hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/metal_suite_diagnostics.mojo" | awk '{print $1}')" = eea1ba691fe3302977540461c6e181a306bd1231fe66f7483f42a6edf3554329 ] || fail 'suite diagnostics hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/metal_suite_answer_exports.mojo" | awk '{print $1}')" = 3cffcc99c04b3f3703c3c2f8bcda724f5f9d32708fb687fc0fdd32d00963d4a5 ] || fail 'answer export source hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/tools/metal_case_metadata.json" | awk '{print $1}')" = 9bae080d8b3ba576448fed603a8c7058224b7b5d9b46ba146c030f1dd4329176 ] || fail 'case metadata hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/metal_coverage_audit.mojo" | awk '{print $1}')" = aa0f80c209314165bac49204a2ddc4ae2f24ede0937b659c6d6d195db2dc6584 ] || fail 'runtime audit hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/metal_suite_common.mojo" | awk '{print $1}')" = 734ea2f1c629e858c1a84f3db5f6493d7b82c67ec7a4f756fe169b615a2733f6 ] || fail 'suite common hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/metal_suite_timing.mojo" | awk '{print $1}')" = 3e4d3c150bb5c63fe320c15cc135da4b8350b824d3472a93aee45df5af971916 ] || fail 'timing source hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/suites/h2o.mojo" | awk '{print $1}')" = bba2b834539145cbd45d1751bf72ed172e94672f5328e64352c32015fbbae05c ] || fail 'H2O source hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/suites/pdsh.mojo" | awk '{print $1}')" = f27444f0de2e128f05ed457b65fa1d1fc93ebae1f3f7eabe1dd9e306819b990e ] || fail 'PDS-H source hash changed'
[ "$(shasum -a 256 "$HARNESS/benchmarks/suites/tpcds.mojo" | awk '{print $1}')" = 14d4f9e0288dcadbb01f4c5aae341951ebb60dd011af478a35010688a8c74e68 ] || fail 'TPC-DS source hash changed'
[ -f "$DATAFRAME_PARQUET_LIBRARY" ] || fail 'native Parquet library missing'
[ "$(shasum -a 256 "$DATAFRAME_PARQUET_LIBRARY" | awk '{print $1}')" = "$PARQUET_SHA" ] || fail 'native Parquet library differs from prepared build'
[ "$(git -C "$HARNESS" rev-parse HEAD)" = "$HARNESS_HEAD" ] || fail 'coverage harness checkout moved'
[ "$(git -C "$HARNESS" rev-parse origin/main)" = "$ENGINE_PIN" ] || fail 'origin/main moved from reviewed pin'
git -C "$HARNESS" merge-base --is-ancestor "$ENGINE_PIN" HEAD || fail 'current origin/main is not an ancestor of the committed harness'
git -C "$HARNESS" diff --quiet "$ENGINE_PIN" HEAD -- dataframe benchmarks/suites benchmarks/tools/bench_suites.py benchmarks/suites/engines.py || fail 'engine or query catalog differs from origin/main'
[ -z "$(git -C "$HARNESS" status --porcelain)" ] || fail 'coverage harness tree is dirty' 

[ ! -e "$OUT" ] || fail 'refusing to overwrite prior receipt directory'
mkdir -p "$OUT"

# Coverage only: no performance-grid commands or timing workers. Preserve the
# complete typed CPU/Metal answer artifacts from the independent coverage pass.
SUITES=h2o_groupby,h2o_join,pdsh,tpcds

set +e
pixi run --locked --manifest-path "$MAIN/pixi.toml" -e oracle python3 \
  "$HARNESS/benchmarks/tools/bench_metal_suites.py" \
  --suites "$SUITES" --scale dev --reps 3 --threads 8 --export-answers \
  --output "$OUT/coverage.json" > "$OUT/coverage-driver.log" 2>&1
COVERAGE_RC=$?
set -e
cat "$OUT/coverage-driver.log"

# The validator checks the complete expected grid, exact-compare success or
# explicitly unsupported cases, every per-collect runtime audit, pinned source
# provenance, complete typed-answer artifacts (including on-disk hash checks),
# and that all input files stayed inside the prepared data tree and retained
# their recorded hashes. It deliberately makes no timing claim.
python3 - "$OUT/coverage.json" "$DATA" "$COVERAGE_RC" "$HARNESS_HEAD" "$HARNESS" <<'PY'
import ast
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
receipt_path = Path(sys.argv[1])
data_root = Path(sys.argv[2]).resolve()
runner_rc = int(sys.argv[3])
expected_head = sys.argv[4]
harness_root = Path(sys.argv[5]).resolve()
sys.path.insert(0, str(harness_root / "benchmarks/tools"))
from metal_answer_exports import file_sha256
receipt = json.loads(receipt_path.read_text())
if receipt.get("format") != "dataframe-metal-suite-coverage-v1":
    raise SystemExit("unexpected coverage receipt format")
provenance = receipt.get("provenance", {})
if provenance.get("engine_revision") != expected_head:
    raise SystemExit("coverage engine revision differs from pinned harness")
if provenance.get("engine_status") != "":
    raise SystemExit("dataframe engine was dirty during coverage execution")
if provenance.get("scale") != "dev" or provenance.get("threads") != 8 or provenance.get("repetitions") != 3:
    raise SystemExit("coverage configuration differs from the prepared plan")
if provenance.get("answer_exports_requested") is not True:
    raise SystemExit("coverage receipt lacks requested typed answer exports")
if provenance.get("implementation") != "mojo":
    raise SystemExit("coverage did not use Mojo workers")

# Derive the exact selected suite/variant/query tuples from the pinned catalog.
catalog_path = harness_root / "benchmarks/tools/bench_suites.py"
catalog = ast.parse(catalog_path.read_text())
assignments = {}
for node in catalog.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"SUITES", "VARIANTS"}:
                assignments[target.id] = node.value
if set(assignments) != {"SUITES", "VARIANTS"}:
    raise SystemExit("cannot locate suite catalog assignments")

def dict_field(node, field):
    if not isinstance(node, ast.Dict):
        raise SystemExit("suite catalog shape changed")
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value == field:
            return value
    raise SystemExit(f"suite catalog lacks {field}")

def suite_queries(node):
    if not isinstance(node, ast.ListComp) or len(node.generators) != 1:
        raise SystemExit("suite query catalog is not the reviewed q-range form")
    generator = node.generators[0]
    if not isinstance(generator.target, ast.Name):
        raise SystemExit("suite query range target changed")
    if not isinstance(generator.iter, ast.Call) or not isinstance(generator.iter.func, ast.Name) or generator.iter.func.id != "range":
        raise SystemExit("suite query range changed")
    bounds = [ast.literal_eval(arg) for arg in generator.iter.args]
    if len(bounds) not in {1, 2} or not isinstance(node.elt, ast.JoinedStr):
        raise SystemExit("suite query range form changed")
    pieces = node.elt.values
    if len(pieces) != 2 or not isinstance(pieces[0], ast.Constant) or not isinstance(pieces[0].value, str) or not isinstance(pieces[1], ast.FormattedValue) or not isinstance(pieces[1].value, ast.Name) or pieces[1].value.id != generator.target.id:
        raise SystemExit("suite query template changed")
    return [pieces[0].value + str(i) for i in range(*bounds)]

suite_node = assignments["SUITES"]
variant_catalog = ast.literal_eval(assignments["VARIANTS"])
selected_suites = ("h2o_groupby", "h2o_join", "pdsh", "tpcds")
expected_cases = set()
for suite in selected_suites:
    suite_queries_node = dict_field(dict_field(suite_node, suite), "queries")
    queries = suite_queries(suite_queries_node)
    variants = variant_catalog.get(suite)
    if not isinstance(variants, dict) or not variants:
        raise SystemExit(f"suite catalog lacks variants for {suite}")
    expected_cases.update((suite, variant, query) for variant in variants for query in queries)
expected_count = len(expected_cases)
if expected_count != 302:
    raise SystemExit(f"pinned suite catalog case count changed: {expected_count}/302")

cases = receipt.get("cases", [])
if len(cases) != expected_count:
    raise SystemExit(f"incomplete coverage grid: {len(cases)}/{expected_count}")
identities = [(c.get("suite"), c.get("variant"), c.get("query")) for c in cases]
if len(set(identities)) != expected_count or set(identities) != expected_cases:
    missing = sorted(expected_cases - set(identities), key=repr)[:20]
    extra = sorted(set(identities) - expected_cases, key=repr)[:20]
    raise SystemExit("coverage grid differs from pinned suite catalog: " + repr({"missing": missing, "extra": extra}))
allowed = {"ok", "unsupported", "reference_unsupported"}
bad = [c for c in cases if c.get("status") not in allowed]
if bad:
    detail = [(c.get("suite"), c.get("variant"), c.get("query"), c.get("status"), c.get("reason")) for c in bad[:30]]
    raise SystemExit("coverage failures require review: " + repr(detail))
output_root = receipt_path.parent.resolve(strict=True)
answers_root = (output_root / "answers").resolve(strict=True)
try:
    answers_root.relative_to(output_root)
except ValueError:
    raise SystemExit("answer artifact directory escaped the coverage receipt directory")
artifact_paths = set()
for case in cases:
    if case.get("status") != "ok":
        continue
    audit = case.get("runtime_adjudication", {})
    if (
        audit.get("status") != "complete"
        or audit.get("observed_calls", 0) < case.get("collections", 0)
        or audit.get("unknown_calls", 0)
        or audit.get("failed_calls", 0)
        or len(audit.get("events", [])) != audit.get("observed_calls", 0)
    ):
        raise SystemExit("runtime audit incomplete: " + repr((case.get("suite"), case.get("variant"), case.get("query"), audit)))
    if case.get("collections") != 3 or type(case.get("ordered_equal")) is not bool:
        raise SystemExit("exact comparison/repetition evidence missing: " + repr((case.get("suite"), case.get("variant"), case.get("query"), case.get("collections"), case.get("ordered_equal"))))
    exports = case.get("answer_exports")
    if not isinstance(exports, list):
        raise SystemExit("typed answer export records are missing: " + repr((case.get("suite"), case.get("variant"), case.get("query"))))
    export_keys = [(item.get("engine"), item.get("repetition")) for item in exports]
    export_status = {key: item.get("status") for key, item in zip(export_keys, exports)}
    required_exports = {("cpu", 0), *(('metal', i) for i in range(3))}
    if (
        len(exports) != 5
        or len(set(export_keys)) != len(exports)
        or set(export_keys) != required_exports | {("cpu_repeat", 0)}
        or any(export_status.get(key) != "written" for key in required_exports)
    ):
        raise SystemExit("complete typed CPU/Metal answers missing: " + repr((case.get("suite"), case.get("variant"), case.get("query"), export_status)))
    for item in exports:
        if (item.get("engine"), item.get("repetition")) not in required_exports:
            continue
        schema = item.get("original_schema", {})
        resolved_artifacts = {}
        if (
            not isinstance(item.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            or not isinstance(item.get("schema_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["schema_sha256"])
            or schema.get("engine") != item.get("engine")
            or schema.get("repetition") != item.get("repetition")
            or schema.get("format") != "dataframe-answer-schema-v1"
        ):
            raise SystemExit("typed answer artifact evidence is incomplete: " + repr((case.get("suite"), case.get("variant"), case.get("query"), item)))
        for field, recorded_hash in (("path", item["sha256"]), ("schema_path", item["schema_sha256"])):
            raw_path = item.get(field)
            if not isinstance(raw_path, str):
                raise SystemExit("typed answer artifact path is missing: " + repr(item))
            artifact = Path(raw_path)
            if artifact.is_symlink():
                raise SystemExit("typed answer artifact must not be a symlink: " + str(artifact))
            try:
                artifact = artifact.resolve(strict=True)
                artifact.relative_to(answers_root)
            except (OSError, ValueError):
                raise SystemExit("typed answer artifact is missing or escaped its receipt directory: " + raw_path)
            if not artifact.is_file() or artifact in artifact_paths:
                raise SystemExit("typed answer artifact path is invalid or duplicated: " + str(artifact))
            artifact_paths.add(artifact)
            resolved_artifacts[field] = artifact
            if file_sha256(artifact) != recorded_hash:
                raise SystemExit("typed answer artifact hash mismatch: " + str(artifact))
        try:
            on_disk_schema = json.loads(resolved_artifacts["schema_path"].read_text())
        except (OSError, ValueError) as error:
            raise SystemExit("typed answer schema artifact cannot be read: " + repr(error))
        if on_disk_schema != schema:
            raise SystemExit("typed answer schema artifact differs from its receipt: " + str(item["schema_path"]))

# Verify each distinct recorded file once, and reject paths outside the pinned
# input tree. This catches accidental replacement or mutation during the run.
recorded = {}
for case in cases:
    inputs = case.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise SystemExit("case lacks its prepared input manifest: " + repr((case.get("suite"), case.get("variant"), case.get("query"))))
    for item in inputs.values():
        path = Path(item["path"]).resolve()
        try:
            path.relative_to(data_root)
        except ValueError:
            raise SystemExit("case input escaped the pinned dataset root: " + str(path))
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise SystemExit("malformed input hash in coverage receipt: " + str(path))
        if path in recorded and recorded[path] != digest:
            raise SystemExit("inconsistent input hashes within the receipt: " + str(path))
        recorded[path] = digest
for path, wanted in recorded.items():
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != wanted:
        raise SystemExit("prepared input changed during coverage: " + str(path))

if runner_rc != 0:
    raise SystemExit(f"coverage driver exited {runner_rc}; inspect retained case receipts and logs")
print(f"coverage statuses: {dict(Counter(c.get('status') for c in cases))}")
print(f"runtime CPU adjudications: {sum(c.get('runtime_adjudication', {}).get('cpu_adjudicated_calls', 0) for c in cases)}; inspect per-case receipt")
print(f"prepared input files revalidated: {len(recorded)}")
print(f"complete coverage receipt: {len(cases)}/{expected_count} cases at {receipt_path}")
print(f"raw worker logs: {receipt_path.parent / 'logs'}")
print(f"complete typed answer artifacts rehashed: {len(artifact_paths)} files under {receipt_path.parent / 'answers'}")
PY
