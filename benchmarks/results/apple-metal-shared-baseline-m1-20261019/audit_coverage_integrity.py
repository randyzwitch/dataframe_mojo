"""Validate receipt integrity while preserving correctness failures as failures."""
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
    print("Preserved correctness failures requiring review: " + repr(detail))
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
    print(f"Preserved coverage driver exit status: {runner_rc}")
print(f"coverage statuses: {dict(Counter(c.get('status') for c in cases))}")
print(f"runtime CPU adjudications: {sum(c.get('runtime_adjudication', {}).get('cpu_adjudicated_calls', 0) for c in cases)}; inspect per-case receipt")
print(f"prepared input files revalidated: {len(recorded)}")
print(f"complete coverage receipt: {len(cases)}/{expected_count} cases at {receipt_path}")
print(f"raw worker logs: {receipt_path.parent / 'logs'}")
print(f"complete typed answer artifacts rehashed: {len(artifact_paths)} files under {receipt_path.parent / 'answers'}")

# Rehash every written export, including failed comparisons and CPU repeats.
all_written = set()
for case in cases:
    for item in case.get("answer_exports", []):
        if item.get("status") != "written":
            continue
        for field, digest_field in (("path", "sha256"), ("schema_path", "schema_sha256")):
            artifact = Path(item[field])
            if artifact.is_symlink():
                raise SystemExit("symlink answer artifact: " + str(artifact))
            artifact = artifact.resolve(strict=True)
            artifact.relative_to(answers_root)
            if artifact in all_written or not artifact.is_file():
                raise SystemExit("duplicate or missing written artifact: " + str(artifact))
            all_written.add(artifact)
            if file_sha256(artifact) != item[digest_field]:
                raise SystemExit("written answer artifact hash mismatch: " + str(artifact))
        if json.loads(Path(item["schema_path"]).read_text()) != item["original_schema"]:
            raise SystemExit("written schema does not match receipt")
summary = dict(
    format="metal-baseline-coverage-integrity-v1",
    receipt_path=str(receipt_path), receipt_sha256=file_sha256(receipt_path),
    integrity_verified=True, correctness_passed=not bad and runner_rc == 0,
    driver_returncode=runner_rc, cases=len(cases),
    statuses=dict(Counter(c.get("status") for c in cases)),
    input_files_rehashed=len(recorded), written_answer_files_rehashed=len(all_written),
    failures=[{k: c.get(k) for k in ("suite", "variant", "query", "status", "reason")} for c in bad],
)
output=receipt_path.parent / "coverage-integrity.json"
if output.exists():
    raise SystemExit("refusing to overwrite " + str(output))
output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
