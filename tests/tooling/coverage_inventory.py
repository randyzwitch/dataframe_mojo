"""Inventory every test entry point and enforce explicit coverage-plan assignments.

The inventory is source evidence, not proof that two cases are equivalent.
Refresh with --write after reviewing new/changed/deleted cases and the plan.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "tests"


def sources(root=ROOT):
    paths = set((root / "tests").rglob("*.mojo"))
    paths.update((root / "tests").rglob("*.py"))
    paths.update((root / "tests").rglob("*.sh"))
    paths.update((root / "native").rglob("test*.py"))
    paths.update((root / "scripts").glob("test*.py"))
    paths.update((root / "scripts").glob("check*.py"))
    paths.update((root / "scripts").glob("check*.sh"))
    paths.update((root / "scripts").glob("*smoke.sh"))
    return sorted(p for p in paths if "__pycache__" not in p.parts)


def entries(path):
    source = path.read_text()
    if path.suffix == ".sh":
        return [("@script", 1)]
    if path.suffix == ".mojo":
        functions = re.findall(r"^def ((?:test_|check_)\w+)\([^\n]*", source, re.M)
        result = [(name, source[:source.index("def " + name + "(")].count("\n") + 1) for name in functions]
        if re.search(r"^def main\(", source, re.M):
            result.append(("@main", source[:source.index("def main(")].count("\n") + 1))
        return result
    tree = ast.parse(source)
    result = []
    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name.startswith(("test_", "check_")):
                    result.append((prefix + child.name, child.lineno))
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")
    visit(tree)
    # Top-level native scripts and fixture producers have executable assertions
    # or are invoked by external harnesses. Keep the entire entry point visible.
    result.append(("@script", 1))
    return result


def inventory(plan, root=ROOT):
    rows = []
    seen = set()
    for path in sources(root):
        cases = entries(path)
        if not cases:
            continue  # reusable support modules are fingerprinted below
        relative = path.relative_to(root).as_posix()
        seen.add(relative)
        if relative not in plan["assignments"]:
            raise ValueError(f"Unmapped test source: {relative}; assign source areas in preservation.json")
        assignment = plan["assignments"][relative]
        if not assignment or not set(assignment) <= set(plan["areas"]):
            raise ValueError(f"Invalid contract assignment: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for name, line in cases:
            case_id = f"{relative}::{name}"
            contracts = plan["case_assignments"].get(case_id)
            if not contracts or not set(contracts) <= set(plan["areas"]):
                raise ValueError(f"Unmapped or invalid case: {case_id}")
            rows.append(dict(id=case_id, line=line, contracts=contracts,
                             obligation=name.removeprefix("test_").removeprefix("check_").replace("_", " "),
                             source_sha256=digest, disposition="retain"))
    stale = set(plan["assignments"]) - seen
    if stale:
        raise ValueError(f"Stale coverage assignments (review removals): {sorted(stale)}")
    ids = {row["id"] for row in rows}
    retired_ids = set()
    retired = plan.get("retired_sources", {})
    archive_info = plan.get("preservation_archive")
    if archive_info:
        archive_path = root / archive_info["path"]
        if (archive_info["path"] != "tests/preservation_archive.json"
                or not archive_path.is_file()
                or hashlib.sha256(archive_path.read_bytes()).hexdigest() != archive_info.get("sha256")):
            raise ValueError("Preservation archive fingerprint mismatch")
        archive = json.loads(archive_path.read_text())
        for source, snapshot in archive["retired_source_inventory"].items():
            review = retired.get(source, {})
            if (review.get("source_sha256") != snapshot["source_sha256"]
                    or set(review.get("entry_points", [])) != set(snapshot["entry_points"])):
                raise ValueError(f"Preservation archive retirement mismatch: {source}")
    for source, review in retired.items():
        digest = review.get("source_sha256", "")
        witnesses = review.get("witnesses", [])
        entry_points = review.get("entry_points", [])
        if (source in seen or (root / source).exists()
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not isinstance(review.get("reason"), str) or not review["reason"].strip()
                or not witnesses or not set(witnesses) <= ids
                or not entry_points or len(set(entry_points)) != len(entry_points)
                or any(not case.startswith(source + "::") for case in entry_points)):
            raise ValueError(f"Invalid reviewed retirement: {source}")
        retired_ids.update(entry_points)
    stale_cases = set(plan["case_assignments"]) - ids
    if stale_cases:
        raise ValueError(f"Stale case assignments (review removals): {sorted(stale_cases)}")
    for regression in plan["protected_regressions"]:
        if regression["legacy"] not in ids | retired_ids:
            raise ValueError(f"Protected regression missing: {regression['legacy']}")
        if regression["legacy"] in retired_ids and not regression.get("replacement"):
            raise ValueError(f"Retired protected regression needs replacement: {regression['legacy']}")
        if regression.get("replacement") and regression["replacement"] not in ids:
            raise ValueError(f"Protected replacement missing: {regression['replacement']}")
    for migration in plan["migrations"]:
        for case in migration["legacy"]:
            if case not in ids | retired_ids:
                raise ValueError(f"Migration witness missing: {case}")
        for case in migration["replacement"]:
            if case not in ids:
                raise ValueError(f"Migration witness missing: {case}")
    for contract in plan.get("contracts", []):
        if contract["area"] not in plan["areas"]:
            raise ValueError(f"Unknown contract area: {contract['id']}")
        if not contract["oracle"] or not contract["required_dimensions"]:
            raise ValueError(f"Contract needs an oracle and dimensions: {contract['id']}")
        if contract["status"] not in ("planned", "partial", "complete"):
            raise ValueError(f"Invalid contract status: {contract['id']}")
        if contract["status"] == "complete" and (contract["open_obligations"] or not contract["witnesses"]):
            raise ValueError(f"Incomplete contract marked complete: {contract['id']}")
        dimensions = contract.get("dimension_witnesses", {})
        if set(dimensions) - set(contract["required_dimensions"]):
            raise ValueError(f"Unknown dimension evidence: {contract['id']}")
        if contract["status"] == "complete" and set(dimensions) != set(contract["required_dimensions"]):
            raise ValueError(f"Complete contract needs evidence for every dimension: {contract['id']}")
        for dimension, witnesses in dimensions.items():
            if not witnesses or not set(witnesses) <= set(contract["witnesses"]):
                raise ValueError(f"Invalid dimension evidence: {contract['id']}: {dimension}")
        for witness in contract["witnesses"]:
            if witness not in ids:
                raise ValueError(f"Contract witness missing: {contract['id']}: {witness}")
    witnesses = {case for contract in plan.get("contracts", []) for case in contract["witnesses"]}
    for case, failure in plan.get("known_failures", {}).items():
        if (case not in witnesses or not isinstance(failure.get("reason"), str)
                or not failure["reason"].strip() or not isinstance(failure.get("error_contains"), str)
                or not failure["error_contains"].strip()):
            raise ValueError(f"Invalid known-failure annotation: {case}")
    for case, failure in plan.get("known_process_failures", {}).items():
        if (case not in witnesses or case in plan.get("known_failures", {})
                or not isinstance(failure.get("reason"), str) or not failure["reason"].strip()
                or not isinstance(failure.get("error_contains"), str) or not failure["error_contains"].strip()
                or type(failure.get("exit_code")) is not int or not 1 <= failure["exit_code"] <= 255):
            raise ValueError(f"Invalid process-failure annotation: {case}")
    capabilities = plan.get("capabilities", {})
    if any(not isinstance(name, str) or not name or not isinstance(reason, str)
           or not reason.strip() for name, reason in capabilities.items()):
        raise ValueError("Invalid capability declaration")
    for case, requirements in plan.get("required_capabilities", {}).items():
        if (case not in witnesses or not isinstance(requirements, list) or not requirements
                or any(not isinstance(name, str) for name in requirements)
                or len(set(requirements)) != len(requirements)
                or not set(requirements) <= set(capabilities)):
            raise ValueError(f"Invalid capability requirement: {case}")
    support = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sources(root) if not entries(p)}
    fixtures = sorted({path for tree in (root / "tests",)
                       for directory in tree.rglob("fixtures") if directory.is_dir()
                       for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts})
    fixture_sources = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in fixtures}
    return dict(version=3, cases=rows, support_sources=support,
                fixture_sources=fixture_sources, retired_sources=retired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    specification = json.loads((DIRECTORY / "plan.json").read_text())
    ledger = json.loads((DIRECTORY / specification["preservation_ledger"]).read_text())
    plan = dict(ledger, areas=specification["areas"], contracts=specification["contracts"],
                known_failures=specification.get("known_failures", {}),
                known_process_failures=specification.get("known_process_failures", {}),
                capabilities=specification.get("capabilities", {}),
                required_capabilities=specification.get("required_capabilities", {}))
    try:
        data = inventory(plan)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path = DIRECTORY / "inventory.json"
    if args.write:
        path.write_text(text)
    elif not path.exists() or path.read_text() != text:
        print("Coverage inventory drift: review case/fixture changes, update plan, then --write", file=sys.stderr)
        return 1
    cases = data["cases"]
    print(f"{len(specification['contracts'])} independently specified contracts")
    for status in ("planned", "partial", "complete"):
        print(f"  contracts {status}: {sum(c['status'] == status for c in specification['contracts'])}")
    print(f"  known implementation failures: {len(plan['known_failures'])}")
    print(f"  isolated fatal implementation failures: {len(plan['known_process_failures'])}")
    print(f"Preservation ledger: {len(cases)} active entry points in {len({r['id'].split('::')[0] for r in cases})} sources; {len(data['retired_sources'])} reviewed source retirements")
    print(f"  static fixture fingerprints: {len(data['fixture_sources'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
