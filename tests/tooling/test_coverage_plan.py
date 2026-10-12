"""Coverage bookkeeping must detect disappearing regression witnesses and fixtures."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("inventory", ROOT / "tests/tooling/coverage_inventory.py")
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class CoveragePlan(unittest.TestCase):
    def test_retirement_requires_recorded_source_and_live_independent_witness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            source = "tests/test_contract.mojo"
            (root / source).write_text("def test_contract() raises:\n    pass\n")
            witness = source + "::test_contract"
            legacy = "tests_legacy/test_old.mojo::test_regression"
            review = dict(source_sha256="a" * 64, entry_points=[legacy],
                          reason="Independent scalar contract preserves the unique null-payload regression",
                          witnesses=[witness])
            plan = dict(areas={"public_contracts": {}},
                        assignments={source: ["public_contracts"]},
                        case_assignments={witness: ["public_contracts"]},
                        retired_sources={"tests_legacy/test_old.mojo": review},
                        protected_regressions=[dict(legacy=legacy, replacement=witness)],
                        migrations=[])
            inventory.inventory(plan, root)
            archive = root / "tests/preservation_archive.json"
            archive.write_text(json.dumps({"retired_source_inventory": {
                "tests_legacy/test_old.mojo": dict(source_sha256="a" * 64, entry_points=[legacy])
            }}))
            plan["preservation_archive"] = dict(path="tests/preservation_archive.json",
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            inventory.inventory(plan, root)
            archive.write_text(archive.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "archive fingerprint"):
                inventory.inventory(plan, root)
            plan["preservation_archive"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            review["source_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "archive retirement mismatch"):
                inventory.inventory(plan, root)
            review["source_sha256"] = "a" * 64
            del plan["preservation_archive"]
            review["witnesses"] = ["missing"]
            with self.assertRaisesRegex(ValueError, "Invalid reviewed retirement"):
                inventory.inventory(plan, root)
            review["witnesses"] = [witness]
            plan["protected_regressions"][0].pop("replacement")
            with self.assertRaisesRegex(ValueError, "needs replacement"):
                inventory.inventory(plan, root)
            plan["protected_regressions"] = []
            review["entry_points"] = ["unrelated::test_case"]
            with self.assertRaisesRegex(ValueError, "Invalid reviewed retirement"):
                inventory.inventory(plan, root)

    def test_python_methods_and_script_entrypoint_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_example.py"
            path.write_text("class Suite:\n    def test_one(self): pass\ndef check_two(): pass\nassert True\n")
            self.assertEqual([name for name, _ in inventory.entries(path)], ["Suite.test_one", "check_two", "@script"])

    def test_unassigned_source_and_missing_protected_witness_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "scripts").mkdir()
            path = root / "tests/test_example.mojo"
            path.write_text("def test_one() raises:\n    pass\n")
            plan = dict(areas={"public_contracts": {}}, assignments={}, protected_regressions=[], migrations=[], case_assignments={"tests/test_example.mojo::test_one": ["public_contracts"]})
            with self.assertRaisesRegex(ValueError, "Unmapped"):
                inventory.inventory(plan, root)
            plan["assignments"]["tests/test_example.mojo"] = ["public_contracts"]
            plan["protected_regressions"] = [{"legacy": "tests/test_example.mojo::test_missing"}]
            with self.assertRaisesRegex(ValueError, "Protected regression missing"):
                inventory.inventory(plan, root)
            plan["protected_regressions"] = []
            fixture = root / "tests/fixtures/example.parquet"
            fixture.parent.mkdir()
            fixture.write_bytes(b"PAR1\x00\x01\xff")
            first = inventory.inventory(plan, root)
            fixture.write_bytes(b"PAR1\x00\x01\xfe")
            self.assertNotEqual(first, inventory.inventory(plan, root))
            first = inventory.inventory(plan, root)
            path.write_text("def test_one() raises:\n    raise Error(\"mutation\")\n")
            self.assertNotEqual(first, inventory.inventory(plan, root))

    def test_complete_contract_requires_evidence_for_each_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "scripts").mkdir()
            source = "tests/test_example.mojo"
            (root / source).write_text("def test_contract() raises:\n    pass\n")
            witness = source + "::test_contract"
            contract = dict(id="example", area="public_contracts", oracle="literal examples",
                            required_dimensions=["empty", "mixed-null"], witnesses=[witness],
                            status="complete", open_obligations=[])
            plan = dict(areas={"public_contracts": {}},
                        assignments={source: ["public_contracts"]},
                        case_assignments={witness: ["public_contracts"]},
                        protected_regressions=[], migrations=[], contracts=[contract])
            with self.assertRaisesRegex(ValueError, "every dimension"):
                inventory.inventory(plan, root)
            contract["dimension_witnesses"] = {"empty": [witness], "mixed-null": []}
            with self.assertRaisesRegex(ValueError, "Invalid dimension evidence"):
                inventory.inventory(plan, root)
            contract["dimension_witnesses"]["mixed-null"] = [witness]
            inventory.inventory(plan, root)
            contract["open_obligations"] = ["still needs coverage"]
            with self.assertRaisesRegex(ValueError, "Incomplete contract marked complete"):
                inventory.inventory(plan, root)
            contract["open_obligations"] = []
            contract["status"] = "partial"
            contract["dimension_witnesses"]["typo"] = [witness]
            with self.assertRaisesRegex(ValueError, "Unknown dimension evidence"):
                inventory.inventory(plan, root)
            del contract["dimension_witnesses"]["typo"]
            plan["known_failures"] = {witness: {"reason": "verified implementation gap", "error_contains": "specific diagnostic"}}
            inventory.inventory(plan, root)
            plan["known_failures"][witness]["error_contains"] = ""
            with self.assertRaisesRegex(ValueError, "Invalid known-failure annotation"):
                inventory.inventory(plan, root)

            plan["known_failures"] = {}
            plan["known_process_failures"] = {witness: {"reason": "verified fatal fixture", "error_contains": "precise crash", "exit_code": 132}}
            inventory.inventory(plan, root)
            plan["known_process_failures"][witness]["exit_code"] = True
            with self.assertRaisesRegex(ValueError, "Invalid process-failure annotation"):
                inventory.inventory(plan, root)
            plan["known_process_failures"] = {}
            plan["capabilities"] = {"metal": "actual native hardware required"}
            plan["required_capabilities"] = {witness: ["metal"]}
            inventory.inventory(plan, root)
            plan["required_capabilities"][witness] = ["typo"]
            with self.assertRaisesRegex(ValueError, "Invalid capability requirement"):
                inventory.inventory(plan, root)
            plan["required_capabilities"] = {}
            plan["known_failures"] = {"missing": {"reason": "gap", "error_contains": "diagnostic"}}
            with self.assertRaisesRegex(ValueError, "Invalid known-failure annotation"):
                inventory.inventory(plan, root)

    def test_mojo_nonstandard_main_and_helpers_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.mojo"
            path.write_text("def check_release() raises:\n    pass\n\ndef main() raises:\n    check_release()\n")
            self.assertEqual([name for name, _ in inventory.entries(path)], ["check_release", "@main"])


if __name__ == "__main__":
    unittest.main()
