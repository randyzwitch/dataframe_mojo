"""Measured costs must not survive CPU/kernel/compiler/settings changes."""
import sys
import tempfile
import json
from unittest.mock import patch
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from metal_build_identity import metal_build_id, metal_identity_source


class MetalBuildIdentityTests(unittest.TestCase):
    def fixture(self, root):
        for directory in ("dataframe", "dataframe_metal", "dataframe_provider"):
            target = root / directory
            target.mkdir()
            (target / "core.mojo").write_text(f'"""{directory}"""\n')
            (target / "_build_identity.mojo").write_text(
                'comptime ID = "generated"\n'
            )
        (root / "pixi.lock").write_text("compiler and SDK pins\n")
        dependency = root / "packages/emberjson"
        dependency.mkdir(parents=True)
        (dependency / "recipe.yaml").write_text("source: pinned upstream\n")
        (dependency / "mojo-nightly.patch").write_text("compiler compatibility\n")

    def test_reproducible_across_checkout_paths(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, second = Path(a), Path(b)
            self.fixture(first)
            self.fixture(second)
            self.assertEqual(
                metal_build_id(first, "Mojo sdk1"),
                metal_build_id(second, "Mojo sdk1"),
            )

    def test_cpu_metal_provider_and_lock_changes_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            original = metal_build_id(root, "Mojo sdk1")
            for relative in (
                "dataframe/core.mojo",
                "dataframe_metal/core.mojo",
                "dataframe_provider/core.mojo",
                "pixi.lock",
                "packages/emberjson/recipe.yaml",
                "packages/emberjson/mojo-nightly.patch",
            ):
                with self.subTest(source=relative):
                    path = root / relative
                    before = path.read_bytes()
                    path.write_bytes(before + b"changed\n")
                    self.assertNotEqual(
                        original, metal_build_id(root, "Mojo sdk1")
                    )
                    path.write_bytes(before)
            self.assertNotEqual(original, metal_build_id(root, "Mojo sdk2"))
            self.assertNotEqual(
                original, metal_build_id(root, "Mojo sdk1", ("-O3",))
            )

    def test_generated_identity_does_not_hash_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            original = metal_build_id(root, "Mojo sdk1")
            for path in root.rglob("_build_identity.mojo"):
                path.write_text("a different generated identity\n")
            self.assertEqual(original, metal_build_id(root, "Mojo sdk1"))
            (root / "dataframe_metal/new_kernel.mojo").write_text(
                "a new kernel\n"
            )
            self.assertNotEqual(original, metal_build_id(root, "Mojo sdk1"))

    def test_only_generated_digest_is_inserted_into_source(self):
        self.assertIn(
            'comptime METAL_BUILD_ID = "' + "a" * 64 + '"',
            metal_identity_source("a" * 64),
        )
        for invalid in ("", "short", "g" * 64, '"; injected', "a" * 65):
            with self.subTest(identity=invalid):
                with self.assertRaises(ValueError):
                    metal_identity_source(invalid)

    def test_package_builder_stamps_both_identities_without_editing_sources(
        self,
    ):
        import build_package
        from build_identity import cpu_build_id

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            source_cpu = (root / "dataframe/_build_identity.mojo").read_bytes()
            source_metal = (
                root / "dataframe_metal/_build_identity.mojo"
            ).read_bytes()
            stamps = []
            dependencies = root / "installed"
            dependencies.mkdir()
            for package in build_package.JSON_PACKAGES:
                (dependencies / f"{package}.mojoc").write_bytes(
                    package.encode()
                )
            license_file = root / "scripts/vendor/ember-LICENSE-APACHE"
            license_file.parent.mkdir(parents=True)
            license_file.write_text("Apache-2.0 test license")

            def output(command, **kwargs):
                if command[:2] == ["mojo", "--version"]:
                    return "Mojo sdk1\n"
                if "rev-parse" in command:
                    return "a" * 40 + "\n"
                return ""

            def compile_package(command, **kwargs):
                stage = Path(command[3])
                stamps.append(
                    (
                        (stage / "dataframe/_build_identity.mojo").read_text(),
                        (
                            stage / "dataframe_metal/_build_identity.mojo"
                        ).read_text(),
                    )
                )
                Path(command[-1]).write_bytes(b"compiled test package")

            with patch.object(build_package, "ROOT", root), patch.object(
                build_package.subprocess, "check_output", side_effect=output
            ), patch.object(
                build_package.subprocess, "run", side_effect=compile_package
            ), patch.object(
                build_package, "dependency_directory", return_value=dependencies
            ):
                build_package.build(root / "dist", "metal")
            cpu = cpu_build_id(root, "Mojo sdk1")
            metal = metal_build_id(root, "Mojo sdk1")
            self.assertEqual(len(stamps), 3)
            self.assertTrue(all(cpu in a and metal in b for a, b in stamps))
            metadata = json.loads((root / "dist/provider.json").read_text())
            self.assertEqual(
                metadata["runtime_build_id"],
                f"dataframe-mojo-metal-v2;cpu={cpu};metal={metal}",
            )
            self.assertEqual(metadata["metal_build_id"], metal)
            for package in build_package.JSON_PACKAGES:
                filename = f"{package}.mojoc"
                self.assertEqual(
                    (root / "dist" / filename).read_bytes(), package.encode()
                )
                self.assertIn(filename, metadata["packages"])
            self.assertEqual(
                (root / "dist/ember-LICENSE-APACHE").read_text(),
                license_file.read_text(),
            )
            self.assertEqual(
                (root / "dataframe/_build_identity.mojo").read_bytes(),
                source_cpu,
            )
            self.assertEqual(
                (root / "dataframe_metal/_build_identity.mojo").read_bytes(),
                source_metal,
            )


if __name__ == "__main__":
    unittest.main()
