import datetime, hashlib, json, os, re, subprocess
from pathlib import Path

home = Path.home()
harness = home / "GitHub/dataframe_mojo-metal-868"
pixi = home / ".pixi/bin/pixi"
manifest = harness / "pixi.toml"
lock = harness / "pixi.lock"
receipt_sha = "087e08b97d18ae2ef0490e3429ab300d1f37c67cf759bca51c47e493daabe874"

def run(argv, timeout=20):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"argv": argv, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": repr(exc)}

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

pixi_base = [str(pixi), "run", "--locked", "--manifest-path", str(manifest), "-e", "oracle"]
hardware = {
    "model": run(["/usr/sbin/sysctl", "-n", "hw.model"]),
    "physical_cpu_count": run(["/usr/sbin/sysctl", "-n", "hw.physicalcpu"]),
    "logical_cpu_count": run(["/usr/sbin/sysctl", "-n", "hw.logicalcpu"]),
    "memory_bytes": run(["/usr/sbin/sysctl", "-n", "hw.memsize"]),
    "system_profiler_hardware": run(["/usr/sbin/system_profiler", "SPHardwareDataType", "-detailLevel", "mini"], timeout=60),
    "macos_version": run(["/usr/bin/sw_vers", "-productVersion"]),
    "macos_build": run(["/usr/bin/sw_vers", "-buildVersion"]),
    "uname": run(["/usr/bin/uname", "-a"]),
}
compiler_path_result = run(pixi_base + ["python3", "-c", "import shutil; print(shutil.which('mojo') or '')"])
compiler_path = compiler_path_result["stdout"].splitlines()[-1] if compiler_path_result["returncode"] == 0 and compiler_path_result["stdout"] else None
sdk_text = lock.read_text()
max_core_matches = re.findall(r"osx-arm64/max-core-([^/]+?)-release\.conda", sdk_text)
result = {
    "format": "metal-coverage-postrun-host-toolchain-snapshot-v1",
    "capture_kind": "read-only post-run snapshot; captured after completion of coverage, not contemporaneous with execution",
    "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "coverage_run": "metal-868-dev-coverage-a52334e7",
    "coverage_receipt_sha256": receipt_sha,
    "harness_revision": "50137ecc2e27f76298559a25e0aa4ecca31d4e96",
    "host": hardware,
    "toolchain": {
        "mojo_version": run(pixi_base + ["mojo", "--version"]),
        "mojo_executable_path": compiler_path,
        "mojo_executable_sha256": digest(compiler_path) if compiler_path and Path(compiler_path).is_file() else None,
        "xcode_version": run(["/usr/bin/xcodebuild", "-version"]),
        "developer_directory": run(["/usr/bin/xcode-select", "-p"]),
        "macos_sdk_version": run(["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-version"]),
        "macos_sdk_path": run(["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"]),
        "pixi_manifest_path": str(manifest),
        "pixi_lock_path": str(lock),
        "pixi_lock_sha256": digest(lock),
        "max_core_sdk_version_from_lock": max_core_matches[0] if max_core_matches else None,
        "compiler_path_lookup": compiler_path_result,
    },
}
out = Path("/tmp/metal-868-host-toolchain-postrun.json")
out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(out)
