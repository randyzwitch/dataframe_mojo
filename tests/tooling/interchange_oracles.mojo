"""Run independent external format consumers against actual suite outputs.

Python here is solely the adapter to PyArrow's external implementation. Process
orchestration, fixture discovery and failure propagation are Mojo.
"""
from std.sys import argv
from std.os import getenv
from std.sys import CompilationTarget
from tests.tooling.contract_runner import capture, run_command


def _consume_outputs(
    root: String, sentinel: String, adapter: String
) raises -> Int:
    var listing = capture(
        [
            "find",
            root,
            "-type",
            "f",
            "-name",
            sentinel,
        ]
    )
    if listing[0]:
        raise Error(listing[1])
    var count = 0
    for part in listing[1].split("\n"):
        var file = String(part)
        if not file:
            continue
        var directory = String(file[byte = 0 : file.rfind("/")])
        run_command(
            [
                "python3",
                adapter,
                "--check-written",
                directory,
            ]
        )
        if adapter == "tests/support/spatial_oracle.py":
            run_command(
                [
                    "python3",
                    "tests/support/spatial_engine_oracle.py",
                    "--check-written",
                    directory,
                ]
            )
        count += 1
    if not count:
        raise Error(
            "No actual writer outputs found for independent validation: "
            + sentinel
        )
    return count


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error(
            "Usage: interchange_oracles.mojo CONTRACT_RUN_OUTPUT_DIRECTORY"
        )
    var parquet_count = _consume_outputs(
        String(args[1]),
        "parquet-written-empty.parquet",
        "tests/support/parquet_oracle.py",
    )
    var spatial_count = _consume_outputs(
        String(args[1]),
        "spatial-written-empty.parquet",
        "tests/support/spatial_oracle.py",
    )
    var ipc_count = _consume_outputs(
        String(args[1]),
        "ipc-written-empty.arrow",
        "tests/support/ipc_oracle.py",
    )
    run_command(["python3", "tests/support/arrow_oracle.py", "--check"])
    var library = getenv("DATAFRAME_PARQUET_LIBRARY")
    if not library:
        comptime if CompilationTarget.is_macos():
            library = "build/dfparquet/libdfparquet.dylib"
        else:
            library = "build/dfparquet/libdfparquet.so"
    run_command(["python3", "tests/support/native_stream_oracle.py", library])
    print(
        "Independent PyArrow consumers passed for",
        parquet_count,
        "Parquet and",
        spatial_count,
        "spatial and",
        ipc_count,
        "IPC contract run directories",
    )
