"""Compile a consumer outside the checkout and exercise packaged identity."""
from std.sys import argv
from std.testing import assert_true
from dataframe.json_value import JsonValue
from tests.tooling.contract_runner import (
    capture,
    command_text,
    run_command,
    shell_quote,
    temporary_directory,
    read_text,
)


def _environment() -> List[String]:
    return [
        "env",
        "-u",
        "DATAFRAME_ACCEL_DEVICE",
        "-u",
        "DATAFRAME_ACCEL_MEMORY_LIMIT",
        "DATAFRAME_THREADS=8",
    ]


def _invoke(work: String, command: List[String]) raises -> String:
    var result = capture(
        ["sh", "-c", "cd " + shell_quote(work) + " && " + command_text(command)]
    )
    if result[0]:
        raise Error(result[1])
    return result[1]


def _run(
    work: String,
    binary: String,
    mode: String,
    setting: String = "",
) raises:
    var command = _environment()
    if setting:
        command.append(setting)
    command.append(binary)
    command.append(mode)
    _ = _invoke(work, command)


def main() raises:
    var args = argv()
    if len(args) < 2 or len(args) > 3:
        raise Error("Usage: package_contracts.mojo PACKAGE_DIRECTORY [--gpu]")
    var gpu = len(args) == 3 and String(args[2]) in List[String](
        ["--gpu", "--native"]
    )
    if len(args) == 3 and not gpu:
        raise Error("Unknown package capability")
    var path = capture(["realpath", String(args[1])])
    if path[0]:
        raise Error(path[1])
    var package = String(path[1].strip())
    var work = temporary_directory("/tmp/dataframe-package-contracts")
    run_command(
        [
            "cp",
            "tests/tooling/accelerator_package_consumer.mojo",
            work + "/consumer.mojo",
        ]
    )
    var binary = work + "/consumer"
    _ = _invoke(
        work,
        [
            "mojo",
            "build",
            "-O3",
            "--fp-mode=contract=off",
            "-I",
            package,
            "consumer.mojo",
            "-o",
            binary,
        ],
    )
    var identity_command = _environment()
    identity_command.append(binary)
    identity_command.append("identity")
    var identity = String(_invoke(work, identity_command).strip())
    assert_true(identity != "development-source-no-calibration")
    assert_true(
        identity
        == JsonValue(read_text(package + "/provider.json"))
        .get("cpu_build_id")
        .string()
    )
    _run(work, binary, "cpu")
    _run(work, binary, "cpu", setting="DATAFRAME_ACCEL_DEVICE=invalid")
    _run(work, binary, "cpu", setting="DATAFRAME_ACCEL_MEMORY_LIMIT=0")
    if gpu:
        _run(work, binary, "gpu")
    else:
        print(
            "SKIP(capability): packaged GPU provider execution; enable --gpu on provisioned hardware"
        )
    run_command(["rm", "-r", work])
    print("Packaged consumer identity and independent cells passed")
