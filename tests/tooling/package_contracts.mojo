"""Compile a consumer outside the checkout and exercise packaged routing identity."""
from std.sys import argv
from std.testing import assert_true
from dataframe.json_value import JsonValue, json_quote
from tests.support.calibration_fixture import calibration_model
from tests.tooling.contract_runner import (
    capture,
    command_text,
    run_command,
    shell_quote,
    temporary_directory,
    read_text,
    write_text,
)


def _record(
    key: JsonValue,
    expiry: Int,
    field: String = "",
    value: String = "",
    accelerator: Float64 = 1,
) raises -> String:
    var result = String("{")
    for name in key.keys():
        if result != "{":
            result += ","
        result += (
            json_quote(name)
            + ":"
            + (value if name == field else key.get(name).text)
        )
    result += ',"validated":' + (value if field == "validated" else "true")
    result += ',"validation_failures":' + (
        value if field == "validation_failures" else "0"
    )
    result += ',"expires_at":' + (
        value if field == "expires_at" else String(expiry)
    )
    return (
        result
        + ',"training_queries":8,"validation_queries":8,"distribution_coverage":"conservative-null-selectivity-envelope","safety_factor":1.5,"cpu":'
        + calibration_model(1e15, max_feature=1e30)
        + ',"accelerator":'
        + calibration_model(accelerator, max_feature=1e30)
        + "}"
    )


def _table(record: String, fraction: String = "0.8") -> String:
    return (
        '{"format":"dataframe-analytical-cost-v1","memory_fraction":'
        + fraction
        + ',"records":['
        + record
        + "]}"
    )


def _environment() -> List[String]:
    return [
        "env",
        "-u",
        "DATAFRAME_ACCEL_COST_TABLE",
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
    table: String = "",
    setting: String = "",
) raises:
    var command = _environment()
    if table:
        write_text(work + "/synthetic-do-not-deploy.json", table)
        command.append(
            "DATAFRAME_ACCEL_COST_TABLE="
            + work
            + "/synthetic-do-not-deploy.json"
        )
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
    if not gpu:
        _run(work, binary, "cpu", setting="DATAFRAME_ACCEL_DEVICE=invalid")
        _run(work, binary, "cpu", setting="DATAFRAME_ACCEL_MEMORY_LIMIT=0")
        print(
            "SKIP(capability): packaged GPU calibration and execution; enable --gpu on provisioned hardware"
        )
        run_command(["rm", "-r", work])
        print(
            "Packaged external import, source identity and independent CPU cells passed"
        )
        return
    var command = _environment()
    command.append(binary)
    command.append("fixture")
    var key = JsonValue(String(_invoke(work, command).strip()))
    assert_true(
        key.get("cpu_build_id").string() != "development-source-no-calibration"
    )
    assert_true(
        key.get("runtime_build_id").string()
        != "development-source-no-calibration"
    )
    var now = capture(["date", "+%s"])
    if now[0]:
        raise Error(now[1])
    var expiry = Int(String(now[1].strip())) + 3600
    var valid = _table(_record(key, expiry))
    _run(work, binary, "cpu")
    _run(work, binary, "cpu", valid, "DATAFRAME_ACCEL_MEMORY_LIMIT=0")
    _run(work, binary, "cpu", valid, "DATAFRAME_ACCEL_DEVICE=invalid")
    _run(work, binary, "cpu", valid, "DATAFRAME_THREADS=1")
    for field in [
        "provider",
        "device",
        "runtime_build_id",
        "cpu_build_id",
        "host",
    ]:
        _run(
            work, binary, "cpu", _table(_record(key, expiry, field, '"stale"'))
        )
    for pair in [
        ("expires_at", "0"),
        ("validated", "false"),
        ("validation_failures", "1"),
    ]:
        _run(
            work, binary, "cpu", _table(_record(key, expiry, pair[0], pair[1]))
        )
    _run(work, binary, "cpu", _table(_record(key, expiry, accelerator=1e18)))
    _run(work, binary, "cpu", _table(String()))
    for fraction in ["0", "-1", "1.1"]:
        _run(work, binary, "cpu", _table(_record(key, expiry), fraction))
    _run(work, binary, "cpu", "invalid JSON")
    if gpu:
        _run(work, binary, "gpu", valid)
        var error_command = _environment()
        error_command.append(binary)
        error_command.append("error-fixture")
        var error_key = JsonValue(String(_invoke(work, error_command).strip()))
        _run(work, binary, "gpu-error", _table(_record(error_key, expiry)))
        if key.get("provider").string() == "metal":
            _run(
                work,
                binary,
                "gpu",
                valid,
                "DATAFRAME_METAL_LIBRARY=/nonexistent/package-native.dylib",
            )
    else:
        print(
            "SKIP(capability): packaged GPU provider execution; enable --gpu on provisioned hardware"
        )
    run_command(["rm", "-r", work])
    print(
        "Packaged consumer identity, independent cells and conservative routing checks passed"
    )
