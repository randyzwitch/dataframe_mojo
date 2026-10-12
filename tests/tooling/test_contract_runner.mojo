"""Runner sensitivity: discovery, quoting, strict failure matching and continuation."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe.json_value import JsonValue, json_quote
from tests.tooling.contract_runner import (
    RunnerOptions,
    DEFAULT_PACKAGING,
    execute_suite,
    temporary_directory,
    write_text,
    read_text,
    run_command,
    capture,
    cases_for,
    selected_witnesses,
    generate_driver,
)


def _root() raises -> String:
    var root = temporary_directory("/tmp/df-contract-runner-selftest")
    run_command(
        [
            "mkdir",
            "-p",
            root + "/tests/public_contracts",
            root + "/tests/storage_ownership",
        ]
    )
    for area in ["public_contracts", "storage_ownership"]:
        write_text(
            root + "/tests/" + area + "/test_operations.mojo",
            "def test_contract() raises:\n    pass\n",
        )
    return root^


def _catalog(process: Bool = False) raises -> JsonValue:
    var public = String(
        "tests/public_contracts/test_operations.mojo::test_contract"
    )
    var storage = String(
        "tests/storage_ownership/test_operations.mojo::test_contract"
    )
    var crash = String("tests/public_contracts/test_crash.mojo::test_fatal")
    return JsonValue(
        '{"contracts":[{"id":"public","witnesses":['
        + json_quote(public)
        + ("," + json_quote(crash) if process else "")
        + ']},{"id":"storage","witnesses":['
        + json_quote(storage)
        + ']}],"known_failures":{},"known_process_failures":{'
        + (
            json_quote(crash)
            + ':{"exit_code":132,"error_contains":"precise fatal diagnostic","reason":"verified fatal fixture"}' if process else ""
        )
        + "}}"
    )


def test_discovery_contract_selection_and_generated_aliases_are_exact() raises:
    var root = _root()
    var catalog = _catalog()
    var public = cases_for(catalog, root, "public")
    var storage = cases_for(catalog, root, "storage")
    assert_equal(
        public, ["tests/public_contracts/test_operations.mojo::test_contract"]
    )
    assert_equal(
        storage, ["tests/storage_ownership/test_operations.mojo::test_contract"]
    )
    assert_equal(selected_witnesses(catalog, ["public"]), public)
    assert_equal(len(selected_witnesses(catalog, ["public", "storage"])), 2)
    with assert_raises():
        _ = selected_witnesses(catalog, ["unknown"])
    var generated = generate_driver(
        ["public", "storage"],
        [public.copy(), storage.copy()],
        root + "/driver",
        JsonValue("{}"),
    )
    assert_true(
        generated.find(
            "from tests.public_contracts.test_operations import test_contract as public__test_operations__test_contract"
        )
        >= 0
    )
    assert_true(
        generated.find(
            "from tests.storage_ownership.test_operations import test_contract as storage__test_operations__test_contract"
        )
        >= 0
    )
    assert_true(generated.find('failed.append("public")') >= 0)
    var extra = root + "/tests/public_contracts/test_extra.mojo"
    write_text(extra, "def test_unmapped() raises:\n    pass\n")
    with assert_raises():
        _ = cases_for(catalog, root, "public")
    write_text(extra, "def test_parameterized(value: Int) raises:\n    pass\n")
    with assert_raises():
        _ = cases_for(catalog, root, "public")


def test_process_arguments_are_literal_and_wait_statuses_retain_failures() raises:
    var payload = String(
        "apostrophe ' and spaces; $(printf injected) `printf injected`\nsecond line"
    )
    var echoed = capture(["printf", "%s", payload.copy()])
    assert_equal(echoed[0], 0)
    assert_equal(echoed[1], payload)
    assert_equal(capture(["sh", "-c", "exit 42"])[0], 42)
    assert_equal(capture(["sh", "-c", "kill -ILL $$"])[0], 132)


def _compiler(root: String, scenario: String) raises -> String:
    var path = root + "/fixture-compiler"
    var script = String(
        '#!/bin/sh\nset -eu\nif [ "$1" = --version ]; then printf \'fixture compiler\\n\'; exit 0; fi\nsource=\noutput=\nwhile [ $# -gt 0 ]; do\n  case "$1" in\n    -o) output=$2; shift 2 ;;\n    *.mojo) source=$1; shift ;;\n    *) shift ;;\n  esac\ndone\ncode=0\nmessage=\ncase "$(cat "$source")" in\n  *public_contracts*)\n'
    )
    if scenario == "compile":
        script += "    exit 3\n"
    elif scenario == "execute":
        script += "    code=4\n"
    script += '    ;;\nesac\ncase "$(cat "$source")" in\n  *test_crash*)\n'
    if scenario == "process" or scenario == "wrong-message":
        script += "    code=132\n"
    elif scenario == "wrong-exit":
        script += "    code=133\n"
    script += (
        "    message="
        + (
            "unrelated" if scenario
            == "wrong-message" else "'precise fatal diagnostic'"
        )
        + '\n    ;;\nesac\nprintf \'#!/bin/sh\\nprintf "%%s\\\\n" "%s"\\nexit %s\\n\' "$message" "$code" > "$output"\nchmod +x "$output"\n'
    )
    write_text(path, script)
    run_command(["chmod", "+x", path])
    return path^


def _run_fixture(
    root: String,
    scenario: String,
    process: Bool,
    packaging: String = DEFAULT_PACKAGING,
) raises -> JsonValue:
    var catalog = _catalog(process)
    write_text(root + "/tests/plan.json", catalog.text)
    if process:
        write_text(
            root + "/tests/public_contracts/test_crash.mojo",
            "def test_fatal() raises:\n    pass\n",
        )
    var failed = False
    try:
        _ = execute_suite(
            RunnerOptions(
                root.copy(),
                ["public", "storage"],
                List[String](),
                packaging.copy(),
                root + "/output",
                _compiler(root, scenario),
            )
        )
    except error:
        if String(error).find("Contract suite failure; provenance:") < 0:
            raise error
        failed = True
    var expected_failure = scenario in [
        "compile",
        "execute",
        "wrong-exit",
        "wrong-message",
        "repair",
    ]
    assert_equal(failed, expected_failure)
    var listing = capture(
        ["find", root + "/output", "-name", "run.metadata", "-type", "f"]
    )
    assert_equal(listing[0], 0)
    return JsonValue(read_text(String(listing[1].strip())))


def test_compile_and_execution_failures_do_not_hide_later_areas() raises:
    assert_equal(String(DEFAULT_PACKAGING), "areas")
    for scenario in ["none", "compile", "execute"]:
        var metadata = _run_fixture(_root(), scenario, False)
        assert_equal(metadata.get("packaging").string(), "areas")
        var events = metadata.get("events").items()
        assert_equal(len(events), 3 if scenario == "compile" else 4)
        assert_equal(events[len(events) - 1].get("label").string(), "storage")
        assert_equal(events[len(events) - 1].get("phase").string(), "execute")
        assert_equal(events[len(events) - 1].get("exit_code").text, "0")
        assert_equal(metadata.get("compiler").string(), "fixture compiler")
        var failures = metadata.get("failed").items()
        assert_equal(len(failures), 0 if scenario == "none" else 1)
        if failures:
            assert_equal(failures[0].string(), "public:" + scenario)


def test_explicit_combined_packaging_keeps_selection_and_reports_shared_process_failures() raises:
    for scenario in ["none", "compile", "execute"]:
        var metadata = _run_fixture(_root(), scenario, False, "combined")
        assert_equal(metadata.get("packaging").string(), "combined")
        assert_equal(len(metadata.get("cases").get("public").items()), 1)
        assert_equal(len(metadata.get("cases").get("storage").items()), 1)
        var events = metadata.get("events").items()
        assert_equal(len(events), 1 if scenario == "compile" else 2)
        for event in events:
            assert_equal(event.get("label").string(), "combined")
        var failures = metadata.get("failed").items()
        assert_equal(len(failures), 0 if scenario == "none" else 1)
        if failures:
            assert_equal(failures[0].string(), "combined:" + scenario)


def test_isolated_failures_require_exact_exit_and_diagnostic_and_reject_xpass() raises:
    for scenario in ["process", "wrong-exit", "wrong-message", "repair"]:
        var metadata = _run_fixture(_root(), scenario, True)
        assert_equal(
            metadata.get("expected_failures").text,
            "1" if scenario == "process" else "0",
        )
        assert_equal(len(metadata.get("process_cases").items()), 1)
        assert_equal(len(metadata.get("events").items()), 6)
        assert_equal(
            len(metadata.get("failed").items()),
            0 if scenario == "process" else 1,
        )


def test_generated_expected_failure_wrappers_match_only_the_named_diagnostic() raises:
    var root = _root()
    var entry = String(
        "tests/public_contracts/test_operations.mojo::test_contract"
    )
    var known = JsonValue(
        "{"
        + json_quote(entry)
        + ':{"error_contains":"precise error","reason":"verified defect"}}'
    )
    var generated = generate_driver(
        ["public"], [[entry.copy()]], root + "/driver", known
    )
    assert_true(generated.find('String(error).find("precise error") < 0') >= 0)
    assert_true(generated.find("raise error") >= 0)
    assert_true(generated.find("XPASS:") >= 0)
    assert_true(
        generated.find("public__test_operations__test_contract__known_failure")
        >= 0
    )


def test_capability_requirements_omit_imports_and_record_skips_separately() raises:
    var root = _root()
    var crash = String("tests/public_contracts/test_crash.mojo::test_fatal")
    write_text(
        root + "/tests/public_contracts/test_crash.mojo",
        "def test_fatal() raises:\n    pass\n",
    )
    var catalog = _catalog(True)
    var text = String(catalog.text[byte = 0 : catalog.text.byte_length() - 1])
    text += (
        ',"capabilities":{"metal":"real native provider"},"required_capabilities":{'
        + json_quote(crash)
        + ':["metal"]}}'
    )
    write_text(root + "/tests/plan.json", text)
    var compiler = _compiler(root, "process")
    var metadata = execute_suite(
        RunnerOptions(
            root.copy(),
            ["public", "storage"],
            List[String](),
            "areas",
            root + "/output",
            compiler.copy(),
        )
    )
    var data = JsonValue(read_text(metadata))
    assert_equal(len(data.get("skipped_capability_cases").items()), 1)
    assert_equal(
        data.get("skipped_capability_cases").items()[0].string(), crash
    )
    assert_equal(len(data.get("process_cases").items()), 0)
    assert_equal(len(data.get("events").items()), 4)
    metadata = execute_suite(
        RunnerOptions(
            root.copy(),
            ["public", "storage"],
            List[String](),
            "areas",
            root + "/output",
            compiler.copy(),
            ["metal"],
        )
    )
    data = JsonValue(read_text(metadata))
    assert_equal(len(data.get("skipped_capability_cases").items()), 0)
    assert_equal(len(data.get("process_cases").items()), 1)
    assert_equal(data.get("expected_failures").text, "1")
    with assert_raises(contains="Unknown runtime capability"):
        _ = execute_suite(
            RunnerOptions(
                root.copy(),
                ["public"],
                List[String](),
                "areas",
                root + "/output",
                compiler.copy(),
                ["typo"],
            )
        )


def main() raises:
    var suite = TestSuite()
    suite.test[
        test_discovery_contract_selection_and_generated_aliases_are_exact
    ]()
    suite.test[
        test_process_arguments_are_literal_and_wait_statuses_retain_failures
    ]()
    suite.test[test_compile_and_execution_failures_do_not_hide_later_areas]()
    suite.test[
        test_explicit_combined_packaging_keeps_selection_and_reports_shared_process_failures
    ]()
    suite.test[
        test_isolated_failures_require_exact_exit_and_diagnostic_and_reject_xpass
    ]()
    suite.test[
        test_generated_expected_failure_wrappers_match_only_the_named_diagnostic
    ]()
    suite.test[
        test_capability_requirements_omit_imports_and_record_skips_separately
    ]()
    suite^.run()
