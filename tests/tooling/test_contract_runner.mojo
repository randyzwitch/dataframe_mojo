"""Runner sensitivity: discovery, quoting, strict failure matching and continuation."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from std.ffi import external_call
from std.sys import CompilationTarget
from tests.support.mapped_pages import assert_mapped_page, assert_unmapped_page
from dataframe.json_value import JsonValue
from tests.tooling.suite_policy import (
    SuitePolicy,
    ExpectedFailure,
    CapabilityRequirement,
    ContractGroup,
    empty_policy,
    policy_from,
)
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
    selected_modules,
    validate_policy,
    generate_driver,
    generate_fatal_driver,
    shell_quote,
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


def _policy(process: Bool = False) -> SuitePolicy:
    var policy = empty_policy()
    policy.groups = [
        ContractGroup(
            "public", ["tests/public_contracts/test_operations.mojo"]
        ),
        ContractGroup(
            "storage", ["tests/storage_ownership/test_operations.mojo"]
        ),
    ]
    if process:
        policy.groups[0].modules.append(
            "tests/public_contracts/test_crash.mojo"
        )
        policy.failures.append(
            ExpectedFailure(
                "tests/public_contracts/test_crash.mojo::test_fatal",
                "verified fatal fixture",
                "precise fatal diagnostic",
                132,
            )
        )
    return policy^


def test_discovery_contract_selection_and_generated_aliases_are_exact() raises:
    var root = _root()
    var policy = _policy()
    var public = cases_for(root, "public")
    var storage = cases_for(root, "storage")
    assert_equal(
        public, ["tests/public_contracts/test_operations.mojo::test_contract"]
    )
    assert_equal(
        storage, ["tests/storage_ownership/test_operations.mojo::test_contract"]
    )
    assert_equal(
        selected_modules(policy, ["public"]),
        ["tests/public_contracts/test_operations.mojo"],
    )
    assert_equal(len(selected_modules(policy, ["public", "storage"])), 2)
    with assert_raises():
        _ = selected_modules(policy, ["unknown"])
    var generated = generate_driver(
        ["public", "storage"],
        [public.copy(), storage.copy()],
        root + "/driver",
        empty_policy(),
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
    assert_equal(len(cases_for(root, "public")), 2)
    assert_equal(len(validate_policy(policy, root)), 3)
    write_text(extra, "def test_parameterized(value: Int) raises:\n    pass\n")
    with assert_raises(contains="Unsupported test entry signature"):
        _ = cases_for(root, "public")
    write_text(
        extra,
        "def test_same() raises:\n    pass\ndef test_same() raises:\n    pass\n",
    )
    with assert_raises(contains="Duplicate test entry"):
        _ = cases_for(root, "public")
    write_text(extra, "def test_unmapped() raises:\n    pass\n")
    var entry = "tests/public_contracts/test_operations.mojo::test_contract"
    policy.failures.append(ExpectedFailure("missing", "reason", "precise", 0))
    with assert_raises(contains="Stale expected failure"):
        _ = validate_policy(policy, root)
    _ = policy.failures.pop()
    policy.failures.append(ExpectedFailure(entry, "reason", "precise", 0))
    policy.failures.append(ExpectedFailure(entry, "reason", "precise", 0))
    with assert_raises(contains="Duplicate expected failure"):
        _ = validate_policy(policy, root)
    _ = policy.failures.pop()
    policy.failures[0].diagnostic = ""
    with assert_raises(contains="Invalid expected failure"):
        _ = validate_policy(policy, root)
    policy.failures = List[ExpectedFailure]()
    policy.requirements.append(CapabilityRequirement("missing", "metal"))
    with assert_raises(contains="Stale capability requirement"):
        _ = validate_policy(policy, root)
    policy.requirements[0].entry = entry
    with assert_raises(contains="Unknown required capability"):
        _ = validate_policy(policy, root)
    policy.requirements = List[CapabilityRequirement]()
    policy.groups[0].modules.append("tests/public_contracts/missing.mojo")
    with assert_raises(contains="Stale or duplicate contract module"):
        _ = validate_policy(policy, root)


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
    var policy = _policy(process)
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
            ),
            policy,
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
        var metadata = _run_fixture(_root(), scenario, False, "areas")
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
        var metadata = _run_fixture(_root(), scenario, True, "areas")
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


def _fatal_entries() -> List[String]:
    return [
        "tests/public_contracts/test_crash.mojo::test_fatal",
        "tests/storage_ownership/test_crash.mojo::test_fatal",
    ]


def _fatal_pair_compiler(root: String, scenario: String) raises -> String:
    # The compiler fixture models processes, not Mojo dispatch semantics. The
    # generated Mojo dispatcher is compiled and exercised separately below.
    var path = root + "/fixture-compiler"
    var entries = _fatal_entries()
    var script = String(
        '#!/bin/sh\nset -eu\nif [ "$1" = --version ]; then printf \'fixture compiler\\n\'; exit 0; fi\nsource=\noutput=\nwhile [ $# -gt 0 ]; do\n  case "$1" in\n    -o) output=$2; shift 2 ;;\n    *.mojo) source=$1; shift ;;\n    *) shift ;;\n  esac\ndone\nprintf \'%s\\n\' "$source" >> '
    )
    script += shell_quote(root + "/builds.log")
    script += '\ncase "$source" in\n  */fatal/driver.mojo)\n'
    if scenario == "compile":
        script += "    exit 3\n"
    script += '    cat > "$output" <<\'FATAL_CHILD\'\n#!/bin/sh\nset -eu\nprintf \'%s|%s\\n\' "$$" "${DATAFRAME_FATAL_CASE-}" >> '
    script += shell_quote(root + "/children.log")
    script += '\ncase "${DATAFRAME_FATAL_CASE-}" in\n  '
    script += shell_quote(entries[0]) + ")\n    printf '%s\\n' "
    script += shell_quote(
        "storage fatal diagnostic" if scenario
        == "wrong-message" else "public fatal diagnostic"
    )
    script += "\n    exit " + String(
        133 if scenario == "wrong-exit" else 0 if scenario == "repair" else 132
    )
    script += "\n    ;;\n  " + shell_quote(entries[1])
    script += ")\n    printf '%s\\n' 'storage fatal diagnostic'\n    exit 133\n    ;;\n  *) exit 1 ;;\nesac\nFATAL_CHILD\n    ;;\n  *) printf '#!/bin/sh\\nexit 0\\n' > \"$output\" ;;\nesac\nchmod +x \"$output\"\n"
    write_text(path, script)
    run_command(["chmod", "+x", path])
    return path^


def test_fatal_bundle_compiles_once_and_validates_each_fresh_process() raises:
    for scenario in [
        "process",
        "wrong-exit",
        "wrong-message",
        "repair",
        "compile",
    ]:
        var root = _root()
        var policy = _policy()
        var entries = _fatal_entries()
        for i in range(len(entries)):
            var module = (
                "tests/"
                + String("public_contracts" if i == 0 else "storage_ownership")
                + "/test_crash.mojo"
            )
            write_text(
                root + "/" + module, "def test_fatal() raises:\n    pass\n"
            )
            policy.groups[i].modules.append(module^)
            policy.failures.append(
                ExpectedFailure(
                    entries[i].copy(),
                    "verified fixture",
                    String("public" if i == 0 else "storage")
                    + " fatal diagnostic",
                    132 + i,
                )
            )
        var failed = False
        try:
            _ = execute_suite(
                RunnerOptions(
                    root.copy(),
                    ["public", "storage"],
                    List[String](),
                    "combined",
                    root + "/output",
                    _fatal_pair_compiler(root, scenario),
                ),
                policy,
            )
        except error:
            if String(error).find("Contract suite failure; provenance:") < 0:
                raise error
            failed = True
        assert_equal(failed, scenario != "process")
        var listing = capture(
            ["find", root + "/output", "-name", "run.metadata"]
        )
        assert_equal(listing[0], 0)
        var metadata = JsonValue(read_text(String(listing[1].strip())))
        var builds = read_text(root + "/builds.log").strip().split("\n")
        assert_equal(
            len(builds), 2, "One ordinary build and one shared fatal build"
        )
        assert_true(String(builds[1]).endswith("/fatal/driver.mojo"))
        assert_equal(len(metadata.get("process_cases").items()), 2)
        var events = metadata.get("events").items()
        assert_equal(len(events), 3 if scenario == "compile" else 5)
        assert_equal(events[2].get("label").string(), "fatal-bundle")
        assert_equal(events[2].get("phase").string(), "compile")
        assert_equal(
            len(metadata.get("failed").items()),
            0 if scenario == "process" else 1,
        )
        assert_equal(
            metadata.get("expected_failures").text,
            "2" if scenario
            == "process" else "0" if scenario
            == "compile" else "1",
        )
        if scenario == "compile":
            assert_equal(
                metadata.get("failed").items()[0].string(),
                "fatal-bundle:compile",
            )
            assert_equal(capture(["test", "-e", root + "/children.log"])[0], 1)
            continue
        if scenario != "process":
            assert_equal(
                metadata.get("failed").items()[0].string(),
                entries[0] + ":execute",
            )
        var children = read_text(root + "/children.log").strip().split("\n")
        assert_equal(len(children), 2)
        var first = String(children[0]).split("|")
        var second = String(children[1]).split("|")
        assert_true(
            String(first[0]) != String(second[0]),
            "Each case needs a fresh process",
        )
        assert_equal(String(first[1]), entries[0])
        assert_equal(String(second[1]), entries[1])
        for i in range(2):
            assert_equal(events[3 + i].get("label").string(), entries[i])
            assert_equal(events[3 + i].get("phase").string(), "execute")
            assert_true(
                events[3 + i]
                .get("log")
                .string()
                .endswith("/isolated-" + String(i) + "/execute.log")
            )


def test_generated_fatal_dispatch_selects_full_ids_and_isolates_fixture_state() raises:
    var root = _root()
    var entries = _fatal_entries()
    var folder = root + "/dispatch"
    var areas: List[String] = ["public", "storage"]
    for i in range(2):
        var module = (
            "tests/"
            + String("public_contracts" if i == 0 else "storage_ownership")
            + "/test_crash.mojo"
        )
        var fixture = folder + "/case-" + String(i) + "/" + areas[i]
        var source = String(
            'from std.testing import assert_equal, assert_false\nfrom std.os import getenv\nfrom std.os.path import exists\nfrom std.ffi import external_call\n\ndef test_fatal() raises:\n    assert_equal(getenv("DATAFRAME_THREADS"), "1")\n    var fixture = getenv("COVERAGE_TEST_TMP")\n    assert_equal(fixture, '
        )
        source += (
            '"'
            + fixture
            + '")\n    assert_false(exists(fixture + "/sentinel"))\n    with open(fixture + "/sentinel", "w") as file:\n        file.write("owned by this case")\n    print("CASE_'
            + String(i)
            + '")\n    print("PID", external_call["getpid", Int32]())\n'
        )
        write_text(root + "/" + module, source)
    run_command(["mkdir", "-p", folder])
    write_text(
        folder + "/driver.mojo", generate_fatal_driver(areas, entries, folder)
    )
    run_command(
        [
            "mojo",
            "build",
            "-I",
            root.copy(),
            folder + "/driver.mojo",
            "-o",
            folder + "/suite",
        ]
    )
    var pids = List[String]()
    for i in range(2):
        var result = capture(
            [
                "env",
                "DATAFRAME_THREADS=9",
                "COVERAGE_TEST_TMP=/invalid/parent/fixture",
                "DATAFRAME_FATAL_CASE=" + entries[i],
                folder + "/suite",
            ]
        )
        assert_equal(result[0], 0, result[1])
        assert_true(result[1].find("CASE_" + String(i)) >= 0)
        assert_true(result[1].find("CASE_" + String(1 - i)) < 0)
        var pid_line = String(result[1].split("PID ")[1]).split("\n")
        pids.append(String(pid_line[0].strip()))
    assert_true(pids[0] != pids[1])
    # Both files must survive independently despite having the same basename.
    for i in range(2):
        assert_equal(
            read_text(
                folder + "/case-" + String(i) + "/" + areas[i] + "/sentinel"
            ),
            "owned by this case",
        )
    var unknown = capture(
        ["env", "DATAFRAME_FATAL_CASE=test_fatal", folder + "/suite"]
    )
    var missing = capture(
        ["env", "-u", "DATAFRAME_FATAL_CASE", folder + "/suite"]
    )
    for result in [unknown, missing]:
        assert_equal(
            result[0],
            1,
            "Invalid selectors are ordinary errors, not native crashes",
        )
        assert_true(result[1].find("Unknown fatal case selector:") >= 0)
        assert_true(result[1].find("CASE_") < 0)


def test_generated_expected_failure_wrappers_match_only_the_named_diagnostic() raises:
    var root = _root()
    var entry = String(
        "tests/public_contracts/test_operations.mojo::test_contract"
    )
    var policy = empty_policy()
    policy.failures.append(
        ExpectedFailure(entry.copy(), "verified defect", "precise error", 0)
    )
    var generated = generate_driver(
        ["public"], [[entry.copy()]], root + "/driver", policy
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
    var policy = _policy(True)
    policy.capabilities = ["metal"]
    policy.requirements.append(CapabilityRequirement(crash.copy(), "metal"))
    var compiler = _compiler(root, "process")
    var metadata = execute_suite(
        RunnerOptions(
            root.copy(),
            ["public", "storage"],
            List[String](),
            "areas",
            root + "/output",
            compiler.copy(),
        ),
        policy,
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
        ),
        policy,
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
            ),
            policy,
        )


def test_mapping_probe_rejects_mapped_pages_and_observes_each_unmap() raises:
    var page = Int(external_call["getpagesize", Int32]())
    var anonymous = 0x1000 if CompilationTarget.is_macos() else 0x20
    var region = external_call["mmap", Int](
        Int(0), 2 * page, Int32(3), Int32(anonymous | 2), Int32(-1), Int(0)
    )
    assert_true(region != -1)
    # Do not touch the pages first: mapped, nonresident pages must not be
    # mistaken for released memory by a residency-only check.
    assert_mapped_page(region)
    assert_mapped_page(region + page)
    with assert_raises(
        contains="producer backing page must really be unmapped"
    ):
        assert_unmapped_page(region)
    assert_equal(external_call["munmap", Int32](region, page), Int32(0))
    assert_unmapped_page(region)
    assert_mapped_page(region + page)
    assert_equal(external_call["munmap", Int32](region + page, page), Int32(0))
    assert_unmapped_page(region + page)
    with assert_raises(contains="source page must be mapped"):
        assert_mapped_page(region)


def test_policy_markers_attach_to_the_next_test_and_reject_strays() raises:
    var root = temporary_directory("/tmp/df-contract-runner-selftest")
    write_text(
        root + "/test_a.mojo",
        (
            '"""Module."""\n# contract-group: zeta\n# contract-group:'
            " alpha\n\n\n# requires: metal\n# requires: mojo_gpu\ndef"
            " test_gpu() raises:\n    pass\n\n\n# expected-failure: Wrong"
            " total\n# expected-failure: on chunks.\n# expected-diagnostic:"
            " sum mismatch\n# expected-exit: variant-abort\ndef"
            " test_known() raises:\n    pass\n\n\ndef test_plain() raises:\n"
            "    # requires: metal\n    pass\n"
        ),
    )
    var policy = policy_from(root)
    var module = root + "/test_a.mojo"
    assert_equal(len(policy.requirements), 2)
    assert_equal(policy.requirements[0].entry, module + "::test_gpu")
    assert_equal(policy.requirements[0].capability, "metal")
    assert_equal(policy.requirements[1].capability, "mojo_gpu")
    assert_equal(len(policy.failures), 1)
    assert_equal(policy.failures[0].entry, module + "::test_known")
    assert_equal(policy.failures[0].reason, "Wrong total on chunks.")
    assert_equal(policy.failures[0].diagnostic, "sum mismatch")
    assert_equal(
        policy.failures[0].exit_code,
        133 if CompilationTarget.is_macos() else 132,
    )
    assert_equal(len(policy.groups), 2)
    assert_equal(policy.groups[0].name, "alpha")
    assert_equal(policy.groups[1].name, "zeta")
    assert_equal(policy.groups[1].modules, [module])
    write_text(
        root + "/test_b.mojo",
        "# requires: metal\nvar stray = 1\n\ndef test_x() raises:\n    pass\n",
    )
    with assert_raises(contains="directly above"):
        _ = policy_from(root)


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
        test_fatal_bundle_compiles_once_and_validates_each_fresh_process
    ]()
    suite.test[
        test_generated_fatal_dispatch_selects_full_ids_and_isolates_fixture_state
    ]()
    suite.test[
        test_generated_expected_failure_wrappers_match_only_the_named_diagnostic
    ]()
    suite.test[
        test_capability_requirements_omit_imports_and_record_skips_separately
    ]()
    suite.test[
        test_mapping_probe_rejects_mapped_pages_and_observes_each_unmap
    ]()
    suite.test[test_policy_markers_attach_to_the_next_test_and_reject_strays]()
    suite^.run()
