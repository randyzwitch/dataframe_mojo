"""Mojo-native suite orchestration; POSIX processes only, no Python runtime.

Source discovery includes every ordinary Mojo test without a registry. The typed
policy only controls expected failures, capabilities and convenient module groups.
Process-isolated failures require both a precise diagnostic and exact exit code.
"""
from std.ffi import external_call
from std.os import getenv
from std.os.path import exists
from dataframe.json_value import json_quote
from tests.tooling.suite_policy import SuitePolicy

comptime DEFAULT_PACKAGING = "areas"
comptime AREAS = (
    "public",
    "storage",
    "kernels",
    "planner",
    "execution",
    "interchange",
)
comptime DIRECTORIES = (
    "public_contracts",
    "storage_ownership",
    "kernel_equivalence",
    "planner_correctness",
    "execution_modes",
    "interchange",
)


def shell_quote(value: String) -> String:
    return "'" + value.replace("'", "'\\''") + "'"


def command_text(args: List[String]) -> String:
    var result = String()
    for i in range(len(args)):
        if i:
            result += " "
        result += shell_quote(args[i])
    return result^


def exit_code(status: Int32) -> Int:
    if status < 0:
        return 255
    var signal = Int(status) & 127
    return 128 + signal if signal else (Int(status) >> 8) & 255


def capture(args: List[String]) raises -> Tuple[Int, String]:
    var command = command_text(args) + " 2>&1"
    var command_bytes = List[UInt8]()
    command_bytes.extend(command.as_bytes())
    command_bytes.append(0)
    var mode: List[UInt8] = [114, 0]
    var stream = external_call["popen", Int](
        Int(command_bytes.unsafe_ptr()), Int(mode.unsafe_ptr())
    )
    _ = command_bytes^
    _ = mode^
    if not stream:
        raise Error("Cannot start command: " + command)
    var output = List[UInt8]()
    while True:
        var byte = external_call["fgetc", Int32](stream)
        if byte < 0:
            break
        output.append(UInt8(byte))
    var status = external_call["pclose", Int32](stream)
    return (exit_code(status), String(unsafe_from_utf8=output))


def run_command(args: List[String]) raises:
    var result = capture(args)
    if result[0]:
        raise Error("Command failed: " + command_text(args) + "\n" + result[1])


def read_text(path: String) raises -> String:
    with open(path, "r") as file:
        return file.read()


def write_text(path: String, text: String) raises:
    with open(path, "w") as file:
        file.write(text)


def temporary_directory(parent: String) raises -> String:
    run_command(["mkdir", "-p", parent])
    var template = parent + "/run-XXXXXX"
    var bytes = List[UInt8]()
    bytes.extend(template.as_bytes())
    bytes.append(0)
    var result = external_call["mkdtemp", Int](bytes.unsafe_ptr())
    if not result:
        raise Error("Cannot create suite temporary directory")
    _ = bytes.pop()
    return String(unsafe_from_utf8=bytes)


def json_strings(values: List[String]) -> String:
    var result = String("[")
    for i in range(len(values)):
        if i:
            result += ","
        result += json_quote(values[i])
    return result + "]"


def _append_unique(mut values: List[String], value: String):
    if value not in values:
        values.append(value.copy())


def area_directory(area: String) raises -> String:
    comptime for i in range(len(AREAS)):
        if area == AREAS[i]:
            return String(DIRECTORIES[i])
    raise Error("Unknown test area: " + area)


def discovered_cases(root: String, area: String) raises -> List[String]:
    var directory = "tests/" + area_directory(area)
    var listing = capture(
        [
            "find",
            root + "/" + directory,
            "-maxdepth",
            "1",
            "-type",
            "f",
            "-name",
            "test_*.mojo",
        ]
    )
    if listing[0]:
        raise Error(listing[1])
    var cases = List[String]()
    for part in listing[1].split("\n"):
        var path = String(part)
        if not path:
            continue
        var slash = path.rfind("/")
        var relative = directory + "/" + String(path[byte = slash + 1 :])
        for part in read_text(path).split("\n"):
            var line = String(part)
            if not line.startswith("def test_"):
                continue
            var paren = line.find("(")
            if paren < 0 or String(line[byte=paren:]) != "() raises:":
                raise Error("Unsupported test entry signature in " + relative)
            var entry = relative + "::" + String(line[byte=4:paren])
            if entry in cases:
                raise Error("Duplicate test entry: " + entry)
            cases.append(entry^)
    # Stable order makes generated sources and run metadata reviewable.
    for i in range(1, len(cases)):
        var at = i
        while at > 0 and cases[at] < cases[at - 1]:
            var previous = cases[at - 1].copy()
            cases[at - 1] = cases[at].copy()
            cases[at] = previous^
            at -= 1
    return cases^


def cases_for(root: String, area: String) raises -> List[String]:
    return discovered_cases(root, area)


def selected_modules(
    policy: SuitePolicy, contracts: List[String]
) raises -> List[String]:
    var known = List[String]()
    var modules = List[String]()
    for group in policy.groups:
        known.append(group.name.copy())
        if group.name in contracts:
            for module in group.modules:
                _append_unique(modules, module)
    for name in contracts:
        if name not in known:
            raise Error("Unknown contract: " + name)
    return modules^


def case_module(entry: String) raises -> String:
    var split = entry.find("::")
    if split < 0:
        raise Error("Invalid entry identifier: " + entry)
    return String(entry[byte=0:split])


def shard_cases(
    cases: List[String], index: Int, count: Int
) raises -> List[String]:
    """Partition sorted discovery by module, keeping its tests together."""
    if count < 1 or index < 0 or index >= count:
        raise Error("Shard count must be positive and index in [0, count)")
    var selected = List[String]()
    var previous = String()
    var module_index = -1
    for entry in cases:
        var module = case_module(entry)
        if module != previous:
            module_index += 1
            previous = module.copy()
        if module_index % count == index:
            selected.append(entry.copy())
    return selected^


def validate_policy(policy: SuitePolicy, root: String) raises -> List[String]:
    var discovered = List[String]()
    var modules = List[String]()
    comptime for i in range(len(AREAS)):
        if exists(root + "/tests/" + String(DIRECTORIES[i])):
            for entry in discovered_cases(root, String(AREAS[i])):
                discovered.append(entry.copy())
                _append_unique(modules, case_module(entry))
    var seen = List[String]()
    for capability in policy.capabilities:
        if not capability or capability in seen:
            raise Error("Invalid or duplicate capability: " + capability)
        seen.append(capability.copy())
    seen = List[String]()
    for failure in policy.failures:
        if failure.entry in seen:
            raise Error("Duplicate expected failure: " + failure.entry)
        if failure.entry not in discovered:
            raise Error("Stale expected failure: " + failure.entry)
        if (
            not failure.reason
            or not failure.diagnostic
            or failure.exit_code < 0
            or failure.exit_code > 255
        ):
            raise Error("Invalid expected failure: " + failure.entry)
        seen.append(failure.entry.copy())
    seen = List[String]()
    for requirement in policy.requirements:
        if requirement.entry not in discovered:
            raise Error("Stale capability requirement: " + requirement.entry)
        if requirement.capability not in policy.capabilities:
            raise Error(
                "Unknown required capability: " + requirement.capability
            )
        var key = requirement.entry + "::" + requirement.capability
        if key in seen:
            raise Error(
                "Duplicate capability requirement: " + requirement.entry
            )
        seen.append(key^)
    seen = List[String]()
    for group in policy.groups:
        if not group.name or group.name in seen or not group.modules:
            raise Error("Invalid or duplicate contract group: " + group.name)
        seen.append(group.name.copy())
        var grouped = List[String]()
        for module in group.modules:
            if module not in modules or module in grouped:
                raise Error("Stale or duplicate contract module: " + module)
            grouped.append(module.copy())
    return discovered^


def case_alias(entry: String, area: String) raises -> String:
    var split = entry.find("::")
    if split < 0:
        raise Error("Invalid entry identifier: " + entry)
    var source = String(entry[byte=0:split])
    var slash = source.rfind("/")
    return (
        area
        + "__"
        + String(source[byte = slash + 1 : source.byte_length() - 5])
        + "__"
        + String(entry[byte = split + 2 :])
    )


def _settings_source() -> String:
    return String(
        "\n\ndef _setting(name: String, value: String) raises:\n    var key ="
        " List[UInt8]()\n    key.extend(name.as_bytes())\n    key.append(0)\n  "
        "  var contents = List[UInt8]()\n   "
        " contents.extend(value.as_bytes())\n    contents.append(0)\n    var"
        ' status = external_call["setenv", Int32](Int(key.unsafe_ptr()),'
        " Int(contents.unsafe_ptr()), Int32(1))\n    _ = key^\n    _ ="
        ' contents^\n    if status != 0:\n        raise Error("Cannot set suite'
        ' environment: " + name)\n'
    )


def generate_driver(
    areas: List[String],
    cases: List[List[String]],
    directory: String,
    policy: SuitePolicy,
) raises -> String:
    var result = String(
        "from std.testing import TestSuite\nfrom std.ffi import external_call\n"
    )
    for i in range(len(areas)):
        run_command(["mkdir", "-p", directory + "/" + areas[i]])
        for entry in cases[i]:
            var split = entry.find("::")
            var module = String(entry[byte = 0 : split - 5]).replace("/", ".")
            result += (
                "from "
                + module
                + " import "
                + String(entry[byte = split + 2 :])
                + " as "
                + case_alias(entry, areas[i])
                + "\n"
            )
    for i in range(len(areas)):
        for entry in cases[i]:
            var index = policy.failure_index(entry)
            if index < 0:
                continue
            ref failure = policy.failures[index]
            var alias = case_alias(entry, areas[i])
            result += (
                "\n\ndef "
                + alias
                + "__known_failure() raises:\n    var failed = False\n   "
                " try:\n        "
                + alias
                + "()\n    except error:\n        if String(error).find("
                + json_quote(failure.diagnostic)
                + ") < 0:\n            raise error\n        print("
                + json_quote("XFAIL: " + entry + ": " + failure.reason)
                + ", error)\n        failed = True\n    if not failed:\n       "
                " raise Error("
                + json_quote(
                    "XPASS: " + entry + "; remove known-failure annotation"
                )
                + ")\n"
            )
    result += _settings_source()
    result += "\n\ndef main() raises:\n    var failed = List[String]()\n"
    for i in range(len(areas)):
        var area = areas[i]
        result += (
            '    _setting("DATAFRAME_THREADS", "1")\n   '
            ' _setting("COVERAGE_TEST_TMP", '
            + json_quote(directory + "/" + area)
            + ")\n    print("
            + json_quote("CONTRACT AREA: " + area)
            + ")\n    var suite_"
            + area
            + " = TestSuite()\n"
        )
        for entry in cases[i]:
            result += (
                "    suite_"
                + area
                + ".test["
                + case_alias(entry, area)
                + (
                    "__known_failure" if policy.failure_index(entry)
                    >= 0 else ""
                )
                + "]()\n"
            )
        result += (
            "    try:\n        suite_"
            + area
            + "^.run()\n    except error:\n        print("
            + json_quote("CONTRACT AREA ERROR: " + area + ":")
            + ", error)\n        failed.append("
            + json_quote(area)
            + ")\n"
        )
    result += (
        '    _setting("DATAFRAME_THREADS", "1")\n    if len(failed):\n       '
        ' raise Error("Contract area failure")\n'
    )
    return result^


def generate_fatal_driver(
    areas: List[String], cases: List[String], directory: String
) raises -> String:
    """Compile cases together while selecting exactly one per fresh process."""
    if not cases or len(areas) != len(cases):
        raise Error(
            "Fatal dispatcher requires matching nonempty areas and cases"
        )
    var result = String(
        "from std.testing import TestSuite\nfrom std.ffi import"
        " external_call\nfrom std.os import getenv\n"
    )
    for i in range(len(cases)):
        var entry = cases[i]
        var split = entry.find("::")
        if split < 0:
            raise Error("Invalid entry identifier: " + entry)
        var module = String(entry[byte = 0 : split - 5]).replace("/", ".")
        result += (
            "from "
            + module
            + " import "
            + String(entry[byte = split + 2 :])
            + " as "
            + case_alias(entry, areas[i])
            + "\n"
        )
        run_command(
            ["mkdir", "-p", directory + "/case-" + String(i) + "/" + areas[i]]
        )
    result += _settings_source()
    result += (
        "\n\ndef main() raises:\n    var selector ="
        ' getenv("DATAFRAME_FATAL_CASE")\n    _setting("DATAFRAME_THREADS",'
        ' "1")\n'
    )
    for i in range(len(cases)):
        result += (
            "    "
            + ("if" if i == 0 else "elif")
            + " selector == "
            + json_quote(cases[i])
            + ':\n        _setting("COVERAGE_TEST_TMP", '
            + json_quote(directory + "/case-" + String(i) + "/" + areas[i])
            + ")\n        var suite = TestSuite()\n        suite.test["
            + case_alias(cases[i], areas[i])
            + "]()\n        suite^.run()\n"
        )
    result += (
        '    else:\n        raise Error("Unknown fatal case selector: " +'
        ' selector)\n    _setting("DATAFRAME_THREADS", "1")\n'
    )
    return result^


struct RunnerOptions(Copyable, Movable):
    var root: String
    var areas: List[String]
    var contracts: List[String]
    var packaging: String
    var output: String
    var compiler: String
    var capabilities: List[String]
    # "metal" builds with the root dataframe_provider (the default). Any other
    # name is a directory under providers/ put ahead of the root on the import
    # path ("cpu" compiles no GPU code).
    var provider: String
    var shard_index: Int
    var shard_count: Int

    def __init__(
        out self,
        root: String,
        areas: List[String],
        contracts: List[String],
        packaging: String,
        output: String,
        compiler: String,
        capabilities: List[String] = List[String](),
        provider: String = "metal",
        shard_index: Int = 0,
        shard_count: Int = 1,
    ):
        self.root = root.copy()
        self.areas = areas.copy()
        self.contracts = contracts.copy()
        self.packaging = packaging.copy()
        self.output = output.copy()
        self.compiler = compiler.copy()
        self.capabilities = capabilities.copy()
        self.provider = provider.copy()
        self.shard_index = shard_index
        self.shard_count = shard_count


def include_flags(options: RunnerOptions) -> List[String]:
    """The `-I` flags of every build: a non-default provider, then the root."""
    var flags = List[String]()
    if options.provider != "metal":
        flags.append("-I")
        flags.append(options.root + "/providers/" + options.provider)
    flags.append("-I")
    flags.append(options.root)
    return flags^


@fieldwise_init
struct RunEvent(Copyable, Movable):
    var label: String
    var phase: String
    var code: Int
    var log: String


def _fingerprints(root: String) raises -> String:
    var dirs = List[String]()
    for path in [root + "/dataframe", root + "/tests"]:
        if exists(path):
            dirs.append(path.copy())
    var arguments: List[String] = ["find"]
    arguments.extend(dirs^)
    arguments.extend(["-type", "f", "-name", "*.mojo"])
    var listing = capture(arguments)
    if listing[0]:
        raise Error(listing[1])
    var result = String("{")
    var count = 0
    for part in listing[1].split("\n"):
        var path = String(part)
        if not path:
            continue
        # sha256sum on Linux, shasum on macOS; paths are positional arguments.
        var hash = capture(
            [
                "sh",
                "-c",
                (
                    "if command -v sha256sum >/dev/null 2>&1; then sha256sum"
                    ' "$1"; else shasum -a 256 "$1"; fi'
                ),
                "hash",
                path,
            ]
        )
        if hash[0]:
            raise Error(hash[1])
        if count:
            result += ","
        result += (
            json_quote(path) + ":" + json_quote(String(hash[1][byte=0:64]))
        )
        count += 1
    return result + "}"


def execute_suite(options: RunnerOptions, policy: SuitePolicy) raises -> String:
    _ = validate_policy(policy, options.root)
    var selected = selected_modules(policy, options.contracts)
    var areas = List[String]()
    var cases = List[List[String]]()
    var process_cases = List[String]()
    var process_areas = List[String]()
    var skipped = List[String]()
    for capability in options.capabilities:
        if capability not in policy.capabilities:
            raise Error("Unknown runtime capability: " + capability)
    if options.provider != "metal" and not exists(
        options.root + "/providers/" + options.provider
    ):
        raise Error("Unknown provider: " + options.provider)
    if options.packaging not in ["combined", "areas"]:
        raise Error("Unknown packaging mode: " + options.packaging)
    var seen_areas = List[String]()
    for area in options.areas:
        if area in seen_areas:
            raise Error("Select each area only once")
        seen_areas.append(area.copy())
        var ordinary = List[String]()
        for entry in shard_cases(
            cases_for(options.root, area),
            options.shard_index,
            options.shard_count,
        ):
            if options.contracts and case_module(entry) not in selected:
                continue
            var missing = List[String]()
            for requirement in policy.requirements:
                if (
                    requirement.entry == entry
                    and requirement.capability not in options.capabilities
                ):
                    missing.append(requirement.capability.copy())
            if missing:
                print("SKIP(capability):", entry, "requires", missing)
                skipped.append(entry.copy())
                continue
            var index = policy.failure_index(entry)
            if index >= 0 and policy.failures[index].exit_code != 0:
                process_cases.append(entry.copy())
                process_areas.append(area.copy())
            else:
                ordinary.append(entry.copy())
        if ordinary:
            areas.append(area.copy())
            cases.append(ordinary^)
    if not areas and not process_cases and not skipped:
        raise Error("Selected contracts have no tests in selected areas")
    var directory = temporary_directory(options.output)
    var version = capture([options.compiler, "--version"])
    if version[0]:
        raise Error(version[1])
    var fingerprints = _fingerprints(options.root)
    var revision = capture(["git", "-C", options.root, "rev-parse", "HEAD"])
    var status = capture(["git", "-C", options.root, "status", "--porcelain"])
    var events = List[RunEvent]()
    var failures = List[String]()
    var expected = 0
    var programs = List[String]()
    var program_areas = List[List[String]]()
    var program_cases = List[List[List[String]]]()
    if options.packaging == "combined" and areas:
        programs.append("combined")
        program_areas.append(areas.copy())
        program_cases.append(cases.copy())
    elif options.packaging == "areas":
        for i in range(len(areas)):
            programs.append(areas[i].copy())
            program_areas.append([areas[i].copy()])
            program_cases.append([cases[i].copy()])
    for i in range(len(programs)):
        var name = programs[i]
        var folder = directory + "/" + name
        run_command(["mkdir", "-p", folder])
        var source = folder + "/driver.mojo"
        var binary = folder + "/suite"
        write_text(
            source,
            generate_driver(program_areas[i], program_cases[i], folder, policy),
        )
        for group in program_cases[i]:
            for entry in group:
                expected += Int(policy.failure_index(entry) >= 0)
        for phase in [String("compile"), String("execute")]:
            var command: List[String] = [binary.copy()]
            if phase == "compile":
                command = [options.compiler, "build"]
                command.extend(include_flags(options))
                command.append(source.copy())
                command.append("-o")
                command.append(binary.copy())
            print("Contract", phase, name)
            var result = capture(command)
            var log = folder + "/" + phase + ".log"
            write_text(log, result[1])
            print(result[1])
            events.append(
                RunEvent(name.copy(), phase.copy(), result[0], log.copy())
            )
            if result[0]:
                failures.append(name + ":" + phase)
                break
    if process_cases:
        var folder = directory + "/fatal"
        run_command(["mkdir", "-p", folder])
        var source = folder + "/driver.mojo"
        var binary = folder + "/suite"
        write_text(
            source,
            generate_fatal_driver(process_areas, process_cases, folder),
        )
        print("Contract compile fatal-bundle")
        var build_command: List[String] = [options.compiler, "build"]
        build_command.extend(include_flags(options))
        build_command.append(source.copy())
        build_command.append("-o")
        build_command.append(binary.copy())
        var build = capture(build_command)
        write_text(folder + "/compile.log", build[1])
        events.append(
            RunEvent(
                "fatal-bundle", "compile", build[0], folder + "/compile.log"
            )
        )
        if build[0]:
            failures.append("fatal-bundle:compile")
            print(build[1])
        else:
            for i in range(len(process_cases)):
                var entry = process_cases[i]
                var case_folder = directory + "/isolated-" + String(i)
                run_command(["mkdir", "-p", case_folder])
                var result = capture(
                    ["env", "DATAFRAME_FATAL_CASE=" + entry, binary]
                )
                write_text(case_folder + "/execute.log", result[1])
                events.append(
                    RunEvent(
                        entry.copy(),
                        "execute",
                        result[0],
                        case_folder + "/execute.log",
                    )
                )
                ref failure = policy.failures[policy.failure_index(entry)]
                var code = failure.exit_code
                if (
                    result[0] == code
                    and result[1].find(failure.diagnostic) >= 0
                ):
                    expected += 1
                    print(
                        "XFAIL(process):",
                        entry,
                        failure.reason,
                        "exit",
                        result[0],
                    )
                else:
                    failures.append(entry + ":execute")
                    print(
                        "XPASS(process):" if result[0]
                        == 0 else "Unexpected process failure:",
                        entry,
                        "exit",
                        result[0],
                    )
                    print(result[1])
    var data = (
        String('{"runner":"mojo","compiler":')
        + json_quote(String(version[1].strip()))
        + ',"revision":'
        + json_quote(String(revision[1].strip()))
        + ',"dirty":'
        + ("true" if status[1] else "false")
        + ',"packaging":'
        + json_quote(options.packaging)
        + ',"shard_index":'
        + String(options.shard_index)
        + ',"shard_count":'
        + String(options.shard_count)
        + ',"contracts":'
        + json_strings(options.contracts)
        + ',"provider":'
        + json_quote(options.provider)
        + ',"enabled_capabilities":'
        + json_strings(options.capabilities)
        + ',"skipped_capability_cases":'
        + json_strings(skipped)
        + ',"failed":'
        + json_strings(failures)
        + ',"expected_failures":'
        + String(expected)
        + ',"source_sha256":'
        + fingerprints
        + ',"cases":{'
    )
    for i in range(len(areas)):
        if i:
            data += ","
        data += json_quote(areas[i]) + ":" + json_strings(cases[i])
    data += '},"process_cases":' + json_strings(process_cases) + ',"events":['
    for i in range(len(events)):
        if i:
            data += ","
        ref event = events[i]
        data += (
            '{"label":'
            + json_quote(event.label)
            + ',"phase":'
            + json_quote(event.phase)
            + ',"exit_code":'
            + String(event.code)
            + ',"log":'
            + json_quote(event.log)
            + "}"
        )
    data += "]}"
    var metadata = directory + "/run.metadata"
    write_text(metadata, data)
    print(
        "Contract suite failures:",
        failures,
        "declared expected failures:",
        expected,
        "provenance:",
        metadata,
    )
    if failures:
        raise Error("Contract suite failure; provenance: " + metadata)
    return metadata^
