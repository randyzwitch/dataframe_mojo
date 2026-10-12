"""Mojo-native suite orchestration; POSIX processes only, no Python runtime.

The catalog defines coverage. Discovery verifies that every ordinary Mojo test
has an assignment. Process-isolated failures require both a precise diagnostic
and an exact exit code, so crashes cannot hide unrelated failures.
"""
from std.ffi import external_call
from std.os import getenv
from std.os.path import exists
from std.memory import Pointer
from dataframe.json_value import JsonValue, json_quote

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
    return (exit_code(status), String(unsafe_from_utf8=output^))


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
    return String(unsafe_from_utf8=bytes^)


def strings(value: JsonValue) raises -> List[String]:
    var result = List[String]()
    for item in value.items():
        result.append(item.string())
    return result^


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


def cases_for(
    catalog: JsonValue, root: String, area: String
) raises -> List[String]:
    var discovered = discovered_cases(root, area)
    var declared = List[String]()
    var prefix = "tests/" + area_directory(area) + "/"
    for contract in catalog.get("contracts").items():
        for witness in strings(contract.get("witnesses")):
            if witness.startswith(prefix):
                _append_unique(declared, witness)
    if not discovered or len(declared) != len(discovered):
        raise Error("Contract catalog/entry mismatch in " + area)
    for entry in discovered:
        if entry not in declared:
            raise Error(
                "Contract catalog/entry mismatch in " + area + ": " + entry
            )
    return discovered^


def selected_witnesses(
    catalog: JsonValue, contracts: List[String]
) raises -> List[String]:
    var known = List[String]()
    var witnesses = List[String]()
    for contract in catalog.get("contracts").items():
        var name = contract.get("id").string()
        known.append(name.copy())
        if name in contracts:
            for witness in strings(contract.get("witnesses")):
                _append_unique(witnesses, witness)
    for name in contracts:
        if name not in known:
            raise Error("Unknown contract: " + name)
    return witnesses^


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


def generate_driver(
    areas: List[String],
    cases: List[List[String]],
    directory: String,
    known: JsonValue,
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
            if not known.has(entry):
                continue
            var failure = known.get(entry)
            var alias = case_alias(entry, areas[i])
            result += (
                "\n\ndef "
                + alias
                + "__known_failure() raises:\n    var failed = False\n    try:\n        "
                + alias
                + "()\n    except error:\n        if String(error).find("
                + json_quote(failure.get("error_contains").string())
                + ") < 0:\n            raise error\n        print("
                + json_quote(
                    "XFAIL: " + entry + ": " + failure.get("reason").string()
                )
                + ", error)\n        failed = True\n    if not failed:\n        raise Error("
                + json_quote(
                    "XPASS: " + entry + "; remove known-failure annotation"
                )
                + ")\n"
            )
    result += '\n\ndef _setting(name: String, value: String) raises:\n    var key = List[UInt8]()\n    key.extend(name.as_bytes())\n    key.append(0)\n    var contents = List[UInt8]()\n    contents.extend(value.as_bytes())\n    contents.append(0)\n    var status = external_call["setenv", Int32](Int(key.unsafe_ptr()), Int(contents.unsafe_ptr()), Int32(1))\n    _ = key^\n    _ = contents^\n    if status != 0:\n        raise Error("Cannot set suite environment: " + name)\n\n\ndef main() raises:\n    var failed = List[String]()\n'
    for i in range(len(areas)):
        var area = areas[i]
        result += (
            '    _setting("DATAFRAME_THREADS", "1")\n    _setting("COVERAGE_TEST_TMP", '
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
                + ("__known_failure" if known.has(entry) else "")
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
    result += '    _setting("DATAFRAME_THREADS", "1")\n    if len(failed):\n        raise Error("Contract area failure")\n'
    return result^


struct RunnerOptions(Copyable, Movable):
    var root: String
    var areas: List[String]
    var contracts: List[String]
    var packaging: String
    var output: String
    var compiler: String
    var capabilities: List[String]

    def __init__(
        out self,
        root: String,
        areas: List[String],
        contracts: List[String],
        packaging: String,
        output: String,
        compiler: String,
        capabilities: List[String] = List[String](),
    ):
        self.root = root.copy()
        self.areas = areas.copy()
        self.contracts = contracts.copy()
        self.packaging = packaging.copy()
        self.output = output.copy()
        self.compiler = compiler.copy()
        self.capabilities = capabilities.copy()


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
    arguments.extend(
        ["-type", "f", "(", "-name", "*.mojo", "-o", "-name", "plan.json", ")"]
    )
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
                'if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi',
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


def execute_suite(options: RunnerOptions) raises -> String:
    var catalog = JsonValue(read_text(options.root + "/tests/plan.json"))
    var selected = selected_witnesses(catalog, options.contracts)
    var areas = List[String]()
    var cases = List[List[String]]()
    var process_cases = List[String]()
    var process_areas = List[String]()
    var skipped = List[String]()
    var capability_catalog = catalog.get("capabilities")
    var requirements = catalog.get("required_capabilities")
    if capability_catalog.kind() == 110:
        capability_catalog = JsonValue("{}")
    if requirements.kind() == 110:
        requirements = JsonValue("{}")
    for capability in options.capabilities:
        if not capability_catalog.has(capability):
            raise Error("Unknown runtime capability: " + capability)
    var known = catalog.get("known_failures")
    var process_known = catalog.get("known_process_failures")
    if process_known.kind() == 110:
        process_known = JsonValue("{}")
    if options.packaging not in ["combined", "areas"]:
        raise Error("Unknown packaging mode: " + options.packaging)
    for area in options.areas:
        if area in areas:
            raise Error("Select each area only once")
        var ordinary = List[String]()
        for entry in cases_for(catalog, options.root, area):
            if options.contracts and entry not in selected:
                continue
            if requirements.has(entry):
                var missing = List[String]()
                for capability in strings(requirements.get(entry)):
                    if capability not in options.capabilities:
                        missing.append(capability.copy())
                if missing:
                    print("SKIP(capability):", entry, "requires", missing)
                    skipped.append(entry.copy())
                    continue
            if process_known.has(entry):
                process_cases.append(entry.copy())
                process_areas.append(area.copy())
            else:
                ordinary.append(entry.copy())
        if ordinary:
            areas.append(area.copy())
            cases.append(ordinary^)
    if not areas and not process_cases and not skipped:
        raise Error(
            "Selected contracts have no implemented witnesses in selected areas"
        )
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
            generate_driver(program_areas[i], program_cases[i], folder, known),
        )
        for group in program_cases[i]:
            for entry in group:
                expected += Int(known.has(entry))
        for phase in [String("compile"), String("execute")]:
            var command: List[String] = [
                options.compiler,
                "build",
                "-I",
                options.root,
                source,
                "-o",
                binary,
            ] if phase == "compile" else [binary.copy()]
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
    for i in range(len(process_cases)):
        var entry = process_cases[i]
        var area = process_areas[i]
        var folder = directory + "/isolated-" + String(i)
        run_command(["mkdir", "-p", folder])
        var source = folder + "/driver.mojo"
        var binary = folder + "/suite"
        write_text(
            source,
            generate_driver(
                [area.copy()], [[entry.copy()]], folder, JsonValue("{}")
            ),
        )
        var build = capture(
            [
                options.compiler,
                "build",
                "-I",
                options.root,
                source,
                "-o",
                binary,
            ]
        )
        write_text(folder + "/compile.log", build[1])
        events.append(
            RunEvent(entry.copy(), "compile", build[0], folder + "/compile.log")
        )
        if build[0]:
            failures.append(entry + ":compile")
            print(build[1])
            continue
        var result = capture([binary])
        write_text(folder + "/execute.log", result[1])
        events.append(
            RunEvent(
                entry.copy(), "execute", result[0], folder + "/execute.log"
            )
        )
        var failure = process_known.get(entry)
        var code = Int(failure.get("exit_code").text)
        if (
            result[0] == code
            and result[1].find(failure.get("error_contains").string()) >= 0
        ):
            expected += 1
            print(
                "XFAIL(process):",
                entry,
                failure.get("reason").string(),
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
        + ',"contracts":'
        + json_strings(options.contracts)
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
