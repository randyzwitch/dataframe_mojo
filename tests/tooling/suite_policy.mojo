"""Execution exceptions and convenient module groups for the maintained suite.

Tests are discovered from source; this policy neither registers ordinary tests
nor describes coverage. Remove an expected failure when its exact defect is fixed.

Each entry is a comment in the test module it is about, so a change adding
tests edits only its own files:

- `# contract-group: NAME` on its own line at the top of a module puts the
  module in contract group NAME (one line per group).
- `# requires: CAPABILITY` directly above a test's `def` runs the test only
  where CAPABILITY is available (`metal` or `mojo_gpu`).
- `# expected-failure: REASON` and `# expected-diagnostic: TEXT` directly
  above a test's `def` record a known defect: why it fails and the text its
  failure must contain. Either may continue on further lines with the same
  prefix. `# expected-exit: variant-abort` marks a defect that ends the
  process through a checked variant abort.

Markers sit at the start of a line, with only other comments between them and
the `def` they describe; a marker followed by anything else is an error.
"""

from std.os import listdir
from std.os.path import isdir
from std.sys import CompilationTarget


@fieldwise_init
struct ExpectedFailure(Copyable, Movable):
    var entry: String
    var reason: String
    var diagnostic: String
    var exit_code: Int


@fieldwise_init
struct CapabilityRequirement(Copyable, Movable):
    var entry: String
    var capability: String


@fieldwise_init
struct ContractGroup(Copyable, Movable):
    var name: String
    var modules: List[String]


@fieldwise_init
struct SuitePolicy(Copyable, Movable):
    var failures: List[ExpectedFailure]
    var requirements: List[CapabilityRequirement]
    var groups: List[ContractGroup]
    var capabilities: List[String]

    def failure_index(self, entry: String) -> Int:
        for i in range(len(self.failures)):
            if self.failures[i].entry == entry:
                return i
        return -1


def empty_policy() -> SuitePolicy:
    return SuitePolicy(
        List[ExpectedFailure](),
        List[CapabilityRequirement](),
        List[ContractGroup](),
        List[String](),
    )


def mojo_variant_abort_exit_code() -> Int:
    # The same checked variant abort traps as SIGILL on Linux and SIGTRAP on
    # Apple Silicon macOS. Each known fatal defect still needs its exact
    # diagnostic and the one status for the platform executing the suite.
    return 133 if CompilationTarget.is_macos() else 132


# Where test modules live, relative to the repository root (the runner's
# working directory).
comptime _TEST_ROOT = "tests"
comptime _GROUP = "# contract-group:"
comptime _REQUIRES = "# requires:"
comptime _FAILURE = "# expected-failure:"
comptime _DIAGNOSTIC = "# expected-diagnostic:"
comptime _EXIT = "# expected-exit:"


def production_policy() raises -> SuitePolicy:
    """The policy written as markers in the test modules under `tests`."""
    return policy_from(_TEST_ROOT)


def policy_from(root: String) raises -> SuitePolicy:
    """The policy the markers in every module under `root` describe, its
    groups in name order and their modules in path order."""
    var policy = empty_policy()
    policy.capabilities = ["metal", "mojo_gpu"]
    var names = List[String]()
    var members = List[List[String]]()
    for path in _modules(root):
        _read_markers(path, policy, names, members)
    var order = List[Int](capacity=len(names))
    for i in range(len(names)):
        order.append(i)
    for i in range(1, len(order)):
        var j = i
        while j > 0 and names[order[j - 1]] > names[order[j]]:
            order.swap_elements(j - 1, j)
            j -= 1
    for i in order:
        policy.groups.append(ContractGroup(names[i], members[i].copy()))
    return policy^


def _modules(root: String) raises -> List[String]:
    """Every `.mojo` file under root, in sorted path order."""
    var entries = listdir(root)
    sort(entries)
    var found = List[String]()
    for name in entries:
        var path = root + "/" + name
        if isdir(path):
            found.extend(_modules(path))
        elif name.endswith(".mojo"):
            found.append(path)
    return found^


def _value(line: String, marker: StaticString) -> String:
    return String(line[byte = marker.byte_length() :].strip())


def _read_markers(
    path: String,
    mut policy: SuitePolicy,
    mut names: List[String],
    mut members: List[List[String]],
) raises:
    var text: String
    with open(path, "r") as file:
        text = file.read()
    var requires = List[String]()
    var reason = String()
    var diagnostic = String()
    var exit_code = 0
    var pending = False
    for raw in text.splitlines():
        var line = String(raw)
        if line.startswith(_GROUP):
            var name = _value(line, _GROUP)
            var at = -1
            for i in range(len(names)):
                if names[i] == name:
                    at = i
            if at < 0:
                names.append(name)
                members.append(List[String]())
                at = len(names) - 1
            members[at].append(path)
        elif line.startswith(_REQUIRES):
            requires.append(_value(line, _REQUIRES))
            pending = True
        elif line.startswith(_FAILURE):
            reason = (
                _value(line, _FAILURE) if not reason else reason
                + " "
                + _value(line, _FAILURE)
            )
            pending = True
        elif line.startswith(_DIAGNOSTIC):
            diagnostic = (
                _value(line, _DIAGNOSTIC) if not diagnostic else diagnostic
                + " "
                + _value(line, _DIAGNOSTIC)
            )
            pending = True
        elif line.startswith(_EXIT):
            var value = _value(line, _EXIT)
            exit_code = (
                mojo_variant_abort_exit_code() if value
                == "variant-abort" else Int(value)
            )
            pending = True
        elif line.startswith("def "):
            if pending:
                var name = String(line[byte=4:].split("(")[0])
                var entry = path + "::" + name
                for capability in requires:
                    policy.requirements.append(
                        CapabilityRequirement(entry, capability)
                    )
                if reason or diagnostic:
                    policy.failures.append(
                        ExpectedFailure(entry, reason, diagnostic, exit_code)
                    )
                requires = List[String]()
                reason = String()
                diagnostic = String()
                exit_code = 0
                pending = False
        elif pending and not line.startswith("#"):
            raise Error(
                path
                + ": a policy marker must sit directly above a `def`: "
                + line
            )
    if pending:
        raise Error(path + ": a policy marker at the end of the file")
