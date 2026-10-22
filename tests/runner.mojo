"""Discover dataframe tests and run them with a Mojo-native process harness."""
from std.sys import argv
from tests.tooling.contract_runner import (
    RunnerOptions,
    execute_suite,
    AREAS,
    DEFAULT_PACKAGING,
)
from tests.tooling.suite_policy import production_policy


def main() raises:
    var arguments = argv()
    var areas = List[String]()
    var contracts = List[String]()
    var capabilities = List[String]()
    var packaging = String(DEFAULT_PACKAGING)
    var output = String("build/contract-tests")
    var compiler = String("mojo")
    var provider = String("")
    var i = 1
    while i < len(arguments):
        var option = String(arguments[i])
        if option == "--help" or option == "-h":
            print(
                "Usage: mojo run -I . tests/runner.mojo [--areas public storage kernels planner execution interchange] [--contracts CONTRACT ...] [--capabilities CAPABILITY ...] [--provider metal] [--packaging areas|combined] [--output DIRECTORY] [--compiler EXECUTABLE]\nDefault packaging: areas (separate executables; continue after area failures)."
            )
            return
        i += 1
        if (
            option == "--areas"
            or option == "--contracts"
            or option == "--capabilities"
        ):
            var count = 0
            while i < len(arguments) and not String(arguments[i]).startswith(
                "--"
            ):
                if option == "--areas":
                    areas.append(String(arguments[i]))
                elif option == "--contracts":
                    contracts.append(String(arguments[i]))
                else:
                    capabilities.append(String(arguments[i]))
                i += 1
                count += 1
            if not count:
                raise Error(option + " requires at least one value")
            continue
        if i >= len(arguments):
            raise Error("Missing value for " + option)
        var value = String(arguments[i])
        if option == "--packaging":
            packaging = value^
        elif option == "--output":
            output = value^
        elif option == "--compiler":
            compiler = value^
        elif option == "--provider":
            provider = value^
        else:
            raise Error("Unknown argument: " + option)
        i += 1
    for capability in capabilities:
        if capability in ["metal", "mojo_gpu"] and provider != "metal":
            raise Error(
                "Capability "
                + capability
                + " needs the Metal provider in the build: pass --provider"
                " metal"
            )
    if not areas:
        comptime for index in range(len(AREAS)):
            areas.append(String(AREAS[index]))
    _ = execute_suite(
        RunnerOptions(
            ".",
            areas,
            contracts,
            packaging,
            output,
            compiler,
            capabilities,
            provider,
        ),
        production_policy(),
    )
