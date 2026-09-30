"""Write driver files that run several test modules in one binary.

Compiling dominates the test run: each module is its own program, and each
program compiles the library code it uses again, 30 to 180 seconds apiece,
while running every test takes under a minute in total. A driver imports the
test functions of several modules and runs each module's tests as its own
TestSuite, so the library is compiled once per driver instead of once per
module.

Usage: scripts/test_drivers.py OUT_DIR GROUPS tests/test_a.mojo ...

Writes OUT_DIR/driver_N.mojo for N in 0..GROUPS-1 and prints the path of
every file to run: the drivers, then any module that cannot join one (a
module whose main does more than run its tests, such as the Parquet modules
that report a missing library and skip). Modules are dealt to drivers in
order of their source size, largest first, onto the driver with the least so
far, which evens out the drivers' compile times.

A driver resets DATAFRAME_THREADS to its starting value before each module,
since several modules set it, runs every module even after one fails, prints
`== tests/test_x.mojo` before each, and fails at the end if any did.
"""
import os
import re
import sys

TEST = re.compile(r"^def (test_\w+)\(\) raises", re.MULTILINE)
# A main that only runs the module's tests; anything else keeps its module
# a program of its own.
STANDARD_MAIN = re.compile(
    r"^def main\(\) raises:\n"
    r"    TestSuite\.discover_tests\[__functions_in_module\(\)\]\(\)\.run\(\)\n"
    r"(?:\s*\n)*(?:(?=\S)|\Z)",
    re.MULTILINE,
)


def main() -> None:
    out, groups, paths = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    joinable, alone = [], []
    for path in paths:
        source = open(path).read()
        if not STANDARD_MAIN.search(source):
            alone.append(path)
        else:
            joinable.append((os.path.getsize(path), path, TEST.findall(source)))
    os.makedirs(out, exist_ok=True)
    buckets = [[] for _ in range(max(1, min(groups, len(joinable))))]
    sizes = [0] * len(buckets)
    for size, path, tests in sorted(joinable, reverse=True):
        smallest = sizes.index(min(sizes))
        buckets[smallest].append((path, tests))
        sizes[smallest] += size
    drivers = []
    for n, bucket in enumerate(buckets):
        if not bucket:
            continue
        bucket.sort()
        lines = ["from std.ffi import external_call", "from std.os import getenv", "from std.testing import TestSuite", ""]
        for path, tests in bucket:
            module = os.path.basename(path)[: -len(".mojo")]
            for test in tests:
                lines.append(f"from {module} import {test} as {module}__{test}")
        lines += [
            "",
            "",
            "def _reset(threads: String):",
            '    """Start each module with the thread setting the run began with."""',
            '    var name = String("DATAFRAME_THREADS")',
            "    if threads:",
            "        var value = threads",
            '        _ = external_call["setenv", Int32](',
            "            Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)",
            "        )",
            "        _ = value^",
            "    else:",
            '        _ = external_call["unsetenv", Int32](Int(name.unsafe_ptr()))',
            "    _ = name^",
            "",
            "",
            "def main() raises:",
            '    var threads = getenv("DATAFRAME_THREADS")',
            "    var failed = List[String]()",
        ]
        for path, tests in bucket:
            module = os.path.basename(path)[: -len(".mojo")]
            lines += [
                "    _reset(threads)",
                f'    print("== {path}")',
                f"    var {module} = TestSuite()",
            ]
            lines += [f"    {module}.test[{module}__{test}]()" for test in tests]
            lines += [
                "    try:",
                f"        {module}^.run()",
                "    except:",
                f'        failed.append("{path}")',
            ]
        lines += [
            "    for name in failed:",
            '        print("FAILED:", name)',
            "    if len(failed) > 0:",
            '        raise Error(String(len(failed)) + " module(s) failed")',
            "",
        ]
        driver = os.path.join(out, f"driver_{n}.mojo")
        with open(driver, "w") as f:
            f.write("\n".join(lines))
        drivers.append(driver)
    print("\n".join(drivers + alone))


if __name__ == "__main__":
    main()
