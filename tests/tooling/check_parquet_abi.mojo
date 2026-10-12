"""The public native Parquet library exports exactly its maintained C ABI."""
from std.sys import argv
from std.testing import assert_equal
from tests.tooling.contract_runner import capture, read_text


def main() raises:
    var args = argv()
    if len(args) > 2:
        raise Error("Usage: check_parquet_abi.mojo [LIBRARY]")
    var platform = capture(["uname", "-s"])
    if platform[0]:
        raise Error(platform[1])
    var darwin = String(platform[1].strip()) == "Darwin"
    var library = String(args[1]) if len(
        args
    ) == 2 else "build/dfparquet/libdfparquet" + (".dylib" if darwin else ".so")
    var command: List[String] = ["nm", "-gjU", library]
    if not darwin:
        command = [
            "nm",
            "-D",
            "--defined-only",
            "--format=posix",
            library,
        ]
    var symbols = capture(command)
    if symbols[0]:
        raise Error(symbols[1])
    var actual = List[String]()
    for line in symbols[1].split("\n"):
        var value = String(line.strip())
        if not value:
            continue
        var name = value if darwin else String(value.split(" ")[0])
        actual.append(
            String(name[byte=1:]) if darwin and name.startswith("_") else name
        )
    var expected = List[String]()
    var specification = read_text("native/dfparquet/dfparquet.sym")
    for line in specification.split("\n"):
        var value = String(line.strip())
        if value:
            expected.append(
                String(value[byte=1:]) if value.startswith("_") else value
            )
    sort(actual)
    sort(expected)
    assert_equal(actual, expected)
    print("Parquet exported C ABI matches the maintained symbol allowlist")
