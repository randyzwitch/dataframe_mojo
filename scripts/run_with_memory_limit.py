#!/usr/bin/env python3
"""Run a compiled CPU query in a memory-limited Linux cgroup."""

import argparse
import os
from pathlib import Path
import shutil
import sys


CGROUP_ROOT = Path("/sys/fs/cgroup")


def positive_bytes(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("bytes must be a positive integer") from error
    if not 0 < result < 2**63:
        raise argparse.ArgumentTypeError("bytes must be between 1 and 2**63 - 1")
    return result


def current_cgroup():
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::/"):
            path = (CGROUP_ROOT / line[4:]).resolve()
            if path.is_relative_to(CGROUP_ROOT):
                return path
    raise OSError("a cgroup v2 mount at /sys/fs/cgroup is required")


def verify_limit(path, limit):
    """Refuse to execute if the manager did not install every required limit."""
    actual = (path / "memory.max").read_text().strip()
    if actual == "max" or int(actual) > limit:
        raise OSError("the requested memory ceiling was not installed")
    if (path / "memory.swap.max").read_text().strip() != "0":
        raise OSError("swap must be disabled for the query cgroup")
    if (path / "memory.oom.group").read_text().strip() != "1":
        raise OSError("whole-query termination on memory exhaustion is required")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Requires Linux cgroup v2 and a running systemd user manager. Compile "
            "before launching. MemoryMax covers charged memory for the query and "
            "its children; swap is disabled. Exhaustion can kill the query. "
            "Device memory is not included. There is no spill to disk."
        ),
    )
    parser.add_argument("--bytes", type=positive_bytes, required=True)
    parser.add_argument("--in-scope", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide a compiled query command after --")
    if sys.platform != "linux":
        parser.error("hard memory enforcement is supported only on Linux")

    try:
        if args.in_scope:
            verify_limit(current_cgroup(), args.bytes)
            os.execvp(command[0], command)
        else:
            if "memory" not in (CGROUP_ROOT / "cgroup.controllers").read_text().split():
                raise OSError("cgroup v2 memory controller is unavailable")
            runner = shutil.which("systemd-run")
            if runner is None:
                raise OSError("systemd-run and a running user manager are required")
            # A scope preserves cwd, environment, stdio and normal exit status.
            # Re-enter to verify the kernel settings before executing the query.
            os.execv(
                runner,
                [
                    runner,
                    "--user",
                    "--scope",
                    "--quiet",
                    "--collect",
                    "--expand-environment=no",
                    "-p",
                    "MemoryAccounting=yes",
                    "-p",
                    f'MemoryMax={args.bytes}',
                    "-p",
                    "MemorySwapMax=0",
                    "-p",
                    "OOMPolicy=kill",
                    "--",
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--bytes",
                    str(args.bytes),
                    "--in-scope",
                    "--",
                    *command,
                ],
            )
    except (OSError, ValueError) as error:
        print(
            f'Unable to run query with memory limit {args.bytes}: {error}',
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
