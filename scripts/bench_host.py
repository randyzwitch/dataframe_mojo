"""Host checks shared by the benchmark drivers."""

from pathlib import Path
import subprocess


def compiler_processes():
    output = subprocess.check_output(["ps", "-eo", "pid,comm,args"], text=True)
    found = []
    for line in output.splitlines()[1:]:
        fields = line.split(None, 2)
        if len(fields) < 3:
            continue
        name = Path(fields[1]).name
        # `mojo run`, `test` and `package` compile too, and every one of
        # them competes with a timed run for the same cores.
        if name in {"clang", "clang++", "cc1", "cc1plus", "rustc"} or (
            name == "mojo"
            and any(
                f" {verb} " in fields[2]
                for verb in ("build", "run", "test", "package", "precompile")
            )
        ):
            found.append((fields[0], name))
    return found
