"""Deterministic local dataset path expansion shared by file readers."""
from std.algorithm import sort
from std.ffi import external_call
from std.memory import Pointer
from std.os import listdir
from std.os.path import exists, isdir


def _c_path(value: String) raises -> List[UInt8]:
    var bytes = List[UInt8](capacity=value.byte_length() + 1)
    for byte in value.as_bytes():
        if byte == 0:
            raise Error("File source paths must not contain NUL")
        bytes.append(byte)
    bytes.append(0)
    return bytes^


def has_pattern(path: String) -> Bool:
    """Whether a path uses glob metacharacters or a glob escape."""
    return "*" in path or "?" in path or "[" in path or "\\" in path


def _matches(pattern: String, name: String) raises -> Bool:
    var wanted = _c_path(pattern)
    var candidate = _c_path(name)
    return (
        external_call["fnmatch", Int32](
            wanted.unsafe_ptr(), candidate.unsafe_ptr(), Int32(0)
        )
        == 0
    )


def _symlink(path: String) raises -> Bool:
    # A one-byte read identifies a symlink without depending on the host's
    # stat layout. Recursive ** does not traverse directory symlinks.
    var source = _c_path(path)
    var byte = UInt8(0)
    return (
        external_call["readlink", Int](
            source.unsafe_ptr(), Pointer(to=byte), Int(1)
        )
        >= 0
    )


def _join(parent: String, name: String) -> String:
    return (
        name if parent == "" else parent + ("" if parent == "/" else "/") + name
    )


def _expand(
    parent: String, parts: List[String], at: Int, mut out: List[String]
) raises:
    if at == len(parts):
        if parent != "" and exists(parent) and not isdir(parent):
            out.append(parent)
        return
    var part = parts[at]
    var directory = "." if parent == "" else parent
    if part == "**":
        _expand(parent, parts, at + 1, out)
        if not isdir(directory):
            return
        for name in listdir(directory):
            if name.startswith("."):
                continue
            var child = _join(parent, name)
            if isdir(child):
                if not _symlink(child):
                    _expand(child, parts, at, out)
            elif at + 1 == len(parts):
                _expand(child, parts, at + 1, out)
        return
    if not has_pattern(part):
        _expand(_join(parent, part), parts, at + 1, out)
        return
    if not isdir(directory):
        return
    for name in listdir(directory):
        if name.startswith(".") and not part.startswith("."):
            continue
        if _matches(part, name):
            _expand(_join(parent, name), parts, at + 1, out)


def resolve_paths(
    sources: List[String], glob: Bool = True
) raises -> List[String]:
    """Expand each source then sort paths, preserving explicit repetitions.

    `**` spans zero or more directories. Dotfiles require an explicit dot
    component; recursive globs do not follow directory symlinks. Every source
    must match at least one file. No file contents are opened here.
    """
    if len(sources) == 0:
        raise Error("File sources require at least one path")
    var paths = List[String]()
    for source in sources:
        _ = _c_path(source)
        if source == "":
            raise Error("File sources require nonempty paths")
        if not glob or not has_pattern(source):
            if not exists(source) or isdir(source):
                raise Error("File source does not name a file: " + source)
            paths.append(source)
            continue
        var parts = List[String]()
        for part in source.split("/"):
            if part != "":
                parts.append(String(part))
        var matched = List[String]()
        _expand("/" if source.startswith("/") else "", parts, 0, matched)
        if len(matched) == 0:
            raise Error("No files matched source: " + source)
        paths.extend(matched)
    sort(paths)
    return paths^
