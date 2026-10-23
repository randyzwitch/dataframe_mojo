"""Generate native Unicode 16 regex property tables from Rust regex 1.12.2.

Run `python3 scripts/regex_property_tables.py [--source-dir DIR]`.
The optional directory is a cache of the pinned upstream unicode_tables/*.rs;
without it, files are downloaded from the versioned upstream repository.
No downloaded sources or Python dependencies are needed at runtime.
"""
import argparse
import ast
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.12.2"
URL = f"https://raw.githubusercontent.com/rust-lang/regex/{VERSION}/regex-syntax/src/unicode_tables/"
KINDS = {
    "General_Category": "general_category",
    "Script": "script",
    "Script_Extensions": "script_extension",
    "Age": "age",
    "Grapheme_Cluster_Break": "grapheme_cluster_break",
    "Sentence_Break": "sentence_break",
    "Word_Break": "word_break",
}


def normalize(value):
    return "".join(c.lower() for c in value if c not in " _-\t\n\r\f\v")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--check-oracle",
        action="store_true",
        help="verify every generated set over all Unicode scalar values with Polars",
    )
    args = parser.parse_args()

    def read(name):
        if args.source_dir:
            return (args.source_dir / (name + ".rs")).read_text()
        return urllib.request.urlopen(URL + name + ".rs").read().decode()

    property_names = dict(
        re.findall(r'\("([^"]+)", "([^"]+)"\)', read("property_names"))
    )
    values = {}
    for name, body in re.findall(
        r'\(\s*"([^"]+)",\s*&\[(.*?)\],\s*\)', read("property_values"), re.S
    ):
        values[name] = dict(re.findall(r'\("([^"]+)", "([^"]+)"\)', body))
    tables = {}
    for kind, filename in {**KINDS, "Binary": "property_bool"}.items():
        text = read(filename)
        by_name = dict(
            re.findall(
                r'\("([^"]+)", ([A-Z0-9_]+)\)', text.split("pub const", 2)[1]
            )
        )
        constants = {}
        for const, body in re.findall(
            r"pub const ([A-Z0-9_]+):[^=]+?=\s*&\[(.*?)\];", text, re.S
        ):

            def codepoint(value):
                escaped = re.sub(
                    r"\\u\{([0-9a-fA-F]+)\}",
                    lambda m: "\\U%08x" % int(m[1], 16),
                    value,
                )
                return ord(ast.literal_eval("'" + escaped + "'"))

            ranges = [
                (codepoint(a), codepoint(b))
                for a, b in re.findall(
                    r"\('((?:\\.|[^'\\])*)', '((?:\\.|[^'\\])*)'\)", body
                )
            ]
            # The age property means assigned *by* this version, not only
            # code points first assigned in this version (regex-syntax).
            constants[const] = ranges
        for name, const in by_name.items():
            tables[kind, name] = constants[const]
    for name in sorted(set(values["Age"].values())):
        if ("Age", name) not in tables:
            continue
        version = tuple(map(int, name[1:].split("_")))
        combined = []
        for (kind, other), ranges in tables.items():
            if (
                kind == "Age"
                and tuple(map(int, other[1:].split("_"))) <= version
            ):
                combined.extend(ranges)
        tables["Age", name] = sorted(combined)
    # Rust also accepts these special general categories.
    tables["General_Category", "Any"] = [(0, 0xD7FF), (0xE000, 0x10FFFF)]
    tables["General_Category", "ASCII"] = [(0, 127)]
    unassigned = tables["General_Category", "Unassigned"]
    assigned, at = [], 0
    for low, high in unassigned:
        if at < low:
            assigned.append((at, low - 1))
        at = high + 1
    if at <= 0x10FFFF:
        assigned.append((at, 0x10FFFF))
    tables["General_Category", "Assigned"] = assigned
    aliases = {}
    for prop, kind in property_names.items():
        if kind in KINDS:
            for alias, name in values[kind].items():
                if (kind, name) in tables:
                    aliases[prop + "=" + alias] = (kind, name)
        elif ("Binary", kind) in tables:
            aliases[prop] = ("Binary", kind)
    # A bare property is a category, script, or binary property, in Rust's
    # precedence order. Prefix `Is` is ignored during lookup.
    for kind in ["Script", "General_Category"]:
        for alias, name in values[kind].items():
            if (kind, name) in tables:
                aliases[alias] = (kind, name)
    for special in ["Any", "ASCII", "Assigned"]:
        aliases[normalize(special)] = ("General_Category", special)
    tables = {key: tables[key] for key in set(aliases.values())}
    blobs, offsets = [], {}
    canonical_tables = {}
    at = 0
    for key, ranges in sorted(tables.items()):
        # Sort/merge, and exclude surrogate code points from scalar sets.
        merged = []
        for low, high in sorted(ranges):
            for a, b in [(low, min(high, 0xD7FF)), (max(low, 0xE000), high)]:
                if a > b:
                    continue
                if merged and a <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
        canonical_tables[key] = merged
        blob = (
            " ".join(f"{a:X}" if a == b else f"{a:X}-{b:X}" for a, b in merged)
            + "\n"
        )
        offsets[key] = (at, at + len(blob))
        at += len(blob)
        blobs.append(blob)
    index = "".join(
        f"{alias}:{offsets[key][0]}:{offsets[key][1]}\n"
        for alias, key in sorted(aliases.items())
    )
    if args.check_oracle:
        import polars as pl

        cps = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
        column = pl.Series("cp", cps, dtype=pl.UInt32)
        text_column = pl.Series("text", [chr(cp) for cp in cps])
        failures = []
        for (kind, name), expected in canonical_tables.items():
            property_name = next(
                alias
                for alias, target in aliases.items()
                if target == (kind, name)
            )
            actual = column.filter(
                text_column.str.contains(r"^\p{" + property_name + r"}$")
            ).to_list()
            actual_ranges = []
            for cp in actual:
                if actual_ranges and cp == actual_ranges[-1][1] + 1:
                    actual_ranges[-1] = (actual_ranges[-1][0], cp)
                else:
                    actual_ranges.append((cp, cp))
            if actual_ranges != expected:
                failures.append(property_name)
        if failures:
            raise AssertionError(
                "Property sets differ from Polars: " + ", ".join(failures)
            )
        print(
            f"All {len(canonical_tables)} sets match Polars {pl.__version__} over every Unicode scalar"
        )
    header = '''"""Unicode 16 property tables generated by scripts/regex_property_tables.py.
Source: rust-lang/regex 1.12.2, regex-syntax/src/unicode_tables (MIT/Apache-2.0).
Aliases use Unicode loose matching. The index maps each alias to byte offsets
in RANGES, which holds canonical scalar-value ranges. Do not edit by hand.
"""\n'''
    text = (
        header
        + '\ncomptime PROPERTY_INDEX = StaticString(\n    """'
        + index
        + '"""\n)\n\ncomptime PROPERTY_RANGES = StaticString(\n    """'
        + "".join(blobs)
        + '"""\n)\n'
    )
    (ROOT / "dataframe/regex/property_tables.mojo").write_text(text)
    print(f"{len(aliases)} aliases, {len(tables)} sets, {len(text)} bytes")


if __name__ == "__main__":
    main()
