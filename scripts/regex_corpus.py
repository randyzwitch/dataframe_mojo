"""Write tests/fixtures/regex/polars_corpus.tsv: regex results from Polars
for the differential test of the regex engine.

Each row is a pattern and a haystack, then what Polars returns for them:
`contains`, `find` (byte offset of the first match), `count_matches`,
`extract_all`, `replace_all` with "<$0>" (which shows where every match
is), `replace` with "[${1}]", `extract` of group 1, and `split` with
`literal=False`. A pattern Polars rejects has the single result ERROR.

Fields escape backslash, tab, newline and other control characters as
`\\\\`, `\\t`, `\\n` and `\\xHH`; `\\N` is null; a list is its length then
its elements, separated by `\\x1f` (written raw).

    pixi run -e oracle python3 scripts/regex_corpus.py
"""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "regex" / "polars_corpus.tsv"

HAYSTACKS = [
    "",
    "a",
    "abc",
    "aaa",
    "abcabcabc",
    "ab ab",
    "foo bar baz",
    "Foo BAR baZ",
    "x1y22z333",
    "hello, world!",
    "été café naïve",
    "日本語のテキスト",
    "Straße STRASSE straẞe",
    "line1\nline2\n",
    "\n",
    "\nfoo\n\nbar",
    "a.b*c+d?",
    "tab\there",
    "αβγ ΑΒΓ σς",
    "emoji 😀 ok 😀",
    "KELVIN K K k",
    "  spaced  out  ",
    "x-y_z",
    "2024-01-15 and 1999-12-31",
    "bob@example.com, amy@test.org",
    "the special big requests here",
    "١٢٣ ٤٥",
    "_under_score_",
    "0",
    "ababab",
    "https://www.example.com/a/b?q=1",
    "http://foo.org/",
]

PATTERNS = [
    # literals and escapes
    "a", "abc", "b", "foo", "é", "日本", "😀", r"\.", r"a\.b", r"\*", r"\+", r"\?",
    r"\t", r"\n", r"\x41", r"\x{1F600}", r"é", r"\U0001F600", r"\-", r"\ ",
    r"\%", r"\_", r"\\", "",
    # dot
    ".", "..", "a.c", "(?s).", r"(?s)a.+", ".+", ".*", "x.y",
    # classes
    "[abc]", "[a-c]+", "[^a-c]+", "[^abc]", "[]a]", "[a-]", "[-a]", "[a-z0-9]+",
    "[éèê]", "[α-ω]+", "[[:alpha:]]+", "[[:digit:]]+", "[[:^alpha:]]+",
    "[[:space:]]", "[[:punct:]]+", "[[:upper:]]", "[[:foo:]]", "[:alpha:]",
    "[a-z&&[^aeiou]]+", "[a-z--[b-y]]+", "[a-c~~b-d]+", "[a[0-9]]+", r"[\d]",
    r"[\w.]+", r"[^\s]+", r"[\n]", r"[\x41-\x43]+", "[😀a]",
    # Perl classes
    r"\d", r"\d+", r"\D+", r"\w+", r"\W+", r"\s+", r"\S+", r"\w", r"\s",
    # anchors
    "^", "$", "^a", "a$", "^abc$", "^$", r"\A", r"\z", r"\Aab", r"b\z",
    "(?m)^", "(?m)$", "(?m)^line", "(?m)\\d$", "(?m)^$", r"\b", r"\B", r"\bfoo\b",
    r"\b\w+\b", r"\B\w", r"\bé", r"ab\b", "^*", "$+",
    # anchored with groups
    "^(a+)(b)?", r"^(\w+)\s", r"\A(?:x|(a))+", "^(|a)", "^([^/]+)/?.*$",
    r"^https?://(?:www\.)?([^/]+)/.*$", "^(a)|b", r"(?m)^(\w+)",
    # alternation and groups
    "a|b", "foo|bar", "a|ab|abc", "abc|ab|a", "(a|b)+", "(?:ab)+", "(ab)+",
    "(a)(b)?(c)?", "(a|ab)(c|bcd)", "()", "a|", "|", "(|a)", "(a)|b",
    "(?P<word>\\w+)", "(?<x>a)(?<y>b)",
    # repetition
    "a*", "a+", "a?", "a{2}", "a{2,}", "a{1,2}", "a{0,1}", "a{ 2 }", "a*?",
    "a+?", "a??", "a{1,2}?", "(a*)*", "(a*)+", "(a|b)*c", "x*", "x+", "a**",
    "x{2}{3}", r"\d{2,3}", r"\d{4}-\d{2}-\d{2}", "(?U)a+", "(?U)a+?",
    "a{0}", "(ab){2}", ".*?b", ".*b",
    # case-insensitive and flags
    "(?i)a", "(?i)foo", "(?i)straße", "(?i)σ", "(?i)k", "(?i)[a-c]+",
    "(?i)[^a]", "(?i)é", "a(?i)b", "(?i:F)oo", "(?i)foo|BAR", "(?x) a b # c",
    "(?x)a{ 2 }", "(?i)ǆ", "(?-i)a", "(?i)(?-i:a)b",
    # realistic
    r"[\w.]+@[\w.]+\.\w+", r"\d+-\d+", "special.*requests", r"^\s+|\s+$",
    r"(\w+)@(\w+)\.com", r"(\d{4})-(\d{2})-(\d{2})", r"[A-Z][a-z]+",
    r"\s{2,}", r"(?i)hello|world", r"^[a-z]+$", r"[^\x00-\x7F]+",
    # errors
    "(?=a)", "(?!a)", "(?<=a)b", "(?<!a)b", r"(a)\1", r"\0", "(?P=n)", "(",
    ")", "a)", "[a", "[]", "[^]", "[z-a]", "a{2,1}", "a{", "{", "*a", "+",
    "?", "a{,3}", r"\e", r"\Z", "(?i-)a", "(?)", "(?<n>a)(?<n>b)",
    "(?P<1a>x)", r"\x{110000}", r"\é", r"[\b]",
]


def escape(text: str) -> str:
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append("\\x%02X" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def scalar(value) -> str:
    if value is None:
        return "\\N"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return escape(value)


def listing(values) -> str:
    if values is None:
        return "\\N"
    return "\x1f".join([str(len(values))] + [escape(v) for v in values])


def main() -> None:
    rows = []
    haystack = pl.Series("s", HAYSTACKS)
    for pattern in PATTERNS:
        try:
            frame = pl.DataFrame({"s": haystack}).select(
                pl.col("s").str.contains(pattern).alias("contains"),
                pl.col("s").str.find(pattern).alias("find"),
                pl.col("s").str.count_matches(pattern).alias("count"),
                pl.col("s").str.extract_all(pattern).alias("all"),
                pl.col("s").str.replace_all(pattern, "<$0>").alias("every"),
                pl.col("s").str.replace(pattern, "[${1}]").alias("first"),
                pl.col("s").str.extract(pattern, 1).alias("group"),
                pl.col("s").str.split(pattern, literal=False).alias("split"),
            )
        except Exception:
            rows.append("\t".join([escape(pattern), "", "ERROR"]))
            continue
        for i, text in enumerate(HAYSTACKS):
            row = frame.row(i, named=True)
            rows.append(
                "\t".join(
                    [
                        escape(pattern),
                        escape(text),
                        scalar(row["contains"]),
                        scalar(row["find"]),
                        scalar(row["count"]),
                        listing(row["all"]),
                        scalar(row["every"]),
                        scalar(row["first"]),
                        scalar(row["group"]),
                        listing(row["split"]),
                    ]
                )
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(rows) + "\n")
    print(len(PATTERNS), "patterns,", len(rows), "rows")


if __name__ == "__main__":
    main()
