"""A linear-time regular expression engine in the family of RE2 and Rust's
`regex` crate: a parser for Rust's syntax, a Thompson NFA, a lazily built
DFA (forward for match ends, reverse for match starts), a Pike VM for
capture groups, and literal prefilters. There is no backtracking, so
look-around and backreferences are not supported.

    var regex = Regex("([a-z]+)-([0-9]+)")
    var matcher = Matcher(regex)
    if matcher.captures(text.as_bytes()):
        ...  # matcher.slots holds the group offsets

A `Regex` is immutable and shared by every worker; each worker searches
with its own `Matcher`.
"""
from .regex import Matcher, Regex, Template, escape
