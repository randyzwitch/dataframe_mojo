"""Regular expression compilation and matching over Unicode code points.

Parse patterns into a Thompson NFA. Lazy forward and reverse DFAs find
match bounds; capture engines record group positions. Bounded caches fall
back to NFA simulation rather than allowing input-dependent exponential
backtracking. Literal prefilters reject impossible matches cheaply.
"""
from .regex import Matcher, Regex, Template, escape
