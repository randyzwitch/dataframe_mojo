"""Original CPU partition-key semantics retained through GPU normalization.

The source dtype owns its original categorical dictionary. Lookup values are
indexed by the query-wide staged dictionary code, never by a source row.
"""
from dataframe.plan_types import WindowKey
