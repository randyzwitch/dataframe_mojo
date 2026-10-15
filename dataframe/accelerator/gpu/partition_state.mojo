"""Shared CPU OVER layout-cache entry snapshots for one resident DAG."""
from std.collections import Dict


struct Context(Movable):
    var layouts: Dict[String, Bool]
    var entries: Dict[String, Bool]

    def __init__(out self):
        self.layouts = Dict[String, Bool]()
        self.entries = Dict[String, Bool]()

    def snapshot(mut self, scope: String, keys: String, state: String) raises:
        var cache = scope + ":" + keys
        self.entries[scope + ":" + state] = cache in self.layouts

    def dispatch(
        mut self, scope: String, keys: String, state: String,
        allow_contiguous: Bool, lanes: Int,
    ) raises -> Bool:
        var cache = scope + ":" + keys
        var entry = scope + ":" + state
        var cached = self.entries[entry] if entry in self.entries else cache in self.layouts
        if allow_contiguous and lanes > 1:
            self.layouts[cache] = True
            cached = True
        return cached
