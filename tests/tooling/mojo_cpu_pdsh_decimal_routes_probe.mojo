"""Inspect actual read_parquet owners and PDS routes without GPU execution.

mojo run -I . -I benchmarks/suites tests/tooling/mojo_cpu_pdsh_decimal_routes_probe.mojo pdsh q1,q3,q5,q7,q8,q9,q10,q15 0 name=path ...
The table arguments are exactly the baseline worker's arguments.
"""
from std.sys import argv
from dataframe import StringColumn
from dataframe.binding import bind
from dataframe.expr import ADD, SUB, MUL, LIT_INT, CAST
from dataframe.lazy import AGG, WITH_COLUMNS
from dataframe.accelerator.constants import fold_literal_text_parsing
from dataframe.accelerator.metadata import schema_frame
from dataframe.accelerator.joins import _decimal_group_producer
from dataframe.metal import MetalRuntime
from pdsh import plan
from suite_common import load_tables


def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error("expected pdsh query-list 0 name=path ...")
    var tables = load_tables()
    for name in tables:
        ref frame = tables[name]
        print("TABLE", name, "rows", frame.height())
        for column in frame._columns:
            var dtype = column.dtype()
            var coded = False
            for chunk in column.chunks():
                if chunk._data.isa[StringColumn]():
                    coded |= Bool(chunk._data[StringColumn]._codes)
            if dtype.is_decimal() or dtype.is_date() or dtype.is_string():
                print(
                    "FIELD",
                    column.name(),
                    dtype.name(),
                    "physical",
                    dtype.physical().name(),
                    "decimal_width",
                    dtype.decimal_width() if dtype.is_decimal() else 0,
                    "StringCodes",
                    coded,
                )
    for part in String(args[2]).split(","):
        var name = String(part)
        print("QUERY", name, "default streaming True batch65536")
        var query = fold_literal_text_parsing(plan(name, tables))
        for index in range(len(query._nodes)):
            ref operation = query._nodes[index]
            if operation.kind != AGG and operation.kind != WITH_COLUMNS:
                continue
            print(
                "NODE", index, "kind", operation.kind, "keys", operation.names
            )
            var schema = schema_frame(query, operation.left)
            if operation.kind == AGG:
                print(
                    "NARROW_GROUP_PROOF", _decimal_group_producer(query, index)
                )
            for expression in operation.exprs:
                var bound = bind(expression, schema._columns)
                print("EXPR", expression._name)
                for at in range(len(bound.expr._nodes)):
                    ref node = bound.expr._nodes[at]
                    if node.op not in [ADD, SUB, MUL, LIT_INT, CAST]:
                        continue
                    var dtype = bound.dtypes[at]
                    if dtype.is_decimal():
                        print(
                            "DECIMAL_NODE",
                            at,
                            "op",
                            node.op,
                            "dtype",
                            dtype.name(),
                            "width",
                            dtype.decimal_width(),
                            "literal_low",
                            node.integer,
                            "literal_high",
                            node.text2,
                        )
        try:
            print(
                "DESCRIBE",
                MetalRuntime().describe(
                    query._with_accelerator_settings(True, True, 65536)
                ),
            )
        except error:
            print("FIRST_GUARD", String(error))
    print("PASS metadata inspection completed without GPU collection")
