"""Full-result CPU/Metal comparison for staged external-suite translations.

The Python driver derives forced-CPU and forced-Metal workers from the same
checked-in query expressions. Table loading, output comparison, and canonical
sorting run on CPU. Every collect in the accelerated query explicitly uses
Metal; unsupported queries are retained rather than retried on CPU.
"""
from std.collections import Dict
from std.sys import argv
from std.time import perf_counter_ns
from dataframe import DataFrame
from suite_common import load_tables
from metal_suite_diagnostics import emit_comparison, environment_keys
from metal_suite_answer_exports import export_answer


def clean(message: String) -> String:
    return message.replace("\n", " ").replace("\t", " ")


def run[
    reference: def(String, Dict[String, DataFrame]) raises thin -> DataFrame,
    accelerated: def(String, Dict[String, DataFrame]) raises thin -> DataFrame,
]() raises:
    """Check every result cell and schema, retaining unsupported references."""
    var args = argv()
    var name = String(args[2])
    var repetitions = Int(String(args[3]))
    var tables = load_tables()
    var alignment_keys = environment_keys("DATAFRAME_BENCH_ALIGNMENT_KEYS")
    var order_keys = environment_keys("DATAFRAME_BENCH_ORDER_KEYS")
    var expected: DataFrame
    try:
        expected = reference(name, tables)
    except error:
        var reason = clean(String(error))
        print(
            "coverage\t"
            + name
            + "\treference_unsupported\t"
            + reason if reason.startswith("unsupported:") else "coverage\t"
            + name
            + "\treference_failed\t"
            + reason
        )
        return
    export_answer(expected, "cpu", 0)
    var ordered_equal = True
    # Suite outputs can leave group/tie order unspecified. Canonical sorting
    # compares complete row multisets, preserving duplicates; also record the
    # stricter original-order comparison instead of silently discarding it.
    var canonical = expected.sort(expected.columns())
    var elapsed = 0
    for repetition in range(repetitions):
        var actual: DataFrame
        try:
            var start = perf_counter_ns()
            actual = accelerated(name, tables)
            elapsed += Int(perf_counter_ns() - start)
        except error:
            var reason = clean(String(error))
            print(
                "coverage\t"
                + name
                + "\tunsupported\t"
                + reason if "unsupported"
                in reason
                or "do not support"
                in reason else "coverage\t"
                + name
                + "\texecution_failed\t"
                + reason
            )
            return
        export_answer(actual, "metal", repetition)
        ordered_equal &= actual.equals(expected)
        if (
            actual.columns() != expected.columns()
            or actual.height() != expected.height()
            or not actual.sort(actual.columns()).equals(canonical)
        ):
            print("coverage\t" + name + "\twrong_answer\tfull result differs")
            print("expected_rows\t" + String(expected.height()))
            print("actual_rows\t" + String(actual.height()))
            print(
                "expected_sample\n"
                + expected.to_string(max_rows=20, max_columns=64)
            )
            print(
                "actual_sample\n"
                + actual.to_string(max_rows=20, max_columns=64)
            )
            # Diagnostics cannot turn a strict mismatch into a passing case.
            try:
                emit_comparison(
                    expected, actual, "cpu_metal", alignment_keys, order_keys
                )
            except diagnostic_error:
                print(
                    "diagnostic_error\tcpu_metal\t"
                    + clean(String(diagnostic_error))
                )
            # Repeat the CPU query without excusing either kind of difference.
            try:
                var repeated = reference(name, tables)
                export_answer(repeated, "cpu_repeat", 0)
                print("cpu_repeat_equal\t" + String(repeated.equals(expected)))
                emit_comparison(
                    expected, repeated, "cpu_repeat", alignment_keys, order_keys
                )
                print(
                    "cpu_repeat_sample\n"
                    + repeated.to_string(max_rows=20, max_columns=64)
                )
            except diagnostic_error:
                print(
                    "diagnostic_error\tcpu_repeat\t"
                    + clean(String(diagnostic_error))
                )
            return
    print("diagnostic\tcpu_metal.schema_equal\tTrue")
    print("diagnostic\tcpu_metal.row_count_equal\tTrue")
    print("diagnostic\tcpu_metal.row_multiset_equal\tTrue")
    print(
        "diagnostic\tcpu_metal.classification\t"
        + ("equal" if ordered_equal else "order_only")
    )
    print("coverage\t" + name + "\tok\t")
    print("ordered_equal\t" + String(ordered_equal))
    print("output_rows\t" + String(expected.height()))
    print("collections\t" + String(repetitions))
    print("collection_ns\t" + String(elapsed))
