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
    var ordered_equal = True
    # Suite outputs can leave group/tie order unspecified. Canonical sorting
    # compares complete row multisets, preserving duplicates; also record the
    # stricter original-order comparison instead of silently discarding it.
    var canonical = expected.sort(expected.columns())
    var elapsed = 0
    for _ in range(repetitions):
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
            # Repeating the original CPU query helps diagnose unspecified
            # limit/tie choices without relaxing the full-result comparison.
            var repeated = reference(name, tables)
            print("cpu_repeat_equal\t" + String(repeated.equals(expected)))
            print(
                "cpu_repeat_sample\n"
                + repeated.to_string(max_rows=20, max_columns=64)
            )
            return
    print("coverage\t" + name + "\tok\t")
    print("ordered_equal\t" + String(ordered_equal))
    print("output_rows\t" + String(expected.height()))
    print("collections\t" + String(repetitions))
    print("collection_ns\t" + String(elapsed))
