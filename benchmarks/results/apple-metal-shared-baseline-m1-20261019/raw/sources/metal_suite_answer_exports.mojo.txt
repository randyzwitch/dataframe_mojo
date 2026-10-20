"""Optional complete answer snapshots outside query collection timing.

The caller owns the output prefix and directory. Failed requested exports are
explicit diagnostics and cannot change a query comparison into a passing case.
"""
from std.os import getenv
from dataframe import DataFrame, DataType, write_parquet
from dataframe.json_value import json_quote


def dtype_receipt(dtype: DataType) -> String:
    var text = (
        '{"name":'
        + json_quote(dtype.name())
        + ',"encoded":'
        + json_quote(dtype._encode())
    )
    if dtype.is_categorical() and dtype.has_dictionary():
        text += ',"dictionary":['
        var dictionary = dtype.dictionary()
        for i in range(len(dictionary[])):
            if i:
                text += ","
            text += json_quote(String(dictionary[].get(i)))
        text += "]"
    return text + "}"


def schema_receipt(
    frame: DataFrame, engine: String, repetition: Int
) raises -> String:
    var text = (
        '{"format":"dataframe-answer-schema-v1","engine":'
        + json_quote(engine)
        + ',"repetition":'
        + String(repetition)
        + ',"rows":'
        + String(frame.height())
        + ',"columns":['
    )
    var names = frame.columns()
    for i in range(len(names)):
        if i:
            text += ","
        text += (
            '{"index":'
            + String(i)
            + ',"name":'
            + json_quote(names[i])
            + ',"dtype":'
            + dtype_receipt(frame.column(names[i]).dtype())
            + "}"
        )
    return text + "]}"


def export_answer(frame: DataFrame, engine: String, repetition: Int):
    var prefix = getenv("DATAFRAME_BENCH_ANSWER_EXPORT_PREFIX")
    if not prefix:
        return
    var stem = prefix + "-" + engine + "-" + String(repetition)
    var path = stem + ".parquet"
    var schema_path = stem + ".schema.json"
    var identity = (
        '{"engine":'
        + json_quote(engine)
        + ',"repetition":'
        + String(repetition)
        + ',"path":'
        + json_quote(path)
        + ',"schema_path":'
        + json_quote(schema_path)
    )
    try:
        # I/O receives the complete original frame. No sort, cast or slicing.
        write_parquet(frame, path)
        with open(schema_path, "w") as file:
            file.write(schema_receipt(frame, engine, repetition))
        print("answer_export\t" + identity + ',"status":"written"}')
    except error:
        print(
            "answer_export\t"
            + identity
            + ',"status":"unavailable","error":'
            + json_quote(String(error))
            + "}"
        )
