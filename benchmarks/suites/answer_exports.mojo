"""Complete suite answers, serialized after stopping the query timer."""
from std.os import getenv

from dataframe import DataFrame, write_parquet
from dataframe.json_value import json_quote


def export_answer(
    frame: DataFrame,
    query: String,
    repetition: Int,
    record_kind: String = "answer",
) raises:
    var directory = getenv("DATAFRAME_SUITE_ANSWER_DIR")
    if not directory:
        return
    var stem = directory + "/" + query + "-" + String(repetition)
    var path = stem + ".parquet"
    var schema_path = stem + ".schema.json"
    # Preserve the original frame and schema: no sorting, casting or slicing.
    var schema = (
        '{"format":"suite-answer-v1","query":'
        + json_quote(query)
        + ',"repetition":'
        + String(repetition)
        + ',"rows":'
        + String(frame.height())
        + ',"columns":['
    )
    var names = frame.columns()
    for i in range(len(names)):
        if i:
            schema += ","
        var dtype = frame.column(names[i]).dtype()
        schema += (
            '{"name":'
            + json_quote(names[i])
            + ',"dtype":'
            + json_quote(dtype.name())
            + ',"encoded":'
            + json_quote(dtype._encode())
            + "}"
        )
    schema += "]}"
    write_parquet(frame, path)
    with open(schema_path, "w") as file:
        file.write(schema)
    print(
        record_kind
        + "\t"
        + query
        + "\t"
        + '{"path":'
        + json_quote(path)
        + ',"schema_path":'
        + json_quote(schema_path)
        + ',"repetition":'
        + String(repetition)
        + "}"
    )
