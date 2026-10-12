"""Adapt logical reference data to production representations; no reference semantics here."""
from dataframe import Column, DataFrame, Series
from tests.support.reference_model import ReferenceTable, generated_model


def materialize_model(model: ReferenceTable) raises -> DataFrame:
    var columns = List[Series]()
    for name in model.names:
        var integers = List[Int64]()
        var floats = List[Float64]()
        var strings = List[String]()
        var booleans = List[Bool]()
        var valid = List[Bool]()
        for row in model.rows:
            if name == "i":
                integers.append(row.i)
                valid.append(row.iv)
            elif name == "f":
                floats.append(row.f)
                valid.append(row.fv)
            elif name == "s":
                strings.append(row.s.copy())
                valid.append(row.sv)
            elif name == "b":
                booleans.append(row.b)
                valid.append(row.bv)
            else:
                raise Error("Unknown model field")
        if name == "i":
            columns.append(Series(name, Column[Int64](integers^, valid^)))
        elif name == "f":
            columns.append(Series(name, Column[Float64](floats^, valid^)))
        elif name == "s":
            columns.append(Series(name, Column[String](strings^, valid^)))
        else:
            columns.append(Series(name, Column[Bool](booleans^, valid^)))
    return DataFrame(columns^, height=len(model.rows))


def frame_from_model(
    model: ReferenceTable, physical_offset: Int = 0
) raises -> DataFrame:
    """Change storage offsets without changing any logical input cell."""
    if physical_offset == 0:
        return materialize_model(model)
    var rows = generated_model(physical_offset).rows.copy()
    for row in model.rows:
        rows.append(row.copy())
    var padded = ReferenceTable(model.names.copy(), rows^)
    return materialize_model(padded).slice(physical_offset, len(model.rows))


def generated_frame(
    n: Int, offset: Int = 0, null_mode: Int = 0
) raises -> DataFrame:
    return frame_from_model(generated_model(n, offset, null_mode), offset)


def chunk_layout(frame: DataFrame) raises -> DataFrame:
    """Different boundaries per column, preserving every typed cell."""
    if frame.height() == 0:
        return frame.copy()
    var columns = List[Series]()
    var index = 0
    for name in frame.columns():
        var series = frame.column(name)
        var cut = min(index + 1, frame.height())
        columns.append(
            Series._from_chunks(
                [
                    series.slice(0, cut),
                    series.slice(cut, frame.height() - cut),
                ]
            )
        )
        index += 1
    return DataFrame(columns^, height=frame.height())


def offset_layout(frame: DataFrame) raises -> DataFrame:
    """A real unaligned window with exactly the original logical rows."""
    if frame.height() == 0:
        return frame.copy()
    return frame.head(1).vstack(frame).rechunk().slice(1, frame.height())
