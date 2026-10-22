"""Private projected file payloads without retained decoder backing buffers."""
from dataframe.frame import DataFrame
from dataframe.series import Series


def compact_source(frame: DataFrame) raises -> DataFrame:
    """Preserve rows/schema while owning only each selected column's window."""
    var columns = List[Series](capacity=frame.width())
    for column in frame._columns:
        columns.append(column._compact_input())
    return DataFrame(columns^, height=frame.height())
