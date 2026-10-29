"""Dataset schema alignment and metadata predicates, independent of readers."""
from std.collections import Optional
from dataframe.dtype import DataType
from dataframe.frame import DataFrame
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.storage.string import StringColumn
from dataframe.expr import (
    Expr,
    AND,
    OR,
    NOT,
    COL,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    IS_NULL,
    IS_NOT_NULL,
    LIT_INT,
    LIT_FLOAT,
    LIT_STRING,
    LIT_BOOL,
    LIT_NULL,
    subtree,
)
from .dataset_options import DatasetSpec, hive_int64


def _compatible_dtype(left: DataType, right: DataType) -> Bool:
    # A batch owns its categorical dictionary; concat reconciles its codes.
    return left == right or (left.is_categorical() and right.is_categorical())


def _hive_column(
    spec: DatasetSpec, file_index: Int, key: Int, rows: Int
) raises -> Series:
    var name = spec.hive_names[key]
    var dtype = spec.hive_dtypes[key]
    if spec.hive_nulls[file_index][key]:
        return Series.full_null(name, dtype, rows)
    var value = spec.hive_values[file_index][key]
    if dtype == DataType.INT64:
        return Series(
            name,
            Column[Int64](List[Int64](length=rows, fill=hive_int64(value))),
        )
    var column = Series(
        name, StringColumn(List[String](length=rows, fill=value))
    )
    return column^ if dtype == DataType.STRING else column.cast(dtype)


def _with_metadata(
    spec: DatasetSpec, file_index: Int, frame: DataFrame
) raises -> DataFrame:
    var result = frame.copy()
    for k in range(len(spec.hive_names)):
        result = result.with_column(
            _hive_column(spec, file_index, k, frame.height())
        )
    var path_name = spec.options.include_file_paths
    if path_name != "":
        if path_name in result.columns():
            raise Error(
                "include_file_paths column already exists: " + path_name
            )
        result = result.with_column(
            Series(
                path_name,
                StringColumn(
                    List[String](
                        length=frame.height(), fill=spec.paths[file_index]
                    )
                ),
            )
        )
    return result^


def dataset_schema(
    spec: DatasetSpec,
    schemas: List[DataFrame],
    required_columns: Optional[List[String]] = None,
    *,
    validate_types: Bool = True,
) raises -> DataFrame:
    """Reconcile supplied physical schemas, checking only required dtypes.

    Name policies always inspect the full schemas. None requests every output
    field; an explicitly empty required list requests a zero-column schema.
    Hive values and dtypes replace retained physical fields in place.
    Metadata discovery can disable dtype checking and keep the first-present
    prototype; decoded kept batches still validate through alignment.
    """
    if len(schemas) != len(spec.paths) or len(schemas) == 0:
        raise Error("Dataset schema requires one schema per resolved file")
    var logical = List[DataFrame]()
    for f in range(len(schemas)):
        logical.append(_with_metadata(spec, f, schemas[f].clear()))
    var names = logical[0].columns()
    if spec.options.union_by_name:
        for frame in logical:
            for name in frame.columns():
                if name not in names:
                    names.append(name)
    else:
        for f in range(1, len(logical)):
            var incoming = logical[f].columns()
            for name in names:
                if (
                    name not in incoming
                    and spec.options.missing_columns == "raise"
                ):
                    raise Error(
                        "Dataset missing column '"
                        + name
                        + "' in "
                        + spec.paths[f]
                    )
            for name in incoming:
                if name not in names and spec.options.extra_columns == "raise":
                    raise Error(
                        "Dataset extra column '"
                        + name
                        + "' in "
                        + spec.paths[f]
                    )
            if (
                spec.options.missing_columns == "raise"
                and spec.options.extra_columns == "raise"
            ):
                var physical_order = List[String]()
                var expected_order = List[String]()
                for name in incoming:
                    if (
                        spec.hive_index(name) < 0
                        and name != spec.options.include_file_paths
                    ):
                        physical_order.append(name)
                for name in names:
                    if (
                        spec.hive_index(name) < 0
                        and name != spec.options.include_file_paths
                    ):
                        expected_order.append(name)
                var same = len(physical_order) == len(expected_order)
                if same:
                    for c in range(len(physical_order)):
                        same &= physical_order[c] == expected_order[c]
                if not same:
                    raise Error(
                        "Dataset column order differs in " + spec.paths[f]
                    )
    var wanted = (
        required_columns.value().copy() if required_columns else names.copy()
    )
    var columns = List[Series]()
    var seen = List[String]()
    for name in wanted:
        if name not in names:
            raise Error("Unknown dataset column: " + name)
        if name in seen:
            raise Error("Dataset column requested twice: " + name)
        seen.append(name)
        var prototype = -1
        for f in range(len(logical)):
            if name not in logical[f].columns():
                continue
            if prototype < 0:
                prototype = f
            elif validate_types and not _compatible_dtype(
                logical[f].column(name).dtype(),
                logical[prototype].column(name).dtype(),
            ):
                raise Error(
                    "Dataset dtype mismatch for column '"
                    + name
                    + "' in "
                    + spec.paths[f]
                )
        columns.append(logical[prototype].column(name).copy())
    return DataFrame(columns^, height=0)


def dataset_physical_columns(
    spec: DatasetSpec,
    file_index: Int,
    schema: DataFrame,
    required_columns: List[String],
) raises -> List[String]:
    """Map logical projection to physical names, excluding path-derived fields."""
    if file_index < 0 or file_index >= len(spec.paths):
        raise Error("Dataset file index out of bounds")
    var names = List[String]()
    for name in required_columns:
        if (
            spec.hive_index(name) >= 0
            or name == spec.options.include_file_paths
        ):
            continue
        if name in schema.columns():
            names.append(name)
    return names^


def align_dataset_frame(
    spec: DatasetSpec,
    file_index: Int,
    frame: DataFrame,
    target_schema: DataFrame,
) raises -> DataFrame:
    """Align a decoded batch, retaining recursive field metadata for nulls."""
    var columns = List[Series]()
    var missing_rows = List[Int](length=frame.height(), fill=-1)
    for prototype in target_schema._columns:
        var name = prototype.name()
        var key = spec.hive_index(name)
        if key >= 0:
            columns.append(_hive_column(spec, file_index, key, frame.height()))
        elif name == spec.options.include_file_paths:
            columns.append(
                Series(
                    name,
                    StringColumn(
                        List[String](
                            length=frame.height(), fill=spec.paths[file_index]
                        )
                    ),
                )
            )
        elif name in frame.columns():
            var column = frame.column(name)
            if not _compatible_dtype(column.dtype(), prototype.dtype()):
                raise Error(
                    "Dataset dtype mismatch for column '"
                    + name
                    + "' in "
                    + spec.paths[file_index]
                )
            columns.append(column.copy())
        else:
            columns.append(prototype.take_or_null(missing_rows))
    return DataFrame(columns^, height=frame.height())


def _metadata_truth(
    spec: DatasetSpec, file_index: Int, predicate: Expr, root: Int
) raises -> Int:
    """0 false/null, 1 true, -1 unknown. Unknown can never reject a file."""
    ref node = predicate._nodes[root]
    if node.op == AND or node.op == OR:
        var left = _metadata_truth(spec, file_index, predicate, node.left)
        var right = _metadata_truth(spec, file_index, predicate, node.right)
        if node.op == AND:
            if left == 0 or right == 0:
                return 0
            return 1 if left == 1 and right == 1 else -1
        if left == 1 or right == 1:
            return 1
        return 0 if left == 0 and right == 0 else -1
    var expression = subtree(predicate, root)
    for item in expression._nodes:
        if item.op == COL:
            if (
                spec.hive_index(item.text) < 0
                and item.text != spec.options.include_file_paths
            ):
                return -1
        elif not (
            item.op == EQ
            or item.op == NE
            or item.op == LT
            or item.op == LE
            or item.op == GT
            or item.op == GE
            or item.op == IS_NULL
            or item.op == IS_NOT_NULL
            or item.op == NOT
            or item.op == LIT_INT
            or item.op == LIT_FLOAT
            or item.op == LIT_STRING
            or item.op == LIT_BOOL
            or item.op == LIT_NULL
        ):
            return -1
    try:
        var metadata = _with_metadata(
            spec, file_index, DataFrame(List[Series](), height=1)
        )
        return 1 if metadata.filter(expression).height() != 0 else 0
    except:
        # Unsupported type comparisons must be reported by the real filter.
        return -1


def dataset_may_match(
    spec: DatasetSpec, file_index: Int, predicate: Expr
) raises -> Bool:
    """Conservatively prune using only known directory/path constants."""
    if len(predicate._nodes) == 0:
        return True
    return (
        _metadata_truth(spec, file_index, predicate, len(predicate._nodes) - 1)
        != 0
    )
