"""Bounded multi-file readers shared by eager and lazy dataset entry points."""
from std.collections import Optional
from dataframe.frame import DataFrame, concat
from dataframe.series import Series
from dataframe.expr import Expr, COL
from .csv_types import CsvSchema, _CsvScanOptions
from .csv_reader import (
    _CsvBatches,
    _open_csv_batches,
    read_csv_explicit,
    read_csv_inferred,
)
from .parquet import (
    _ParquetBatches,
    _open_parquet_stream,
    read_parquet,
    parquet_row_group_statistics,
    _pruned_row_groups,
)
from .dataset_options import DatasetSpec
from .dataset_sources import (
    dataset_schema,
    dataset_physical_columns,
    align_dataset_frame,
    dataset_may_match,
)


def _csv_fallback(
    path: String,
    schema: Optional[CsvSchema],
    scan: _CsvScanOptions,
    columns: List[String],
    limit: Int,
) raises -> DataFrame:
    if schema:
        return read_csv_explicit(
            path,
            schema.value(),
            has_header=scan.has_header,
            separator=scan.separator,
            quote_char=scan.quote_char,
            comment_prefix=scan.comment_prefix,
            skip_rows=scan.skip_rows,
            n_rows=limit,
            columns=columns,
            null_values=scan.null_values,
            ignore_errors=scan.ignore_errors,
            truncate_ragged_lines=scan.truncate_ragged_lines,
            encoding=scan.encoding,
        )
    return read_csv_inferred(
        path,
        infer_schema_length=scan.infer_schema_length,
        schema_overrides=scan.schema_overrides,
        try_parse_dates=scan.try_parse_dates,
        has_header=scan.has_header,
        separator=scan.separator,
        quote_char=scan.quote_char,
        comment_prefix=scan.comment_prefix,
        skip_rows=scan.skip_rows,
        n_rows=limit,
        columns=columns,
        null_values=scan.null_values,
        ignore_errors=scan.ignore_errors,
        truncate_ragged_lines=scan.truncate_ragged_lines,
        encoding=scan.encoding,
    )


def _physical_schemas(
    spec: DatasetSpec,
    parquet: Bool,
    schema: Optional[CsvSchema],
    scan: _CsvScanOptions,
    batch_size: Int,
) raises -> List[DataFrame]:
    var schemas = List[DataFrame]()
    for path in spec.paths:
        if parquet:
            schemas.append(
                read_parquet(path, row_groups=List[Int](), glob=False)
            )
            continue
        var source = _open_csv_batches(
            path, schema, List[String](), 0, batch_size, scan
        )
        if not source:
            schemas.append(
                _csv_fallback(path, schema, scan, List[String](), 0).clear()
            )
            continue
        var batches = source.take()
        try:
            var next = batches.next()
            if not next:
                raise Error("CSV schema discovery produced no schema: " + path)
            var job = next.take()
            job.run()
            schemas.append(job^.into_frame().clear())
            batches.finish()
        except e:
            batches.finish()
            raise e^
    return schemas^


def _target_schema(
    spec: DatasetSpec,
    schemas: List[DataFrame],
    columns: Optional[List[String]],
) raises -> DataFrame:
    if not spec.active() and columns:
        # Legacy list readers compare only selected file schemas. Unselected
        # extra/missing fields do not suddenly become dataset policy errors.
        var projected = List[DataFrame]()
        for frame in schemas:
            projected.append(frame.select(columns.value()))
        return dataset_schema(spec, projected, columns, validate_types=False)
    return dataset_schema(spec, schemas, columns, validate_types=False)


struct _DatasetBatches(Movable):
    """One live file at a time, with schema discovery separate from payloads.

    predicate is pruning-only: callers retain the actual FILTER operation.
    limit counts source rows from retained files, before any caller filter.
    """

    var schema: DataFrame
    var spec: DatasetSpec
    var parquet: Bool
    var csv_schema: Optional[CsvSchema]
    var scan: _CsvScanOptions
    var physical_schemas: List[DataFrame]
    var decode_schema: DataFrame
    var predicate: Optional[Expr]
    var row_groups: Optional[List[Int]]
    var use_threads: Bool
    var batch_size: Int
    var limit: Int
    var emitted: Int
    var cursor: Int
    var current: Int
    var csv: List[_CsvBatches]
    var parquet_sources: List[_ParquetBatches]
    var fallback: Optional[DataFrame]
    var fallback_offset: Int
    var virtual_rows: Int
    var closed: Bool

    def __init__(
        out self,
        spec: DatasetSpec,
        parquet: Bool,
        *,
        schema: Optional[CsvSchema] = None,
        scan: _CsvScanOptions = _CsvScanOptions.defaults(),
        columns: Optional[List[String]] = None,
        limit: Int = -1,
        batch_size: Int = 65536,
        row_groups: Optional[List[Int]] = None,
        use_threads: Bool = True,
        predicate: Optional[Expr] = None,
    ) raises:
        if batch_size <= 0:
            raise Error("Dataset batch_size must be positive")
        if limit < -1:
            raise Error("Dataset limit must be nonnegative or -1")
        _ = scan.reader()
        if row_groups:
            for group in row_groups.value():
                if group < 0:
                    raise Error("Dataset row group indices must be nonnegative")
        self.spec = spec.copy()
        self.parquet = parquet
        self.csv_schema = schema.copy()
        self.scan = scan.copy()
        self.physical_schemas = _physical_schemas(
            spec, parquet, schema, scan, batch_size
        )
        self.schema = _target_schema(spec, self.physical_schemas, columns)
        self.decode_schema = self.schema.copy()
        self.predicate = predicate.copy()
        self.row_groups = row_groups.copy()
        self.use_threads = use_threads
        self.batch_size = batch_size
        self.limit = limit
        self.emitted = 0
        self.cursor = 0
        self.current = -1
        self.csv = List[_CsvBatches]()
        self.parquet_sources = List[_ParquetBatches]()
        self.fallback = None
        self.fallback_offset = 0
        self.virtual_rows = 0
        self.closed = False

    def _close_current(mut self) raises:
        if len(self.csv):
            self.csv[0].discard()
            self.csv[0].finish()
        self.csv.clear()
        self.parquet_sources.clear()
        self.fallback = None
        self.fallback_offset = 0
        self.virtual_rows = 0

    def finish(mut self) raises:
        self.closed = True
        self._close_current()

    def _groups(mut self, statistics: DataFrame) raises -> List[Int]:
        var groups = List[Int]()
        if self.row_groups:
            groups = self.row_groups.value().copy()
        else:
            for group in range(statistics.height()):
                groups.append(group)
        for group in groups:
            if group >= statistics.height():
                raise Error(
                    "Dataset row group index out of bounds: " + String(group)
                )
        if self.predicate:
            var physical = True
            for item in self.predicate.value()._nodes:
                if item.op == COL and (
                    self.spec.hive_index(item.text) >= 0
                    or item.text == self.spec.options.include_file_paths
                ):
                    physical = False
            if physical:
                var pruned = _pruned_row_groups(
                    statistics, self.predicate.value()
                )
                if pruned:
                    var selected = List[Int]()
                    for group in groups:
                        if group in pruned.value():
                            selected.append(group)
                    groups = selected^
        return groups^

    def _open_next(mut self) raises -> Bool:
        self._close_current()
        while self.cursor < len(self.spec.paths):
            self.current = self.cursor
            self.cursor += 1
            if self.predicate and not dataset_may_match(
                self.spec, self.current, self.predicate.value()
            ):
                continue
            # Validate required fields on surviving schemas before any payload
            # stream opens. Rejected partition dtypes cannot poison the query.
            _ = align_dataset_frame(
                self.spec,
                self.current,
                self.physical_schemas[self.current],
                self.decode_schema,
            )
            var columns = dataset_physical_columns(
                self.spec,
                self.current,
                self.physical_schemas[self.current],
                self.decode_schema.columns(),
            )
            var path = self.spec.paths[self.current]
            if self.parquet:
                var statistics = parquet_row_group_statistics(path)
                var groups = self._groups(statistics)
                if len(columns) == 0:
                    for group in groups:
                        self.virtual_rows += Int(
                            statistics.column("rows").get(group).int64()
                        )
                else:
                    var selected = List[Int32]()
                    for group in groups:
                        selected.append(Int32(group))
                    self.parquet_sources.append(
                        _ParquetBatches(
                            _open_parquet_stream(
                                path,
                                columns,
                                selected,
                                len(selected),
                                self.use_threads,
                            ),
                            self.batch_size,
                        )
                    )
            else:
                var remaining = -1 if self.limit < 0 else max(
                    0, self.limit - self.emitted
                )
                var source = _open_csv_batches(
                    path,
                    self.csv_schema,
                    columns,
                    remaining,
                    self.batch_size,
                    self.scan,
                )
                if source:
                    var batches = source.take()
                    if len(columns) == 0:
                        batches.project_none()
                    self.csv.append(batches^)
                else:
                    self.fallback = _csv_fallback(
                        path, self.csv_schema, self.scan, columns, remaining
                    )
            return True
        return False

    def _next_physical(mut self) raises -> Optional[DataFrame]:
        if self.virtual_rows > 0:
            var rows = min(self.virtual_rows, self.batch_size)
            self.virtual_rows -= rows
            return DataFrame(List[Series](), height=rows)
        if len(self.csv):
            var next = self.csv[0].next()
            if next:
                var job = next.take()
                job.run()
                return job^.into_frame()
        elif len(self.parquet_sources):
            return self.parquet_sources[0].next()
        elif (
            self.fallback
            and self.fallback_offset < self.fallback.value().height()
        ):
            var frame = self.fallback.value().slice(
                self.fallback_offset, self.batch_size
            )
            self.fallback_offset += frame.height()
            return frame^
        return None

    def next(mut self) raises -> Optional[DataFrame]:
        if self.closed:
            return None
        if self.limit >= 0 and self.emitted >= self.limit:
            self.finish()
            return None
        try:
            while True:
                if self.current < 0:
                    if not self._open_next():
                        self.finish()
                        return None
                var physical = self._next_physical()
                if not physical:
                    if not self._open_next():
                        self.finish()
                        return None
                    continue
                var frame = align_dataset_frame(
                    self.spec,
                    self.current,
                    physical.value(),
                    self.decode_schema,
                )
                if self.limit >= 0:
                    frame = frame.head(max(0, self.limit - self.emitted))
                if frame.height() == 0:
                    continue
                frame = frame.select(self.schema.columns())
                self.emitted += frame.height()
                return frame^
        except e:
            self.finish()
            raise e^


def read_dataset(
    spec: DatasetSpec,
    parquet: Bool,
    *,
    schema: Optional[CsvSchema] = None,
    scan: _CsvScanOptions = _CsvScanOptions.defaults(),
    columns: Optional[List[String]] = None,
    limit: Int = -1,
    batch_size: Int = 65536,
    row_groups: Optional[List[Int]] = None,
    use_threads: Bool = True,
    predicate: Optional[Expr] = None,
) raises -> DataFrame:
    """Collect aligned source batches; predicate only prunes files/groups."""
    var batches = _DatasetBatches(
        spec,
        parquet,
        schema=schema,
        scan=scan,
        columns=columns,
        limit=limit,
        batch_size=batch_size,
        row_groups=row_groups,
        use_threads=use_threads,
        predicate=predicate,
    )
    var frames = List[DataFrame]()
    try:
        while True:
            var next = batches.next()
            if not next:
                break
            frames.append(next.take())
        batches.finish()
    except e:
        batches.finish()
        raise e^
    return (
        batches.schema.copy() if len(frames)
        == 0 else concat(frames)._with_statistics()
    )


def dataset_schema_frame(
    spec: DatasetSpec,
    parquet: Bool,
    *,
    schema: Optional[CsvSchema] = None,
    scan: _CsvScanOptions = _CsvScanOptions.defaults(),
    columns: Optional[List[String]] = None,
    limit: Int = -1,
    batch_size: Int = 65536,
    row_groups: Optional[List[Int]] = None,
    use_threads: Bool = True,
    predicate: Optional[Expr] = None,
) raises -> DataFrame:
    var batches = _DatasetBatches(
        spec,
        parquet,
        schema=schema,
        scan=scan,
        columns=columns,
        limit=limit,
        batch_size=batch_size,
        row_groups=row_groups,
        use_threads=use_threads,
        predicate=predicate,
    )
    var schema_frame = batches.schema.copy()
    batches.finish()
    return schema_frame^
