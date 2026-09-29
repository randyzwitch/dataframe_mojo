// dfparquet: a minimal C ABI over Arrow C++'s Parquet reader.
//
// dfq_read_parquet_stream decodes one selected row group per stream batch.
// The legacy dfq_read_parquet still exports one combined batch for existing
// ABI consumers, and dfq_row_group_statistics
// exports the footer's per-row-group statistics as a record batch, so a
// caller can decide which row groups a filter needs before decoding any.
// Both export through the Arrow C Data Interface. Only the dfq_* symbols
// are visible; Arrow and its bundled codecs are linked in statically.
//
// Column types the caller cannot hold are coerced on the way out:
// dictionary columns are decoded to their value type, float16 widens to
// float32, and string/binary views become plain string/binary. Timestamps
// keep their time zone.
#include <arrow/array.h>
#include <arrow/array/builder_base.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/c/bridge.h>
#include <arrow/compute/cast.h>
#include <arrow/datum.h>
#include <arrow/io/file.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/scalar.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/config.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/arrow/schema.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <parquet/properties.h>
#include <parquet/statistics.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

int fail(const arrow::Status& status, char** error_out) {
  std::string text = status.ToString();
  *error_out = static_cast<char*>(std::malloc(text.size() + 1));
  std::memcpy(*error_out, text.c_str(), text.size() + 1);
  return 1;
}

std::shared_ptr<arrow::DataType> CoercedType(
    const std::shared_ptr<arrow::DataType>& type) {
  switch (type->id()) {
    case arrow::Type::DICTIONARY:
      return CoercedType(
          static_cast<const arrow::DictionaryType&>(*type).value_type());
    case arrow::Type::HALF_FLOAT:
      return arrow::float32();
    case arrow::Type::STRING_VIEW:
      return arrow::utf8();
    case arrow::Type::BINARY_VIEW:
      return arrow::binary();
    default:
      return type;
  }
}

arrow::Result<std::shared_ptr<arrow::Array>> Coerce(
    const std::shared_ptr<arrow::Array>& array) {
  auto target = CoercedType(array->type());
  if (target->Equals(*array->type())) return array;
  ARROW_ASSIGN_OR_RAISE(auto datum,
                        arrow::compute::Cast(arrow::Datum(array), target));
  return datum.make_array();
}

arrow::Result<std::shared_ptr<arrow::Scalar>> CoerceScalar(
    const std::shared_ptr<arrow::Scalar>& scalar,
    const std::shared_ptr<arrow::DataType>& target) {
  if (target->Equals(*scalar->type)) return scalar;
  ARROW_ASSIGN_OR_RAISE(auto datum,
                        arrow::compute::Cast(arrow::Datum(scalar), target));
  return datum.scalar();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> CoerceBatch(
    const std::shared_ptr<arrow::RecordBatch>& batch) {
  std::vector<std::shared_ptr<arrow::Array>> columns;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (int i = 0; i < batch->num_columns(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto column, Coerce(batch->column(i)));
    columns.push_back(column);
    const auto& field = batch->schema()->field(i);
    fields.push_back(arrow::field(field->name(), column->type(), field->nullable()));
  }
  return arrow::RecordBatch::Make(arrow::schema(fields), batch->num_rows(), columns);
}

arrow::Status Open(const char* path, bool use_threads,
                   std::unique_ptr<parquet::arrow::FileReader>* reader) {
  ARROW_ASSIGN_OR_RAISE(auto file, arrow::io::ReadableFile::Open(path));
  parquet::ArrowReaderProperties properties;
  properties.set_use_threads(use_threads);
  properties.set_pre_buffer(true);
  parquet::arrow::FileReaderBuilder builder;
  ARROW_RETURN_NOT_OK(builder.Open(file));
  builder.properties(properties);
  return builder.Build(reader);
}

void CollectLeaves(const parquet::arrow::SchemaField& field, std::vector<int>* out) {
  if (field.is_leaf()) {
    out->push_back(field.column_index);
    return;
  }
  for (const auto& child : field.children) CollectLeaves(child, out);
}

// The Parquet leaf columns behind each named Arrow field. A nested field
// (list, struct) spans several leaves; ReadRowGroups wants all of them.
arrow::Result<std::vector<int>> LeafIndices(parquet::arrow::FileReader& reader,
                                            const char** columns, int n) {
  std::shared_ptr<arrow::Schema> schema;
  ARROW_RETURN_NOT_OK(reader.GetSchema(&schema));
  const auto& manifest = reader.manifest();
  std::vector<int> leaves;
  for (int i = 0; i < n; ++i) {
    int index = schema->GetFieldIndex(columns[i]);
    if (index < 0) return arrow::Status::KeyError("no column named ", columns[i]);
    CollectLeaves(manifest.schema_fields[index], &leaves);
  }
  return leaves;
}

// Own the file reader for the stream lifetime. Decode only the next selected
// row group; never create a table or prebuffer spanning the whole selection.
class RowGroupReader final : public arrow::RecordBatchReader {
 public:
  RowGroupReader(std::unique_ptr<parquet::arrow::FileReader> reader,
                 std::vector<int> groups, std::vector<int> leaves,
                 bool projected, std::shared_ptr<arrow::Schema> schema)
      : reader_(std::move(reader)), groups_(std::move(groups)),
        leaves_(std::move(leaves)), projected_(projected),
        schema_(std::move(schema)) {}

  std::shared_ptr<arrow::Schema> schema() const override { return schema_; }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out) override {
    *out = nullptr;
    // An empty batch preserves the schema for an empty selection/file.
    if (groups_.empty() && next_ == 0) {
      ++next_;
      ARROW_ASSIGN_OR_RAISE(*out, arrow::RecordBatch::MakeEmpty(schema_));
      return arrow::Status::OK();
    }
    if (next_ >= groups_.size()) return arrow::Status::OK();
    std::vector<int> group{groups_[next_]};
    ARROW_ASSIGN_OR_RAISE(auto table, projected_
        ? reader_->ReadRowGroups(group, leaves_) : reader_->ReadRowGroups(group));
    ARROW_ASSIGN_OR_RAISE(auto batch, table->CombineChunksToBatch());
    ARROW_ASSIGN_OR_RAISE(*out, CoerceBatch(batch));
    ++next_;
    return arrow::Status::OK();
  }

 private:
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::vector<int> groups_;
  std::vector<int> leaves_;
  bool projected_;
  std::shared_ptr<arrow::Schema> schema_;
  size_t next_ = 0;
};

bool IsBinaryLike(const arrow::DataType& type) {
  return arrow::is_base_binary_like(type.id()) || arrow::is_binary_view_like(type.id()) ||
         type.id() == arrow::Type::FIXED_SIZE_BINARY;
}

}  // namespace

extern "C" {

// Returns 0 and fills out_array/out_schema (caller releases both through
// their release callbacks). Otherwise returns 1 and *error_out holds a
// message to free with dfq_free.
//
// n_columns == 0 reads every column. n_row_groups < 0 reads every row
// group; n_row_groups == 0 reads none and returns an empty batch that still
// carries the schema.
int dfq_read_parquet(const char* path, int use_threads, const char** columns,
                     int n_columns, const int* row_groups, int n_row_groups,
                     struct ArrowArray* out_array, struct ArrowSchema* out_schema,
                     char** error_out) {
  *error_out = nullptr;
  std::unique_ptr<parquet::arrow::FileReader> reader;
  arrow::Status status = Open(path, use_threads != 0, &reader);
  if (!status.ok()) return fail(status, error_out);

  std::vector<int> leaves;
  if (n_columns > 0) {
    auto result = LeafIndices(*reader, columns, n_columns);
    if (!result.ok()) return fail(result.status(), error_out);
    leaves = *result;
  }

  std::vector<int> groups;
  if (n_row_groups < 0) {
    for (int g = 0; g < reader->num_row_groups(); ++g) groups.push_back(g);
  } else {
    for (int i = 0; i < n_row_groups; ++i) {
      if (row_groups[i] < 0 || row_groups[i] >= reader->num_row_groups()) {
        return fail(arrow::Status::IndexError("row group ", row_groups[i],
                                              " is out of range"),
                    error_out);
      }
      groups.push_back(row_groups[i]);
    }
  }

  auto table = n_columns > 0 ? reader->ReadRowGroups(groups, leaves)
                             : reader->ReadRowGroups(groups);
  if (!table.ok()) return fail(table.status(), error_out);
  auto batch = (*table)->CombineChunksToBatch();
  if (!batch.ok()) return fail(batch.status(), error_out);
  auto coerced = CoerceBatch(*batch);
  if (!coerced.ok()) return fail(coerced.status(), error_out);
  status = arrow::ExportRecordBatch(**coerced, out_array, out_schema);
  if (!status.ok()) return fail(status, error_out);
  return 0;
}

// Export an Arrow C stream, one batch per selected row group. The caller
// releases every array/schema and the stream, including on early termination.
int dfq_read_parquet_stream(const char* path, int use_threads,
                            const char** columns, int n_columns,
                            const int* row_groups, int n_row_groups,
                            struct ArrowArrayStream* out, char** error_out) {
  *error_out = nullptr;
  out->release = nullptr;
  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto status = Open(path, use_threads != 0, &reader);
  if (!status.ok()) return fail(status, error_out);
  std::shared_ptr<arrow::Schema> source;
  status = reader->GetSchema(&source);
  if (!status.ok()) return fail(status, error_out);
  std::vector<int> leaves;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  if (n_columns > 0) {
    auto indices = LeafIndices(*reader, columns, n_columns);
    if (!indices.ok()) return fail(indices.status(), error_out);
    leaves = *indices;
    for (int i = 0; i < n_columns; ++i) {
      auto field = source->GetFieldByName(columns[i]);
      fields.push_back(arrow::field(field->name(), CoercedType(field->type()),
                                    field->nullable()));
    }
  } else {
    for (const auto& field : source->fields()) {
      fields.push_back(arrow::field(field->name(), CoercedType(field->type()),
                                    field->nullable()));
    }
  }
  std::vector<int> groups;
  if (n_row_groups < 0) {
    for (int g = 0; g < reader->num_row_groups(); ++g) groups.push_back(g);
  } else {
    for (int i = 0; i < n_row_groups; ++i) {
      if (row_groups[i] < 0 || row_groups[i] >= reader->num_row_groups()) {
        return fail(arrow::Status::IndexError("row group ", row_groups[i],
                                             " is out of range"), error_out);
      }
      groups.push_back(row_groups[i]);
    }
  }
  auto stream = std::make_shared<RowGroupReader>(std::move(reader),
      std::move(groups), std::move(leaves), n_columns > 0, arrow::schema(fields));
  status = arrow::ExportRecordBatchReader(std::move(stream), out);
  return status.ok() ? 0 : fail(status, error_out);
}

// Import owns the exports on success; the caller releases any structs left
// unconsumed on failure. store_schema preserves duration and Arrow type widths.
int dfq_write_parquet(const char* path, struct ArrowArray* array,
                      struct ArrowSchema* schema, const char* compression,
                      int64_t row_group_size, char** error_out) {
  *error_out = nullptr;
  parquet::Compression::type codec;
  std::string name(compression);
  if (name == "uncompressed") codec = parquet::Compression::UNCOMPRESSED;
  else if (name == "snappy") codec = parquet::Compression::SNAPPY;
  else if (name == "zstd") codec = parquet::Compression::ZSTD;
  else return fail(arrow::Status::Invalid("unsupported compression: ", name), error_out);
  if (row_group_size <= 0) {
    return fail(arrow::Status::Invalid("row_group_size must be positive"), error_out);
  }
  auto imported = arrow::ImportRecordBatch(array, schema);
  if (!imported.ok()) return fail(imported.status(), error_out);
  auto table = arrow::Table::FromRecordBatches({*imported});
  if (!table.ok()) return fail(table.status(), error_out);
  auto output = arrow::io::FileOutputStream::Open(path);
  if (!output.ok()) return fail(output.status(), error_out);
  auto properties = parquet::WriterProperties::Builder().compression(codec)->build();
  auto arrow_properties = parquet::ArrowWriterProperties::Builder().store_schema()->build();
  auto status = parquet::arrow::WriteTable(**table, arrow::default_memory_pool(),
      *output, row_group_size, properties, arrow_properties);
  auto closed = (*output)->Close();
  if (!status.ok()) return fail(status, error_out);
  return closed.ok() ? 0 : fail(closed, error_out);
}

// One row per row group: `row_group` (int32), `rows` (int64), then for each
// column of a flat schema `min:<name>` and `max:<name>` in the column's
// (coerced) type and `nulls:<name>` (int64). A bound is null when the
// footer has none, or when a string/binary bound is not marked exact, so a
// null bound always means "unknown", never a value.
int dfq_row_group_statistics(const char* path, struct ArrowArray* out_array,
                             struct ArrowSchema* out_schema, char** error_out) {
  *error_out = nullptr;
  std::unique_ptr<parquet::arrow::FileReader> reader;
  arrow::Status status = Open(path, false, &reader);
  if (!status.ok()) return fail(status, error_out);
  std::shared_ptr<arrow::Schema> schema;
  status = reader->GetSchema(&schema);
  if (!status.ok()) return fail(status, error_out);
  auto metadata = reader->parquet_reader()->metadata();
  const int n_groups = metadata->num_row_groups();
  const bool flat = metadata->num_columns() == schema->num_fields();
  const int n_fields = flat ? schema->num_fields() : 0;

  arrow::Int32Builder group_builder;
  arrow::Int64Builder rows_builder;
  std::vector<std::shared_ptr<arrow::DataType>> types;
  std::vector<std::unique_ptr<arrow::ArrayBuilder>> min_builders, max_builders;
  std::vector<std::unique_ptr<arrow::Int64Builder>> null_builders;
  for (int j = 0; j < n_fields; ++j) {
    auto type = CoercedType(schema->field(j)->type());
    types.push_back(type);
    std::unique_ptr<arrow::ArrayBuilder> min_builder, max_builder;
    status = arrow::MakeBuilder(arrow::default_memory_pool(), type, &min_builder);
    if (!status.ok()) return fail(status, error_out);
    status = arrow::MakeBuilder(arrow::default_memory_pool(), type, &max_builder);
    if (!status.ok()) return fail(status, error_out);
    min_builders.push_back(std::move(min_builder));
    max_builders.push_back(std::move(max_builder));
    null_builders.push_back(std::make_unique<arrow::Int64Builder>());
  }

  for (int g = 0; g < n_groups; ++g) {
    auto row_group = metadata->RowGroup(g);
    status = group_builder.Append(g);
    if (!status.ok()) return fail(status, error_out);
    status = rows_builder.Append(row_group->num_rows());
    if (!status.ok()) return fail(status, error_out);
    for (int j = 0; j < n_fields; ++j) {
      auto chunk = row_group->ColumnChunk(j);
      std::shared_ptr<parquet::Statistics> stats =
          chunk->is_stats_set() ? chunk->statistics() : nullptr;
      bool have_bounds = false;
      std::shared_ptr<arrow::Scalar> min_scalar, max_scalar;
      if (stats && stats->HasMinMax()) {
        bool exact = true;
        if (IsBinaryLike(*types[j])) {
          auto encoded = stats->Encode();
          exact = encoded.is_min_value_exact.value_or(false) &&
                  encoded.is_max_value_exact.value_or(false);
        }
        if (exact) {
          auto converted =
              parquet::arrow::StatisticsAsScalars(*stats, &min_scalar, &max_scalar);
          if (converted.ok()) {
            auto low = CoerceScalar(min_scalar, types[j]);
            auto high = CoerceScalar(max_scalar, types[j]);
            if (low.ok() && high.ok()) {
              min_scalar = *low;
              max_scalar = *high;
              have_bounds = true;
            }
          }
        }
      }
      if (have_bounds) {
        status = min_builders[j]->AppendScalar(*min_scalar, 1);
        if (!status.ok()) return fail(status, error_out);
        status = max_builders[j]->AppendScalar(*max_scalar, 1);
        if (!status.ok()) return fail(status, error_out);
      } else {
        status = min_builders[j]->AppendNull();
        if (!status.ok()) return fail(status, error_out);
        status = max_builders[j]->AppendNull();
        if (!status.ok()) return fail(status, error_out);
      }
      if (stats && stats->HasNullCount()) {
        status = null_builders[j]->Append(stats->null_count());
      } else {
        status = null_builders[j]->AppendNull();
      }
      if (!status.ok()) return fail(status, error_out);
    }
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::shared_ptr<arrow::Array> array;
  status = group_builder.Finish(&array);
  if (!status.ok()) return fail(status, error_out);
  fields.push_back(arrow::field("row_group", arrow::int32(), false));
  arrays.push_back(array);
  status = rows_builder.Finish(&array);
  if (!status.ok()) return fail(status, error_out);
  fields.push_back(arrow::field("rows", arrow::int64(), false));
  arrays.push_back(array);
  for (int j = 0; j < n_fields; ++j) {
    const std::string& name = schema->field(j)->name();
    status = min_builders[j]->Finish(&array);
    if (!status.ok()) return fail(status, error_out);
    fields.push_back(arrow::field("min:" + name, types[j]));
    arrays.push_back(array);
    status = max_builders[j]->Finish(&array);
    if (!status.ok()) return fail(status, error_out);
    fields.push_back(arrow::field("max:" + name, types[j]));
    arrays.push_back(array);
    status = null_builders[j]->Finish(&array);
    if (!status.ok()) return fail(status, error_out);
    fields.push_back(arrow::field("nulls:" + name, arrow::int64()));
    arrays.push_back(array);
  }
  auto batch = arrow::RecordBatch::Make(arrow::schema(fields), n_groups, arrays);
  status = arrow::ExportRecordBatch(*batch, out_array, out_schema);
  if (!status.ok()) return fail(status, error_out);
  return 0;
}

void dfq_free(void* pointer) { std::free(pointer); }

const char* dfq_arrow_version() { return ARROW_VERSION_STRING; }

}  // extern "C"
