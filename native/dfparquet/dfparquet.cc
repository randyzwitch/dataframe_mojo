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
// keep their time zone. dfq_read_parquet_stream_dict is the stream except
// that a top-level string column with a dictionary page in every selected
// row group stays dictionary-encoded: each batch's codes index one running
// dictionary per column, which only grows, so the caller can keep the codes
// beside the strings.
#include <arrow/array.h>
#include <arrow/array/array_dict.h>
#include <arrow/array/builder_base.h>
#include <arrow/array/builder_binary.h>
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
#include <unordered_map>
#include <vector>

namespace {

int fail(const arrow::Status& status, char** error_out) {
  std::string text = status.ToString();
  *error_out = static_cast<char*>(std::malloc(text.size() + 1));
  std::memcpy(*error_out, text.c_str(), text.size() + 1);
  return 1;
}

bool StringDictionary(const arrow::DataType& type) {
  if (type.id() != arrow::Type::DICTIONARY) return false;
  auto value = static_cast<const arrow::DictionaryType&>(type).value_type()->id();
  return value == arrow::Type::STRING || value == arrow::Type::LARGE_STRING;
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
  // A decimal stored as INT32 or INT64 reads as decimal32 or decimal64,
  // the width its file declares, not widened to decimal128.
  properties.set_smallest_decimal_enabled(true);
  parquet::arrow::FileReaderBuilder builder;
  ARROW_RETURN_NOT_OK(builder.Open(file));
  builder.properties(properties);
  return builder.Build(reader);
}

// Open for dfq_read_parquet_dict: string leaves dictionary-encoded in every
// selected row group read as dictionary arrays.
arrow::Status OpenDictionary(const char* path, bool use_threads,
                             const std::vector<int>& groups,
                             const char** columns, int n_columns, int* chosen,
                             std::unique_ptr<parquet::arrow::FileReader>* reader) {
  *chosen = 0;
  ARROW_ASSIGN_OR_RAISE(auto file, arrow::io::ReadableFile::Open(path));
  parquet::arrow::FileReaderBuilder builder;
  ARROW_RETURN_NOT_OK(builder.Open(file));
  parquet::ArrowReaderProperties properties;
  properties.set_use_threads(use_threads);
  properties.set_pre_buffer(true);
  auto metadata = builder.raw_reader()->metadata();
  const auto* schema = metadata->schema();
  for (int c = 0; c < schema->num_columns(); ++c) {
    const auto* column = schema->Column(c);
    if (column->physical_type() != parquet::Type::BYTE_ARRAY) continue;
    if (column->max_repetition_level() != 0) continue;
    if (column->path()->ToDotVector().size() != 1) continue;
    const auto& logical = column->logical_type();
    bool text = (logical && logical->is_string()) ||
                column->converted_type() == parquet::ConvertedType::UTF8;
    if (!text || groups.empty()) continue;
    if (n_columns > 0) {
      bool selected = false;
      for (int i = 0; i < n_columns; ++i) {
        selected = selected || column->path()->ToDotVector()[0] == columns[i];
      }
      if (!selected) continue;
    }
    bool dictionary = true;
    for (int g : groups) {
      dictionary = dictionary &&
                   metadata->RowGroup(g)->ColumnChunk(c)->has_dictionary_page();
    }
    if (dictionary) {
      properties.set_read_dictionary(c, true);
      ++*chosen;
    }
  }
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
                 bool projected, std::shared_ptr<arrow::Schema> schema,
                 bool keep_dictionaries = false)
      : reader_(std::move(reader)), groups_(std::move(groups)),
        leaves_(std::move(leaves)), projected_(projected),
        schema_(std::move(schema)), keep_dictionaries_(keep_dictionaries) {}

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
    if (keep_dictionaries_) {
      ARROW_ASSIGN_OR_RAISE(*out, Recode(batch));
    } else {
      ARROW_ASSIGN_OR_RAISE(*out, CoerceBatch(batch));
    }
    ++next_;
    return arrow::Status::OK();
  }

 private:
  // Hash and compare std::string keys by std::string_view, so a lookup
  // allocates nothing.
  struct ViewHash {
    using is_transparent = void;
    size_t operator()(std::string_view text) const {
      return std::hash<std::string_view>{}(text);
    }
  };

  // One column's running dictionary: every value seen so far, in first-seen
  // order, so a code once given never changes. `built` is the dictionary
  // array as of `built_size` values, rebuilt only when it grows.
  struct Running {
    std::unordered_map<std::string, int32_t, ViewHash, std::equal_to<>> codes;
    std::vector<std::string> values;
    std::shared_ptr<arrow::Array> built;
    size_t built_size = 0;
  };

  template <typename Values>
  static void Remember(const Values& values, Running& running,
                       std::vector<int32_t>& remap) {
    for (int64_t k = 0; k < values.length(); ++k) {
      std::string_view text = values.GetView(k);
      auto found = running.codes.find(text);
      if (found == running.codes.end()) {
        int32_t code = static_cast<int32_t>(running.values.size());
        running.values.emplace_back(text);
        running.codes.emplace(running.values.back(), code);
        remap[k] = code;
      } else {
        remap[k] = found->second;
      }
    }
  }

  // Coerce as CoerceBatch does, except string dictionary columns: their
  // indices move onto the column's running dictionary, which the batch
  // carries whole.
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Recode(
      const std::shared_ptr<arrow::RecordBatch>& batch) {
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (int i = 0; i < batch->num_columns(); ++i) {
      auto column = batch->column(i);
      if (!StringDictionary(*column->type())) {
        ARROW_ASSIGN_OR_RAISE(column, Coerce(column));
        columns.push_back(column);
        continue;
      }
      auto& running = running_[i];
      const auto& dictionary = static_cast<const arrow::DictionaryArray&>(*column);
      auto values = dictionary.dictionary();
      std::vector<int32_t> remap(values->length());
      if (values->type_id() == arrow::Type::LARGE_STRING) {
        Remember(static_cast<const arrow::LargeStringArray&>(*values), running, remap);
      } else {
        Remember(static_cast<const arrow::StringArray&>(*values), running, remap);
      }
      // New index values over the old validity bitmap.
      auto widened = dictionary.indices();
      if (widened->type_id() != arrow::Type::INT32) {
        ARROW_ASSIGN_OR_RAISE(auto cast, arrow::compute::Cast(
            arrow::Datum(widened), arrow::int32()));
        widened = cast.make_array();
      }
      const auto& old_indices = static_cast<const arrow::Int32Array&>(*widened);
      int64_t n = old_indices.length();
      ARROW_ASSIGN_OR_RAISE(auto data, arrow::AllocateBuffer(n * sizeof(int32_t)));
      auto* out = reinterpret_cast<int32_t*>(data->mutable_data());
      const int32_t* in = old_indices.raw_values();
      bool nulls = old_indices.null_count() > 0;
      for (int64_t r = 0; r < n; ++r) {
        out[r] = (nulls && old_indices.IsNull(r)) ? 0 : remap[in[r]];
      }
      auto index_array = std::make_shared<arrow::Int32Array>(
          n, std::move(data), old_indices.null_bitmap(), old_indices.null_count(),
          old_indices.offset());
      if (!running.built || running.built_size != running.values.size()) {
        arrow::StringBuilder text;
        for (const auto& value : running.values) {
          ARROW_RETURN_NOT_OK(text.Append(value));
        }
        ARROW_RETURN_NOT_OK(text.Finish(&running.built));
        running.built_size = running.values.size();
      }
      ARROW_ASSIGN_OR_RAISE(
          auto recoded,
          arrow::DictionaryArray::FromArrays(
              arrow::dictionary(arrow::int32(), arrow::utf8()), index_array,
              running.built));
      columns.push_back(recoded);
    }
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int i = 0; i < batch->num_columns(); ++i) {
      const auto& field = batch->schema()->field(i);
      fields.push_back(arrow::field(field->name(), columns[i]->type(), field->nullable()));
    }
    return arrow::RecordBatch::Make(arrow::schema(fields), batch->num_rows(), columns);
  }

 public:

 private:
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::vector<int> groups_;
  std::vector<int> leaves_;
  bool projected_;
  std::shared_ptr<arrow::Schema> schema_;
  size_t next_ = 0;
  bool keep_dictionaries_;
  std::unordered_map<int, Running> running_;
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

// As dfq_read_parquet_stream, with string columns that have a dictionary
// page in every selected row group streamed as dictionary<int32, utf8>
// whose dictionary is the column's running one (see RowGroupReader). Returns
// 2, opening no stream, when no selected column is such a column.
int dfq_read_parquet_stream_dict(const char* path, int use_threads,
                                 const char** columns, int n_columns,
                                 const int* row_groups, int n_row_groups,
                                 struct ArrowArrayStream* out, char** error_out) {
  *error_out = nullptr;
  out->release = nullptr;
  std::vector<int> groups;
  {
    std::unique_ptr<parquet::ParquetFileReader> probe;
    try {
      probe = parquet::ParquetFileReader::OpenFile(path, false);
    } catch (const std::exception& e) {
      return fail(arrow::Status::IOError(e.what()), error_out);
    }
    int count = probe->metadata()->num_row_groups();
    if (n_row_groups < 0) {
      for (int g = 0; g < count; ++g) groups.push_back(g);
    } else {
      for (int i = 0; i < n_row_groups; ++i) {
        if (row_groups[i] < 0 || row_groups[i] >= count) {
          return fail(arrow::Status::IndexError("row group ", row_groups[i],
                                                " is out of range"),
                      error_out);
        }
        groups.push_back(row_groups[i]);
      }
    }
  }
  std::unique_ptr<parquet::arrow::FileReader> reader;
  int chosen = 0;
  auto status = OpenDictionary(path, use_threads != 0, groups, columns, n_columns,
                               &chosen, &reader);
  if (!status.ok()) return fail(status, error_out);
  if (chosen == 0) return 2;
  std::shared_ptr<arrow::Schema> source;
  status = reader->GetSchema(&source);
  if (!status.ok()) return fail(status, error_out);
  auto out_type = [](const std::shared_ptr<arrow::DataType>& type) {
    return StringDictionary(*type) ? arrow::dictionary(arrow::int32(), arrow::utf8())
                                   : CoercedType(type);
  };
  std::vector<int> leaves;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  if (n_columns > 0) {
    auto indices = LeafIndices(*reader, columns, n_columns);
    if (!indices.ok()) return fail(indices.status(), error_out);
    leaves = *indices;
    for (int i = 0; i < n_columns; ++i) {
      auto field = source->GetFieldByName(columns[i]);
      fields.push_back(arrow::field(field->name(), out_type(field->type()),
                                    field->nullable()));
    }
  } else {
    for (const auto& field : source->fields()) {
      fields.push_back(arrow::field(field->name(), out_type(field->type()),
                                    field->nullable()));
    }
  }
  auto stream = std::make_shared<RowGroupReader>(
      std::move(reader), std::move(groups), std::move(leaves), n_columns > 0,
      arrow::schema(fields), true);
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
