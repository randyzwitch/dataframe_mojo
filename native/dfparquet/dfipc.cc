// IPC file/stream I/O through the same Arrow C ABI as the Parquet backend.
#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/compute/cast.h>
#include <arrow/io/file.h>
#include <arrow/ipc/api.h>
#include <arrow/util/compression.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
int Fail(const arrow::Status& status, char** error) {
  const auto text = status.ToString();
  *error = static_cast<char*>(std::malloc(text.size() + 1));
  std::memcpy(*error, text.c_str(), text.size() + 1);
  return 1;
}

// Preserve logical metadata and decimal widths. Only unsupported physical
// string/binary views and float16 need normalization for the Mojo importer.
std::shared_ptr<arrow::DataType> ImportType(
    const std::shared_ptr<arrow::DataType>& type) {
  switch (type->id()) {
    case arrow::Type::STRING_VIEW: return arrow::large_utf8();
    case arrow::Type::BINARY_VIEW: return arrow::large_binary();
    case arrow::Type::HALF_FLOAT: return arrow::float32();
    case arrow::Type::LIST:
    case arrow::Type::LARGE_LIST: {
      auto field = type->field(0);
      auto value = field->WithType(ImportType(field->type()));
      return type->id() == arrow::Type::LIST ? arrow::list(value)
                                            : arrow::large_list(value);
    }
    case arrow::Type::STRUCT: {
      arrow::FieldVector fields;
      for (const auto& field : type->fields()) {
        fields.push_back(field->WithType(ImportType(field->type())));
      }
      return arrow::struct_(fields);
    }
    case arrow::Type::DICTIONARY: {
      const auto& dict = static_cast<const arrow::DictionaryType&>(*type);
      return arrow::dictionary(dict.index_type(), ImportType(dict.value_type()),
                               dict.ordered());
    }
    default: return type;
  }
}

std::shared_ptr<arrow::Schema> ImportSchema(
    const std::shared_ptr<arrow::Schema>& schema) {
  arrow::FieldVector fields;
  for (const auto& field : schema->fields()) {
    fields.push_back(field->WithType(ImportType(field->type())));
  }
  return arrow::schema(fields, schema->metadata());
}

arrow::Result<std::shared_ptr<arrow::Array>> ImportArray(
    const std::shared_ptr<arrow::Array>& array) {
  auto target = ImportType(array->type());
  if (target->Equals(array->type())) return array;
  if (array->type_id() == arrow::Type::DICTIONARY) {
    const auto& dict = static_cast<const arrow::DictionaryArray&>(*array);
    ARROW_ASSIGN_OR_RAISE(auto values, ImportArray(dict.dictionary()));
    return arrow::DictionaryArray::FromArrays(target, dict.indices(), values);
  }
  ARROW_ASSIGN_OR_RAISE(auto converted,
                        arrow::compute::Cast(arrow::Datum(array), target));
  return converted.make_array();
}

struct Source {
  std::shared_ptr<arrow::io::RandomAccessFile> file;
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> indexed;
  std::shared_ptr<arrow::RecordBatchReader> sequential;
  int next = 0;

  std::shared_ptr<arrow::Schema> schema() const {
    return indexed ? indexed->schema() : sequential->schema();
  }
  arrow::Status OpenReader(bool stream, const arrow::ipc::IpcReadOptions& options) {
    next = 0;
    indexed.reset();
    sequential.reset();
    ARROW_RETURN_NOT_OK(file->Seek(0));
    if (stream) {
      ARROW_ASSIGN_OR_RAISE(sequential,
                            arrow::ipc::RecordBatchStreamReader::Open(file, options));
    } else {
      ARROW_ASSIGN_OR_RAISE(indexed,
                            arrow::ipc::RecordBatchFileReader::Open(file, options));
    }
    return arrow::Status::OK();
  }
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next() {
    if (indexed) {
      if (next == indexed->num_record_batches()) return nullptr;
      return indexed->ReadRecordBatch(next++);
    }
    return sequential->Next();
  }
};

arrow::Result<std::shared_ptr<Source>> Open(const char* path, bool stream,
                                           bool mmap) {
  auto source = std::make_shared<Source>();
  if (mmap) {
    ARROW_ASSIGN_OR_RAISE(source->file,
                          arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ));
  } else {
    ARROW_ASSIGN_OR_RAISE(source->file, arrow::io::ReadableFile::Open(path));
  }
  ARROW_RETURN_NOT_OK(source->OpenReader(stream, arrow::ipc::IpcReadOptions::Defaults()));
  return source;
}

class Reader final : public arrow::RecordBatchReader {
 public:
  Reader(std::shared_ptr<Source> source, std::vector<int> fields,
         std::shared_ptr<arrow::Schema> schema, int64_t remaining)
      : source_(std::move(source)), fields_(std::move(fields)),
        schema_(std::move(schema)), remaining_(remaining) {}
  std::shared_ptr<arrow::Schema> schema() const override { return schema_; }
  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out) override {
    *out = nullptr;
    std::shared_ptr<arrow::RecordBatch> batch;
    if (remaining_ != 0) {
      ARROW_ASSIGN_OR_RAISE(batch, source_->Next());
    }
    if (!batch) {
      if (!emitted_) {
        ARROW_ASSIGN_OR_RAISE(*out, arrow::RecordBatch::MakeEmpty(schema_));
        emitted_ = true;
      }
      return arrow::Status::OK();
    }
    if (remaining_ >= 0 && batch->num_rows() > remaining_) {
      batch = batch->Slice(0, remaining_);
    }
    if (remaining_ >= 0) remaining_ -= batch->num_rows();
    arrow::ArrayVector columns;
    for (int field : fields_) {
      ARROW_ASSIGN_OR_RAISE(auto column, ImportArray(batch->column(field)));
      columns.push_back(std::move(column));
    }
    *out = arrow::RecordBatch::Make(schema_, batch->num_rows(), std::move(columns));
    emitted_ = true;
    return arrow::Status::OK();
  }
 private:
  std::shared_ptr<Source> source_;
  std::vector<int> fields_;
  std::shared_ptr<arrow::Schema> schema_;
  int64_t remaining_;
  bool emitted_ = false;
};

struct Writer {
  std::shared_ptr<arrow::io::FileOutputStream> output;
  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
};
}  // namespace

extern "C" {
// n_columns=-1 selects all, zero selects no fields while retaining row count.
// n_rows=-1 has no limit. The exported stream owns the native reader/file.
int dfq_read_ipc(const char* path, int stream, int mmap, const char** columns,
                 int n_columns, int64_t n_rows, ArrowArrayStream* out, char** error) {
  *error = nullptr;
  out->release = nullptr;
  if (n_columns < -1 || n_rows < -1) {
    return Fail(arrow::Status::Invalid("invalid IPC selection or row limit"), error);
  }
  auto opened = Open(path, stream != 0, mmap != 0);
  if (!opened.ok()) return Fail(opened.status(), error);
  auto source = *opened;
  auto original = source->schema();
  std::vector<int> requested;
  if (n_columns < 0) {
    for (int i = 0; i < original->num_fields(); ++i) requested.push_back(i);
  } else {
    for (int i = 0; i < n_columns; ++i) {
      auto matches = original->GetAllFieldIndices(columns[i]);
      if (matches.size() != 1) {
        return Fail(arrow::Status::Invalid("missing or ambiguous IPC column: ", columns[i]), error);
      }
      if (std::find(requested.begin(), requested.end(), matches[0]) != requested.end()) {
        return Fail(arrow::Status::Invalid("IPC column requested twice: ", columns[i]), error);
      }
      requested.push_back(matches[0]);
    }
  }
  // Arrow's included_fields returns schema order; explicitly restore caller order.
  if (!requested.empty() && n_columns >= 0) {
    auto options = arrow::ipc::IpcReadOptions::Defaults();
    options.included_fields = requested;
    std::sort(options.included_fields.begin(), options.included_fields.end());
    auto status = source->OpenReader(stream != 0, options);
    if (!status.ok()) return Fail(status, error);
  }
  arrow::FieldVector fields;
  std::vector<int> positions;
  for (int field : requested) {
    if (original->GetFieldIndex(original->field(field)->name()) < 0) {
      return Fail(arrow::Status::Invalid("ambiguous IPC column: ", original->field(field)->name()), error);
    }
    fields.push_back(original->field(field)->WithType(ImportType(original->field(field)->type())));
    positions.push_back(source->schema()->GetFieldIndex(original->field(field)->name()));
  }
  auto schema = arrow::schema(fields, original->metadata());
  auto reader = std::make_shared<Reader>(source, positions, schema, n_rows);
  auto status = arrow::ExportRecordBatchReader(reader, out);
  return status.ok() ? 0 : Fail(status, error);
}

// Reads only the schema/header/footer, never a record-batch body.
int dfq_read_ipc_schema(const char* path, int stream, ArrowArray* array,
                        ArrowSchema* schema, char** error) {
  *error = nullptr;
  array->release = nullptr;
  schema->release = nullptr;
  auto source = Open(path, stream != 0, false);
  if (!source.ok()) return Fail(source.status(), error);
  auto batch = arrow::RecordBatch::MakeEmpty(ImportSchema((*source)->schema()));
  if (!batch.ok()) return Fail(batch.status(), error);
  auto status = arrow::ExportRecordBatch(**batch, array, schema);
  return status.ok() ? 0 : Fail(status, error);
}

int dfq_ipc_writer_open(const char* path, ArrowSchema* schema,
                        const char* compression, int stream, void** out, char** error) {
  *error = nullptr;
  *out = nullptr;
  auto options = arrow::ipc::IpcWriteOptions::Defaults();
  std::string codec(compression);
  if (codec != "uncompressed") {
    if (codec != "lz4" && codec != "zstd") {
      return Fail(arrow::Status::Invalid("unsupported IPC compression: ", codec), error);
    }
    auto created = arrow::util::Codec::Create(codec == "lz4" ? arrow::Compression::LZ4_FRAME
                                                            : arrow::Compression::ZSTD);
    if (!created.ok()) return Fail(created.status(), error);
    options.codec = std::move(*created);
  }
  auto imported = arrow::ImportSchema(schema);
  if (!imported.ok()) return Fail(imported.status(), error);
  auto handle = std::make_unique<Writer>();
  auto output = arrow::io::FileOutputStream::Open(path);
  if (!output.ok()) return Fail(output.status(), error);
  handle->output = *output;
  auto writer = stream ? arrow::ipc::MakeStreamWriter(handle->output, *imported, options)
                       : arrow::ipc::MakeFileWriter(handle->output, *imported, options);
  if (!writer.ok()) return Fail(writer.status(), error);
  handle->writer = *writer;
  *out = handle.release();
  return 0;
}

int dfq_ipc_writer_write(void* handle, ArrowArray* array, ArrowSchema* schema,
                         char** error) {
  *error = nullptr;
  auto batch = arrow::ImportRecordBatch(array, schema);
  if (!batch.ok()) return Fail(batch.status(), error);
  auto status = static_cast<Writer*>(handle)->writer->WriteRecordBatch(**batch);
  return status.ok() ? 0 : Fail(status, error);
}

int dfq_ipc_writer_close(void* handle, char** error) {
  *error = nullptr;
  std::unique_ptr<Writer> owned(static_cast<Writer*>(handle));
  auto status = owned->writer->Close();
  auto closed = owned->output->Close();
  if (!status.ok()) return Fail(status, error);
  return closed.ok() ? 0 : Fail(closed, error);
}

void dfq_ipc_writer_abort(void* handle) {
  delete static_cast<Writer*>(handle);
}
}  // extern "C"
