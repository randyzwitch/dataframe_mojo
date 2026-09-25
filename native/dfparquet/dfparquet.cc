// dfparquet: a minimal C ABI over Arrow C++'s Parquet reader.
//
// One entry point reads a file (optionally a column subset) into a single
// record batch and exports it through the Arrow C Data Interface. Only the
// dfq_* symbols are visible; Arrow and its bundled codecs are linked in
// statically and hidden.
#include <arrow/c/bridge.h>
#include <arrow/io/file.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/util/config.h>
#include <parquet/arrow/reader.h>
#include <parquet/properties.h>

#include <chrono>
#include <cstdio>
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

}  // namespace

extern "C" {

// Returns 0 and fills out_array/out_schema (caller releases both through
// their release callbacks). Otherwise returns 1 and *error_out holds a
// message to free with dfq_free.
int dfq_read_parquet(const char* path, int use_threads, const char** columns,
                     int n_columns, struct ArrowArray* out_array,
                     struct ArrowSchema* out_schema, char** error_out) {
  *error_out = nullptr;
  auto file = arrow::io::ReadableFile::Open(path);
  if (!file.ok()) return fail(file.status(), error_out);

  parquet::ArrowReaderProperties properties;
  properties.set_use_threads(use_threads != 0);
  properties.set_pre_buffer(true);
  parquet::arrow::FileReaderBuilder builder;
  arrow::Status status = builder.Open(*file);
  if (!status.ok()) return fail(status, error_out);
  builder.properties(properties);
  std::unique_ptr<parquet::arrow::FileReader> reader;
  status = builder.Build(&reader);
  if (!status.ok()) return fail(status, error_out);

  auto t0 = std::chrono::steady_clock::now();
  arrow::Result<std::shared_ptr<arrow::Table>> table;
  if (n_columns > 0) {
    // Flat schemas only for now: Arrow field index == Parquet leaf index.
    std::shared_ptr<arrow::Schema> schema;
    status = reader->GetSchema(&schema);
    if (!status.ok()) return fail(status, error_out);
    std::vector<int> indices;
    for (int i = 0; i < n_columns; ++i) {
      int index = schema->GetFieldIndex(columns[i]);
      if (index < 0) {
        return fail(arrow::Status::KeyError("no column named ", columns[i]),
                    error_out);
      }
      indices.push_back(index);
    }
    table = reader->ReadTable(indices);
  } else {
    table = reader->ReadTable();
  }
  if (!table.ok()) return fail(table.status(), error_out);
  auto t1 = std::chrono::steady_clock::now();
  int chunks = (*table)->num_columns() > 0 ? (*table)->column(0)->num_chunks() : 0;
  auto batch = (*table)->CombineChunksToBatch();
  if (!batch.ok()) return fail(batch.status(), error_out);
  auto t2 = std::chrono::steady_clock::now();
  status = arrow::ExportRecordBatch(**batch, out_array, out_schema);
  if (!status.ok()) return fail(status, error_out);
  if (std::getenv("DFQ_TIMING")) {
    using ms = std::chrono::duration<double, std::milli>;
    std::fprintf(stderr, "dfq: read_table %.1f ms (%d chunks), combine %.1f ms\n",
                 ms(t1 - t0).count(), chunks, ms(t2 - t1).count());
  }
  return 0;
}

void dfq_free(void* pointer) { std::free(pointer); }

const char* dfq_arrow_version() { return ARROW_VERSION_STRING; }

}  // extern "C"
