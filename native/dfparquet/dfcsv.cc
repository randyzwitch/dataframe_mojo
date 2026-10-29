// Incremental CSV transport only. CSV grammar and typed decoding stay in Mojo.
#include <arrow/buffer.h>
#include <arrow/io/compressed.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <arrow/util/compression.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>

namespace {
int Fail(const arrow::Status& status, char** error) {
  if (error) {
    const auto text = status.ToString();
    *error = static_cast<char*>(std::malloc(text.size() + 1));
    if (*error) std::memcpy(*error, text.c_str(), text.size() + 1);
  }
  return 1;
}

struct CsvInput {
  // The codec outlives the stream. Closing stream also closes raw.
  std::unique_ptr<arrow::util::Codec> codec;
  std::shared_ptr<arrow::io::InputStream> raw;
  std::shared_ptr<arrow::io::CompressedInputStream> stream;
  ~CsvInput() {
    if (stream) (void)stream->Close();
    else if (raw) (void)raw->Close();
  }
};

arrow::Result<std::unique_ptr<CsvInput>> Open(
    std::shared_ptr<arrow::io::InputStream> raw, int codec) {
  auto input = std::make_unique<CsvInput>();
  input->raw = std::move(raw);
  arrow::Compression::type kind;
  switch (codec) {
    case 1: kind = arrow::Compression::GZIP; break;
    case 2: kind = arrow::Compression::ZSTD; break;
    default: return arrow::Status::Invalid("unsupported CSV compression codec");
  }
  ARROW_ASSIGN_OR_RAISE(input->codec, arrow::util::Codec::Create(kind));
  ARROW_ASSIGN_OR_RAISE(input->stream,
                       arrow::io::CompressedInputStream::Make(
                           input->codec.get(), input->raw));
  return input;
}
}  // namespace

extern "C" {
// codec: 1=gzip, 2=zstd. Output handles and errors are initialized on entry.
int dfq_csv_input_open_path(const char* path, int codec, void** out,
                           char** error) {
  *out = nullptr;
  *error = nullptr;
  auto raw = arrow::io::ReadableFile::Open(path);
  if (!raw.ok()) return Fail(raw.status(), error);
  auto input = Open(*raw, codec);
  if (!input.ok()) return Fail(input.status(), error);
  *out = input->release();
  return 0;
}

// Borrowed compressed bytes: the caller MUST retain the byte owner until close.
// Native code never mutates or frees the bytes.
int dfq_csv_input_open_buffer(const uint8_t* data, int64_t length, int codec,
                             void** out, char** error) {
  *out = nullptr;
  *error = nullptr;
  if (length < 0 || (!data && length))
    return Fail(arrow::Status::Invalid("invalid CSV input buffer"), error);
  auto buffer = std::make_shared<arrow::Buffer>(data, length);
  auto raw = std::make_shared<arrow::io::BufferReader>(std::move(buffer));
  auto input = Open(std::move(raw), codec);
  if (!input.ok()) return Fail(input.status(), error);
  *out = input->release();
  return 0;
}

int dfq_csv_input_read(void* handle, uint8_t* destination, int64_t capacity,
                      int64_t* bytes_read, char** error) {
  *bytes_read = 0;
  *error = nullptr;
  if (!handle || capacity <= 0 || !destination)
    return Fail(arrow::Status::Invalid("invalid CSV read request"), error);
  auto input = static_cast<CsvInput*>(handle);
  auto result = input->stream->Read(capacity, destination);
  if (!result.ok()) return Fail(result.status(), error);
  if (*result == 0) {
    auto position = input->raw->Tell();
    if (!position.ok()) return Fail(position.status(), error);
    if (*position == 0)
      return Fail(arrow::Status::IOError("Empty compressed CSV input"), error);
  }
  *bytes_read = *result;
  return 0;
}

void dfq_csv_input_close(void* handle) { delete static_cast<CsvInput*>(handle); }
}  // extern "C"
