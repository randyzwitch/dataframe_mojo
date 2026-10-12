// Scratch diagnostic: no Arrow/PyArrow link or headers, only native C ABI.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <vector>

struct ArrowArray {
  int64_t length, null_count, offset, n_buffers, n_children;
  const void** buffers;
  ArrowArray** children;
  ArrowArray* dictionary;
  void (*release)(ArrowArray*);
  void* private_data;
};
struct ArrowSchema {
  const char *format, *name, *metadata;
  int64_t flags, n_children;
  ArrowSchema** children;
  ArrowSchema* dictionary;
  void (*release)(ArrowSchema*);
  void* private_data;
};
struct ArrowArrayStream {
  int (*get_schema)(ArrowArrayStream*, ArrowSchema*);
  int (*get_next)(ArrowArrayStream*, ArrowArray*);
  const char* (*get_last_error)(ArrowArrayStream*);
  void (*release)(ArrowArrayStream*);
  void* private_data;
};
void require(bool ok, const std::string& message) {
  if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
void check(ArrowArray& array, ArrowSchema& schema, int group) {
  require(array.length == 1000, "Expected 1000 rows per row group");
  for (int field = 0; field < array.n_children; ++field) {
    auto* child = array.children[field];
    auto* type = schema.children[field];
    if (!std::strcmp(type->format, "l")) {
      auto* values = static_cast<const int64_t*>(child->buffers[1]);
      std::cout << "Raw C-only id group=" << group << " head="
                << values[child->offset] << ',' << values[child->offset + 1]
                << ',' << values[child->offset + 2] << '\n' << std::flush;
      for (int row = 0; row < child->length; ++row)
        require(values[child->offset + row] == group * 1000 + row,
                "C-only int64 mismatch group=" + std::to_string(group) +
                " row=" + std::to_string(row));
    } else if (!std::strcmp(type->format, "u")) {
      auto* offsets = static_cast<const int32_t*>(child->buffers[1]);
      auto* bytes = static_cast<const char*>(child->buffers[2]);
      for (int row = 0; row < child->length; ++row) {
        int64_t i = child->offset + row;
        require(std::string(bytes + offsets[i], offsets[i + 1] - offsets[i]) ==
                    "row-" + std::to_string(group * 1000 + row),
                "C-only string mismatch");
      }
    } else require(false, "Unexpected C schema field format");
  }
}
int main(int argc, char** argv) {
  require(argc == 3, "Usage: native_stream_plain library fixture");
  auto* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  require(lib, "Native dlopen failed");
  using Read = int (*)(const char*, int, const char**, int, const int*, int,
                      ArrowArrayStream*, char**);
  auto read = reinterpret_cast<Read>(dlsym(lib, "dfq_read_parquet_stream"));
  require(read, "Missing native stream ABI");
  for (int selection = 0; selection < 2; ++selection) {
    ArrowArrayStream stream{};
    ArrowSchema schema{};
    char* error = nullptr;
    const char* names[] = {"text", "id"};
    const int groups[] = {2, 0, 2};
    require(read(argv[2], 0, selection ? names : nullptr, selection ? 2 : 0,
                 selection ? groups : nullptr, selection ? 3 : -1,
                 &stream, &error) == 0,
            error ? error : "Opening native stream failed");
    require(stream.get_schema(&stream, &schema) == 0, "C stream schema failed");
    std::vector<ArrowArray> retained;
    for (int batch = 0; batch < 3; ++batch) {
      ArrowArray array{};
      require(stream.get_next(&stream, &array) == 0, "C stream next failed");
      require(array.release, "Unexpected early EOF");
      check(array, schema, selection ? groups[batch] : batch);
      retained.push_back(array);
    }
    ArrowArray eof{};
    require(stream.get_next(&stream, &eof) == 0 && !eof.release,
            "Missing expected EOF");
    stream.release(&stream);
    for (int batch = 0; batch < 3; ++batch) {
      check(retained[batch], schema, selection ? groups[batch] : batch);
      retained[batch].release(&retained[batch]);
    }
    schema.release(&schema);
  }
  dlclose(lib);
  std::cout << "Pure C ABI native stream values, projection, order and retention passed\n";
}
