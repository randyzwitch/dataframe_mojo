#ifndef DATAFRAME_METAL_H
#define DATAFRAME_METAL_H

#include <stdint.h>

// The Mojo wrapper passes addresses of these structs; no C++ or Objective-C
// object crosses the ABI. Bump the version when layouts or meanings change.
#define DFM_ABI_VERSION 2
#define DFM_API __attribute__((visibility("default")))

enum DFMType {
  DFM_FLOAT32 = 1,
  DFM_INT32 = 2,
  DFM_INT64 = 3,
  DFM_BOOL = 4,
  DFM_INT8 = 5,
  DFM_INT16 = 6,
  DFM_UINT8 = 7,
  DFM_UINT16 = 8,
  DFM_UINT32 = 9,
  DFM_UINT64 = 10,
};

// Fixed native opcodes. The Mojo wrapper translates the shared expression IR
// rather than relying on its numeric constants remaining unchanged.
enum DFMOp {
  DFM_COLUMN = 0,
  DFM_LITERAL_INT = 1,
  DFM_LITERAL_FLOAT = 2,
  DFM_LITERAL_BOOL = 3,
  DFM_ADD = 5,
  DFM_SUB = 6,
  DFM_MUL = 7,
  DFM_GT = 8,
  DFM_EQ = 9,
  DFM_SUM = 10,
  DFM_COUNT = 11,
  DFM_LITERAL_NULL = 12,
  DFM_LT = 20,
  DFM_GE = 21,
  DFM_LE = 22,
  DFM_NE = 23,
  DFM_AND = 30,
  DFM_OR = 31,
  DFM_XOR = 32,
  DFM_FILL_NULL = 33,
  DFM_NEG = 50,
  DFM_NOT = 60,
  DFM_IS_NULL = 61,
  DFM_IS_NOT_NULL = 62,
  DFM_FLOORDIV = 25,
  DFM_MOD = 26,
  DFM_POW = 27,
  DFM_CLIP_LOW = 28,
  DFM_CLIP_HIGH = 29,
  DFM_FILL_NAN = 34,
  DFM_KEEP_NULLS = 35,
  DFM_ABS = 51,
  DFM_FLOOR = 55,
  DFM_CEIL = 56,
  DFM_ROUND = 57,
  DFM_IS_NAN = 63,
  DFM_IS_NOT_NAN = 64,
  DFM_IS_FINITE = 65,
  DFM_IS_INFINITE = 66,
  DFM_WHEN = 100,
  DFM_CAST = 79,
  DFM_MIN = 80,
  DFM_MAX = 81,
  DFM_LEN = 90,
  DFM_SORT_KEY = 200,
};

typedef struct DFMInput {
  const void *values;      // Numeric row zero or Boolean byte zero.
  const uint8_t *validity; // Selected byte window; null if all valid.
  int64_t dtype;
  int64_t bit_offset; // Row zero's bit within the selected byte.
  int64_t has_validity;
} DFMInput;

typedef struct DFMStep {
  int64_t start;
  int64_t nodes;
  int64_t slot;
  int64_t filter; // 0 projection, 1 stable filter, 2 stable sort.
  int64_t gather_start;
  int64_t gather_count;
} DFMStep;

typedef struct DFMOutput {
  void *values;      // Caller-owned capacity: rows, or one scalar.
  uint8_t *validity; // Caller-owned packed bitmap, initially zero.
  int64_t dtype;
  int64_t slot;
  int64_t reduction; // -1 for row output.
  int64_t min_count;
} DFMOutput;

typedef struct DFMRequest {
  int64_t abi_version;
  int64_t rows;
  int64_t dtype; // Common numeric type, or zero for typed 64-bit word storage.
  int64_t slots;
  int64_t input_count;
  int64_t code_words;
  int64_t literal_count;
  int64_t step_count;
  int64_t gather_count;
  int64_t output_count;
  int64_t limit; // -1 or final head length.
  int64_t reductions;
  int64_t profiling;
  int64_t memory_budget_bytes; // -1 uses runtime headroom; zero denies payload.
  const DFMInput *inputs;
  const int64_t *code;      // Four words per bound expression node.
  const uint64_t *literals; // Host literal bits; converted before GPU use.
  const DFMStep *steps;
  const int64_t *gathers;
  const DFMOutput *outputs;
  const int64_t *node_types; // Required for typed storage; one per node.
  const int64_t
      *slot_types; // Required for typed storage; one per logical slot.
} DFMRequest;

typedef struct DFMMemory {
  int64_t shared_bytes;
  int64_t result_bytes;
  int64_t peak_bytes;
  int64_t upload_bytes;
  int64_t launches;
  int64_t blocks;
  int64_t capacity;
  int64_t packed_bytes;
} DFMMemory;

typedef struct DFMStats {
  DFMMemory memory;
  int64_t output_rows;
  int64_t download_bytes;
  int64_t synchronizations;
  int64_t pipeline_cache_hit;
  int64_t pipeline_compile_ns;
  int64_t staging_copy_ns;
  int64_t submit_wait_ns;
  int64_t result_copy_ns;
  int64_t gpu_ns; // -1 when no valid GPU interval was observed.
} DFMStats;

typedef struct DFMDeviceInfo {
  const char *name; // Valid while the context is retained.
  uint64_t registry_id;
  int64_t recommended_bytes;
  int64_t allocated_bytes;
  int64_t max_threads;
  int64_t unified_memory;
} DFMDeviceInfo;

#ifdef __cplusplus
extern "C" {
#endif

DFM_API int64_t dfm_abi_version(void);
DFM_API const char *dfm_build_id(void);
DFM_API int64_t dfm_device_count(void);
DFM_API void *dfm_context_create(int64_t device_id, int64_t reuse_default,
                                 char **error);
DFM_API void dfm_context_retain(void *context);
DFM_API void dfm_context_release(void *context);
DFM_API int32_t dfm_context_info(void *context, DFMDeviceInfo *info,
                                 char **error);

// Pure shape validation and allocation accounting; no device initialization.
DFM_API int32_t dfm_estimate(const DFMRequest *request, DFMMemory *memory,
                             char **error);
// Checks an existing cache without compiling or submitting GPU work.
// Returns 1 if cached, 0 if missing, or -1 with an allocated error message.
DFM_API int32_t dfm_plan_cached(void *context, const DFMRequest *request,
                                char **error);
// Synchronous boundary: all GPU users have stopped before success or error.
// Returns 0 on success or 1 with an allocated error message; free via dfm_free.
DFM_API int32_t dfm_execute(void *context, const DFMRequest *request,
                            DFMStats *stats, char **error);
DFM_API void dfm_free(void *pointer);

#ifdef __cplusplus
}
#endif

#endif
