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
  DFM_FIRST = 83,
  DFM_LAST = 84,
  DFM_ANY = 91,
  DFM_ALL = 92,
  DFM_NULL_COUNT = 93,
  DFM_ARG_MIN = 95,
  DFM_ARG_MAX = 96,
  DFM_LEN = 90,
  DFM_SORT_KEY = 200,
  DFM_GROUP_HEADER = 201,
  DFM_GROUP_KEY = 202,
  DFM_GROUP_OUTPUT = 203,
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
  int64_t filter; // 0 projection, 1 stable filter, 2 stable sort, 3 grouped
                  // reduction.
  int64_t gather_start;
  int64_t gather_count;
} DFMStep;

typedef struct DFMOutput {
  void *values;      // Caller-owned capacity: rows, or one scalar.
  uint8_t *validity; // Caller-owned packed bitmap, initially zero.
  int64_t dtype;
  int64_t slot;
  int64_t reduction; // -1 for row output.
  int64_t min_count; // Sum minimum; Boolean any/all ignore_nulls flag.
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

// Additive workspace inspection; existing request/stats layouts stay intact.
typedef struct DFMWorkspaceInfo {
  int64_t idle_bytes;
  int64_t idle_buffers;
  int64_t reuse_hits;
  int64_t allocations;
  int64_t limit_bytes;
} DFMWorkspaceInfo;

// Versioned two-phase join interface; row request ABI remains unchanged.
#define DFM_JOIN_ABI_VERSION 1

typedef struct DFMJoinInput {
  DFMInput input;
  int64_t side; // 0 left, 1 right.
} DFMJoinInput;
typedef struct DFMJoinKey {
  int64_t left, right; // Indices into the input descriptors.
} DFMJoinKey;
typedef struct DFMJoinOutput {
  void *values;
  uint8_t *validity;
  int64_t dtype;
  int64_t left, right;  // Input indices; -1 for a missing source.
  int64_t prefer_right; // Right joins retain the right key even on matches.
} DFMJoinOutput;
typedef struct DFMJoinRequest {
  int64_t abi_version;
  int64_t left_rows, right_rows;
  int64_t input_count, key_count, output_count;
  int64_t how;   // inner, left, right, full, semi, anti, cross = 0..6.
  int64_t limit; // -1 or final head.
  int64_t memory_budget_bytes;
  int64_t profiling;
  const DFMJoinInput *inputs;
  const DFMJoinKey *keys;
  const DFMJoinOutput *outputs; // Storage pointers may be null during prepare.
} DFMJoinRequest;

// Resident ABI 1. A null table selects the ordinary CPU input descriptor.
// A non-null table selects an immutable native column; its rows and dtype must
// match the descriptor, and it must belong to this query scope.
#define DFM_RESIDENT_ABI_VERSION 1
typedef struct DFMResidentInput {
  void *table;
  int64_t column;
} DFMResidentInput;
typedef struct DFMResidentInfo {
  int64_t rows;
  int64_t columns;
  int64_t retained_bytes;
} DFMResidentInfo;
typedef struct DFMQueryInfo {
  int64_t live_bytes;
  int64_t peak_bytes;
  int64_t memory_budget_bytes;
} DFMQueryInfo;

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
DFM_API int32_t dfm_workspace_info(void *context, DFMWorkspaceInfo *info,
                                  char **error);
// Releases only idle buffers; live join sessions and in-flight work keep owners.
DFM_API int32_t dfm_workspace_trim(void *context, char **error);
DFM_API int64_t dfm_resident_abi_version(void);
DFM_API void *dfm_query_create(void *context, int64_t memory_budget_bytes,
                               char **error);
DFM_API void dfm_query_release(void *query);
DFM_API int32_t dfm_query_info(void *query, DFMQueryInfo *info, char **error);
// Every call drains its submissions before returning. Resident tables retain
// their query/context, never alias recycled storage, and can feed multiple
// consumers. final_output packs bitmaps and reserves the CPU result payload;
// intermediate calls retain native words and validity without materialization.
DFM_API int32_t dfm_resident_rows(void *query, const DFMRequest *request,
    const DFMResidentInput *inputs, int64_t input_count, int64_t final_output,
    void **table, DFMStats *stats, char **error);
DFM_API int32_t dfm_resident_join(void *query, const DFMJoinRequest *request,
    const DFMResidentInput *inputs, int64_t input_count, int64_t final_output,
    void **table, DFMStats *stats, char **error);
DFM_API int32_t dfm_resident_info(void *table, DFMResidentInfo *info, char **error);
// Export checks all metadata and the complete CPU result budget before touching
// any caller output. Request dtype and column order must match the table.
DFM_API int32_t dfm_resident_export(void *table, const DFMOutput *outputs,
    int64_t output_count, DFMStats *stats, char **error);
DFM_API void dfm_resident_release(void *table);

// Pure shape validation and allocation accounting; no device initialization.
DFM_API int32_t dfm_estimate(const DFMRequest *request, DFMMemory *memory,
                             char **error);
// Pure flags for the implementation selected by this runtime build. Bit zero
// denotes predicate-aware scalar reduction without row compaction.
DFM_API int64_t dfm_plan_features(const DFMRequest *request, char **error);
// Checks an existing cache without compiling or submitting GPU work.
// Returns 1 if cached, 0 if missing, or -1 with an allocated error message.
DFM_API int32_t dfm_plan_cached(void *context, const DFMRequest *request,
                                char **error);
// Synchronous boundary: all GPU users have stopped before success or error.
// Returns 0 on success or 1 with an allocated error message; free via dfm_free.
DFM_API int32_t dfm_execute(void *context, const DFMRequest *request,
                            DFMStats *stats, char **error);
// Estimate the known matching payload without device initialization. Actual
// output allocations are checked separately after GPU matching determines size.
DFM_API int32_t dfm_join_estimate(const DFMJoinRequest *request,
                                  DFMMemory *memory, char **error);
DFM_API int32_t dfm_join_prepare(void *context, const DFMJoinRequest *request,
                                 void **session, DFMStats *stats, char **error);
DFM_API int32_t dfm_join_finish(void *session, const DFMJoinOutput *outputs,
                                int64_t output_count, DFMStats *stats,
                                char **error);
DFM_API void dfm_join_release(void *session);
DFM_API void dfm_free(void *pointer);

#ifdef __cplusplus
}
#endif

#endif
