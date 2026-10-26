/* Benchmark-only Mojo allocation requests, at the pinned runtime boundary.
 * Arrow and other native allocators are excluded; RSS covers the whole process. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <dlfcn.h>
static _Atomic uint64_t requested = 0, calls = 0;
static _Atomic int enabled = 0;
void df_bench_alloc_reset(void) {
    atomic_store(&enabled, 0);
    atomic_store(&requested, 0);
    atomic_store(&calls, 0);
    atomic_store(&enabled, 1);
}
uint64_t df_bench_alloc_bytes(void) { return atomic_load(&requested); }
uint64_t df_bench_alloc_calls(void) { return atomic_load(&calls); }
static void record(void *result, size_t bytes) {
    if (result && atomic_load(&enabled)) {
        atomic_fetch_add(&requested, bytes);
        atomic_fetch_add(&calls, 1);
    }
}
typedef void *(*AlignedAlloc)(size_t alignment, size_t bytes);
#ifdef __APPLE__
extern void *KGEN_CompilerRT_AlignedAlloc(size_t alignment, size_t bytes);
static void *observed_alloc(size_t alignment, size_t bytes) {
    void *p = KGEN_CompilerRT_AlignedAlloc(alignment, bytes);
    record(p, bytes);
    return p;
}
__attribute__((used)) static struct { const void *replacement; const void *original; }
interpose_alloc __attribute__((section("__DATA,__interpose"))) =
{(const void *)observed_alloc, (const void *)KGEN_CompilerRT_AlignedAlloc};
#else
void *KGEN_CompilerRT_AlignedAlloc(size_t alignment, size_t bytes) {
    static _Atomic(AlignedAlloc) original = NULL;
    AlignedAlloc real = atomic_load(&original);
    if (!real) {
        real = (AlignedAlloc)dlsym(RTLD_NEXT, "KGEN_CompilerRT_AlignedAlloc");
        if (!real) abort();
        atomic_store(&original, real);
    }
    void *p = real(alignment, bytes);
    record(p, bytes);
    return p;
}
#endif
