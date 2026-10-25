// Diagnostic child-process interposition of the exported Mojo allocation ABI.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <sys/types.h>
#include <unordered_map>

struct State {
    std::mutex mutex;
    std::unordered_map<void *, std::size_t> sizes;
    std::uint64_t live = 0, peak = 0, allocated = 0, released = 0;
    std::uint64_t alloc_calls = 0, free_calls = 0, unknown_frees = 0, duplicate_allocs = 0;
};
static State &state() { static State *s = new State; return *s; }
static void *symbol(const char *name) {
    void *p = dlsym(RTLD_NEXT, name);
    if (!p) { std::fprintf(stderr, "allocation_probe_missing_symbol %s\n", name); std::abort(); }
    return p;
}
extern "C" void *KGEN_CompilerRT_AlignedAlloc(ssize_t alignment, ssize_t size) {
    using Fn = void *(*)(ssize_t, ssize_t);
    static Fn real = reinterpret_cast<Fn>(symbol("KGEN_CompilerRT_AlignedAlloc"));
    void *p = real(alignment, size);
    if (p) {
        auto &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        std::size_t bytes = size > 0 ? static_cast<std::size_t>(size) : 0;
        if (!s.sizes.emplace(p, bytes).second) ++s.duplicate_allocs;
        else { s.live += bytes; s.peak = std::max(s.peak, s.live); s.allocated += bytes; }
        ++s.alloc_calls;
    }
    return p;
}
extern "C" void KGEN_CompilerRT_AlignedFree(void *p) {
    using Fn = void (*)(void *);
    static Fn real = reinterpret_cast<Fn>(symbol("KGEN_CompilerRT_AlignedFree"));
    if (p) {
        auto &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        auto found = s.sizes.find(p);
        if (found == s.sizes.end()) ++s.unknown_frees;
        else { s.live -= found->second; s.released += found->second; s.sizes.erase(found); }
        ++s.free_calls;
    }
    real(p);
}
__attribute__((destructor)) static void report() {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    std::fprintf(stderr, "compiler_alloc_probe live=%llu peak=%llu allocated=%llu released=%llu alloc_calls=%llu free_calls=%llu outstanding=%zu unknown_frees=%llu duplicate_allocs=%llu\n",
        static_cast<unsigned long long>(s.live), static_cast<unsigned long long>(s.peak),
        static_cast<unsigned long long>(s.allocated), static_cast<unsigned long long>(s.released),
        static_cast<unsigned long long>(s.alloc_calls), static_cast<unsigned long long>(s.free_calls),
        s.sizes.size(), static_cast<unsigned long long>(s.unknown_frees),
        static_cast<unsigned long long>(s.duplicate_allocs));
}
