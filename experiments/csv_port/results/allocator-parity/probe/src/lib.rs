//! Isolated LD_PRELOAD experiment matching Polars 1.44.2's Unix allocator.
//! Polars uses a prefixed `_rjem_` jemalloc build; these C wrappers only expose
//! the standard names to the test process and are not a production dependency.
use core::ffi::c_void;
use libc::{c_int, size_t};
use tikv_jemallocator::Jemalloc;
use tikv_jemalloc_sys as rjem;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[no_mangle]
pub unsafe extern "C" fn malloc(size: size_t) -> *mut c_void {
    rjem::malloc(size)
}

#[no_mangle]
pub unsafe extern "C" fn calloc(count: size_t, size: size_t) -> *mut c_void {
    rjem::calloc(count, size)
}

#[no_mangle]
pub unsafe extern "C" fn realloc(pointer: *mut c_void, size: size_t) -> *mut c_void {
    rjem::realloc(pointer, size)
}

#[no_mangle]
pub unsafe extern "C" fn free(pointer: *mut c_void) {
    rjem::free(pointer)
}

#[no_mangle]
pub unsafe extern "C" fn posix_memalign(
    result: *mut *mut c_void, alignment: size_t, size: size_t,
) -> c_int {
    rjem::posix_memalign(result, alignment, size)
}

#[no_mangle]
pub unsafe extern "C" fn aligned_alloc(alignment: size_t, size: size_t) -> *mut c_void {
    rjem::aligned_alloc(alignment, size)
}

#[no_mangle]
pub unsafe extern "C" fn malloc_usable_size(pointer: *const c_void) -> size_t {
    rjem::malloc_usable_size(pointer)
}

#[no_mangle]
pub unsafe extern "C" fn polars_jemalloc_probe_smoke() -> c_int {
    let pointer = rjem::malloc(4096);
    if pointer.is_null() { return 1; }
    rjem::free(pointer);
    0
}
