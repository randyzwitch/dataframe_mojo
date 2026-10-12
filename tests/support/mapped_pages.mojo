"""Observe actual virtual mappings without depending on resident page state."""
from std.ffi import external_call
from std.memory import Pointer
from std.sys import CompilationTarget
from std.testing import assert_equal


def _mapping_probe(address: Int) -> Tuple[Int32, Int32]:
    var page = Int(external_call["getpagesize", Int32]())
    # MS_ASYNC is 1 on Linux and Darwin. Unlike Darwin mincore, msync reports
    # ENOMEM for an unmapped page; a mapped but nonresident page still succeeds.
    var result = external_call["msync", Int32](address, page, Int32(1))
    var location: Pointer[Int32, MutAnyOrigin]
    comptime if CompilationTarget.is_macos():
        location = external_call["__error", Pointer[Int32, MutAnyOrigin]]()
    else:
        location = external_call[
            "__errno_location", Pointer[Int32, MutAnyOrigin]
        ]()
    var error = location[]
    return (result, error)


def assert_mapped_page(address: Int) raises:
    assert_equal(
        _mapping_probe(address)[0], Int32(0), "source page must be mapped"
    )


def assert_unmapped_page(address: Int) raises:
    var result = _mapping_probe(address)
    assert_equal(
        result[0], Int32(-1), "producer backing page must really be unmapped"
    )
    assert_equal(result[1], Int32(12), "unmapped page must report ENOMEM")
