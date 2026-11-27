"""Own aligned host memory borrowed by a Metal buffer until synchronization."""
from std.ffi import external_call
from std.memory import Pointer


struct SharedPages(Movable):
    # Metal borrows these pages until the queued reductions have finished.
    var address: Int

    def __init__(out self, size: Int, alignment: Int) raises:
        var address = 0
        if (
            size
            and external_call["posix_memalign", Int32](
                Pointer(to=address), alignment, size
            )
            != 0
        ):
            raise Error("Could not allocate page-aligned shared Metal storage")
        self.address = address

    def __deinit__(deinit self):
        if self.address:
            external_call["free", NoneType](self.address)
