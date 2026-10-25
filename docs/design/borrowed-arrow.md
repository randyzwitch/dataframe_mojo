# Retained Arrow buffer ownership

`_Buffer[T]` separates shared immutable payload access from writable storage.
Owned buffers contain Mojo lists. A borrowed buffer contains a pointer, a
length, and a reference-counted `_ForeignOwner`. The owner holds an opaque
context and a C-compatible release callback. Wrapping an existing owned
`ArcPointer[List[T]]` also retains its allocation without copying its values.

Reading, copying and slicing a column share this storage. `writable()` first
detaches any shared or externally owned buffer into a private Mojo list.
Column append and reserve paths also compact the current window before
mutating it. Read-only access never materializes a borrowed payload. Existing
unsafe bulk pointers retain their read-only contract: keeping a column alive
does not authorize modifying shared or mapped memory through them.

For `import_arrow(..., copy=False)`, the importer transfers the parent
`ArrowArray` into a stable heap context and clears the caller's release slot.
Parallel column jobs share that owner. Imported columns retain it only when
they borrow a payload. After binding and metadata copying, the schema export
is released independently. A producer-array callback runs exactly once when
the final dependent buffer dies, including when only a slice or another
Arrow export remains. Import errors join all jobs before dropping their
partial columns and the parent owner.

Compatible fixed-width values can retain their original address. Alignment
and conversion requirements still apply: unaligned payloads are byte-copied
to aligned owned storage; date/time scaling or storage-width changes retain
the existing conversion path. This foundation leaves other buffer layouts on
the copy path until their borrowing support is integrated.

The storage contracts exercise actual exporter callbacks, a native mapped
IPC stream, retained views, mutation isolation, ordinary copy-by-default
imports, and serial/parallel error paths. They check cleanup after scoped
helpers return, so temporary typed views are also gone before final release
counts are asserted. Native IPC tests require the IPC-enabled shared library
from the preceding I/O changes.
