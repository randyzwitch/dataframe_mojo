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

The buffer abstraction covers fixed-width values, validity and Boolean
bitmaps, string/binary bytes and Int64 offsets, native view descriptors and
blocks, list offsets and recursively imported children, and compatible
categorical codes and labels. Foreign windows align buffer bases to a bitmap
byte and keep at most seven prefix rows. This preserves the same row offset
for payload and validity without copying or rotating a bitmap. A dictionary
label window can retain a buffer state through another opaque owner.

Unaligned typed data is byte-copied to aligned storage. Int32 offsets widen
without copying byte payloads or children. Non-UInt32 categorical indices and
unsupported temporal storage units convert as before. Null Arrow view slots
may contain undefined descriptors; unsafe null descriptors are replaced in
owned descriptor storage, while byte blocks remain shared. Canonical Arrow
`U`/`Z` exports of view storage materialize a contiguous adapter and retain it
until export release. Numeric rechunking and other operations that explicitly
require contiguous output also allocate.

The IPC bridge preserves `vu`/`vz` buffers rather than casting them to large
strings/binary. Eager and lazy IPC readers borrow compatible buffers and
retain their parent array independently from the reader stream. Decompression
necessarily creates decoded Arrow buffers; those buffers are then retained
without a second payload copy. Explicit C Data imports still default to
copying, and layout conversions retain their documented allocation paths.

The storage contracts exercise actual exporter callbacks, a native mapped
IPC stream, retained views, mutation isolation, ordinary copy-by-default
imports, and serial/parallel error paths. They check cleanup after scoped
helpers return, so temporary typed views are also gone before final release
counts are asserted. Native IPC tests require the IPC-enabled shared library
from the preceding I/O changes.

Mapped projections use Arrow's cached range reader after prebuffering only
the requested batch's metadata. In the pinned Arrow 24 library, the ordinary
synchronous selective path assembles a copied body even for mapped input.
The cached path retains mapped buffer ranges and still skips decoding
excluded fields. The bridge separately validates the message/footer body
length, which the cached Arrow path does not compare. It never prebuffers
later record-batch bodies, preserving early stops and skipped corrupt-body
behavior. Native-address tests observe upstream buffers before import, close
the reader, and verify actual unmapping after the final column view dies.
