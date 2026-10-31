"""Physical eligibility checks before scalar-kernel shortcuts."""
from dataframe.series import Series


@no_inline
def _has_physical_storage[T: Copyable & Deinitable](input: Series) -> Bool:
    """Validate every cold window, or the actual immutable published cache.

    A private constructor can combine unlike storage under one logical tag.
    Callers retain their logical-type admission and legacy rejection path.
    This check neither materializes payloads nor waits for a cache.
    """
    if input.is_chunked():
        ref chunks = input._chunked.value()[]
        if chunks.cache[].state.load() == 2:
            return chunks.cache[].merged.value().isa[T]()
        for array in chunks.arrays:
            if not array.isa[T]():
                return False
        return True
    return input._data.isa[T]()
