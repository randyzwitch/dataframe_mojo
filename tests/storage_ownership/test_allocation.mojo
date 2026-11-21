"""Allocation advice preserves initialized bytes and caller-owned writable storage."""
# contract-group: storage.isolation
from std.testing import assert_equal
from dataframe.parallel import Job, run_jobs
from dataframe.storage.allocation import (
    advise_huge_pages,
    huge_list,
    huge_uninit,
    release_huge,
)
from tests.support.group_strategy_reference import ScopedEnvironment


def test_advised_allocations_preserve_fills_writes_and_unaligned_range_contents() raises:
    # Zero, ordinary-page and huge-page byte boundaries, independent of whether
    # the operating system elects to provide huge pages.
    for n in [0, 1, 511, 512, 513, 262143, 262144, 262145]:
        var filled = huge_list(n, UInt64(0xDA7A5500FEDCBA98))
        var written = huge_uninit[UInt64](n)
        assert_equal(len(filled), n)
        assert_equal(len(written), n)
        for row in range(n):
            assert_equal(filled[row], UInt64(0xDA7A5500FEDCBA98))
            written[row] = UInt64(row * 17 + 3)
        for row in range(n):
            assert_equal(written[row], UInt64(row * 17 + 3))
    var bytes = huge_uninit[UInt8]((4 << 20) + 8193)
    for row in range(len(bytes)):
        bytes[row] = UInt8(row % 251)
    var address = Int(bytes.unsafe_ptr())
    for offset in [0, 1, 4095, 4096, 4097]:
        for length in [0, 1, (2 << 20) - 1, 2 << 20, (2 << 20) + 1]:
            advise_huge_pages(address + offset, length)
    for row in range(len(bytes)):
        assert_equal(bytes[row], UInt8(row % 251))


struct _PoolJob(Job):
    """Takes, fills, checks and releases huge tables, many times over: on
    several workers at once they all share the process-wide pool of
    retained blocks."""

    var salt: Int
    var bad: Int

    def __init__(out self, salt: Int):
        self.salt = salt
        self.bad = 0

    def run(mut self) raises:
        for round in range(8):
            # Sizes around the retention band, so blocks are reused by
            # tables of other sizes too.
            var n = 300_000 + 37_000 * ((self.salt + round) % 5)
            var value = UInt64(self.salt * 1000 + round)
            var table = huge_list(n, value)
            for row in range(0, n, 997):
                if table[row] != value:
                    self.bad += 1
            release_huge(table^)


def test_workers_share_the_block_pool_without_losing_blocks() raises:
    # Each worker's table is its own block while it holds it: a block
    # handed to two workers at once, or a pool list changed by two at
    # once, shows up as a wrong fill or a crash.
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var jobs = List[_PoolJob]()
    for salt in range(16):
        jobs.append(_PoolJob(salt))
    run_jobs(jobs)
    for k in range(len(jobs)):
        assert_equal(jobs[k].bad, 0)
    _ = threads^
