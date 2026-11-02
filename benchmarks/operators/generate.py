"""Synthetic Parquet inputs for the operator parity harness.

Every dataset is a pure function of its spec (matrices.py) and a seed
derived from it, so a cell always sees the same rows. Datasets are written
once under `<data root>/operators/<kind>-<hash>/` (the data root is
benchmarks/suites/datagen.py's) and reused.

Key values come from integer ids through bijections (an xor-shift-multiply
mix, or an odd-multiplier affine map for dates and decimals), so distinct
ids give distinct keys while the values themselves look random: no
progressions, no dense ranges unless the cell asks for dense keys.
Strings are fixed-width base-36 encodings of mixed ids followed by a
padding of id-dependent length.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

U64 = np.uint64
ALPHABET = np.frombuffer(b"0123456789abcdefghijklmnopqrstuvwxyz", np.uint8)
PAD = np.frombuffer(b"qwertyuiopasdfghjklzxcvbnm", np.uint8)


def seed_of(spec, salt=0):
    text = json.dumps(spec, sort_keys=True) + f"/{salt}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def mix64(ids, salt):
    """splitmix64's finalizer after adding a salt: a bijection on uint64."""
    with np.errstate(over="ignore"):
        z = ids.astype(U64) + U64(salt & 0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
        return z ^ (z >> U64(31))


def mix32(ids, salt):
    """murmur3's fmix32 after adding a salt: a bijection on uint32 (ids
    must be below 2**32)."""
    with np.errstate(over="ignore"):
        h = ids.astype(np.uint32) + np.uint32(salt & 0xFFFFFFFF)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x85EBCA6B)
        h ^= h >> np.uint32(13)
        h *= np.uint32(0xC2B2AE35)
        h ^= h >> np.uint32(16)
        return h


def affine(ids, bits, salt):
    """(a * id + b) mod 2**bits with odd a: a bijection on [0, 2**bits)."""
    mask = U64((1 << bits) - 1)
    a = U64((salt | 1) & 0xFFFFFFFF)
    b = U64((salt >> 32) & 0xFFFFFFFF)
    with np.errstate(over="ignore"):
        return (ids.astype(U64) * a + b) & mask


def _bits(count):
    return max(1, math.ceil(math.log2(max(2, count))))


def strings(values, width, pad_min, pad_max, prefix=b""):
    """Unique strings for unique uint64 values: prefix, `width` base-36
    digits of the value, then pad_min..pad_max padding letters chosen by
    the value."""
    values = values.astype(U64)
    n = len(values)
    pads = pad_min + (
        mix64(values, 7) % U64(pad_max - pad_min + 1)
    ).astype(np.int64)
    total = len(prefix) + width + pad_max
    mat = np.empty((n, total), np.uint8)
    for i, byte in enumerate(prefix):
        mat[:, i] = byte
    v = values.copy()
    for j in range(width):
        mat[:, len(prefix) + width - 1 - j] = ALPHABET[
            (v % U64(36)).astype(np.int64)
        ]
        v //= U64(36)
    letters = mix64(values, 11)
    for j in range(pad_max):
        mat[:, len(prefix) + width + j] = PAD[
            ((letters >> U64(5 * (j % 12))) % U64(26)).astype(np.int64)
        ]
    lengths = len(prefix) + width + pads
    keep = np.arange(total)[None, :] < lengths[:, None]
    data = mat[keep]
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return pa.LargeStringArray.from_buffers(
        n, pa.py_buffer(offsets), pa.py_buffer(data)
    ).cast(pa.string())


def str_short(ids, salt):
    """4..12 bytes."""
    return strings(mix32(ids, salt).astype(U64), 7, 0, 5)


def str_long(ids, salt):
    """20..40 bytes."""
    return strings(mix64(ids, salt), 13, 2, 22, b"item-")


def str25(ids, salt):
    """Exactly 25 bytes."""
    return strings(mix64(ids, salt), 13, 12, 12)


def decimal_array(unscaled, precision=18, scale=2, valid=None):
    unscaled = unscaled.astype(np.int64)
    words = np.empty((len(unscaled), 2), np.int64)
    words[:, 0] = unscaled
    words[:, 1] = np.where(unscaled < 0, -1, 0)
    array = pa.Array.from_buffers(
        pa.decimal128(precision, scale),
        len(unscaled),
        [None, pa.py_buffer(words)],
    )
    return _with_nulls(array, valid)


def _with_nulls(array, valid):
    if valid is None or valid.all():
        return array
    return pa.compute.if_else(
        pa.array(valid), array, pa.nulls(len(array), array.type)
    )


def key_values(kind, ids, salt, valid=None):
    """A key column (or columns, for two-part keys) for ids; distinct ids
    give distinct keys."""
    ids = ids.astype(U64)
    if kind == "int32":
        return [_with_nulls(pa.array(mix32(ids, salt).view(np.int32)), valid)]
    if kind == "int64":
        return [_with_nulls(pa.array(mix64(ids, salt).view(np.int64)), valid)]
    if kind == "date":
        # Up to 2**21 distinct dates stay within years 1..9999 (centred
        # near year 4700); more spread further.
        bits = max(21, _bits(int(ids.max()) + 1 if len(ids) else 1))
        days = (
            affine(ids, bits, salt).astype(np.int64)
            - (1 << (bits - 1))
            + 1_000_000
        )
        return [_with_nulls(pa.array(days.astype(np.int32)).cast(pa.date32()), valid)]
    if kind == "decimal":
        unscaled = affine(ids, 40, salt).astype(np.int64) - (1 << 39)
        return [decimal_array(unscaled, valid=valid)]
    if kind == "str_short":
        return [_with_nulls(str_short(ids, salt), valid)]
    if kind == "str_long":
        return [_with_nulls(str_long(ids, salt), valid)]
    if kind == "int_int":
        a = mix32(ids & U64(0xFFF), salt).view(np.int32)
        b = mix32(ids >> U64(12), salt + 1).view(np.int32)
        return [_with_nulls(pa.array(a), valid), pa.array(b)]
    if kind == "int_str":
        a = mix32(ids & U64(0xFF), salt).view(np.int32)
        return [_with_nulls(pa.array(a), valid), str_short(ids >> U64(8), salt + 1)]
    raise ValueError(f"unknown key kind {kind}")


def payload(kind, rng, n, salt):
    if kind == "int64":
        return pa.array(rng.integers(-(10**9), 10**9, n))
    if kind == "int32":
        return pa.array(rng.integers(-(10**6), 10**6, n).astype(np.int32))
    if kind == "float64":
        return pa.array(rng.standard_normal(n) * 1000.0)
    if kind == "decimal":
        return decimal_array(rng.integers(-(10**9), 10**9, n))
    if kind == "str_short":
        return str_short(rng.integers(0, 1 << 32, n, dtype=np.uint64), salt)
    raise ValueError(kind)


def _valid(rng, n, share):
    if share <= 0:
        return None
    return rng.random(n) >= share


# --- datasets ---------------------------------------------------------------


def join_dataset(spec, seed):
    from matrices import PAYLOAD_TYPES

    rng = np.random.default_rng(seed)
    build_rows, probe_rows = spec["build_rows"], spec["probe_rows"]
    if spec["dups"] == "1":
        counts = np.ones(build_rows, np.int64)
    elif spec["dups"] == "4":
        counts = np.full(-(-build_rows // 4), 4, np.int64)
    else:
        # Zipf(2) rows per key, at most 64 (mean about 3.5).
        parts, total = [], 0
        while total < build_rows:
            draw = np.minimum(rng.zipf(2.0, build_rows), 64)
            parts.append(draw)
            total += int(draw.sum())
        counts = np.concatenate(parts)
        counts = counts[: np.searchsorted(np.cumsum(counts), build_rows) + 1]
    distinct = len(counts)
    build_ids = np.repeat(np.arange(distinct, dtype=np.int64), counts)[
        :build_rows
    ]
    matched = rng.random(probe_rows) < spec["match"]
    probe_ids = np.where(
        matched,
        rng.integers(0, distinct, probe_rows),
        distinct + rng.integers(0, max(distinct, probe_rows), probe_rows),
    )
    salt = seed & 0xFFFFFFFF
    kind = spec["key"]
    tables = {}
    for side, ids, names, prefix in (
        ("probe", probe_ids, "k", "p"),
        ("build", build_ids, "bk", "b"),
    ):
        n = len(ids)
        columns = key_values(kind, ids, salt, _valid(rng, n, spec["nulls"]))
        fields = {f"{names}{i}": c for i, c in enumerate(columns)}
        for i in range(spec["payload"]):
            fields[f"{prefix}{i}"] = payload(PAYLOAD_TYPES[i], rng, n, i)
        table = pa.table(fields)
        keys = [f"{names}{i}" for i in range(len(columns))]
        if spec["order"] == "sorted":
            table = table.sort_by([(k, "ascending") for k in keys])
        else:
            table = table.take(pa.array(rng.permutation(n)))
        tables[side] = table
    return tables


def agg_dataset(spec, seed):
    rng = np.random.default_rng(seed)
    n, groups = spec["rows"], spec["groups"]
    perm = rng.permutation(groups)
    if spec["skew"] == "uniform":
        ranks = rng.integers(0, groups, n)
    elif spec["skew"] == "one50":
        ranks = np.where(rng.random(n) < 0.5, 0, rng.integers(0, groups, n))
    else:
        weights = np.arange(1, groups + 1, dtype=np.float64) ** -1.1
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        ranks = np.minimum(np.searchsorted(cdf, rng.random(n)), groups - 1)
    gid = perm[ranks].astype(U64)
    salt = seed & 0xFFFFFFFF
    nulls = spec["nulls"]
    fields = {}
    kind = spec["key"]
    if kind == "int32_dense":
        dense = rng.permutation(groups).astype(np.int32)
        fields["k_i32"] = pa.array(dense[gid.astype(np.int64)])
    elif kind == "int64_sparse":
        fields["k_i64"] = key_values("int64", gid, salt)[0]
    elif kind == "decimal":
        fields["k_dec"] = key_values("decimal", gid, salt)[0]
    elif kind == "str_short":
        fields["k_ss"] = str_short(gid, salt)
    elif kind == "str_long":
        fields["k_sl"] = str_long(gid, salt)
    elif kind == "int_str":
        fields["k_a"] = pa.array(mix32(gid % U64(1000), salt).view(np.int32))
        fields["k_s"] = str_short(gid // U64(1000), salt + 1)
    else:
        fields["m_i"] = pa.array(mix32(gid % U64(16), salt).view(np.int32))
        fields["m_s"] = str_short((gid // U64(16)) % U64(64), salt + 1)
        fields["m_d"] = key_values("date", (gid // U64(1024)) % U64(32), salt + 2)[0]
        fields["m_dec"] = key_values("decimal", gid // U64(32768), salt + 3)[0]
    any_null = np.zeros(n, bool)
    if nulls > 0:
        for name in list(fields):
            valid = rng.random(n) >= nulls
            any_null |= ~valid
            fields[name] = _with_nulls(fields[name], valid)
    fields["v_i"] = _with_nulls(
        pa.array(rng.integers(-(10**6), 10**6, n)), _valid(rng, n, nulls)
    )
    fields["v_f"] = _with_nulls(
        pa.array(rng.standard_normal(n) * 1000.0), _valid(rng, n, nulls)
    )
    fields["v_d"] = decimal_array(
        rng.integers(-(10**9), 10**9, n), valid=_valid(rng, n, nulls)
    )
    fields["v_s"] = _with_nulls(
        str_short(rng.integers(0, 1 << 32, n, dtype=np.uint64), salt + 4),
        _valid(rng, n, nulls),
    )
    # Constant within each group, so first and last are determined.
    fd = mix64(gid, salt + 5).view(np.int64)
    fields["fd"] = pa.array(np.where(any_null, 0, fd))
    fields["u_i"] = pa.array(
        mix64(rng.integers(0, 1000, n).astype(U64), salt + 6).view(np.int64)
    )
    fields["u_s"] = str_short(rng.integers(0, 1000, n).astype(U64), salt + 7)
    return {"t": pa.table(fields)}


def filter_dataset(spec, seed):
    rng = np.random.default_rng(seed)
    n = spec["rows"]
    fields = {"f": pa.array(rng.integers(0, 1000, n))}
    for name in ("a", "b"):
        if spec["type"] == "int":
            fields[name] = pa.array(rng.integers(-1000, 1001, n))
        elif spec["type"] == "float":
            fields[name] = pa.array(rng.random(n) * 2000.0 - 1000.0)
        else:
            fields[name] = decimal_array(rng.integers(-(10**6), 10**6 + 1, n))
    return {"t": pa.table(fields)}


def sort_dataset(spec, seed):
    rng = np.random.default_rng(seed)
    n = spec["rows"]
    salt = seed & 0xFFFFFFFF

    def draw(high):
        return rng.integers(0, high, n).astype(U64)

    fields = {
        "s_i": pa.array(mix64(draw(n), salt).view(np.int64)),
        "s_dec": key_values("decimal", draw(n), salt + 1)[0],
        "s_s25": str25(draw(n), salt + 2),
        "s_a": pa.array(mix32(draw(100), salt + 3).view(np.int32)),
        "s_b": str25(draw(max(1, n // 10)), salt + 4),
        "s_c": key_values("decimal", draw(n), salt + 5)[0],
        "p_i": pa.array(rng.permutation(n).astype(np.int64)),
        "p_s": str_short(draw(1 << 32), salt + 6),
    }
    return {"t": pa.table(fields)}


def window_dataset(spec, seed):
    rng = np.random.default_rng(seed)
    n, groups = spec["rows"], spec["groups"]
    salt = seed & 0xFFFFFFFF
    gid = rng.integers(0, groups, n).astype(U64)
    perm = rng.permutation(n).astype(U64)
    half = perm // U64(2)
    fields = {
        "g": pa.array(mix64(gid, salt).view(np.int64)),
        "o_i": pa.array(mix64(perm, salt + 1).view(np.int64)),
        "t_i": pa.array(mix64(half, salt + 1).view(np.int64)),
        "o_s": str_short(perm, salt + 2),
        "t_s": str_short(half, salt + 2),
        "o_a": pa.array(mix32(perm % U64(1000), salt + 3).view(np.int32)),
        "o_b": str_short(perm // U64(1000), salt + 4),
        "t_a": pa.array(mix32(half % U64(1000), salt + 3).view(np.int32)),
        "t_b": str_short(half // U64(1000), salt + 4),
        "v": pa.array(rng.integers(-1000, 1001, n)),
    }
    return {"t": pa.table(fields)}


def sched_dataset(spec, seed):
    rng = np.random.default_rng(seed)
    probe, build = spec["probe_rows"], spec["build_rows"]
    salt = seed & 0xFFFFFFFF
    fields = {}
    tables = {}
    for j in (1, 2, 3):
        ids = rng.permutation(build).astype(U64)
        tables[f"b{j}"] = pa.table(
            {
                f"bk{j}": pa.array(mix64(ids, salt + j).view(np.int64)),
                f"x{j}": pa.array(rng.integers(-(10**9), 10**9, build)),
            }
        )
        fields[f"k{j}"] = pa.array(
            mix64(rng.integers(0, build, probe).astype(U64), salt + j).view(
                np.int64
            )
        )
    fields["p"] = pa.array(rng.integers(-(10**9), 10**9, probe))
    tables["t"] = pa.table(fields)
    return tables


DATASETS = {
    "join": join_dataset,
    "agg": agg_dataset,
    "filter": filter_dataset,
    "sort": sort_dataset,
    "window": window_dataset,
    "sched": sched_dataset,
}


def build_tables(spec):
    """The dataset's tables in memory (tests use this at tiny sizes)."""
    return DATASETS[spec["kind"]](spec, seed_of(spec))


def dataset_dir(spec, root):
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()[:16]
    return Path(root) / "operators" / f"{spec['kind']}-{digest}"


def generate(spec, root):
    """Write the dataset once; returns {table name: Parquet path}."""
    directory = dataset_dir(spec, root)
    done = directory / "spec.json"
    if not done.exists():
        directory.mkdir(parents=True, exist_ok=True)
        for name, table in build_tables(spec).items():
            pq.write_table(
                table,
                directory / f"{name}.parquet",
                store_decimal_as_integer=True,
            )
        done.write_text(json.dumps(spec, sort_keys=True))
    names = [p.stem for p in sorted(directory.glob("*.parquet"))]
    return {name: directory / f"{name}.parquet" for name in names}
