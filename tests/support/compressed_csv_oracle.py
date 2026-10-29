"""Independent gzip/zstd fixtures and Polars checks for the Mojo CSV reader.

Regenerate checked-in fixtures with the oracle environment and independently
verify their values and corruption behavior. Mojo contracts read these files
without requiring Python at runtime.
"""
import argparse
import gzip
import io
from pathlib import Path

import polars as pl
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2] / "tests/fixtures/compressed_csv"
CORE = 'id,text\n1,"α\nβ"\n2,"a ""quote"""\n3,last'.encode()
OPTIONS = '\ufeff\n# before\nskip;row\nid;text\n1;\'α,\nβ\'\n# between\n2;NA\n3;\'\'\n'.encode()


def compress(data, codec):
    return gzip.compress(data, mtime=0) if codec == "gzip" else pa.compress(data, codec="zstd").to_pybytes()


def generate(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    (root / "core.csv").write_bytes(CORE)
    expected = {"id": [1, 2, 3], "text": ["α\nβ", 'a "quote"', "last"]}
    for codec, suffix, wrong in [("gzip", "gz", "zst"), ("zstd", "zst", "gz")]:
        payload = compress(CORE, codec)
        for name in [f"core.csv.{suffix}", f"magic-{codec}.bin", f"conflict-{codec}.csv.{wrong}"]:
            (root / name).write_bytes(payload)
            for reader in [pl.read_csv, pl.scan_csv]:
                result = reader(io.BytesIO(payload))
                if isinstance(result, pl.LazyFrame):
                    result = result.collect()
                assert result.to_dict(as_series=False) == expected
        # The frame split is inside a quoted, multiline CSV field.
        split = CORE.index(b'\xce\xb1') + 2
        (root / f"concat-{codec}.csv.{suffix}").write_bytes(
            compress(CORE[:split], codec) + compress(CORE[split:], codec)
        )
        (root / f"options.csv.{suffix}").write_bytes(compress(OPTIONS, codec))
        (root / f"infer.csv.{suffix}").write_bytes(compress(b"x\n1\n2\nlate\n", codec))
        (root / f"header-only.csv.{suffix}").write_bytes(compress(b"id,text\n", codec))
        (root / f"empty.csv.{suffix}").write_bytes(compress(b"", codec))
        (root / f"disabled-quote.csv.{suffix}").write_bytes(compress(b'x|s\n1|a"b\n2|c\n', codec))
        (root / f"no-header.csv.{suffix}").write_bytes(compress(b"skip\n1|1.5|keep\n2|NA|other\n3|3.5|last\n", codec))
        # Quotes, an escaped quote, UTF-8 and a record LF span read boundaries.
        long_text = "x" * 65523 + '"α\n' + "y" * 65536
        long_csv = ('id,text\n1,"' + long_text.replace('"', '""') + '"\n2,tail\n').encode()
        (root / f"boundary.csv.{suffix}").write_bytes(compress(long_csv, codec))
        late = compress(b"x\n" + b"1\n" * 2000000, codec)
        corruptions = {
            "truncated": payload[:-3],
            "junk": payload + b"junk",
            "late-truncated": late[:-3],
            "invalid": b"not a compressed stream",
        }
        if codec == "gzip":
            damaged = bytearray(payload)
            damaged[-8] ^= 1
            corruptions["checksum"] = bytes(damaged)
        for name, data in corruptions.items():
            (root / f"{name}.csv.{suffix}").write_bytes(data)
            try:
                pl.read_csv(io.BytesIO(data))
            except Exception:
                pass
            else:
                # Extension is the only signal for the invalid-data fixture.
                assert name == "invalid", (codec, name)
        # Large unread tails remain unread under finite-inference row limits.
        assert pl.read_csv(io.BytesIO(late[:-3]), n_rows=1).to_dict(as_series=False) == {"x": [1]}
        assert pl.scan_csv(io.BytesIO(late[:-3])).head(1).collect().to_dict(as_series=False) == {"x": [1]}
        assert pl.read_csv(io.BytesIO(compress(OPTIONS, codec)), separator=";", quote_char="'", comment_prefix="#", skip_rows=1, null_values=["NA"]).to_dict(as_series=False) == {
            "id": [1, 2, 3], "text": ["α,\nβ", None, ""]
        }
        assert pl.read_csv(io.BytesIO(compress(long_csv, codec))).item(0, "text") == long_text
    print(f"CSV compression fixtures and Polars {pl.__version__} checks passed: {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT)
    generate(parser.parse_args().output)
