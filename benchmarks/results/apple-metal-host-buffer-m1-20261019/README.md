# Explicit HostBuffer download diagnostic

This diagnostic compares copying one Float32 device buffer into a raw `List` pointer with copying it into a `HostBuffer`, then separately copying the HostBuffer contents into owned List storage. It uploads the same input through a raw List pointer for both paths, synchronizes each stage, alternates download order by repetition, and validates all returned Float32 bit patterns. It does not run a GPU kernel or a dataframe query.

The raw sample output contains 10/10 validated samples: five repetitions each at 4,097 rows (16,388 bytes) and 1,048,576 rows (4,194,304 bytes). For each row count, rep 0 is the first sample for that payload size and reps 1–4 are warm samples. The Metal device context was initialized before either row-count series; these are not cold-device measurements. Context initialization was reported separately as 50.928 ms.

## Results

Times are milliseconds. First-sample columns show rep 0; warm columns show the median and observed range over reps 1–4. The HostBuffer-to-owned-List copy is outside the transfer enqueue/wait timings.

| Rows | Path | First enqueue | First wait | First owned-list copy | Warm enqueue median (range) | Warm wait median (range) | Warm owned-list copy median (range) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4,097 | raw List download | 0.326 | 0.289 | — | 0.301 (0.272–0.361) | 0.239 (0.201–0.282) | — |
| 4,097 | HostBuffer download | 8.472 | 0.516 | 0.106 | 0.012 (0.010–0.014) | 0.255 (0.214–0.288) | 0.087 (0.085–0.092) |
| 1,048,576 | raw List download | 17.319 | 0.625 | — | 13.779 (13.753–13.822) | 0.477 (0.392–0.588) | — |
| 1,048,576 | HostBuffer download | 0.017 | 0.759 | 15.304 | 0.012 (0.010–0.013) | 0.673 (0.646–0.689) | 13.617 (13.453–13.753) |

For the 1,048,576-row warm samples, the median of each sample's raw-List enqueue-plus-wait time was 14.269 ms. The median of each HostBuffer enqueue-plus-wait-plus-owned-List-copy total was 14.288 ms. This small diagnostic does not demonstrate a meaningful end-to-end win: HostBuffer enqueue is much shorter, but the owned-storage copy consumes about 13.6 ms. These component timings do not establish bandwidth, a full-query speedup, or performance for other payloads or devices.

## Reproduction identity

The compiled source was `source-built.mojo`, captured from source revision `700dcf4ab1887df74130253b47450c57ee2d9259`. Its SHA-256 is `d83b2fab74cf420b03d919a2a66d1cef07bf98de3b3af099ac0946cea1c06f33`. The build source included the final top-level `main()` invocation. The tracked `benchmarks/bench_gpu_host_buffers.mojo` has that top-level invocation removed for normal source hygiene; its current SHA-256 is `65cdee019ceb060029985683b35f528053e40ed5d33ebd69fabc9dc9400c5feb`.

The HostBuffer build subcommand was:

```sh
cd /Users/randyzwitch/github/dataframe_mojo-metal-868-host-buffer-probe
/Users/randyzwitch/github/dataframe_mojo-metal-868/.pixi/envs/gpu/bin/mojo build -O3 -I . benchmarks/bench_gpu_host_buffers.mojo -o /tmp/metal-868-host-buffer-probe
```

It was invoked by `pixi run --manifest-path /Users/randyzwitch/github/dataframe_mojo-metal-868/pixi.toml --locked -e gpu bash /tmp/metal-868-sdk-diagnostics.sh`. That script also ran separate SDK probes; their output is not included or attributed to this HostBuffer result. The built binary was run with default arguments; its recorded exit status was 0.

| Item | Identity |
|---|---|
| Mojo | `1.2.0.dev2026100905` (`3b71af75`); executable SHA-256 `ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b` |
| Pixi | `0.81.0`; invocation used `--locked -e gpu` |
| `pixi.lock` SHA-256 | `098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277` |
| Built binary SHA-256 | `533a2242067abf1b7caa95b9d8d2f1da0139703b97b7d4e56375ea269dc30351` |
| Host / Metal device | Mac mini `Macmini9,1`, Apple M1, 8 CPU cores, 16 GiB RAM; Metal device reported as `Apple M1` |
| OS | macOS `26.5.2` (`25F84`), Darwin arm64 |
| Xcode / Metal SDK | Xcode `26.6` (`17F113`); macOS SDK `26.5` (`25F70`) |

The compiler emitted four warnings that the initial zero assignments for timing variables were overwritten before being read. The raw build log is retained below; compilation and the recorded run completed successfully.

## Raw artifacts

- `source-built.mojo` — exact source bytes used for the binary hash above.
- `build.log` — captured compiler output for this probe build.
- `samples.log` — captured context and all ten sample records.
- `run-status.txt` — recorded successful binary exit status (`0`).

SHA-256: `source-built.mojo` `d83b2fab74cf420b03d919a2a66d1cef07bf98de3b3af099ac0946cea1c06f33`; `build.log` `6cd9d644851523c8e8938bf5b54d1320aa1c3f2d1761268071acb46f83d7f33c`; `samples.log` `f85124fa736ba99aeb63a125b6a192661622832538feb7f989cb3d4f02afcb76`.
