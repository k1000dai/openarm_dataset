# Parallelize OpenArm → LeRobot conversion (v2.1 + v3.0)

Date: 2026-07-01
Status: Approved (design)

## Problem

Converting an OpenArm dataset to LeRobot v2.1 / v3.0 (and GR00T) is fully
serial and uses ~1 of the machine's cores. On a 256-core SLURM node, a
6-episode subset baseline shows:

| Stage | v2.1 | v3.0 |
|---|---|---|
| load + downsample (`_collect_downsampled_data`) | 11.1 s (NFS, cold) | ~1.6 s (warm) |
| write parquet | 0.02 s | 0.04 s |
| **video encode (ffmpeg)** | **85.6 s** | **21.3 s** |
| **stats (JPEG decode)** | **16.6 s** | **16.6 s** |
| **TOTAL** | **113 s** | **40 s** |

Extrapolated: v2.1 ≈ 31 min / 100 episodes, ≈ 4 h / 786 episodes. Everything
is embarrassingly parallel across episodes.

Dataset shape: 5 cameras (1280×720 JPEG) on an NFS mount, obs/action at 250 Hz,
target output fps 30 (~200 frames/episode). Full dataset = 786 episodes.

## Goal

Speed up both `to_lerobotv21` and `to_lerobotv30` (and `to_gr00t`) on a single
multi-core node using in-process parallelism, while keeping output equivalent
to the current serial code. Design so a future SLURM shard→aggregate layer can
be added without reworking the per-episode logic. The SLURM layer itself is
**out of scope** for this change.

## Guiding principle

**Parallelize the map, keep assembly serial and ordered.** Per-episode compute
runs in a worker pool; results are assembled in episode order. The data
parquet, per-episode stat values, stats reduction, and metadata are therefore
identical to today. Video packing has one intentional change (below).

## Design

### 1. `parallel.py` helper (new module)

- `parallel_map(func, items, jobs, initializer=None, initargs=())`:
  returns results **in input order**. Backed by
  `concurrent.futures.ProcessPoolExecutor`. When `jobs <= 1`, runs serially
  in-process (identical behavior, easy debugging, no pickling).
- `resolve_jobs(jobs)`: `None`/`0` → `os.cpu_count()`; otherwise the given int.
- Stdlib only; no new dependencies.

### 2. Parallelize the three hot stages

**a. Load + downsample.** `_collect_downsampled_data` maps over episodes with a
process pool. A pool `initializer` builds one `Dataset` (with smoothing set) per
worker process from the dataset root + camera names + cutoff, so metadata is not
re-pickled per call. Each worker returns the same
`(episode_index, num_frames, sampled_obs, sampled_actions, sampled_cameras)`
record tuple the serial code produces. Records are collected in order.

**b. Image-stats decode.** The expensive part of stats is `_describe_images`
decoding sampled JPEGs per `(episode, camera)`. Run these decodes as a parallel
pre-pass keyed by `(episode_index, camera)`; feed the resulting per-camera image
stat dicts into `_calc_episode_stats` (v2.1) / `_calc_episode_stats_numpy`
(v3.0), which are refactored to accept precomputed image stats instead of frame
lists. Numeric stat values are unchanged.

To avoid a second NFS/JPEG pass, the per-episode worker in (a) computes its own
image stats (calling `_describe_images` on its sampled frames) and returns them
alongside the record. Decode thus happens exactly once, in parallel, in the same
worker that already loaded the episode. The full camera frame lists are still
returned for the video stage (they are cheap `Frame` refs, not decoded images).

**c. Video encode (ffmpeg).** Independent encodes run in parallel:
- v2.1: one job per `(episode, camera)` (`_write_videos`).
- v3.0: one job per packed output `file-XXX.mp4` (`_write_packed_videos`).

`encode_mp4` gains an optional `threads` argument mapped to ffmpeg `-threads`.
The video stage sets `ffmpeg_threads ≈ max(1, cpu_count / active_jobs)` where
`active_jobs = min(jobs, num_encodes)`, filling cores without oversubscribing
(few large v3.0 encodes each get more threads; many small v2.1 encodes each get
fewer).

### 3. Intentional v3.0 video-packing change

Today `_write_packed_videos` packs via an **online** compression-ratio feedback
loop (each written file refines the ratio for the next), which is inherently
serial. To parallelize file encodes, precompute all file assignments up front
from a **fixed** compression ratio calibrated once per camera (single sample
encode, as today). Consequences:

- File boundaries may differ slightly from the current serial layout, and file
  sizes only approximately track `VIDEO_FILES_SIZE_IN_MB` (already documented as
  approximate; user-accepted).
- Output remains a valid v3.0 dataset and still satisfies the existing v3.0
  tests: single-pass encode per file (uniform PTS), exact frame-grid
  timestamps, correct per-episode frame counts, and correct
  `from_timestamp`/`to_timestamp`.

The **data-parquet** packing (`_write_packed_parquet`) stays byte-identical: it
uses deterministic per-episode measured sizes in order. That stage is already
<0.1 s and remains serial.

### 4. Plumb `jobs` through the API

Add a `jobs: int | None = None` parameter to `to_lerobotv21`, `to_lerobotv30`,
`to_gr00t`, forwarded through `Dataset.write(...)`. Add `--jobs` to the
`openarm-dataset-convert` CLI (default = all cores; `1` = serial).

### 5. Keep the SLURM layer ready (not built now)

Per-episode workers are pure `process_episode(index) → record/stats` units and
assembly is separable. A later shard→aggregate step can reuse the workers and
the existing `_aggregate_stats` / `_aggregate_feature_stats` (already written to
fold per-episode stats into global stats). No code for this is added here, but
the refactor must not couple worker logic to the full-dataset assembly.

## Components / boundaries

- `parallel.py` — generic ordered parallel map + job resolution. No project
  knowledge. Testable in isolation.
- `lerobot_v21.py` — per-episode worker(s) + serial assembly; uses `parallel_map`.
- `lerobot_v30.py` — per-file video packing (precomputed assignments) + serial
  assembly; uses `parallel_map`; reuses v21 workers where shared.
- `ffmpeg.py` — `encode_mp4(..., threads=None)`.
- `convert.py` — `--jobs` CLI flag.

## Error handling

- A worker exception propagates and aborts the conversion with the episode
  index in the message (fail fast; a partial LeRobot dataset is not useful).
- `jobs <= 1` bypasses the pool entirely (no multiprocessing overhead, direct
  tracebacks).
- ffmpeg failures already raise via `subprocess.run(check=True)`; that behavior
  is preserved, surfaced per encode job.

## Testing / verification

- **New**: a regression test that runs the fixture conversion with `jobs=1` and
  `jobs=4` and asserts equivalence — identical `data` parquet bytes/values and
  identical `stats.json` numeric values; videos present and valid. This test
  does not require lerobot (operates on the emitted files) so it runs in CI.
- **Existing**: run `tests/test_lerobot_v21.py` and `tests/test_lerobot_v30.py`
  (install the `lerobot-dataset-v2.1` / `lerobot-dataset-v3.0` extras locally to
  exercise the `importorskip`-gated tests).
- **Benchmark**: convert the 100-episode sample end-to-end, record wall-clock
  vs. the serial baseline.
- `pre-commit run --all-files` for format/lint.

## Non-goals

- SLURM / multi-node distribution (kept ready, not implemented).
- Changing the numeric downsampling, smoothing, or stat formulas.
- Changing the openarm→openarm (`_write`) path or other formats (rrd).
- Reducing peak memory (records still held in memory as today; the 2 TB node
  and future sharding cover this).
