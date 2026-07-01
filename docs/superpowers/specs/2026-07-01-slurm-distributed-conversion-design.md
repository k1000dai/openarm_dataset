# SLURM distributed OpenArm → LeRobot conversion (Phase B)

Date: 2026-07-01
Status: Approved (design)

Builds on [Phase A](2026-07-01-parallelize-lerobot-conversion-design.md)
(in-process parallelism). This phase distributes a conversion across many SLURM
array tasks (across nodes) and aggregates the results into one dataset.

## Goal

Convert a large OpenArm dataset (e.g. 786 episodes, multi-TB) to LeRobot
**v3.0, v2.1, and gr00t** by:

1. Splitting episodes into `N` shards, each converted by an independent SLURM
   array task (reusing Phase-A in-process parallelism within each task).
2. Aggregating the shards into one final dataset with a single job — moving and
   renumbering files, **not** re-encoding video.

Scope: v3.0, v2.1, gr00t. Not in scope: rrd; changing numeric/stat formulas;
chunked `meta/episodes` output (single episodes file, as today).

## Core idea

Each shard converts a slice of episodes into a **self-contained partial
dataset**, reusing the Phase-A writers unchanged by injecting **global**
episode/task indices but **local** row-index and file numbering. A single
`aggregate` step stitches shards by moving/renumbering files and fixing up the
global row index.

**Key invariant:** global `episode_index` and `task_index` are assigned at shard
time (both are derivable from the full metadata, so every array task agrees
without coordination). Only the row `index` (global frame counter) and the file
numbering — which genuinely depend on other shards' sizes — are deferred to
aggregate.

## Components

### 1. Refactor: extract post-remap writer bodies

Extract the body of `to_lerobotv21` / `to_lerobotv30` that runs *after* remaps
are built into `_write_v21(dataset, records, episode_image_stats,
remap_episode_index, remap_task_index, output_dir, fps, train_split,
joint_names)` and the v3.0 equivalent. The normal (non-distributed) path builds
local identity-ish remaps via `_build_remaps` and calls these — behavior
unchanged. The shard path injects **global** remaps. This keeps one
orchestration implementation per format.

### 2. `distributed.py`

- `convert_shard(input, shards_dir, format, num_shards, shard_index, *, fps,
  smoothing_cutoff, train_split, success_only, jobs)`:
  1. Build the **filtered** episode list from the full metadata (respecting
     `success_only`). Deterministic → every array task computes the same list.
  2. Assign each filtered episode its **global** `episode_index` (its position
     in the filtered list). Compute a **global** task remap over the full
     filtered list.
  3. Slice the filtered list into `num_shards` contiguous shards
     (`numpy.array_split`-style, near-equal sizes); take this `shard_index`.
  4. Convert only this shard's episodes through `_write_v21` / `_write_v30`,
     passing the global remaps. Writers emit correct global
     `episode_index`/`task_index`, with **local** row `index` (0-based) and
     **local** file numbering (`file-000…`). gr00t additionally writes
     `modality.json`.
  5. Output → `shards_dir/shard-{shard_index:05d}/` (a valid partial dataset).

- `aggregate_shards(shards_dir, output, format)`:
  Read each shard's `info.json` (for `total_frames`) and episodes metadata, in
  shard-index order. Compute per-shard `row_offset` (cumulative frames) and
  running global data/video file counters. Then per shard, in order:
  - **v3.0:** move each `data/chunk/file.parquet` to the next global
    `data/chunk/file`, rewriting the `index` column (`+= row_offset`). Move each
    `videos/<cam>/chunk/file.mp4` to its global renumber (pure file move). Fix
    the shard's episodes rows: `dataset_from_index`/`dataset_to_index +=
    row_offset`; `data/chunk_index`,`data/file_index` and
    `videos/<cam>/chunk_index`,`file_index` → global values via the move
    mapping. `episode_index`, `from_timestamp`, `to_timestamp` already correct.
  - **v2.1 / gr00t:** files are already globally named (global `episode_index`),
    so move them into the combined tree; rewrite each data parquet `index +=
    row_offset`; concatenate `episodes.jsonl`, and concatenate
    `episodes_stats.jsonl` after adding `row_offset` to each record's
    `dataset_from_index` / `dataset_to_index`.
  - Concatenate all episodes rows → final `meta/episodes/chunk-000/file-000.parquet`
    (v3.0) or `episodes.jsonl` (v2.1).
  - **stats.json:** v3.0 reconstructs per-episode stats from the
    episodes-parquet `stats/*` columns and folds them with the existing
    `_aggregate_stats` → identical to a single-process v3.0 run. v2.1 re-reads
    the (small, numeric-only) data parquet and vstacks → matches single-process
    v2.1 exactly.
  - `tasks` copied from any shard (identical global remap). `info.json`
    recomputed with global totals (`total_episodes`, `total_frames`,
    `total_tasks`, splits, chunk counts); `features` copied from a shard.
    gr00t: copy `modality.json` from a shard.

Move uses `os.replace`/`shutil.move` when shards and output share a filesystem
(fast, no copy); falls back to copy across filesystems.

### 3. CLIs

- `openarm-dataset-convert … --num-shards N --shard-index i` → `convert_shard`
  (writes to `OUTPUT/shard-{i:05d}/`; without `--num-shards` behavior is
  unchanged). `--num-shards`/`--shard-index` must be given together and valid
  (`0 <= i < N`).
- `openarm-dataset-aggregate SHARDS_DIR OUTPUT --format {lerobot_v2.1,
  lerobot_v3.0,gr00t}` → `aggregate_shards`.
- `openarm-dataset-slurm-submit INPUT OUTPUT --format … --num-shards N
  --partition … --cpus-per-task … [--mem …] [--time …] [--dry-run]`: generates
  an sbatch **array** job (`--array=0-(N-1)`) running the shard convert, plus a
  dependent (`--dependency=afterok:<arrayjobid>`) aggregate job. Always writes
  both scripts to `OUTPUT/_slurm/` for inspection; submits via `sbatch` by
  default; `--dry-run` writes without submitting. The library itself never
  imports or requires SLURM.

## Directory conventions

```
SHARDS_DIR/
  shard-00000/   # a valid partial v3.0/v2.1 dataset
  shard-00001/
  ...
OUTPUT/          # final aggregated dataset
OUTPUT/_slurm/   # generated sbatch scripts (slurm-submit only)
```

## Error handling

- `convert_shard`: validate `0 <= shard_index < num_shards`; a shard with zero
  episodes (more shards than episodes) writes a valid empty partial dataset that
  aggregate skips.
- `aggregate_shards`: validate every expected `shard-*` exists and all share the
  same `codebase_version`, `robot_type`, `features`, and camera set; fail fast
  with the offending shard named. Refuse if `OUTPUT` exists.
- `slurm-submit`: fail clearly if `sbatch` is absent (unless `--dry-run`).

## Testing (no cluster required)

- **Equivalence:** run `convert_shard` for `num_shards=3` locally (subprocesses),
  then `aggregate_shards`, and assert the result is **equivalent to a
  single-process conversion** for each format:
  - per-episode data equal when joined on `episode_index` (v3.0 and v2.1),
  - `stats.json` numerically identical (v3.0) / equal (v2.1),
  - `info.json` totals match (`total_episodes`, `total_frames`, `total_tasks`),
  - every `data`/`videos` pointer in the episodes metadata resolves to a real
    file; `dataset_from/to_index` contiguous and covering `[0, total_frames)`,
  - `from_timestamp`/`to_timestamp` land on the frame grid.
  Video file *layout* differs (sharded packing) and is not compared byte-wise.
- **Edge:** `num_shards > num_episodes`; `success_only=True`; single shard
  (`num_shards=1`) equals the non-distributed output except file layout.
- **slurm-submit `--dry-run`:** asserts the generated sbatch scripts contain the
  expected `--array`, `--dependency=afterok`, and command lines.
- `pre-commit` / ruff.

## Non-goals

- No re-encoding of video during aggregation.
- No exact file-size targets at shard boundaries (sizes remain approximate, as
  accepted in Phase A).
- No datatrove dependency; no SLURM import in the library.
