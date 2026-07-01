# Remove SLURM distributed sharding; keep single-node parallelism

Date: 2026-07-01
Status: Approved (design)

Reverses **Phase B** — SLURM distributed sharded conversion + aggregation; its
design spec (`2026-07-01-slurm-distributed-conversion-design.md`) is removed in
this change — while keeping
[Phase A](2026-07-01-parallelize-lerobot-conversion-design.md) (in-process,
single-node parallelism). Goal: a clean, PR-to-`main` branch whose diff is only
the single-node parallelism feature.

## Rationale — sharding vs. single-node parallelism

| Aspect | Single-node parallel (Phase A) | SLURM sharding (Phase B) |
|--------|--------------------------------|--------------------------|
| Mechanism | One node, all cores; per-episode parallelism (processes for load/downsample, threads for ffmpeg) | Split episodes into N shards → SLURM array job → `aggregate_shards` stitches |
| Extra cost | None | A mandatory aggregate pass that moves/renumbers every parquet+video file and rewrites the global row `index` |
| Code size | `parallel.py` (~140 lines) | `distributed.py` (659) + `aggregate.py` + `slurm_submit.py` + generalized shared writers (~950+ lines) |
| Wins when | Always (saturates one node's cores) | Only when one node's cores are the bottleneck *and* multiple nodes can be thrown at it |
| External deps | None | A SLURM cluster / `sbatch` |

For this workload video encoding (`ffmpeg`) is the wall and already saturates a
single node's cores, so sharding's aggregation I/O and cluster dependency add
complexity without a real speedup. Single-node parallelism is simpler, has no
external dependencies, and is effectively faster here. Keep only Phase A.

Sharding work is preserved on the `sharding` branch (and `origin/sharding`), so
removing it here loses nothing.

## Scope

### Delete
- `src/openarm_dataset/distributed.py` — `convert_shard` / `aggregate_shards`
- `src/openarm_dataset/aggregate.py` — `openarm-dataset-aggregate` CLI
- `src/openarm_dataset/slurm_submit.py` — `openarm-dataset-slurm-submit` CLI
- `tests/test_distributed.py`
- `docs/superpowers/specs/2026-07-01-slurm-distributed-conversion-design.md`
- `pyproject.toml`: the `openarm-dataset-aggregate` and
  `openarm-dataset-slurm-submit` `[project.scripts]` entries
- `convert.py`: the `--num-shards` / `--shard-index` arguments and the shard
  branch that calls `convert_shard`

### Revert Phase-B generalizations in shared writers (back to Phase-A form)
- `lerobot_v21.py`:
  - `_collect_downsampled_data`: drop the `episode_indices` parameter (only the
    shard path passed it); restore the plain `success_only` filter.
  - Inline `_write_v21` back into `to_lerobotv21` (single caller now).
- `lerobot_v30.py`:
  - Inline `_write_v30` back into `to_lerobotv30` (single caller now).

### Keep
- `parallel.py` (all-core parallelism + SLURM/cgroup affinity via
  `available_cpus`), tqdm progress, `test_lerobot_parallel.py`, and the Phase-A
  design spec. `available_cpus` respecting cgroup affinity is generally correct
  and not sharding-specific, so it stays.

## Verification
- `uv run pytest` — full suite green (minus the deleted `test_distributed.py`).
- `pre-commit run --all-files` — ruff/format/typecheck clean.
- `git diff main..HEAD` contains no `shard`/`slurm`/`aggregate`/`distributed`
  conversion code; only the single-node parallelism feature remains.
- No import of `distributed` / `aggregate` / `slurm_submit` remains in `src` or
  `tests`.

## Git strategy
Add removal commit(s) on top of the current branch history (sharding is safely
on the `sharding` branch). The final `main..HEAD` diff is clean even though the
history shows add-then-remove.
