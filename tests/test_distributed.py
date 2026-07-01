# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sharded conversion + aggregation must equal a single-process conversion.

These tests need no cluster and no lerobot: they convert the fixture in shards,
aggregate, and compare per-episode data / stats / info totals against a
single-process run. Video *layout* differs (sharded packing) and is not
compared byte-wise, only checked for presence.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openarm_dataset import Dataset
from openarm_dataset.distributed import (
    _shard_bounds,
    aggregate_shards,
    convert_shard,
)

FIXTURE_DIR = Path(__file__).parent / "fixture"
DATASET_0_3_0_PATH = FIXTURE_DIR / "dataset_0.3.0"
FPS = 30
FORMATS = ["lerobot_v2.1", "lerobot_v3.0", "gr00t"]


def _single_process(out: Path, fmt: str, success_only: bool = False) -> Path:
    dataset = Dataset(DATASET_0_3_0_PATH)
    dataset.set_smoothing(1.0)
    dataset.write(
        out,
        format=fmt,
        fps=FPS,
        train_split=0.8,
        success_only=success_only,
        jobs=1,
    )
    return out


def _sharded(
    shards_dir: Path,
    out: Path,
    fmt: str,
    num_shards: int,
    success_only: bool = False,
) -> Path:
    for shard_index in range(num_shards):
        convert_shard(
            DATASET_0_3_0_PATH,
            shards_dir,
            fmt,
            num_shards,
            shard_index,
            fps=FPS,
            success_only=success_only,
            jobs=1,
        )
    aggregate_shards(shards_dir, out, format=fmt)
    return out


def _data_by_episode(root: Path) -> dict[int, pd.DataFrame]:
    frames: dict[int, list] = {}
    for parquet in sorted(Path(root).glob("data/**/*.parquet")):
        df = pd.read_parquet(parquet)
        for episode_index, sub in df.groupby("episode_index"):
            frames.setdefault(int(episode_index), []).append(sub)
    return {
        episode_index: pd.concat(parts).sort_values("index").reset_index(drop=True)
        for episode_index, parts in frames.items()
    }


def _stats_close(a, b, path: str = "") -> None:
    if isinstance(a, dict):
        assert set(a) == set(b), f"stats keys differ at {path}: {set(a) ^ set(b)}"
        for key in a:
            _stats_close(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"stats list length differs at {path}"
        for x, y in zip(a, b):
            _stats_close(x, y, path)
    elif a is None or b is None:
        assert a == b, f"stats differ at {path}: {a} vs {b}"
    else:
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9, err_msg=path)


def _assert_equivalent(ref: Path, agg: Path, fmt: str) -> None:
    ref_data = _data_by_episode(ref)
    agg_data = _data_by_episode(agg)
    assert set(ref_data) == set(agg_data), "episode sets differ"

    for episode_index, ref_df in ref_data.items():
        agg_df = agg_data[episode_index]
        assert len(ref_df) == len(agg_df)
        for column in ("action", "observation.state"):
            for xr, xa in zip(ref_df[column].to_list(), agg_df[column].to_list()):
                np.testing.assert_array_equal(np.asarray(xr), np.asarray(xa))
        for column in ("index", "frame_index", "episode_index", "task_index"):
            np.testing.assert_array_equal(
                ref_df[column].to_numpy(), agg_df[column].to_numpy()
            )

    ref_stats = json.loads((ref / "meta" / "stats.json").read_text())
    agg_stats = json.loads((agg / "meta" / "stats.json").read_text())
    _stats_close(ref_stats, agg_stats)

    ref_info = json.loads((ref / "meta" / "info.json").read_text())
    agg_info = json.loads((agg / "meta" / "info.json").read_text())
    for key in ("total_episodes", "total_frames", "total_tasks", "splits"):
        assert ref_info[key] == agg_info[key], f"info {key} differs"

    videos = list(agg.glob("videos/**/*.mp4"))
    assert videos, "no aggregated video files"
    for video in videos:
        assert video.stat().st_size > 0

    if fmt == "gr00t":
        assert (agg / "meta" / "modality.json").exists()


@pytest.mark.parametrize("fmt", FORMATS)
def test_sharded_equals_single_process(tmp_path, fmt):
    ref = _single_process(tmp_path / "ref", fmt)
    agg = _sharded(tmp_path / "shards", tmp_path / "agg", fmt, num_shards=2)
    _assert_equivalent(ref, agg, fmt)


@pytest.mark.parametrize("fmt", ["lerobot_v2.1", "lerobot_v3.0"])
def test_more_shards_than_episodes(tmp_path, fmt):
    # Fixture has 2 episodes; the 3rd shard is empty and must be skipped.
    ref = _single_process(tmp_path / "ref", fmt)
    agg = _sharded(tmp_path / "shards", tmp_path / "agg", fmt, num_shards=3)
    _assert_equivalent(ref, agg, fmt)


@pytest.mark.parametrize("fmt", ["lerobot_v2.1", "lerobot_v3.0"])
def test_single_shard(tmp_path, fmt):
    ref = _single_process(tmp_path / "ref", fmt)
    agg = _sharded(tmp_path / "shards", tmp_path / "agg", fmt, num_shards=1)
    _assert_equivalent(ref, agg, fmt)


def test_success_only(tmp_path):
    fmt = "lerobot_v3.0"
    ref = _single_process(tmp_path / "ref", fmt, success_only=True)
    agg = _sharded(
        tmp_path / "shards", tmp_path / "agg", fmt, num_shards=2, success_only=True
    )
    _assert_equivalent(ref, agg, fmt)
    # Only the success episode (id 3) survives.
    info = json.loads((agg / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1


def test_shard_bounds_covers_all_items():
    n, k = 8, 3
    bounds = [_shard_bounds(n, k, i) for i in range(k)]
    assert bounds[0] == (0, 3)
    assert bounds[1] == (3, 6)
    assert bounds[2] == (6, 8)
    # contiguous, non-overlapping, covering [0, n)
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    for (_, prev_stop), (next_start, _) in zip(bounds, bounds[1:]):
        assert prev_stop == next_start


def test_shard_bounds_more_shards_than_items():
    n, k = 2, 3
    assert [_shard_bounds(n, k, i) for i in range(k)] == [(0, 1), (1, 2), (2, 2)]


def test_convert_shard_rejects_bad_index(tmp_path):
    with pytest.raises(ValueError):
        convert_shard(DATASET_0_3_0_PATH, tmp_path, "lerobot_v3.0", 2, 2, jobs=1)


def test_aggregate_rejects_existing_output(tmp_path):
    shards = tmp_path / "shards"
    convert_shard(DATASET_0_3_0_PATH, shards, "lerobot_v3.0", 1, 0, jobs=1)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        aggregate_shards(shards, out, format="lerobot_v3.0")


def test_slurm_submit_writes_scripts(tmp_path):
    from types import SimpleNamespace

    from openarm_dataset import slurm_submit

    out = tmp_path / "out"
    args = SimpleNamespace(
        input=tmp_path / "in",
        output=out,
        format="lerobot_v3.0",
        num_shards=4,
        partition="defq",
        cpus_per_task=8,
        mem="32G",
        time="04:00:00",
        account=None,
        job_name="openarm",
        shards_dir=None,
        fps=30,
        smoothing_cutoff=1.0,
        train_split=0.8,
        success_only=True,
        jobs=None,
    )
    convert_path, aggregate_path, shards_dir, output = slurm_submit._write_scripts(args)

    convert_text = convert_path.read_text()
    assert "#SBATCH --array=0-3" in convert_text
    assert "--num-shards 4" in convert_text
    assert "--shard-index ${SLURM_ARRAY_TASK_ID}" in convert_text
    assert "--jobs ${SLURM_CPUS_PER_TASK}" in convert_text
    assert "--success-only" in convert_text

    aggregate_text = aggregate_path.read_text()
    assert "openarm_dataset.aggregate" in aggregate_text
    assert str(shards_dir) in aggregate_text
    assert str(output) in aggregate_text
