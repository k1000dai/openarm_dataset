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

"""Distributed (sharded) OpenArm -> LeRobot conversion.

A large conversion is split into ``num_shards`` shards. Each shard converts a
contiguous slice of the (filtered) episode list into a self-contained *partial*
LeRobot dataset under ``shards_dir/shard-XXXXX/``, then :func:`aggregate_shards`
stitches the partials into one dataset by moving/renumbering files and fixing up
the global row ``index`` — without re-encoding video.

The key invariant: global ``episode_index`` and ``task_index`` are assigned at
shard time (both derive from the full metadata, so every shard agrees without
coordination); only the row ``index`` and file numbering — which depend on other
shards' sizes — are deferred to aggregation.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import Dataset

SUPPORTED_FORMATS = ("lerobot_v2.1", "lerobot_v3.0", "gr00t")
SHARD_META_NAME = "shard_meta.json"
# Files per chunk; matches CHUNK_SIZE in lerobot_v21/lerobot_v30.
CHUNK_SIZE = 1000


def _shard_dir(shards_dir: Path, shard_index: int) -> Path:
    return Path(shards_dir) / f"shard-{shard_index:05d}"


def _filtered_episode_indices(dataset: Dataset, success_only: bool) -> list[int]:
    """Source episode indices included in the output, in metadata order."""
    return [
        index
        for index, episode in enumerate(dataset.meta.episodes)
        if episode["success"] or not success_only
    ]


def _global_task_remap(dataset: Dataset, filtered_indices: list[int]) -> dict[int, int]:
    """Map source task_index -> contiguous output task_index over all shards.

    Computed over the full filtered episode list so every shard produces the
    same mapping independently.
    """
    seen: set[int] = set()
    used: list[int] = []
    for index in filtered_indices:
        task_index = int(dataset.meta.episodes[index]["task_index"])
        if task_index not in seen:
            seen.add(task_index)
            used.append(task_index)
    used.sort()
    return {original: new for new, original in enumerate(used)}


def _shard_bounds(n_items: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    """Contiguous near-equal split: the first ``n % k`` shards get one extra."""
    base, remainder = divmod(n_items, num_shards)
    if shard_index < remainder:
        start = shard_index * (base + 1)
        return start, start + base + 1
    start = remainder * (base + 1) + (shard_index - remainder) * base
    return start, start + base


def _read_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def convert_shard(
    input_path: str | os.PathLike,
    shards_dir: str | os.PathLike,
    format: str,
    num_shards: int,
    shard_index: int,
    *,
    fps: int = 30,
    smoothing_cutoff: float = 1.0,
    train_split: float = 0.8,
    success_only: bool = False,
    jobs: int | None = None,
) -> Path:
    """Convert one shard of ``input_path`` to a partial LeRobot dataset.

    Writes to ``shards_dir/shard-{shard_index:05d}/`` and returns that path.
    ``episode_index`` and ``task_index`` are global (final) values; the row
    ``index`` and file numbering are local and fixed up by
    :func:`aggregate_shards`.
    """
    if format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format {format!r}; expected one of {SUPPORTED_FORMATS}"
        )
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(
            f"shard_index must satisfy 0 <= shard_index < {num_shards}, "
            f"got {shard_index}"
        )

    # Imported lazily so the module has no import-time cost for non-distributed use.
    from .lerobot_v21 import (
        _collect_downsampled_data,
        _collect_keys_and_joint_names,
        _write_modality_json,
        _write_v21,
    )
    from .lerobot_v30 import _write_v30

    dataset = Dataset(input_path)
    dataset.set_smoothing(cutoff=smoothing_cutoff)

    filtered = _filtered_episode_indices(dataset, success_only)
    task_remap = _global_task_remap(dataset, filtered)
    start, stop = _shard_bounds(len(filtered), num_shards, shard_index)
    shard_source_indices = filtered[start:stop]

    shard_out = _shard_dir(shards_dir, shard_index)
    if shard_out.exists():
        raise FileExistsError(f"Shard output already exists: {shard_out}")

    if not shard_source_indices:
        # More shards than episodes: emit a valid empty marker only.
        _write_shard_meta(
            shard_out, format, shard_index, start, start, 0, train_split, fps
        )
        return shard_out

    joint_keys, joint_names = _collect_keys_and_joint_names(dataset)
    records, episode_image_stats = _collect_downsampled_data(
        dataset,
        fps,
        joint_keys,
        jobs=jobs,
        episode_indices=shard_source_indices,
    )

    # Global episode_index = position in the filtered list.
    remap_episode_index = {
        source_index: start + offset
        for offset, source_index in enumerate(shard_source_indices)
    }
    total_frames = sum(num_frames for _, num_frames, *_ in records)

    if format == "lerobot_v3.0":
        _write_v30(
            dataset,
            records,
            episode_image_stats,
            shard_out,
            fps,
            train_split,
            joint_names,
            remap_episode_index,
            task_remap,
            jobs=jobs,
        )
    else:  # lerobot_v2.1 or gr00t (v2.1 + modality.json)
        _write_v21(
            dataset,
            records,
            episode_image_stats,
            shard_out,
            fps,
            train_split,
            joint_names,
            remap_episode_index,
            task_remap,
            jobs=jobs,
        )
        if format == "gr00t":
            _write_modality_json(dataset, shard_out)

    _write_shard_meta(
        shard_out, format, shard_index, start, stop, total_frames, train_split, fps
    )
    return shard_out


def _write_shard_meta(
    shard_out: Path,
    format: str,
    shard_index: int,
    episode_start: int,
    episode_stop: int,
    total_frames: int,
    train_split: float,
    fps: int,
) -> None:
    _write_json(
        Path(shard_out) / SHARD_META_NAME,
        {
            "format": format,
            "shard_index": shard_index,
            "global_episode_start": episode_start,
            "global_episode_stop": episode_stop,
            "num_episodes": episode_stop - episode_start,
            "total_frames": total_frames,
            "train_split": train_split,
            "fps": fps,
        },
    )


def _move_file(src: Path, dst: Path) -> None:
    """Move ``src`` to ``dst`` (rename when possible, copy across filesystems)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError:
        shutil.copy2(src, dst)
        Path(src).unlink()


def _global_chunk_file(counter: int, chunk_size: int) -> tuple[int, int]:
    """Global (chunk_index, file_index) for the ``counter``-th file."""
    return counter // chunk_size, counter % chunk_size


def _to_nested_list(value):
    """Recursively convert arrays/lists to nested Python scalar lists."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_nested_list(item) for item in value]
    return value


def _clean_stat_array(value) -> np.ndarray:
    """Recover a numeric ndarray from a parquet stats cell.

    pandas returns nested list columns (e.g. per-channel image stats) as
    ``object`` arrays of arrays; a single ``.tolist()`` only unwraps the outer
    level. Fully flattening to nested Python scalars first lets ``np.asarray``
    rebuild a clean numeric array with the natural dtype (int counts stay int,
    floats stay float).
    """
    return np.asarray(_to_nested_list(value))


def aggregate_shards(
    shards_dir: str | os.PathLike,
    output: str | os.PathLike,
    format: str | None = None,
) -> None:
    """Aggregate the partial datasets under ``shards_dir`` into ``output``.

    Files are moved/renumbered and the global row ``index`` is fixed up; video
    is never re-encoded. ``format`` is validated against the shards if given.
    """
    shards_dir = Path(shards_dir)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output path already exists: {output}")

    shard_dirs = sorted(d for d in shards_dir.glob("shard-*") if d.is_dir())
    if not shard_dirs:
        raise ValueError(f"No shard-* directories found under {shards_dir}")

    metas = []
    for shard_dir in shard_dirs:
        meta_path = shard_dir / SHARD_META_NAME
        if not meta_path.exists():
            raise ValueError(f"Missing {SHARD_META_NAME} in {shard_dir}")
        metas.append(_read_json(meta_path))

    formats = {meta["format"] for meta in metas}
    if len(formats) != 1:
        raise ValueError(f"Inconsistent shard formats: {sorted(formats)}")
    shard_format = formats.pop()
    if format is not None and format != shard_format:
        raise ValueError(f"Requested format {format!r} but shards are {shard_format!r}")
    format = shard_format

    order = sorted(range(len(shard_dirs)), key=lambda i: metas[i]["shard_index"])
    shard_dirs = [shard_dirs[i] for i in order]
    metas = [metas[i] for i in order]

    non_empty = [
        (shard_dir, meta)
        for shard_dir, meta in zip(shard_dirs, metas)
        if meta["num_episodes"] > 0
    ]
    if not non_empty:
        raise ValueError("All shards are empty; nothing to aggregate.")

    if format == "lerobot_v3.0":
        _aggregate_v30(non_empty, output)
    else:  # lerobot_v2.1 or gr00t
        _aggregate_v21(non_empty, output, is_gr00t=(format == "gr00t"))


def _episode_image_names(columns) -> list[str]:
    """Camera image names from ``videos/<name>/chunk_index`` episode columns."""
    prefix, suffix = "videos/", "/chunk_index"
    return [
        col[len(prefix) : -len(suffix)]
        for col in columns
        if col.startswith(prefix) and col.endswith(suffix)
    ]


def _aggregate_v30(non_empty, output: Path) -> None:
    from .lerobot_v21 import _describe_scalar
    from .lerobot_v30 import (
        CHUNK_SIZE,
        DATA_PATH,
        EPISODES_PATH,
        INFO_PATH,
        STATS_PATH,
        TASKS_PATH,
        VIDEO_PATH,
        _aggregate_stats,
        _serialize_stats,
    )

    row_offset = 0
    data_counter = 0
    video_counter: dict[str, int] = {}
    all_episode_rows: list[dict] = []
    all_episode_stats: list[dict] = []

    for shard_dir, meta in non_empty:
        ep = pd.read_parquet(
            shard_dir / EPISODES_PATH.format(chunk_index=0, file_index=0)
        )
        ep = ep.sort_values("episode_index").reset_index(drop=True)
        image_names = _episode_image_names(ep.columns)
        stats_cols = [c for c in ep.columns if c.startswith("stats/")]

        # Data files: assign each local (chunk,file) a global one, rewriting the
        # per-row global ``index`` while copying to the output location.
        data_pairs = sorted(
            {
                (int(c), int(f))
                for c, f in zip(ep["data/chunk_index"], ep["data/file_index"])
            }
        )
        data_map: dict[tuple[int, int], tuple[int, int]] = {}
        for local_chunk, local_file in data_pairs:
            global_chunk, global_file = _global_chunk_file(data_counter, CHUNK_SIZE)
            data_map[(local_chunk, local_file)] = (global_chunk, global_file)
            src = shard_dir / DATA_PATH.format(
                chunk_index=local_chunk, file_index=local_file
            )
            dst = output / DATA_PATH.format(
                chunk_index=global_chunk, file_index=global_file
            )
            df = pd.read_parquet(src)
            df["index"] = df["index"].to_numpy() + row_offset
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False)
            data_counter += 1

        # Video files: per-camera global renumber, pure move (no re-encode).
        video_map: dict[tuple[str, int, int], tuple[int, int]] = {}
        for image_name in image_names:
            counter = video_counter.get(image_name, 0)
            pairs = sorted(
                {
                    (int(c), int(f))
                    for c, f in zip(
                        ep[f"videos/{image_name}/chunk_index"],
                        ep[f"videos/{image_name}/file_index"],
                    )
                }
            )
            for local_chunk, local_file in pairs:
                global_chunk, global_file = _global_chunk_file(counter, CHUNK_SIZE)
                video_map[(image_name, local_chunk, local_file)] = (
                    global_chunk,
                    global_file,
                )
                src = shard_dir / VIDEO_PATH.format(
                    video_key=image_name, chunk_index=local_chunk, file_index=local_file
                )
                dst = output / VIDEO_PATH.format(
                    video_key=image_name,
                    chunk_index=global_chunk,
                    file_index=global_file,
                )
                _move_file(src, dst)
                counter += 1
            video_counter[image_name] = counter

        # Fix up the per-episode metadata rows.
        for _, series in ep.iterrows():
            row = series.to_dict()
            global_from = int(row["dataset_from_index"]) + row_offset
            global_to = int(row["dataset_to_index"]) + row_offset
            row["dataset_from_index"] = global_from
            row["dataset_to_index"] = global_to

            local_data = (int(row["data/chunk_index"]), int(row["data/file_index"]))
            gc, gf = data_map[local_data]
            row["data/chunk_index"] = gc
            row["data/file_index"] = gf

            for image_name in image_names:
                key = (
                    image_name,
                    int(row[f"videos/{image_name}/chunk_index"]),
                    int(row[f"videos/{image_name}/file_index"]),
                )
                gvc, gvf = video_map[key]
                row[f"videos/{image_name}/chunk_index"] = gvc
                row[f"videos/{image_name}/file_index"] = gvf

            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0

            # The per-episode ``index`` stats were computed over a shard-local
            # frame range; recompute them over the global range.
            index_stats = _describe_scalar(
                np.arange(global_from, global_to, dtype=np.int64)
            )
            for stat_name, value in index_stats.items():
                row[f"stats/index/{stat_name}"] = value

            all_episode_rows.append(row)
            all_episode_stats.append(
                {c[len("stats/") :]: _clean_stat_array(row[c]) for c in stats_cols}
            )

        row_offset += int(meta["total_frames"])

    total_frames = row_offset
    total_episodes = len(all_episode_rows)

    ep_df = pd.DataFrame(all_episode_rows).sort_values("episode_index")
    ep_df = ep_df.reset_index(drop=True)
    episodes_path = output / EPISODES_PATH.format(chunk_index=0, file_index=0)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    ep_df.to_parquet(episodes_path, index=False)

    overall_stats = _aggregate_stats(all_episode_stats)
    _write_json(output / STATS_PATH, _serialize_stats(overall_stats))

    src_tasks = non_empty[0][0] / TASKS_PATH
    dst_tasks = output / TASKS_PATH
    dst_tasks.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_tasks, dst_tasks)
    total_tasks = len(pd.read_parquet(dst_tasks))

    _write_aggregated_info(
        non_empty[0][0] / INFO_PATH,
        output / INFO_PATH,
        non_empty[0][1],
        total_episodes,
        total_frames,
        total_tasks,
    )


def _aggregate_v21(non_empty, output: Path, is_gr00t: bool) -> None:
    from .lerobot_v21 import (
        METADATA_DIR,
        _describe_scalar,
        _describe_vector,
        _get_chunk_name,
        _get_image_name_from_key,
    )

    row_offset = 0
    episodes_meta: list[dict] = []
    episodes_stats: list[dict] = []
    tasks_lines: list[str] | None = None
    camera_names: list[str] | None = None

    # For the overall stats we vstack the data columns, matching single-process.
    action_all, obs_all = [], []
    scalar_all: dict[str, list] = {
        k: []
        for k in (
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "success",
            "last_frame_index",
        )
    }

    for shard_dir, meta in non_empty:
        with (shard_dir / METADATA_DIR / "episodes.jsonl").open() as f:
            shard_episode_meta = [json.loads(line) for line in f]
        with (shard_dir / METADATA_DIR / "episodes_stats.jsonl").open() as f:
            shard_episode_stats = [json.loads(line) for line in f]
        if tasks_lines is None:
            with (shard_dir / METADATA_DIR / "tasks.jsonl").open() as f:
                tasks_lines = [line.rstrip("\n") for line in f]

        info = _read_json(shard_dir / METADATA_DIR / "info.json")
        if camera_names is None:
            camera_names = [
                key[len("observation.images.") :]
                for key in info["features"]
                if key.startswith("observation.images.")
            ]

        # Each episode file is already globally named (global episode_index);
        # move it in, rewriting the global ``index`` column.
        for record in shard_episode_meta:
            global_ep = int(record["episode_index"])
            rel = (
                Path("data")
                / _get_chunk_name(global_ep)
                / f"episode_{global_ep:06d}.parquet"
            )
            df = pd.read_parquet(shard_dir / rel)
            df["index"] = df["index"].to_numpy() + row_offset
            dst = output / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False)

            action_all.append(np.vstack(df["action"].to_list()).astype(np.float32))
            obs_all.append(
                np.vstack(df["observation.state"].to_list()).astype(np.float32)
            )
            for key in scalar_all:
                scalar_all[key].append(df[key].to_numpy())

            for cam in camera_names:
                image_name = _get_image_name_from_key(cam)
                vrel = (
                    Path("videos")
                    / _get_chunk_name(global_ep)
                    / image_name
                    / f"episode_{global_ep:06d}.mp4"
                )
                _move_file(shard_dir / vrel, output / vrel)

            episodes_meta.append(record)

        for stats_record in shard_episode_stats:
            stats_record["dataset_from_index"] = (
                int(stats_record["dataset_from_index"]) + row_offset
            )
            stats_record["dataset_to_index"] = (
                int(stats_record["dataset_to_index"]) + row_offset
            )
            global_from = stats_record["dataset_from_index"]
            global_to = stats_record["dataset_to_index"]
            index_stats = _describe_scalar(
                np.arange(global_from, global_to, dtype=np.int64)
            )
            stats_record["stats"]["index"] = index_stats
            episodes_stats.append(stats_record)

        row_offset += int(meta["total_frames"])

    meta_dir = output / METADATA_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_meta.sort(key=lambda record: record["episode_index"])
    episodes_stats.sort(key=lambda record: record["episode_index"])
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for record in episodes_meta:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (meta_dir / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for record in episodes_stats:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for line in tasks_lines or []:
            f.write(line + "\n")

    # Overall stats.json by vstacking all frames (matches single-process v2.1).
    action = np.vstack(action_all)
    observation = np.vstack(obs_all)
    overall = {
        "action": _describe_vector(action),
        "observation.state": _describe_vector(observation),
    }
    for key in scalar_all:
        overall[key] = _describe_scalar(np.concatenate(scalar_all[key]))
    _write_json(meta_dir / "stats.json", overall)

    total_episodes = len(episodes_meta)
    total_frames = row_offset
    total_tasks = len(tasks_lines or [])
    _write_aggregated_info(
        non_empty[0][0] / METADATA_DIR / "info.json",
        meta_dir / "info.json",
        non_empty[0][1],
        total_episodes,
        total_frames,
        total_tasks,
    )

    if is_gr00t:
        shutil.copy2(
            non_empty[0][0] / METADATA_DIR / "modality.json",
            meta_dir / "modality.json",
        )


def _write_aggregated_info(
    src_info_path: Path,
    dst_info_path: Path,
    meta: dict,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> None:
    """Recompute totals/splits on a shard's info.json for the merged dataset."""
    info = _read_json(src_info_path)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = total_tasks

    train_split = float(meta["train_split"])
    train_end = round(total_episodes * train_split)
    splits = {"train": f"0:{train_end}"}
    if train_end < total_episodes:
        splits["val"] = f"{train_end}:{total_episodes}"
    info["splits"] = splits

    if "total_chunks" in info:  # v2.1 only
        info["total_chunks"] = (
            max((total_episodes - 1) // CHUNK_SIZE + 1, 0) if total_episodes else 0
        )
    if "total_videos" in info:  # v2.1 only
        num_cameras = sum(
            1 for key in info["features"] if key.startswith("observation.images.")
        )
        info["total_videos"] = total_episodes * num_cameras

    _write_json(dst_info_path, info)
