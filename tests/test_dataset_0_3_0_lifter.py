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

from pathlib import Path
import pytest

from openarm_dataset.dataset import Dataset

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.3.0_lifter"

ARM_JOINT_COLUMNS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "gripper",
]

LIFTER_JOINT_COLUMNS = ["elevation"]


@pytest.fixture
def dataset():
    return Dataset(DATASET_DIR)


def test_num_episodes(dataset):
    assert dataset.num_episodes == 2


def test_load_obs_includes_lifter(dataset):
    obs = dataset.load_obs(0)
    assert set(obs) == {
        "arms/left/qpos",
        "arms/left/qvel",
        "arms/left/qtorque",
        "arms/right/qpos",
        "arms/right/qvel",
        "arms/right/qtorque",
        "lifter/elevation",
    }
    assert list(obs["lifter/elevation"].columns) == LIFTER_JOINT_COLUMNS
    assert obs["lifter/elevation"].index.name == "timestamp"
    assert obs["lifter/elevation"].shape == (745, 1)


def test_load_action_includes_lifter(dataset):
    action = dataset.load_action(0)
    assert set(action) == {
        "arms/left/qpos",
        "arms/right/qpos",
        "lifter/elevation",
    }
    assert list(action["lifter/elevation"].columns) == LIFTER_JOINT_COLUMNS
    assert action["lifter/elevation"].shape == (90, 1)


def test_load_all_episodes_have_lifter(dataset):
    for i in range(dataset.num_episodes):
        obs = dataset.load_obs(i)
        action = dataset.load_action(i)
        assert "lifter/elevation" in obs
        assert "lifter/elevation" in action
        assert not obs["lifter/elevation"].empty
        assert not action["lifter/elevation"].empty


def test_sample_includes_lifter(dataset):
    samples = dataset.sample(hz=30, episode_index=0)
    assert len(samples) > 1
    assert "lifter/elevation" in samples[0].obs
    assert "lifter/elevation" in samples[0].action
    assert samples[0].obs["lifter/elevation"].shape == (1,)
    assert samples[0].action["lifter/elevation"].shape == (1,)


def test_write_preserves_lifter(dataset, tmp_path):
    output = tmp_path / "out"
    dataset.write(output)
    for episode_id in ("0", "3"):
        assert (
            output / "episodes" / episode_id / "obs" / "lifter" / "elevation.parquet"
        ).exists()
        assert (
            output / "episodes" / episode_id / "action" / "lifter" / "elevation.parquet"
        ).exists()
    rewritten = Dataset(output)
    obs = rewritten.load_obs(0)
    assert "lifter/elevation" in obs
    assert list(obs["lifter/elevation"].columns) == LIFTER_JOINT_COLUMNS
