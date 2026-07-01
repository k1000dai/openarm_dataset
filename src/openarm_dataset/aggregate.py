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

"""Aggregate sharded LeRobot partial datasets into one dataset."""

import argparse
import pathlib

from .distributed import SUPPORTED_FORMATS, aggregate_shards


def main():
    """CLI entry point for aggregating sharded conversions."""
    parser = argparse.ArgumentParser(
        description="Aggregate sharded LeRobot partial datasets into one dataset"
    )
    parser.add_argument(
        "shards_dir",
        help="Directory containing shard-XXXXX/ partial datasets",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="Path of the aggregated output dataset (must not exist)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--format",
        help="Expected format of the shards; validated against them if given",
        default=None,
        choices=list(SUPPORTED_FORMATS),
    )

    args = parser.parse_args()
    aggregate_shards(args.shards_dir, args.output, format=args.format)


if __name__ == "__main__":
    main()
