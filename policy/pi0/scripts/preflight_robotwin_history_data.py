"""Validate the nine supported RoboTwin history datasets and report length buckets."""

import argparse
import collections
import json
import math
from pathlib import Path

import h5py

TASKS = (
    "observe_and_pickup",
    "rearrange_blocks",
    "put_back_block",
    "swap_blocks",
    "swap_T",
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "press_button",
)


def _episode_number(path: Path) -> int:
    return int(path.stem.removeprefix("episode"))


def inspect_task(raw_root: Path, task: str, setting: str, episodes: int, bucket_size: int) -> tuple[int, int]:
    task_root = raw_root / task / setting
    hdf5_paths = sorted((task_root / "data").glob("episode*.hdf5"), key=_episode_number)
    instruction_paths = sorted((task_root / "instructions").glob("episode*.json"), key=_episode_number)
    if len(hdf5_paths) != episodes:
        raise ValueError(f"{task}: expected {episodes} hdf5 episodes, found {len(hdf5_paths)}")
    if len(instruction_paths) != episodes:
        raise ValueError(f"{task}: expected {episodes} instruction files, found {len(instruction_paths)}")

    lengths = []
    buckets = collections.Counter()
    required_cameras = ("head_camera", "left_camera", "right_camera")
    for episode_id, hdf5_path, instruction_path in zip(range(episodes), hdf5_paths, instruction_paths, strict=True):
        if _episode_number(hdf5_path) != episode_id or _episode_number(instruction_path) != episode_id:
            raise ValueError(f"{task}: episode files are not numerically contiguous at {episode_id}")
        instruction_payload = json.loads(instruction_path.read_text())
        if len(instruction_payload["seen"]) == 0:
            raise ValueError(f"{task}: episode {episode_id} has no seen instruction")

        with h5py.File(hdf5_path, "r") as handle:
            joints = handle["/joint_action"]
            raw_length = int(joints["left_gripper"].shape[0])
            if joints["left_arm"].shape != (raw_length, 6):
                raise ValueError(f"{task}: episode {episode_id} left arm is not (T, 6)")
            if joints["right_arm"].shape != (raw_length, 6):
                raise ValueError(f"{task}: episode {episode_id} right arm is not (T, 6)")
            if joints["right_gripper"].shape != (raw_length,):
                raise ValueError(f"{task}: episode {episode_id} right gripper length differs")
            if joints["vector"].shape != (raw_length, 14):
                raise ValueError(f"{task}: episode {episode_id} joint vector is not (T, 14)")
            for camera in required_cameras:
                if handle[f"/observation/{camera}/rgb"].shape[0] != raw_length:
                    raise ValueError(f"{task}: episode {episode_id} {camera} length differs")

        processed_length = raw_length - 1
        lengths.append(processed_length)
        buckets[math.ceil(processed_length / bucket_size) * bucket_size] += 1

    ordered = sorted(lengths)
    median = (ordered[(episodes - 1) // 2] + ordered[episodes // 2]) / 2
    print(
        f"{task}: episodes={episodes} frames={sum(lengths)} "
        f"min={ordered[0]} median={median:g} max={ordered[-1]} "
        f"buckets={','.join(f'{key}:{buckets[key]}' for key in sorted(buckets))}"
    )
    return episodes, sum(lengths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("../../data/data"))
    parser.add_argument("--setting", default="demo_clean")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--bucket-size", type=int, default=128)
    parser.add_argument("--spatial-tokens", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--feature-bytes", type=int, default=2)
    parser.add_argument("tasks", nargs="*", default=TASKS)
    args = parser.parse_args()

    for task in args.tasks:
        TASKS.index(task)
    totals = [inspect_task(args.raw_root, task, args.setting, args.episodes, args.bucket_size) for task in args.tasks]
    total_frames = sum(x[1] for x in totals)
    cache_gib = total_frames * args.spatial_tokens * args.feature_dim * args.feature_bytes / 1024**3
    print(
        f"total: tasks={len(args.tasks)} episodes={sum(x[0] for x in totals)} "
        f"frames={total_frames} siglip_fp16_cache_gib={cache_gib:.2f}"
    )


if __name__ == "__main__":
    main()
