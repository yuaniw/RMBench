import argparse
import hashlib
import json
import pathlib

import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np

from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


def _parse_args():
    parser = argparse.ArgumentParser(description="Precompute frozen Pi0 SigLIP features for full-history training.")
    parser.add_argument("--config-name", default="pi0_base_aloha_robotwin_lora")
    parser.add_argument(
        "--checkpoint-params",
        default="./checkpoints/pi0_base_aloha_robotwin_lora/put_back_block-demo_clean-50/10000/params",
    )
    parser.add_argument("--output-dir", default="./history_cache/put_back_block-demo_clean-50")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pooled-grid-size", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def _hash_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = _parse_args()
    config = _config.get_config(args.config_name)
    if config.model.history is not None:
        raise ValueError("Feature extraction must use the baseline Pi0 config without a history module.")
    checkpoint_params = pathlib.Path(args.checkpoint_params).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = config.model.load(_model.restore_params(checkpoint_params, dtype=jax.numpy.bfloat16))
    encode = nnx_utils.module_jit(model.encode_history_images, static_argnames=("pooled_grid_size",))
    data_config = config.data.create(config.assets_dirs, config.model)
    metadata = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / metadata.fps for t in range(config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    prompted = dataset
    if data_config.prompt_from_task:
        prompted = _data_loader.TransformedDataset(
            prompted, [_transforms.PromptFromLeRobotTask(metadata.tasks)]
        )
    transformed = _data_loader.transform_dataset(prompted, data_config)
    feature_dim = None

    episode_ids = sorted(metadata.episodes)
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]

    for episode_id in episode_ids:
        start = int(dataset.episode_data_index["from"][episode_id])
        end = int(dataset.episode_data_index["to"][episode_id])
        episode_features = []
        for batch_start in range(start, end, args.batch_size):
            batch_end = min(batch_start + args.batch_size, end)
            images = np.stack(
                [np.asarray(transformed[index]["image"]["base_0_rgb"]) for index in range(batch_start, batch_end)]
            )
            if images.dtype != np.uint8:
                raise TypeError(f"Expected uint8 transformed images, got {images.dtype}.")
            images = images.astype(np.float32) / 255.0 * 2.0 - 1.0
            features = encode(jax.numpy.asarray(images), pooled_grid_size=args.pooled_grid_size)
            episode_features.append(np.asarray(features, dtype=np.float16))
        episode_features = np.concatenate(episode_features, axis=0)
        feature_dim = int(episode_features.shape[-1])
        np.save(output_dir / f"episode_{episode_id:06d}.npy", episode_features, allow_pickle=False)
        print(f"episode={episode_id} shape={episode_features.shape}")

    norm_stats_path = pathlib.Path(data_config.assets_dir) / data_config.asset_id / "norm_stats.json"
    manifest = {
        "repo_id": data_config.repo_id,
        "checkpoint_params": str(checkpoint_params),
        "checkpoint_metadata_sha256": _hash_file(checkpoint_params / "_METADATA"),
        "camera": "cam_high",
        "pooled_grid_size": args.pooled_grid_size,
        "spatial_tokens": args.pooled_grid_size**2,
        "feature_dim": feature_dim,
        "dtype": "float16",
        "total_episodes": len(episode_ids),
        "total_frames": sum(int(metadata.episodes[episode_id]["length"]) for episode_id in episode_ids),
        "norm_stats_path": str(norm_stats_path.resolve()),
        "norm_stats_sha256": _hash_file(norm_stats_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
