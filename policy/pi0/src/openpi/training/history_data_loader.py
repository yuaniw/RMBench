from collections.abc import Iterator
import hashlib
import json
import logging
import pathlib

import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


logger = logging.getLogger(__name__)


def episode_padding_length(
    episode_length: int,
    *,
    max_episode_steps: int | None,
    episode_length_bucket_size: int | None,
) -> int:
    """Returns the static sequence length used for one episode."""
    if episode_length_bucket_size is not None:
        if max_episode_steps is not None:
            raise ValueError("Bucketed history loading requires max_episode_steps=None.")
        if episode_length_bucket_size <= 0:
            raise ValueError("episode_length_bucket_size must be positive.")
        return (
            (episode_length + episode_length_bucket_size - 1)
            // episode_length_bucket_size
            * episode_length_bucket_size
        )
    if max_episode_steps is None:
        raise ValueError("Set max_episode_steps or episode_length_bucket_size.")
    if episode_length > max_episode_steps:
        raise ValueError(
            f"Episode has {episode_length} steps, exceeding {max_episode_steps}."
        )
    return max_episode_steps


def sample_stratified_anchors(rng: np.random.Generator, episode_length: int, num_anchors: int) -> np.ndarray:
    if num_anchors > episode_length:
        raise ValueError("num_anchors cannot exceed the episode length.")
    edges = np.linspace(0, episode_length, num_anchors + 1, dtype=np.int64)
    anchors = np.empty((num_anchors,), dtype=np.int64)
    for segment in range(num_anchors):
        anchors[segment] = rng.integers(edges[segment], edges[segment + 1])
    return anchors


def make_action_valid_mask(
    anchors: np.ndarray, *, episode_length: int, action_horizon: int
) -> np.ndarray:
    action_offsets = np.arange(action_horizon, dtype=np.int64)
    return anchors[:, None] + action_offsets[None] < episode_length


def prepare_episode_action_targets(
    raw_states: np.ndarray,
    raw_actions: np.ndarray,
    *,
    action_dim: int,
    action_target_dim: int,
    adapt_to_pi: bool,
    use_delta_joint_actions: bool,
    action_norm_stats: _transforms.NormStats,
) -> np.ndarray:
    states = aloha_policy.prepare_aloha_state(raw_states, action_dim, adapt_to_pi=adapt_to_pi)
    actions = aloha_policy.prepare_aloha_actions(raw_actions, action_dim, adapt_to_pi=adapt_to_pi)
    if use_delta_joint_actions:
        delta_mask = np.asarray(_transforms.make_bool_mask(6, -1, 6, -1))
        actions[:, : delta_mask.shape[0]] -= np.where(
            delta_mask, states[:, : delta_mask.shape[0]], 0
        )
    normalize = _transforms.Normalize({"actions": action_norm_stats}, strict=True)
    return normalize({"actions": actions})["actions"][:, :action_target_dim].astype(np.float32)


class HistoryEpisodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_config: _config.DataConfig,
        model_config: _model.BaseModelConfig,
        *,
        cache_dir: pathlib.Path | None,
        max_episode_steps: int | None,
        episode_length_bucket_size: int | None = None,
        anchors_per_episode: int,
        seed: int,
        source_checkpoint_params: pathlib.Path | None,
        online_image_history: bool = False,
        adapt_to_pi: bool,
        use_delta_joint_actions: bool,
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.cache_dir = cache_dir
        self.online_image_history = online_image_history
        self.max_episode_steps = max_episode_steps
        self.episode_length_bucket_size = episode_length_bucket_size
        self.anchors_per_episode = anchors_per_episode
        self.adapt_to_pi = adapt_to_pi
        self.use_delta_joint_actions = use_delta_joint_actions
        self.rng = np.random.default_rng(seed)

        if not online_image_history:
            manifest_path = cache_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest["repo_id"] != data_config.repo_id:
                raise ValueError(f"Cache repo_id {manifest['repo_id']} does not match {data_config.repo_id}.")
            if int(manifest["feature_dim"]) != model_config.history.feature_dim:
                raise ValueError("History cache feature dimension does not match the model config.")
            if int(manifest["spatial_tokens"]) != model_config.history.spatial_tokens:
                raise ValueError("History cache spatial token count does not match the model config.")
            source_checkpoint_params = source_checkpoint_params.expanduser().resolve()
            if pathlib.Path(manifest["checkpoint_params"]).resolve() != source_checkpoint_params:
                raise ValueError("History cache checkpoint does not match the configured source checkpoint.")
            checkpoint_metadata = source_checkpoint_params / "_METADATA"
            checkpoint_digest = hashlib.sha256(checkpoint_metadata.read_bytes()).hexdigest()
            if checkpoint_digest != manifest["checkpoint_metadata_sha256"]:
                raise ValueError("History cache source checkpoint hash does not match its manifest.")
            norm_stats_path = pathlib.Path(self.data_config.assets_dir) / self.data_config.asset_id / "norm_stats.json"
            digest = hashlib.sha256(norm_stats_path.read_bytes()).hexdigest()
            if digest != manifest["norm_stats_sha256"]:
                raise ValueError("History cache normalization statistics hash does not match its manifest.")

        metadata = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
        self.dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [t / metadata.fps for t in range(model_config.action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        prompted = self.dataset
        if data_config.prompt_from_task:
            prompted = _data_loader.TransformedDataset(
                prompted, [_transforms.PromptFromLeRobotTask(metadata.tasks)]
            )
        self.transformed = _data_loader.transform_dataset(prompted, data_config)
        self.episode_ids = tuple(sorted(metadata.episodes))
        self.episode_lengths = {episode_id: int(metadata.episodes[episode_id]["length"]) for episode_id in self.episode_ids}

        if not online_image_history and int(manifest["total_episodes"]) != len(self.episode_ids):
            raise ValueError("History cache episode count does not match the dataset metadata.")
        if data_config.norm_stats is None:
            raise ValueError("History training requires normalization statistics.")
        self.normalize_state = _transforms.Normalize({"state": data_config.norm_stats["state"]}, strict=True)

    def __len__(self) -> int:
        return len(self.episode_ids)

    def _sample_stratified_anchors(self, episode_length: int) -> np.ndarray:
        return sample_stratified_anchors(self.rng, episode_length, self.anchors_per_episode)

    def __getitem__(self, index: int) -> tuple:
        episode_id = self.episode_ids[index]
        episode_length = self.episode_lengths[episode_id]
        padded_episode_steps = episode_padding_length(
            episode_length,
            max_episode_steps=self.max_episode_steps,
            episode_length_bucket_size=self.episode_length_bucket_size,
        )
        logger.info(
            "History episode repo=%s episode=%d length=%d bucket=%d anchors=%d",
            self.data_config.repo_id,
            episode_id,
            episode_length,
            padded_episode_steps,
            self.anchors_per_episode,
        )
        episode_start = int(self.dataset.episode_data_index["from"][episode_id])
        global_indices = np.arange(episode_start, episode_start + episode_length, dtype=np.int64)
        raw_states = np.stack(
            [np.asarray(x) for x in self.dataset.hf_dataset.select(global_indices.tolist())["observation.state"]]
        )
        prepared_states = aloha_policy.prepare_aloha_state(
            raw_states, self.model_config.action_dim, adapt_to_pi=self.adapt_to_pi
        )
        states = self.normalize_state({"state": prepared_states})["state"].astype(np.float32)
        raw_actions = np.stack(
            [np.asarray(x) for x in self.dataset.hf_dataset.select(global_indices.tolist())["action"]]
        )
        episode_actions = prepare_episode_action_targets(
            raw_states,
            raw_actions,
            action_dim=self.model_config.action_dim,
            action_target_dim=self.model_config.history.action_target_dim,
            adapt_to_pi=self.adapt_to_pi,
            use_delta_joint_actions=self.use_delta_joint_actions,
            action_norm_stats=self.data_config.norm_stats["actions"],
        )

        if self.online_image_history:
            episode_images = np.stack(
                [np.asarray(self.transformed[episode_start + index]["image"]["base_0_rgb"])
                 for index in range(episode_length)]
            )
            padded_visual = np.zeros(
                (padded_episode_steps, *episode_images.shape[1:]), dtype=episode_images.dtype
            )
            padded_visual[:episode_length] = episode_images
        else:
            cache_path = self.cache_dir / f"episode_{episode_id:06d}.npy"
            visual = np.load(cache_path, mmap_mode="r")
            expected_shape = (
                episode_length,
                self.model_config.history.spatial_tokens,
                self.model_config.history.feature_dim,
            )
            if visual.shape != expected_shape:
                raise ValueError(f"Cache {cache_path} has shape {visual.shape}, expected {expected_shape}.")
            padded_visual = np.zeros(
                (
                    padded_episode_steps,
                    self.model_config.history.spatial_tokens,
                    self.model_config.history.feature_dim,
                ),
                dtype=np.float16,
            )
            padded_visual[:episode_length] = visual
        padded_states = np.zeros((padded_episode_steps, self.model_config.action_dim), dtype=np.float32)
        padded_actions = np.zeros(
            (padded_episode_steps, self.model_config.history.action_target_dim), dtype=np.float32
        )
        valid_mask = np.zeros((padded_episode_steps,), dtype=np.bool_)
        padded_states[:episode_length] = states
        padded_actions[:episode_length] = episode_actions[:, : self.model_config.history.action_target_dim]
        valid_mask[:episode_length] = True

        anchors = self._sample_stratified_anchors(episode_length)
        current_items = [self.transformed[episode_start + int(anchor)] for anchor in anchors]
        current_batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *current_items)
        np.testing.assert_allclose(
            padded_actions[anchors],
            np.asarray(current_batch["actions"])[:, 0, : self.model_config.history.action_target_dim],
            rtol=1e-5,
            atol=1e-5,
        )
        current_batch = jax.tree.map(jax.numpy.asarray, current_batch)
        action_valid_mask = make_action_valid_mask(
            anchors,
            episode_length=episode_length,
            action_horizon=self.model_config.action_horizon,
        )
        observation = _model.Observation.from_dict(current_batch)
        return (
            observation,
            current_batch["actions"],
            action_valid_mask,
            padded_visual[None],
            padded_states[None],
            valid_mask[None],
            anchors[None],
            padded_actions[None],
        )


class HistoryDataLoader:
    def __init__(self, dataset: HistoryEpisodeDataset, *, num_workers: int, shuffle: bool, seed: int):
        if num_workers != 0:
            raise ValueError("HistoryDataLoader currently requires num_workers=0 for deterministic episode sampling.")
        self.dataset = dataset
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def data_config(self) -> _config.DataConfig:
        return self.dataset.data_config

    def __iter__(self) -> Iterator[tuple]:
        while True:
            indices = np.arange(len(self.dataset))
            if self.shuffle:
                self.rng.shuffle(indices)
            for index in indices:
                batch = self.dataset[int(index)]
                yield jax.tree.map(jax.numpy.asarray, batch)


def create_history_data_loader(
    config: _config.TrainConfig, *, num_workers: int, shuffle: bool = True
) -> HistoryDataLoader:
    if config.history_data is None:
        raise ValueError("History training requires history_data configuration.")
    if config.batch_size != 1:
        raise ValueError("History dataloader batch_size is the number of episodes and must equal 1.")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = HistoryEpisodeDataset(
        data_config,
        config.model,
        cache_dir=(pathlib.Path(config.history_data.cache_dir)
                   if config.history_data.cache_dir is not None else None),
        max_episode_steps=config.history_data.max_episode_steps,
        episode_length_bucket_size=config.history_data.episode_length_bucket_size,
        anchors_per_episode=config.history_data.anchors_per_episode,
        seed=config.seed,
        source_checkpoint_params=(
            pathlib.Path(config.history_data.source_checkpoint_params).expanduser()
            if config.history_data.source_checkpoint_params is not None else None
        ),
        online_image_history=config.history_data.online_image_history,
        adapt_to_pi=config.data.adapt_to_pi,
        use_delta_joint_actions=config.data.use_delta_joint_actions,
    )
    return HistoryDataLoader(dataset, num_workers=num_workers, shuffle=shuffle, seed=config.seed)
