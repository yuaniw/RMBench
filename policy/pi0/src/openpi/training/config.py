"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.history as history
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
from openpi.shared import franka_memory
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=False)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=False)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    assets_dir: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions", )

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False


class GroupFactory(Protocol):

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=False)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(inputs=[
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len), ),
                ], )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(_tokenizer.FASTTokenizer(model_config.max_token_len), ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=False)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            assets_dir=str(self.assets.assets_dir or assets_dirs),
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=False)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=False)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=False)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # Shared by current-observation transforms, history targets and policy outputs.
    delta_joint_mask: tuple[bool, ...] = _transforms.make_bool_mask(6, -1, 6, -1)
    action_output_dim: int = 14
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(default=_transforms.Group(inputs=[
        _transforms.RepackTransform({
            "images": {
                "cam_high": "observation.images.top"
            },
            "state": "observation.state",
            "actions": "action",
        })
    ]))
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action", )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(
                adapt_to_pi=self.adapt_to_pi, action_output_dim=self.action_output_dim,
            )],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = self.delta_joint_mask
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=False)
class LeRobotFrankaLeftDataConfig(LeRobotAlohaDataConfig):
    """Native left Franka: seven joints and an absolute gripper command.

    LeRobot stores absolute targets. DeltaActions references every future target
    to the current observation, and AbsoluteActions reverses that at inference.
    With adapt_to_pi=False the shared image/padding adapter performs no Aloha
    joint or gripper coordinate conversion.
    """

    delta_joint_mask: tuple[bool, ...] = _transforms.make_bool_mask(7, -1)
    action_output_dim: int = 8

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.adapt_to_pi or not self.use_delta_joint_actions:
            raise ValueError("Franka left requires native coordinates and chunk-origin joint deltas.")
        if self.action_output_dim != 8 or self.delta_joint_mask != _transforms.make_bool_mask(7, -1):
            raise ValueError("Franka left layout must be seven joints followed by one gripper.")
        if model_config.action_dim != 32:
            raise ValueError("Franka left preserves the pretrained Pi0 action dimension of 32.")
        return super().create(assets_dirs, model_config)


@dataclasses.dataclass(frozen=False)
class LeRobotLiberoDataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(inputs=[
            _transforms.RepackTransform({
                "observation/image": "image",
                "observation/wrist_image": "wrist_image",
                "observation/state": "state",
                "actions": "actions",
                "prompt": "prompt",
            })
        ])

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[
                libero_policy.LiberoInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[libero_policy.LiberoOutputs()],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=False)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    history_data: "HistoryDataConfig | None" = None

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints/"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 5000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    # Keep it disabled by default so training does not contact the WandB service
    # unless a config/CLI override explicitly opts in.
    wandb_enabled: bool = False

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


@dataclasses.dataclass(frozen=True)
class HistoryDataConfig:
    cache_dir: str | None = None
    source_checkpoint_params: str | None = None
    online_image_history: bool = False
    max_episode_steps: int | None = 384
    episode_length_bucket_size: int | None = None
    anchors_per_episode: int = 8
    gradient_accumulate_episodes: int = 4
    # Single-token anti-shortcut controls. Defaults reproduce the original FiLM behavior
    # (current observation always fully visible, no auxiliary losses).
    augment_current_observation: bool = False
    full_input_probability: float = 1.0
    image_dropout_probability: float = 0.0
    strict_past_probability: float = 0.0
    history_action_loss_weight: float = 0.0
    state_action_loss_weight: float = 0.0
    # Optional frame-aligned annotation supervision for FiLM/AdaLN history
    # representations.  The directory must contain language_annotation.json.
    annotation_dir: str | None = None
    progress_loss_weight: float = 0.0
    event_loss_weight: float = 0.0
    event_sigma: float = 5.0
    remaining_loss_weight: float = 1.0
    phase_loss_weight: float = 1.0
    anchor_raw_tokens: bool = False
    # Keep decoded episode tensors in host memory after their first access.
    # This avoids repeatedly reading the same state/action arrays and history
    # feature files on every pass through the episode stream.
    cache_in_memory: bool = True
    # Limit the small per-frame cache used for current-observation transforms.
    # None keeps every transformed frame; a finite value bounds host memory.
    max_cached_frames: int | None = 512
    # Optional RoboTwin task names to mix during history training.  When set,
    # the history loader creates one episode dataset per task and samples them
    # in a shuffled stream proportional to episode counts. An empty tuple keeps the original
    # single-repository behavior.
    tasks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.anchors_per_episode <= 0:
            raise ValueError("anchors_per_episode must be positive")
        if self.gradient_accumulate_episodes <= 0:
            raise ValueError("gradient_accumulate_episodes must be positive")
        if min(self.progress_loss_weight, self.event_loss_weight, self.remaining_loss_weight, self.phase_loss_weight) < 0:
            raise ValueError("Auxiliary loss weights must be non-negative")
        if self.event_sigma <= 0:
            raise ValueError("event_sigma must be positive")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    ###
    ### finetune config for robotwin
    ###
    # pi0_base by lora
    TrainConfig(
        name="pi0_base_aloha_robotwin_lora",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                            action_expert_variant="gemma_300m_lora",
                            max_token_len=64
                            ),
        data=LeRobotAlohaDataConfig(
            repo_id="put_back_block-demo_clean-50",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=128,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=10000,
        fsdp_devices=4,  # refer line 359
    ),
    TrainConfig(
        name="pi0_base_aloha_robotwin_lora_history_transformer",
        model=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=64,
            history=history.HistoryEncoderConfig(encoder_type="transformer"),
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="put_back_block-demo_clean-50",
            adapt_to_pi=False,
            assets=AssetsConfig(
                assets_dir="./assets/pi0_base_aloha_robotwin_lora",
                asset_id="put_back_block-demo_clean-50",
            ),
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(local_files_only=True, prompt_from_task=True),
        ),
        history_data=HistoryDataConfig(
            cache_dir="./history_cache/put_back_block-demo_clean-50",
            source_checkpoint_params=(
                "./checkpoints/pi0_base_aloha_robotwin_lora/put_back_block-demo_clean-50/10000/params"
            ),
        ),
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            history=history.HistoryEncoderConfig(encoder_type="transformer"),
        ).get_freeze_filter(),
        batch_size=1,
        num_workers=0,
        weight_loader=weight_loaders.HistoryCheckpointWeightLoader(
            "./checkpoints/pi0_base_aloha_robotwin_lora/put_back_block-demo_clean-50/10000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6),
        ema_decay=None,
        num_train_steps=10_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi0_base_aloha_robotwin_lora_history_mamba",
        model=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=64,
            history=history.HistoryEncoderConfig(encoder_type="mamba"),
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="put_back_block-demo_clean-50",
            adapt_to_pi=False,
            assets=AssetsConfig(
                assets_dir="./assets/pi0_base_aloha_robotwin_lora",
                asset_id="put_back_block-demo_clean-50",
            ),
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(local_files_only=True, prompt_from_task=True),
        ),
        history_data=HistoryDataConfig(
            cache_dir="./history_cache/put_back_block-demo_clean-50",
            source_checkpoint_params=(
                "./checkpoints/pi0_base_aloha_robotwin_lora/put_back_block-demo_clean-50/10000/params"
            ),
        ),
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            history=history.HistoryEncoderConfig(encoder_type="mamba"),
        ).get_freeze_filter(),
        batch_size=1,
        num_workers=0,
        weight_loader=weight_loaders.HistoryCheckpointWeightLoader(
            "./checkpoints/pi0_base_aloha_robotwin_lora/put_back_block-demo_clean-50/10000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6),
        ema_decay=None,
        num_train_steps=10_000,
        fsdp_devices=1,
    ),
    # pi0_fast_base by lora
    TrainConfig(
        name="pi0_fast_aloha_robotwin_lora",
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0_fast.Pi0FASTConfig(
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30000,
        fsdp_devices=2,  # refer line 359
    ),
    # pi0_base by full
    TrainConfig(
        name="pi0_base_aloha_robotwin_full",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0.Pi0Config().get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        fsdp_devices=4,  # refer line 359
    ),
    # pi0_fast_base by full
    TrainConfig(
        name="pi0_fast_aloha_robotwin_full",
        model=pi0_fast.Pi0FASTConfig(),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0_fast.Pi0FASTConfig().get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30000,
        fsdp_devices=1,  # refer line 359
    ),
]

# Four-device FSDP variants keep one episode replicated and shard its gathered anchor batch.
for history_config_name in (
    "pi0_base_aloha_robotwin_lora_history_transformer",
    "pi0_base_aloha_robotwin_lora_history_mamba",
):
    history_config = next(config for config in _CONFIGS if config.name == history_config_name)
    _CONFIGS.append(
        dataclasses.replace(
            history_config,
            name=f"{history_config_name}_fsdp4",
            history_data=dataclasses.replace(
                history_config.history_data,
                anchors_per_episode=32,
                gradient_accumulate_episodes=1,
            ),
            fsdp_devices=4,
        )
    )

# Direct experiment: official dense Pi0 -> frozen SigLIP cache -> action-expert LoRA + history Transformer.
pretrained_history_base = next(
    config for config in _CONFIGS if config.name == "pi0_base_aloha_robotwin_lora_history_transformer"
)
pretrained_pi0_params = "~/.cache/openpi/openpi-assets/checkpoints/pi0_base/params"
trained_robotwin_params = (
    "./checkpoints/pi0_base_aloha_robotwin_lora/put_back_block-demo_clean-50/10000/params"
)
pretrained_history_model = pi0.Pi0Config(
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m_lora",
    max_token_len=64,
    history=history.HistoryEncoderConfig(encoder_type="transformer"),
)
_CONFIGS.append(
    dataclasses.replace(
        pretrained_history_base,
        name="pi0_base_aloha_robotwin_pretrained",
        model=pi0.Pi0Config(max_token_len=64),
        history_data=None,
        freeze_filter=pi0.Pi0Config(max_token_len=64).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(pretrained_pi0_params),
        fsdp_devices=1,
    )
)
_CONFIGS.append(
    dataclasses.replace(
        pretrained_history_base,
        name="pi0_base_aloha_robotwin_lora_history_transformer_pretrained_fsdp4",
        model=pretrained_history_model,
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            cache_dir="./history_cache/put_back_block-demo_clean-50-pi0-base",
            source_checkpoint_params=pretrained_pi0_params,
            anchors_per_episode=48,
            gradient_accumulate_episodes=1,
        ),
        freeze_filter=pretrained_history_model.get_freeze_filter(),
        weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
        fsdp_devices=4,
    )
)

# Longer direct-pretrained experiment: train LoRA in both the PaliGemma VLM and action expert,
# together with the history Transformer and FiLM. Dense VLM/SigLIP/action weights remain frozen.
pretrained_vlm_lora_history_model = dataclasses.replace(
    pretrained_history_model,
    paligemma_variant="gemma_2b_lora",
    history_train_scope="history_vlm_action_lora",
)
pretrained_vlm_lora_history_config = dataclasses.replace(
    pretrained_history_base,
    name="pi0_base_aloha_robotwin_lora_history_transformer_pretrained_vlm_lora_30k_fsdp4",
    model=pretrained_vlm_lora_history_model,
    history_data=dataclasses.replace(
        pretrained_history_base.history_data,
        cache_dir="./history_cache/put_back_block-demo_clean-50-pi0-base",
        source_checkpoint_params=pretrained_pi0_params,
        anchors_per_episode=32,
        gradient_accumulate_episodes=1,
    ),
    freeze_filter=pretrained_vlm_lora_history_model.get_freeze_filter(),
    weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
    lr_schedule=_optimizer.CosineDecaySchedule(
        peak_lr=2.5e-5,
        decay_steps=30_000,
        decay_lr=2.5e-6,
    ),
    num_train_steps=30_000,
    fsdp_devices=4,
)
_CONFIGS.append(pretrained_vlm_lora_history_config)

# Controlled positional-encoding ablation: keep the pretrained VLM-LoRA
# baseline data, cache, padding, and optimization recipe unchanged, run on one
# GPU, and replace only the history Transformer's learned absolute positions
# with RoPE.
pretrained_vlm_lora_history_rope_baseline_model = dataclasses.replace(
    pretrained_vlm_lora_history_model,
    history=dataclasses.replace(
        pretrained_vlm_lora_history_model.history,
        position_encoding="rope",
        remat_policy="none",
    ),
)
pretrained_vlm_lora_history_rope_baseline_config = dataclasses.replace(
    pretrained_vlm_lora_history_config,
    name="pi0_base_aloha_robotwin_lora_history_transformer_pretrained_vlm_lora_rope_baseline_30k_1gpu",
    model=pretrained_vlm_lora_history_rope_baseline_model,
    fsdp_devices=1,
)
_CONFIGS.append(pretrained_vlm_lora_history_rope_baseline_config)

pretrained_vlm_lora_mamba_history_model = dataclasses.replace(
    pretrained_vlm_lora_history_model,
    history=dataclasses.replace(pretrained_vlm_lora_history_model.history, encoder_type="mamba"),
)
_CONFIGS.append(
    dataclasses.replace(
        pretrained_vlm_lora_history_config,
        name="pi0_base_aloha_robotwin_lora_history_mamba_pretrained_vlm_lora_30k_fsdp4",
        model=pretrained_vlm_lora_mamba_history_model,
        freeze_filter=pretrained_vlm_lora_mamba_history_model.get_freeze_filter(),
    )
)

# Mamba2/SSD history counterpart.  Keep this as a separate config so the
# legacy selective-SSM checkpoint and its parameter tree remain untouched.
pretrained_vlm_lora_mamba2_history_model = dataclasses.replace(
    pretrained_vlm_lora_history_model,
    history=dataclasses.replace(
        pretrained_vlm_lora_history_model.history,
        encoder_type="mamba2",
        conditioning_mode="adaln",
        mamba2_state_size=64,
        mamba2_head_dim=64,
        mamba2_chunk_size=64,
    ),
)
_CONFIGS.append(
    dataclasses.replace(
        pretrained_vlm_lora_history_config,
        name="pi0_base_aloha_robotwin_lora_history_mamba2_pretrained_vlm_lora_30k_1gpu",
        model=pretrained_vlm_lora_mamba2_history_model,
        freeze_filter=pretrained_vlm_lora_mamba2_history_model.get_freeze_filter(),
        fsdp_devices=1,
    )
)

# Online shared-SigLIP history experiment: encode the whole episode with the same SigLIP used for
# the current observation, stop gradients before the history encoder, and retain current-image
# gradients through the VLM/action loss.
pretrained_vlm_lora_siglip_history_model = dataclasses.replace(
    pretrained_vlm_lora_history_model,
    separate_history_image_encoder=False,
    history_train_scope="history_vlm_action_lora_siglip",
)
_CONFIGS.append(
    dataclasses.replace(
        pretrained_history_base,
        name=(
            "pi0_base_aloha_robotwin_lora_history_transformer_pretrained_vlm_siglip1_30k_fsdp4"
        ),
        model=pretrained_vlm_lora_siglip_history_model,
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            cache_dir=None,
            source_checkpoint_params=None,
            online_image_history=True,
            anchors_per_episode=32,
            gradient_accumulate_episodes=1,
        ),
        freeze_filter=pretrained_vlm_lora_siglip_history_model.get_freeze_filter(),
        weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        fsdp_devices=4,
    )
)
_CONFIGS.append(
    dataclasses.replace(
        pretrained_history_base,
        name="pi0_base_aloha_robotwin_lora_history_transformer_vlm_lora_30k_fsdp4",
        model=pretrained_vlm_lora_history_model,
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            cache_dir="./history_cache/put_back_block-demo_clean-50",
            source_checkpoint_params=trained_robotwin_params,
            anchors_per_episode=32,
            gradient_accumulate_episodes=1,
        ),
        freeze_filter=pretrained_vlm_lora_history_model.get_freeze_filter(),
        weight_loader=weight_loaders.OfficialPi0WithTrainedSiglipWeightLoader(
            official_params_path=pretrained_pi0_params,
            trained_siglip_params_path=trained_robotwin_params,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        fsdp_devices=4,
    )
)

# Single-token memory experiment initialized from official Pi0 except for the Robotwin-trained
# SigLIP. Its history cache uses the same trained SigLIP, while both Gemma branches use new LoRA.
single_token_history_model = pi0.Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    max_token_len=64,
    history=history.HistoryEncoderConfig(
        encoder_type="transformer",
        conditioning_mode="single_token",
        num_condition_tokens=1,
        action_target_dim=14,
    ),
    history_train_scope="history_vlm_action_lora",
)
_CONFIGS.append(
    dataclasses.replace(
        pretrained_history_base,
        name=(
            "pi0_base_aloha_robotwin_lora_history_transformer_single_token_image_dropout_fsdp4"
        ),
        model=single_token_history_model,
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            cache_dir="./history_cache/put_back_block-demo_clean-50",
            source_checkpoint_params=trained_robotwin_params,
            anchors_per_episode=32,
            gradient_accumulate_episodes=1,
            full_input_probability=0.5,
            image_dropout_probability=0.5,
            strict_past_probability=0.0,
            history_action_loss_weight=0.0,
            state_action_loss_weight=0.0,
        ),
        freeze_filter=single_token_history_model.get_freeze_filter(),
        weight_loader=weight_loaders.OfficialPi0WithTrainedSiglipWeightLoader(
            official_params_path=pretrained_pi0_params,
            trained_siglip_params_path=trained_robotwin_params,
        ),
        fsdp_devices=4,
    )
)

_CONFIGS.append(
    dataclasses.replace(
        pretrained_history_base,
        name=(
            "pi0_base_aloha_robotwin_lora_history_transformer_single_token_image_fsdp4"
        ),
        model=single_token_history_model,
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            cache_dir="./history_cache/put_back_block-demo_clean-50",
            source_checkpoint_params=trained_robotwin_params,
            anchors_per_episode=32,
            gradient_accumulate_episodes=1,
            full_input_probability=1,
            image_dropout_probability=0.0,
            strict_past_probability=0.0,
            history_action_loss_weight=0.0,
            state_action_loss_weight=0.0,
        ),
        freeze_filter=single_token_history_model.get_freeze_filter(),
        weight_loader=weight_loaders.OfficialPi0WithTrainedSiglipWeightLoader(
            official_params_path=pretrained_pi0_params,
            trained_siglip_params_path=trained_robotwin_params,
        ),
        fsdp_devices=4,
    )
)

_CONFIGS.append(
    dataclasses.replace(
        pretrained_history_base,
        name=(
            "pi0_base_aloha_robotwin_pretrained_history_transformer_single_token_image_fsdp4"
        ),
        model=single_token_history_model,
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            cache_dir="./history_cache/put_back_block-demo_clean-50-pi0-base",
            source_checkpoint_params=pretrained_pi0_params,
            anchors_per_episode=32,
            gradient_accumulate_episodes=1,
            full_input_probability=1,
            image_dropout_probability=0.0,
            strict_past_probability=0.0,
            history_action_loss_weight=0.0,
            state_action_loss_weight=0.0,
        ),
        freeze_filter=single_token_history_model.get_freeze_filter(),
        weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
        fsdp_devices=4,
    )
)


ROBOTWIN_HISTORY_TASKS = (
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
# Local NVMe mirror of the precomputed history features.  Keeping this path
# outside the Ceph workspace avoids per-step mmap reads over the shared FS.
ROBOTWIN_HISTORY_CACHE_DIR = "/data/home/gzy/openpi_history_cache"
ROBOTWIN_HISTORY_ARTIFACT_VERSION = "history-rope-v1"
ROBOTWIN_HISTORY_ADALN_ARTIFACT_VERSION = "history-adaln-v1"
ROBOTWIN_HISTORY_ANCHOR_ADALN_ARTIFACT_VERSION = "history-anchor-adaln-v1"
# Separate namespace for the two-layer Anchor-AdaLN recipe with 64 sampled
# history anchors per episode.  The cached history features remain compatible
# with the regular ``history-rope-v1`` cache.
ROBOTWIN_HISTORY_ANCHOR_ADALN_BZ64_ARTIFACT_VERSION = "history-anchor-adaln-bz64-v1"
ROBOTWIN_HISTORY_ANCHOR_ONLY_ARTIFACT_VERSION = "history-anchor-only-v1"
ROBOTWIN_HISTORY_ASSETS_DIR = "./assets/pi0_base_aloha_robotwin_history-rope-v1"

# Keep the larger-anchor experiment scoped to the three requested tasks.  The
# regular Anchor-AdaLN recipe below remains available for every RoboTwin task.
ROBOTWIN_HISTORY_ANCHOR_ADALN_BZ64_TASKS = (
    "swap_blocks",
    "blocks_ranking_try",
    "cover_blocks",
)


def robotwin_history_repo_id(task: str) -> str:
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return f"{task}-demo_clean-50-{ROBOTWIN_HISTORY_ARTIFACT_VERSION}"


def robotwin_history_precompute_config_name(task: str) -> str:
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return f"pi0_base_aloha_robotwin_pretrained_{task}_{ROBOTWIN_HISTORY_ARTIFACT_VERSION}"


def robotwin_history_train_config_name(task: str) -> str:
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return (
        "pi0_base_aloha_robotwin_lora_history_transformer_rope_vlm_lora_30k_1gpu_"
        f"{task}_{ROBOTWIN_HISTORY_ARTIFACT_VERSION}"
    )

def robotwin_history_adaln_train_config_name(task: str) -> str:
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return (
        "pi0_base_aloha_robotwin_lora_history_transformer_rope_adaln_vlm_lora_30k_1gpu_"
        f"{task}_{ROBOTWIN_HISTORY_ADALN_ARTIFACT_VERSION}"
    )


def robotwin_history_aux_adaln_train_config_name(task: str) -> str:
    """Name for annotation-supervised AdaLN history experiments.

    The auxiliary heads are part of ``HistoryConditioner`` and are therefore
    reusable for every RoboTwin task.  Keeping a separate name makes the
    press-button pilot checkpoint unambiguous while allowing the same recipe
    to be instantiated for another task later.
    """
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return (
        "pi0_base_aloha_robotwin_lora_history_transformer_rope_adaln_aux_"
        f"vlm_lora_30k_1gpu_{task}_history-adaln-aux-v1"
    )


def robotwin_history_anchor_adaln_train_config_name(task: str) -> str:
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return (
        "pi0_base_aloha_robotwin_lora_history_transformer_rope_anchor_adaln_vlm_lora_30k_1gpu_"
        f"{task}_{ROBOTWIN_HISTORY_ANCHOR_ADALN_ARTIFACT_VERSION}"
    )


def robotwin_history_anchor_adaln_bz64_train_config_name(task: str) -> str:
    """Name for the two-layer Anchor-AdaLN recipe with 64 anchors/episode."""
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return (
        "pi0_base_aloha_robotwin_lora_history_transformer_rope_anchor_adaln_bz64_"
        "vlm_lora_30k_1gpu_"
        f"{task}_{ROBOTWIN_HISTORY_ANCHOR_ADALN_BZ64_ARTIFACT_VERSION}"
    )


def robotwin_history_anchor_only_train_config_name(task: str) -> str:
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return (
        "pi0_base_aloha_robotwin_lora_history_transformer_rope_anchor_only_adaln_vlm_lora_30k_1gpu_"
        f"{task}_{ROBOTWIN_HISTORY_ANCHOR_ONLY_ARTIFACT_VERSION}"
    )


def robotwin_history_anchor_adaln_4layer_train_config_name(task: str) -> str:
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return (
        "pi0_base_aloha_robotwin_lora_history_transformer_rope_anchor_adaln4_"
        f"vlm_lora_10k_1gpu_{task}_history-anchor-adaln4-v1"
    )


def robotwin_history_anchor_adaln_4layer_30k_train_config_name(task: str) -> str:
    """Name for the four-layer Anchor-AdaLN recipe trained for 30k steps."""
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return (
        "pi0_base_aloha_robotwin_lora_history_transformer_rope_anchor_adaln4_"
        f"vlm_lora_30k_1gpu_{task}_history-anchor-adaln4-v1"
    )


def robotwin_history_anchor_adaln_4layer_bz64_10k_train_config_name(task: str) -> str:
    """Four-layer Anchor-AdaLN recipe with 64 history anchors per update."""
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return (
        "pi0_base_aloha_robotwin_lora_history_transformer_rope_anchor_adaln4_bz64_"
        f"vlm_lora_10k_1gpu_{task}_history-anchor-adaln4-bz64-v1"
    )


def _robotwin_history_data_config(task: str) -> LeRobotAlohaDataConfig:
    repo_id = robotwin_history_repo_id(task)
    return dataclasses.replace(
        pretrained_history_base.data,
        repo_id=repo_id,
        assets=AssetsConfig(
            assets_dir=ROBOTWIN_HISTORY_ASSETS_DIR,
            asset_id=repo_id,
        ),
    )


def robotwin_history_data_config(task: str) -> LeRobotAlohaDataConfig:
    return _robotwin_history_data_config(task)


robotwin_history_rope_model = dataclasses.replace(
    pretrained_vlm_lora_history_model,
    history=dataclasses.replace(
        pretrained_vlm_lora_history_model.history,
        position_encoding="rope",
        remat_policy="nothing_saveable",
    ),
)
robotwin_history_adaln_model = dataclasses.replace(
    robotwin_history_rope_model,
    history=dataclasses.replace(
        robotwin_history_rope_model.history,
        conditioning_mode="adaln",
    ),
)
robotwin_history_anchor_adaln_model = dataclasses.replace(
    robotwin_history_adaln_model,
    history=dataclasses.replace(
        robotwin_history_adaln_model.history,
        anchor_frame=True,
        anchor_num_layers=2,
    ),
)
# Annotation-supervised AdaLN variant.  The auxiliary progress/event heads are
# enabled by the AdaLN conditioning mode; no annotation is fed at inference.
# This model is intentionally task agnostic and is used below for the
# press-button pilot configuration.
robotwin_history_aux_adaln_model = dataclasses.replace(
    robotwin_history_anchor_adaln_model,
    history=dataclasses.replace(
        robotwin_history_anchor_adaln_model.history,
        conditioning_mode="adaln",
        anchor_frame=True,
        anchor_only=False,
        anchor_num_layers=2,
    ),
)
robotwin_history_anchor_adaln_raw_model = dataclasses.replace(
    robotwin_history_aux_adaln_model,
    history=dataclasses.replace(
        robotwin_history_aux_adaln_model.history,
        anchor_visual_mode="raw",
        # SigLIP So400m/14 receives 224x224 images in the current transform,
        # hence the raw patch grid is 16x16=256 tokens.
        anchor_raw_tokens=256,
    ),
)
robotwin_history_anchor_only_model = dataclasses.replace(
    robotwin_history_anchor_adaln_model,
    history=dataclasses.replace(
        robotwin_history_anchor_adaln_model.history,
        anchor_only=True,
    ),
)
robotwin_history_anchor_adaln_4layer_model = dataclasses.replace(
    robotwin_history_anchor_adaln_model,
    history=dataclasses.replace(
        robotwin_history_anchor_adaln_model.history,
        num_layers=4,
    ),
)

# Learned-absolute counterpart used to isolate positional encoding from the
# Robotwin history-rope-v1 data/cache and training recipe.  Clamp positions
# beyond the 384-entry table so offline training and streaming evaluation use
# exactly the same overflow behavior without temporal wrap-around.
robotwin_history_absolute_model = dataclasses.replace(
    pretrained_vlm_lora_history_model,
    history=dataclasses.replace(
        pretrained_vlm_lora_history_model.history,
        position_encoding="learned_absolute",
        position_overflow="clamp",
        remat_policy="none",
    ),
)

# Online visual-encoder ablation for the RobotWin absolute-position recipe.
# Keep every training setting aligned with the cached baseline and only change
# history image loading plus the trainable shared SigLIP scope.
robotwin_history_absolute_online_siglip_model = dataclasses.replace(
    robotwin_history_absolute_model,
    separate_history_image_encoder=False,
    history_train_scope="history_vlm_action_lora_siglip",
)


def robotwin_history_absolute_train_config_name(task: str) -> str:
    if task not in ROBOTWIN_HISTORY_TASKS:
        raise ValueError(f"Unsupported RoboTwin history task: {task}")
    return (
        "pi0_base_aloha_robotwin_lora_history_transformer_absolute_vlm_lora_30k_1gpu_"
        f"{task}_history-absolute-v1"
    )


ROBOTWIN_MULTITASK_HISTORY_TASKS = (
    "put_back_block",
    "rearrange_blocks",
    "battery_try",
)
ROBOTWIN_MULTITASK_HISTORY_CONFIG_NAME = (
    "pi0_base_aloha_robotwin_lora_history_transformer_absolute_vlm_lora_"
    "30k_3gpu_batch48_put_back_rearrange_battery_history-absolute-v1"
)
ROBOTWIN_MULTITASK_HISTORY_1GPU_CONFIG_NAME = (
    "pi0_base_aloha_robotwin_lora_history_transformer_absolute_vlm_lora_"
    "30k_1gpu_batch32_put_back_rearrange_battery_history-absolute-v1"
)
ROBOTWIN_FOURTASK_HISTORY_4GPU_CONFIG_NAME = (
    "pi0_base_aloha_robotwin_lora_history_transformer_rope_vlm_lora_"
    "30k_4gpu_dp_cover_press_putback_rearrange_history-rope-v1"
)
ROBOTWIN_FOURTASK_HISTORY_TASKS = (
    "cover_blocks",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
)
ROBOTWIN_ALLTASK_HISTORY_ANCHOR_ADALN_BZ32_4GPU_CONFIG_NAME = (
    "pi0_base_aloha_robotwin_lora_history_transformer_rope_anchor_adaln_bz32_"
    "vlm_lora_30k_4gpu_all9_history-anchor-adaln-bz32-v1"
)
ROBOTWIN_ALLTASK_HISTORY_ANCHOR_ADALN_BZ32_4GPU_80K_CONFIG_NAME = (
    "pi0_base_aloha_robotwin_lora_history_transformer_rope_anchor_adaln_bz32_"
    "vlm_lora_80k_4gpu_all9_history-anchor-adaln-bz32-v1"
)

for robotwin_history_task in ROBOTWIN_HISTORY_TASKS:
    robotwin_data = _robotwin_history_data_config(robotwin_history_task)
    robotwin_repo_id = robotwin_history_repo_id(robotwin_history_task)
    _CONFIGS.append(
        dataclasses.replace(
            pretrained_history_base,
            name=robotwin_history_precompute_config_name(robotwin_history_task),
            model=pi0.Pi0Config(max_token_len=64),
            data=robotwin_data,
            history_data=None,
            freeze_filter=pi0.Pi0Config(max_token_len=64).get_freeze_filter(),
            weight_loader=weight_loaders.CheckpointWeightLoader(pretrained_pi0_params),
            batch_size=1,
            num_workers=0,
            fsdp_devices=1,
        )
    )
    if robotwin_history_task in ("observe_and_pickup", "cover_blocks"):
        _CONFIGS.append(
            dataclasses.replace(
                pretrained_history_base,
                name=robotwin_history_anchor_only_train_config_name(robotwin_history_task),
                model=robotwin_history_anchor_only_model,
                data=robotwin_data,
                history_data=dataclasses.replace(
                    pretrained_history_base.history_data,
                    cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                    source_checkpoint_params=pretrained_pi0_params,
                    max_episode_steps=None,
                    episode_length_bucket_size=128,
                    anchors_per_episode=32,
                    gradient_accumulate_episodes=1,
                ),
                freeze_filter=robotwin_history_anchor_only_model.get_freeze_filter(),
                weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
                lr_schedule=_optimizer.CosineDecaySchedule(
                    peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6
                ),
                batch_size=1,
                num_workers=0,
                num_train_steps=30_000,
                fsdp_devices=1,
            )
        )
    _CONFIGS.append(
        dataclasses.replace(
            pretrained_history_base,
            name=robotwin_history_anchor_adaln_train_config_name(robotwin_history_task),
            model=robotwin_history_anchor_adaln_model,
            data=robotwin_data,
            history_data=dataclasses.replace(
                pretrained_history_base.history_data,
                cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                source_checkpoint_params=pretrained_pi0_params,
                max_episode_steps=None,
                episode_length_bucket_size=128,
                anchors_per_episode=32,
                gradient_accumulate_episodes=1,
            ),
            freeze_filter=robotwin_history_anchor_adaln_model.get_freeze_filter(),
            weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6
            ),
            batch_size=1,
            num_workers=0,
            num_train_steps=30_000,
            fsdp_devices=1,
        )
    )
    # Two-layer Anchor-AdaLN variant with 64 sampled anchors per episode.
    # Keep this as a separate checkpoint namespace so the existing bz32 runs
    # can be resumed or evaluated independently.
    if robotwin_history_task in ROBOTWIN_HISTORY_ANCHOR_ADALN_BZ64_TASKS:
        _CONFIGS.append(
            dataclasses.replace(
                pretrained_history_base,
                name=robotwin_history_anchor_adaln_bz64_train_config_name(robotwin_history_task),
                model=robotwin_history_anchor_adaln_model,
                data=robotwin_data,
                history_data=dataclasses.replace(
                    pretrained_history_base.history_data,
                    cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                    source_checkpoint_params=pretrained_pi0_params,
                    max_episode_steps=None,
                    episode_length_bucket_size=128,
                    anchors_per_episode=64,
                    gradient_accumulate_episodes=1,
                ),
                freeze_filter=robotwin_history_anchor_adaln_model.get_freeze_filter(),
                weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
                lr_schedule=_optimizer.CosineDecaySchedule(
                    peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6
                ),
                batch_size=1,
                num_workers=0,
                num_train_steps=30_000,
                fsdp_devices=1,
            )
        )
    _CONFIGS.append(
        dataclasses.replace(
            pretrained_history_base,
            name=robotwin_history_adaln_train_config_name(robotwin_history_task),
            model=robotwin_history_adaln_model,
            data=robotwin_data,
            history_data=dataclasses.replace(
                pretrained_history_base.history_data,
                cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                source_checkpoint_params=pretrained_pi0_params,
                max_episode_steps=None,
                episode_length_bucket_size=128,
                anchors_per_episode=32,
                gradient_accumulate_episodes=1,
            ),
            freeze_filter=robotwin_history_adaln_model.get_freeze_filter(),
            weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6
            ),
            batch_size=1,
            num_workers=0,
            num_train_steps=10_000,
            fsdp_devices=1,
        )
    )
    # Pilot configuration for annotation-supervised remaining/phase learning.
    if robotwin_history_task == "press_button":
        _CONFIGS.append(
            dataclasses.replace(
                pretrained_history_base,
                name=robotwin_history_aux_adaln_train_config_name(robotwin_history_task),
                model=robotwin_history_aux_adaln_model,
                data=robotwin_data,
                history_data=dataclasses.replace(
                    pretrained_history_base.history_data,
                    cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                    source_checkpoint_params=pretrained_pi0_params,
                    annotation_dir="data/data/press_button/demo_clean",
                    remaining_loss_weight=1.0,
                    phase_loss_weight=1.0,
                    max_episode_steps=None,
                    episode_length_bucket_size=128,
                    anchors_per_episode=32,
                    gradient_accumulate_episodes=1,
                ),
                freeze_filter=robotwin_history_aux_adaln_model.get_freeze_filter(),
                weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
                lr_schedule=_optimizer.CosineDecaySchedule(
                    peak_lr=2.5e-5,
                    decay_steps=30_000,
                    decay_lr=2.5e-6,
                ),
                batch_size=1,
                num_workers=0,
                num_train_steps=30_000,
                fsdp_devices=1,
            )
        )
    if robotwin_history_task == "observe_and_pickup":
        _CONFIGS.append(
            dataclasses.replace(
                pretrained_history_base,
                name="pi0_base_aloha_robotwin_lora_history_transformer_rope_anchor_adaln_raw_30k_1gpu_observe_and_pickup",
                model=dataclasses.replace(
                    robotwin_history_anchor_adaln_raw_model,
                    history=dataclasses.replace(
                        robotwin_history_anchor_adaln_raw_model.history,
                        anchor_raw_tokens=256,
                    ),
                ),
                data=robotwin_data,
                history_data=dataclasses.replace(
                    pretrained_history_base.history_data,
                    cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                    source_checkpoint_params=pretrained_pi0_params,
                    annotation_dir=None,
                    remaining_loss_weight=0.0,
                    phase_loss_weight=0.0,
                    anchor_raw_tokens=True,
                    max_episode_steps=None,
                    episode_length_bucket_size=128,
                    anchors_per_episode=32,
                    gradient_accumulate_episodes=1,
                ),
                freeze_filter=dataclasses.replace(
                    robotwin_history_anchor_adaln_raw_model,
                    history=dataclasses.replace(
                        robotwin_history_anchor_adaln_raw_model.history,
                        anchor_raw_tokens=256,
                    ),
                ).get_freeze_filter(),
                weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
                lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6),
                batch_size=1,
                num_workers=0,
                num_train_steps=30_000,
                fsdp_devices=1,
            )
        )
    _CONFIGS.append(
        dataclasses.replace(
            pretrained_history_base,
            name=robotwin_history_anchor_adaln_4layer_train_config_name(robotwin_history_task),
            model=robotwin_history_anchor_adaln_4layer_model,
            data=robotwin_data,
            history_data=dataclasses.replace(
                pretrained_history_base.history_data,
                cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                source_checkpoint_params=pretrained_pi0_params,
                max_episode_steps=None,
                episode_length_bucket_size=128,
                anchors_per_episode=128,
                gradient_accumulate_episodes=1,
            ),
            freeze_filter=robotwin_history_anchor_adaln_4layer_model.get_freeze_filter(),
            weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6
            ),
            batch_size=1,
            num_workers=0,
            num_train_steps=10_000,
            fsdp_devices=1,
        )
    )

    # Four-layer Anchor-AdaLN recipe for the longer 30k-step run.  This is
    # intentionally a separate config/checkpoint namespace from the 10k
    # recipe above so the two experiments can coexist and be compared.
    _CONFIGS.append(
        dataclasses.replace(
            pretrained_history_base,
            name=robotwin_history_anchor_adaln_4layer_30k_train_config_name(robotwin_history_task),
            model=robotwin_history_anchor_adaln_4layer_model,
            data=robotwin_data,
            history_data=dataclasses.replace(
                pretrained_history_base.history_data,
                cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                source_checkpoint_params=pretrained_pi0_params,
                max_episode_steps=None,
                episode_length_bucket_size=128,
                anchors_per_episode=128,
                gradient_accumulate_episodes=1,
            ),
            freeze_filter=robotwin_history_anchor_adaln_4layer_model.get_freeze_filter(),
            weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6
            ),
            batch_size=1,
            num_workers=0,
            num_train_steps=30_000,
            fsdp_devices=1,
        )
    )

    # Four-layer Anchor-AdaLN 10k ablation with 64 history anchors per update.
    # Keep a distinct config/checkpoint namespace from the 128-anchor recipes.
    _CONFIGS.append(
        dataclasses.replace(
            pretrained_history_base,
            name=robotwin_history_anchor_adaln_4layer_bz64_10k_train_config_name(robotwin_history_task),
            model=robotwin_history_anchor_adaln_4layer_model,
            data=robotwin_data,
            history_data=dataclasses.replace(
                pretrained_history_base.history_data,
                cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                source_checkpoint_params=pretrained_pi0_params,
                max_episode_steps=None,
                episode_length_bucket_size=128,
                anchors_per_episode=64,
                gradient_accumulate_episodes=1,
            ),
            freeze_filter=robotwin_history_anchor_adaln_4layer_model.get_freeze_filter(),
            weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6
            ),
            batch_size=1,
            num_workers=0,
            num_train_steps=10_000,
            fsdp_devices=1,
        )
    )

    _CONFIGS.append(
        dataclasses.replace(
            pretrained_history_base,
            name=robotwin_history_absolute_train_config_name(robotwin_history_task),
            model=robotwin_history_absolute_model,
            data=robotwin_data,
            history_data=dataclasses.replace(
                pretrained_history_base.history_data,
                cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                source_checkpoint_params=pretrained_pi0_params,
                max_episode_steps=None,
                episode_length_bucket_size=128,
                anchors_per_episode=32,
                gradient_accumulate_episodes=1,
            ),
            freeze_filter=robotwin_history_absolute_model.get_freeze_filter(),
            weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=2.5e-5,
                decay_steps=30_000,
                decay_lr=2.5e-6,
            ),
            batch_size=1,
            num_workers=0,
            num_train_steps=30_000,
            fsdp_devices=1,
        )
    )
    _CONFIGS.append(
        dataclasses.replace(
            pretrained_history_base,
            name=robotwin_history_train_config_name(robotwin_history_task),
            model=robotwin_history_rope_model,
            data=robotwin_data,
            history_data=dataclasses.replace(
                pretrained_history_base.history_data,
                cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                source_checkpoint_params=pretrained_pi0_params,
                max_episode_steps=None,
                episode_length_bucket_size=128,
                anchors_per_episode=32,
                gradient_accumulate_episodes=1,
            ),
            freeze_filter=robotwin_history_rope_model.get_freeze_filter(),
            weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=2.5e-5,
                decay_steps=30_000,
                decay_lr=2.5e-6,
            ),
            batch_size=1,
            num_workers=0,
            num_train_steps=30_000,
            fsdp_devices=1,
        )
    )
    # Mamba history counterpart.  Keep the RoboTwin data/cache recipe identical
    # to the Transformer runs and replace only the history encoder.
    if robotwin_history_task == "swap_blocks":
        mamba_robotwin_model = dataclasses.replace(
            pretrained_vlm_lora_history_model,
            history=dataclasses.replace(
                pretrained_vlm_lora_history_model.history,
                encoder_type="mamba",
            ),
        )
        _CONFIGS.append(
            dataclasses.replace(
                pretrained_history_base,
                name="pi0_base_aloha_robotwin_lora_history_mamba_vlm_lora_10k_swap_blocks",
                model=mamba_robotwin_model,
                data=robotwin_data,
                history_data=dataclasses.replace(
                    pretrained_history_base.history_data,
                    cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                    source_checkpoint_params=pretrained_pi0_params,
                    max_episode_steps=None,
                    episode_length_bucket_size=128,
                    anchors_per_episode=96,
                    gradient_accumulate_episodes=1,
                ),
                freeze_filter=mamba_robotwin_model.get_freeze_filter(),
                weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(
                    pretrained_pi0_params
                ),
                lr_schedule=_optimizer.CosineDecaySchedule(
                    peak_lr=2.5e-5,
                    decay_steps=10_000,
                    decay_lr=2.5e-6,
                ),
                batch_size=1,
                num_workers=0,
                num_train_steps=10_000,
                fsdp_devices=1,
            )
        )
        mamba2_robotwin_model = dataclasses.replace(
            pretrained_vlm_lora_history_model,
            history=dataclasses.replace(
                pretrained_vlm_lora_history_model.history,
                encoder_type="mamba2",
                conditioning_mode="adaln",
                mamba2_state_size=64,
                mamba2_head_dim=64,
                mamba2_chunk_size=64,
            ),
        )
        _CONFIGS.append(
            dataclasses.replace(
                pretrained_history_base,
                name="pi0_base_aloha_robotwin_lora_history_mamba2_vlm_lora_10k_swap_blocks",
                model=mamba2_robotwin_model,
                data=robotwin_data,
                history_data=dataclasses.replace(
                    pretrained_history_base.history_data,
                    cache_dir=f"{ROBOTWIN_HISTORY_CACHE_DIR}/{robotwin_repo_id}-pi0-base",
                    source_checkpoint_params=pretrained_pi0_params,
                    max_episode_steps=None,
                    episode_length_bucket_size=128,
                    anchors_per_episode=96,
                    gradient_accumulate_episodes=1,
                ),
                freeze_filter=mamba2_robotwin_model.get_freeze_filter(),
                weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(
                    pretrained_pi0_params
                ),
                lr_schedule=_optimizer.CosineDecaySchedule(
                    peak_lr=2.5e-5,
                    decay_steps=10_000,
                    decay_lr=2.5e-6,
                ),
                batch_size=1,
                num_workers=0,
                num_train_steps=10_000,
                fsdp_devices=1,
            )
        )
    _CONFIGS.append(
        dataclasses.replace(
            pretrained_history_base,
            name=(
                "pi0_base_aloha_robotwin_lora_history_transformer_absolute_"
                "vlm_lora_online_siglip_30k_1gpu_"
                f"{robotwin_history_task}_history-absolute-v1"
            ),
            model=robotwin_history_absolute_online_siglip_model,
            data=robotwin_data,
            history_data=dataclasses.replace(
                pretrained_history_base.history_data,
                cache_dir=None,
                source_checkpoint_params=None,
                online_image_history=True,
                max_episode_steps=None,
                episode_length_bucket_size=128,
                anchors_per_episode=32,
                gradient_accumulate_episodes=1,
            ),
            freeze_filter=robotwin_history_absolute_online_siglip_model.get_freeze_filter(),
            weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=2.5e-5,
                decay_steps=30_000,
                decay_lr=2.5e-6,
            ),
            batch_size=1,
            num_workers=0,
            num_train_steps=30_000,
            fsdp_devices=1,
        )
    )

# Three-task history training. Each GPU owns one episode and the loader
# performs episode-level data parallelism across the three task sources.
_CONFIGS.append(
    dataclasses.replace(
        pretrained_history_base,
        name=ROBOTWIN_MULTITASK_HISTORY_CONFIG_NAME,
        model=robotwin_history_absolute_model,
        data=_robotwin_history_data_config("put_back_block"),
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            tasks=ROBOTWIN_MULTITASK_HISTORY_TASKS,
            cache_dir=None,
            source_checkpoint_params=pretrained_pi0_params,
            max_episode_steps=None,
            episode_length_bucket_size=128,
            anchors_per_episode=16,
            gradient_accumulate_episodes=1,
        ),
        freeze_filter=robotwin_history_absolute_model.get_freeze_filter(),
        weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(
            pretrained_pi0_params
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        batch_size=3,
        num_workers=0,
        num_train_steps=30_000,
        fsdp_devices=1,
    )
)

# Four-task throughput-oriented training.  Each optimizer step loads four
# episodes from one task and the history loader places one episode on each GPU
# through the batch axis.  fsdp_devices=1 keeps the model replicated, avoiding
# the expensive long-history FSDP anchor split used by the older recipe.
_CONFIGS.append(
    dataclasses.replace(
        pretrained_history_base,
        name=ROBOTWIN_FOURTASK_HISTORY_4GPU_CONFIG_NAME,
        model=robotwin_history_rope_model,
        data=_robotwin_history_data_config("cover_blocks"),
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            tasks=ROBOTWIN_FOURTASK_HISTORY_TASKS,
            cache_dir=None,
            source_checkpoint_params=pretrained_pi0_params,
            max_episode_steps=None,
            episode_length_bucket_size=128,
            anchors_per_episode=32,
            gradient_accumulate_episodes=1,
        ),
        freeze_filter=robotwin_history_rope_model.get_freeze_filter(),
        weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        batch_size=4,
        # Concurrently decode one episode per data-parallel GPU.
        num_workers=4,
        num_train_steps=30_000,
        fsdp_devices=1,
    )
)

# One complete episode per GPU, 32 anchors per episode: 128 anchors/update.
# Tasks share model weights but retain their own prompts and normalization.
_CONFIGS.append(
    dataclasses.replace(
        pretrained_history_base,
        name=ROBOTWIN_ALLTASK_HISTORY_ANCHOR_ADALN_BZ32_4GPU_CONFIG_NAME,
        model=robotwin_history_anchor_adaln_model,
        data=_robotwin_history_data_config(ROBOTWIN_HISTORY_TASKS[0]),
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            tasks=ROBOTWIN_HISTORY_TASKS,
            cache_dir=ROBOTWIN_HISTORY_CACHE_DIR,
            source_checkpoint_params=pretrained_pi0_params,
            max_episode_steps=None,
            episode_length_bucket_size=128,
            anchors_per_episode=32,
            gradient_accumulate_episodes=1,
        ),
        freeze_filter=robotwin_history_anchor_adaln_model.get_freeze_filter(),
        weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6,
        ),
        batch_size=4,
        # Decode selected episodes from independent RoboTwin tasks concurrently.
        num_workers=4,
        num_train_steps=30_000,
        fsdp_devices=1,
    )
)

# Longer four-GPU continuation recipe. Keep the data-parallel batch and
# loader settings aligned with the 30k recipe, while stretching LR decay over
# the full 80k updates.
_CONFIGS.append(
    dataclasses.replace(
        pretrained_history_base,
        name=ROBOTWIN_ALLTASK_HISTORY_ANCHOR_ADALN_BZ32_4GPU_80K_CONFIG_NAME,
        model=robotwin_history_anchor_adaln_model,
        data=_robotwin_history_data_config(ROBOTWIN_HISTORY_TASKS[0]),
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            tasks=ROBOTWIN_HISTORY_TASKS,
            cache_dir=ROBOTWIN_HISTORY_CACHE_DIR,
            source_checkpoint_params=pretrained_pi0_params,
            max_episode_steps=None,
            episode_length_bucket_size=128,
            anchors_per_episode=32,
            gradient_accumulate_episodes=1,
        ),
        freeze_filter=robotwin_history_anchor_adaln_model.get_freeze_filter(),
        weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5, decay_steps=80_000, decay_lr=2.5e-6,
        ),
        batch_size=4,
        num_workers=4,
        num_train_steps=80_000,
        fsdp_devices=1,
    )
)

# Real Franka data has its own robot layout, dataset identity and artifacts.
# Do not add it to ROBOTWIN_HISTORY_TASKS, whose names encode simulator data.
franka_memory_data = LeRobotFrankaLeftDataConfig(
    repo_id=franka_memory.REPO_ID,
    assets=AssetsConfig(
        assets_dir="./assets/pi0_franka_left_memory_260915_h50",
        asset_id=franka_memory.REPO_ID,
    ),
    base_config=DataConfig(local_files_only=True, prompt_from_task=True),
    default_prompt=franka_memory.PROMPT,
    repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform({
        "images": {
            "cam_high": "observation.images.cam_high",
            "cam_left_wrist": "observation.images.cam_left_wrist",
        },
        "state": "observation.state",
        "actions": "action",
        "prompt": "prompt",
    })]),
)
franka_memory_model = dataclasses.replace(
    robotwin_history_anchor_adaln_model,
    action_horizon=franka_memory.HORIZON,
    history=dataclasses.replace(robotwin_history_anchor_adaln_model.history, action_target_dim=8),
)
franka_memory_single_data = dataclasses.replace(
    franka_memory_data,
    repo_id=franka_memory.SINGLE_REPO_ID,
    assets=AssetsConfig(assets_dir="./assets/pi0_franka_left_memory_260915_front_h50", asset_id=franka_memory.SINGLE_REPO_ID),
    repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform({
        "images": {"cam_high": "observation.images.cam_high"},
        "state": "observation.state", "actions": "action", "prompt": "prompt",
    })]),
)
_CONFIGS.extend([
    dataclasses.replace(
        pretrained_history_base,
        name=franka_memory.PRECOMPUTE_CONFIG,
        model=pi0.Pi0Config(max_token_len=64, action_horizon=franka_memory.HORIZON),
        data=franka_memory_data,
        history_data=None,
        freeze_filter=pi0.Pi0Config().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(pretrained_pi0_params),
        batch_size=1, num_workers=0, fsdp_devices=1,
    ),
    dataclasses.replace(
        pretrained_history_base,
        name=franka_memory.TRAIN_CONFIG,
        model=franka_memory_model,
        data=franka_memory_data,
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            cache_dir=f"./history_cache/{franka_memory.REPO_ID}-pi0-base",
            source_checkpoint_params=pretrained_pi0_params,
            max_episode_steps=None, episode_length_bucket_size=128,
            anchors_per_episode=32, gradient_accumulate_episodes=1,
        ),
        freeze_filter=franka_memory_model.get_freeze_filter(),
        weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6,
        ),
        batch_size=1, num_workers=0, fsdp_devices=1, num_train_steps=30_000,
        policy_metadata={
            "robot": "franka_left", "action_output_dim": 8,
            "action_representation": "absolute_joint_targets_and_absolute_gripper",
            "training_action_representation": "joint_delta_from_chunk_origin",
            "dataset_fps": franka_memory.FPS,
        },
    ),
    dataclasses.replace(
        pretrained_history_base,
        name=franka_memory.SINGLE_PRECOMPUTE_CONFIG,
        model=pi0.Pi0Config(max_token_len=64, action_horizon=franka_memory.HORIZON),
        data=franka_memory_single_data, history_data=None,
        freeze_filter=pi0.Pi0Config().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(pretrained_pi0_params),
        batch_size=1, num_workers=0, fsdp_devices=1,
    ),
    dataclasses.replace(
        pretrained_history_base,
        name=franka_memory.SINGLE_TRAIN_CONFIG,
        model=franka_memory_model, data=franka_memory_single_data,
        history_data=dataclasses.replace(
            pretrained_history_base.history_data,
            cache_dir=f"./history_cache/{franka_memory.SINGLE_REPO_ID}-pi0-base",
            source_checkpoint_params=pretrained_pi0_params, max_episode_steps=None,
            episode_length_bucket_size=128, anchors_per_episode=32, gradient_accumulate_episodes=1,
        ),
        freeze_filter=franka_memory_model.get_freeze_filter(),
        weight_loader=weight_loaders.PretrainedHistoryCheckpointWeightLoader(pretrained_pi0_params),
        lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6),
        batch_size=1, num_workers=0, fsdp_devices=1, num_train_steps=30_000,
        policy_metadata={"robot": "franka_left", "camera_setup": "front_only", "action_output_dim": 8,
                         "action_representation": "absolute_joint_targets_and_absolute_gripper",
                         "training_action_representation": "joint_delta_from_chunk_origin", "dataset_fps": franka_memory.FPS},
    ),
])

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
