import dataclasses
import logging
from typing import Literal

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.history as _history
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    # Attention masks are discrete bookkeeping values. Prevent autodiff from
    # tracing their prefix sums into the training graph (which otherwise
    # lowers to a large reduce-window JVP for every long history batch).
    # ``jnp.cumsum`` lowers to a full-width ``reduce-window``.  For the long
    # history batches this makes XLA spend tens of seconds constant-folding the
    # discrete mask during the first multi-device executable compile.  The
    # associative tree scan has identical integer semantics, but exposes a
    # logarithmic-depth graph that compiles reliably on all replicas.
    cumsum = jax.lax.associative_scan(
        jnp.add, jax.lax.stop_gradient(mask_ar).astype(jnp.int32), axis=1
    )
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float,
                  max_period: float) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period)**fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    history: _history.HistoryEncoderConfig | None = None
    separate_history_image_encoder: bool = False
    history_train_scope: Literal[
        "history_action_lora",
        "history_vlm_action_lora",
        "history_vlm_action_lora_siglip",
        "stage1_except_siglip",
        "full_except_siglip",
        "state_action_baseline",
    ] = "history_action_lora"

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if self.history is not None:
            if self.history_train_scope == "history_action_lora":
                trainable = nnx.Any(
                    nnx_utils.PathRegex(".*history_conditioner.*"),
                    nnx_utils.PathRegex(".*history_modulation.*"),
                    nnx_utils.PathRegex(".*history_adaln.*"),
                    nnx_utils.PathRegex(".*history_token_proj.*"),
                    nnx_utils.PathRegex(".*llm.*_1.*lora.*"),
                )
            elif self.history_train_scope == "history_vlm_action_lora":
                trainable = nnx.Any(
                    nnx_utils.PathRegex(".*history_conditioner.*"),
                    nnx_utils.PathRegex(".*history_modulation.*"),
                    nnx_utils.PathRegex(".*history_adaln.*"),
                    nnx_utils.PathRegex(".*history_token_proj.*"),
                    nnx_utils.PathRegex(".*llm.*lora.*"),
                )
            elif self.history_train_scope == "history_vlm_action_lora_siglip":
                trainable = nnx.Any(
                    nnx_utils.PathRegex(".*history_conditioner.*"),
                    nnx_utils.PathRegex(".*history_modulation.*"),
                    nnx_utils.PathRegex(".*history_adaln.*"),
                    nnx_utils.PathRegex(".*history_token_proj.*"),
                    nnx_utils.PathRegex(".*llm.*lora.*"),
                    nnx_utils.PathRegex(".*PaliGemma/img.*"),
                )
            elif self.history_train_scope == "stage1_except_siglip":
                trainable = nnx.Any(
                    nnx_utils.PathRegex(".*history_conditioner.*"),
                    nnx_utils.PathRegex(".*history_modulation.*"),
                    nnx_utils.PathRegex(".*history_adaln.*"),
                    nnx_utils.PathRegex(".*history_token_proj.*"),
                    nnx_utils.PathRegex(".*llm.*lora.*"),
                    nnx_utils.PathRegex(
                        ".*(state_proj|action_in_proj|action_time_mlp_in|action_time_mlp_out|action_out_proj).*"
                    ),
                )
            elif self.history_train_scope == "full_except_siglip":
                return nnx_utils.PathRegex(".*PaliGemma/img.*")
            elif self.history_train_scope == "state_action_baseline":
                trainable = nnx_utils.PathRegex(
                    ".*history_conditioner/(state_action_head|state_action_head_norm).*"
                )
            else:
                raise ValueError(f"Unsupported history training scope: {self.history_train_scope}")
            return nnx.Not(trainable)
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter, )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(nnx.Not(action_expert_params_filter), )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter, )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")), )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):

    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        if config.separate_history_image_encoder and config.history is None:
            raise ValueError("A separate history image encoder requires history conditioning.")
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            ))
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            ))
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.history_image_encoder = None
        if config.separate_history_image_encoder:
            self.history_image_encoder = nnx_bridge.ToNNX(
                _siglip.Module(
                    num_classes=paligemma_config.width,
                    variant="So400m/14",
                    pool_type="none",
                    scan=True,
                    dtype_mm=config.dtype,
                )
            )
            self.history_image_encoder.lazy_init(
                next(iter(config.fake_obs().images.values())), train=False, rngs=rngs
            )
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.history_conditioner = None
        self.history_modulation = None
        self.history_adaln = None
        self.history_adaln_layer_scale = None
        self.history_adaln_layer_bias = None
        self.history_token_proj = None
        if config.history is not None:
            self.history_conditioner = _history.HistoryConditioner(config.history, rngs=rngs)
            if config.history.conditioning_mode == "film":
                self.history_modulation = nnx.Linear(
                    config.history.d_model,
                    2 * action_expert_config.width,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                    rngs=rngs,
                )
            elif config.history.conditioning_mode == "adaln":
                self.history_adaln = nnx.Linear(
                    config.history.condition_dim,
                    4 * action_expert_config.width,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                    rngs=rngs,
                )
                self.history_adaln_layer_scale = nnx.Param(
                    jnp.ones((action_expert_config.depth, 4 * action_expert_config.width))
                )
                self.history_adaln_layer_bias = nnx.Param(
                    jnp.zeros((action_expert_config.depth, 4 * action_expert_config.width))
                )
            elif config.history.conditioning_mode == "prefix_tokens":
                self.history_token_proj = nnx.Linear(
                    config.history.d_model, action_expert_config.width, rngs=rngs
                )
            else:
                self.history_token_proj = nnx.Linear(
                    config.history.d_model, paligemma_config.width, rngs=rngs
                )

    def encode_history_images(self, images: jax.Array, *, pooled_grid_size: int = 4) -> jax.Array:
        image_encoder = self.PaliGemma.img
        if self.history_image_encoder is not None:
            image_encoder = self.history_image_encoder
        image_tokens, _ = image_encoder(images, train=False)
        source_grid_size = int(image_tokens.shape[1]**0.5)
        if source_grid_size**2 != image_tokens.shape[1]:
            raise ValueError(f"Expected a square image-token grid, got {image_tokens.shape[1]} tokens.")
        if source_grid_size % pooled_grid_size != 0:
            raise ValueError(f"Cannot pool a {source_grid_size}x{source_grid_size} grid to {pooled_grid_size}x{pooled_grid_size}.")
        cell_size = source_grid_size // pooled_grid_size
        features = image_tokens.reshape(
            images.shape[0], pooled_grid_size, cell_size, pooled_grid_size, cell_size, image_tokens.shape[-1]
        )
        features = jnp.mean(features, axis=(2, 4))
        return features.reshape(images.shape[0], pooled_grid_size**2, image_tokens.shape[-1])

    def encode_history_conditions(
        self,
        visual_features: jax.Array,
        states: jax.Array,
        valid_mask: jax.Array,
        anchor_indices: jax.Array,
        *,
        train: bool = False,
    ) -> jax.Array:
        if self.history_conditioner is None:
            raise ValueError("History conditioning is not enabled for this Pi0 model.")
        conditions = self.history_conditioner.gather_anchor_conditions(
            visual_features, states, valid_mask, anchor_indices, train=train
        )
        if self.history_conditioner.config.conditioning_mode in ("film", "adaln"):
            return conditions.reshape((-1, conditions.shape[-1]))
        if self.history_conditioner.config.conditioning_mode == "single_token":
            return conditions.reshape((-1, 1, conditions.shape[-1]))
        return conditions.reshape((-1, conditions.shape[-2], conditions.shape[-1]))

    def encode_single_token_history(
        self,
        visual_features: jax.Array,
        states: jax.Array,
        valid_mask: jax.Array,
        anchor_indices: jax.Array,
        use_previous: jax.Array,
        *,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        if self.history_conditioner is None:
            raise ValueError("History conditioning is not enabled for this Pi0 model.")
        if self.history_conditioner.config.conditioning_mode != "single_token":
            raise ValueError("Single-token encoding requires single-token conditioning.")
        sequence = self.history_conditioner.encode_sequence(
            visual_features, states, valid_mask, train=train
        )
        conditions = self.history_conditioner.gather_single_token_conditions(
            sequence, anchor_indices, use_previous
        )
        return conditions.reshape((-1, 1, conditions.shape[-1])), sequence

    def predict_history_actions(self, encoded_sequence: jax.Array) -> jax.Array:
        if self.history_conditioner is None:
            raise ValueError("History conditioning is not enabled for this Pi0 model.")
        return self.history_conditioner.predict_actions(encoded_sequence)

    def predict_state_actions(self, states: jax.Array) -> jax.Array:
        if self.history_conditioner is None:
            raise ValueError("History conditioning is not enabled for this Pi0 model.")
        return self.history_conditioner.predict_actions_from_states(states)

    def init_history_cache(self, batch_size: int):
        if self.history_conditioner is None:
            raise ValueError("History conditioning is not enabled for this Pi0 model.")
        return self.history_conditioner.init_cache(batch_size)

    def history_attention_diagnostics(self, cache) -> jax.Array:
        if self.history_conditioner is None:
            raise ValueError("History attention diagnostics require a history-enabled policy.")
        return self.history_conditioner.attention_diagnostics(cache)

    def history_film_diagnostics(self, history_condition: jax.Array) -> dict[str, jax.Array]:
        if self.history_modulation is None:
            raise ValueError("History conditioning is not enabled for this Pi0 model.")
        modulation = self.history_modulation(history_condition)
        zero_modulation = self.history_modulation(jnp.zeros_like(history_condition))
        scale, shift = jnp.split(modulation, 2, axis=-1)
        zero_scale, zero_shift = jnp.split(zero_modulation, 2, axis=-1)
        return {
            "history_condition": history_condition,
            "scale": scale,
            "shift": shift,
            "scale_dynamic": scale - zero_scale,
            "shift_dynamic": shift - zero_shift,
        }

    def history_adaln_modulations(self, history_condition: jax.Array) -> jax.Array:
        if self.history_adaln is None:
            raise ValueError("AdaLN history conditioning is not enabled for this Pi0 model.")
        base = self.history_adaln(history_condition)
        return base[None, ...] * self.history_adaln_layer_scale.value[:, None, :] + self.history_adaln_layer_bias.value[:, None, :]

    def history_adaln_diagnostics(self, history_condition: jax.Array) -> dict[str, jax.Array]:
        modulation = self.history_adaln_modulations(history_condition)
        attn_scale, attn_shift, mlp_scale, mlp_shift = jnp.split(modulation, 4, axis=-1)
        return {
            "history_condition": history_condition,
            "attn_scale": attn_scale,
            "attn_shift": attn_shift,
            "mlp_scale": mlp_scale,
            "mlp_shift": mlp_shift,
            "attn_scale_dynamic": attn_scale,
            "attn_shift_dynamic": attn_shift,
            "mlp_scale_dynamic": mlp_scale,
            "mlp_shift_dynamic": mlp_shift,
        }

    def _history_adaln_for_llm(self, history_condition):
        if self.history_adaln is None or history_condition is None:
            return None
        return self.history_adaln_modulations(history_condition)

    def history_prefix_diagnostics(self, history_condition: jax.Array) -> dict[str, jax.Array]:
        if self.history_token_proj is None:
            raise ValueError("History prefix diagnostics require prefix-token conditioning.")
        tokens = self.history_token_proj(history_condition)
        normalized = tokens / jnp.maximum(jnp.linalg.norm(tokens, axis=-1, keepdims=True), 1e-6)
        cosine = jnp.einsum("bkd,bld->bkl", normalized, normalized)
        token_count = tokens.shape[1]
        off_diagonal = jnp.ones((tokens.shape[0],), dtype=tokens.dtype)
        if token_count > 1:
            off_diagonal = (jnp.sum(cosine, axis=(-2, -1)) - token_count) / (
                token_count * (token_count - 1)
            )
        return {
            "history_tokens": tokens,
            "token_norms": jnp.linalg.norm(tokens, axis=-1),
            "token_variance": jnp.mean(jnp.var(tokens, axis=1), axis=-1),
            "token_pairwise_cosine": off_diagonal,
        }

    def update_history(
        self,
        observation: _model.Observation,
        cache,
    ):
        if self.history_conditioner is None:
            raise ValueError("History conditioning is not enabled for this Pi0 model.")
        observation = _model.preprocess_observation(None, observation, train=False)
        visual_features = self.encode_history_images(observation.images["base_0_rgb"])
        valid = observation.image_masks["base_0_rgb"]
        condition, cache = self.history_conditioner.step(visual_features, observation.state, valid, cache)
        if self.history_conditioner.config.conditioning_mode == "single_token":
            condition = condition[:, None, :]
        return condition, cache

    @at.typecheck
    def embed_prefix(
        self,
        obs: _model.Observation,
        history_condition: jax.Array | None = None,
        current_image_visible: jax.Array | None = None,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        if current_image_visible is None:
            current_image_visible = jnp.ones(obs.state.shape[:-1], dtype=jnp.bool_)
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(einops.repeat(
                obs.image_masks[name] & current_image_visible,
                "b -> b s",
                s=image_tokens.shape[1],
            ))
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        if history_condition is not None and self.history_conditioner.config.conditioning_mode == "single_token":
            if self.history_conditioner is None or (
                self.history_conditioner.config.conditioning_mode != "single_token"
            ):
                raise ValueError("VLM-prefix history requires single-token conditioning.")
            if self.history_token_proj is None:
                raise ValueError("Single-token history projection is not initialized.")
            history_tokens = self.history_token_proj(history_condition)
            tokens.append(history_tokens)
            input_mask.append(jnp.ones(history_tokens.shape[:2], dtype=jnp.bool_))
            ar_mask += [False] * history_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        history_condition: jax.Array | None = None,
        current_state_visible: jax.Array | None = None,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        if current_state_visible is None:
            current_state_visible = jnp.ones(obs.state.shape[:-1], dtype=jnp.bool_)
        input_mask.append(current_state_visible[:, None])
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        if history_condition is not None and self.history_conditioner.config.conditioning_mode == "film":
            if self.history_modulation is None:
                raise ValueError("Received a history condition for a Pi0 model without history conditioning.")
            scale, shift = jnp.split(self.history_modulation(history_condition), 2, axis=-1)
            action_time_tokens = action_time_tokens * (1 + scale[:, None, :]) + shift[:, None, :]
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def embed_history_prefix(
        self, history_condition: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if self.history_token_proj is None:
            raise ValueError("History-prefix embedding requires prefix-token conditioning.")
        tokens = self.history_token_proj(history_condition)
        input_mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.concatenate([
            jnp.ones((1,), dtype=jnp.bool_),
            jnp.zeros((tokens.shape[1] - 1,), dtype=jnp.bool_),
        ])
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(self,
                     rng: at.KeyArrayLike,
                     observation: _model.Observation,
                     actions: _model.Actions,
                     *,
                     train: bool = False,
                     history_condition: jax.Array | None = None,
                     current_image_visible: jax.Array | None = None,
                     current_state_visible: jax.Array | None = None) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        single_token_mode = self.history_conditioner is not None and (
            self.history_conditioner.config.conditioning_mode == "single_token"
        )
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation,
            history_condition=history_condition if single_token_mode else None,
            current_image_visible=current_image_visible,
        )
        prefix_mode = self.history_conditioner is not None and (
            self.history_conditioner.config.conditioning_mode == "prefix_tokens"
        )
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
            observation,
            x_t,
            time,
            history_condition=None if prefix_mode or single_token_mode else history_condition,
            current_state_visible=current_state_visible,
        )
        if prefix_mode:
            if history_condition is None:
                raise ValueError("Prefix-token history conditioning requires history tokens.")
            history_tokens, history_mask, history_ar_mask = self.embed_history_prefix(history_condition)
            suffix_tokens = jnp.concatenate([history_tokens, suffix_tokens], axis=1)
            suffix_mask = jnp.concatenate([history_mask, suffix_mask], axis=1)
            suffix_ar_mask = jnp.concatenate([history_ar_mask, suffix_ar_mask], axis=0)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jax.lax.associative_scan(
            jnp.add, jax.lax.stop_gradient(input_mask).astype(jnp.int32), axis=1
        ) - 1
        llm_adaln = (
            self._history_adaln_for_llm(history_condition)
            if self.history_conditioner is not None
            and self.history_conditioner.config.conditioning_mode == "adaln"
            else None
        )
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adaln_modulation=llm_adaln,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def compute_loss_with_history(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        visual_features: jax.Array,
        history_states: jax.Array,
        history_valid_mask: jax.Array,
        anchor_indices: jax.Array,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        history_condition = self.encode_history_conditions(
            visual_features, history_states, history_valid_mask, anchor_indices, train=train
        )
        return self.compute_loss(
            rng, observation, actions, train=train, history_condition=history_condition
        )

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        history_condition: jax.Array | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_mode = self.history_conditioner is not None and (
            self.history_conditioner.config.conditioning_mode == "prefix_tokens"
        )
        if prefix_mode and history_condition is None:
            raise ValueError("Prefix-token history conditioning requires history tokens.")

        # First fill the KV cache with image/language and, in prefix mode, history tokens.
        single_token_mode = self.history_conditioner is not None and (
            self.history_conditioner.config.conditioning_mode == "single_token"
        )
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation, history_condition=history_condition if single_token_mode else None
        )
        history_tokens = None
        if prefix_mode:
            history_tokens, history_mask, history_ar_mask = self.embed_history_prefix(history_condition)
            prefill_mask = jnp.concatenate([prefix_mask, history_mask], axis=1)
            prefill_ar_mask = jnp.concatenate([prefix_ar_mask, history_ar_mask], axis=0)
        else:
            prefill_mask = prefix_mask
            prefill_ar_mask = prefix_ar_mask
        prefill_attn_mask = make_attn_mask(prefill_mask, prefill_ar_mask)
        positions = jnp.cumsum(prefill_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, history_tokens], mask=prefill_attn_mask, positions=positions,
            adaln_modulation=None,
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
                history_condition=None if prefix_mode or single_token_mode else history_condition,
            )
            llm_adaln = (
                self._history_adaln_for_llm(history_condition)
                if self.history_conditioner is not None
                and self.history_conditioner.config.conditioning_mode == "adaln"
                else None
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefill_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefill_mask.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefill_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens],
                                                             mask=full_attn_mask,
                                                             positions=positions,
                                                             kv_cache=kv_cache,
                                                             adaln_modulation=llm_adaln)
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
