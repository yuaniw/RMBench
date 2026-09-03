import dataclasses
from typing import Literal, NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

HistoryEncoderType = Literal["transformer", "mamba"]
HistoryConditioningMode = Literal["film", "adaln", "prefix_tokens", "single_token"]
HistoryPositionEncoding = Literal["learned_absolute", "rope"]
HistoryPositionOverflow = Literal["cycle", "clamp"]
HistoryRematPolicy = Literal["none", "nothing_saveable"]


@dataclasses.dataclass(frozen=True)
class HistoryEncoderConfig:
    encoder_type: HistoryEncoderType = "transformer"
    conditioning_mode: HistoryConditioningMode = "film"
    num_condition_tokens: int = 8
    action_target_dim: int = 14
    feature_dim: int = 2048
    spatial_tokens: int = 16
    state_dim: int = 32
    d_model: int = 512
    num_layers: int = 2
    num_heads: int = 8
    mlp_dim: int = 2048
    max_length: int = 384
    dropout_rate: float = 0.0
    position_encoding: HistoryPositionEncoding = "learned_absolute"
    # Behavior for learned absolute positions beyond the learned table. ``cycle``
    # preserves the legacy checkpoint behavior; ``clamp`` reuses the final
    # learned position and is preferable when training and evaluation may exceed
    # max_length.
    position_overflow: HistoryPositionOverflow = "cycle"
    rope_theta: float = 10000.0
    remat_policy: HistoryRematPolicy = "none"
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: int = 32
    mamba_dt_min: float = 0.001
    mamba_dt_max: float = 0.1
    mamba_dt_init_floor: float = 1e-4
    mamba_dt_scale: float = 1.0
    anchor_frame: bool = False
    anchor_only: bool = False
    anchor_num_layers: int = 2

    @property
    def condition_dim(self) -> int:
        return self.d_model if self.anchor_only else (2 * self.d_model if self.anchor_frame else self.d_model)


class TransformerLayerCache(NamedTuple):
    keys: jax.Array
    values: jax.Array


class TransformerHistoryCache(NamedTuple):
    layers: tuple[TransformerLayerCache, ...]
    input_tokens: jax.Array
    valid_mask: jax.Array
    encoded_outputs: jax.Array
    length: jax.Array
    anchor_tokens: jax.Array | None = None


def grow_transformer_history_cache(
    cache: TransformerHistoryCache, new_capacity: int
) -> TransformerHistoryCache:
    """Expands a Transformer cache without discarding any encoded history."""
    old_capacity = cache.input_tokens.shape[1]
    if new_capacity <= old_capacity:
        raise ValueError("New Transformer history capacity must exceed the current capacity.")
    padding = new_capacity - old_capacity
    layers = tuple(
        TransformerLayerCache(
            keys=jnp.pad(layer.keys, ((0, 0), (0, padding), (0, 0), (0, 0))),
            values=jnp.pad(layer.values, ((0, 0), (0, padding), (0, 0), (0, 0))),
        )
        for layer in cache.layers
    )
    return TransformerHistoryCache(
        layers=layers,
        input_tokens=jnp.pad(cache.input_tokens, ((0, 0), (0, padding), (0, 0))),
        valid_mask=jnp.pad(cache.valid_mask, ((0, 0), (0, padding))),
        encoded_outputs=jnp.pad(cache.encoded_outputs, ((0, 0), (0, padding), (0, 0))),
        length=cache.length,
        anchor_tokens=cache.anchor_tokens,
    )


class MambaLayerCache(NamedTuple):
    conv_state: jax.Array
    ssm_state: jax.Array


class MambaHistoryCache(NamedTuple):
    layers: tuple[MambaLayerCache, ...]
    length: jax.Array
    anchor_tokens: jax.Array | None = None


class SpatialAttentionPool(nnx.Module):
    def __init__(self, feature_dim: int, d_model: int, *, rngs: nnx.Rngs):
        self.feature_proj = nnx.Linear(feature_dim, d_model, use_bias=False, rngs=rngs)
        self.score = nnx.Linear(d_model, 1, use_bias=False, rngs=rngs)

    def project(self, features: jax.Array) -> jax.Array:
        return jnp.tanh(self.feature_proj(features))

    def __call__(self, features: jax.Array) -> jax.Array:
        projected = self.project(features)
        weights = jax.nn.softmax(self.score(projected), axis=-2)
        return jnp.sum(weights * projected, axis=-2)


class AnchorCrossAttentionBlock(nnx.Module):
    def __init__(self, config: HistoryEncoderConfig, *, rngs: nnx.Rngs):
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.query_norm = nnx.LayerNorm(config.d_model, rngs=rngs)
        self.kv_norm = nnx.LayerNorm(config.d_model, rngs=rngs)
        self.query_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.key_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.value_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.mlp_norm = nnx.LayerNorm(config.d_model, rngs=rngs)
        self.mlp_in = nnx.Linear(config.d_model, config.mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(config.mlp_dim, config.d_model, rngs=rngs)

    def __call__(self, query: jax.Array, key_value: jax.Array) -> jax.Array:
        q = self.query_proj(self.query_norm(query)).reshape(
            query.shape[0], query.shape[1], self.num_heads, self.head_dim
        )
        source = self.kv_norm(key_value)
        k = self.key_proj(source).reshape(
            source.shape[0], source.shape[1], self.num_heads, self.head_dim
        )
        v = self.value_proj(source).reshape(
            source.shape[0], source.shape[1], self.num_heads, self.head_dim
        )
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * self.head_dim**-0.5
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(query.shape)
        x = query + self.out_proj(attended)
        return x + self.mlp_out(nnx.gelu(self.mlp_in(self.mlp_norm(x))))


class HistoryInputAdapter(nnx.Module):
    def __init__(self, config: HistoryEncoderConfig, *, rngs: nnx.Rngs):
        self.spatial_pool = SpatialAttentionPool(config.feature_dim, config.d_model, rngs=rngs)
        self.state_proj = nnx.Linear(config.state_dim, config.d_model, use_bias=False, rngs=rngs)
        self.norm = nnx.LayerNorm(config.d_model, rngs=rngs)

    def __call__(self, visual_features: jax.Array, states: jax.Array) -> jax.Array:
        return self.norm(self.spatial_pool(visual_features) + self.state_proj(states))


def apply_rotary_position_embedding(
    tensor: jax.Array, positions: jax.Array, theta: float
) -> jax.Array:
    """Applies split-half RoPE to tensors shaped [..., heads, head_dim]."""
    head_dim = tensor.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even attention head dimension.")
    half_dim = head_dim // 2
    inverse_frequency = theta ** (
        -jnp.arange(half_dim, dtype=jnp.float32) / half_dim
    )
    positions = jnp.asarray(positions, dtype=jnp.float32)
    angles = positions[..., None] * inverse_frequency
    angles = angles.reshape((1, *positions.shape, 1, half_dim))
    cosine = jnp.cos(angles).astype(tensor.dtype)
    sine = jnp.sin(angles).astype(tensor.dtype)
    first, second = jnp.split(tensor, 2, axis=-1)
    return jnp.concatenate(
        (first * cosine - second * sine, second * cosine + first * sine), axis=-1
    )


class TransformerHistoryBlock(nnx.Module):
    def __init__(self, config: HistoryEncoderConfig, *, rngs: nnx.Rngs):
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if config.dropout_rate != 0.0:
            raise ValueError("History Transformer checkpointing currently requires dropout_rate=0.0.")
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.max_length = config.max_length
        self.position_encoding = config.position_encoding
        self.rope_theta = config.rope_theta
        self.attn_norm = nnx.LayerNorm(config.d_model, rngs=rngs)
        self.qkv_proj = nnx.Linear(config.d_model, 3 * config.d_model, use_bias=False, rngs=rngs)
        self.attn_out = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.mlp_norm = nnx.LayerNorm(config.d_model, rngs=rngs)
        self.mlp_in = nnx.Linear(config.d_model, config.mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(config.mlp_dim, config.d_model, rngs=rngs)

    def project_qkv(
        self, normalized: jax.Array, positions: jax.Array | None
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        query, key, value = jnp.split(self.qkv_proj(normalized), 3, axis=-1)
        query = query.reshape((*query.shape[:-1], self.num_heads, self.head_dim))
        key = key.reshape((*key.shape[:-1], self.num_heads, self.head_dim))
        value = value.reshape((*value.shape[:-1], self.num_heads, self.head_dim))
        if self.position_encoding == "rope":
            if positions is None:
                raise ValueError("RoPE attention requires explicit positions.")
            query = apply_rotary_position_embedding(query, positions, self.rope_theta)
            key = apply_rotary_position_embedding(key, positions, self.rope_theta)
        return query, key, value

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array,
        positions: jax.Array | None,
        *,
        train: bool,
    ) -> jax.Array:
        del train
        y = self.attn_norm(x)
        query, key, value = self.project_qkv(y, positions)
        scores = jnp.einsum("bthd,bshd->bhts", query, key) * self.head_dim**-0.5
        scores = jnp.where(attention_mask, scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores, axis=-1)
        y = jnp.einsum("bhts,bshd->bthd", weights, value).reshape(x.shape)
        y = self.attn_out(y)
        x = x + y
        y = self.mlp_out(nnx.gelu(self.mlp_in(self.mlp_norm(x))))
        return x + y

    def init_cache(self, batch_size: int) -> TransformerLayerCache:
        shape = (batch_size, self.max_length, self.num_heads, self.head_dim)
        return TransformerLayerCache(
            keys=jnp.zeros(shape, dtype=jnp.float32),
            values=jnp.zeros(shape, dtype=jnp.float32),
        )

    def step(
        self,
        token: jax.Array,
        valid: jax.Array,
        valid_mask: jax.Array,
        index: jax.Array,
        cache: TransformerLayerCache,
    ) -> tuple[jax.Array, TransformerLayerCache]:
        y = self.attn_norm(token)
        query, key, value = self.project_qkv(y, index)
        keys = cache.keys.at[:, index].set(jnp.where(valid[:, None, None], key, 0))
        values = cache.values.at[:, index].set(jnp.where(valid[:, None, None], value, 0))
        scores = jnp.einsum("bhd,bshd->bhs", query, keys) * self.head_dim**-0.5
        scores = jnp.where(valid_mask[:, None, :], scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores, axis=-1)
        y = jnp.einsum("bhs,bshd->bhd", weights, values).reshape(token.shape)
        x = token + self.attn_out(y)
        y = self.mlp_out(nnx.gelu(self.mlp_in(self.mlp_norm(x))))
        output = x + y
        return jnp.where(valid[:, None], output, 0), TransformerLayerCache(keys=keys, values=values)


def _call_transformer_block(
    block: TransformerHistoryBlock,
    x: jax.Array,
    attention_mask: jax.Array,
    positions: jax.Array | None,
    train: bool,  # noqa: FBT001
) -> jax.Array:
    return block(x, attention_mask, positions, train=train)


_remat_transformer_block = nnx.remat(
    _call_transformer_block,
    prevent_cse=False,
    static_argnums=(4,),
    policy=jax.checkpoint_policies.nothing_saveable,
)


class CausalTransformerHistoryEncoder(nnx.Module):
    def __init__(self, config: HistoryEncoderConfig, *, rngs: nnx.Rngs):
        self.config = config
        if config.position_encoding == "learned_absolute":
            self.position_embedding = nnx.Param(
                jax.random.normal(
                    rngs.params(), (config.max_length, config.d_model), dtype=jnp.float32
                )
                * 0.02
            )
        elif config.position_encoding == "rope":
            self.position_embedding = None
        else:
            raise ValueError(f"Unsupported history position encoding: {config.position_encoding}")
        if config.remat_policy not in ("none", "nothing_saveable"):
            raise ValueError(f"Unsupported history remat policy: {config.remat_policy}")
        self.blocks = nnx.Dict(
            {f"layer_{index}": TransformerHistoryBlock(config, rngs=rngs) for index in range(config.num_layers)}
        )
        self.final_norm = nnx.LayerNorm(config.d_model, rngs=rngs)

    def _position_embeddings(self, length: int) -> jax.Array:
        if self.position_embedding is None:
            raise ValueError("Learned position embeddings are disabled for RoPE.")
        if length <= self.config.max_length:
            return self.position_embedding.value[:length]
        indices = jnp.arange(length, dtype=jnp.int32)
        if self.config.position_overflow == "cycle":
            indices = indices % self.config.max_length
        elif self.config.position_overflow == "clamp":
            indices = jnp.minimum(indices, self.config.max_length - 1)
        else:
            raise ValueError(f"Unsupported learned-position overflow: {self.config.position_overflow}")
        return self.position_embedding.value[indices]

    def _position_embedding(self, index: jax.Array) -> jax.Array:
        if self.position_embedding is None:
            raise ValueError("Learned position embeddings are disabled for RoPE.")
        if self.config.position_overflow == "cycle":
            position = index % self.config.max_length
        elif self.config.position_overflow == "clamp":
            position = jnp.minimum(index, self.config.max_length - 1)
        else:
            raise ValueError(f"Unsupported learned-position overflow: {self.config.position_overflow}")
        return self.position_embedding.value[position]

    def encode_sequence(self, tokens: jax.Array, valid_mask: jax.Array, *, train: bool = False) -> jax.Array:
        length = tokens.shape[1]
        positions = None
        if self.config.position_encoding == "learned_absolute":
            x = tokens + self._position_embeddings(length)[None]
        else:
            x = tokens
            positions = jnp.arange(length, dtype=jnp.int32)
        causal = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))
        attention_mask = causal[None, None] & valid_mask[:, None, None, :]
        for index in range(self.config.num_layers):
            block = self.blocks[f"layer_{index}"]
            if self.config.remat_policy == "nothing_saveable":
                x = _remat_transformer_block(block, x, attention_mask, positions, train)
            else:
                x = _call_transformer_block(block, x, attention_mask, positions, train)
        x = self.final_norm(x)
        return jnp.where(valid_mask[..., None], x, 0)

    def init_cache(self, batch_size: int) -> TransformerHistoryCache:
        return TransformerHistoryCache(
            layers=tuple(
                self.blocks[f"layer_{index}"].init_cache(batch_size) for index in range(self.config.num_layers)
            ),
            input_tokens=jnp.zeros(
                (batch_size, self.config.max_length, self.config.d_model), dtype=jnp.float32
            ),
            valid_mask=jnp.zeros((batch_size, self.config.max_length), dtype=jnp.bool_),
            encoded_outputs=jnp.zeros(
                (batch_size, self.config.max_length, self.config.d_model), dtype=jnp.float32
            ),
            length=jnp.zeros((), dtype=jnp.int32),
            anchor_tokens=None,
        )

    def rebuild_cache(self, tokens: jax.Array, valid_mask: jax.Array) -> TransformerHistoryCache:
        capacity = tokens.shape[1]
        positions = None
        if self.config.position_encoding == "learned_absolute":
            x = tokens + self._position_embeddings(capacity)[None]
        else:
            x = tokens
            positions = jnp.arange(capacity, dtype=jnp.int32)
        causal = jnp.tril(jnp.ones((capacity, capacity), dtype=jnp.bool_))
        attention_mask = causal[None, None] & valid_mask[:, None, None, :]
        layers = []
        for index in range(self.config.num_layers):
            block = self.blocks[f"layer_{index}"]
            _, key, value = block.project_qkv(block.attn_norm(x), positions)
            cache_mask = valid_mask[..., None, None]
            layers.append(TransformerLayerCache(
                keys=jnp.where(cache_mask, key, 0),
                values=jnp.where(cache_mask, value, 0),
            ))
            x = block(x, attention_mask, positions, train=False)
        encoded_outputs = self.final_norm(x)
        encoded_outputs = jnp.where(valid_mask[..., None], encoded_outputs, 0)
        return TransformerHistoryCache(
            layers=tuple(layers),
            input_tokens=tokens,
            valid_mask=valid_mask,
            encoded_outputs=encoded_outputs,
            length=jnp.asarray(capacity, dtype=jnp.int32),
        )

    def step(
        self, token: jax.Array, valid: jax.Array, cache: TransformerHistoryCache
    ) -> tuple[jax.Array, TransformerHistoryCache]:
        def append_one(
            current_cache: TransformerHistoryCache, inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[TransformerHistoryCache, jax.Array]:
            current_token, current_valid = inputs
            index = current_cache.length
            input_tokens = current_cache.input_tokens.at[:, index].set(current_token)
            valid_mask = current_cache.valid_mask.at[:, index].set(current_valid)
            if self.config.position_encoding == "learned_absolute":
                x = current_token + self._position_embedding(index)
            else:
                x = current_token
            new_layers = []
            for layer_index, layer_cache in enumerate(current_cache.layers):
                x, new_layer_cache = self.blocks[f"layer_{layer_index}"].step(
                    x, current_valid, valid_mask, index, layer_cache
                )
                new_layers.append(new_layer_cache)
            output = self.final_norm(x)
            output = jnp.where(current_valid[:, None], output, 0)
            encoded_outputs = current_cache.encoded_outputs.at[:, index].set(output)
            return TransformerHistoryCache(
                layers=tuple(new_layers),
                input_tokens=input_tokens,
                valid_mask=valid_mask,
                encoded_outputs=encoded_outputs,
                length=index + 1,
                anchor_tokens=current_cache.anchor_tokens,
            ), output

        def rebuild_sliding_window(
            full_cache: TransformerHistoryCache,
        ) -> tuple[jax.Array, TransformerHistoryCache]:
            window_tokens = jnp.concatenate([full_cache.input_tokens[:, 1:], token[:, None]], axis=1)
            window_valid = jnp.concatenate([full_cache.valid_mask[:, 1:], valid[:, None]], axis=1)
            rebuilt_cache = self.rebuild_cache(window_tokens, window_valid)
            rebuilt_cache = rebuilt_cache._replace(anchor_tokens=full_cache.anchor_tokens)
            return rebuilt_cache.encoded_outputs[:, -1], rebuilt_cache

        def append_to_cache(
            current_cache: TransformerHistoryCache,
        ) -> tuple[jax.Array, TransformerHistoryCache]:
            updated_cache, output = append_one(current_cache, (token, valid))
            return output, updated_cache

        # Rebuild retained tokens so their absolute positions and hidden states match the new window.
        return jax.lax.cond(
            cache.length == cache.input_tokens.shape[1],
            rebuild_sliding_window,
            append_to_cache,
            cache,
        )

    def attention_diagnostics(self, cache: TransformerHistoryCache) -> jax.Array:
        """Compute temporal attention for the latest history token.

        Returns ``[num_layers, batch, num_heads, capacity]``. Invalid/padded
        positions are masked to zero. This replay path is only intended for
        debugging; normal recurrent inference does not materialize attention.
        """
        capacity = cache.input_tokens.shape[1]
        index = cache.length - 1
        positions = jnp.arange(capacity, dtype=jnp.int32)
        valid_mask = cache.valid_mask
        if self.config.position_encoding == "learned_absolute":
            x = cache.input_tokens + self._position_embeddings(capacity)[None]
        else:
            x = cache.input_tokens
        causal = jnp.tril(jnp.ones((capacity, capacity), dtype=jnp.bool_))
        attention_mask = causal[None, None] & valid_mask[:, None, None, :]
        diagnostics = []
        for layer_index in range(self.config.num_layers):
            block = self.blocks[f"layer_{layer_index}"]
            normalized = block.attn_norm(x)
            query, _, _ = block.project_qkv(normalized, positions)
            query_last = query[:, index]
            keys = cache.layers[layer_index].keys
            scores = jnp.einsum("bhd,bshd->bhs", query_last, keys) * block.head_dim**-0.5
            scores = jnp.where(valid_mask[:, None, :], scores, jnp.finfo(scores.dtype).min)
            layer_weights = jax.nn.softmax(scores, axis=-1)
            layer_weights = jnp.where(valid_mask[:, None, :], layer_weights, 0)
            diagnostics.append(layer_weights)
            x = block(x, attention_mask, positions if self.config.position_encoding == "rope" else None, train=False)
        return jnp.stack(diagnostics, axis=0)


class SelectiveSSMLayer(nnx.Module):
    """A correctness-first selective SSM whose full scan reuses the recurrent step."""

    def __init__(self, config: HistoryEncoderConfig, *, rngs: nnx.Rngs):
        self.d_model = config.d_model
        self.d_inner = config.d_model * config.mamba_expand
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.dt_rank = config.mamba_dt_rank
        self.norm = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.in_proj = nnx.Linear(config.d_model, 2 * self.d_inner, use_bias=False, rngs=rngs)
        self.conv_weight = nnx.Param(
            jax.random.normal(rngs.params(), (self.d_conv, self.d_inner), dtype=jnp.float32)
            / jnp.sqrt(self.d_conv)
        )
        self.conv_bias = nnx.Param(jnp.zeros((self.d_inner,), dtype=jnp.float32))
        self.x_proj = nnx.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, use_bias=False, rngs=rngs
        )
        self.dt_proj = nnx.Linear(self.dt_rank, self.d_inner, rngs=rngs)
        dt_init_std = self.dt_rank**-0.5 * config.mamba_dt_scale
        self.dt_proj.kernel.value = jax.random.uniform(
            rngs.params(),
            self.dt_proj.kernel.value.shape,
            dtype=jnp.float32,
            minval=-dt_init_std,
            maxval=dt_init_std,
        )
        initial_dt = jnp.exp(
            jax.random.uniform(
                rngs.params(),
                self.dt_proj.bias.value.shape,
                dtype=jnp.float32,
                minval=jnp.log(config.mamba_dt_min),
                maxval=jnp.log(config.mamba_dt_max),
            )
        ).clip(min=config.mamba_dt_init_floor)
        self.dt_proj.bias.value = initial_dt + jnp.log(-jnp.expm1(-initial_dt))
        self.a_log = nnx.Param(
            jnp.log(jnp.arange(1, self.d_state + 1, dtype=jnp.float32))[None].repeat(self.d_inner, axis=0)
        )
        self.skip = nnx.Param(jnp.ones((self.d_inner,), dtype=jnp.float32))
        self.out_proj = nnx.Linear(self.d_inner, config.d_model, use_bias=False, rngs=rngs)

    def init_cache(self, batch_size: int) -> MambaLayerCache:
        return MambaLayerCache(
            conv_state=jnp.zeros((batch_size, self.d_conv, self.d_inner), dtype=jnp.float32),
            ssm_state=jnp.zeros((batch_size, self.d_inner, self.d_state), dtype=jnp.float32),
        )

    def step(
        self, token: jax.Array, valid: jax.Array, cache: MambaLayerCache
    ) -> tuple[jax.Array, MambaLayerCache]:
        residual = token
        projected = self.in_proj(self.norm(token)).astype(jnp.float32)
        x, gate = jnp.split(projected, 2, axis=-1)
        candidate_conv = jnp.concatenate([cache.conv_state[:, 1:], x[:, None]], axis=1)
        convolved = jnp.sum(candidate_conv * self.conv_weight.value[None], axis=1) + self.conv_bias.value
        x = nnx.silu(convolved)
        selective = self.x_proj(x)
        dt_features = selective[:, : self.dt_rank]
        b = selective[:, self.dt_rank : self.dt_rank + self.d_state]
        c = selective[:, self.dt_rank + self.d_state :]
        dt = jax.nn.softplus(self.dt_proj(dt_features).astype(jnp.float32))
        a = -jnp.exp(self.a_log.value)
        decay = jnp.exp(dt[..., None] * a[None])
        candidate_ssm = decay * cache.ssm_state + dt[..., None] * b[:, None, :] * x[..., None]
        y = jnp.sum(candidate_ssm * c[:, None, :], axis=-1) + self.skip.value * x
        update = self.out_proj((y * nnx.silu(gate)).astype(token.dtype))
        output = residual + update
        valid_state = valid[:, None, None]
        new_cache = MambaLayerCache(
            conv_state=jnp.where(valid_state, candidate_conv, cache.conv_state),
            ssm_state=jnp.where(valid_state, candidate_ssm, cache.ssm_state),
        )
        return jnp.where(valid[:, None], output, 0), new_cache


class RecurrentMambaHistoryEncoder(nnx.Module):
    def __init__(self, config: HistoryEncoderConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.layers = nnx.Dict(
            {f"layer_{index}": SelectiveSSMLayer(config, rngs=rngs) for index in range(config.num_layers)}
        )
        self.final_norm = nnx.RMSNorm(config.d_model, rngs=rngs)

    def init_cache(self, batch_size: int) -> MambaHistoryCache:
        return MambaHistoryCache(
            layers=tuple(
                self.layers[f"layer_{index}"].init_cache(batch_size) for index in range(self.config.num_layers)
            ),
            length=jnp.zeros((), dtype=jnp.int32),
            anchor_tokens=None,
        )

    def step(
        self, token: jax.Array, valid: jax.Array, cache: MambaHistoryCache
    ) -> tuple[jax.Array, MambaHistoryCache]:
        x = token
        new_layers = []
        for index, layer_cache in enumerate(cache.layers):
            layer = self.layers[f"layer_{index}"]
            x, new_cache = layer.step(x, valid, layer_cache)
            new_layers.append(new_cache)
        x = self.final_norm(x)
        return jnp.where(valid[:, None], x, 0), MambaHistoryCache(
            layers=tuple(new_layers), length=cache.length + 1, anchor_tokens=cache.anchor_tokens
        )

    def encode_sequence(self, tokens: jax.Array, valid_mask: jax.Array, *, train: bool = False) -> jax.Array:
        del train
        cache = self.init_cache(tokens.shape[0])

        def scan_step(carry, inputs):
            token, valid = inputs
            output, carry = self.step(token, valid, carry)
            return carry, output

        _, outputs = jax.lax.scan(scan_step, cache, (jnp.swapaxes(tokens, 0, 1), jnp.swapaxes(valid_mask, 0, 1)))
        return jnp.swapaxes(outputs, 0, 1)


class LearnedQueryHistoryResampler(nnx.Module):
    """Compresses every causal history prefix into a fixed set of learned-query tokens."""

    def __init__(self, config: HistoryEncoderConfig, *, rngs: nnx.Rngs):
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.queries = nnx.Param(
            jax.random.normal(
                rngs.params(), (config.num_condition_tokens, config.d_model), dtype=jnp.float32
            ) * 0.02
        )
        self.query_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.key_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.value_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.attn_norm = nnx.LayerNorm(config.d_model, rngs=rngs)
        self.mlp_norm = nnx.LayerNorm(config.d_model, rngs=rngs)
        self.mlp_in = nnx.Linear(config.d_model, config.mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(config.mlp_dim, config.d_model, rngs=rngs)

    def __call__(
        self, encoded_sequence: jax.Array, valid_mask: jax.Array, anchor_indices: jax.Array
    ) -> jax.Array:
        batch_size, length, width = encoded_sequence.shape
        query = self.query_proj(self.queries.value).reshape(
            self.queries.value.shape[0], self.num_heads, self.head_dim
        )
        source = self.attn_norm(encoded_sequence)
        key = self.key_proj(source).reshape(batch_size, length, self.num_heads, self.head_dim)
        value = self.value_proj(source).reshape(batch_size, length, self.num_heads, self.head_dim)
        scores = jnp.einsum("qhd,bthd->bqht", query, key) * self.head_dim**-0.5
        scores = scores[:, None]

        positions = jnp.arange(length)[None, None, :]
        causal_valid = valid_mask[:, None, :] & (positions <= anchor_indices[..., None])
        scores = jnp.where(
            causal_valid[:, :, None, None, :], scores, jnp.finfo(scores.dtype).min
        )
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("baqht,bthd->baqhd", weights, value).reshape(
            batch_size, anchor_indices.shape[1], self.queries.value.shape[0], width
        )
        x = self.queries.value[None, None] + self.out_proj(attended)
        return x + self.mlp_out(nnx.gelu(self.mlp_in(self.mlp_norm(x))))


class HistoryConditioner(nnx.Module):
    def __init__(self, config: HistoryEncoderConfig, *, rngs: nnx.Rngs):
        self.config = config
        if config.anchor_frame and config.conditioning_mode != "adaln":
            raise ValueError("Anchor-frame conditioning currently requires AdaLN mode.")
        if config.anchor_only and not config.anchor_frame:
            raise ValueError("anchor_only requires anchor_frame to be enabled.")
        if config.anchor_frame and config.anchor_num_layers <= 0:
            raise ValueError("anchor_num_layers must be positive when anchor conditioning is enabled.")
        self.input_adapter = HistoryInputAdapter(config, rngs=rngs)
        self.anchor_attention = None
        if config.anchor_frame:
            self.anchor_attention = nnx.Dict(
                {f"layer_{index}": AnchorCrossAttentionBlock(config, rngs=rngs)
                 for index in range(config.anchor_num_layers)}
            )
        if config.encoder_type == "transformer":
            self.encoder = CausalTransformerHistoryEncoder(config, rngs=rngs)
        elif config.encoder_type == "mamba":
            self.encoder = RecurrentMambaHistoryEncoder(config, rngs=rngs)
        else:
            raise ValueError(f"Unsupported history encoder type: {config.encoder_type}")
        self.resampler = None
        self.action_head_norm = None
        self.action_head = None
        self.state_action_head_norm = None
        self.state_action_head = None
        if config.conditioning_mode == "prefix_tokens":
            if config.encoder_type != "transformer":
                raise ValueError("Prefix-token conditioning currently requires the Transformer history encoder.")
            if config.num_condition_tokens < 2:
                raise ValueError("Prefix-token conditioning requires at least two condition tokens.")
            self.resampler = LearnedQueryHistoryResampler(config, rngs=rngs)
        elif config.conditioning_mode == "single_token":
            if config.encoder_type != "transformer":
                raise ValueError("Single-token conditioning currently requires the Transformer history encoder.")
            self.action_head_norm = nnx.LayerNorm(config.d_model, rngs=rngs)
            self.action_head = nnx.Linear(config.d_model, config.action_target_dim, rngs=rngs)
            self.state_action_head_norm = nnx.LayerNorm(config.state_dim, rngs=rngs)
            self.state_action_head = nnx.Linear(config.state_dim, config.action_target_dim, rngs=rngs)

    def encode_sequence(
        self, visual_features: jax.Array, states: jax.Array, valid_mask: jax.Array, *, train: bool = False
    ) -> jax.Array:
        tokens = self.input_adapter(visual_features, states)
        return self.encoder.encode_sequence(tokens, valid_mask, train=train)

    def init_cache(self, batch_size: int):
        cache = self.encoder.init_cache(batch_size)
        if self.config.anchor_frame:
            anchor_tokens = jnp.zeros(
                (batch_size, self.config.spatial_tokens, self.config.d_model), dtype=jnp.float32
            )
            cache = cache._replace(anchor_tokens=anchor_tokens)
        return cache

    def _anchor_condition(self, visual_features: jax.Array, anchor_indices: jax.Array) -> jax.Array:
        projected = self.input_adapter.spatial_pool.project(visual_features)
        batch_size, anchor_count = anchor_indices.shape
        anchor_tokens = jnp.broadcast_to(projected[:, 0, None], (batch_size, anchor_count, projected.shape[2], projected.shape[3]))
        current = jax.vmap(lambda values, indices: values[indices])(projected, anchor_indices)
        anchor_tokens = anchor_tokens.reshape((-1, anchor_tokens.shape[-2], anchor_tokens.shape[-1]))
        current = current.reshape((-1, current.shape[-2], current.shape[-1]))
        weights = jax.nn.softmax(self.input_adapter.spatial_pool.score(current), axis=-2)
        query = jnp.sum(weights * current, axis=-2, keepdims=True)
        for index in range(self.config.anchor_num_layers):
            query = self.anchor_attention[f"layer_{index}"](query, anchor_tokens)
        return query.squeeze(axis=1).reshape((batch_size, anchor_count, -1))

    def attention_diagnostics(self, cache):
        if self.config.encoder_type != "transformer":
            raise ValueError("Attention diagnostics require the Transformer history encoder.")
        return self.encoder.attention_diagnostics(cache)

    def step(self, visual_features: jax.Array, states: jax.Array, valid: jax.Array, cache):
        was_empty = cache.length == 0
        token = self.input_adapter(visual_features, states)
        output, cache = self.encoder.step(token, valid, cache)
        if self.config.anchor_frame:
            projected = self.input_adapter.spatial_pool.project(visual_features)
            first_valid = was_empty & valid[:, None, None]
            anchor_tokens = jnp.where(first_valid, projected, cache.anchor_tokens)
            cache = cache._replace(anchor_tokens=anchor_tokens)
            weights = jax.nn.softmax(self.input_adapter.spatial_pool.score(projected), axis=-2)
            query = jnp.sum(weights * projected, axis=-2, keepdims=True)
            for index in range(self.config.anchor_num_layers):
                query = self.anchor_attention[f"layer_{index}"](query, cache.anchor_tokens)
            output = jnp.concatenate([output, query.squeeze(axis=1)], axis=-1)
            if self.config.anchor_only:
                output = query.squeeze(axis=1)
            output = jnp.where(valid[:, None], output, 0)
        if self.config.conditioning_mode in ("film", "adaln", "single_token"):
            return output, cache
        anchor_indices = jnp.full((token.shape[0], 1), cache.length - 1, dtype=jnp.int32)
        condition_tokens = self.resampler(cache.encoded_outputs, cache.valid_mask, anchor_indices)
        return condition_tokens[:, 0], cache

    def gather_anchor_conditions(
        self,
        visual_features: jax.Array,
        states: jax.Array,
        valid_mask: jax.Array,
        anchor_indices: jax.Array,
        *,
        train: bool = False,
    ) -> jax.Array:
        sequence = self.encode_sequence(visual_features, states, valid_mask, train=train)
        if self.config.conditioning_mode in ("film", "adaln", "single_token"):
            conditions = jax.vmap(lambda encoded, indices: encoded[indices])(sequence, anchor_indices)
            if self.config.anchor_frame:
                anchor = self._anchor_condition(visual_features, anchor_indices)
                conditions = anchor if self.config.anchor_only else jnp.concatenate([conditions, anchor], axis=-1)
            return conditions
        return self.resampler(sequence, valid_mask, anchor_indices)

    def gather_single_token_conditions(
        self, encoded_sequence: jax.Array, anchor_indices: jax.Array, use_previous: jax.Array
    ) -> jax.Array:
        if self.config.conditioning_mode != "single_token":
            raise ValueError("Single-token gathering requires single-token conditioning.")
        indices = anchor_indices - use_previous.astype(anchor_indices.dtype)
        return jax.vmap(lambda encoded, selected: encoded[selected])(encoded_sequence, indices)

    def predict_actions(self, encoded_sequence: jax.Array) -> jax.Array:
        if self.action_head is None or self.action_head_norm is None:
            raise ValueError("History action prediction requires single-token conditioning.")
        return self.action_head(self.action_head_norm(encoded_sequence))

    def predict_actions_from_states(self, states: jax.Array) -> jax.Array:
        if self.state_action_head is None or self.state_action_head_norm is None:
            raise ValueError("State action prediction requires single-token conditioning.")
        return self.state_action_head(self.state_action_head_norm(states))
