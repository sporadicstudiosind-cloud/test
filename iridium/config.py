"""Configuration ladder and executable parameter accounting for Iridium-1.

One dataclass tree describes every scale from ``tiny`` (0.3 M parameters, runs
in a unit test) to ``extreme`` (33 T parameters, runs on nothing that exists).
The *same* dataclasses drive the *same* modules; scale is data, not a rewrite.
That is the whole point of the ladder: what is checked at ``nano`` is the
architecture that would be instantiated at ``base``.

Every count here is derived from tensor shapes, not transcribed from a
document. ``tests/unit/test_config_inventory.py`` re-derives them against the
real ``torch`` modules at the small scales, so the formulae cannot drift away
from the implementation.

Counting conventions
--------------------
* Attention: ``W_Q`` and ``W_O`` are ``d x d``; ``W_K`` and ``W_V`` are
  ``d x d_kv`` where ``d_kv = n_kv_heads * d_head`` (grouped-query attention).
* SwiGLU: three ``d x d_ff`` matrices (gate, up, down).
* RMSNorm: one vector of length ``d`` per norm site, two per transformer block.
* Bridge cross-attention (superstack -> core KV): ``W_Q`` and ``W_O`` are
  ``d_s x d_s``; ``W_K`` and ``W_V`` map the *core* width, ``d_c x d_kv_s``.
* Spectral (Fourier) blocks: complex weights are counted as two real numbers.

Nothing in this module imports torch, so the multi-trillion configurations can
be costed on a laptop.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "CoreConfig",
    "SuperstackConfig",
    "RouterConfig",
    "CodecConfig",
    "IridiumConfig",
    "LADDER",
    "get_config",
    "load_config",
    "ParameterReport",
]

#: Discrete ids below this are control slots (BOS, EOS, SEP, PAD_CTRL and
#: headroom); a text token id is its vocabulary index plus this. Mirrors
#: ``training.tasks.TEXT_OFFSET``, which is the definition — duplicated here
#: only because nothing in this module may import torch, and ``tasks`` does.
#: ``tests/unit/test_config_inventory.py`` asserts the two agree, so the copy
#: cannot drift.
TEXT_ID_OFFSET = 16

KIB = 1024
MIB = 1024 ** 2
GIB = 1024 ** 3
TIB = 1024 ** 4


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


# --------------------------------------------------------------------------
# Leaf configurations
# --------------------------------------------------------------------------


def _deltanet_params(d: int, h: int, dh: int, conv: int) -> int:
    """Gated DeltaNet layer parameters; mirrors ``GatedDeltaNet.param_count``.

    Five bias-free ``d x d_qkv`` projections (q, k, v, output gate, out),
    two per-head gates (forget ``alpha`` and write strength ``beta``) with
    biases, three depthwise causal convs with biases, one RMSNorm.
    """
    d_qkv = h * dh
    return 5 * d * d_qkv + 2 * (d * h + h) + 3 * (d_qkv * conv + d_qkv) + d_qkv


@dataclass(frozen=True)
class CoreConfig:
    """The control stack: every token enters and leaves through these layers.

    ``n_layers`` is split into two stages at ``stage_split``. Stage I builds the
    representation the router reads; stage II integrates whatever the
    superstacks returned. Both stages own the canonical causal KV cache.
    """

    d_model: int
    n_layers: int
    n_query_heads: int
    n_kv_heads: int
    d_head: int
    d_ff: int
    stage_split: int | None = None      # default: n_layers // 2
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    # -- architecture options. Every default reproduces the original core. --
    #: Per-layer attention, repeated cyclically across the core. Entries:
    #: ``"global"`` (full causal GQA), ``"local"`` (sliding-window GQA of
    #: ``local_window`` tokens), ``"mla"`` (multi-head latent attention),
    #: ``"deltanet"`` (Gated DeltaNet linear attention, fixed-size state).
    #: ``("deltanet", "deltanet", "deltanet", "global")`` is the 3:1 hybrid
    #: Qwen3-Next and Kimi Linear use; ``("local",) * 5 + ("global",)`` is
    #: Gemma 3's. Empty means every layer is ``"global"``.
    layer_pattern: tuple[str, ...] = ()
    local_window: int = 4096
    #: MLA geometry. The cache holds ``mla_kv_rank + mla_rope_dim`` numbers
    #: per token per layer *regardless of head count*, which is the whole
    #: reason to use it: it is how many query heads become affordable. At the
    #: head counts the shipped rungs use, narrow GQA is already cheaper.
    mla_kv_rank: int = 512
    mla_q_rank: int = 0
    mla_rope_dim: int = 64
    deltanet_conv: int = 4
    #: ``"rms"`` (default), ``"dyt"`` or ``"derf"`` -- the normalisation-free
    #: elementwise replacements. Not ``"adaptive"``: a core block has no
    #: conditioning input to drive it.
    norm_kind: str = "rms"
    #: ``"sequential"`` (attention then FFN) or ``"parallel"`` (both from one
    #: normed input, PaLM style; one norm per block instead of two).
    block_kind: str = "sequential"
    #: Rank of the token-conditioned correction on every FFN projection; 0 off.
    ffn_dynamic_rank: int = 0
    #: mHC residual streams. 1 is the ordinary single residual stream; 4 is
    #: what DeepSeek-V4 uses. Streams live only inside one run of core layers
    #: -- expanded on entry, collapsed on exit -- so the router, superstacks,
    #: bridge and cache all keep seeing a single ``d_model`` state.
    hyper_streams: int = 1
    hyper_dynamic: bool = True
    hyper_sinkhorn_iters: int = 20

    def __post_init__(self) -> None:
        kinds = set(self.layer_pattern) or {"global"}
        unknown = kinds - {"global", "local", "mla", "deltanet"}
        if unknown:
            raise ConfigError(f"unknown layer kinds {sorted(unknown)}")
        if kinds & {"global", "local"} and self.d_model != self.n_query_heads * self.d_head:
            # Only a GQA layer needs its heads to tile d_model exactly. MLA's
            # per-head content width is independent of d_model, which is what
            # lets it run many heads without making each one narrow.
            raise ConfigError(
                f"d_model {self.d_model} != n_query_heads {self.n_query_heads} "
                f"* d_head {self.d_head}"
            )
        if self.n_query_heads % self.n_kv_heads:
            raise ConfigError("n_query_heads must be a multiple of n_kv_heads")
        if self.norm_kind not in ("rms", "dyt", "derf"):
            raise ConfigError(f"norm_kind must be rms, dyt or derf, got {self.norm_kind!r}")
        if self.block_kind not in ("sequential", "parallel"):
            raise ConfigError(f"block_kind must be sequential or parallel, got {self.block_kind!r}")
        if min(self.local_window, self.mla_kv_rank, self.hyper_streams,
               self.hyper_sinkhorn_iters, self.deltanet_conv) < 1:
            raise ConfigError("window, ranks, stream count and iteration counts must be positive")
        if self.mla_rope_dim % 2 or self.mla_q_rank < 0 or self.ffn_dynamic_rank < 0:
            raise ConfigError("mla_rope_dim must be even; ranks must be nonnegative")
        if self.n_layers < 2:
            raise ConfigError("the control core needs at least two layers")
        if self.split < 1 or self.split >= self.n_layers:
            raise ConfigError(
                f"stage_split {self.split} must lie in [1, n_layers - 1]"
            )

    @property
    def split(self) -> int:
        return self.n_layers // 2 if self.stage_split is None else self.stage_split

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.d_head

    def layer_kinds(self) -> tuple[str, ...]:
        """The attention kind of every core layer, in order."""
        pattern = self.layer_pattern or ("global",)
        return tuple(pattern[i % len(pattern)] for i in range(self.n_layers))

    def attention_params(self, kind: str) -> int:
        """One layer's attention module. Each branch restates the module's own
        ``param_count`` as arithmetic, because this file may not import torch;
        ``tests/unit/test_core_blocks.py`` asserts every branch equals the
        instantiated module, so a restated formula cannot drift."""
        d, h, dh = self.d_model, self.n_query_heads, self.d_head
        if kind in ("global", "local"):
            return 2 * d * d + 2 * d * self.d_kv
        if kind == "mla":
            kvr, qr, rr = self.mla_kv_rank, self.mla_q_rank, self.mla_rope_dim
            q = (d * qr + qr + qr * h * dh + qr * h * rr) if qr else (d * h * dh + d * h * rr)
            return q + (d * kvr + kvr + 2 * kvr * h * dh) + d * rr + h * dh * d
        if kind == "deltanet":
            return _deltanet_params(d, h, dh, self.deltanet_conv)
        raise ConfigError(f"unknown layer kind {kind!r}")

    def ffn_params(self) -> int:
        d, f, r = self.d_model, self.d_ff, self.ffn_dynamic_rank
        if not r:
            return 3 * d * f
        dyn = lambda i, o: i * o + i * r + r * o + i * r + r      # noqa: E731
        return 2 * dyn(d, f) + dyn(f, d)

    def norm_params(self) -> int:
        """Per block: two norms sequentially, one shared norm in parallel form."""
        one = {"rms": self.d_model, "dyt": 2 * self.d_model + 1,
               "derf": 2 * self.d_model + 2}[self.norm_kind]
        return one if self.block_kind == "parallel" else 2 * one

    def layer_params(self, kind: str) -> int:
        return self.attention_params(kind) + self.ffn_params() + self.norm_params()

    @property
    def params_per_layer(self) -> int:
        """A ``"global"`` layer. Kept for the callers that cost a homogeneous
        core; use :meth:`layer_params` or :attr:`layers_params` for a mixed one."""
        return self.layer_params("global")

    @property
    def layers_params(self) -> int:
        return sum(self.layer_params(k) for k in self.layer_kinds())

    @property
    def hyper_params(self) -> int:
        """mHC maps: per layer a stream mix and two stream vectors, plus one
        collapse. Zero with a single stream, where no module is built."""
        n, d = self.hyper_streams, self.d_model
        if n == 1:
            return 0
        dyn = self.hyper_dynamic
        vector = n + (d * n + n if dyn else 0)
        mix = n * n + (d * n * n + n * n if dyn else 0)
        return self.n_layers * (mix + 2 * vector) + vector

    @property
    def params(self) -> int:
        """Layers, output norm, loop re-injection gate, loop halting head.

        The gating parameters the *router* owns are counted in RouterConfig;
        the loop halting head reads the stage-II output and so belongs here.
        """
        return (
            self.layers_params
            + self.hyper_params
            + self.d_model          # output RMSNorm
            + self.d_model          # loop re-injection gate
            + self.d_model + 1      # loop halting head
        )

    def kv_bytes_per_token(self, bytes_per_element: int = 2) -> int:
        """Core cache: K and V, every layer, every recurrence slot.

        Recurrence multiplicity is applied by the caller (see
        :meth:`IridiumConfig.kv_bytes_per_token`) because it depends on the
        ponder budget, not on the core geometry.
        """
        return 2 * self.d_kv * bytes_per_element * self.n_layers


@dataclass(frozen=True)
class SuperstackConfig:
    """The domain banks. Deep, narrow-traffic, cross-attentive.

    A superstack is *not* a feed-forward expert. It is a deep transformer that
    attends to two things: the subset of the sequence that was also routed to
    it (stack-local sparse KV), and the control core's global KV through a
    bridge. §"KV coherence" in docs/architecture.md explains why that pair is
    exactly what makes incremental decoding equal teacher forcing.
    """

    n_stacks: int
    n_layers: int
    d_model: int
    n_query_heads: int
    n_kv_heads: int
    d_head: int
    d_ff: int
    cross_stride: int = 4          # a bridge cross-attention every N layers
    spectral_stride: int = 0       # 0 disables; else an FNO block every N layers
    spectral_modes: int = 8
    spectral_dims: int = 2         # spatial rank of the spectral operator
    spectral_channels: int = 64    # FNO channel width, NOT d_model
    spectral_stacks: tuple[int, ...] = ()   # which stacks carry FNO blocks
    min_depth: int = 2
    core_d_model: int = 0          # filled in by IridiumConfig
    io_projection: bool | None = None       # None -> only when widths differ
    specializations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.d_model != self.n_query_heads * self.d_head:
            raise ConfigError(
                f"superstack d_model {self.d_model} != "
                f"{self.n_query_heads} * {self.d_head}"
            )
        if self.n_query_heads % self.n_kv_heads:
            raise ConfigError("n_query_heads must be a multiple of n_kv_heads")
        if self.cross_stride < 1:
            raise ConfigError("cross_stride must be >= 1")
        if self.min_depth < 1 or self.min_depth > self.n_layers:
            raise ConfigError("min_depth must lie in [1, n_layers]")
        if self.specializations and len(self.specializations) != self.n_stacks:
            raise ConfigError(
                f"{len(self.specializations)} specializations for "
                f"{self.n_stacks} stacks"
            )
        if any(i < 0 or i >= self.n_stacks for i in self.spectral_stacks):
            raise ConfigError("spectral_stacks contains an out-of-range index")
        if self.spectral_stride > 0 and not self.spectral_stacks:
            raise ConfigError(
                "spectral_stride > 0 requires an explicit spectral_stacks list; "
                "spectral weights scale as C^2 * modes^2 and must not be given "
                "to every stack by accident"
            )

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.d_head

    @property
    def n_cross_layers(self) -> int:
        """Layers carrying a bridge. Layer indices 0, stride, 2*stride, ..."""
        return len(range(0, self.n_layers, self.cross_stride))

    @property
    def n_spectral_layers(self) -> int:
        """Spectral blocks *within a stack that has them*."""
        if self.spectral_stride < 1:
            return 0
        return len(range(0, self.n_layers, self.spectral_stride))

    def has_spectral(self, stack_index: int) -> bool:
        return self.spectral_stride > 0 and stack_index in self.spectral_stacks

    @property
    def n_spectral_channels(self) -> int:
        return self.spectral_channels

    @property
    def uses_io_projection(self) -> bool:
        if self.io_projection is not None:
            return self.io_projection
        return (self.core_d_model or self.d_model) != self.d_model

    @property
    def params_per_layer(self) -> int:
        """Self-attention + SwiGLU + norms, before optional blocks."""
        d, d_kv, d_ff = self.d_model, self.d_kv, self.d_ff
        return 2 * d * d + 2 * d * d_kv + 3 * d * d_ff + 2 * d

    @property
    def params_per_bridge(self) -> int:
        """Cross-attention onto the core cache, plus its norm.

        ``W_K``/``W_V`` read the *core* hidden width, which is why this is not
        simply another self-attention block.
        """
        d, d_kv = self.d_model, self.d_kv
        d_c = self.core_d_model or d
        return 2 * d * d + 2 * d_c * d_kv + d

    @property
    def params_per_spectral(self) -> int:
        """Lift, ``2**(dims-1)`` corner mode blocks, pointwise path, project.

        ``rfftn`` halves only the last axis, so a real ``d``-dimensional
        spectral kernel has ``2**(d-1)`` independent corners: 1 in 1-D, 2 in
        2-D, 4 in 3-D. Complex weights count as two reals. See the correction
        note in ``model/fno.py``.
        """
        c = self.n_spectral_channels
        m = self.spectral_modes
        blocks = 2 ** (self.spectral_dims - 1)
        lift = self.d_model * c + c
        project = c * self.d_model + self.d_model
        spectral = blocks * (c * c * m ** self.spectral_dims) * 2
        pointwise = c * c + c
        return lift + project + spectral + pointwise + self.d_model

    def params_for_stack(self, stack_index: int) -> int:
        base = self.n_layers * self.params_per_layer
        bridges = self.n_cross_layers * self.params_per_bridge
        spectral = (
            self.n_spectral_layers * self.params_per_spectral
            if self.has_spectral(stack_index)
            else 0
        )
        d_c = self.core_d_model or self.d_model
        io = (
            2 * d_c * self.d_model + self.d_model + d_c
            if self.uses_io_projection
            else 0
        )
        # Per-layer halting head for the depth ladder (PonderNet-in-depth),
        # the shared exit norm, and the scalar focus gain.
        halting = self.n_layers * (self.d_model + 1)
        exit_norm = self.d_model
        focus_gain = 1
        return base + bridges + spectral + io + halting + exit_norm + focus_gain

    @property
    def params_per_stack(self) -> int:
        """A stack with no spectral blocks. Use ``params_for_stack`` per index."""
        return self.params_for_stack(
            next((i for i in range(self.n_stacks) if not self.has_spectral(i)), 0)
        )

    @property
    def params(self) -> int:
        return sum(self.params_for_stack(i) for i in range(self.n_stacks))

    def kv_bytes_per_token(self, bytes_per_element: int = 2) -> int:
        """Stack-local cache for one token, in the one stack it visited."""
        return 2 * self.d_kv * bytes_per_element * self.n_layers


@dataclass(frozen=True)
class RouterConfig:
    """Macro-router, focus head and ponder controller.

    ``top_k`` superstacks receive each token. ``max_loops`` bounds how many
    times a token may re-enter the routing matrix. Every decision here is
    computed from a *causal* state: see the note in ``model/router.py`` about
    why a chunk-mean focus statistic is not admissible.
    """

    top_k: int = 2
    balance_alpha: float = 1e-2
    z_alpha: float = 1e-3
    max_loops: int = 3
    loop_entry: int = 0            # core layer index re-entered on a ponder loop
    ponder_prior_p_stop: float = 0.4
    ponder_beta: float = 1e-2
    depth_prior_p_stop: float = 0.25
    depth_beta: float = 1e-2
    capacity_factor: float = 0.0   # 0 disables token dropping (see docs)
    router_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigError("top_k must be >= 1")
        if self.max_loops < 1:
            raise ConfigError("max_loops must be >= 1")
        if not 0.0 < self.ponder_prior_p_stop <= 1.0:
            raise ConfigError("ponder_prior_p_stop must lie in (0, 1]")

    def params(self, d_model: int, n_stacks: int) -> int:
        gate = d_model * n_stacks
        focus = d_model + 1
        halt = d_model + 1
        loop_embed = (self.max_loops + 1) * d_model
        norm = d_model
        return gate + focus + halt + loop_embed + norm


@dataclass(frozen=True)
class CodecConfig:
    """Input and output codec sizing.

    Codecs are Iridium-1 components, trained inside the same checkpoint. They
    are listed separately only so that the parameter report can say how much of
    the model is perception and emission rather than reasoning.
    """

    vocab_size: int = 384
    image_patch: int = 8
    image_channels: int = 3
    video_patch_t: int = 2
    audio_mels: int = 64
    audio_frames: int = 4
    field_patch: int = 8
    field_channels: int = 4
    point_features: int = 10
    action_ops: int = 24
    action_scalars: int = 6
    quantity_roles: int = 16
    max_position: int = 8192
    n_modalities: int = 9
    tie_text_embedding: bool = True
    continuous_head: str = "flow"      # "flow" (CFM) or "regression" (MSE)
    flow_tau_features: int = 64
    #: How the flow head reads the model's hidden state: ``"add"`` projects it
    #: and adds it into the mixer, ``"adaln"`` uses it to modulate a LayerNorm.
    #: Adaptive normalisation is what DiT and SD3 use and is the stronger of the
    #: two at matched width; ``"add"`` remains the default because changing it
    #: changes the parameter set, and a checkpoint is not portable across that.
    continuous_conditioning: str = "add"
    #: Timestep distribution for the flow objective. ``"logit_normal"`` (SD3)
    #: concentrates training signal on the hard middle of the trajectory rather
    #: than spreading it uniformly over the easy ends. Costs no parameters, so
    #: unlike ``continuous_conditioning`` it can be changed between runs freely
    #: -- though not mid-run, since it changes what the loss is an average over.
    flow_timestep_sampling: str = "uniform"
    #: Hashed n-gram input embeddings (Over-Tokenized Transformer, arXiv
    #: 2501.16975): each text token also adds embeddings of the 2-gram and
    #: 3-gram ending at it, hashed into tables of this many rows. The paper's
    #: finding is that loss falls log-linearly in input-vocabulary size, and a
    #: hashed table grows that vocabulary without growing the softmax. 0 off.
    #: Costs ``len(ngram_orders) * ngram_table_size * d`` parameters and almost
    #: no FLOPs -- it is a lookup.
    ngram_table_size: int = 0
    ngram_orders: tuple[int, ...] = (2, 3)
    #: Width of a camera token (6 = one Plucker ray per patch). 0 disables the
    #: camera modality entirely; enabling it also needs ``n_modalities >= 10``
    #: so the modality embedding and slot head have a row for it. Input-only:
    #: costs an encoder, never a decoder. See ``iridium.world.tokens``.
    camera_features: int = 0

    def __post_init__(self) -> None:
        # Validated here rather than in the module, because an unknown value
        # would otherwise be caught only when torch builds the head -- by which
        # point the parameter report has already been printed, and it would be
        # wrong. The accounting and the modules have to agree about what these
        # strings mean or the ladder stops being costable.
        if self.continuous_head not in ("flow", "regression"):
            raise ConfigError(
                f"continuous_head must be 'flow' or 'regression', "
                f"got {self.continuous_head!r}"
            )
        if self.continuous_conditioning not in ("add", "adaln"):
            raise ConfigError(
                f"continuous_conditioning must be 'add' or 'adaln', "
                f"got {self.continuous_conditioning!r}"
            )
        if self.camera_features < 0:
            raise ConfigError("camera_features must be >= 0")
        if self.camera_features and self.n_modalities < 10:
            raise ConfigError(
                "camera_features needs n_modalities >= 10: the camera modality "
                "is index 9, appended after the original nine"
            )
        if self.ngram_table_size < 0 or any(n < 2 for n in self.ngram_orders):
            raise ConfigError("ngram_table_size must be >= 0 and every order >= 2")
        if self.flow_timestep_sampling not in ("uniform", "logit_normal"):
            raise ConfigError(
                f"flow_timestep_sampling must be 'uniform' or 'logit_normal', "
                f"got {self.flow_timestep_sampling!r}"
            )

    def continuous_dims(self) -> dict[str, int]:
        return {
            "image": self.image_channels * self.image_patch ** 2,
            "video": self.image_channels * self.video_patch_t * self.image_patch ** 2,
            "audio": self.audio_mels * self.audio_frames,
            "field": self.field_channels * self.field_patch ** 2,
            "geometry": self.point_features,
            "quantity": 3 + self.quantity_roles,
        }

    def _flow_head_params(self, d: int, d_out: int) -> int:
        """FlowMatchingHead: norm, tau/x/condition projections, mixer, output.

        ``null_cond`` is the learned "no conditioning" vector classifier-free
        guidance substitutes for the real conditioning during its unconditional
        pass. It is allocated unconditionally rather than only when
        ``cfg_dropout > 0``, and that is the right call even though it costs a
        vector per head on a run that never uses guidance: making it
        conditional would mean a checkpoint trained without dropout has a
        different parameter *set* from one trained with it, so enabling
        guidance later could not load the earlier weights. A vector of length
        ``d`` per continuous modality is a rounding error against the head it
        sits in; a checkpoint that will not load is not.

        ``adaln`` conditioning replaces the additive conditioning path with
        LayerNorm modulation (scale and shift predicted from the conditioning),
        which is what DiT and SD3 use and is the stronger of the two at matched
        width. The normalisation itself is affine-free, so the cost is exactly
        the modulation projection.
        """
        d_h = max(d_out * 2, d)
        n_tau = self.flow_tau_features
        adaln = (d_h * 2 * d_h + 2 * d_h) if self.continuous_conditioning == "adaln" else 0
        return (
            d                                   # RMSNorm
            + 2 * n_tau * d_h + d_h             # tau features
            + d_out * d_h + d_h                 # x_tau
            + d * d_h + d_h                     # conditioning
            + d_h * d_h + d_h                   # mixer
            + d_h * d_out + d_out               # velocity output
            + d                                 # null_cond (see above)
            + adaln                             # ada_mod: Linear(d_h, 2*d_h)
        )

    def params(self, d: int) -> dict[str, int]:
        dims = self.continuous_dims()
        out: dict[str, int] = {
            "text_embedding": self.vocab_size * d,
            "modality_embedding": self.n_modalities * d,
        }
        for name, dim in dims.items():
            out[f"{name}_encoder"] = dim * d + d
            if name == "quantity":
                # A quantity decodes to one number - its log magnitude - not to
                # a patch, and it is a regression rather than a flow: there is
                # no distribution to sample here, there is an answer.
                out["quantity_decoder"] = d + d * 1 + 1
                continue
            out[f"{name}_decoder"] = (
                self._flow_head_params(d, dim)
                if self.continuous_head == "flow"
                else d + d * dim + dim          # RMSNorm + Linear
            )
        out["action_encoder"] = (self.action_ops + self.action_scalars) * d + d
        out["action_head"] = (
            d
            + d * self.action_ops + self.action_ops
            + d * self.action_scalars + self.action_scalars
        )
        out["slot_type_head"] = d + d * self.n_modalities + self.n_modalities
        out["confidence_head"] = d + d + 1
        if self.camera_features:
            out["camera_encoder"] = self.camera_features * d + d
        if self.ngram_table_size:
            out["ngram_embedding"] = len(self.ngram_orders) * self.ngram_table_size * d
        out["text_head"] = (
            d if self.tie_text_embedding
            else d + d * self.vocab_size + self.vocab_size
        )
        return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterReport:
    """A parameter breakdown that can be printed, tested and diffed."""

    name: str
    parts: Mapping[str, int]
    active_min: int
    active_max: int

    @property
    def total(self) -> int:
        return sum(self.parts.values())

    def bytes_at(self, bits_per_param: float) -> float:
        return self.total * bits_per_param / 8.0

    def render(self) -> str:
        width = max(len(k) for k in self.parts)
        lines = [f"{self.name}", "-" * (width + 22)]
        for key, value in self.parts.items():
            lines.append(f"  {key:<{width}}  {value:>18,}")
        lines.append("-" * (width + 22))
        lines.append(f"  {'TOTAL':<{width}}  {self.total:>18,}")
        lines.append(
            f"  {'active / token (min)':<{width}}  {self.active_min:>18,}"
        )
        lines.append(
            f"  {'active / token (max)':<{width}}  {self.active_max:>18,}"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Whole-model configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IridiumConfig:
    name: str
    core: CoreConfig
    stacks: SuperstackConfig
    router: RouterConfig = field(default_factory=RouterConfig)
    codecs: CodecConfig = field(default_factory=CodecConfig)
    max_seq_len: int = 4096
    dropout: float = 0.0
    gated_bank: bool = False
    loop_identity: bool = False
    qk_norm: bool = False  # opt-in head RMS normalization before rotary attention
    controller_mode: bool = False
    memory_slots: int = 0
    memory_stride: int = 32
    memory_rank: int = 64
    perception_layers: int = 0
    perception_rank: int = 64
    #: Subword vocabulary the text pipeline trains and uses. 0 keeps the
    #: byte-level path, where a text id is a byte plus a fixed offset.
    #:
    #: This is the single most consequential setting on this list for how the
    #: model reads, and the reason is arithmetic rather than taste. Byte-level
    #: text costs about one token per character; a subword vocabulary of this
    #: size costs roughly one per four. At a fixed ``max_seq_len`` and a fixed
    #: step budget that is a ~4x difference in how much *text* the model sees,
    #: and a ~4x difference in how far back its context actually reaches — and
    #: it spends the difference learning orthography, which is not the thing
    #: anyone wants a small model's capacity spent on. Byte-level's real
    #: advantages (nothing to go stale, no vocabulary/corpus mismatch, lossless
    #: on arbitrary input) are genuine and are nowhere near worth that at a
    #: scale where capacity is the binding constraint.
    #:
    #: Must leave room for the control offset: ``codecs.vocab_size`` has to be
    #: at least ``text_vocab_size + 16``. Checked in ``__post_init__``.
    text_vocab_size: int = 0
    #: Where the trained tokenizer artifact is cached. Recorded in the run
    #: manifest so a checkpoint can say which vocabulary produced it — a model
    #: served with a different tokenizer than it was trained with does not fail,
    #: it just produces confident nonsense, which is the worst way to fail.
    text_tokenizer_cache: str = "artifacts/tokenizers"
    #: Per-layer embeddings (Gemma 3n): every core layer also receives a small
    #: per-token embedding, looked up from the token id and projected into
    #: that layer's input. Capacity that lives in a lookup table -- almost no
    #: FLOPs, and offloadable from accelerator memory -- at a cost of
    #: ``vocab * n_layers * ple_dim + n_layers * ple_dim * d`` parameters.
    #: The projections start at zero, so enabling it is function-preserving.
    #: 0 disables.
    ple_dim: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.perception_layers < 0 or self.perception_rank < 1:
            raise ConfigError("invalid perceptual encoder dimensions")
        if self.memory_slots < 0 or self.memory_stride < 1 or self.memory_rank < 1:
            raise ConfigError("invalid context memory dimensions")
        if self.ple_dim < 0:
            raise ConfigError("ple_dim must be nonnegative (0 = off)")
        if self.text_vocab_size < 0:
            raise ConfigError("text_vocab_size must be nonnegative (0 = byte level)")
        if self.text_vocab_size:
            # Text ids are shifted past the control tokens (see TEXT_OFFSET in
            # training.tasks). Getting this wrong does not raise anywhere: the
            # high-id tokens simply index past the embedding table, or worse,
            # wrap into a control id and the model emits an EOS in the middle of
            # a word. Caught here, where the number is still a config value.
            if self.codecs.vocab_size < self.text_vocab_size + TEXT_ID_OFFSET:
                raise ConfigError(
                    f"codecs.vocab_size {self.codecs.vocab_size} cannot hold a "
                    f"{self.text_vocab_size}-token vocabulary plus the "
                    f"{TEXT_ID_OFFSET} reserved control ids; it needs at least "
                    f"{self.text_vocab_size + TEXT_ID_OFFSET}"
                )
        if self.controller_mode and self.gated_bank:
            raise ConfigError("controller_mode integrates in the core; disable gated_bank")
        if self.stacks.core_d_model == 0:
            object.__setattr__(
                self, "stacks", replace(self.stacks, core_d_model=self.core.d_model)
            )
        if self.stacks.core_d_model != self.core.d_model:
            raise ConfigError(
                "superstack.core_d_model must equal core.d_model "
                f"({self.stacks.core_d_model} != {self.core.d_model})"
            )
        if self.router.top_k > self.stacks.n_stacks:
            raise ConfigError("top_k exceeds the number of superstacks")
        if self.router.loop_entry >= self.core.n_layers:
            raise ConfigError("loop_entry must be a valid core layer index")

    # -- parameters ------------------------------------------------------

    def parameter_parts(self) -> dict[str, int]:
        d = self.core.d_model
        parts = {
            "control_core": self.core.params,
            "superstacks": self.stacks.params,
            "router": self.router.params(d, self.stacks.n_stacks),
        }
        if self.controller_mode:
            parts["controller_dispatch"] = d + 1
        if self.memory_slots:
            parts["context_memory"] = 4 * d * self.memory_rank + self.memory_rank + d + 2
        if self.ple_dim:
            parts["per_layer_embedding"] = (self.codecs.vocab_size * self.core.n_layers * self.ple_dim
                                            + self.core.n_layers * self.ple_dim * d)
        if self.perception_layers:
            parts["perceptual_encoders"] = (len(self.codecs.continuous_dims()) + bool(self.codecs.camera_features)) * self.perception_layers * (2 * d * self.perception_rank + self.perception_rank + 3 * d)
        if self.gated_bank:
            parts["bank_gate"] = d
        parts.update(
            {f"codec.{k}": v for k, v in self.codecs.params(d).items()}
        )
        return parts

    def _stack_cost(self, depth: int, spectral: bool) -> int:
        st = self.stacks
        n_cross = len(range(0, depth, st.cross_stride))
        n_spec = (
            len(range(0, depth, st.spectral_stride))
            if (spectral and st.spectral_stride > 0)
            else 0
        )
        io = (
            2 * self.core.d_model * st.d_model if st.uses_io_projection else 0
        )
        return (
            depth * st.params_per_layer
            + n_cross * st.params_per_bridge
            + n_spec * st.params_per_spectral
            + io
        )

    def active_parameters(self) -> tuple[int, int]:
        """*Distinct* parameters touched by one token, at min and max focus.

        This is deliberately not the same quantity as FLOPs. A core layer
        executed three times by the ponder loop touches one set of parameters
        and costs three passes. Conflating the two is how a specification ends
        up quoting an "active parameter" number that no memory system has to
        hold. Use :meth:`flops_per_token` for cost.

        Codecs are excluded: a text token does not run the video decoder.

        The upper bound sums the ``top_k`` *most expensive actual stacks*, not
        ``top_k`` copies of the most expensive one. Only some stacks carry
        spectral blocks, and assuming every routed stack does can push the
        reported "active" count above the model's total parameter count — which
        is how a specification ends up quoting an active figure that is
        arithmetically impossible.
        """
        core = self.core.params
        k = min(self.router.top_k, self.stacks.n_stacks)
        costs = sorted(
            (
                self._stack_cost(
                    self.stacks.n_layers, spectral=self.stacks.has_spectral(i)
                )
                for i in range(self.stacks.n_stacks)
            ),
            reverse=True,
        )
        lo = core + k * self._stack_cost(self.stacks.min_depth, spectral=False)
        hi = core + sum(costs[:k])
        if self.controller_mode:
            extras = self.core.d_model + 1
            if self.memory_slots:
                extras += 4 * self.core.d_model * self.memory_rank + self.memory_rank + self.core.d_model + 2
            lo = core + extras
            hi = core + extras + sum(costs[:min(len(costs), k * (self.router.max_loops - 1))])
        return lo, hi

    def flops_per_token(self) -> tuple[int, int]:
        """Forward multiply-accumulate FLOPs for one token, min and max.

        ``2 * params`` per matmul parameter, counting every repetition: the
        ponder loop multiplies the core, the focus ladder scales the stacks.
        Attention score/​value FLOPs (the ``O(T)`` part) are excluded; they
        depend on context length and are reported by the parallel planner.
        """
        core = self.core.params
        k = min(self.router.top_k, self.stacks.n_stacks)
        costs = sorted(
            (
                self._stack_cost(
                    self.stacks.n_layers, spectral=self.stacks.has_spectral(i)
                )
                for i in range(self.stacks.n_stacks)
            ),
            reverse=True,
        )
        lo = 2 * (core + k * self._stack_cost(self.stacks.min_depth, spectral=False))
        hi = 2 * self.router.max_loops * (core + sum(costs[:k]))
        return lo, hi

    def report(self) -> ParameterReport:
        lo, hi = self.active_parameters()
        return ParameterReport(
            name=self.name, parts=self.parameter_parts(), active_min=lo, active_max=hi
        )

    @property
    def n_params(self) -> int:
        return sum(self.parameter_parts().values())

    # -- memory ----------------------------------------------------------

    def kv_bytes_per_token(
        self, loops: int = 1, bytes_per_element: int = 2
    ) -> int:
        """Worst-case cache for one token.

        A token occupies ``loops`` slots in the core cache (a layer run twice
        is two cache entries, not one) and ``top_k`` stack-local slots. The
        stack entries are what a naive analysis forgets.
        """
        core = loops * self.core.kv_bytes_per_token(bytes_per_element)
        stack = self.router.top_k * self.stacks.kv_bytes_per_token(bytes_per_element)
        return core + stack

    def weight_bytes(self, bits_per_param: float = 16.0) -> float:
        return self.n_params * bits_per_param / 8.0

    def training_state_bytes(self, bytes_per_param: float = 16.0) -> float:
        """BF16 weights + BF16 grads + FP32 master + two FP32 moments."""
        return self.n_params * bytes_per_param

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "core": asdict(self.core),
            "stacks": asdict(self.stacks),
            "router": asdict(self.router),
            "codecs": asdict(self.codecs),
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "controller_mode": self.controller_mode,
            "memory_slots": self.memory_slots,
            "memory_stride": self.memory_stride,
            "memory_rank": self.memory_rank,
            "perception_layers": self.perception_layers,
            "perception_rank": self.perception_rank,
            "qk_norm": self.qk_norm,
            "gated_bank": self.gated_bank,
            "loop_identity": self.loop_identity,
            # The vocabulary a checkpoint was trained with. Omitting these made a
            # BPE-trained config reload as byte-level -- a tokenizer mismatch,
            # which does not raise; it produces fluent nonsense.
            "text_vocab_size": self.text_vocab_size,
            "text_tokenizer_cache": self.text_tokenizer_cache,
            "ple_dim": self.ple_dim,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IridiumConfig":
        data = dict(data)
        stacks = dict(data["stacks"])
        spec = stacks.get("specializations") or ()
        stacks["specializations"] = tuple(spec)
        core = dict(data["core"])
        # JSON has no tuples; a list here would make the frozen config unhashable.
        core["layer_pattern"] = tuple(core.get("layer_pattern") or ())
        return cls(
            name=data["name"],
            core=CoreConfig(**core),
            stacks=SuperstackConfig(**stacks),
            router=RouterConfig(**data.get("router", {})),
            codecs=CodecConfig(**{**data.get("codecs", {}),
                                  **({"ngram_orders": tuple(data["codecs"]["ngram_orders"])}
                                     if "ngram_orders" in data.get("codecs", {}) else {})}),
            max_seq_len=data.get("max_seq_len", 4096),
            dropout=data.get("dropout", 0.0),
            controller_mode=data.get("controller_mode", False),
            memory_slots=data.get("memory_slots", 0),
            memory_stride=data.get("memory_stride", 32),
            memory_rank=data.get("memory_rank", 64),
            perception_layers=data.get("perception_layers", 0),
            perception_rank=data.get("perception_rank", 64),
            qk_norm=data.get("qk_norm", False),
            gated_bank=data.get("gated_bank", False),
            loop_identity=data.get("loop_identity", False),
            text_vocab_size=data.get("text_vocab_size", 0),
            text_tokenizer_cache=data.get("text_tokenizer_cache", "artifacts/tokenizers"),
            ple_dim=data.get("ple_dim", 0),
            notes=data.get("notes", ""),
        )


def load_config(path: str | Path) -> IridiumConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return IridiumConfig.from_dict(yaml.safe_load(handle))


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------

SPECIALIZATIONS_32: tuple[str, ...] = (
    "fluid_dynamics_navier_stokes",
    "electrodynamics_plasma",
    "astrophysics_relativity_nbody",
    "quantum_chemistry_molecular",
    "solid_mechanics_elastodynamics",
    "cellular_biology_genomics",
    "symbolic_mathematics_pde",
    "numerical_analysis_solvers",
    "compilers_systems_code",
    "software_architecture_api",
    "ui_dom_os_actuation",
    "geometry_cad_blender",
    "photorealistic_video_spatiotemporal",
    "image_composition_raster",
    "audio_spectral_synthesis",
    "speech_prosody_language",
    "natural_language_pragmatics",
    "multilingual_translation",
    "long_form_reasoning_planning",
    "formal_logic_proof",
    "statistics_probability_inference",
    "optimization_control",
    "signal_processing_estimation",
    "materials_thermodynamics",
    "earth_climate_ocean",
    "medicine_physiology",
    "economics_operations_research",
    "law_policy_structure",
    "retrieval_citation_grounding",
    "uncertainty_calibration_critique",
    "tool_protocol_orchestration",
    "safety_refusal_boundary",
)


def _ladder() -> dict[str, IridiumConfig]:
    rungs: dict[str, IridiumConfig] = {}

    # -- tiny: unit-test scale. Everything runs in milliseconds. ----------
    rungs["tiny"] = IridiumConfig(
        name="iridium-1-tiny",
        core=CoreConfig(
            d_model=64, n_layers=4, n_query_heads=4, n_kv_heads=2, d_head=16, d_ff=176
        ),
        stacks=SuperstackConfig(
            n_stacks=2,
            n_layers=4,
            d_model=64,
            n_query_heads=4,
            n_kv_heads=2,
            d_head=16,
            d_ff=176,
            cross_stride=2,
            spectral_stride=4,
            spectral_modes=4,
            spectral_channels=8,
            spectral_stacks=(0,),
            min_depth=1,
            specializations=("fluid_dynamics_navier_stokes", "natural_language_pragmatics"),
        ),
        router=RouterConfig(top_k=1, max_loops=2),
        codecs=CodecConfig(vocab_size=320, image_patch=4, field_patch=4),
        max_seq_len=256,
        notes="Unit-test rung. Exercises every mechanism at negligible cost.",
    )

    # -- nano: the trainable demonstration. CPU-feasible. -----------------
    rungs["nano"] = IridiumConfig(
        name="iridium-1-nano",
        core=CoreConfig(
            d_model=256, n_layers=8, n_query_heads=8, n_kv_heads=2, d_head=32, d_ff=704
        ),
        stacks=SuperstackConfig(
            n_stacks=4,
            n_layers=8,
            d_model=256,
            n_query_heads=8,
            n_kv_heads=2,
            d_head=32,
            d_ff=704,
            cross_stride=4,
            spectral_stride=4,
            spectral_modes=8,
            spectral_channels=32,
            spectral_stacks=(0,),
            min_depth=2,
            specializations=(
                "fluid_dynamics_navier_stokes",
                "symbolic_mathematics_pde",
                "natural_language_pragmatics",
                "ui_dom_os_actuation",
            ),
        ),
        router=RouterConfig(top_k=2, max_loops=2),
        codecs=CodecConfig(vocab_size=384),
        max_seq_len=1024,
        notes="Trained rung. Every claim in docs/results.md comes from here.",
    )

    # -- test1b: the smallest rung with the *full* stack geometry. --------
    # Five general-purpose superstacks rather than 32 domain ones, a core
    # narrow enough to instantiate on a laptop, and every mechanism of the
    # base configuration present: two-stage core, top-2 macro-routing, the
    # bridge, the focus ladder, the ponder loop, spectral blocks in the
    # science stack. This is the rung to point a smoke test at when the
    # question is "does the architecture hold together at a real size".
    rungs["test1b"] = IridiumConfig(
        name="iridium-1-test1b",
        core=CoreConfig(
            d_model=768, n_layers=16, n_query_heads=12, n_kv_heads=4,
            d_head=64, d_ff=2048,
        ),
        stacks=SuperstackConfig(
            n_stacks=5,
            n_layers=26,            # deeper than the core, as the design requires
            d_model=768,
            n_query_heads=12,
            n_kv_heads=4,
            d_head=64,
            d_ff=2048,
            cross_stride=8,
            spectral_stride=8,
            spectral_modes=16,
            spectral_channels=64,
            spectral_stacks=(0,),
            min_depth=4,
            specializations=(
                "science_physics_simulation",
                "mathematics_symbolic_proof",
                "code_systems_tools",
                "language_reasoning_intent",
                "perception_geometry_action",
            ),
        ),
        router=RouterConfig(top_k=2, max_loops=3),
        codecs=CodecConfig(vocab_size=32_768),
        max_seq_len=4096,
        notes=(
            "1.00 B test rung: 5 general superstacks, full mechanism set. "
            "Instantiates and runs on CPU; fp32 Adam does not fit in 15 GB, "
            "so it is trained through LoRA adapters."
        ),
    )

    # -- nano100m: ~0.1 B, the largest rung that trains on a laptop CPU. ---
    # Five general-purpose superstacks and the full mechanism set, at a width
    # chosen so a CPU-only box can reach convergence rather than merely
    # execute a forward pass.
    rungs["nano100m"] = IridiumConfig(
        name="iridium-1-nano100m",
        core=CoreConfig(
            d_model=512, n_layers=6, n_query_heads=8, n_kv_heads=2,
            d_head=64, d_ff=1152,
        ),
        stacks=SuperstackConfig(
            n_stacks=5,
            n_layers=6,
            d_model=512,
            n_query_heads=8,
            n_kv_heads=2,
            d_head=64,
            d_ff=1152,
            cross_stride=4,
            spectral_stride=4,
            spectral_modes=8,
            spectral_channels=32,
            spectral_stacks=(0,),
            min_depth=2,
            specializations=(
                "science_physics_simulation",
                "mathematics_symbolic_proof",
                "code_systems_tools",
                "language_reasoning_intent",
                "perception_geometry_action",
            ),
        ),
        router=RouterConfig(top_k=2, max_loops=3),
        codecs=CodecConfig(vocab_size=8192),
        max_seq_len=2048,
        notes=(
            "~0.1 B trainable-on-CPU rung. The stacks are the same depth as "
            "the core rather than deeper: at this width the depth ratio buys "
            "less than the steps it costs, and converging is the point."
        ),
    )

    # -- modern: the ~0.7 B rung with the recommended architecture options. --
    # What this session's options look like assembled, choosing only those
    # with published evidence at scale, and leaving out the ones without:
    #
    # * core layers 3 Gated DeltaNet : 1 latent attention -- the Qwen3-Next /
    #   Kimi Linear ratio. DeltaNet's decode state is fixed-size, so the core's
    #   cache stops growing with context on three layers in four;
    # * mHC with 4 residual streams (DeepSeek-V4), QK-norm for low-precision
    #   stability, a 32k subword vocabulary, hashed n-gram input embeddings,
    #   SD3-style flow heads (AdaLN conditioning, logit-normal timesteps), the
    #   camera modality and 14-wide splat geometry for the world model.
    #
    # Left out on purpose: DyT/Derf (the DyT paper reports its alpha is
    # sensitive for LLMs), parallel blocks (PaLM measured a small loss at 8B),
    # dynamic FFN weights (no evidence at scale), per-layer embeddings (at a
    # 32k vocabulary they would be 33 M parameters of table on a 0.7 B model).
    #
    # UNTRAINED. The superstacks keep ordinary attention, so at long context
    # their KV cache dominates (docs/long-context.md); max_seq_len is what the
    # architecture supports, not what anything has been trained to use.
    rungs["modern"] = IridiumConfig(
        name="iridium-1-modern",
        core=CoreConfig(
            d_model=1024, n_layers=16, n_query_heads=16, n_kv_heads=4,
            d_head=64, d_ff=2816,
            layer_pattern=("deltanet", "deltanet", "deltanet", "mla"),
            mla_kv_rank=256, mla_rope_dim=64, hyper_streams=4,
        ),
        stacks=SuperstackConfig(
            n_stacks=5,
            n_layers=6,
            d_model=1024,
            n_query_heads=16,
            n_kv_heads=4,
            d_head=64,
            d_ff=2816,
            cross_stride=3,
            spectral_stride=6,
            spectral_modes=12,
            spectral_channels=48,
            spectral_stacks=(0,),
            min_depth=2,
            specializations=(
                "science_physics_simulation",
                "mathematics_symbolic_proof",
                "code_systems_tools",
                "language_reasoning_intent",
                "perception_geometry_action",
            ),
        ),
        router=RouterConfig(top_k=2, max_loops=3),
        codecs=CodecConfig(
            vocab_size=32_768 + TEXT_ID_OFFSET, point_features=14,
            continuous_conditioning="adaln", flow_timestep_sampling="logit_normal",
            ngram_table_size=16_384, camera_features=6, n_modalities=10,
        ),
        max_seq_len=32_768,
        qk_norm=True,
        text_vocab_size=32_768,
        notes=(
            "Recommended-options rung, untrained: 3:1 DeltaNet:MLA core, mHC x4, "
            "32k BPE, n-gram embeddings, SD3-style flow heads, camera modality."
        ),
    )

    # -- micro: one modern accelerator. -----------------------------------
    rungs["micro"] = IridiumConfig(
        name="iridium-1-micro",
        core=CoreConfig(
            d_model=768, n_layers=16, n_query_heads=12, n_kv_heads=4, d_head=64, d_ff=2048
        ),
        stacks=SuperstackConfig(
            n_stacks=8,
            n_layers=16,
            d_model=768,
            n_query_heads=12,
            n_kv_heads=4,
            d_head=64,
            d_ff=2048,
            cross_stride=4,
            spectral_stride=8,
            spectral_modes=12,
            spectral_channels=64,
            spectral_stacks=(0,),
            min_depth=4,
            specializations=SPECIALIZATIONS_32[:8],
        ),
        router=RouterConfig(top_k=2, max_loops=3),
        codecs=CodecConfig(vocab_size=32_768),
        max_seq_len=8192,
        notes="Single-accelerator rung: the first scale where routing can pay.",
    )

    # -- small: a real cluster job. ---------------------------------------
    rungs["small"] = IridiumConfig(
        name="iridium-1-small",
        core=CoreConfig(
            d_model=2048, n_layers=24, n_query_heads=16, n_kv_heads=4, d_head=128, d_ff=5632
        ),
        stacks=SuperstackConfig(
            n_stacks=16,
            n_layers=32,
            d_model=2048,
            n_query_heads=16,
            n_kv_heads=4,
            d_head=128,
            d_ff=5632,
            cross_stride=4,
            spectral_stride=8,
            spectral_modes=16,
            spectral_channels=96,
            spectral_stacks=(0, 1),
            min_depth=4,
            specializations=SPECIALIZATIONS_32[:16],
        ),
        router=RouterConfig(top_k=2, max_loops=3),
        codecs=CodecConfig(vocab_size=65_536),
        max_seq_len=32_768,
        notes="Multi-node rung: the first scale that needs the parallel planner.",
    )

    # -- base: the specified 8.8 T target. --------------------------------
    rungs["base"] = IridiumConfig(
        name="iridium-1-base",
        core=CoreConfig(
            d_model=14_336,
            n_layers=64,
            n_query_heads=112,
            n_kv_heads=14,
            d_head=128,
            d_ff=38_400,
        ),
        stacks=SuperstackConfig(
            n_stacks=32,
            n_layers=128,
            d_model=14_336,
            n_query_heads=112,
            n_kv_heads=14,
            d_head=128,
            d_ff=38_400,
            cross_stride=8,
            spectral_stride=16,
            spectral_modes=32,
            spectral_channels=128,
            spectral_stacks=(0, 1, 2, 3, 4, 23, 24),
            min_depth=16,
            specializations=SPECIALIZATIONS_32,
        ),
        router=RouterConfig(top_k=2, max_loops=3),
        codecs=CodecConfig(vocab_size=128_000, image_patch=16, field_patch=16),
        max_seq_len=1_048_576,
        notes="The target configuration. Costed, not built.",
    )

    # -- extreme: the corrected 38.4 T target. ----------------------------
    # The source plan paired N_S=48, L_S=196, d=18432 with a stated 38.41 T
    # total. Those are inconsistent: 48 stacks of that geometry total 32.9 T.
    # 56 stacks reproduce the stated headline. See docs/verification.md F-01.
    rungs["extreme"] = IridiumConfig(
        name="iridium-1-extreme",
        core=CoreConfig(
            d_model=18_432,
            n_layers=64,
            n_query_heads=144,
            n_kv_heads=18,
            d_head=128,
            d_ff=49_152,
        ),
        stacks=SuperstackConfig(
            n_stacks=56,
            n_layers=196,
            d_model=18_432,
            n_query_heads=144,
            n_kv_heads=18,
            d_head=128,
            d_ff=49_152,
            cross_stride=8,
            spectral_stride=16,
            spectral_modes=32,
            spectral_channels=160,
            spectral_stacks=(0, 1, 2, 3, 4, 23, 24),
            min_depth=16,
            specializations=(),
        ),
        router=RouterConfig(top_k=2, max_loops=3),
        codecs=CodecConfig(vocab_size=128_000, image_patch=16, field_patch=16),
        max_seq_len=1_048_576,
        notes="Corrected extreme rung: 56 stacks, not 48, to reach ~38 T.",
    )
    return rungs


LADDER: dict[str, IridiumConfig] = _ladder()


def get_config(name: str) -> IridiumConfig:
    key = name.replace("iridium-1-", "")
    if key not in LADDER:
        raise ConfigError(
            f"unknown rung {name!r}; known rungs: {sorted(LADDER)}"
        )
    return LADDER[key]


def ladder_table() -> str:
    """One-screen summary of the whole ladder."""
    header = (
        f"{'rung':<10} {'total params':>18} {'active/tok max':>16} "
        f"{'GFLOP/tok max':>13} {'BF16 wts':>10} {'MXFP4 wts':>10} {'KV/token':>10}"
    )
    lines = [header, "-" * len(header)]
    for key, cfg in LADDER.items():
        _, hi = cfg.active_parameters()
        _, fhi = cfg.flops_per_token()
        kv = cfg.kv_bytes_per_token(loops=cfg.router.max_loops)
        lines.append(
            f"{key:<10} {cfg.n_params:>18,} {hi:>16,} "
            f"{fhi / 1e9:>13.2f} "
            f"{_si(cfg.weight_bytes(16.0)):>10} "
            f"{_si(cfg.weight_bytes(4.25)):>10} "
            f"{_si(kv):>10}"
        )
    return "\n".join(lines)


def _si(num_bytes: float) -> str:
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if num_bytes >= scale:
            return f"{num_bytes / scale:.2f} {unit}"
    return f"{num_bytes:.0f} B"


if __name__ == "__main__":  # pragma: no cover
    print(ladder_table())
    print()
    for rung in ("tiny", "nano", "base"):
        print(LADDER[rung].report().render())
        print()
