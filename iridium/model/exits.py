"""Intermediate-depth exits for the control core: LayerSkip's training recipe
and the head that reads out of a partial forward.

An "exit" reads a control-core hidden state that stopped at layer ``e < n_layers``
and produces the same text-logit distribution the model's real output head
would. The naive way to build one is a fresh ``Linear(d_model, vocab)`` per
exit depth. That is wasteful and wrong here: the model already has an output
head (``CodecBank.text_head``, weight-tied to the input embedding at most
scales), and the *only* thing that differs between an intermediate state and
the final one is where its activation statistics sit -- an intermediate
residual stream has not been normalized by ``ControlCore.finalize`` and has
not seen the layers that would otherwise re-scale it. So ``ExitHead`` adds
exactly one cheap, exit-specific ``RMSNorm`` and reuses the shared text head
verbatim (passed in at call time, never copied), plus a tiny scalar
confidence head for the budget-streaming path in
``iridium/runtime/streaming.py``. This keeps each exit's marginal parameter
cost at ``2 * d_model + 1`` instead of ``d_model * vocab_size`` per exit,
which is the difference between "one exit per layer is basically free" and
"one exit per layer roughly doubles the model".

``early_exit_loss`` and ``layer_dropout_schedule`` are the training-side half
of LayerSkip (Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and
Self-Speculative Decoding", arXiv:2404.16710, ACL 2024 -- verified against the
abstract and ACL Anthology listing). Both default to off / unused unless a
training script explicitly calls them: nothing here changes what
``Iridium1.forward`` or ``losses`` compute when they are not invoked. This
matters because an exit that has never been trained is not a cheap version of
the model, it is an untrained linear probe -- its argmax is close to uniform
noise over the vocabulary, and self-speculative decoding built on it will
still be *exact* (see ``streaming.py``: acceptance is checked against the
full-depth model, not the exit) but its acceptance rate, and therefore any
speed benefit, will be poor until ``early_exit_loss`` has actually been
optimized. ``tests/unit/test_streaming.py`` measures and reports this rather
than assuming it away.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


class ExitHead(nn.Module):
    """Reads an intermediate control-core state; reuses the model's text head.

    ``text_head`` (a ``codecs.heads.TextHead``) is passed to :meth:`forward`
    rather than stored as a submodule on purpose: storing it here would make
    it a child of *two* places in the module tree (``CodecBank`` and every
    ``ExitHead``), and while ``nn.Module.parameters()`` deduplicates tied
    tensors by identity so the aggregate count would still be right, the
    per-module "sum of this module's own parameters" that
    ``tests/unit/test_exits.py`` checks against :meth:`count_params` would
    then depend on which other modules happen to hold the same tensor --
    fragile, and it would silently stop being true the day someone untied the
    heads. Taking the head as an argument instead makes ``ExitHead``'s own
    parameter count exactly ``2 * d_model + 1``, unconditionally.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        # A single scalar per token: "how much do I trust this exit's logits
        # right now". Trained by early_exit_loss's confidence term (below);
        # read by StreamingSession's budget mode as the CALM-style stopping
        # signal.
        self.confidence = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor, text_head) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm(h)
        logits = text_head(normed)
        conf = torch.sigmoid(self.confidence(h).float()).squeeze(-1)
        return logits, conf

    @staticmethod
    def count_params(d_model: int) -> int:
        """``RMSNorm`` weight (``d_model``) + confidence ``Linear`` weight and
        bias (``d_model + 1``). Does *not* include the reused text head --
        see the class docstring for why that would be double-counting."""
        return d_model + (d_model + 1)


def layer_dropout_schedule(n_layers: int, max_rate: float = 0.2) -> list[float]:
    """LayerSkip's per-layer stochastic-depth rate (Sec. 3.1 of the paper).

    Dropout rate rises linearly with depth, from ``0`` at layer 0 to
    ``max_rate`` at the last layer: ``rate(l) = max_rate * l / (n_layers - 1)``.
    This is deliberately *not* a flat rate. Every exit at layer ``e`` depends
    on layers ``0..e-1`` having run; a flat dropout rate would make the
    *earliest* layers -- load-bearing for every exit, not just the last one
    -- unreliable exactly as often as the layers only the final exit depends
    on. Concentrating dropout at the end trains the deep layers to be
    genuinely optional (which is what lets an early exit skip them) while
    keeping the shared shallow prefix stable for every exit depth at once.
    A caller applies ``rate(l)`` as a per-sample Bernoulli skip of layer
    ``l``'s residual update during training; this function only returns the
    schedule; it does not wire it into ``ControlCore`` (see the note on
    ``control_core.py`` in the final report -- that hook belongs to the
    model owner).
    """
    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    if n_layers == 1:
        return [0.0]
    return [max_rate * i / (n_layers - 1) for i in range(n_layers)]


def curriculum_weights(exit_layers: Sequence[int], progress: float) -> list[float]:
    """LayerSkip's early-exit-loss curriculum coefficient, ``C_t`` in the paper.

    ``progress`` is training fraction in ``[0, 1]``. At ``progress=0`` almost
    all weight sits on the *last* listed exit (so training starts out close
    to plain next-token training with no early-exit tax); as ``progress``
    increases weight is redistributed toward the shallower exits, so they are
    not asked to do useful work with a representation that is still being
    formed by a moving target. This is the schedule the paper motivates
    qualitatively (progressively easier early layers, harder curriculum for
    exits as training proceeds) rather than a numerically transcribed
    constant from it, since the paper's formula is defined over total decoder
    layers, not over an arbitrary caller-chosen exit subset; treat exact
    constants here as an approximation to verify against the reference
    implementation before relying on it for a real training run.

    Returns one non-negative weight per entry of ``exit_layers``, summing to 1.
    """
    if not exit_layers:
        raise ValueError("exit_layers must be non-empty")
    progress = min(max(progress, 0.0), 1.0)
    n = len(exit_layers)
    if n == 1:
        return [1.0]
    # Rank 0 = shallowest exit. early_share ramps 0 -> 1 over training.
    early_share = progress
    ranks = [i / (n - 1) for i in range(n)]
    raw = [early_share * (1.0 - r) + (1.0 - early_share) * r for r in ranks]
    total = sum(raw)
    return [w / total for w in raw]


def early_exit_loss(
    model,
    batch,
    exit_layers: Sequence[int],
    weights: Optional[Sequence[float]] = None,
    exit_heads: Optional[Sequence[ExitHead]] = None,
) -> dict[str, torch.Tensor]:
    """LayerSkip's curriculum-weighted early-exit cross-entropy (text/control
    slots only -- this is the streaming-relevant subset of ``codecs.losses``,
    not a replacement for it).

    Runs the control core's stage-one layers *once*, in increasing depth
    order, capturing the residual stream at each requested boundary, then
    scores each captured state with its own :class:`ExitHead` against the
    same next-token targets ``CodecBank.losses`` uses. Running once and
    slicing out intermediate states -- rather than calling
    ``ControlCore._run`` once per exit layer -- is an ``O(n_layers)``
    forward instead of the ``O(len(exit_layers) * max(exit_layers))`` that
    repeating the prefix per exit would cost; the two are numerically
    identical for the deterministic (non-dropout) core because each layer's
    output depends only on the layers before it, not on which future exit
    will read the state, so the shared-prefix optimization changes nothing
    but wall-clock cost.

    ``exit_layers`` must be non-decreasing and lie in ``[1, model.cfg.core.split]``
    -- restricted to stage-one, so this never depends on the router's output
    (which would make the loss depend on a routing decision the shallow
    layers have not produced anything for yet). A caller wanting exits inside
    stage two needs a model-owner change (the core's own forward already runs
    stage two only after the bank, so an exit there is not "skip the rest of
    the core", it is "skip the bank", a materially different mechanism from
    what LayerSkip trains).

    Returns ``{"early_exit": weighted_sum, "early_exit_layer_<e>": per-layer}``.
    Optional per default: nothing calls this unless a training script does.
    """
    core = model.core
    split = core.split
    if any(e < 1 or e > split for e in exit_layers):
        raise ValueError(f"exit_layers must be in [1, {split}] (stage-one only)")
    if list(exit_layers) != sorted(exit_layers):
        raise ValueError("exit_layers must be non-decreasing")
    if weights is None:
        weights = curriculum_weights(exit_layers, progress=1.0)
    if len(weights) != len(exit_layers):
        raise ValueError("weights must match exit_layers")
    if exit_heads is None:
        exit_heads = [ExitHead(model.cfg.core.d_model) for _ in exit_layers]
    if len(exit_heads) != len(exit_layers):
        raise ValueError("exit_heads must match exit_layers")

    h = model.codecs.embed(batch)
    positions = batch.positions
    keep = model._stream_keep(batch, 0)

    tgt = model.codecs.next_slot_targets(batch)
    from ..codecs.spans import MODALITY_INDEX
    text_mask = tgt["valid"] & (
        (tgt["modality"] == MODALITY_INDEX["text"])
        | (tgt["modality"] == MODALITY_INDEX["control"])
    )

    losses: dict[str, torch.Tensor] = {}
    total = h.new_zeros(())
    layer_ptr = 0
    next_boundary = exit_layers[layer_ptr]
    for i in range(split):
        h = core.layers[i](h, positions, keep, None, None)
        while layer_ptr < len(exit_layers) and i + 1 == next_boundary:
            state = h[:, :-1]
            logits, conf = exit_heads[layer_ptr](state, model.codecs.text_head)
            logits = logits.float()
            if bool(text_mask.any()):
                ce = F.cross_entropy(
                    logits[text_mask], tgt["discrete"][text_mask], reduction="mean"
                )
                # Confidence target: 1 where the exit's own greedy choice
                # matches the label, 0 otherwise -- teaches the confidence
                # head to predict its own correctness, which is exactly what
                # StreamingSession's budget mode needs it for.
                with torch.no_grad():
                    correct = (logits[text_mask].argmax(-1) == tgt["discrete"][text_mask]).float()
                conf_loss = F.binary_cross_entropy(conf[text_mask].clamp(1e-6, 1 - 1e-6), correct)
            else:
                ce = h.new_zeros(())
                conf_loss = h.new_zeros(())
            layer_loss = ce + 0.1 * conf_loss
            losses[f"early_exit_layer_{next_boundary}"] = layer_loss
            total = total + weights[layer_ptr] * layer_loss
            layer_ptr += 1
            if layer_ptr < len(exit_layers):
                next_boundary = exit_layers[layer_ptr]
    losses["early_exit"] = total
    return losses
